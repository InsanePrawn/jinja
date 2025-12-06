"""Parse tokens from the lexer into nodes for the compiler."""

import typing
import typing as t

from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

_ImportInclude = t.TypeVar("_ImportInclude", nodes.Import, nodes.Include)
_MacroCall = t.TypeVar("_MacroCall", nodes.Macro, nodes.CallBlock)

_statement_keywords = frozenset(
    [
        "for",
        "if",
        "block",
        "extends",
        "print",
        "macro",
        "include",
        "from",
        "import",
        "set",
        "with",
        "autoescape",
    ]
)
_compare_operators = frozenset(["eq", "ne", "lt", "lteq", "gt", "gteq"])

_math_nodes: dict[str, type[nodes.Expr]] = {
    "add": nodes.Add,
    "sub": nodes.Sub,
    "mul": nodes.Mul,
    "div": nodes.Div,
    "floordiv": nodes.FloorDiv,
    "mod": nodes.Mod,
}


class Parser:
    """This is the central parsing class Jinja uses.  It's passed to
    extensions and can be used to parse expressions or statements.
    """

    def __init__(
        self,
        environment: "Environment",
        source: str,
        name: str | None = None,
        filename: str | None = None,
        state: str | None = None,
    ) -> None:
        self.environment = environment
        self.stream = environment._tokenize(source, name, filename, state)
        self.name = name
        self.filename = filename
        self.closed = False
        self.extensions: dict[
            str, t.Callable[[Parser], nodes.Node | list[nodes.Node]]
        ] = {}
        for extension in environment.iter_extensions():
            for tag in extension.tags:
                self.extensions[tag] = extension.parse
        self._last_identifier = 0
        self._tag_stack: list[str] = []
        self._end_token_stack: list[tuple[str, ...]] = []

    def fail(
        self,
        msg: str,
        lineno: int | None = None,
        exc: type[TemplateSyntaxError] = TemplateSyntaxError,
    ) -> "te.NoReturn":
        """Convenience method that raises `exc` with the message, passed
        line number or last line number as well as the current name and
        filename.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        raise exc(msg, lineno, self.name, self.filename)

    def _fail_ut_eof(
        self,
        name: str | None,
        end_token_stack: list[tuple[str, ...]],
        lineno: int | None,
    ) -> "te.NoReturn":
        expected: set[str] = set()
        for exprs in end_token_stack:
            expected.update(map(describe_token_expr, exprs))
        if end_token_stack:
            currently_looking: str | None = " or ".join(
                map(repr, map(describe_token_expr, end_token_stack[-1]))
            )
        else:
            currently_looking = None

        if name is None:
            message = ["Unexpected end of template."]
        else:
            message = [f"Encountered unknown tag {name!r}."]

        if currently_looking:
            if name is not None and name in expected:
                message.append(
                    "You probably made a nesting mistake. Jinja is expecting this tag,"
                    f" but currently looking for {currently_looking}."
                )
            else:
                message.append(
                    f"Jinja was looking for the following tags: {currently_looking}."
                )

        if self._tag_stack:
            message.append(
                "The innermost block that needs to be closed is"
                f" {self._tag_stack[-1]!r}."
            )

        self.fail(" ".join(message), lineno)

    def fail_unknown_tag(self, name: str, lineno: int | None = None) -> "te.NoReturn":
        """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
        self._fail_ut_eof(name, self._end_token_stack, lineno)

    def fail_eof(
        self,
        end_tokens: tuple[str, ...] | None = None,
        lineno: int | None = None,
    ) -> "te.NoReturn":
        """Like fail_unknown_tag but for end of template situations."""
        stack = list(self._end_token_stack)
        if end_tokens is not None:
            stack.append(end_tokens)
        self._fail_ut_eof(None, stack, lineno)

    def is_tuple_end(self, extra_end_rules: tuple[str, ...] | None = None) -> bool:
        """Are we at the end of a tuple?"""
        if self.stream.current.type in ("variable_end", "block_end", "rparen"):
            return True
        elif extra_end_rules is not None:
            return self.stream.current.test_any(extra_end_rules)  # type: ignore
        return False

    def free_identifier(
        self, lineno: int | None = None, linepos: int | None = None
    ) -> nodes.InternalName:
        """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
        self._last_identifier += 1
        rv = object.__new__(nodes.InternalName)
        nodes.Node.__init__(
            rv,
            f"fi{self._last_identifier}",
            lineno=lineno,
            linepos=linepos,
            lineno_end=lineno,
            linepos_end=linepos,
        )
        return rv

    def parse_statement(self) -> nodes.Node | list[nodes.Node]:
        """Parse a single statement."""
        token = self.stream.current
        if token.type != "name":
            if not self.environment.parser_tolerate_faults:
                self.fail("tag name expected", token.lineno)
            nxt = self.stream.look() if not self.stream.closed else self.stream.current
            return nodes.EmptyStatement(
                message="Tag name expected",
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=nxt.lineno,
                linepos_end=nxt.linepos,
                issue_context="tag",
            )
        self._tag_stack.append(token.value)
        pop_tag = True
        try:
            if token.value in _statement_keywords:
                f = getattr(self, f"parse_{self.stream.current.value}")
                return f()  # type: ignore
            if token.value == "call":
                return self.parse_call_block()
            if token.value == "filter":
                return self.parse_filter_block()
            ext = self.extensions.get(token.value)
            if ext is not None:
                res = ext(self)
                if hasattr(res, "linepos") and res.linepos is None:
                    res.linepos = token.linepos
                return res

            # did not work out, remove the token we pushed by accident
            # from the stack so that the unknown tag fail function can
            # produce a proper error message.
            self._tag_stack.pop()
            pop_tag = False
            self.fail_unknown_tag(token.value, token.lineno)
        finally:
            if pop_tag:
                self._tag_stack.pop()

    def parse_statements(
        self, end_tokens: tuple[str, ...], drop_needle: bool = False
    ) -> list[nodes.Node]:
        """Parse multiple statements into a list until one of the end tokens
        is reached.  This is used to parse the body of statements as it also
        parses template data if appropriate.  The parser checks first if the
        current token is a colon and skips it if there is one.  Then it checks
        for the block end and parses until if one of the `end_tokens` is
        reached.  Per default the active token in the stream at the end of
        the call is the matched end token.  If this is not wanted `drop_needle`
        can be set to `True` and the end token is removed.
        """
        # the first token may be a colon for python compatibility
        self.stream.skip_if("colon")

        # in the future it would be possible to add whole code sections
        # by adding some sort of end of statement token and parsing those here.
        self.stream.expect("block_end")
        result = self.subparse(end_tokens)

        # we reached the end of the template too early, the subparser
        # does not check for this, so we do that now
        if self.stream.current.type == "eof":
            self.fail_eof(end_tokens)

        if drop_needle:
            next(self.stream)
        return result

    def parse_set(self) -> nodes.Assign | nodes.AssignBlock:
        """Parse an assign statement."""
        _next = next(self.stream)
        lineno = _next.lineno
        linepos = _next.linepos
        target = self.parse_assign_target(with_namespace=True)
        expr_start = self.stream.next_if("assign")
        if expr_start:
            expr = self.parse_tuple(allow_empty=self.environment.parser_tolerate_faults)
            end_token = self.stream.current
            result = nodes.Assign(
                target,
                expr,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            if isinstance(expr, nodes.EmptyExpression):
                expr.message = "Assignment to empty expression"
                expr.issue_context = "assignment"
                expr.lineno, expr.linepos = expr_start.lineno, expr_start.linepos
                expr.linepos_end += 1
            return result
        filter_node = self.parse_filter(None)
        body = self.parse_statements(("name:endset",), drop_needle=True)
        end_token = self.stream.current
        return nodes.AssignBlock(
            target,
            filter_node,
            body,
            lineno=lineno,
            linepos=linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_for(self) -> nodes.For:
        """Parse a for loop."""
        _next = self.stream.expect("name:for")
        lineno = _next.lineno
        linepos = _next.linepos
        target = self.parse_assign_target(extra_end_rules=("name:in",))
        iter_start = self.stream.expect("name:in")
        iter = self.parse_tuple(
            with_condexpr=False,
            extra_end_rules=("name:recursive",),
            allow_empty=self.environment.parser_tolerate_faults,
        )
        if self.environment.parser_tolerate_faults and isinstance(
            iter, nodes.EmptyExpression
        ):
            iter.message = "Empty For-loop iterator"
            iter.lineno, iter.linepos = iter_start.lineno, iter_start.linepos
            assert iter.linepos_end is not None
            iter.linepos_end += 1
            iter.issue_context = "for_iterator"

        test = None
        if self.stream.skip_if("name:if"):
            test = self.parse_expression()
        recursive = self.stream.skip_if("name:recursive")
        body = self.parse_statements(("name:endfor", "name:else"))
        if next(self.stream).value == "endfor":
            else_ = []
        else:
            else_ = self.parse_statements(("name:endfor",), drop_needle=True)
        end_token = self.stream.current
        return nodes.For(
            target,
            iter,
            body,
            else_,
            test,
            recursive,
            lineno=lineno,
            linepos=linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_if(self) -> nodes.If:
        """Parse an if construct."""
        current = self.stream.current
        _next = self.stream.expect("name:if")
        node = result = nodes.If(
            lineno=current.lineno,
            linepos=current.linepos,
            lineno_end=_next.lineno,
            linepos_end=_next.linepos,
        )
        while True:
            node.test = self.parse_tuple(
                with_condexpr=False,
                allow_empty=self.environment.parser_tolerate_faults,
            )
            node.body = self.parse_statements(("name:elif", "name:else", "name:endif"))
            node.elif_ = []
            node.else_ = []
            token = next(self.stream)
            nxt = self.stream.look() if not self.stream.closed else self.stream.current
            if token.test("name:elif"):
                node = nodes.If(
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                )
                result.elif_.append(node)
                continue
            elif token.test("name:else"):
                result.else_ = self.parse_statements(("name:endif",), drop_needle=True)
            break
        end_token = self.stream.current
        result.lineno_end = end_token.lineno
        result.linepos_end = end_token.linepos
        return result

    def parse_with(self) -> nodes.With:
        _next = next(self.stream)
        node = nodes.With(lineno=_next.lineno, linepos=_next.linepos)
        targets: list[nodes.Expr] = []
        values: list[nodes.Expr] = []
        while self.stream.current.type != "block_end":
            if targets:
                self.stream.expect("comma")
            target = self.parse_assign_target()
            target.set_ctx("param")
            targets.append(target)
            self.stream.expect("assign")
            values.append(self.parse_expression())
        node.targets = targets
        node.values = values
        node.body = self.parse_statements(("name:endwith",), drop_needle=True)
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_autoescape(self) -> nodes.Scope:
        _next = next(self.stream)
        node = nodes.ScopedEvalContextModifier(
            lineno=_next.lineno, linepos=_next.linepos
        )
        node.options = [nodes.Keyword("autoescape", self.parse_expression())]
        node.body = self.parse_statements(("name:endautoescape",), drop_needle=True)
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return nodes.Scope(
            [node],
            lineno=node.lineno,
            linepos=node.linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_block(self) -> nodes.Block:
        _next = next(self.stream)
        node = nodes.Block(lineno=_next.lineno, linepos=_next.linepos)
        node.name = self.stream.expect("name").value
        node.scoped = self.stream.skip_if("name:scoped")
        node.required = self.stream.skip_if("name:required")

        # common problem people encounter when switching from django
        # to jinja.  we do not support hyphens in block names, so let's
        # raise a nicer error message in that case.
        if self.stream.current.type == "sub":
            self.fail(
                "Block names in Jinja have to be valid Python identifiers and may not"
                " contain hyphens, use an underscore instead."
            )

        node.body = self.parse_statements(("name:endblock",), drop_needle=True)

        # enforce that required blocks only contain whitespace or comments
        # by asserting that the body, if not empty, is just TemplateData nodes
        # with whitespace data
        if node.required:
            for body_node in node.body:
                if not isinstance(body_node, nodes.Output) or any(
                    not isinstance(output_node, nodes.TemplateData)
                    or not output_node.data.isspace()
                    for output_node in body_node.nodes
                ):
                    self.fail("Required blocks can only contain comments or whitespace")

        if not self.environment.parser_tolerate_faults:
            self.stream.skip_if("name:" + node.name)
        elif self.stream.current.test("name"):
            node.endblock_with_name = True
            wrong = self.stream.expect("name")
            if wrong.value != node.name:
                node.issues = node.issues or []
                node.issues.append(
                    nodes.ParserIssue(
                        message=f"endblock used with incorrect name {wrong.value!r} for block {node.name!r}",
                        lineno=wrong.lineno,
                        linepos=wrong.linepos,
                        lineno_end=wrong.lineno,
                        linepos_end=wrong.linepos + len(wrong.value),
                        issue_context="endblock",
                    )
                )
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_extends(self) -> nodes.Extends:
        _next = next(self.stream)
        node = nodes.Extends(lineno=_next.lineno, linepos=_next.linepos)
        node.template = self.parse_expression()
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_import_context(
        self, node: _ImportInclude, default: bool
    ) -> _ImportInclude:
        if self.stream.current.test_any(
            "name:with", "name:without"
        ) and self.stream.look().test("name:context"):
            node.with_context = next(self.stream).value == "with"
            self.stream.skip()
        else:
            node.with_context = default
        return node

    def parse_include(self) -> nodes.Include:
        _next = next(self.stream)
        node = nodes.Include(lineno=_next.lineno, linepos=_next.linepos)
        node.template = self.parse_expression()
        if self.stream.current.test("name:ignore") and self.stream.look().test(
            "name:missing"
        ):
            node.ignore_missing = True
            self.stream.skip(2)
        else:
            node.ignore_missing = False
        result = self.parse_import_context(node, True)
        end_token = self.stream.current
        result.lineno_end = end_token.lineno
        result.linepos_end = end_token.linepos
        return result

    def parse_import(self) -> nodes.Import:
        _next = next(self.stream)
        node = nodes.Import(lineno=_next.lineno, linepos=_next.linepos)
        node.template = self.parse_expression()
        self.stream.expect("name:as")
        node.target = self.parse_assign_target(name_only=True).name
        result = self.parse_import_context(node, False)
        end_token = self.stream.current
        result.lineno_end = end_token.lineno
        result.linepos_end = end_token.linepos
        return result

    def parse_from(self) -> nodes.FromImport:
        _next = next(self.stream)
        node = nodes.FromImport(lineno=_next.lineno, linepos=_next.linepos)
        node.template = self.parse_expression()
        self.stream.expect("name:import")
        node.names = []

        def parse_context() -> bool:
            if self.stream.current.value in {
                "with",
                "without",
            } and self.stream.look().test("name:context"):
                node.with_context = next(self.stream).value == "with"
                self.stream.skip()
                return True
            return False

        while True:
            if node.names:
                self.stream.expect("comma")
            if self.stream.current.type == "name":
                if parse_context():
                    break
                target = self.parse_assign_target(name_only=True)
                if target.name.startswith("_"):
                    self.fail(
                        "names starting with an underline can not be imported",
                        target.lineno,
                        exc=TemplateAssertionError,
                    )
                if self.stream.skip_if("name:as"):
                    alias = self.parse_assign_target(name_only=True)
                    node.names.append((target.name, alias.name))
                else:
                    node.names.append(target.name)
                if parse_context() or self.stream.current.type != "comma":
                    break
            else:
                self.stream.expect("name")
        if not hasattr(node, "with_context"):
            node.with_context = False
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_signature(
        self, node: _MacroCall
    ) -> None | nodes.EmptyExpression | nodes.InvalidExpression:
        args = node.args = []
        defaults = node.defaults = []
        if (
            self.environment.parser_tolerate_faults
            and self.stream.current.type != "lparen"
        ):
            if self.stream.current.type == "block_end":
                node.issues = node.issues or []
                issue = nodes.EmptyExpression(  # type: ignore[assignment]
                    lineno=node.lineno,
                    linepos=node.linepos,
                    lineno_end=self.stream.current.lineno,
                    linepos_end=self.stream.current.linepos,
                    message=f"Missing {type(node).__name__} signature",
                    issue_context="signature",
                )
                node.issues.append(issue)
                return issue

        self.stream.expect("lparen")
        while self.stream.current.type != "rparen":
            if args:
                self.stream.expect("comma")
            arg = self.parse_assign_target(name_only=True)
            arg.set_ctx("param")
            if self.stream.skip_if("assign"):
                defaults.append(self.parse_expression())
            elif defaults:
                msg = "non-default argument follows default argument"
                if not self.environment.parser_tolerate_faults:
                    self.fail(msg)
                err = nodes.InvalidExpression(
                    lineno=arg.lineno,
                    linepos=arg.linepos,
                    lineno_end=self.stream.current.lineno,
                    linepos_end=self.stream.current.linepos,
                    message=msg,
                    original_str=arg.name,
                )
                arg.issues = arg.issues or []
                arg.issues.append(err)
            args.append(arg)
        self.stream.expect("rparen")
        return None

    def parse_call_block(self) -> nodes.CallBlock:
        _next = next(self.stream)
        node = nodes.CallBlock(lineno=_next.lineno, linepos=_next.linepos)
        if self.stream.current.type == "lparen":
            signature_issue = self.parse_signature(node)
            if signature_issue:
                assert self.environment.parser_tolerate_faults
                node.args = signature_issue  # type: ignore[assignment]
        else:
            node.args = []
            node.defaults = []

        call_node = self.parse_expression()
        if not isinstance(call_node, nodes.Call):
            if not (
                self.environment.parser_tolerate_faults or isinstance(call_node, Name)
            ):
                self.fail("expected call", node.lineno)
            call_node.issues = call_node.issues or []
            issue = nodes.EmptyExpression(
                message="Expected function call; missing parentheses",
                lineno=call_node.lineno,
                linepos=call_node.linepos,
                lineno_end=call_node.lineno_end,
                linepos_end=call_node.linepos_end,
                issue_context="function_call",
            )
            call_node.issues.append(issue)
        node.call = call_node
        node.body = self.parse_statements(("name:endcall",), drop_needle=True)
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_filter_block(self) -> nodes.FilterBlock:
        _next = next(self.stream)
        node = nodes.FilterBlock(lineno=_next.lineno, linepos=_next.linepos)
        node.filter = self.parse_filter(None, start_inline=True)  # type: ignore
        node.body = self.parse_statements(("name:endfilter",), drop_needle=True)
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_macro(self) -> nodes.Macro:
        _next = next(self.stream)
        node = nodes.Macro(lineno=_next.lineno, linepos=_next.linepos, issues=None)
        node.name = self.parse_assign_target(name_only=True).name
        signature_issue = self.parse_signature(node)
        node.body = self.parse_statements(("name:endmacro",), drop_needle=True)
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    def parse_print(self) -> nodes.Output:
        _next = next(self.stream)
        node = nodes.Output(lineno=_next.lineno, linepos=_next.linepos)
        node.nodes = []
        while self.stream.current.type != "block_end":
            if node.nodes:
                self.stream.expect("comma")
            node.nodes.append(self.parse_expression())
        end_token = self.stream.current
        node.lineno_end = end_token.lineno
        node.linepos_end = end_token.linepos
        return node

    @typing.overload
    def parse_assign_target(
        self, with_tuple: bool = ..., name_only: "te.Literal[True]" = ...
    ) -> nodes.Name: ...

    @typing.overload
    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: tuple[str, ...] | None = None,
        with_namespace: bool = False,
    ) -> nodes.NSRef | nodes.Name | nodes.Tuple: ...

    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: tuple[str, ...] | None = None,
        with_namespace: bool = False,
    ) -> nodes.NSRef | nodes.Name | nodes.Tuple:
        """Parse an assignment target.  As Jinja allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.  If
        `with_namespace` is enabled, a namespace assignment may be parsed.
        """
        target: nodes.Expr

        if name_only:
            token = self.stream.expect("name")
            nxt = self.stream.look() if not self.stream.closed else self.stream.current
            target = nodes.Name(
                token.value,
                "store",
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=nxt.lineno,
                linepos_end=nxt.linepos,
            )
        else:
            if with_tuple:
                target = self.parse_tuple(
                    simplified=True,
                    extra_end_rules=extra_end_rules,
                    with_namespace=with_namespace,
                )
            else:
                target = self.parse_primary(with_namespace=with_namespace)

            target.set_ctx("store")

        if not target.can_assign():
            self.fail(
                f"can't assign to {type(target).__name__.lower()!r}", target.lineno
            )

        return target  # type: ignore

    def parse_expression(self, with_condexpr: bool = True) -> nodes.Expr:
        """Parse an expression.  Per default all expressions are parsed, if
        the optional `with_condexpr` parameter is set to `False` conditional
        expressions are not parsed.
        """
        if with_condexpr:
            return self.parse_condexpr()
        return self.parse_or()

    def parse_condexpr(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        expr1 = self.parse_or()
        expr3: nodes.Expr | None

        while self.stream.skip_if("name:if"):
            expr2 = self.parse_or()
            if self.stream.skip_if("name:else"):
                expr3 = self.parse_condexpr()
            else:
                expr3 = None
            end_token = self.stream.current
            expr1 = nodes.CondExpr(
                expr2,
                expr1,
                expr3,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return expr1

    def parse_or(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        left = self.parse_and()
        while self.stream.current.test("name:or"):
            token = next(self.stream)
            right = self.parse_and()
            end_token = self.stream.current
            if isinstance(right, nodes.EmptyExpression):
                right.lineno, right.linepos = token.lineno, token.linepos
            left = nodes.Or(
                left,
                right,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return left

    def parse_and(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        left = self.parse_not()
        while self.stream.current.test("name:and"):
            token = next(self.stream)
            right = self.parse_not()
            if isinstance(right, nodes.EmptyExpression):
                right.lineno, right.linepos = token.lineno, token.linepos
            end_token = self.stream.current
            left = nodes.And(
                left,
                right,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return left

    def parse_not(self) -> nodes.Expr:
        if self.stream.current.test("name:not"):
            _next = next(self.stream)
            lineno = _next.lineno
            linepos = _next.linepos
            result = self.parse_not()
            end_token = self.stream.current
            return nodes.Not(
                result,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        return self.parse_compare()

    def parse_compare(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        expr = self.parse_math1()
        ops = []
        while True:
            token = self.stream.current
            token_type = token.type
            if token_type in _compare_operators:
                next(self.stream)
                nxt = self.stream.current
                ops.append(
                    nodes.Operand(
                        token_type,
                        self.parse_math1(),
                        lineno=token.lineno,
                        linepos=token.linepos,
                        lineno_end=nxt.lineno,
                        linepos_end=nxt.linepos,
                    )
                )
            elif self.stream.skip_if("name:in"):
                nxt = self.stream.look() if not self.stream.closed else token
                ops.append(
                    nodes.Operand(
                        "in",
                        self.parse_math1(),
                        lineno=token.lineno,
                        linepos=token.linepos,
                        lineno_end=nxt.lineno,
                        linepos_end=nxt.linepos,
                    )
                )
            elif self.stream.current.test("name:not") and self.stream.look().test(
                "name:in"
            ):
                token = self.stream.current
                self.stream.skip(2)
                nxt = (
                    self.stream.look()
                    if not self.stream.closed
                    else self.stream.current
                )
                ops.append(
                    nodes.Operand(
                        "notin",
                        self.parse_math1(),
                        lineno=token.lineno,
                        linepos=token.linepos,
                        lineno_end=nxt.lineno,
                        linepos_end=nxt.linepos,
                    )
                )
            else:
                break
        if not ops:
            if isinstance(expr, nodes.EmptyExpression):
                expr.lineno, expr.linepos = lineno, linepos
            return expr
        end_token = self.stream.current
        return nodes.Compare(
            expr,
            ops,
            lineno=lineno,
            linepos=linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_math1(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        left = self.parse_concat()
        while self.stream.current.type in ("add", "sub"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_concat()
            end_token = self.stream.current
            left = cls(
                left,
                right,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return left

    def parse_concat(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        args = [self.parse_math2()]
        while self.stream.current.type == "tilde":
            next(self.stream)
            args.append(self.parse_math2())
        if len(args) == 1:
            return args[0]
        end_token = self.stream.current
        return nodes.Concat(
            args,
            lineno=lineno,
            linepos=linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_math2(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        left = self.parse_pow()
        while self.stream.current.type in ("mul", "div", "floordiv", "mod"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_pow()
            end_token = self.stream.current
            left = cls(
                left,
                right,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return left

    def parse_pow(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        left = self.parse_unary()
        while self.stream.current.type == "pow":
            next(self.stream)
            right = self.parse_unary()
            end_token = self.stream.current
            left = nodes.Pow(
                left,
                right,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
            lineno = self.stream.current.lineno
            linepos = self.stream.current.linepos
        return left

    def parse_unary(self, with_filter: bool = True) -> nodes.Expr:
        token_type = self.stream.current.type
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        node: nodes.Expr

        if token_type == "sub":
            next(self.stream)
            inner = self.parse_unary(False)
            end_token = self.stream.current
            node = nodes.Neg(
                inner,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        elif token_type == "add":
            next(self.stream)
            inner = self.parse_unary(False)
            end_token = self.stream.current
            node = nodes.Pos(
                inner,
                lineno=lineno,
                linepos=linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        else:
            node = self.parse_primary()
            node.lineno, node.linepos = lineno, linepos
        node = self.parse_postfix(node)
        if with_filter:
            node = self.parse_filter_expr(node)
        return node

    def parse_primary(self, with_namespace: bool = False) -> nodes.Expr:
        """Parse a name or literal value. If ``with_namespace`` is enabled, also
        parse namespace attr refs, for use in assignments."""
        token = self.stream.current
        node: nodes.Expr
        if token.type == "name":
            next(self.stream)
            if token.value in ("true", "false", "True", "False"):
                node = nodes.Const(
                    token.value in ("true", "True"),
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=token.lineno,
                    linepos_end=token.linepos + len(token.value),
                )
            elif token.value in ("none", "None"):
                node = nodes.Const(
                    None,
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=token.lineno,
                    linepos_end=token.linepos + len(token.value),
                )
            elif with_namespace and self.stream.current.type == "dot":
                # If namespace attributes are allowed at this point, and the next
                # token is a dot, produce a namespace reference.
                next(self.stream)
                attr = self.stream.expect("name")
                nxt = self.stream.current
                node = nodes.NSRef(
                    token.value,
                    attr.value,
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                )
            else:
                nxt = (
                    self.stream.look()
                    if not self.stream.closed
                    else self.stream.current
                )
                node = nodes.Name(
                    token.value,
                    "load",
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                )
        elif token.type == "string":
            next(self.stream)
            buf = [token.value]
            lineno = token.lineno
            linepos_end = token.linepos
            while self.stream.current.type == "string":
                buf.append(self.stream.current.value)
                linepos_end = self.stream.current.linepos
                next(self.stream)
            node = nodes.Const(
                "".join(buf),
                lineno=lineno,
                linepos=token.linepos,
                lineno_end=self.stream.current.lineno,
                linepos_end=linepos_end,
            )
        elif token.type in ("integer", "float"):
            next(self.stream)
            nxt = self.stream.look() if not self.stream.closed else self.stream.current
            node = nodes.Const(
                token.value,
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=nxt.lineno,
                linepos_end=nxt.linepos,
            )
        elif token.type == "lparen":
            next(self.stream)
            node = self.parse_tuple(explicit_parentheses=True)
            self.stream.expect("rparen")
        elif token.type == "lbracket":
            node = self.parse_list()
        elif token.type == "lbrace":
            node = self.parse_dict()
        else:
            msg = f"unexpected {describe_token(token)!r}"
            if not self.environment.parser_tolerate_faults:
                self.fail(msg, token.lineno)
            if token.type == "variable_end":
                nxt = (
                    self.stream.look()
                    if not self.stream.closed
                    else self.stream.current
                )
                node = nodes.EmptyExpression(
                    message="Unexpected end of primary statement",
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                    issue_context="primary",
                )
            else:
                self.fail(msg, token.lineno)
        return node

    def parse_tuple(
        self,
        simplified: bool = False,
        with_condexpr: bool = True,
        extra_end_rules: tuple[str, ...] | None = None,
        explicit_parentheses: bool = False,
        with_namespace: bool = False,
        allow_empty: bool = False,
    ) -> nodes.Tuple | nodes.Expr:
        """Works like `parse_expression` but if multiple expressions are
        delimited by a comma a :class:`~jinja2.nodes.Tuple` node is created.
        This method could also return a regular expression instead of a tuple
        if no commas where found.

        The default parsing mode is a full tuple.  If `simplified` is `True`
        only names and literals are parsed; ``with_namespace`` allows namespace
        attr refs as well. The `no_condexpr` parameter is forwarded to
        :meth:`parse_expression`.

        Because tuples do not require delimiters and may end in a bogus comma
        an extra hint is needed that marks the end of a tuple.  For example
        for loops support tuples between `for` and `in`.  In that case the
        `extra_end_rules` is set to ``['name:in']``.

        `explicit_parentheses` is true if the parsing was triggered by an
        expression in parentheses.  This is used to figure out if an empty
        tuple is a valid expression or not.
        """
        lineno = self.stream.current.lineno
        lineno_start = lineno
        if simplified:

            def parse() -> nodes.Expr:
                return self.parse_primary(with_namespace=with_namespace)

        else:

            def parse() -> nodes.Expr:
                return self.parse_expression(with_condexpr=with_condexpr)

        args: list[nodes.Expr] = []
        is_tuple = False
        linepos_start = self.stream.current.linepos

        while True:
            if args:
                self.stream.expect("comma")
            if self.is_tuple_end(extra_end_rules):
                break
            args.append(parse())
            if self.stream.current.type == "comma":
                is_tuple = True
            else:
                break
            lineno = self.stream.current.lineno

        if not is_tuple:
            if args:
                return args[0]

            # if we don't have explicit parentheses, an empty tuple is
            # not a valid expression.  This would mean nothing (literally
            # nothing) in the spot of an expression would be an empty
            # tuple.
            if not explicit_parentheses:
                if allow_empty:
                    empty = nodes.EmptyExpression(
                        lineno=lineno_start,
                        linepos=linepos_start,
                        lineno_end=self.stream.current.lineno,
                        linepos_end=self.stream.current.linepos,
                        message="Expected an expression",
                    )
                    return empty
                self.fail(
                    "Expected an expression,"
                    f" got {describe_token(self.stream.current)!r}"
                )
        end_token = self.stream.current
        return nodes.Tuple(
            args,
            "load",
            lineno=lineno,
            linepos=linepos_start,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_list(self) -> nodes.List:
        token = self.stream.expect("lbracket")
        items: list[nodes.Expr] = []
        while self.stream.current.type != "rbracket":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbracket":
                break
            items.append(self.parse_expression())
        end_token = self.stream.expect("rbracket")
        return nodes.List(
            items,
            lineno=token.lineno,
            linepos=token.linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_dict(self) -> nodes.Dict:
        token = self.stream.expect("lbrace")
        items: list[nodes.Pair] = []
        while self.stream.current.type != "rbrace":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbrace":
                break
            key = self.parse_expression()
            self.stream.expect("colon")
            value = self.parse_expression()
            items.append(nodes.Pair(key, value, lineno=key.lineno, linepos=key.linepos))
        end_token = self.stream.expect("rbrace")
        return nodes.Dict(
            items,
            lineno=token.lineno,
            linepos=token.linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_postfix(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "dot" or token_type == "lbracket":
                node = self.parse_subscript(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_filter_expr(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "pipe":
                node = self.parse_filter(node)  # type: ignore
            elif token_type == "name" and self.stream.current.value == "is":
                node = self.parse_test(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_subscript(self, node: nodes.Expr) -> nodes.Getattr | nodes.Getitem:
        token = next(self.stream)
        arg: nodes.Expr

        if token.type == "dot":
            attr_token = self.stream.current
            if attr_token.type == "name":
                next(self.stream)
                nxt = self.stream.current
                return nodes.Getattr(
                    node,
                    attr_token.value,
                    "load",
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                )
            if attr_token.type != "integer":
                if not self.environment.parser_tolerate_faults:
                    self.fail("expected name or number", attr_token.lineno)
                arg = nodes.EmptyExpression(
                    message=f"Missing name for dot access! Got {attr_token.type}",
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=attr_token.lineno,
                    linepos_end=attr_token.linepos,
                    issue_context="attribute",
                )
            else:
                next(self.stream)
                nxt = self.stream.current
                arg = nodes.Const(
                    attr_token.value,
                    lineno=attr_token.lineno,
                    linepos=attr_token.linepos,
                    lineno_end=nxt.lineno,
                    linepos_end=nxt.linepos,
                )
            end_token = self.stream.current
            return nodes.Getitem(
                node,
                arg,
                "load",
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        if token.type == "lbracket":
            args: list[nodes.Expr] = []
            while self.stream.current.type != "rbracket":
                if args:
                    self.stream.expect("comma")
                args.append(self.parse_subscribed())
            end_token = self.stream.expect("rbracket")
            if len(args) == 1:
                arg = args[0]
            else:
                arg = nodes.Tuple(
                    args,
                    "load",
                    lineno=token.lineno,
                    linepos=token.linepos,
                    lineno_end=end_token.lineno,
                    linepos_end=end_token.linepos,
                )
            return nodes.Getitem(
                node,
                arg,
                "load",
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        self.fail("expected subscript expression", token.lineno)

    def parse_subscribed(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        linepos = self.stream.current.linepos
        args: list[nodes.Expr | None]

        if self.stream.current.type == "colon":
            next(self.stream)
            args = [None]
        else:
            node = self.parse_expression()
            if self.stream.current.type != "colon":
                return node
            next(self.stream)
            args = [node]

        if self.stream.current.type == "colon":
            args.append(None)
        elif self.stream.current.type not in ("rbracket", "comma"):
            args.append(self.parse_expression())
        else:
            args.append(None)

        if self.stream.current.type == "colon":
            next(self.stream)
            if self.stream.current.type not in ("rbracket", "comma"):
                args.append(self.parse_expression())
            else:
                args.append(None)
        else:
            args.append(None)

        end_token = self.stream.current
        return nodes.Slice(
            lineno=lineno,
            linepos=linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
            *args,
        )  # noqa: B026

    def parse_call_args(
        self,
    ) -> tuple[
        list[nodes.Expr],
        list[nodes.Keyword],
        nodes.Expr | None,
        nodes.Expr | None,
    ]:
        token = self.stream.expect("lparen")
        args = []
        kwargs = []
        dyn_args = None
        dyn_kwargs = None
        require_comma = False

        def ensure(expr: bool) -> None:
            if not expr:
                self.fail("invalid syntax for function call expression", token.lineno)

        while self.stream.current.type != "rparen":
            if require_comma:
                self.stream.expect("comma")

                # support for trailing comma
                if self.stream.current.type == "rparen":
                    break

            if self.stream.current.type == "mul":
                ensure(dyn_args is None and dyn_kwargs is None)
                next(self.stream)
                dyn_args = self.parse_expression()
            elif self.stream.current.type == "pow":
                ensure(dyn_kwargs is None)
                next(self.stream)
                dyn_kwargs = self.parse_expression()
            else:
                if (
                    self.stream.current.type == "name"
                    and self.stream.look().type == "assign"
                ):
                    # Parsing a kwarg
                    ensure(dyn_kwargs is None)
                    key = self.stream.current.value
                    self.stream.skip(2)
                    value = self.parse_expression()
                    kwargs.append(
                        nodes.Keyword(
                            key, value, lineno=value.lineno, linepos=value.linepos
                        )
                    )
                else:
                    # Parsing an arg
                    ensure(dyn_args is None and dyn_kwargs is None and not kwargs)
                    args.append(self.parse_expression())

            require_comma = True

        self.stream.expect("rparen")
        return args, kwargs, dyn_args, dyn_kwargs

    def parse_call(self, node: nodes.Expr) -> nodes.Call:
        # The lparen will be expected in parse_call_args, but the lineno
        # needs to be recorded before the stream is advanced.
        token = self.stream.current
        args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        end_token = self.stream.current
        return nodes.Call(
            node,
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=token.lineno,
            linepos=token.linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
        )

    def parse_filter(
        self, node: nodes.Expr | None, start_inline: bool = False
    ) -> nodes.Expr | None:
        while self.stream.current.type == "pipe" or start_inline:
            if not start_inline:
                next(self.stream)
            issues: list[nodes.ExprIssue] = []

            def _get_name() -> str:
                nonlocal issues
                if (
                    self.environment.parser_tolerate_faults
                    and not self.stream.current.test("name")
                ):
                    issues.append(
                        nodes.EmptyExpression(
                            message="Missing name: Filter expected",
                            lineno=self.stream.current.lineno,
                            linepos=self.stream.current.linepos,
                            lineno_end=self.stream.current.lineno,
                            linepos_end=self.stream.current.linepos,
                            issue_context="filter",
                        )
                    )
                    return ""
                return self.stream.expect("name").value

            name = _get_name()
            token = self.stream.current
            while self.stream.current.type == "dot":
                next(self.stream)
                name += "." + self.stream.expect("name").value
            if self.stream.current.type == "lparen":
                args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
            else:
                args = []
                kwargs = []
                dyn_args = dyn_kwargs = None
            end_token = self.stream.current
            node = nodes.Filter(
                node,
                name,
                args,
                kwargs,
                dyn_args,
                dyn_kwargs,
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
                issues=issues,
            )
            start_inline = False
        return node

    def parse_test(self, node: nodes.Expr) -> nodes.Expr:
        token = next(self.stream)
        if self.stream.current.test("name:not"):
            next(self.stream)
            negated = True
        else:
            negated = False
        issues: list[nodes.ExprIssue] = []

        def _get_name() -> str:
            nonlocal issues
            if self.environment.parser_tolerate_faults and not self.stream.current.test(
                "name"
            ):
                issues.append(
                    nodes.EmptyExpression(
                        message="Missing name: Test expected",
                        lineno=self.stream.current.lineno,
                        linepos=self.stream.current.linepos,
                        lineno_end=self.stream.current.lineno,
                        linepos_end=self.stream.current.linepos,
                        issue_context="test",
                    )
                )
                return ""

            return self.stream.expect("name").value

        name = _get_name()

        while self.stream.current.type == "dot":
            next(self.stream)
            name += "." + _get_name()
        dyn_args = dyn_kwargs = None
        kwargs: list[nodes.Keyword] = []
        if self.stream.current.type == "lparen":
            args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        elif self.stream.current.type in {
            "name",
            "string",
            "integer",
            "float",
            "lparen",
            "lbracket",
            "lbrace",
        } and not self.stream.current.test_any("name:else", "name:or", "name:and"):
            if self.stream.current.test("name:is"):
                self.fail("You cannot chain multiple tests with is")
            arg_node = self.parse_primary()
            arg_node = self.parse_postfix(arg_node)
            args = [arg_node]
        else:
            args = []
        end_token = self.stream.current
        node = nodes.Test(
            node,
            name,
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=token.lineno,
            linepos=token.linepos,
            lineno_end=end_token.lineno,
            linepos_end=end_token.linepos,
            issues=issues,
        )
        if negated:
            node = nodes.Not(
                node,
                lineno=token.lineno,
                linepos=token.linepos,
                lineno_end=end_token.lineno,
                linepos_end=end_token.linepos,
            )
        return node

    def subparse(self, end_tokens: tuple[str, ...] | None = None) -> list[nodes.Node]:
        body: list[nodes.Node] = []
        data_buffer: list[nodes.Node] = []
        add_data = data_buffer.append

        if end_tokens is not None:
            self._end_token_stack.append(end_tokens)

        def flush_data() -> None:
            if data_buffer:
                lineno = data_buffer[0].lineno
                linepos = data_buffer[0].linepos
                lineno_end = data_buffer[-1].lineno_end
                linepos_end = data_buffer[-1].linepos_end
                body.append(
                    nodes.Output(
                        data_buffer[:],
                        lineno=lineno,
                        linepos=linepos,
                        lineno_end=lineno_end,
                        linepos_end=linepos_end,
                    )
                )
                del data_buffer[:]

        try:
            while self.stream:
                token = self.stream.current
                if token.type == "data":
                    if "\n" not in token.value:
                        end = token.lineno, token.linepos + len(token.value)
                    else:
                        end = (
                            token.lineno + token.value.count("\n"),
                            len(token.value.rsplit("\n", 1)[-1]),
                        )
                    if token.value:
                        add_data(
                            nodes.TemplateData(
                                token.value,
                                lineno=token.lineno,
                                linepos=token.linepos,
                                lineno_end=end[0],
                                linepos_end=end[1],
                            )
                        )
                    next(self.stream)
                elif token.type == "variable_begin":
                    next(self.stream)
                    data = self.parse_tuple(
                        with_condexpr=True,
                        allow_empty=self.environment.parser_tolerate_faults,
                    )
                    if isinstance(data, nodes.EmptyExpression):
                        data.lineno, data.linepos = token.lineno, token.linepos
                        nxt = self.stream.current
                        data.lineno_end, data.linepos_end = nxt.lineno, nxt.linepos
                        if nxt.type == "variable_end":
                            data.linepos_end += len(nxt.value)
                        data.message = "Empty expression inside print statement"
                        data.issue_context = "print"
                    add_data(data)
                    self.stream.expect("variable_end")
                elif token.type == "block_begin":
                    flush_data()
                    next(self.stream)
                    if end_tokens is not None and self.stream.current.test_any(
                        *end_tokens
                    ):
                        return body
                    rv = self.parse_statement()
                    if isinstance(rv, list):
                        body.extend(rv)
                    else:
                        nxt = self.stream.current
                        if self.environment.parser_tolerate_faults and isinstance(
                            rv, (nodes.ParserIssue, nodes.EmptyStatement)
                        ):
                            rv.lineno, rv.linepos = token.lineno, token.linepos
                            rv = nodes.Output(
                                [rv],
                                lineno=token.lineno,
                                linepos=token.linepos,
                                lineno_end=nxt.lineno,
                                linepos_end=nxt.linepos,
                            )
                        body.append(rv)
                    self.stream.expect("block_end")
                else:
                    raise AssertionError("internal parsing error")

            flush_data()
        finally:
            if end_tokens is not None:
                self._end_token_stack.pop()
        return body

    def parse(self) -> nodes.Template:
        """Parse the whole template into a `Template` node."""
        result = nodes.Template(self.subparse(), lineno=1, linepos=0)
        result.set_environment(self.environment)
        # Set end position to the last token
        end_token = self.stream.current
        end_element = result.body[-1] if result.body else None
        end = max(
            (end_token.lineno, end_token.linepos),
            (-1, -1)
            if not end_element
            else (end_element.lineno_end, end_element.linepos_end),
        )
        result.lineno_end = end[0]
        result.linepos_end = end[1]
        return result
