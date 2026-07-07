"""
forge_core/ast_validator.py
===========================
Dynamic AST static analysis engine for Clean Lab architectural policy enforcement.

Provides the :func:`ast_policy` decorator to scan student implementations for
forbidden constructs (loops, disallowed calls, disallowed imports) or missing
required calls before execution or benchmarking occurs.
"""

from __future__ import annotations

import abc
import ast
import contextlib
import functools
import inspect
import sys
import textwrap
from collections.abc import Callable, Sequence
from typing import Any

from nguyenpanda.swan import c24, reset

_ERR = c24["ff3333"]
_WARN = c24["ff8800"]
_CYAN = c24["00d7ff"]
_GOLD = c24["ffd700"]


class PolicyViolationError(SyntaxError):
    """Exception raised when student implementation violates architectural AST policy."""

    def __init__(self, message: str, filename: str = "", lineno: int | None = None):
        super().__init__(message)
        self.filename = filename
        if lineno is not None:
            self.lineno = lineno


class VisitorContext:
    """Context state maintained during AST traversal."""

    def __init__(self) -> None:
        self.in_function_depth: int = 0


class PolicyRule(abc.ABC):
    """Abstract base class for all AST policy inspection rules."""

    def __init__(self, feedback: dict[str, str] | None = None) -> None:
        self.feedback: dict[str, str] = feedback or {}
        self.violations: list[tuple[bool, str, int | None]] = []

    @abc.abstractmethod
    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        """Return the AST node types inspected by this rule."""

    @abc.abstractmethod
    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        """Inspect an AST node and record any policy violations."""

    def check_post_traversal(self) -> None:  # noqa: B027
        """Perform any required post-traversal validation checks."""
        pass


def _resolve_call_name(func_node: ast.AST) -> str:
    if isinstance(func_node, ast.Name):
        return func_node.id
    elif isinstance(func_node, ast.Attribute):
        val_name = _resolve_call_name(func_node.value)
        return f"{val_name}.{func_node.attr}" if val_name else func_node.attr
    return ""


def _call_matches(actual: str, pattern: str) -> bool:
    if not actual or not pattern:
        return False
    if actual == pattern:
        return True
    pattern_attr = pattern.split(".")[-1]
    actual_attr = actual.split(".")[-1]
    return pattern_attr == actual_attr


class ForbidLoopsRule(PolicyRule):
    """Rule forbidding execution loops beyond defined maximum thresholds."""

    def __init__(
        self,
        max_for_loops: int = 0,
        max_while_loops: int = 0,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.max_for_loops = max_for_loops
        self.max_while_loops = max_while_loops
        self.for_count = 0
        self.while_count = 0

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.For, ast.AsyncFor, ast.While)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self.for_count += 1
            if self.for_count > self.max_for_loops:
                custom = (
                    self.feedback.get("max_for_loops")
                    or self.feedback.get("for_loops")
                    or self.feedback.get("for")
                )
                lineno = getattr(node, "lineno", None)
                if custom:
                    self.violations.append((True, custom, lineno))
                else:
                    loop_type = "async for-loops" if isinstance(node, ast.AsyncFor) else "for-loops"
                    self.violations.append(
                        (
                            False,
                            f"Exceeded maximum allowed {loop_type} ({self.for_count} > {self.max_for_loops}) at line {lineno}.",
                            lineno,
                        )
                    )
        elif isinstance(node, ast.While):
            self.while_count += 1
            if self.while_count > self.max_while_loops:
                custom = (
                    self.feedback.get("max_while_loops")
                    or self.feedback.get("while_loops")
                    or self.feedback.get("while")
                )
                lineno = getattr(node, "lineno", None)
                if custom:
                    self.violations.append((True, custom, lineno))
                else:
                    self.violations.append(
                        (
                            False,
                            f"Exceeded maximum allowed while-loops ({self.while_count} > {self.max_while_loops}) at line {lineno}.",
                            lineno,
                        )
                    )


class ForbidImportsRule(PolicyRule):
    """Rule forbidding import of specified modules or packages."""

    def __init__(
        self,
        forbid_imports: set[str] | None = None,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.forbid_imports = forbid_imports or set()

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Import, ast.ImportFrom)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in self.forbid_imports:
                    if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                        custom = (
                            self.feedback.get(alias.name)
                            or self.feedback.get(forbidden)
                            or self.feedback.get("forbid_imports")
                        )
                        lineno = getattr(node, "lineno", None)
                        if custom:
                            self.violations.append((True, custom, lineno))
                        else:
                            self.violations.append(
                                (
                                    False,
                                    f"Forbidden module import '{alias.name}' detected at line {lineno}.",
                                    lineno,
                                )
                            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in self.forbid_imports:
                if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                    custom = (
                        self.feedback.get(node.module)
                        or self.feedback.get(forbidden)
                        or self.feedback.get("forbid_imports")
                    )
                    lineno = getattr(node, "lineno", None)
                    if custom:
                        self.violations.append((True, custom, lineno))
                    else:
                        self.violations.append(
                            (
                                False,
                                f"Forbidden module import 'from {node.module}' detected at line {lineno}.",
                                lineno,
                            )
                        )


class ForbidFunctionImportsRule(PolicyRule):
    """Rule forbidding local imports inside function definitions."""

    def __init__(
        self,
        forbid_function_imports: bool = False,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.forbid_function_imports = forbid_function_imports

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Import, ast.ImportFrom)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if self.forbid_function_imports and context.in_function_depth > 0:
            custom = self.feedback.get("forbid_function_imports") or self.feedback.get(
                "no_function_imports"
            )
            lineno = getattr(node, "lineno", None)
            msg = (
                custom
                or f"Forbidden import inside function body at line {lineno}. Clean Lab forbids local function imports."
            )
            self.violations.append((bool(custom), msg, lineno))


class ForbidCallsRule(PolicyRule):
    """Rule forbidding execution of specified function or method calls."""

    def __init__(
        self,
        forbid_calls: set[str] | None = None,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.forbid_calls = forbid_calls or set()

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Call,)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if isinstance(node, ast.Call):
            call_name = _resolve_call_name(node.func)
            if call_name:
                for forbidden in self.forbid_calls:
                    if _call_matches(call_name, forbidden):
                        custom = (
                            self.feedback.get(call_name)
                            or self.feedback.get(forbidden)
                            or self.feedback.get("forbid_calls")
                        )
                        lineno = getattr(node, "lineno", None)
                        if custom:
                            self.violations.append((True, custom, lineno))
                        else:
                            self.violations.append(
                                (
                                    False,
                                    f"Forbidden function call '{call_name}' (matching rule '{forbidden}') detected at line {lineno}.",
                                    lineno,
                                )
                            )


class RequireCallsRule(PolicyRule):
    """Rule requiring execution of specified function or method calls."""

    def __init__(
        self,
        require_calls: set[str] | None = None,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.require_calls = require_calls or set()
        self.found_calls: set[str] = set()

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Call,)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if isinstance(node, ast.Call):
            call_name = _resolve_call_name(node.func)
            if call_name:
                self.found_calls.add(call_name)

    def check_post_traversal(self) -> None:
        for req in self.require_calls:
            if not any(_call_matches(actual, req) for actual in self.found_calls):
                custom = self.feedback.get(req) or self.feedback.get("require_calls")
                if custom:
                    self.violations.append((True, custom, None))
                else:
                    self.violations.append(
                        (
                            False,
                            f"Required function call matching '{req}' was not found in the implementation.",
                            None,
                        )
                    )


class ForbidHardcodedLiteralsRule(PolicyRule):
    """Rule forbidding hardcoded numeric literals in arithmetic calculations."""

    def __init__(
        self,
        forbid_hardcoded_literals: bool = False,
        feedback: dict[str, str] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.forbid_hardcoded_literals = forbid_hardcoded_literals

    def target_node_types(self) -> tuple[type[ast.AST], ...]:
        return (ast.Constant, ast.UnaryOp)

    def inspect_node(self, node: ast.AST, context: VisitorContext) -> None:
        if not self.forbid_hardcoded_literals:
            return
        if isinstance(node, ast.Constant):
            if (
                isinstance(node.value, (int, float, complex))
                and not isinstance(node.value, bool)
                and node.value not in (0, 1, -1, 0.0, 1.0, -1.0)
            ):
                custom = self.feedback.get("forbid_hardcoded_literals") or self.feedback.get(
                    "no_hardcoded_literals"
                )
                lineno = getattr(node, "lineno", None)
                msg = (
                    custom
                    or f"Forbidden hardcoded numeric literal '{node.value}' at line {lineno}. Require mathematical constants or parameters."
                )
                self.violations.append((bool(custom), msg, lineno))
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            val = -node.operand.value
            if val not in (0, 1, -1, 0.0, 1.0, -1.0):
                custom = self.feedback.get("forbid_hardcoded_literals") or self.feedback.get(
                    "no_hardcoded_literals"
                )
                lineno = getattr(node, "lineno", None)
                msg = (
                    custom
                    or f"Forbidden hardcoded numeric literal '{val}' at line {lineno}. Require mathematical constants or parameters."
                )
                self.violations.append((bool(custom), msg, lineno))


class ASTPolicyVisitor(ast.NodeVisitor):
    """AST Visitor that traverses syntax trees checking against defined policy rules."""

    def __init__(
        self,
        max_for_loops: int = 0,
        max_while_loops: int = 0,
        forbid_imports: Sequence[str] | None = None,
        require_calls: Sequence[str] | None = None,
        forbid_calls: Sequence[str] | None = None,
        forbid_function_imports: bool = False,
        forbid_hardcoded_literals: bool = False,
        feedback: dict[str, str] | None = None,
    ):
        self.feedback = feedback or {}
        self.context = VisitorContext()
        self.rules: list[PolicyRule] = [
            ForbidLoopsRule(max_for_loops, max_while_loops, self.feedback),
            ForbidImportsRule(set(forbid_imports or []), self.feedback),
            ForbidFunctionImportsRule(forbid_function_imports, self.feedback),
            ForbidCallsRule(set(forbid_calls or []), self.feedback),
            RequireCallsRule(set(require_calls or []), self.feedback),
            ForbidHardcodedLiteralsRule(forbid_hardcoded_literals, self.feedback),
        ]
        self._dispatch_map: dict[type[ast.AST], list[PolicyRule]] = {}
        for rule in self.rules:
            for node_type in rule.target_node_types():
                self._dispatch_map.setdefault(node_type, []).append(rule)

    @property
    def violations(self) -> list[tuple[bool, str, int | None]]:
        res: list[tuple[bool, str, int | None]] = []
        for rule in self.rules:
            res.extend(rule.violations)
        return res

    def check_requirements(self) -> None:
        for rule in self.rules:
            rule.check_post_traversal()

    def _dispatch(self, node: ast.AST) -> None:
        node_type = type(node)
        rules = self._dispatch_map.get(node_type)
        if rules is None:
            rules = [rule for rule in self.rules if isinstance(node, rule.target_node_types())]
            self._dispatch_map[node_type] = rules
        for rule in rules:
            rule.inspect_node(node, self.context)

    def visit(self, node: ast.AST) -> Any:
        self._dispatch(node)
        return super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.context.in_function_depth += 1
        self.generic_visit(node)
        self.context.in_function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.context.in_function_depth += 1
        self.generic_visit(node)
        self.context.in_function_depth -= 1


def _is_stub_node(node: ast.AST) -> bool:
    """Check if an AST node is a stub containing only docstrings, pass, or raise NotImplementedError."""
    if isinstance(node, ast.Module):
        if len(node.body) == 1 and isinstance(
            node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body[0].body
        else:
            body = node.body
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = node.body
    else:
        return False

    stmts = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        stmts.append(stmt)

    if not stmts:
        return True
    if len(stmts) == 1:
        s = stmts[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Raise):
            if isinstance(s.exc, ast.Call) and isinstance(s.exc.func, ast.Name):
                if s.exc.func.id == "NotImplementedError":
                    return True
            elif isinstance(s.exc, ast.Name) and s.exc.id == "NotImplementedError":
                return True
    return False


def _find_target_node_in_module(mod_ast: ast.Module, target_name: str) -> ast.AST | None:
    """Locate a FunctionDef or AsyncFunctionDef matching target_name inside a module AST."""
    for node in ast.walk(mod_ast):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
            return node
    return None


def ast_policy(
    max_for_loops: int = 0,
    max_while_loops: int = 0,
    forbid_imports: Sequence[str] | None = None,
    require_calls: Sequence[str] | None = None,
    forbid_calls: Sequence[str] | None = None,
    forbid_function_imports: bool = False,
    forbid_hardcoded_literals: bool = False,
    no_loops: bool | None = None,
    no_function_imports: bool | None = None,
    no_hardcoded_literals: bool | None = None,
    target: Callable[..., Any] | str | None = None,
    feedback: dict[str, str] | None = None,
) -> Callable[..., Any]:
    """Decorator to enforce static AST architectural constraints on student code."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_name = ""
            node_to_check: ast.AST | None = None

            if callable(target):
                target_name = getattr(
                    target, "__qualname__", getattr(target, "__name__", str(target))
                )
                try:
                    source = textwrap.dedent(inspect.getsource(target))
                    node_to_check = ast.parse(source)
                except (TypeError, OSError):
                    node_to_check = None
            elif isinstance(target, str):
                target_name = target
            else:
                raw_name = fn.__name__
                target_name = raw_name[5:] if raw_name.startswith("test_") else raw_name

            if node_to_check is None and "student_code" in sys.modules:
                student_mod = sys.modules["student_code"]
                try:
                    source = textwrap.dedent(inspect.getsource(student_mod))
                    mod_ast = ast.parse(source)
                    node_to_check = _find_target_node_in_module(mod_ast, target_name)
                    if node_to_check is None:
                        node_to_check = mod_ast
                        target_name = "student_code (module)"
                except (TypeError, OSError):
                    node_to_check = None

            if node_to_check is not None and not _is_stub_node(node_to_check):
                eff_for = 0 if (no_loops is True or max_for_loops == -1) else max_for_loops
                eff_while = 0 if (no_loops is True or max_while_loops == -1) else max_while_loops
                eff_func_imp = bool(no_function_imports is True or forbid_function_imports is True)
                eff_lit = bool(no_hardcoded_literals is True or forbid_hardcoded_literals is True)

                visitor = ASTPolicyVisitor(
                    max_for_loops=eff_for,
                    max_while_loops=eff_while,
                    forbid_imports=forbid_imports,
                    require_calls=require_calls,
                    forbid_calls=forbid_calls,
                    forbid_function_imports=eff_func_imp,
                    forbid_hardcoded_literals=eff_lit,
                    feedback=feedback,
                )
                visitor.visit(node_to_check)
                visitor.check_requirements()

                if visitor.violations:
                    formatted_items = []
                    first_lineno = None
                    target_file = ""
                    if callable(target):
                        with contextlib.suppress(TypeError, OSError):
                            target_file = inspect.getfile(target)
                    elif "student_code" in sys.modules:
                        with contextlib.suppress(Exception):
                            target_file = getattr(sys.modules["student_code"], "__file__", "")

                    for is_custom, msg, lineno in visitor.violations:
                        if first_lineno is None and lineno is not None:
                            first_lineno = lineno
                        if is_custom:
                            formatted_items.append(f"{_GOLD}{msg}{reset}")
                        else:
                            formatted_items.append(f"{_ERR}{msg}{reset}")
                    details_str = "\n  - ".join([""] + formatted_items)
                    msg = (
                        f"\n{_ERR}{'=' * 65}\n"
                        f"🛑 AST POLICY VIOLATION DETECTED\n"
                        f"{'=' * 65}{reset}\n"
                        f"  Target  : {_CYAN}{target_name}{reset}\n"
                        f"  File    : {_CYAN}{target_file}{reset}\n"
                        f"  Details :{details_str}\n"
                        f"{_ERR}{'-' * 65}\n"
                        f"{_GOLD}  💡 Hint: Clean Lab enforces architectural constraints using AST static\n"
                        f"           analysis. Please modify your implementation to comply with\n"
                        f"           the rule specified above.{reset}\n"
                        f"{_ERR}{'=' * 65}{reset}\n"
                    )
                    raise PolicyViolationError(msg, filename=target_file, lineno=first_lineno)

            return fn(*args, **kwargs)

        wrapper.__dict__["_ast_policy_spec"] = {
            "max_for_loops": max_for_loops,
            "max_while_loops": max_while_loops,
            "forbid_imports": forbid_imports,
            "require_calls": require_calls,
            "forbid_calls": forbid_calls,
            "forbid_function_imports": forbid_function_imports,
            "forbid_hardcoded_literals": forbid_hardcoded_literals,
        }
        return wrapper

    return decorator
