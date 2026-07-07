"""
forge_core/ast_validator.py
===========================
Dynamic AST static analysis engine for Clean Lab architectural policy enforcement.

Provides the :func:`ast_policy` decorator to scan student implementations for
forbidden constructs (loops, disallowed calls, disallowed imports) or missing
required calls before execution or benchmarking occurs.
"""

from __future__ import annotations

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


class ASTPolicyVisitor(ast.NodeVisitor):
    """AST Visitor that traverses syntax trees checking against defined rules."""

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
        self.max_for_loops = max_for_loops
        self.max_while_loops = max_while_loops
        self.forbid_imports = set(forbid_imports or [])
        self.require_calls = set(require_calls or [])
        self.forbid_calls = set(forbid_calls or [])
        self.forbid_function_imports = forbid_function_imports
        self.forbid_hardcoded_literals = forbid_hardcoded_literals
        self.feedback = feedback or {}

        self.for_count = 0
        self.while_count = 0
        self.found_calls: set[str] = set()
        self.violations: list[tuple[bool, str, int | None]] = []
        self._in_function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._in_function_depth += 1
        self.generic_visit(node)
        self._in_function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._in_function_depth += 1
        self.generic_visit(node)
        self._in_function_depth -= 1

    def visit_For(self, node: ast.For) -> Any:
        self.for_count += 1
        if self.for_count > self.max_for_loops:
            custom = (
                self.feedback.get("max_for_loops")
                or self.feedback.get("for_loops")
                or self.feedback.get("for")
            )
            if custom:
                self.violations.append((True, custom, node.lineno))
            else:
                self.violations.append(
                    (
                        False,
                        f"Exceeded maximum allowed for-loops ({self.for_count} > {self.max_for_loops}) at line {node.lineno}.",
                        node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.for_count += 1
        if self.for_count > self.max_for_loops:
            custom = (
                self.feedback.get("max_for_loops")
                or self.feedback.get("for_loops")
                or self.feedback.get("for")
            )
            if custom:
                self.violations.append((True, custom, node.lineno))
            else:
                self.violations.append(
                    (
                        False,
                        f"Exceeded maximum allowed async for-loops ({self.for_count} > {self.max_for_loops}) at line {node.lineno}.",
                        node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.while_count += 1
        if self.while_count > self.max_while_loops:
            custom = (
                self.feedback.get("max_while_loops")
                or self.feedback.get("while_loops")
                or self.feedback.get("while")
            )
            if custom:
                self.violations.append((True, custom, node.lineno))
            else:
                self.violations.append(
                    (
                        False,
                        f"Exceeded maximum allowed while-loops ({self.while_count} > {self.max_while_loops}) at line {node.lineno}.",
                        node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        if self.forbid_function_imports and self._in_function_depth > 0:
            custom = self.feedback.get("forbid_function_imports") or self.feedback.get(
                "no_function_imports"
            )
            msg = (
                custom
                or f"Forbidden import inside function body at line {node.lineno}. Clean Lab forbids local function imports."
            )
            self.violations.append((bool(custom), msg, node.lineno))
        for alias in node.names:
            for forbidden in self.forbid_imports:
                if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                    custom = (
                        self.feedback.get(alias.name)
                        or self.feedback.get(forbidden)
                        or self.feedback.get("forbid_imports")
                    )
                    if custom:
                        self.violations.append((True, custom, node.lineno))
                    else:
                        self.violations.append(
                            (
                                False,
                                f"Forbidden module import '{alias.name}' detected at line {node.lineno}.",
                                node.lineno,
                            )
                        )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if self.forbid_function_imports and self._in_function_depth > 0:
            custom = self.feedback.get("forbid_function_imports") or self.feedback.get(
                "no_function_imports"
            )
            msg = (
                custom
                or f"Forbidden import inside function body at line {node.lineno}. Clean Lab forbids local function imports."
            )
            self.violations.append((bool(custom), msg, node.lineno))
        if node.module:
            for forbidden in self.forbid_imports:
                if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                    custom = (
                        self.feedback.get(node.module)
                        or self.feedback.get(forbidden)
                        or self.feedback.get("forbid_imports")
                    )
                    if custom:
                        self.violations.append((True, custom, node.lineno))
                    else:
                        self.violations.append(
                            (
                                False,
                                f"Forbidden module import 'from {node.module}' detected at line {node.lineno}.",
                                node.lineno,
                            )
                        )
        self.generic_visit(node)

    def _resolve_call_name(self, func_node: ast.AST) -> str:
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            val_name = self._resolve_call_name(func_node.value)
            return f"{val_name}.{func_node.attr}" if val_name else func_node.attr
        return ""

    def _call_matches(self, actual: str, pattern: str) -> bool:
        if not actual or not pattern:
            return False
        if actual == pattern:
            return True
        pattern_attr = pattern.split(".")[-1]
        actual_attr = actual.split(".")[-1]
        return pattern_attr == actual_attr

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = self._resolve_call_name(node.func)
        if call_name:
            self.found_calls.add(call_name)
            for forbidden in self.forbid_calls:
                if self._call_matches(call_name, forbidden):
                    custom = (
                        self.feedback.get(call_name)
                        or self.feedback.get(forbidden)
                        or self.feedback.get("forbid_calls")
                    )
                    if custom:
                        self.violations.append((True, custom, node.lineno))
                    else:
                        self.violations.append(
                            (
                                False,
                                f"Forbidden function call '{call_name}' (matching rule '{forbidden}') detected at line {node.lineno}.",
                                node.lineno,
                            )
                        )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if (
            self.forbid_hardcoded_literals
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
            and node.value not in (0, 1, -1, 0.0, 1.0, -1.0)
        ):
            custom = self.feedback.get("forbid_hardcoded_literals") or self.feedback.get(
                "no_hardcoded_literals"
            )
            msg = (
                custom
                or f"Forbidden hardcoded numeric literal '{node.value}' at line {node.lineno}. Require mathematical constants or parameters."
            )
            self.violations.append((bool(custom), msg, getattr(node, "lineno", None)))
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if (
            self.forbid_hardcoded_literals
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
                msg = (
                    custom
                    or f"Forbidden hardcoded numeric literal '{val}' at line {node.lineno}. Require mathematical constants or parameters."
                )
                self.violations.append((bool(custom), msg, getattr(node, "lineno", None)))
            return
        self.generic_visit(node)

    def check_requirements(self) -> None:
        for req in self.require_calls:
            if not any(self._call_matches(actual, req) for actual in self.found_calls):
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


def _is_stub_node(node: ast.AST) -> bool:
    """Check if an AST node is just a stub containing only docstrings, pass, or raise NotImplementedError."""
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
