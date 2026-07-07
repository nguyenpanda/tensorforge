#!/usr/bin/env python3
"""
validate_baselines.py — TensorForge Full Baseline Validation
=============================================================
Runs three tiers of validation without needing student code.

Validation Tiers
----------------
Tier 1 — Baseline self-consistency
    Each baseline method is compared against itself using compare_and_benchmark().
    Correctness is guaranteed (same function), and performance ratio ≈ 1.0.

Tier 2 — Infrastructure integrity
    Verifies the benchmark system correctly detects wrong output and slow code.

Tier 3 — Registry completeness
    Confirms every expected lesson module and method is registered in the hint
    system using composite keys (``"<tier>/<lesson_dir>"``).

Lesson Path Convention
----------------------
All paths use Tier-Scoped Numbering: ``"<tier>/<lesson_dir>"``.

    basic/01_array_creation        (was 00_array_creation)
    basic/02_properties_reshaping  (was 01_properties_reshaping)
    basic/03_indexing_slicing      (was 02_indexing_slicing)
    intermediate/01_vectorized_math    (was 03_vectorized_math)
    intermediate/02_array_manipulation (was 04_array_manipulation)
    intermediate/03_boolean_logic      (was 05_boolean_logic)
    advanced/01_memory_layout          (was 06_memory_layout)

Usage:
    uv run python validate_baselines.py
    uv run python validate_baselines.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import sys
import textwrap
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: make project root importable
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from nguyenpanda.swan import c24, reset

from forge_core.benchmark import BenchmarkConfig, BenchmarkResult, compare_and_benchmark

_GREEN = c24["00ff87"]
_RED = c24["ff5f5f"]
_YELLOW = c24["ffd700"]
_CYAN = c24["00d7ff"]
_BOLD = "\033[1m"
_RESET = reset


def _green(s: str) -> str:
    return f"{_GREEN}{s}{_RESET}"


def _red(s: str) -> str:
    return f"{_RED}{s}{_RESET}"


def _yellow(s: str) -> str:
    return f"{_YELLOW}{s}{_RESET}"


def _cyan(s: str) -> str:
    return f"{_CYAN}{s}{_RESET}"


def _bold(s: str) -> str:
    return f"{_BOLD}{s}{_RESET}"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    detail: str = ""
    timing: str | None = None


@dataclass
class ValidationReport:
    """Aggregated results for one validation tier."""

    tier_name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when every check in this tier passed."""
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        """Return the number of passing checks."""
        return sum(1 for r in self.results if r.passed)


def _load_baseline(tier_lesson_path: str, class_name: str) -> Any:
    """Import a baseline class dynamically via sys.path modification.

    Args:
        tier_lesson_path: Tier-scoped path (e.g. ``"basic/01_array_creation"``).
        class_name: The name of the baseline class inside ``_baseline.py``.

    Returns:
        Any: The baseline class object.
    """
    test_path = str(ROOT / "tests" / "curriculum" / "arraysmith" / tier_lesson_path)
    student_path = str(ROOT / "arraysmith" / tier_lesson_path)

    # Evict stale entries for shared module names.
    for name in ("_baseline", "student_code"):
        sys.modules.pop(name, None)

    for path in (test_path, student_path):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, student_path)
    sys.path.insert(0, test_path)

    mod = importlib.import_module("_baseline")
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Tier 1 — Baseline self-consistency
# ---------------------------------------------------------------------------

# Fixed-seed inputs shared across exercises to keep the validation deterministic.

_RNG_03 = np.random.default_rng(seed=7)
_MAT_03 = _RNG_03.uniform(-100.0, 100.0, size=(50, 8))

_RNG_04 = np.random.default_rng(seed=13)
_ROWS_04 = [_RNG_04.integers(0, 100, size=5, dtype=np.int32) for _ in range(3)]
_LEFT_04 = _RNG_04.integers(0, 50, size=(4, 3), dtype=np.int32)
_RIGHT_04 = _RNG_04.integers(0, 50, size=(4, 5), dtype=np.int32)

_RNG_02 = np.random.default_rng(seed=42)
_ARR_02 = _RNG_02.integers(-50, 51, size=200, dtype=np.int32)
_IDX_02 = np.array([7, 0, 15, 3, 7, 99, 42, 128, 0, 199], dtype=np.int64)

_RNG_05 = np.random.default_rng(seed=101)
_PRICES_05 = _RNG_05.uniform(10.0, 500.0, size=(100, 100)).astype(np.float64)
_MAT_A_05 = _RNG_05.uniform(-10.0, 10.0, size=(100, 20)).astype(np.float64)
_MAT_B_05 = _MAT_A_05.copy()
_MAT_B_05[_RNG_05.choice(100, size=50, replace=False)] += 1.0

_RNG_06 = np.random.default_rng(seed=202)
_ARR_06 = _RNG_06.uniform(0.0, 10.0, size=(50, 50))
_ARR_06_F = np.asfortranarray(_ARR_06)

# Each entry: tier_lesson_path, class_name, method, fn_factory(cls) → zero-arg callable.
TIER1_EXERCISES: list[dict[str, Any]] = [
    # basic/01_array_creation
    {
        "tier_lesson_path": "basic/01_array_creation",
        "class_name": "ArrayCreationBaseline",
        "method": "create_integer_range",
        "fn_factory": lambda cls: lambda: cls.create_integer_range(),
    },
    {
        "tier_lesson_path": "basic/01_array_creation",
        "class_name": "ArrayCreationBaseline",
        "method": "create_squared_range",
        "fn_factory": lambda cls: lambda: cls.create_squared_range(),
    },
    # basic/02_properties_reshaping
    {
        "tier_lesson_path": "basic/02_properties_reshaping",
        "class_name": "PropertiesReshapingBaseline",
        "method": "reshape_to_matrix",
        "fn_factory": lambda cls: lambda: cls.reshape_to_matrix(np.arange(12, dtype=np.int32)),
    },
    {
        "tier_lesson_path": "basic/02_properties_reshaping",
        "class_name": "PropertiesReshapingBaseline",
        "method": "flatten_and_cast",
        "fn_factory": lambda cls: (
            lambda: cls.flatten_and_cast(np.arange(12, dtype=np.int32).reshape(2, 6))
        ),
    },
    # basic/03_indexing_slicing
    {
        "tier_lesson_path": "basic/03_indexing_slicing",
        "class_name": "IndexingSlicingBaseline",
        "method": "filter_above_threshold",
        "fn_factory": lambda cls: lambda: cls.filter_above_threshold(_ARR_02.copy(), threshold=0.0),
    },
    {
        "tier_lesson_path": "basic/03_indexing_slicing",
        "class_name": "IndexingSlicingBaseline",
        "method": "gather_by_indices",
        "fn_factory": lambda cls: lambda: cls.gather_by_indices(_ARR_02.copy(), _IDX_02.copy()),
    },
    # intermediate/01_vectorized_math
    {
        "tier_lesson_path": "intermediate/01_vectorized_math",
        "class_name": "VectorizedMathBaseline",
        "method": "row_means",
        "fn_factory": lambda cls: lambda: cls.row_means(_MAT_03.copy()),
    },
    {
        "tier_lesson_path": "intermediate/01_vectorized_math",
        "class_name": "VectorizedMathBaseline",
        "method": "normalize_columns",
        "fn_factory": lambda cls: lambda: cls.normalize_columns(_MAT_03.copy()),
    },
    # intermediate/02_array_manipulation
    {
        "tier_lesson_path": "intermediate/02_array_manipulation",
        "class_name": "ArrayManipulationBaseline",
        "method": "stack_rows",
        "fn_factory": lambda cls: lambda: cls.stack_rows([a.copy() for a in _ROWS_04]),
    },
    {
        "tier_lesson_path": "intermediate/02_array_manipulation",
        "class_name": "ArrayManipulationBaseline",
        "method": "concatenate_side_by_side",
        "fn_factory": lambda cls: (
            lambda: cls.concatenate_side_by_side(_LEFT_04.copy(), _RIGHT_04.copy())
        ),
    },
    # intermediate/03_boolean_logic
    {
        "tier_lesson_path": "intermediate/03_boolean_logic",
        "class_name": "BooleanLogicBaseline",
        "method": "apply_discount",
        "fn_factory": lambda cls: lambda: cls.apply_discount(_PRICES_05.copy()),
    },
    {
        "tier_lesson_path": "intermediate/03_boolean_logic",
        "class_name": "BooleanLogicBaseline",
        "method": "identify_close_rows",
        "fn_factory": lambda cls: (
            lambda: cls.identify_close_rows(_MAT_A_05.copy(), _MAT_B_05.copy())
        ),
    },
    # advanced/01_memory_layout
    {
        "tier_lesson_path": "advanced/01_memory_layout",
        "class_name": "MemoryLayoutBaseline",
        "method": "ensure_c_contiguous",
        "fn_factory": lambda cls: lambda: cls.ensure_c_contiguous(_ARR_06_F.copy()),
    },
    {
        "tier_lesson_path": "advanced/01_memory_layout",
        "class_name": "MemoryLayoutBaseline",
        "method": "check_memory_share",
        "fn_factory": lambda cls: lambda: cls.check_memory_share(_ARR_06, _ARR_06[:10]),
    },
    {
        "tier_lesson_path": "advanced/01_memory_layout",
        "class_name": "MemoryLayoutBaseline",
        "method": "get_row_stride_bytes",
        "fn_factory": lambda cls: lambda: cls.get_row_stride_bytes(_ARR_06),
    },
]


def _verify_baseline_ast_policy(
    tier_lesson_path: str, class_name: str, method_name: str, baseline_cls: Any
) -> None:
    """Verify that the baseline solution complies with any ast_policy defined in its test suite."""
    test_dir = ROOT / "tests" / "curriculum" / "arraysmith" / tier_lesson_path
    test_files = list(test_dir.glob("test_*.py"))
    if not test_files:
        return

    test_path = test_files[0]
    mod_name = test_path.stem
    if mod_name in sys.modules:
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(mod_name, test_path)
    if spec is None or spec.loader is None:
        return
    test_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_mod)

    test_cls = None
    for attr_name in dir(test_mod):
        if attr_name.startswith("Test") and isinstance(getattr(test_mod, attr_name), type):
            test_cls = getattr(test_mod, attr_name)
            break

    if test_cls is None:
        return

    test_fn = getattr(test_cls, f"test_{method_name}", None)
    if test_fn is None:
        return

    policy_spec = getattr(test_fn, "_ast_policy_spec", None)
    if policy_spec is None:
        return

    from forge_core.ast_validator import ASTPolicyVisitor

    baseline_fn = getattr(baseline_cls, method_name)
    source = textwrap.dedent(inspect.getsource(baseline_fn))
    node_to_check = ast.parse(source)

    visitor = ASTPolicyVisitor(
        max_for_loops=policy_spec.get("max_for_loops", 0),
        max_while_loops=policy_spec.get("max_while_loops", 0),
        forbid_imports=policy_spec.get("forbid_imports"),
        require_calls=policy_spec.get("require_calls"),
        forbid_calls=policy_spec.get("forbid_calls"),
    )
    visitor.visit(node_to_check)
    visitor.check_requirements()

    if visitor.violations:
        violations_str = ", ".join(msg for _, msg, _ in visitor.violations)
        raise RuntimeError(
            f"Baseline {class_name}.{method_name} violated AST policy: {violations_str}"
        )


def run_tier1(verbose: bool) -> ValidationReport:
    """Execute Tier 1 — Baseline self-consistency checks.

    Each baseline method is called with itself as both student and baseline.
    Correctness is trivially guaranteed; the check validates the benchmarking
    harness can load and execute all baselines without error.

    Args:
        verbose: When True, print full tracebacks for failures.

    Returns:
        ValidationReport: Results for every exercise.
    """
    report = ValidationReport("Tier 1 — Baseline Self-Consistency")
    config = BenchmarkConfig(n_runs=20, check_performance=False, silent=True)

    for ex in TIER1_EXERCISES:
        label = f"{ex['tier_lesson_path']}::{ex['method']}"
        try:
            cls = _load_baseline(ex["tier_lesson_path"], ex["class_name"])
            _verify_baseline_ast_policy(ex["tier_lesson_path"], ex["class_name"], ex["method"], cls)
            fn = ex["fn_factory"](cls)
            result: BenchmarkResult = compare_and_benchmark(
                student_fn=fn,
                baseline_fn=fn,
                config=config,
            )
            timing = f"{result.baseline_time_ms:.4f} ms | {result.baseline_peak_kb:.2f} KB"
            report.results.append(CheckResult(label, True, timing=timing))
        except Exception as exc:
            detail = traceback.format_exc() if verbose else str(exc)
            report.results.append(CheckResult(label, False, detail=detail))

    return report


# ---------------------------------------------------------------------------
# Tier 2 — Infrastructure integrity
# ---------------------------------------------------------------------------


def _infra_correctness_detection() -> CheckResult:
    """Verify that compare_and_benchmark raises on wrong output.

    Returns:
        CheckResult: Passed if a correctness error is correctly raised.
    """
    name = "Benchmark detects wrong output"
    cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
    try:
        compare_and_benchmark(
            student_fn=lambda: np.zeros(100, dtype=np.int8),
            baseline_fn=lambda: cls.create_integer_range(),
            config=BenchmarkConfig(n_runs=5, check_performance=False, silent=True),
        )
        return CheckResult(name, False, "No error raised for wrong output — INFRA BUG")
    except BaseException:
        return CheckResult(name, True)


def _infra_performance_detection() -> CheckResult:
    """Verify that compare_and_benchmark raises when code is too slow.

    Returns:
        CheckResult: Passed if a performance violation is correctly raised.
    """
    name = "Benchmark detects slow code (loop)"
    cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")

    def slow_student() -> Any:
        result = []
        for i in range(100):
            result.append(i)
        return np.array(result, dtype=np.int8)

    try:
        compare_and_benchmark(
            student_fn=slow_student,
            baseline_fn=lambda: cls.create_integer_range(),
            config=BenchmarkConfig(n_runs=50, slowdown_threshold=0.001, silent=True),
        )
        return CheckResult(name, False, "Slow code was not flagged — INFRA BUG")
    except BaseException:
        return CheckResult(name, True)


def _infra_dtype_int_check() -> CheckResult:
    """Verify integer dtype uses assert_array_equal (exact), not allclose.

    Returns:
        CheckResult: Passed if an off-by-one integer value is caught.
    """
    name = "Integer arrays use exact comparison"
    cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")

    def slightly_wrong() -> Any:
        arr = cls.create_integer_range().copy().astype(np.int16)
        arr[50] = 51
        return arr.astype(np.int8)

    try:
        compare_and_benchmark(
            student_fn=slightly_wrong,
            baseline_fn=lambda: cls.create_integer_range(),
            config=BenchmarkConfig(n_runs=5, check_performance=False, silent=True),
        )
        return CheckResult(name, False, "Off-by-one integer not caught — INFRA BUG")
    except BaseException:
        return CheckResult(name, True)


def run_tier2(verbose: bool) -> ValidationReport:
    """Execute Tier 2 — Infrastructure integrity checks.

    Args:
        verbose: When True, print full tracebacks for failures.

    Returns:
        ValidationReport: Results for each infrastructure check.
    """
    report = ValidationReport("Tier 2 — Infrastructure Integrity")
    checks: list[Callable[[], CheckResult]] = [
        _infra_correctness_detection,
        _infra_performance_detection,
        _infra_dtype_int_check,
    ]
    for check in checks:
        try:
            report.results.append(check())
        except BaseException as exc:
            detail = traceback.format_exc() if verbose else str(exc)
            report.results.append(CheckResult(check.__name__, False, detail=detail))
    return report


# ---------------------------------------------------------------------------
# Tier 3 — Registry completeness
# ---------------------------------------------------------------------------

# Composite key → expected method names.
_EXPECTED_REGISTRY: dict[str, list[str]] = {
    "basic/01_array_creation": ["create_integer_range", "create_squared_range"],
    "basic/02_properties_reshaping": ["reshape_to_matrix", "flatten_and_cast"],
    "basic/03_indexing_slicing": ["filter_above_threshold", "gather_by_indices"],
    "intermediate/01_vectorized_math": ["row_means", "normalize_columns"],
    "intermediate/02_array_manipulation": ["stack_rows", "concatenate_side_by_side"],
    "intermediate/03_boolean_logic": ["apply_discount", "identify_close_rows"],
    "advanced/01_memory_layout": [
        "ensure_c_contiguous",
        "check_memory_share",
        "get_row_stride_bytes",
    ],
    "basic/01_cpp_integration": ["run_cpp_addition"],
    "intermediate/01_cuda_gemm": ["run_naive_gemm", "run_tiled_gemm"],
}

# Composite key → expected student class name (without "Baseline" suffix).
_EXPECTED_CLASS_NAMES: dict[str, str] = {
    "basic/01_array_creation": "ArrayCreation",
    "basic/02_properties_reshaping": "PropertiesReshaping",
    "basic/03_indexing_slicing": "IndexingSlicing",
    "intermediate/01_vectorized_math": "VectorizedMath",
    "intermediate/02_array_manipulation": "ArrayManipulation",
    "intermediate/03_boolean_logic": "BooleanLogic",
    "advanced/01_memory_layout": "MemoryLayout",
    "basic/01_cpp_integration": "CppIntegration",
    "intermediate/01_cuda_gemm": "CudaGemm",
}


def run_tier3(verbose: bool) -> ValidationReport:
    """Execute Tier 3 — Hint registry completeness checks.

    Verifies that every expected lesson, class, and method is present in the
    auto-discovered ``HINTS_REGISTRY`` with non-empty ``major`` and ``minor``
    hint strings.

    Args:
        verbose: Unused; retained for API consistency.

    Returns:
        ValidationReport: Results for each registry entry.
    """
    report = ValidationReport("Tier 3 — Registry Completeness")

    from hint import HINTS_REGISTRY

    registry = HINTS_REGISTRY

    for module_key, expected_methods in _EXPECTED_REGISTRY.items():
        class_name = _EXPECTED_CLASS_NAMES[module_key]

        if module_key not in registry:
            report.results.append(
                CheckResult(
                    f"registry['{module_key}'] present",
                    False,
                    f"Module '{module_key}' missing from hint registry",
                )
            )
            continue
        report.results.append(CheckResult(f"registry['{module_key}'] present", True))

        if class_name not in registry[module_key]:
            report.results.append(
                CheckResult(
                    f"  └─ class '{class_name}'",
                    False,
                    f"Class '{class_name}' missing under registry['{module_key}']",
                )
            )
            continue
        report.results.append(CheckResult(f"  └─ class '{class_name}'", True))

        for method in expected_methods:
            name_mth = f"       └─ method '{method}'"
            if method not in registry[module_key][class_name]:
                report.results.append(CheckResult(name_mth, False, "Method missing"))
            else:
                hint = registry[module_key][class_name][method]
                has_major = bool(hint.get("major", "").strip())
                has_minor = bool(hint.get("minor", "").strip())
                detail = "" if (has_major and has_minor) else _yellow("⚠ empty hint text")
                report.results.append(CheckResult(name_mth, True, detail=detail))

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(report: ValidationReport, verbose: bool) -> None:
    """Print a formatted report for one validation tier.

    Args:
        report: The :class:`ValidationReport` to display.
        verbose: When True, print detail strings for passing checks too.
    """
    sep = "─" * 65
    print(f"\n{_bold(_cyan(sep))}")
    print(_bold(f"  {report.tier_name}"))
    print(_bold(_cyan(sep)))
    for r in report.results:
        icon = _green("✅") if r.passed else _red("❌")
        timing = f"  {_yellow(r.timing)}" if r.timing else ""
        print(f"  {icon}  {r.name}{timing}")
        if r.detail and (not r.passed or verbose):
            for line in r.detail.splitlines():
                print(f"       {_yellow(line) if r.passed else _red(line)}")
    total = len(report.results)
    passed = report.pass_count
    colour = _green if report.passed else _red
    print(colour(f"\n  {passed}/{total} checks passed"))


def _print_summary(reports: list[ValidationReport]) -> None:
    """Print the overall pass/fail summary banner.

    Args:
        reports: All :class:`ValidationReport` objects from the run.
    """
    sep = "=" * 65
    all_passed = all(r.passed for r in reports)
    colour = _green if all_passed else _red
    print(f"\n{colour(_bold(sep))}")
    if all_passed:
        print(colour(_bold("  🎉  ALL VALIDATION CHECKS PASSED")))
    else:
        print(colour(_bold("  ⚠   SOME CHECKS FAILED — see details above")))
    total = sum(len(r.results) for r in reports)
    passed = sum(r.pass_count for r in reports)
    print(colour(f"  {passed}/{total} total checks passed"))
    print(colour(_bold(sep)) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all three validation tiers and return exit code.

    Returns:
        int: 0 if every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="TensorForge baseline validation script")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full tracebacks for failed checks",
    )
    args = parser.parse_args()

    print(_bold(_cyan("\n" + "=" * 65)))
    print(_bold(_cyan("  TensorForge — Baseline Validation")))
    print(_bold(_cyan("=" * 65)))

    reports = [
        run_tier1(args.verbose),
        run_tier2(args.verbose),
        run_tier3(args.verbose),
    ]

    for report in reports:
        _print_report(report, args.verbose)

    _print_summary(reports)

    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
