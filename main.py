"""
main.py — TensorForge CLI Entry Point
=====================================
Provides a command-line interface for checking student solutions and validating
project baselines in a Clean Lab architecture with Tier-Scoped Numbering.

The ``arraysmith`` curriculum is organised into semantic tiers
(``basic``, ``intermediate``, ``advanced``, ``applications``), each with
lesson IDs that reset at ``01`` per tier, eliminating the sequential-numbering
ripple-effect present in flat curricula.

Usage Examples
--------------
Check a lesson:
    tforge check arraysmith basic 01
    tforge check arraysmith basic 01 create_squared_range
    tforge check arraysmith intermediate
    tforge check arraysmith

Run infrastructure unit tests:
    tforge check infra

Run all tests:
    tforge check all

Validate baselines:
    tforge validate

Show curriculum progress:
    tforge status

Scaffold a new lesson:
    tforge generate arraysmith basic 02_new_lesson
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from nguyenpanda.swan import c24, reset

ROOT = Path(__file__).parent.resolve()

ERR_COLOR  = c24["ff3333"]
INFO_COLOR = c24["00d7ff"]
GREEN_COLOR = c24["00ff87"]
WARN_COLOR = c24["ff8800"]
BOLD = "\033[1m"

# Ordered tier names used for consistent display.
ARRAYSMITH_TIERS = ["basic", "intermediate", "advanced", "applications"]


def _ensure_torch() -> None:
    """Guard against missing optional ``torch`` dependency.

    When a ``tensorsmith`` command is requested but the ``torch`` extra has not
    been installed, print a descriptive warning via swan and terminate with
    exit code 1.  This prevents an ImportError traceback from confusing the
    student with an irrelevant stack trace.
    """
    if importlib.util.find_spec("torch") is None:
        print(
            f"{WARN_COLOR}⚠  Missing dependency: 'torch' is not installed.{reset}\n"
            f"   Run {INFO_COLOR}`uv sync --extra torch`{reset} to enable "
            f"tensorsmith commands.",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_target(
    curriculum: str,
    tier: str | None,
    lesson: str | None,
    method: str | None = None,
) -> list[str]:
    """Resolve CLI arguments to pytest target paths.

    Args:
        curriculum: Curriculum name (``"arraysmith"``, ``"tensorsmith"``,
            ``"infra"``, or ``"all"``).
        tier: Optional tier name (``"basic"``, ``"intermediate"``, etc.).
            When ``None``, the entire curriculum is tested.
        lesson: Optional lesson ID or prefix (e.g. ``"01"``).  When ``None``,
            the entire tier is tested.
        method: Optional pytest ``-k`` filter for a specific method name.

    Returns:
        list[str]: Argument list suitable for ``python -m pytest``.
    """
    if curriculum == "all":
        args = ["tests/", "-v", "-s"]
    elif curriculum == "infra":
        args = ["tests/test_infrastructure.py", "-v", "-s"]
    else:
        curriculum_dir = ROOT / "tests" / curriculum
        if not curriculum_dir.is_dir():
            print(
                f"{ERR_COLOR}Error: Curriculum directory not found: "
                f"tests/{curriculum}{reset}",
                file=sys.stderr,
            )
            sys.exit(1)

        if not tier or tier == "all":
            # Run the full curriculum.
            args = [f"tests/{curriculum}", "-v", "-s"]
        else:
            tier_dir = curriculum_dir / tier
            if not tier_dir.is_dir():
                available = [d.name for d in curriculum_dir.iterdir() if d.is_dir() and not d.name.startswith("__")]
                print(
                    f"{ERR_COLOR}Error: Tier '{tier}' not found in tests/{curriculum}.\n"
                    f"Available tiers: {', '.join(sorted(available))}{reset}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not lesson or lesson == "all":
                # Run the full tier.
                args = [str(tier_dir.relative_to(ROOT)), "-v", "-s"]
            else:
                # Resolve lesson by exact name or leading-ID prefix.
                matches = [
                    d for d in tier_dir.iterdir()
                    if d.is_dir() and (d.name == lesson or d.name.startswith(f"{lesson}_"))
                ]
                if not matches:
                    print(
                        f"{ERR_COLOR}Error: No lesson matching '{lesson}' found in "
                        f"tests/{curriculum}/{tier}{reset}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if len(matches) > 1:
                    names = ", ".join(m.name for m in matches)
                    print(
                        f"{ERR_COLOR}Error: Ambiguous lesson prefix '{lesson}'. "
                        f"Matches: {names}{reset}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                matched_dir = matches[0]
                args = [str(matched_dir.relative_to(ROOT)), "-v", "-s"]

    if method:
        args.extend(["-k", method])
    return args


def run_check(
    curriculum: str,
    tier: str | None,
    lesson: str | None,
    method: str | None = None,
) -> int:
    """Run pytest against the specified curriculum, tier, and lesson.

    Args:
        curriculum: Curriculum identifier.
        tier: Tier name (``"basic"``, ``"intermediate"``, etc.) or ``None``.
        lesson: Lesson ID/prefix or ``None``.
        method: Optional ``-k`` filter for a specific test method.

    Returns:
        int: pytest return code (0 = all passed).
    """
    pytest_args = _resolve_target(curriculum, tier, lesson, method)
    print(f"{INFO_COLOR}Running: pytest {' '.join(pytest_args)}{reset}\n")
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    result = subprocess.run(cmd)
    return result.returncode


def run_validate() -> int:
    """Run the multi-tier baseline validation script.

    Returns:
        int: Script return code (0 = all passed).
    """
    cmd = [sys.executable, str(ROOT / "validate_baselines.py")]
    result = subprocess.run(cmd)
    return result.returncode


def run_status() -> int:
    """Run tests silently and output a tier-grouped colorized progress report.

    Iterates over the canonical tier order (basic → intermediate → advanced →
    applications), printing a pass/fail status line per lesson within each tier.
    A final progress bar summarises overall curriculum completion.

    Returns:
        int: 0 always (errors are displayed inline).
    """
    print(f"{INFO_COLOR}Scanning curriculum progress...{reset}\n")
    tests_dir = ROOT / "tests" / "arraysmith"
    if not tests_dir.is_dir():
        print(
            f"{ERR_COLOR}Error: Curriculum directory not found: {tests_dir}{reset}",
            file=sys.stderr,
        )
        return 1

    total_lessons = 0
    passed_lessons = 0

    print(f"{BOLD}{INFO_COLOR}  Module Progress Report — arraysmith{reset}")
    print(f"{INFO_COLOR}  {'─' * 50}{reset}")

    for tier_name in ARRAYSMITH_TIERS:
        tier_dir = tests_dir / tier_name
        if not tier_dir.is_dir():
            continue

        lessons = sorted([
            d for d in tier_dir.iterdir()
            if d.is_dir() and not d.name.startswith("__")
        ])
        if not lessons:
            continue

        print(f"\n  {BOLD}{INFO_COLOR}[{tier_name.upper()}]{reset}")

        for lesson in lessons:
            total_lessons += 1
            cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", str(lesson)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                passed_lessons += 1
                status_str = f"{GREEN_COLOR}✅ PASSED{reset}"
            else:
                status_str = f"{WARN_COLOR}░░ PENDING / FAILED{reset}"
            print(f"    {lesson.name:<32} {status_str}")

    print(f"\n{INFO_COLOR}  {'─' * 50}{reset}")

    pct = int((passed_lessons / total_lessons) * 100) if total_lessons > 0 else 0
    bar_len = 10
    filled = int(bar_len * (passed_lessons / total_lessons)) if total_lessons > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    bar_color = GREEN_COLOR if pct == 100 else (WARN_COLOR if pct >= 50 else ERR_COLOR)
    print(f"\n  {BOLD}Curriculum Status: {bar_color}[{bar}] {pct}% Passed{reset}\n")
    return 0


def _to_pascal_case(name: str) -> str:
    """Convert a lesson directory name to PascalCase for class naming.

    Args:
        name: Lesson directory name (e.g. ``"01_array_creation"``).

    Returns:
        str: PascalCase string (e.g. ``"ArrayCreation"``).
    """
    clean = re.sub(r"^\d+_?", "", name)
    return "".join(w.capitalize() for w in clean.split("_") if w)


def run_generate(module: str, tier: str, lesson_name: str) -> int:
    """Dynamically scaffold a new lesson directory and mirror it in tests/.

    Creates the student workspace under ``<module>/<tier>/<lesson_name>/``
    and the mirrored test workspace under ``tests/<module>/<tier>/<lesson_name>/``.
    Generates ``student_code.py``, ``INSTRUCTION.md``, ``_baseline.py``, and
    ``test_XX.py`` from canonical templates.

    Args:
        module: Curriculum module name (e.g. ``"arraysmith"``).
        tier: Tier name (e.g. ``"basic"``).
        lesson_name: Lesson folder name (e.g. ``"04_new_topic"``).

    Returns:
        int: 0 on success, 1 if the directory already exists.
    """
    student_dir = ROOT / module / tier / lesson_name
    test_dir = ROOT / "tests" / module / tier / lesson_name

    if student_dir.exists() or test_dir.exists():
        print(
            f"{ERR_COLOR}Error: Lesson directory '{lesson_name}' already exists in "
            f"{module}/{tier} or tests/{module}/{tier}.{reset}",
            file=sys.stderr,
        )
        return 1

    student_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    cls_name = _to_pascal_case(lesson_name)
    prefix_match = re.match(r"^(\d+)", lesson_name)
    num_prefix = prefix_match.group(1) if prefix_match else "01"

    student_code_py = student_dir / "student_code.py"
    student_code_py.write_text(
        f'"""\nstudent_code.py — {tier.capitalize()} Tier, Lesson {num_prefix}: {lesson_name}\n"""\n\n'
        f"import numpy as np\n"
        f"from hint import show_hint\n\n\n"
        f"class {cls_name}:\n"
        f"    @classmethod\n"
        f"    def example_method(cls):\n"
        f'        """Example method docstring."""\n'
        f'        # Replace the line below with show_hint() if you need help.\n'
        f'        raise NotImplementedError("Implement example_method()")\n',
        encoding="utf-8",
    )

    instruction_md = student_dir / "INSTRUCTION.md"
    instruction_md.write_text(
        f"# Lesson: {lesson_name}\n\n"
        f"## Learning Objectives\n"
        f"- Master NumPy array manipulations and vectorized operations.\n\n"
        f"## Task\n"
        f"Implement the methods in `student_code.py` complying with architectural constraints.\n\n"
        f"## Verification\n"
        f"Run your tests using the TensorForge CLI:\n"
        f"```bash\n"
        f"tforge check {module} {tier} {num_prefix}\n"
        f"```\n",
        encoding="utf-8",
    )

    baseline_py = test_dir / "_baseline.py"
    baseline_py.write_text(
        f'"""\n_baseline.py — Reference Solution for {lesson_name}\n"""\n\n'
        f"import numpy as np\n\n\n"
        f"class {cls_name}Baseline:\n"
        f"    @classmethod\n"
        f"    def example_method(cls):\n"
        f'        """Reference implementation."""\n'
        f"        return np.ones((5, 5), dtype=np.int32)\n",
        encoding="utf-8",
    )

    test_py = test_dir / f"test_{num_prefix}.py"
    test_py.write_text(
        f'"""\ntest_{num_prefix}.py — Verification Suite for {lesson_name}\n"""\n\n'
        f"import numpy as np\n"
        f"import pytest\n"
        f"from forge_core.benchmark import compare_and_benchmark\n"
        f"from forge_core.ast_validator import ast_policy\n"
        f"from _baseline import {cls_name}Baseline\n"
        f"from student_code import {cls_name}\n\n\n"
        f"class Test{cls_name}:\n"
        f"    @ast_policy(max_for_loops=0, max_while_loops=0)\n"
        f"    def test_example_method(self, benchmark_config):\n"
        f"        compare_and_benchmark(\n"
        f"            student_fn={cls_name}.example_method,\n"
        f"            baseline_fn={cls_name}Baseline.example_method,\n"
        f"            config=benchmark_config,\n"
        f"        )\n",
        encoding="utf-8",
    )

    print(f"{GREEN_COLOR}🎉 Successfully scaffolded lesson '{lesson_name}' in '{module}/{tier}'!{reset}\n")
    print(f"  Created student workspace : {INFO_COLOR}{student_dir.relative_to(ROOT)}{reset}")
    print(f"    └─ INSTRUCTION.md")
    print(f"    └─ student_code.py")
    print(f"  Created test workspace    : {INFO_COLOR}{test_dir.relative_to(ROOT)}{reset}")
    print(f"    └─ _baseline.py")
    print(f"    └─ test_{num_prefix}.py\n")
    return 0


def main() -> None:
    """TensorForge CLI entry point.

    Dispatches subcommands: ``check``, ``validate``, ``status``, ``generate``.
    """
    parser = argparse.ArgumentParser(
        description="TensorForge CLI — Interactive Lab & Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tforge check arraysmith basic 01\n"
            "  tforge check arraysmith basic 01 create_squared_range\n"
            "  tforge check arraysmith intermediate\n"
            "  tforge check infra\n"
            "  tforge validate\n"
            "  tforge status\n"
            "  tforge generate arraysmith basic 04_new_lesson\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ------------------------------------------------------------------ check
    check_parser = subparsers.add_parser("check", help="Run tests for student solutions")
    check_parser.add_argument(
        "curriculum",
        nargs="?",
        default="all",
        help=(
            "Target curriculum ('arraysmith', 'tensorsmith', 'infra', or 'all'). "
            "Default: 'all'"
        ),
    )
    check_parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        help=(
            "Tier within the curriculum ('basic', 'intermediate', 'advanced', "
            "'applications'). Omitting runs all tiers."
        ),
    )
    check_parser.add_argument(
        "lesson",
        nargs="?",
        default=None,
        help="Lesson ID or prefix (e.g. '01'). Omitting runs all lessons in the tier.",
    )
    check_parser.add_argument(
        "method",
        nargs="?",
        default=None,
        help="Optional specific method name to test (e.g. 'create_squared_range').",
    )
    check_parser.add_argument(
        "--method", "-m",
        dest="method_flag",
        default=None,
        help="Optional specific method name to test (flag form).",
    )

    # --------------------------------------------------------------- validate
    subparsers.add_parser("validate", help="Run baseline self-consistency and registry checks")

    # ----------------------------------------------------------------- status
    subparsers.add_parser("status", help="Show curriculum completion progress report")

    # --------------------------------------------------------------- generate
    generate_parser = subparsers.add_parser("generate", help="Scaffold a new curriculum lesson")
    generate_parser.add_argument("module", help="Curriculum module name (e.g. 'arraysmith')")
    generate_parser.add_argument("tier", help="Tier name (e.g. 'basic')")
    generate_parser.add_argument("lesson_name", help="Lesson folder name (e.g. '04_broadcasting')")

    args = parser.parse_args()

    if args.command == "check" or args.command is None:
        if args.command is None and len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        curriculum = getattr(args, "curriculum", "all")
        tier = getattr(args, "tier", None)
        lesson = getattr(args, "lesson", None)
        method = getattr(args, "method", None) or getattr(args, "method_flag", None)

        # Guard tensorsmith commands against missing optional dependency.
        if curriculum == "tensorsmith":
            _ensure_torch()

        exit_code = run_check(curriculum, tier, lesson, method)
        sys.exit(exit_code)

    elif args.command == "validate":
        exit_code = run_validate()
        sys.exit(exit_code)

    elif args.command == "status":
        exit_code = run_status()
        sys.exit(exit_code)

    elif args.command == "generate":
        exit_code = run_generate(args.module, args.tier, args.lesson_name)
        sys.exit(exit_code)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
