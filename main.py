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
Check a lesson (full benchmark):
    tforge check arraysmith basic 01

Check a lesson (Fast Mode — correctness only, no timing loops):
    tforge check arraysmith basic 01 --fast
    tforge check arraysmith basic 01 -f
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
import sys
import textwrap

from nguyenpanda.swan import reset

from forge_core.cli_controllers import (
    GREEN_COLOR,
    INFO_COLOR,
    WARN_COLOR,
    _ensure_torch,
    run_autocomplete,
    run_check,
    run_generate,
    run_lint,
    run_status,
    run_validate,
)

__all__ = [
    "run_autocomplete",
    "run_check",
    "run_generate",
    "run_lint",
    "run_status",
    "run_validate",
]


def main() -> None:
    """TensorForge CLI entry point.

    Dispatches subcommands: ``check``, ``validate``, ``status``, ``generate``, ``autocomplete``.
    Delegates all business logic and execution workflows to ``forge_core.cli_controllers``.
    """
    parser = argparse.ArgumentParser(
        prog="tforge",
        description=f"{INFO_COLOR}TensorForge CLI — Interactive Lab Training Ground & Clean Lab Architecture{reset}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""
            {GREEN_COLOR}Examples:{reset}
            {INFO_COLOR}tforge check arraysmith basic 01{reset}
            {INFO_COLOR}tforge check arraysmith basic 01 --fast{reset}
            {INFO_COLOR}tforge check arraysmith basic 01 -t create_squared_range{reset}
            {INFO_COLOR}tforge check arraysmith intermediate{reset}
            {INFO_COLOR}tforge check infra{reset}
            {INFO_COLOR}tforge validate{reset}
            {INFO_COLOR}tforge status{reset}
            {INFO_COLOR}tforge generate arraysmith basic 04_new_lesson{reset}
            {INFO_COLOR}tforge autocomplete{reset}
        """
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", help=f"{INFO_COLOR}Available subcommands{reset}"
    )

    check_parser = subparsers.add_parser(
        "check",
        help=f"{GREEN_COLOR}Run comprehensive unit tests and benchmarks against student solutions{reset}",
        description=f"{INFO_COLOR}Execute pytest benchmark suites for specified curricula, tiers, and lessons with granular method targeting.{reset}",
    )
    check_parser.add_argument(
        "curriculum",
        nargs="?",
        default="all",
        help=f"{INFO_COLOR}Target curriculum ('arraysmith', 'tensorsmith', 'hpcsmith', 'infra', or 'all'). Default: 'all'{reset}",
    )
    check_parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        help=f"{INFO_COLOR}Tier within the curriculum ('basic', 'intermediate', 'advanced', 'applications'). Omitting runs all tiers.{reset}",
    )
    check_parser.add_argument(
        "lesson",
        nargs="?",
        default=None,
        help=f"{INFO_COLOR}Lesson ID or prefix (e.g., '01'). Omitting runs all lessons in the tier.{reset}",
    )
    check_parser.add_argument(
        "method",
        nargs="?",
        default=None,
        help=f"{INFO_COLOR}Specific method name to test via pytest -k filter (e.g., 'create_squared_range').{reset}",
    )
    check_parser.add_argument(
        "--method",
        "-m",
        "--target",
        "-t",
        dest="method_flag",
        default=None,
        help=f"{INFO_COLOR}Specific method name to test via pytest -k filter (flag form).{reset}",
    )
    check_parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        default=False,
        help=f"{WARN_COLOR}Fast Mode: run each function exactly once for correctness only. Skips timing loops and performance scorecards.{reset}",
    )

    subparsers.add_parser(
        "validate",
        help=f"{GREEN_COLOR}Run baseline self-consistency and registry checks{reset}",
        description=f"{INFO_COLOR}Validate all baseline reference implementations and verify hint registry completeness across all tiers.{reset}",
    )

    subparsers.add_parser(
        "lint",
        help=f"{GREEN_COLOR}Run static analysis (ruff formatting and mypy strict type checking){reset}",
        description=f"{INFO_COLOR}Perform static code analysis across the entire TensorForge repository using ruff and mypy in strict mode.{reset}",
    )

    subparsers.add_parser(
        "status",
        help=f"{GREEN_COLOR}Show curriculum completion progress report and scorecards{reset}",
        description=f"{INFO_COLOR}Display an interactive progress report showing passed lessons and overall completion percentages per tier.{reset}",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help=f"{GREEN_COLOR}Scaffold a new curriculum lesson directory and starter templates{reset}",
        description=f"{INFO_COLOR}Create a new lesson directory with starter instruction files, baseline solutions, and student code scaffolds.{reset}",
    )
    generate_parser.add_argument(
        "module", help=f"{INFO_COLOR}Curriculum module name (e.g., 'arraysmith'){reset}"
    )
    generate_parser.add_argument("tier", help=f"{INFO_COLOR}Tier name (e.g., 'basic'){reset}")
    generate_parser.add_argument(
        "lesson_name", help=f"{INFO_COLOR}Lesson folder name (e.g., '04_broadcasting'){reset}"
    )

    autocomplete_parser = subparsers.add_parser(
        "autocomplete",
        help=f"{GREEN_COLOR}Generate native shell autocomplete script (.tforge-autocomplete.sh){reset}",
        description=f"{INFO_COLOR}Dynamically generate shell completion scripts for bash, zsh, or fish using native CLI parser introspection.{reset}",
    )
    autocomplete_parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish"],
        default=None,
        help=f"{INFO_COLOR}Target shell for completion script (default: auto-detect from $SHELL){reset}",
    )

    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if args.command == "check" or args.command is None:
        if args.command is None and len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        curriculum = getattr(args, "curriculum", "all")
        tier = getattr(args, "tier", None)
        lesson = getattr(args, "lesson", None)
        method = getattr(args, "method", None) or getattr(args, "method_flag", None)
        fast = getattr(args, "fast", False)

        if curriculum in ("tensorsmith", "hpcsmith"):
            _ensure_torch()

        exit_code = run_check(curriculum, tier, lesson, method, fast=fast)
        sys.exit(exit_code)

    elif args.command == "validate":
        exit_code = run_validate()
        sys.exit(exit_code)

    elif args.command == "lint":
        exit_code = run_lint()
        sys.exit(exit_code)

    elif args.command == "status":
        exit_code = run_status()
        sys.exit(exit_code)

    elif args.command == "generate":
        exit_code = run_generate(args.module, args.tier, args.lesson_name)
        sys.exit(exit_code)

    elif args.command == "autocomplete":
        exit_code = run_autocomplete(shell=args.shell)
        sys.exit(exit_code)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
