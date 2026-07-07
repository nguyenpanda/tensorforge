"""
cli_controllers.py — TensorForge CLI Workflow Controllers
=========================================================
Encapsulates all execution logic, test targeting, scaffolding, validation, and status reporting
for the TensorForge command-line interface. Keeps the main CLI router thin and decoupled.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from nguyenpanda.swan import c24, reset

ROOT = Path(__file__).parent.parent.resolve()


def _ensure_venv_in_path(root_dir: Path) -> None:
    """Ensure virtual environment bin/Scripts directory is in os.environ['PATH']."""
    paths_to_add = []
    exec_dir = os.path.dirname(sys.executable)
    if exec_dir and os.path.isdir(exec_dir):
        paths_to_add.append(exec_dir)

    venv_bin = root_dir / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    if venv_bin.is_dir() and str(venv_bin) not in paths_to_add:
        paths_to_add.append(str(venv_bin))

    current_path = os.environ.get("PATH", "")
    path_list = current_path.split(os.pathsep) if current_path else []

    added = False
    for p in reversed(paths_to_add):
        if p not in path_list:
            path_list.insert(0, p)
            added = True

    if added:
        os.environ["PATH"] = os.pathsep.join(path_list)


_ensure_venv_in_path(ROOT)

ERR_COLOR = c24["ff3333"]
INFO_COLOR = c24["00d7ff"]
GREEN_COLOR = c24["00ff87"]
WARN_COLOR = c24["ff8800"]
BOLD = "\033[1m"

ARRAYSMITH_TIERS = ["basic", "intermediate", "advanced", "applications"]


def _ensure_torch() -> None:
    """Guard against missing optional ``torch`` dependency and set default CUDA arch list.

    When a ``tensorsmith`` or ``hpcsmith`` command is requested but the ``torch`` extra has not
    been installed, print a descriptive warning via swan and terminate with
    exit code 1. This prevents an ImportError traceback from confusing the
    student with an irrelevant stack trace.
    """
    if importlib.util.find_spec("torch") is None:
        print(
            f"{WARN_COLOR}⚠  Missing dependency: 'torch' is not installed.{reset}\n"
            f"   Run {INFO_COLOR}`uv sync --extra torch`{reset} to enable "
            f"tensorsmith and hpcsmith commands.",
            file=sys.stderr,
        )
        sys.exit(1)

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch

            if torch.cuda.is_available():
                archs = {
                    f"{cap[0]}.{cap[1]}"
                    for i in range(torch.cuda.device_count())
                    for cap in [torch.cuda.get_device_capability(i)]
                }
                if archs:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(archs))
        except Exception:
            pass


def _resolve_target(
    curriculum: str,
    tier: str | None,
    lesson: str | None,
    method: str | None = None,
) -> list[str]:
    """Resolve CLI arguments to pytest target paths.

    Args:
        curriculum: Curriculum name (``"arraysmith"``, ``"tensorsmith"``,
            ``"hpcsmith"``, ``"infra"``, or ``"all"``).
        tier: Optional tier name (``"basic"``, ``"intermediate"``, etc.).
            When ``None``, the entire curriculum is tested.
        lesson: Optional lesson ID or prefix (e.g. ``"01"``). When ``None``,
            the entire tier is tested.
        method: Optional pytest ``-k`` filter for a specific method name.

    Returns:
        list[str]: Argument list suitable for ``python -m pytest``.
    """
    if curriculum == "all":
        args = ["tests/", "-v", "-s"]
    elif curriculum == "infra":
        args = ["tests/infra/", "-v", "-s"]
    else:
        curriculum_dir = ROOT / "tests" / "curriculum" / curriculum
        if not curriculum_dir.is_dir():
            print(
                f"{ERR_COLOR}Error: Curriculum directory not found: tests/curriculum/{curriculum}{reset}",
                file=sys.stderr,
            )
            sys.exit(1)

        if not tier or tier == "all":
            args = [f"tests/curriculum/{curriculum}", "-v", "-s"]
        else:
            tier_dir = curriculum_dir / tier
            if not tier_dir.is_dir():
                available = [
                    d.name
                    for d in curriculum_dir.iterdir()
                    if d.is_dir() and not d.name.startswith("__")
                ]
                print(
                    f"{ERR_COLOR}Error: Tier '{tier}' not found in tests/curriculum/{curriculum}.\n"
                    f"Available tiers: {', '.join(sorted(available))}{reset}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not lesson or lesson == "all":
                args = [str(tier_dir.relative_to(ROOT)), "-v", "-s"]
            else:
                matches = [
                    d
                    for d in tier_dir.iterdir()
                    if d.is_dir() and (d.name == lesson or d.name.startswith(f"{lesson}_"))
                ]
                if not matches:
                    print(
                        f"{ERR_COLOR}Error: No lesson matching '{lesson}' found in "
                        f"tests/curriculum/{curriculum}/{tier}{reset}",
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


def _to_pascal_case(name: str) -> str:
    """Convert a lesson directory name to PascalCase for class naming.

    Args:
        name: Lesson directory name (e.g. ``"01_array_creation"``).

    Returns:
        str: PascalCase string (e.g. ``"ArrayCreation"``).
    """
    clean = re.sub(r"^\d+_?", "", name)
    return "".join(w.capitalize() for w in clean.split("_") if w)


class CLIController:
    """Encapsulates execution workflows and dispatch logic for TensorForge CLI commands."""

    @staticmethod
    def run_check(
        curriculum: str,
        tier: str | None,
        lesson: str | None,
        method: str | None = None,
        fast: bool = False,
    ) -> int:
        """Run pytest against the specified curriculum, tier, and lesson.

        When *fast* is ``True``, the environment variable ``TFORGE_FAST_MODE=1``
        is injected into the subprocess environment before pytest launches. The
        benchmark engine reads this variable and bypasses timing loops,
        ``tracemalloc``, and the performance scorecard, performing only a single
        correctness assertion per function.

        Args:
            curriculum: Curriculum identifier.
            tier: Tier name (``"basic"``, ``"intermediate"``, etc.) or ``None``.
            lesson: Lesson ID/prefix or ``None``.
            method: Optional ``-k`` filter for a specific test method.
            fast: When ``True``, set ``TFORGE_FAST_MODE=1`` in the subprocess
                environment to skip heavy profiling overhead.

        Returns:
            int: pytest return code (0 = all passed).
        """
        pytest_args = _resolve_target(curriculum, tier, lesson, method)
        mode_label = f"{WARN_COLOR}[Fast Mode]{reset} " if fast else ""
        print(f"{INFO_COLOR}{mode_label}Running: pytest {' '.join(pytest_args)}{reset}\n")
        cmd = [sys.executable, "-m", "pytest", *pytest_args]
        env = os.environ.copy()
        if fast:
            env["TFORGE_FAST_MODE"] = "1"
        result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        return result.returncode

    @staticmethod
    def run_validate() -> int:
        """Run the multi-tier baseline validation script.

        Returns:
            int: Script return code (0 = all passed).
        """
        cmd = [sys.executable, str(ROOT / "validate_baselines.py")]
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        return result.returncode

    @staticmethod
    def run_lint() -> int:
        """Run static analysis checks (ruff and mypy) sequentially.

        Returns:
            int: 0 if both checks pass, otherwise non-zero exit code.
        """
        print(f"{INFO_COLOR}[TensorForge Lint] Running ruff check...{reset}")
        res_ruff = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(ROOT)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res_ruff.returncode != 0:
            return res_ruff.returncode

        print(f"{INFO_COLOR}[TensorForge Lint] Running mypy strict checks...{reset}")
        res_mypy = subprocess.run(
            [sys.executable, "-m", "mypy", str(ROOT)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res_mypy.returncode != 0:
            return res_mypy.returncode

        print(f"{GREEN_COLOR}[TensorForge Lint] All static analysis checks passed!{reset}")
        return 0

    @staticmethod
    def run_status() -> int:
        """Run tests silently and output a tier-grouped colorized progress report.

        Iterates over all available curricula (arraysmith, tensorsmith, hpcsmith) and their
        canonical tier order (basic → intermediate → advanced → applications),
        printing a pass/fail status line per lesson within each tier. A final
        progress bar summarises overall curriculum completion.

        Returns:
            int: 0 always (errors are displayed inline).
        """
        print(f"{INFO_COLOR}Scanning curriculum progress...{reset}\n")
        total_lessons = 0
        passed_lessons = 0

        for curriculum_name in ["arraysmith", "tensorsmith", "hpcsmith"]:
            tests_dir = ROOT / "tests" / "curriculum" / curriculum_name
            if not tests_dir.is_dir():
                continue

            has_lessons = any(
                (tests_dir / t).is_dir()
                and any(
                    d.is_dir() and not d.name.startswith("__") for d in (tests_dir / t).iterdir()
                )
                for t in ARRAYSMITH_TIERS
                if (tests_dir / t).is_dir()
            )
            if not has_lessons:
                continue

            print(f"{BOLD}{INFO_COLOR}  Module Progress Report — {curriculum_name}{reset}")
            print(f"{INFO_COLOR}  {'─' * 50}{reset}")

            for tier_name in ARRAYSMITH_TIERS:
                tier_dir = tests_dir / tier_name
                if not tier_dir.is_dir():
                    continue

                lessons = sorted(
                    [d for d in tier_dir.iterdir() if d.is_dir() and not d.name.startswith("__")]
                )
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

            print(f"\n{INFO_COLOR}  {'─' * 50}{reset}\n")

        if total_lessons == 0:
            print(f"{WARN_COLOR}No curriculum lessons found.{reset}\n")
            return 1

        pct = int((passed_lessons / total_lessons) * 100) if total_lessons > 0 else 0
        bar_len = 10
        filled = int(bar_len * (passed_lessons / total_lessons)) if total_lessons > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        bar_color = GREEN_COLOR if pct == 100 else (WARN_COLOR if pct >= 50 else ERR_COLOR)
        print(f"  {BOLD}Curriculum Status: {bar_color}[{bar}] {pct}% Passed{reset}\n")
        return 0

    @staticmethod
    def run_autocomplete(shell: str | None = None) -> int:
        """Generate shell autocomplete script and print sourcing instructions."""
        if shell:
            shell_name = shell.lower()
        else:
            shell_path = os.environ.get("SHELL", "/bin/bash")
            shell_name = os.path.basename(shell_path).lower()
            if shell_name not in ("bash", "zsh", "fish"):
                shell_name = "bash"

        if sys.stdout.isatty():
            print(
                f"{INFO_COLOR}============================================================{reset}"
            )
            print(f"{INFO_COLOR}  TensorForge Autocomplete Generator{reset}")
            print(
                f"{INFO_COLOR}============================================================{reset}\n"
            )

        script_path = ROOT / ".tforge-autocomplete.sh"

        try:
            import argcomplete

            script_content = argcomplete.shellcode(["tforge"], shell=shell_name, use_defaults=True)  # type: ignore[attr-defined, no-untyped-call]
            wrapper = textwrap.dedent(
                """

                # TensorForge CLI wrapper: automatically routes 'tforge' to '.venv/bin/tforge' or 'uv run tforge'
                if ! command -v tforge >/dev/null 2>&1; then
                    tforge() {
                        if [ -x "./.venv/bin/tforge" ]; then
                            "./.venv/bin/tforge" "$@"
                        elif command -v uv >/dev/null 2>&1; then
                            uv run tforge "$@"
                        else
                            echo "Error: tforge executable not found. Please run 'source .venv/bin/activate' or install uv." >&2
                            return 1
                        fi
                    }
                fi
                """
            )
            script_content += wrapper
        except Exception as e:
            print(
                f"{ERR_COLOR}Failed to generate native autocomplete script via argcomplete: {e}{reset}",
                file=sys.stderr,
            )
            return 1

        try:
            script_path.write_text(script_content, encoding="utf-8")
            if not sys.stdout.isatty():
                print(script_content, end="", file=sys.stdout)
                print(
                    f"{GREEN_COLOR}[SUCCESS]{reset} Generated autocomplete script for '{shell_name}' to stdout.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{GREEN_COLOR}[SUCCESS]{reset} Generated native autocomplete script for '{shell_name}':"
                )
                print(f"  {INFO_COLOR}{script_path.absolute()}{reset}\n")
                print("To enable autocomplete in your current shell, run:")
                print(
                    f"  {GREEN_COLOR}[ -f {script_path.absolute()} ] && source {script_path.absolute()}{reset}\n"
                )
                print(
                    "To persist this permanently, add the line above to your ~/.zshrc or ~/.bashrc."
                )
            return 0
        except Exception as e:
            print(
                f"{ERR_COLOR}[ERROR]{reset} Failed to generate autocomplete script: {e}",
                file=sys.stderr,
            )
            return 1

    @staticmethod
    def run_generate(module: str, tier: str, lesson_name: str) -> int:
        """Dynamically scaffold a new lesson directory and mirror it in tests/.

        Creates the student workspace under ``<module>/<tier>/<lesson_name>/``
        and the mirrored test workspace under ``tests/curriculum/<module>/<tier>/<lesson_name>/``.
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
        test_dir = ROOT / "tests" / "curriculum" / module / tier / lesson_name

        if student_dir.exists() or test_dir.exists():
            print(
                f"{ERR_COLOR}Error: Lesson directory '{lesson_name}' already exists in "
                f"{module}/{tier} or tests/curriculum/{module}/{tier}.{reset}",
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
            f"from hint import show_hint  # noqa: F401\n\n\n"
            f"class {cls_name}:\n"
            f"    @classmethod\n"
            f"    def example_method(cls):\n"
            f'        """Example method docstring."""\n'
            f"        # Replace the line below with show_hint() if you need help.\n"
            f'        raise NotImplementedError("TODO: Implement this function")\n',
            encoding="utf-8",
        )

        instruction_md = student_dir / "INSTRUCTION.md"
        instruction_md.write_text(
            f"# Lesson: {lesson_name}\n\n"
            f"## Learning Objectives\n"
            f"- Master array manipulations and vectorized operations.\n\n"
            f"## Task\n"
            f"Implement the methods in `student_code.py` complying with architectural constraints.\n"
            f"Call `show_hint()` if you are stuck.\n\n"
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

        print(
            f"{GREEN_COLOR}🎉 Successfully scaffolded lesson '{lesson_name}' in '{module}/{tier}'!{reset}\n"
        )
        print(f"  Created student workspace : {INFO_COLOR}{student_dir.relative_to(ROOT)}{reset}")
        print("    └─ INSTRUCTION.md")
        print("    └─ student_code.py")
        print(f"  Created test workspace    : {INFO_COLOR}{test_dir.relative_to(ROOT)}{reset}")
        print("    └─ _baseline.py")
        print(f"    └─ test_{num_prefix}.py\n")
        return 0


def run_check(
    curriculum: str,
    tier: str | None,
    lesson: str | None,
    method: str | None = None,
    fast: bool = False,
) -> int:
    """Execute run_check workflow."""
    return CLIController.run_check(curriculum, tier, lesson, method, fast=fast)


def run_validate() -> int:
    """Execute run_validate workflow."""
    return CLIController.run_validate()


def run_lint() -> int:
    """Execute run_lint workflow."""
    return CLIController.run_lint()


def run_status() -> int:
    """Execute run_status workflow."""
    return CLIController.run_status()


def run_autocomplete(shell: str | None = None) -> int:
    """Execute run_autocomplete workflow."""
    return CLIController.run_autocomplete(shell=shell)


def run_generate(module: str, tier: str, lesson_name: str) -> int:
    """Execute run_generate workflow."""
    return CLIController.run_generate(module, tier, lesson_name)
