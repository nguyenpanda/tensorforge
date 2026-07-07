"""
tests/conftest.py — Consolidated Test Path & Module Resolution
==============================================================
Consolidates all path-injection and module-clearing logic for TensorForge tests.

When running tests in a Clean Lab architecture where `_baseline.py` lives inside
`tests/curriculum/arraysmith/<lesson>/` and `student_code.py` lives inside
`arraysmith/<lesson>/`, pytest must resolve imports dynamically based on the test
module currently being collected or executed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

_TESTS_ROOT = Path(__file__).parent.resolve()
_PROJECT_ROOT = _TESTS_ROOT.parent.resolve()


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


_ensure_venv_in_path(_PROJECT_ROOT)
_CURRENT_TEST_DIR: Path | None = None


def _switch_lesson_context(test_dir_path: Path) -> None:
    """Dynamically configure sys.path and sys.modules for the target lesson directory.

    Args:
        test_dir_path: Absolute Path to the directory containing the test file.
    """
    global _CURRENT_TEST_DIR
    resolved_dir = test_dir_path.resolve()
    if resolved_dir == _CURRENT_TEST_DIR:
        return
    _CURRENT_TEST_DIR = resolved_dir

    for name in ("_baseline", "student_code"):
        sys.modules.pop(name, None)

    test_dir_str = str(resolved_dir)

    try:
        curriculum_root = _TESTS_ROOT / "curriculum"
        try:
            rel_path = test_dir_path.resolve().relative_to(curriculum_root)
        except ValueError:
            rel_path = test_dir_path.resolve().relative_to(_TESTS_ROOT)
        student_dir_str = str((_PROJECT_ROOT / rel_path).resolve())
    except ValueError:
        student_dir_str = test_dir_str

    for path_str in (test_dir_str, student_dir_str):
        while path_str in sys.path:
            sys.path.remove(path_str)

    if student_dir_str != test_dir_str and Path(student_dir_str).is_dir():
        sys.path.insert(0, student_dir_str)
    sys.path.insert(0, test_dir_str)


def pytest_pycollect_makemodule(
    module_path: Path, parent: pytest.Collector | None = None, **kwargs: Any
) -> None:
    """Hook invoked before pytest collects a Python test module file."""
    if module_path and module_path.parent:
        _switch_lesson_context(module_path.parent)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hook invoked before pytest executes an individual test item."""
    test_path = getattr(item, "path", None)
    if test_path and test_path.parent:
        _switch_lesson_context(test_path.parent)
