"""
tests/conftest.py — Consolidated Test Path & Module Resolution
==============================================================
Consolidates all path-injection and module-clearing logic for TensorForge tests.

When running tests in a Clean Lab architecture where `_baseline.py` lives inside
`tests/arraysmith/<lesson>/` and `student_code.py` lives inside
`arraysmith/<lesson>/`, pytest must resolve imports dynamically based on the test
module currently being collected or executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

_TESTS_ROOT = Path(__file__).parent.resolve()
_PROJECT_ROOT = _TESTS_ROOT.parent.resolve()


def _switch_lesson_context(test_dir_path: Path) -> None:
    """Dynamically configure sys.path and sys.modules for the target lesson directory.

    Args:
        test_dir_path: Absolute Path to the directory containing the test file.
    """
    for name in ("_baseline", "student_code"):
        sys.modules.pop(name, None)

    test_dir_str = str(test_dir_path.resolve())

    try:
        rel_path = test_dir_path.resolve().relative_to(_TESTS_ROOT)
        student_dir_str = str((_PROJECT_ROOT / rel_path).resolve())
    except ValueError:
        student_dir_str = test_dir_str

    for path_str in (test_dir_str, student_dir_str):
        while path_str in sys.path:
            sys.path.remove(path_str)

    if student_dir_str != test_dir_str:
        sys.path.insert(0, student_dir_str)
    sys.path.insert(0, test_dir_str)


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector | None = None, **kwargs) -> None:
    """Hook invoked before pytest collects a Python test module file."""
    if module_path and module_path.parent:
        _switch_lesson_context(module_path.parent)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hook invoked before pytest executes an individual test item."""
    test_path = getattr(item, "path", None)
    if test_path and test_path.parent:
        _switch_lesson_context(test_path.parent)
