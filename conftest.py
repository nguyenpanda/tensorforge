"""
conftest.py — Root pytest configuration for TensorForge
=========================================================
Responsibilities
----------------
1. **benchmark_config fixture** — Provides a session-scoped
   :class:`~forge_core.benchmark.BenchmarkConfig` that every lesson's test
   file receives automatically.  Individual lessons can override it by
   re-declaring the fixture locally.

2. **Per-lesson sys.path injection** — Each lesson directory contains its own
   ``conftest.py`` that prepends the local path and evicts the module cache so
   that ``from student_code import ...`` and ``from _baseline import ...``
   always resolve to the lesson-local files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_ROOT_DIR = Path(__file__).parent.resolve()


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


_ensure_venv_in_path(_ROOT_DIR)

if TYPE_CHECKING:
    from forge_core.benchmark import BenchmarkConfig


@pytest.fixture(scope="session")
def benchmark_config() -> BenchmarkConfig:
    """Session-scoped default :class:`~forge_core.benchmark.BenchmarkConfig`.

    Individual test modules can override this fixture to customise thresholds
    for specific lessons::

        # inside a lesson's conftest.py or test_XX.py
        import pytest
        from forge_core.benchmark import BenchmarkConfig

        @pytest.fixture(scope="module")
        def benchmark_config():
            return BenchmarkConfig(slowdown_threshold=10.0, n_runs=100)

    Returns:
        BenchmarkConfig: Default configuration (5× threshold, 50 runs).
    """
    from forge_core.benchmark import BenchmarkConfig

    return BenchmarkConfig()
