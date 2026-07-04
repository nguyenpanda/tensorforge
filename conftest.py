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

from typing import TYPE_CHECKING

import pytest

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
