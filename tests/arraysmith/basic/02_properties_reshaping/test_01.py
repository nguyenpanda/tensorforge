"""
test_01.py — Test Suite for Module 01: Properties & Reshaping
==============================================================
Run this module's tests:
    tforge check arraysmith 01
"""

import numpy as np
import pytest

from forge_core.benchmark import compare_and_benchmark, BenchmarkConfig

from _baseline import PropertiesReshapingBaseline
from student_code import PropertiesReshaping


# ---------------------------------------------------------------------------
# Fixed test inputs (deterministic — same values every run)
# ---------------------------------------------------------------------------

_ARR_1D_12 = np.arange(12, dtype=np.int32)
_ARR_2D_2x6 = np.arange(12, dtype=np.int32).reshape(2, 6)


class TestPropertiesReshaping:
    """Test suite for Module 01 — Properties & Reshaping."""

    def test_reshape_to_matrix(self, benchmark_config):
        """Student must reshape (12,) → (3, 4) matching baseline output."""
        arr = _ARR_1D_12.copy()
        compare_and_benchmark(
            student_fn=lambda: PropertiesReshaping.reshape_to_matrix(arr),
            baseline_fn=lambda: PropertiesReshapingBaseline.reshape_to_matrix(arr),
            config=benchmark_config,
        )

    def test_flatten_and_cast(self, benchmark_config):
        """Student must flatten a 2-D int array and cast to float64."""
        arr = _ARR_2D_2x6.copy()
        compare_and_benchmark(
            student_fn=lambda: PropertiesReshaping.flatten_and_cast(arr),
            baseline_fn=lambda: PropertiesReshapingBaseline.flatten_and_cast(arr),
            config=benchmark_config,
        )
