"""
test_04.py — Test Suite for Module 04: Array Manipulation
==========================================================
Run this module's tests:
    tforge check arraysmith 04
"""

import numpy as np
import pytest

from forge_core.benchmark import compare_and_benchmark

from _baseline import ArrayManipulationBaseline
from student_code import ArrayManipulation


# ---------------------------------------------------------------------------
# Fixed test inputs (deterministic)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=13)

# Three 1-D arrays to be stacked into a matrix.
_ROW_ARRAYS: list[np.ndarray] = [
    _RNG.integers(0, 100, size=5, dtype=np.int32),
    _RNG.integers(0, 100, size=5, dtype=np.int32),
    _RNG.integers(0, 100, size=5, dtype=np.int32),
]

# Two 2-D arrays for horizontal concatenation: same row count, different columns.
_LEFT:  np.ndarray = _RNG.integers(0, 50, size=(4, 3), dtype=np.int32)
_RIGHT: np.ndarray = _RNG.integers(0, 50, size=(4, 5), dtype=np.int32)


class TestArrayManipulation:
    """Test suite for Module 04 — Array Manipulation."""

    def test_stack_rows(self, benchmark_config):
        """Student must stack 1-D arrays into a 2-D matrix using np.vstack."""
        arrays = [a.copy() for a in _ROW_ARRAYS]
        compare_and_benchmark(
            student_fn=lambda: ArrayManipulation.stack_rows(arrays),
            baseline_fn=lambda: ArrayManipulationBaseline.stack_rows(arrays),
            config=benchmark_config,
        )

        # Extra shape check.
        result = ArrayManipulation.stack_rows(arrays)
        assert result.shape == (3, 5), (
            f"Expected shape (3, 5) but got {result.shape}.\n"
            "Each input array should become one row of the output."
        )

    def test_concatenate_side_by_side(self, benchmark_config):
        """Student must join two matrices column-wise using np.concatenate(axis=1)."""
        left, right = _LEFT.copy(), _RIGHT.copy()
        compare_and_benchmark(
            student_fn=lambda: ArrayManipulation.concatenate_side_by_side(left, right),
            baseline_fn=lambda: ArrayManipulationBaseline.concatenate_side_by_side(
                left, right
            ),
            config=benchmark_config,
        )

        # Extra shape check.
        result = ArrayManipulation.concatenate_side_by_side(left, right)
        assert result.shape == (4, 8), (
            f"Expected shape (4, 8) but got {result.shape}.\n"
            "Columns of left (3) + right (5) should equal 8 total columns."
        )
