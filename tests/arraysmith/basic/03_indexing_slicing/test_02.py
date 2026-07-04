"""
test_02.py — Test Suite for Module 02: Indexing & Slicing
==========================================================
Run this module's tests:
    tforge check arraysmith 02
"""

import numpy as np
from _baseline import IndexingSlicingBaseline
from student_code import IndexingSlicing

from forge_core.benchmark import compare_and_benchmark

# ---------------------------------------------------------------------------
# Fixed test inputs
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)

# Signed integer array with values spread around zero — ensures both
# positive and negative elements exist for the threshold filter test.
_ARR_SIGNED: np.ndarray = _RNG.integers(-50, 51, size=200, dtype=np.int32)

# Unsorted, potentially-duplicate index list for fancy-indexing test.
_INDICES: np.ndarray = np.array(
    [7, 0, 15, 3, 7, 99, 42, 128, 0, 199], dtype=np.int64
)

_THRESHOLD: float = 10.0


class TestIndexingSlicing:
    """Test suite for Module 02 — Indexing & Slicing."""

    def test_filter_above_threshold(self, benchmark_config):
        """Student must return only elements > 10 using boolean masking."""
        arr = _ARR_SIGNED.copy()
        compare_and_benchmark(
            student_fn=lambda: IndexingSlicing.filter_above_threshold(arr, _THRESHOLD),
            baseline_fn=lambda: IndexingSlicingBaseline.filter_above_threshold(
                arr, _THRESHOLD
            ),
            config=benchmark_config,
        )

    def test_gather_by_indices(self, benchmark_config):
        """Student must gather elements at given positions using fancy indexing."""
        arr = _ARR_SIGNED.copy()
        indices = _INDICES.copy()
        compare_and_benchmark(
            student_fn=lambda: IndexingSlicing.gather_by_indices(arr, indices),
            baseline_fn=lambda: IndexingSlicingBaseline.gather_by_indices(
                arr, indices
            ),
            config=benchmark_config,
        )
