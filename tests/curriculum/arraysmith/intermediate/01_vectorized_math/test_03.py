"""
test_03.py — Test Suite for Module 03: Vectorized Math
=======================================================
Run this module's tests:
    tforge check arraysmith 03
"""

import numpy as np
from _baseline import VectorizedMathBaseline
from student_code import VectorizedMath

from forge_core.ast_validator import ast_policy
from forge_core.benchmark import compare_and_benchmark

# ---------------------------------------------------------------------------
# Fixed test inputs (deterministic)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=7)

# Float matrix — (50 rows × 8 cols) provides enough computation for timing.
_MATRIX: np.ndarray = _RNG.uniform(-100.0, 100.0, size=(50, 8))


class TestVectorizedMath:
    """Test suite for Module 03 — Vectorized Math."""

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        feedback={
            "max_for_loops": "Do not loop over array elements!",
        },
    )
    def test_row_means(self, benchmark_config):
        """Student must compute per-row means using np.mean(axis=1)."""
        mat = _MATRIX.copy()
        compare_and_benchmark(
            student_fn=lambda: VectorizedMath.row_means(mat),
            baseline_fn=lambda: VectorizedMathBaseline.row_means(mat),
            config=benchmark_config,
        )

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        feedback={
            "max_for_loops": "Do not loop over array elements!",
        },
    )
    def test_normalize_columns(self, benchmark_config):
        """Student must zero-mean every column via broadcasting."""
        mat = _MATRIX.copy()
        compare_and_benchmark(
            student_fn=lambda: VectorizedMath.normalize_columns(mat),
            baseline_fn=lambda: VectorizedMathBaseline.normalize_columns(mat),
            config=benchmark_config,
        )

        # Extra semantic check: every column of the output must have mean ≈ 0.
        student_output = VectorizedMath.normalize_columns(mat)
        col_means = np.abs(np.mean(student_output, axis=0))
        assert np.all(col_means < 1e-10), (
            f"Column means are not zero after normalization.\n"
            f"Got column means: {col_means}\n"
            f"Each column must have mean = 0.0 (within floating-point tolerance)."
        )
