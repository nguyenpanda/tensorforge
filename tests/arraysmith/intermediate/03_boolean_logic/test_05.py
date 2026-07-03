"""
test_05.py — Verification Suite for Module 05: Boolean Logic
============================================================
Tests student implementation against reference baselines and enforces
clean architectural principles via static AST analysis.
"""

import numpy as np
import pytest

from forge_core.benchmark import compare_and_benchmark
from forge_core.ast_validator import ast_policy
from _baseline import BooleanLogicBaseline
from student_code import BooleanLogic

# Fixed test inputs for deterministic verification
_RNG = np.random.default_rng(42)
_PRICES = _RNG.uniform(10.0, 500.0, size=(100, 100)).astype(np.float64)
_MAT_A = _RNG.uniform(-10.0, 10.0, size=(200, 20)).astype(np.float64)
_MAT_B = _MAT_A.copy()
_MAT_B[_RNG.choice(200, size=100, replace=False)] += 1.0


class TestBooleanLogic:
    """Test suite for Module 05 — Boolean Logic."""

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        require_calls=["np.where"],
        feedback={
            "max_for_loops": "Do not iterate over prices with a for-loop! Use np.where(condition, x, y) for instantaneous C-speed conditional logic.",
            "np.where": "You must invoke np.where(condition, discounted, original) to vectorize the conditional price discount.",
        },
    )
    def test_apply_discount(self, benchmark_config):
        """Student output must use np.where without loops and match baseline precision."""
        prices = _PRICES.copy()
        compare_and_benchmark(
            student_fn=lambda: BooleanLogic.apply_discount(prices),
            baseline_fn=lambda: BooleanLogicBaseline.apply_discount(prices),
            config=benchmark_config,
        )

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        require_calls=["np.isclose", "np.all"],
        feedback={
            "max_for_loops": "Do not loop over rows! Combine np.isclose with np.all(..., axis=1) to check row-wise approximate equality vectorially.",
            "np.isclose": "You must use np.isclose(a, b, rtol=...) to properly handle floating-point tolerances.",
            "np.all": "You must use np.all(..., axis=1) to collapse column comparisons across each row.",
        },
    )
    def test_identify_close_rows(self, benchmark_config):
        """Student output must use np.isclose and np.all without loops."""
        mat_a = _MAT_A.copy()
        mat_b = _MAT_B.copy()
        compare_and_benchmark(
            student_fn=lambda: BooleanLogic.identify_close_rows(mat_a, mat_b),
            baseline_fn=lambda: BooleanLogicBaseline.identify_close_rows(mat_a, mat_b),
            config=benchmark_config,
        )
