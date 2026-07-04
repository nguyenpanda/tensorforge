"""
test_00.py — Test Suite for Module 00: Array Creation
======================================================
This file is intentionally kept slim.  All correctness checking and
performance benchmarking is delegated to ``compare_and_benchmark()``.

What each test does
-------------------
1. Calls the student's method and the baseline method (zero-arg callables).
2. Asserts outputs are equal (dtype + values).
3. Times both over ``benchmark_config.n_runs`` iterations.
4. Fails with a vectorization hint if the student's code exceeds the
   slowdown threshold.

Run this module's tests:
    tforge check arraysmith 00
"""


from _baseline import ArrayCreationBaseline
from student_code import ArrayCreation

from forge_core.benchmark import compare_and_benchmark


class TestArrayCreation:
    """Test suite for Module 00 — Array Creation."""

    def test_create_integer_range(self, benchmark_config):
        """Student output must match [0..99] as int8, within 5× of baseline speed."""
        compare_and_benchmark(
            student_fn=ArrayCreation.create_integer_range,
            baseline_fn=ArrayCreationBaseline.create_integer_range,
            config=benchmark_config,
        )

    def test_create_squared_range(self, benchmark_config):
        """Student output must match [0,1,4,9,...,9801], within 5× of baseline speed."""
        compare_and_benchmark(
            student_fn=ArrayCreation.create_squared_range,
            baseline_fn=ArrayCreationBaseline.create_squared_range,
            config=benchmark_config,
        )
