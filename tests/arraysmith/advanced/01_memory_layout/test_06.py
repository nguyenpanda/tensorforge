"""
test_06.py — Verification Suite for Module 06: Memory Layout & Strides
======================================================================
Tests student implementation against reference baselines and enforces
clean architectural principles via static AST analysis.
"""

import numpy as np
from _baseline import MemoryLayoutBaseline
from student_code import MemoryLayout

from forge_core.ast_validator import ast_policy
from forge_core.benchmark import compare_and_benchmark

# Fixed test inputs for deterministic verification
_RNG = np.random.default_rng(42)
_ARR_F = _RNG.uniform(0.0, 100.0, size=(100, 100)).T.astype(np.float64)
_ARR_SHARE_1 = _RNG.uniform(0.0, 10.0, size=(100, 100)).astype(np.float64)
_ARR_SHARE_2 = _ARR_SHARE_1[::2, ::2]
_ARR_STRIDE = _RNG.uniform(0.0, 10.0, size=(50, 80)).astype(np.float64)


class TestMemoryLayout:
    """Test suite for Module 06 — Memory Layout & Strides."""

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        feedback={
            "max_for_loops": "Do not loop over array elements or dimensions! Inspect array flags and use np.ascontiguousarray(arr) if re-ordering is needed.",
        },
    )
    def test_ensure_c_contiguous(self, benchmark_config):
        """Student output must be C-contiguous and preserve existing buffer if already contiguous."""
        arr = _ARR_F.copy()
        compare_and_benchmark(
            student_fn=lambda: MemoryLayout.ensure_c_contiguous(arr),
            baseline_fn=lambda: MemoryLayoutBaseline.ensure_c_contiguous(arr),
            config=benchmark_config,
        )

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        feedback={
            "max_for_loops": "Do not iterate over elements to check memory equality! Use np.may_share_memory(a1, a2) or inspect the .base attribute directly.",
        },
    )
    def test_check_memory_share(self, benchmark_config):
        """Student output must correctly detect shared memory views vs copies."""
        a1 = _ARR_SHARE_1
        a2 = _ARR_SHARE_2
        compare_and_benchmark(
            student_fn=lambda: MemoryLayout.check_memory_share(a1, a2),
            baseline_fn=lambda: MemoryLayoutBaseline.check_memory_share(a1, a2),
            config=benchmark_config,
        )

    @ast_policy(
        max_for_loops=0,
        max_while_loops=0,
        feedback={
            "max_for_loops": "Do not use a loop to compute stride offsets! Simply inspect and extract the row stride from arr.strides[0].",
        },
    )
    def test_get_row_stride_bytes(self, benchmark_config):
        """Student output must correctly inspect row stride bytes from array metadata."""
        arr = _ARR_STRIDE.copy()
        compare_and_benchmark(
            student_fn=lambda: MemoryLayout.get_row_stride_bytes(arr),
            baseline_fn=lambda: MemoryLayoutBaseline.get_row_stride_bytes(arr),
            config=benchmark_config,
        )
