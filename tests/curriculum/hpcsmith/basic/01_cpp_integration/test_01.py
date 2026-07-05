"""
test_01.py — Verification Suite for Lesson 01: C++ Integration
==============================================================
Verifies element-wise tensor addition via C++ extension against PyTorch baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
from _baseline import CppIntegrationBaseline
from student_code import CppIntegration


class TestCppIntegration:
    """Verification suite for C++ Integration."""

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 4, 5)])
    def test_addition_correctness(self, shape: tuple[int, ...]) -> None:
        """Verify C++ addition kernel matches PyTorch baseline."""
        np.random.seed(42)
        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)

        expected = CppIntegrationBaseline.run_cpp_addition(a, b)
        result = CppIntegration.run_cpp_addition(a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
