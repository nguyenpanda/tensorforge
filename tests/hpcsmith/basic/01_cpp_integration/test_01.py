"""
tests/hpcsmith/basic/01_cpp_integration/test_01.py
==================================================
Test suite for Lesson 01: C++ Integration.

Verifies that :meth:`~CppIntegration.run_cpp_addition` correctly delegates to
the JIT-compiled C++ extension via :class:`~forge_core.backends.cuda_backend.CudaJitBackend`
and returns numerically accurate results.

Strategy
--------
Tests use real JIT compilation against the actual
``hpcsmith/native/01_cpp_addition.cpp`` source.  This is an integration test:
it exercises the full chain from NumPy → PyTorch tensor → C++ ATen kernel →
NumPy, proving that the HPC Bridge pipeline is functional end-to-end.

The ``tests/hpcsmith/conftest.py`` guards this entire directory tree with
``pytest.importorskip("torch")``, so all tests here are automatically skipped
in environments where the ``torch`` extra has not been installed.

Run with:
    uv run pytest tests/hpcsmith/basic/01_cpp_integration/test_01.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_LESSON_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "hpcsmith"
    / "basic"
    / "01_cpp_integration"
)
if str(_LESSON_DIR) not in sys.path:
    sys.path.insert(0, str(_LESSON_DIR))

from student_code import CppIntegration  # noqa: E402


class TestCppIntegration:
    """Integration tests for the C++ tensor addition HPC Bridge lesson."""

    def test_addition_float32_small(self) -> None:
        """C++ kernel must correctly add two small float32 arrays."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = CppIntegration.run_cpp_addition(a, b)

        np.testing.assert_array_equal(result, np.array([5.0, 7.0, 9.0], dtype=np.float32))

    def test_addition_preserves_dtype_float32(self) -> None:
        """Output dtype must match the float32 input dtype."""
        a = np.ones((10,), dtype=np.float32)
        b = np.ones((10,), dtype=np.float32)

        result = CppIntegration.run_cpp_addition(a, b)

        assert result.dtype == np.float32, (
            f"Expected float32, got {result.dtype}"
        )

    def test_addition_random_float32_large(self) -> None:
        """C++ kernel must produce results numerically identical to NumPy addition."""
        rng = np.random.default_rng(seed=42)
        a = rng.standard_normal((1024,)).astype(np.float32)
        b = rng.standard_normal((1024,)).astype(np.float32)

        result = CppIntegration.run_cpp_addition(a, b)
        expected = a + b

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_addition_2d_matrix_float32(self) -> None:
        """C++ kernel must handle 2-D matrix inputs correctly."""
        rng = np.random.default_rng(seed=7)
        a = rng.standard_normal((32, 64)).astype(np.float32)
        b = rng.standard_normal((32, 64)).astype(np.float32)

        result = CppIntegration.run_cpp_addition(a, b)
        expected = a + b

        assert result.shape == expected.shape, (
            f"Shape mismatch: got {result.shape}, expected {expected.shape}"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_addition_returns_numpy_array(self) -> None:
        """run_cpp_addition must return a numpy.ndarray, not a torch.Tensor."""
        a = np.zeros((5,), dtype=np.float32)
        b = np.ones((5,), dtype=np.float32)

        result = CppIntegration.run_cpp_addition(a, b)

        assert isinstance(result, np.ndarray), (
            f"Expected numpy.ndarray, got {type(result).__name__}"
        )

    def test_addition_zero_tensors(self) -> None:
        """Adding two zero arrays must produce an all-zero result."""
        a = np.zeros((100,), dtype=np.float32)
        b = np.zeros((100,), dtype=np.float32)

        result = CppIntegration.run_cpp_addition(a, b)

        np.testing.assert_array_equal(result, np.zeros((100,), dtype=np.float32))

    def test_addition_commutativity(self) -> None:
        """a + b must equal b + a (commutativity via C++ ATen kernel)."""
        rng = np.random.default_rng(seed=99)
        a = rng.standard_normal((256,)).astype(np.float32)
        b = rng.standard_normal((256,)).astype(np.float32)

        result_ab = CppIntegration.run_cpp_addition(a, b)
        result_ba = CppIntegration.run_cpp_addition(b, a)

        np.testing.assert_allclose(result_ab, result_ba, rtol=1e-6, atol=1e-6)

    def test_addition_does_not_mutate_inputs(self) -> None:
        """The C++ kernel must not modify the original input arrays."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a_copy = a.copy()
        b_copy = b.copy()

        CppIntegration.run_cpp_addition(a, b)

        np.testing.assert_array_equal(a, a_copy)
        np.testing.assert_array_equal(b, b_copy)
