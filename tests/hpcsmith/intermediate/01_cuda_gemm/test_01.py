"""
test_01.py — Verification Suite for Lesson 01: CUDA GEMM
========================================================
Verifies the correctness of both naive and shared-memory tiled CUDA matrix
multiplication kernels against the PyTorch golden standard baseline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from _baseline import CudaGemmBaseline
from student_code import CudaGemm


class TestCudaGemm:
    """Verification suite for CUDA General Matrix Multiplication implementations."""

    @pytest.mark.parametrize("size", [16, 32, 64, 128, 256])
    @pytest.mark.parametrize("method_name", ["run_naive_gemm", "run_tiled_gemm"])
    def test_gemm_correctness_square_matrices(self, size: int, method_name: str) -> None:
        """Verify CUDA GEMM implementations against PyTorch baseline on square matrices."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        a = torch.randn((size, size), dtype=torch.float32, device=device)
        b = torch.randn((size, size), dtype=torch.float32, device=device)

        student_fn = getattr(CudaGemm, method_name)
        baseline_fn = getattr(CudaGemmBaseline, method_name)

        expected = baseline_fn(a, b)
        result = student_fn(a, b)

        assert result.shape == expected.shape, (
            f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        )
        assert result.dtype == expected.dtype, (
            f"Data type mismatch: expected {expected.dtype}, got {result.dtype}"
        )
        assert result.device == expected.device, (
            f"Device mismatch: expected {expected.device}, got {result.device}"
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("size", [1, 7, 15, 33, 100])
    @pytest.mark.parametrize("method_name", ["run_naive_gemm", "run_tiled_gemm"])
    def test_gemm_correctness_non_power_of_two(self, size: int, method_name: str) -> None:
        """Verify CUDA GEMM implementations on matrix sizes not divisible by tile size."""
        device = torch.device("cuda")
        torch.manual_seed(123)
        a = torch.randn((size, size), dtype=torch.float32, device=device)
        b = torch.randn((size, size), dtype=torch.float32, device=device)

        student_fn = getattr(CudaGemm, method_name)
        baseline_fn = getattr(CudaGemmBaseline, method_name)

        expected = baseline_fn(a, b)
        result = student_fn(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("method_name", ["run_naive_gemm", "run_tiled_gemm"])
    def test_gemm_does_not_mutate_inputs(self, method_name: str) -> None:
        """Verify that CUDA GEMM execution does not mutate operand tensors in place."""
        device = torch.device("cuda")
        torch.manual_seed(777)
        a = torch.randn((32, 32), dtype=torch.float32, device=device)
        b = torch.randn((32, 32), dtype=torch.float32, device=device)
        a_copy = a.clone()
        b_copy = b.clone()

        student_fn = getattr(CudaGemm, method_name)
        _ = student_fn(a, b)

        torch.testing.assert_close(a, a_copy)
        torch.testing.assert_close(b, b_copy)
