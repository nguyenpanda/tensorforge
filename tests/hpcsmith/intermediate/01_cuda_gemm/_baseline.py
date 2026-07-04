"""
_baseline.py — Golden Standard Baseline for Lesson 01: CUDA GEMM
=================================================================
Provides the golden standard reference implementation using PyTorch's native
matrix multiplication (:func:`torch.matmul`).
"""

from __future__ import annotations

import torch


class CudaGemmBaseline:
    """Golden standard baseline for CUDA General Matrix Multiplication."""

    @classmethod
    def run_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the matrix multiplication of tensors a and b using PyTorch.

        Args:
            a: Left operand tensor of shape (M, K).
            b: Right operand tensor of shape (K, N).

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        return torch.matmul(a, b)

    @classmethod
    def run_naive_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute matrix product using PyTorch native matmul reference."""
        return torch.matmul(a, b)

    @classmethod
    def run_tiled_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute matrix product using PyTorch native matmul reference."""
        return torch.matmul(a, b)
