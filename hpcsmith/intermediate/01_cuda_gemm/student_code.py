"""
student_code.py — Lesson 01: CUDA GEMM
======================================
Transparent JIT compiler wrapper for the CUDA GEMM extension.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import torch
from torch.utils.cpp_extension import load

from hint import show_hint  # noqa: F401

_CU_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_code.cu")

if torch.cuda.is_available():
    os.makedirs("./.jit_cache/", exist_ok=True)
    _cpp_module = load(
        name="hpc_intermediate_01_cuda_gemm",
        sources=[_CU_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory="./.jit_cache/",
        verbose=False,
    )
    run_gemm = _cpp_module.run_gemm
else:
    _cpp_module = MagicMock()
    run_gemm = _cpp_module.run_gemm


class CudaGemm:
    """Exercises covering CUDA General Matrix Multiplication."""

    @classmethod
    def run_naive_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Execute the naive CUDA GEMM kernel.

        Args:
            a: Left operand matrix of shape (M, K).
            b: Right operand matrix of shape (K, N).

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        return run_gemm(a, b, False)

    @classmethod
    def run_tiled_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Execute the shared-memory tiled CUDA GEMM kernel.

        Args:
            a: Left operand matrix of shape (M, K).
            b: Right operand matrix of shape (K, N).

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        return run_gemm(a, b, True)

    @classmethod
    def run_gemm(cls, a: torch.Tensor, b: torch.Tensor, use_tiled: bool = False) -> torch.Tensor:
        """Execute CUDA GEMM kernel (naive or tiled).

        Args:
            a: Left operand matrix of shape (M, K).
            b: Right operand matrix of shape (K, N).
            use_tiled: If True, uses shared memory tiling.

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        return run_gemm(a, b, use_tiled)
