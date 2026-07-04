"""
student_code.py — Lesson 01: CUDA General Matrix Multiplication (GEMM)
======================================================================
Provides the student-facing Python interface for invoking custom CUDA GEMM
kernels (naive and shared-memory tiled variants) compiled via the TensorForge
HPC Bridge.

The CUDA extension source is located at ``hpcsmith/native/01_cuda_gemm.cu`` and
exposes a single function, ``run_gemm(a, b, use_tiled)``, which performs 2D
float32 matrix multiplication on PyTorch tensors.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from forge_core.backends.cuda_backend import CudaJitBackend

_NATIVE_DIR: Path = Path(__file__).parent.parent.parent / "native"
_CU_SOURCE: str = str(_NATIVE_DIR / "01_cuda_gemm.cu")


@functools.lru_cache
def _get_backend() -> CudaJitBackend:
    """Return a cached CudaJitBackend instance for the CUDA GEMM kernel.

    Decorating with lru_cache ensures the backend instance and its underlying
    compiled module are instantiated only once per process.

    Returns:
        CudaJitBackend: Configured backend pointing to 01_cuda_gemm.cu.
    """
    return CudaJitBackend(
        source_path=_CU_SOURCE,
        module_name="cuda_gemm",
        function_name="run_gemm",
    )


class CudaGemm:
    """Exercises covering CUDA General Matrix Multiplication via the HPC Bridge."""

    @classmethod
    def run_naive_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Execute matrix multiplication using the naive CUDA kernel.

        Invokes the C++ extension kernel with use_tiled set to False, causing each
        thread to read directly from global memory without shared memory caching.

        Args:
            a: Left operand tensor of shape (M, K). Must be a 2D contiguous float32 tensor.
            b: Right operand tensor of shape (K, N). Must be a 2D contiguous float32 tensor.

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        backend = _get_backend()
        backend.setup(a, b, use_tiled=False)
        backend.warmup()
        result: torch.Tensor = backend.execute()
        backend.teardown()
        return result

    @classmethod
    def run_tiled_gemm(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Execute matrix multiplication using the shared-memory tiled CUDA kernel.

        Invokes the C++ extension kernel with use_tiled set to True, utilizing
        shared memory tiling to reduce global memory bandwidth consumption.

        Args:
            a: Left operand tensor of shape (M, K). Must be a 2D contiguous float32 tensor.
            b: Right operand tensor of shape (K, N). Must be a 2D contiguous float32 tensor.

        Returns:
            torch.Tensor: Matrix product of shape (M, N).
        """
        backend = _get_backend()
        backend.setup(a, b, use_tiled=True)
        backend.warmup()
        result: torch.Tensor = backend.execute()
        backend.teardown()
        return result
