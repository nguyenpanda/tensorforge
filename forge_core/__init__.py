"""
forge_core — Shared TensorForge Infrastructure
===============================================
This package contains utilities shared across all TensorForge curriculum modules
(``arraysmith``, ``tensorsmith``, and any future additions).

Public API
----------
- :func:`~forge_core.benchmark.compare_and_benchmark`
- :class:`~forge_core.benchmark.BenchmarkConfig`
- :class:`~forge_core.benchmark.BenchmarkResult`
- :class:`~forge_core.backends.base.ExecutionBackend`
- :class:`~forge_core.backends.numpy_backend.NumpyBackend`
- :class:`~forge_core.backends.cuda_backend.CudaJitBackend`
"""

from forge_core.backends import CudaJitBackend, ExecutionBackend, NumpyBackend
from forge_core.benchmark import BenchmarkConfig, BenchmarkResult, compare_and_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "compare_and_benchmark",
    "CudaJitBackend",
    "ExecutionBackend",
    "NumpyBackend",
]
