"""
forge_core — Shared TensorForge Infrastructure
===============================================
This package contains architectural infrastructure shared across all TensorForge
curriculum modules (``arraysmith``, ``tensorsmith``, ``hpcsmith``).

Public API
----------
- :func:`~forge_core.benchmark.compare_and_benchmark`
- :class:`~forge_core.benchmark.BenchmarkConfig`
- :class:`~forge_core.benchmark.BenchmarkResult`
- :class:`~forge_core.dataset_manager.DatasetManager`
- :class:`~forge_core.backends.base.ExecutionBackend`
- :class:`~forge_core.backends.numpy_backend.NumpyBackend`
- :class:`~forge_core.backends.cuda_backend.CudaJitBackend`
- :func:`~hint.show_hint`
"""

from forge_core.backends import CudaJitBackend, ExecutionBackend, NumpyBackend
from forge_core.benchmark import BenchmarkConfig, BenchmarkResult, compare_and_benchmark
from forge_core.dataset_manager import DatasetManager
from hint import show_hint

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CudaJitBackend",
    "DatasetManager",
    "ExecutionBackend",
    "NumpyBackend",
    "compare_and_benchmark",
    "show_hint",
]
