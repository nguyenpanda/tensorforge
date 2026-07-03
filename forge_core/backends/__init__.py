"""
forge_core/backends/__init__.py
================================
Public API for the TensorForge execution backend plugin system.

Plugin Architecture
-------------------
Backends implement the :class:`ExecutionBackend` ABC defined in
:mod:`forge_core.backends.base`.  New compute targets (CUDA, OpenCL, Triton,
MKL) are added by:

1. Creating a new module in this package (e.g. ``cuda_backend.py``).
2. Implementing the four lifecycle methods: ``setup``, ``warmup``, ``execute``,
   ``teardown``.
3. Exporting the class from this ``__init__.py``.

No changes to :mod:`forge_core.benchmark` or any lesson test file are required
when adding a new backend — the harness accepts any :class:`ExecutionBackend`
instance via dependency injection.
"""

from __future__ import annotations

from forge_core.backends.base import ExecutionBackend
from forge_core.backends.numpy_backend import NumpyBackend

__all__ = [
    "ExecutionBackend",
    "NumpyBackend",
]
