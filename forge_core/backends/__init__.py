"""
forge_core/backends/__init__.py
================================
Public API for the TensorForge execution backend plugin system.

Plugin Architecture
-------------------
Backends implement the :class:`ExecutionBackend` ABC defined in
:mod:`forge_core.backends.base`.  New compute targets are added by:

1. Creating a module in this package (e.g. ``cuda_backend.py``).
2. Implementing the four lifecycle methods: ``setup``, ``warmup``, ``execute``,
   ``teardown``.
3. Exporting the class from this ``__init__.py``.

No changes to :mod:`forge_core.benchmark` or any lesson test file are required
when adding a new backend — the harness accepts any :class:`ExecutionBackend`
instance via dependency injection.

Optional Dependencies
---------------------
:class:`CudaJitBackend` requires PyTorch (``torch >= 2.0.0``), declared as an
optional dependency under ``[project.optional-dependencies] torch``.  When
PyTorch is **not** installed, :class:`CudaJitBackend` is bound to a stub class
that raises a descriptive :class:`RuntimeError` on instantiation, keeping the
NumPy-only ``arraysmith`` path free of any PyTorch import at module load time.

Install the optional dependency with::

    uv sync --extra torch
"""

from __future__ import annotations

from forge_core.backends.base import ExecutionBackend
from forge_core.backends.numpy_backend import NumpyBackend

try:
    from forge_core.backends.cuda_backend import CudaJitBackend
except ImportError:

    class _CudaJitBackendStub:  # type: ignore[no-redef]
        """Placeholder exported when the ``torch`` optional dependency is absent.

        Raises :class:`RuntimeError` on instantiation with an actionable message
        directing the user to install the ``torch`` extra via ``uv``.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Raise :class:`RuntimeError` unconditionally.

            Raises:
                RuntimeError: Always; includes the ``uv`` install command.
            """
            raise RuntimeError(
                "PyTorch is not installed in the current environment. "
                "To use the HPC Bridge, run:  uv sync --extra torch"
            )

    CudaJitBackend = _CudaJitBackendStub  # type: ignore[assignment,misc]


__all__ = [
    "ExecutionBackend",
    "NumpyBackend",
    "CudaJitBackend",
]
