"""
forge_core/backends/numpy_backend.py
======================================
NumPy implementation of the :class:`~forge_core.backends.base.ExecutionBackend` ABC.

This backend wraps an arbitrary zero-argument callable and executes it on the
CPU using NumPy.  It is the default backend used by
:func:`~forge_core.benchmark.compare_and_benchmark` when no explicit backend
is supplied by the caller.

CUDA Cold-Start Notes
---------------------
``warmup()`` is intentionally a ``pass`` here because NumPy operations are
executed eagerly by the CPython interpreter with no JIT compilation step.
Future GPU backends (``CudaBackend``, ``TritonBackend``) MUST override
``warmup`` to call the target kernel once before any timed measurement.

Memory Management
-----------------
NumPy allocates and frees CPU memory automatically via the Python garbage
collector.  ``setup()`` and ``teardown()`` are therefore no-ops for this
backend.  GPU backends must override both to manage device-side allocations
explicitly.
"""

from __future__ import annotations

from typing import Any, Callable

from forge_core.backends.base import ExecutionBackend


class NumpyBackend(ExecutionBackend):
    """CPU-only NumPy execution backend.

    Wraps a zero-argument callable and exposes it through the standard
    :class:`~forge_core.backends.base.ExecutionBackend` lifecycle.

    Args:
        fn: Zero-argument callable whose return value is the computation result.
            Must not perform side effects that corrupt state across repeated calls.

    Example::

        from forge_core.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend(fn=lambda: np.arange(100, dtype=np.int8))
        with backend:
            result = backend.execute()
    """

    def __init__(self, fn: Callable[[], Any]) -> None:
        """Initialise the backend with the target callable.

        Args:
            fn: Zero-argument callable to be executed and timed.
        """
        self._fn: Callable[[], Any] = fn

    def setup(self) -> None:
        """No-op for NumPy: CPU memory is managed by the Python allocator.

        Retained in the interface to allow future subclasses (e.g. a backend
        that pre-pins host memory for faster DMA transfers) to extend behaviour
        without changing the call site.
        """

    def warmup(self) -> None:
        """No-op for NumPy: eager execution requires no JIT warm-up.

        GPU backends MUST override this to trigger kernel compilation before
        the first timed ``execute()`` call, preventing cold-start latency from
        inflating benchmark measurements.
        """

    def execute(self) -> Any:
        """Invoke the wrapped callable and return its result.

        This is the sole method whose wall-clock time is measured by the
        :func:`~forge_core.benchmark.compare_and_benchmark` harness.

        Returns:
            Any: The return value of the wrapped callable.
        """
        return self._fn()

    def teardown(self) -> None:
        """No-op for NumPy: the Python garbage collector handles deallocation.

        GPU backends MUST override this to call ``cudaFree`` or equivalent
        device-memory release APIs.
        """


__all__ = ["NumpyBackend"]
