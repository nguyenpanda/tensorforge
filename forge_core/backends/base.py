"""
forge_core/backends/base.py
============================
Abstract base class enforcing the HPC execution backend lifecycle.

Architecture
------------
Every compute backend — whether NumPy, CUDA, OpenCL, or a custom compiled
kernel — MUST implement the four-stage lifecycle defined here.  The strict
ordering guarantees safe resource management and eliminates cold-start timing
artefacts in benchmarks:

    setup()   → Allocate host/device memory and transfer data (H2D).
    warmup()  → Execute a throw-away run to trigger JIT compilation or
                CUDA stream initialisation, preventing first-call latency
                from contaminating timed measurements.
    execute() → Run the actual computation.  This is the ONLY call whose
                wall-clock time is included in benchmark measurements.
    teardown() → Release device memory, destroy CUDA contexts, and free
                 any other acquired resources.

The class also implements the context manager protocol so backends can be
used in ``with`` blocks.  ``__enter__`` calls ``setup`` + ``warmup``;
``__exit__`` calls ``teardown``, guaranteeing cleanup even on exception.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Literal


class ExecutionBackend(ABC):
    """Abstract contract for TensorForge execution backends.

    All concrete backends must implement the four lifecycle methods below.
    The ordering—``setup → warmup → execute → teardown``—is mandatory and
    encodes the requirements of latency-sensitive HPC benchmarking.

    Lifecycle
    ---------
    setup()
        Allocate host or device memory, transfer input tensors from host to
        device (H2D), and perform any one-time initialisation that should
        not be included in timed execution.

    warmup()
        Execute the target computation once without measuring wall-clock
        time.  For JIT-compiled backends (Triton, torch.compile, CUDA) this
        forces kernel compilation so that the first *timed* call does not
        include compilation overhead.  NumPy backends may implement this as
        a ``pass``.

    execute()
        Run the actual timed computation and return its result.  Only the
        wall-clock time of this method is included in benchmark measurements.

    teardown()
        Free all memory allocations, destroy CUDA streams or contexts, and
        reset backend state.  Must be idempotent — safe to call even if
        ``setup`` was never completed.
    """

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Allocate resources and transfer data to the computation device.

        Implementations should:
        - Allocate host / device buffers.
        - Copy input tensors from host to device (H2D transfer).
        - Perform any one-time initialisation that must not pollute timing.

        Args:
            *args: Positional arguments to pass to the setup step.
            **kwargs: Keyword arguments to pass to the setup step.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Perform a cold-start run to eliminate JIT / compilation overhead.

        For NumPy backends this is a no-op.  For CUDA or Triton backends
        this forces kernel compilation so the first *measured* call in
        ``execute()`` reflects steady-state performance.
        """

    @abstractmethod
    def execute(self) -> Any:
        """Execute the computation and return its result.

        This is the ONLY method whose wall-clock time is measured by the
        benchmarking harness.  Implementations must not perform allocation
        or H2D transfers here — those belong in ``setup``.

        Returns:
            Any: The computation result, passed directly to the correctness
                 verification layer.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release all acquired resources.

        Must be idempotent: safe to call repeatedly or even if ``setup``
        was never invoked.  For CUDA backends this destroys streams and
        frees device memory.  For NumPy backends this is typically a no-op.
        """

    # ------------------------------------------------------------------
    # Context manager protocol — guarantees teardown on exception.
    # ------------------------------------------------------------------

    def __enter__(self) -> ExecutionBackend:
        """Invoke ``setup`` and ``warmup``; return ``self`` for use in ``with`` blocks.

        Returns:
            ExecutionBackend: The backend instance after initialisation.
        """
        self.setup()
        self.warmup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Invoke ``teardown`` unconditionally, then propagate any exception.

        Args:
            exc_type: Exception class, or ``None`` if no exception was raised.
            exc_val: Exception instance, or ``None``.
            exc_tb: Traceback object, or ``None``.

        Returns:
            bool: Always ``False`` — exceptions are never suppressed.
        """
        self.teardown()
        return False


__all__ = ["ExecutionBackend"]
