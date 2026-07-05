"""
forge_core/backends/cuda_backend.py
=====================================
PyTorch JIT C++/CUDA extension backend for the TensorForge HPC Bridge.

Overview
--------
:class:`CudaJitBackend` compiles student-authored C++ and CUDA kernel source
files on-the-fly using ``torch.utils.cpp_extension.load``, eliminating any
CMake or ahead-of-time build step.

Dependency model
----------------
PyTorch is an **optional** dependency of TensorForge (``[project.optional-dependencies]
torch`` in ``pyproject.toml``).  Every ``torch`` import is guarded with
``try/except ImportError`` so the pure-NumPy ``arraysmith`` path remains fully
functional without PyTorch installed.

Device-aware execution
----------------------
On macOS and other platforms without a CUDA-capable GPU,
``torch.cuda.is_available()`` returns ``False``.  Calling ``.cuda()`` or
``torch.cuda.synchronize()`` on such systems raises
``AssertionError: Torch not compiled with CUDA enabled``.

The instance flag ``self._cuda_available``, evaluated once in ``__init__``,
gates every CUDA-specific operation:

- ``setup()``    — skips ``.cuda()``; tensors remain on CPU.
- ``execute()``  — skips ``torch.cuda.synchronize()``; CPU ops are synchronous.
- ``teardown()`` — skips ``torch.cuda.empty_cache()``; no VRAM to reclaim.

C++-only extensions (no ``__global__`` CUDA kernels) execute transparently in
CPU-only mode without any modification to the kernel source or student code.

Module caching
--------------
``_module_cache`` is a class-level dictionary keyed by ``(module_name, resolved_source_path)``.
Once compiled, a JIT extension module is reused across all subsequent instances
that share the same key, eliminating redundant invocations of
``torch.utils.cpp_extension.load`` within a single process.

Lifecycle
---------
::

    setup(*args, **kwargs)  — Tensors → device (CUDA) or CPU pass-through.
    warmup()                — Cold-start invocation; absorbs JIT/context-init penalty.
    execute()               — Timed invocation; ``torch.cuda.synchronize()`` on CUDA only.
    teardown()              — Clear ``_device_args`` and ``_device_kwargs``; ``empty_cache()`` on CUDA only.

Synchronization
---------------
CUDA kernel launches are **asynchronous** with respect to the host CPU.  Without
``torch.cuda.synchronize()`` in ``execute()``, the CPU-side timer records only
kernel *dispatch* latency (µs), not actual GPU *execution* time.  The barrier
ensures every measurement in the 50-iteration benchmark loop reflects true
device-side completion.
"""

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import Any, ClassVar

from forge_core.backends.base import ExecutionBackend

try:
    import torch  # type: ignore[import-untyped]
    from torch.utils.cpp_extension import load as _cpp_load  # type: ignore[import-untyped]

    _TORCH_AVAILABLE: bool = True
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()  # type: ignore[assignment]
    _cpp_load = MagicMock()  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from nguyenpanda.swan import c24
    from nguyenpanda.swan import reset as _reset

    _WARN_COLOR: str = c24["ff8800"]
except Exception:  # noqa: BLE001
    _WARN_COLOR = ""
    _reset = ""

_HPC_PREFIX: str = f"{_WARN_COLOR}[HPC Bridge]{_reset}"


class CudaJitBackend(ExecutionBackend):
    """PyTorch JIT-compiled C++/CUDA kernel execution backend.

    Compiles a student-authored C++ or CUDA source file at instantiation time
    using ``torch.utils.cpp_extension.load``, then exposes the compiled extension
    through the standard four-stage
    :class:`~forge_core.backends.base.ExecutionBackend` lifecycle.

    The backend is **device-aware**: when ``torch.cuda.is_available()`` returns
    ``False`` (macOS, CPU-only Linux), all CUDA-specific operations are skipped
    automatically.  C++-only extensions execute on CPU without modification.

    Compiled modules are cached in ``_module_cache`` at the class level.  The
    cache key is ``(module_name, resolved_source_path)``, so repeated
    instantiations within a process reuse the already-loaded extension without
    invoking the compiler again.

    Args:
        source_path: Absolute or relative path to the ``.cpp`` or ``.cu`` source
            file.  The file must expose a top-level C++ function whose name
            matches *function_name*.
        module_name: Unique Python-identifier name for the compiled extension.
            Use the lesson slug to avoid cache collisions across extensions.
        function_name: Name of the callable exported by the extension module.
            Bound to ``self.kernel`` after compilation via ``getattr``.

    Raises:
        RuntimeError: If PyTorch is not installed.  The message includes the
            exact ``uv`` install command.
        FileNotFoundError: If *source_path* does not refer to an existing file.
        AttributeError: If the compiled extension does not export *function_name*.
            The error message enumerates the available public symbols.

    Warns:
        UserWarning: When ``torch.cuda.is_available()`` returns ``False``.
            Execution continues in CPU-only mode.

    Example::

        backend = CudaJitBackend(
            source_path="tensorsmith/native/my_kernel.cu",
            module_name="my_kernel",
            function_name="forward",
        )
        backend.setup(torch.randn(1024))
        backend.warmup()
        result = backend.execute()
        backend.teardown()
    """

    _module_cache: ClassVar[dict[tuple[str, str], Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        source_path: str,
        module_name: str,
        function_name: str,
    ) -> None:
        """Compile the extension and extract the target callable.

        Args:
            source_path: Path to the ``.cpp`` or ``.cu`` source file.
            module_name: Unique Python-identifier name for the compiled module.
            function_name: Name of the C++ function to bind as ``self.kernel``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed in the current environment. "
                "To use the HPC Bridge, run:  uv sync --extra torch"
            )

        self._cuda_available: bool = torch.cuda.is_available()

        if not self._cuda_available:
            warnings.warn(
                f"{_HPC_PREFIX} torch.cuda.is_available() returned False.  "
                "No CUDA-capable GPU was detected (expected on macOS / CPU-only Linux).  "
                "Running in CPU-only mode: device transfers and synchronization are disabled.  "
                "C++-only extensions execute correctly; CUDA kernels will fail at the call site.",
                UserWarning,
                stacklevel=2,
            )

        self._source_path: str = source_path
        self._module_name: str = module_name
        self._function_name: str = function_name
        self._device_args: tuple[Any, ...] = ()
        self._device_kwargs: dict[str, Any] = {}

        resolved_source = str(Path(source_path).resolve())
        cache_key = (module_name, resolved_source)

        with self._lock:
            if cache_key in self._module_cache:
                self.module: Any = self._module_cache[cache_key]
            else:
                if not Path(source_path).exists():
                    raise FileNotFoundError(f"Source file not found: {source_path}")
                load_kwargs: dict[str, Any] = {
                    "name": module_name,
                    "sources": [resolved_source],
                    "verbose": False,
                }
                if os.environ.get("TFORGE_DEBUG_CPP") == "1":
                    load_kwargs["extra_cflags"] = [
                        "-fsanitize=address",
                        "-fno-omit-frame-pointer",
                        "-g",
                    ]
                    load_kwargs["extra_ldflags"] = ["-fsanitize=address"]
                self.module = _cpp_load(**load_kwargs)
                self._module_cache[cache_key] = self.module

        try:
            self.kernel: Any = getattr(self.module, function_name)
        except AttributeError as exc:
            available = [k for k in dir(self.module) if not k.startswith("_")]
            raise AttributeError(
                f"Compiled module '{module_name}' does not export function '{function_name}'. "
                f"Available public symbols: {available}"
            ) from exc

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Transfer input tensors to the computation device.

        On CUDA systems, each tensor argument (positional and keyword) is moved
        to device via ``.cuda()``.  On CPU-only systems, arguments are stored
        unchanged; the C++ kernel operates on CPU tensors without modification.

        Args:
            *args: :class:`torch.Tensor` instances to supply to the kernel.
            **kwargs: Keyword :class:`torch.Tensor` instances to supply to the kernel.
        """
        if self._cuda_available:
            self._device_args = tuple(t.cuda() if hasattr(t, "cuda") else t for t in args)
            self._device_kwargs = {
                k: v.cuda() if hasattr(v, "cuda") else v for k, v in kwargs.items()
            }
        else:
            self._device_args = tuple(args)
            self._device_kwargs = dict(kwargs)

    def warmup(self) -> None:
        """Execute the kernel once to absorb the cold-start penalty.

        On CUDA systems, the first launch initialises the CUDA context, JIT-
        compiles PTX, and creates the CUDA stream.  Running once here ensures
        ``execute()`` measures only steady-state throughput.

        Precondition: ``setup()`` must have been called.
        """
        self.kernel(*self._device_args, **self._device_kwargs)

    def execute(self) -> Any:
        """Invoke the kernel and, on CUDA systems, synchronize before returning.

        ``torch.cuda.synchronize()`` is mandatory on CUDA: launches are
        asynchronous, so without it the CPU timer records only dispatch latency
        rather than true GPU execution time.  On CPU-only systems the call is
        omitted because CPU operations are inherently synchronous.

        Returns:
            Any: The return value of the compiled kernel function.
        """
        result = self.kernel(*self._device_args, **self._device_kwargs)
        if self._cuda_available:
            torch.cuda.synchronize()
        return result

    def teardown(self) -> None:
        """Release device arguments and reclaim VRAM on CUDA systems.

        Clears ``_device_args`` and ``_device_kwargs`` to drop tensor references.
        On CUDA systems, ``torch.cuda.empty_cache()`` returns VRAM to the allocator
        pool, preventing OOM across successive benchmark iterations over large tensors.

        Idempotent: safe to call multiple times or before ``setup()``.
        """
        self._device_args = ()
        self._device_kwargs = {}
        if self._cuda_available:
            torch.cuda.empty_cache()


__all__ = ["CudaJitBackend"]
