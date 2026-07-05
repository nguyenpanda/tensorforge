"""
student_code.py — Lesson 01: C++ Integration
=============================================
Introduction
------------
This lesson demonstrates the TensorForge HPC Bridge workflow: writing a C++
PyTorch extension, compiling it on-the-fly, and invoking it through the
:class:`~forge_core.backends.cuda_backend.CudaJitBackend` lifecycle.

The C++ kernel is located at ``hpcsmith/native/01_cpp_addition.cpp`` and
exposes a single function, ``add_tensors``, which returns the element-wise
sum of two :class:`torch.Tensor` operands.  No manual build step is required;
the backend compiles the extension at first instantiation.

HPC Bridge lifecycle
--------------------
::

    backend.setup(tensor_a, tensor_b)
    backend.warmup()
    result = backend.execute()
    backend.teardown()

On macOS (no CUDA GPU), the backend transparently runs in CPU-only mode.

Run your tests with:
    uv run pytest tests/hpcsmith/basic/01_cpp_integration/test_01.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from forge_core.backends.cuda_backend import CudaJitBackend

_NATIVE_DIR: Path = Path(__file__).parent.parent.parent / "native"
_CPP_SOURCE: str = str(_NATIVE_DIR / "01_cpp_addition.cpp")


def _get_backend() -> CudaJitBackend:
    """Return a :class:`CudaJitBackend` for the addition kernel.

    The backend automatically caches the compiled extension in ``_module_cache``,
    invoking ``torch.utils.cpp_extension.load`` only once per process.

    Returns:
        CudaJitBackend: Configured backend instance for the addition kernel.
    """
    return CudaJitBackend(
        source_path=_CPP_SOURCE,
        module_name="cpp_addition",
        function_name="add_tensors",
    )


class CppIntegration:
    """Exercises covering C++ PyTorch extension integration via the HPC Bridge.

    Each method demonstrates the journey from raw NumPy arrays to C++-accelerated
    computation and back, using
    :class:`~forge_core.backends.cuda_backend.CudaJitBackend` as the bridge.
    """

    @classmethod
    def run_cpp_addition(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two NumPy arrays using a C++ PyTorch extension via the HPC Bridge.

        Converts *a* and *b* to :class:`torch.Tensor`, invokes the ``add_tensors``
        C++ kernel through the :class:`~forge_core.backends.cuda_backend.CudaJitBackend`
        lifecycle, and returns the result as :class:`numpy.ndarray`.

        Args:
            a: First operand.  Must be a NumPy array with a dtype supported by
               PyTorch (e.g., ``float32``, ``float64``, ``int32``).
            b: Second operand.  Must match the shape and dtype of *a*.

        Returns:
            np.ndarray: Element-wise sum ``a + b`` with the same shape and dtype
            as the inputs.
        """
        tensor_a: torch.Tensor = torch.from_numpy(np.ascontiguousarray(a))
        tensor_b: torch.Tensor = torch.from_numpy(np.ascontiguousarray(b))

        backend = _get_backend()
        backend.setup(tensor_a, tensor_b)
        backend.warmup()
        result_tensor: torch.Tensor = backend.execute()
        backend.teardown()

        return result_tensor.cpu().numpy()
