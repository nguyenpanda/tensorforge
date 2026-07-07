"""
student_code.py — Lesson 01: C++ Integration
=============================================
Transparent JIT compiler wrapper for the C++ tensor addition extension.
"""

from __future__ import annotations

import os
import shutil
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.cpp_extension import load

from hint import show_hint  # noqa: F401

_CPP_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_code.cpp")
_has_cpp = any(shutil.which(c) is not None for c in ("c++", "g++", "clang", "clang++"))

if _has_cpp:
    os.makedirs("./.jit_cache/", exist_ok=True)
    _cpp_module = load(
        name="hpc_basic_01_cpp_integration",
        sources=[_CPP_SOURCE],
        extra_cflags=["-O3"],
        build_directory="./.jit_cache/",
        verbose=False,
    )
    add_tensors = _cpp_module.add_tensors
else:
    _cpp_module = MagicMock()
    add_tensors = _cpp_module.add_tensors


class CppIntegration:
    """Exercises covering C++ PyTorch extension integration."""

    @classmethod
    def run_cpp_addition(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two NumPy arrays using the C++ PyTorch extension kernel.

        Args:
            a: First operand array.
            b: Second operand array.

        Returns:
            np.ndarray: Element-wise sum of a and b.
        """
        tensor_a = torch.from_numpy(np.ascontiguousarray(a))
        tensor_b = torch.from_numpy(np.ascontiguousarray(b))
        result_tensor = add_tensors(tensor_a, tensor_b)
        return result_tensor.cpu().numpy()
