"""
_baseline.py — Golden Standard Baseline for Lesson 01: C++ Integration
======================================================================
Provides the golden standard reference implementation using PyTorch native addition.
"""

from __future__ import annotations

import numpy as np
import torch


class CppIntegrationBaseline:
    """Golden standard baseline for C++ Integration."""

    @classmethod
    def run_cpp_addition(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two NumPy arrays using PyTorch native addition.

        Args:
            a: First operand array.
            b: Second operand array.

        Returns:
            np.ndarray: Element-wise sum of a and b.
        """
        tensor_a = torch.from_numpy(np.ascontiguousarray(a))
        tensor_b = torch.from_numpy(np.ascontiguousarray(b))
        return (tensor_a + tensor_b).cpu().numpy()
