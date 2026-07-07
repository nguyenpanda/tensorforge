"""
_baseline.py — Module 06 Reference Implementation
=================================================
Reference implementations for Module 06 (Memory Layout & Strides).
Used by ``compare_and_benchmark()`` to verify student correctness and speed.
"""

import numpy as np


class MemoryLayoutBaseline:
    """Reference implementations for Module 06 — Memory Layout & Strides."""

    @classmethod
    def ensure_c_contiguous(cls, arr: np.ndarray) -> np.ndarray:
        """Ensure an array is stored in C-contiguous memory layout without redundant copies.

        If *arr* is already C-contiguous (check ``arr.flags.c_contiguous``), return *arr*
        directly without copying so the memory buffer is preserved.
        If it is not C-contiguous (e.g. Fortran order or sliced), return a C-contiguous
        copy using ``np.ascontiguousarray``.

        Args:
            arr: Input NumPy array of arbitrary dimension and layout.

        Returns:
            C-contiguous NumPy array with identical elements to *arr*.

        """
        if arr.flags.c_contiguous:
            return arr
        return np.ascontiguousarray(arr)

    @classmethod
    def check_memory_share(cls, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        """Determine if two arrays share the same underlying memory buffer.

        An array slice or view shares data with its parent (inspecting ``base`` or
        using ``np.may_share_memory``), whereas an explicit copy allocates new memory.

        Args:
            arr1: First NumPy array.
            arr2: Second NumPy array.

        Returns:
            True if *arr1* and *arr2* share memory, False otherwise.

        """
        return bool(np.may_share_memory(arr1, arr2))

    @classmethod
    def get_row_stride_bytes(cls, arr: np.ndarray) -> int:
        """Extract the number of bytes required to step from one row to the next in a 2D array.

        Access the array's ``strides`` tuple to determine the byte step for axis 0.

        Args:
            arr: 2D NumPy array of shape (N, M).

        Returns:
            Integer number of bytes for stride along axis 0.

        """
        return int(arr.strides[0])
