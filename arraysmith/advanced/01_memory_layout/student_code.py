"""
student_code.py — Module 06: Memory Layout & Strides
====================================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops or while-loops.
- Understand C-contiguous vs Fortran-contiguous memory ordering, array strides,
  and how NumPy distinguishes between views (shared data buffer via ``base``) and copies.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith advanced 01
"""

import numpy as np

from hint import show_hint  # noqa: F401


class MemoryLayout:
    """Exercises covering array contiguity, byte strides, and view vs copy mechanics."""

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
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("TODO: Implement this function")

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
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("TODO: Implement this function")

    @classmethod
    def get_row_stride_bytes(cls, arr: np.ndarray) -> int:
        """Extract the number of bytes required to step from one row to the next in a 2D array.

        Access the array's ``strides`` tuple to determine the byte step for axis 0.

        Args:
            arr: 2D NumPy array of shape (N, M).

        Returns:
            Integer number of bytes for stride along axis 0.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("TODO: Implement this function")
