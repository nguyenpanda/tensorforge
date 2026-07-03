"""
student_code.py — Module 02: Indexing & Slicing
================================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops, list comprehensions, or generator
  expressions.
- Use NumPy boolean masking and/or fancy indexing exclusively.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith basic 03
"""

import numpy as np
from hint import show_hint


class IndexingSlicing:
    """Exercises covering boolean masking and integer (fancy) indexing."""

    @classmethod
    def filter_above_threshold(
        cls, arr: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return a 1-D array containing only elements strictly greater than threshold.

        Construct a boolean mask by evaluating the condition directly on the
        array, then apply that mask as an index: ``arr[mask]``.
        Do NOT modify the input array.

        Example:
            arr       = np.array([-3, -1, 0, 2, 5, 8])
            threshold = 1.0
            Output    = np.array([2, 5, 8])

        Allowed API : boolean indexing ``arr[arr > threshold]``
        Forbidden   : Python loops, list comprehensions, np.where (for extraction)

        Args:
            arr: 1-D NumPy array of numeric values.
            threshold: Scalar comparison value.

        Returns:
            np.ndarray: 1-D array containing only the elements > threshold,
                        preserving their original order.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement filter_above_threshold()")

    @classmethod
    def gather_by_indices(
        cls, arr: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Select elements from *arr* at positions given by *indices*.

        Use fancy (integer) indexing — pass the ``indices`` array directly into
        ``arr[...]`` to retrieve all requested elements in a single operation.

        Example:
            arr     = np.array([10, 20, 30, 40, 50])
            indices = np.array([4, 0, 2])
            Output  = np.array([50, 10, 30])

        Note: ``indices`` may contain duplicates and need not be sorted.
        The output order must follow the order of ``indices``.

        Allowed API : fancy indexing ``arr[indices]``
        Forbidden   : Python loops, explicit element-by-element access

        Args:
            arr: 1-D source NumPy array.
            indices: 1-D integer array of valid positions into *arr*.

        Returns:
            np.ndarray: 1-D array of elements gathered at the requested positions.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement gather_by_indices()")
