"""
student_code.py — Module 01: Properties & Reshaping
=====================================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops or list comprehensions.
- Every function must return a ``numpy.ndarray``.
- Use ONLY NumPy array attributes and methods — no manual size calculations.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith basic 02
"""

import numpy as np


class PropertiesReshaping:
    """Exercises for array metadata attributes and structural reshaping."""

    @classmethod
    def reshape_to_matrix(cls, arr: np.ndarray) -> np.ndarray:
        """Reshape a 1-D array of 12 elements into a 2-D matrix of shape (3, 4).

        The transformation must preserve the order of elements (row-major / C-order,
        which is the default in NumPy).  No data copying should occur if possible
        (i.e., return a view over the existing buffer).

        Example:
            Input:  np.arange(12)  -> [0, 1, 2, ..., 11]
            Output: [[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]]

        Allowed API : arr.reshape()
        Forbidden   : Python loops, manual index arithmetic

        Args:
            arr: 1-D NumPy array with exactly 12 elements.

        Returns:
            np.ndarray: 2-D array of shape (3, 4).
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement reshape_to_matrix()")

    @classmethod
    def flatten_and_cast(cls, arr: np.ndarray) -> np.ndarray:
        """Flatten a 2-D integer array to 1-D and cast every element to float64.

        The output must be a contiguous 1-D array (not a view).
        Row-major (C-order) flattening is expected.

        Example:
            Input:  np.array([[1, 2], [3, 4]], dtype=int32)
            Output: np.array([1., 2., 3., 4.], dtype=float64)
            Output shape: (4,)

        Allowed API : arr.flatten(), arr.astype()
        Forbidden   : Python loops, list(), manual element access

        Args:
            arr: 2-D NumPy array of any integer dtype.

        Returns:
            np.ndarray: 1-D float64 array containing all elements in row-major order.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement flatten_and_cast()")
