"""
_baseline.py — Module 04: Array Manipulation (Reference Solutions)
===================================================================
This file contains the optimal NumPy solutions used as ground truth by the
test suite.  Do NOT modify this file.
"""

import numpy as np


class ArrayManipulationBaseline:
    """Reference implementations for Module 04 exercises."""

    @classmethod
    def stack_rows(cls, arrays: list[np.ndarray]) -> np.ndarray:
        """Stack a list of 1-D arrays into a 2-D matrix (one array per row).

        Each element of *arrays* becomes a single row in the output matrix.
        Use ``np.vstack`` — do not construct the result with a loop.

        Example:
            arrays = [np.array([1, 2, 3]),
                      np.array([4, 5, 6]),
                      np.array([7, 8, 9])]

            Output = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])   shape=(3, 3)

        Allowed API : np.vstack()
        Forbidden   : Python loops, np.array(arrays) as a workaround loop

        Args:
            arrays: List of 1-D NumPy arrays, all with the same length.

        Returns:
            np.ndarray: 2-D array of shape (len(arrays), len(arrays[0])).

        """
        return np.vstack(arrays)

    @classmethod
    def concatenate_side_by_side(cls, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Concatenate two 2-D arrays horizontally (column-wise).

        *left* and *right* must have the same number of rows.  The output
        contains all columns from *left* followed by all columns from *right*.

        Use ``np.concatenate`` with the correct ``axis`` argument.

        Example:
            left  = np.array([[1, 2],
                                [3, 4]])   shape=(2, 2)

            right = np.array([[5, 6, 7],
                                [8, 9, 0]])   shape=(2, 3)

            Output = np.array([[1, 2, 5, 6, 7],
                                [3, 4, 8, 9, 0]])   shape=(2, 5)

        Allowed API : np.concatenate() with ``axis=1``
        Forbidden   : Python loops, np.hstack (use concatenate explicitly)

        Args:
            left:  2-D NumPy array of shape (M, K).
            right: 2-D NumPy array of shape (M, L).

        Returns:
            np.ndarray: 2-D array of shape (M, K+L).

        """
        return np.concatenate([left, right], axis=1)
