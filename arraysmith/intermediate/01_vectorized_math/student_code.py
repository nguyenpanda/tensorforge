"""
student_code.py — Module 03: Vectorized Math
=============================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops or list comprehensions.
- Use the ``axis`` parameter of NumPy reduction functions — do not call
  ``np.mean`` or ``np.sum`` on individual rows/columns inside a loop.
- Broadcasting must be used where subtraction or addition spans the whole array.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith intermediate 01
"""

import numpy as np

from hint import show_hint  # noqa: F401


class VectorizedMath:
    """Exercises covering axis-aware reductions and NumPy broadcasting."""

    @classmethod
    def row_means(cls, matrix: np.ndarray) -> np.ndarray:
        """Compute the arithmetic mean of each row of a 2-D array.

        The result must have one element per row of *matrix*.
        Use the ``axis`` parameter — do not loop over rows.

        Example:
            matrix = np.array([[1, 2, 3],
                                [4, 5, 6]])   shape=(2, 3)

            Output = np.array([2.0, 5.0])     shape=(2,)
                               ↑       ↑
                            mean(row0) mean(row1)

        Allowed API : np.mean() with the correct ``axis``
        Forbidden   : Python loops, computing individual row means manually

        Args:
            matrix: 2-D NumPy array of shape (M, N).

        Returns:
            np.ndarray: 1-D array of shape (M,) containing the mean of each row.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("TODO: Implement this function")

    @classmethod
    def normalize_columns(cls, matrix: np.ndarray) -> np.ndarray:
        """Subtract each column's mean from every element in that column.

        After this operation, every column of the result has a mean of zero.
        Use broadcasting — compute all column means in one call, then subtract
        the resulting 1-D mean vector from the 2-D matrix in a single expression.

        Example:
            matrix = np.array([[1.0, 10.0],
                                [3.0, 20.0],
                                [5.0, 30.0]])   shape=(3, 2)

            col_means = [3.0, 20.0]             shape=(2,)

            Output = np.array([[-2., -10.],
                                [ 0.,   0.],
                                [ 2.,  10.]])    shape=(3, 2)

        Allowed API : np.mean() with ``axis=0``, then broadcast subtraction
        Forbidden   : Python loops, normalizing one column at a time

        Args:
            matrix: 2-D NumPy array of shape (M, N) with float values.

        Returns:
            np.ndarray: 2-D array of the same shape where every column is
                        zero-mean.
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("TODO: Implement this function")
