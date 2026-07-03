"""
student_code.py — Module 05: Boolean Logic
==========================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops, while-loops, or list comprehensions.
- Use NumPy boolean logic functions (``np.where``, ``np.isclose``, ``np.all``, ``np.any``) exclusively.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith intermediate 03
"""

import numpy as np
from hint import show_hint


class BooleanLogic:
    """Exercises covering conditional array construction and approximate comparisons."""

    @classmethod
    def apply_discount(
        cls,
        prices: np.ndarray,
        threshold: float = 100.0,
        discount_rate: float = 0.20,
    ) -> np.ndarray:
        """Apply a conditional discount to item prices using ``np.where``.

        For every price in *prices* strictly greater than *threshold*, apply the
        discount by multiplying the price by ``(1.0 - discount_rate)``.
        Prices less than or equal to *threshold* remain unchanged.

        Args:
            prices: 1D or 2D array of item prices (float64).
            threshold: Price limit above which the discount applies.
            discount_rate: Fractional discount (e.g., 0.20 for 20% off).

        Returns:
            New float64 array of discounted prices with identical shape to *prices*.

        Example:
            >>> prices = np.array([50.0, 100.0, 150.0, 200.0])
            >>> BooleanLogic.apply_discount(prices, threshold=100.0, discount_rate=0.20)
            array([ 50., 100., 120., 160.])
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement apply_discount()")

    @classmethod
    def identify_close_rows(
        cls,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        rtol: float = 1e-4,
    ) -> np.ndarray:
        """Identify which rows of two matrices are approximately equal.

        Two rows are considered approximately equal if all corresponding elements
        satisfy the relative tolerance *rtol* using ``np.isclose`` and ``np.all``.

        Args:
            matrix_a: 2D array of shape (N, M).
            matrix_b: 2D array of shape (N, M) to compare against *matrix_a*.
            rtol: Relative tolerance parameter for comparison.

        Returns:
            1D boolean array of shape (N,) where True indicates the entire row is close.

        Example:
            >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> B = np.array([[1.00001, 2.0], [3.1, 4.0]])
            >>> BooleanLogic.identify_close_rows(A, B, rtol=1e-3)
            array([ True, False])
        """
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement identify_close_rows()")
