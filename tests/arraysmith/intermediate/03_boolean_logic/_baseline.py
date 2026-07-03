"""
_baseline.py — Module 05 Reference Implementation
=================================================
Reference implementations for Module 05 (Boolean Logic).
Used by ``compare_and_benchmark()`` to verify student correctness and speed.
"""

import numpy as np


class BooleanLogicBaseline:
    """Reference implementations for Module 05 — Boolean Logic."""

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
        return np.where(prices > threshold, prices * (1.0 - discount_rate), prices).astype(np.float64)

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
        return np.all(np.isclose(matrix_a, matrix_b, rtol=rtol), axis=1)
