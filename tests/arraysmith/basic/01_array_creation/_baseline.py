"""
_baseline.py — Module 00: Array Creation (Reference Solutions)
===============================================================
This file contains the optimal, vectorized NumPy solutions for every exercise
in this module.  It is the ground truth used by the test suite for:

- Correctness comparison  (output must match exactly)
- Performance benchmarking (student code must be within the slowdown threshold)

Do NOT modify this file.
"""

import numpy as np


class ArrayCreationBaseline:
    """Reference implementations for Module 00 exercises."""

    @classmethod
    def create_integer_range(cls) -> np.ndarray:
        """Return a 1-D array containing integers from 0 to 99 (inclusive).

                Expected output shape : (100,)
                Expected dtype        : int8
                Expected values       : [0, 1, 2, ..., 98, 99]

                Allowed API           : np.arange
                Forbidden             : Python for-loops, list comprehensions, range()

                Returns:
                    np.ndarray: 1-D int8 array with values [0, 1, 2, ..., 99].

        """
        return np.arange(100, dtype=np.int8)

    @classmethod
    def create_squared_range(cls) -> np.ndarray:
        """Return an array of the first 100 perfect squares.

                Expected output shape : (100,)
                Expected dtype        : int64
                Expected values       : [0, 1, 4, 9, 16, ..., 9801]

                Hint: Create the integer sequence first, then apply an element-wise
                power operation — NumPy supports the ``**`` operator on arrays.

                Allowed API           : np.arange, ``**`` operator or np.power
                Forbidden             : Python for-loops, list comprehensions

                Returns:
                    np.ndarray: 1-D array with values [0, 1, 4, 9, ..., 9801].

        """
        return np.arange(100) ** 2
