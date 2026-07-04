"""
student_code.py — Module 00: Array Creation
============================================
Instructions
------------
Implement every method marked with ``raise NotImplementedError``.

Rules (enforced by the test suite):
- You MUST NOT use Python for-loops, list comprehensions, or generator
  expressions to produce the array values.
- Every function must return a ``numpy.ndarray``.
- Call ``show_hint()`` (replace the NotImplementedError line) for a guided hint if you are stuck.

Run your tests with:
    tforge check arraysmith basic 01
"""

import numpy as np


class ArrayCreation:
    """Exercises covering fundamental NumPy array-creation routines."""

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
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement create_integer_range()")

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
        # Replace the line below with show_hint() if you need help.
        raise NotImplementedError("Implement create_squared_range()")
