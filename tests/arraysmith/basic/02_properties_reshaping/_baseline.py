"""
_baseline.py — Module 01: Properties & Reshaping (Reference Solutions)
=======================================================================
This file contains the optimal NumPy solutions used as ground truth by the
test suite.  Do NOT modify this file.
"""

import numpy as np


class PropertiesReshapingBaseline:
    """Reference implementations for Module 01 exercises."""

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
        return arr.reshape(3, 4)

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
        return arr.flatten().astype(np.float64)
