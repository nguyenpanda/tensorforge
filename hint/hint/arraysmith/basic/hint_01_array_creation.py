"""
hint_01_array_creation.py
==========================
Hints for tier ``basic``, lesson ``01_array_creation``.

Structure:
    HINT[ClassName][method_name] = {"major": ..., "minor": ...}
"""

MODULE = "basic/01_array_creation"

HINT = {
    "ArrayCreation": {
        "create_integer_range": {
            "major": (
                "Use np.arange(start, stop) to create an integer sequence.\n"
                "np.arange returns integers by default when both arguments are ints."
            ),
            "minor": (
                "np.linspace also works but produces floats by default.\n"
                "np.arange(0, 100) gives exactly what you need here."
            ),
        },
        "create_squared_range": {
            "major": (
                "Create the integer sequence first, then apply the power operator.\n"
                "NumPy supports element-wise exponentiation with ** or np.power()."
            ),
            "minor": (
                "arr = np.arange(0, 100)\n"
                "result = arr ** 2\n"
                "No for-loop needed — NumPy broadcasts the operation over every element."
            ),
        },
    },
}
