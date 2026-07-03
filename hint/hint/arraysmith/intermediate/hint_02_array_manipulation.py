"""
hint_02_array_manipulation.py
==============================
Hints for tier ``intermediate``, lesson ``02_array_manipulation``.
"""

MODULE = "intermediate/02_array_manipulation"

HINT = {
    "ArrayManipulation": {
        "stack_rows": {
            "major": (
                "Use np.vstack() and pass the list of arrays directly.\n"
                "vstack treats each 1-D array as one row and stacks them vertically."
            ),
            "minor": (
                "np.vstack([a, b, c])  →  2-D array, one row per input array\n\n"
                "If your input is already a list, pass it directly:\n"
                "  np.vstack(arrays)"
            ),
        },
        "concatenate_side_by_side": {
            "major": (
                "Use np.concatenate() with axis=1 to join columns side by side.\n"
                "axis=0 adds rows (wrong here); axis=1 adds columns (correct)."
            ),
            "minor": (
                "np.concatenate([left, right], axis=1)\n\n"
                "Both arrays must have the same number of rows (shape[0]).\n"
                "The output width = left.shape[1] + right.shape[1]."
            ),
        },
    },
}
