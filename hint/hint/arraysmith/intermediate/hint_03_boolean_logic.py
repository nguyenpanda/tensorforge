"""
hint_03_boolean_logic.py
=========================
Hints for tier ``intermediate``, lesson ``03_boolean_logic``.
"""

MODULE = "intermediate/03_boolean_logic"

HINT = {
    "BooleanLogic": {
        "apply_discount": {
            "major": (
                "Use np.where(condition, x, y) to choose between discounted and original prices.\n"
                "The condition evaluates to True where prices strictly exceed the threshold."
            ),
            "minor": (
                "np.where(prices > threshold, prices * (1.0 - discount_rate), prices)\n\n"
                "Avoid writing loops; np.where broadcasts conditionals over the entire array."
            ),
        },
        "identify_close_rows": {
            "major": (
                "Use np.isclose(A, B, rtol=rtol) to compare elements, then use np.all(..., axis=1)\n"
                "to check if an entire row is close."
            ),
            "minor": (
                "mask = np.isclose(matrix_a, matrix_b, rtol=rtol)\n"
                "return np.all(mask, axis=1)\n\n"
                "axis=1 evaluates across columns for each row index."
            ),
        },
    },
}
