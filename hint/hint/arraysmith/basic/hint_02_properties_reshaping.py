"""
hint_02_properties_reshaping.py
================================
Hints for tier ``basic``, lesson ``02_properties_reshaping``.
"""

MODULE = "basic/02_properties_reshaping"

HINT = {
    "PropertiesReshaping": {
        "reshape_to_matrix": {
            "major": "Use arr.reshape(rows, cols) to change the shape of the array.",
            "minor": (
                "The target shape is (3, 4).\n"
                "arr.reshape(3, 4) returns a view with the new shape — no data is copied."
            ),
        },
        "flatten_and_cast": {
            "major": (
                "Chain two method calls: arr.flatten() then .astype(np.float64).\n"
                "flatten() always returns a 1-D copy in row-major order."
            ),
            "minor": (
                "arr.flatten()           → 1-D int array (copy)\n"
                "arr.flatten().astype(np.float64)  → 1-D float64 array\n"
                "No loop is needed — astype() converts every element at C speed."
            ),
        },
    },
}
