"""
hint_03_indexing_slicing.py
============================
Hints for tier ``basic``, lesson ``03_indexing_slicing``.
"""

MODULE = "basic/03_indexing_slicing"

HINT = {
    "IndexingSlicing": {
        "filter_above_threshold": {
            "major": (
                "Create a boolean mask by comparing the array to the threshold,\n"
                "then use that mask to index the array.\n"
                "Both steps can be written as a single expression."
            ),
            "minor": (
                "mask = arr > threshold      ← boolean array of True/False\n"
                "result = arr[mask]          ← selects only True positions\n"
                "Or shorter: arr[arr > threshold]"
            ),
        },
        "gather_by_indices": {
            "major": (
                "Pass the indices array directly inside the square brackets.\n"
                "NumPy will fetch every requested position in one C-speed operation."
            ),
            "minor": (
                "arr[indices]  is all you need.\n"
                "This is called 'fancy indexing' — the output order follows\n"
                "the order of values in the indices array."
            ),
        },
    },
}
