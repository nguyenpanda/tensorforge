"""
hint_01_vectorized_math.py
===========================
Hints for tier ``intermediate``, lesson ``01_vectorized_math``.
"""

MODULE = "intermediate/01_vectorized_math"

HINT = {
    "VectorizedMath": {
        "row_means": {
            "major": (
                "Use np.mean() with the axis parameter.\n"
                "axis=1 collapses the column dimension, leaving one value per row."
            ),
            "minor": (
                "np.mean(matrix, axis=1)  →  shape (M,)\n\n"
                "Memory trick: axis=1 moves 'sideways' across columns,\n"
                "producing a single mean per row."
            ),
        },
        "normalize_columns": {
            "major": (
                "Two steps, both vectorised:\n"
                "  1. Compute column means:  col_means = np.mean(matrix, axis=0)\n"
                "  2. Subtract with broadcasting:  matrix - col_means\n"
                "Broadcasting stretches col_means (shape N,) across all M rows."
            ),
            "minor": (
                "col_means = np.mean(matrix, axis=0)   # shape (N,)\n"
                "result    = matrix - col_means         # broadcasting: (M,N) - (N,)\n\n"
                "After this, np.mean(result, axis=0) should be ≈ [0, 0, ..., 0]."
            ),
        },
    },
}
