"""
hint_01_memory_layout.py
=========================
Hints for tier ``advanced``, lesson ``01_memory_layout``.
"""

MODULE = "advanced/01_memory_layout"

HINT = {
    "MemoryLayout": {
        "ensure_c_contiguous": {
            "major": (
                "Check arr.flags.c_contiguous. If True, return arr unmodified.\n"
                "If False, use np.ascontiguousarray(arr) to convert it."
            ),
            "minor": (
                "if arr.flags.c_contiguous:\n"
                "    return arr\n"
                "return np.ascontiguousarray(arr)\n\n"
                "This guarantees C-contiguous memory layout while avoiding redundant copies."
            ),
        },
        "check_memory_share": {
            "major": (
                "Use np.may_share_memory(arr1, arr2) or inspect the base attribute.\n"
                "A slice shares memory buffer with its parent array."
            ),
            "minor": (
                "return bool(np.may_share_memory(arr1, arr2))\n\n"
                "np.may_share_memory safely checks bounding memory addresses."
            ),
        },
        "get_row_stride_bytes": {
            "major": (
                "Access arr.strides[0] to determine the number of bytes between rows.\n"
                "In a 2D array, axis 0 represents rows."
            ),
            "minor": (
                "return int(arr.strides[0])\n\n"
                "Strides define step sizes in bytes for multidimensional memory traversal."
            ),
        },
    },
}
