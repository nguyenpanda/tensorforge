"""
hint_01_cpp_integration.py
==========================
Hints for HPCSmith tier ``basic``, lesson ``01_cpp_integration``.
"""

MODULE = "basic/01_cpp_integration"

HINT = {
    "CppIntegration": {
        "run_cpp_addition": {
            "major": (
                "Write your C++/CUDA logic in student_code.cpp. The Python file is only used to compile your code.\n"
                "In student_code.cpp, implement element-wise addition using PyTorch's ATen C++ API.\n"
                "In C++, torch::Tensor objects overload standard arithmetic operators like +."
            ),
            "minor": (
                "Check tensor device and dtype matching first using TORCH_CHECK.\n"
                "Then simply return a + b; to perform vectorized ATen addition."
            ),
        },
    },
}
