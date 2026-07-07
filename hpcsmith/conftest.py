"""
hpcsmith/conftest.py
==========================
Pytest configuration for the ``hpcsmith`` curriculum module.

Guards the entire test suite against environments where the optional ``torch``
extra has not been installed, or where no C++/CUDA compiler or GPU is detected.
"""

from __future__ import annotations

import shutil

import pytest

torch = pytest.importorskip(
    "torch",
    reason=(
        "Optional dependency 'torch' is not installed. "
        "Run `uv sync --extra torch` to enable hpcsmith tests."
    ),
)


def _has_cpp_compiler() -> bool:
    """Check whether a valid C++ compiler exists in PATH."""
    return any(
        shutil.which(compiler) is not None
        for compiler in ("c++", "g++", "clang", "clang++", "nvcc")
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Dynamically skip C++ or CUDA dependent HPC test items when compiler or GPU is unavailable."""
    has_cuda = torch.cuda.is_available()
    has_cpp = _has_cpp_compiler()

    skip_cuda = pytest.mark.skip(reason="No CUDA-capable device detected.")
    skip_cpp = pytest.mark.skip(reason="No C++ compiler (c++/g++/clang) detected in PATH.")

    for item in items:
        path_str = str(getattr(item, "path", item.fspath)).lower()
        if ("cuda" in path_str or ".cu" in path_str) and not has_cuda:
            item.add_marker(skip_cuda)
        elif (
            "cpp" in path_str or "c++" in path_str or ".cpp" in path_str or ".cc" in path_str
        ) and not has_cpp:
            item.add_marker(skip_cpp)
