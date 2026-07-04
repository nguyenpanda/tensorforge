"""
tests/hpcsmith/conftest.py
==========================
Pytest configuration for the ``hpcsmith`` curriculum module.

Guards the entire test suite against environments where the optional ``torch``
extra has not been installed, or where no CUDA-capable GPU is detected.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch",
    reason=(
        "Optional dependency 'torch' is not installed. "
        "Run `uv sync --extra torch` to enable hpcsmith tests."
    ),
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Dynamically skip CUDA-dependent HPC test items when CUDA GPU is unavailable."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="No CUDA-capable device detected.")
        for item in items:
            if "cuda" in str(item.fspath).lower():
                item.add_marker(skip_cuda)
