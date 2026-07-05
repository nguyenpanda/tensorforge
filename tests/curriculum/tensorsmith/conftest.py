"""
tests/tensorsmith/conftest.py
==============================
Pytest configuration for the ``tensorsmith`` curriculum module.

Guards the entire test suite against environments where the optional ``torch``
extra has not been installed.  Importing this conftest at collection time causes
pytest to skip every test in this directory tree when ``torch`` is absent,
preventing ImportError noise during global ``tforge check all`` or CI runs that
do not install the heavy PyTorch dependency.

Install the extra before running tensorsmith tests::

    uv sync --extra torch
"""

from __future__ import annotations

import pytest

# Skip every test in this subtree when torch is not present.
torch = pytest.importorskip(
    "torch",
    reason=(
        "Optional dependency 'torch' is not installed. "
        "Run `uv sync --extra torch` to enable tensorsmith tests."
    ),
)
