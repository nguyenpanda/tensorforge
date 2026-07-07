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

import gc
from collections.abc import Callable, Generator
from typing import Any

import pytest

# Skip every test in this subtree when torch is not present.
torch = pytest.importorskip(
    "torch",
    reason=(
        "Optional dependency 'torch' is not installed. "
        "Run `uv sync --extra torch` to enable tensorsmith tests."
    ),
)


def apple_silicon_trap(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute PyTorch forward/backward calls with Apple Silicon MPS trap.

    Catch BOTH NotImplementedError and RuntimeError (verifying 'MPS' is in the error string)
    to seamlessly fallback to CPU.
    """
    try:
        return fn(*args, **kwargs)
    except (NotImplementedError, RuntimeError) as e:
        err_str = str(e).upper()
        if isinstance(e, NotImplementedError) or "MPS" in err_str:
            cpu_args = [arg.to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args]
            cpu_kwargs = {
                k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
            }
            return fn(*cpu_args, **cpu_kwargs)
        raise


@pytest.fixture(name="apple_silicon_trap")
def apple_silicon_trap_fixture() -> Callable[..., Any]:
    """Fixture returning the apple_silicon_trap wrapper for PyTorch forward/backward calls."""
    return apple_silicon_trap


@pytest.fixture
def device() -> Generator[str, None, None]:
    """Yields optimal computing device (cuda -> mps -> cpu) with memory GC teardown."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    yield dev

    gc.collect()
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


@pytest.fixture(autouse=True)
def _tensorsmith_memory_gc_teardown() -> Generator[None, None, None]:
    """Autouse fixture to enforce memory GC teardown after every test to prevent OOM state bleed."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()
