"""
forge_core/benchmark.py
=======================
Shared benchmarking utility for TensorForge curriculum modules.

Provides :func:`compare_and_benchmark`, the primary entry-point used by every
``test_XX.py`` in any TensorForge curriculum module (``arraysmith``,
``tensorsmith``, etc.).

Responsibilities
----------------
1. **Correctness & Strict Dtype** — verifies the student's output matches the baseline using
   ``np.testing.assert_array_equal`` for integer arrays and
   ``np.testing.assert_allclose`` for floating-point arrays. Strictly enforces
   matching ``dtype`` between student and baseline outputs.

2. **Performance & Memory Profiling** — measures mean execution time of both implementations over
   ``n_runs`` iterations via :mod:`timeit`, and tracks peak memory usage via
   :mod:`tracemalloc`.  Fails the test if the student's code exceeds
   ``slowdown_threshold`` times the baseline's speed.

3. **Backend Plugin Support** — accepts an optional
   :class:`~forge_core.backends.base.ExecutionBackend` instance to route timed
   execution through the HPC lifecycle (``setup → warmup → execute → teardown``).
   When ``backend`` is ``None`` (the default), a :class:`~forge_core.backends.numpy_backend.NumpyBackend`
   is constructed transparently, preserving backward compatibility for all
   existing ``test_XX.py`` call sites.

4. **Fast Mode** — when the environment variable ``TFORGE_FAST_MODE=1`` is
   set (via ``tforge check --fast``), each backend is executed exactly once,
   only correctness assertions run, and the function returns immediately.
   Timing loops, ``tracemalloc``, and the performance scorecard are all skipped.

5. **Reporting** — returns a :class:`BenchmarkResult` namedtuple for optional
   inspection in test files.

Usage::

    from forge_core.benchmark import compare_and_benchmark, BenchmarkConfig

    def test_my_exercise():
        compare_and_benchmark(
            student_fn=lambda: MySolution.my_method(),
            baseline_fn=lambda: MyBaseline.my_method(),
        )

    # With an explicit backend (future CUDA / HPC use):
    from forge_core.backends import NumpyBackend
    compare_and_benchmark(
        student_fn=lambda: MySolution.my_method(),
        baseline_fn=lambda: MyBaseline.my_method(),
        student_backend=NumpyBackend(fn=MySolution.my_method),
    )
"""

from __future__ import annotations

import os
import timeit
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pytest
from nguyenpanda.swan import c24, reset

from forge_core.backends.base import ExecutionBackend

_ERR = c24["ff3333"]
_WARN = c24["ff8800"]
_CYAN = c24["00d7ff"]
_GOLD = c24["ffd700"]
_CHAD = c24["ff00ff"]
_OPT = c24["00ff87"]
_SUB = c24["ffd700"]
_MID = c24["ff8800"]
_SOY = c24["ff3333"]


@dataclass
class BenchmarkConfig:
    """Configuration parameters for :func:`compare_and_benchmark`.

    Attributes:
        slowdown_threshold: Maximum allowed ``student_time / baseline_time``
            ratio before the performance check fails.  Default ``5.0`` (5×).
        n_runs: Number of timed repetitions for computing mean execution time.
            Default ``50``.
        rtol: Relative tolerance for ``np.testing.assert_allclose`` (floats).
            Default ``1e-5``.
        atol: Absolute tolerance for ``np.testing.assert_allclose`` (floats).
            Default ``1e-8``.
        check_performance: When ``False``, only correctness is verified.
            Default ``True``.
        silent: When ``True``, suppresses printing the tier badge to console.
            Default ``False``.
    """

    slowdown_threshold: float = 5.0
    n_runs: int = 50
    rtol: float = 1e-5
    atol: float = 1e-8
    check_performance: bool = True
    silent: bool = False


class BenchmarkResult(NamedTuple):
    """Timing and memory summary returned by :func:`compare_and_benchmark`.

    In **Fast Mode** (``TFORGE_FAST_MODE=1``), all numeric fields are set to
    ``0.0`` and ``passed_performance`` is ``True`` because no profiling is
    performed — the result is a sentinel conveying that correctness passed.

    Attributes:
        student_time_ms: Mean student execution time in milliseconds.
            ``0.0`` in Fast Mode.
        baseline_time_ms: Mean baseline execution time in milliseconds.
            ``0.0`` in Fast Mode.
        slowdown_ratio: ``student_time_ms / baseline_time_ms``.
            ``0.0`` in Fast Mode.
        passed_performance: ``True`` if ratio is within the configured threshold
            or if Fast Mode bypassed the performance check entirely.
        student_peak_kb: Peak memory usage of student function in kilobytes.
            ``0.0`` in Fast Mode.
        baseline_peak_kb: Peak memory usage of baseline function in kilobytes.
            ``0.0`` in Fast Mode.
    """

    student_time_ms: float
    baseline_time_ms: float
    slowdown_ratio: float
    passed_performance: bool
    student_peak_kb: float = 0.0
    baseline_peak_kb: float = 0.0


def _time_function(fn: Callable[[], Any], n_runs: int) -> tuple[float, float, Any]:
    """Execute *fn* once for output capture and memory tracing, then time it over *n_runs* repeats.

    Args:
        fn: Zero-argument callable returning the value to check.
        n_runs: Number of repetitions for timing.

    Returns:
        ``(mean_time_seconds, peak_memory_kb, output)`` — output and memory are from the first call.
    """
    tracemalloc.start()
    output = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak / 1024.0

    elapsed = timeit.timeit(fn, number=n_runs)
    return elapsed / n_runs, peak_kb, output


def _assert_outputs_equal(
    student_output: Any,
    baseline_output: Any,
    rtol: float,
    atol: float,
) -> None:
    """Assert that *student_output* matches *baseline_output*.

    Dispatches to the appropriate NumPy testing function based on dtype:

    - Integer / boolean arrays → :func:`np.testing.assert_array_equal` (exact).
    - Floating-point arrays    → :func:`np.testing.assert_allclose` (tolerance).
    - Other types              → plain ``assert`` equality.

    Strictly enforces matching dtypes between student and baseline array outputs.

    Args:
        student_output: Value produced by the student's function.
        baseline_output: Ground-truth value from the baseline.
        rtol: Relative tolerance (float arrays only).
        atol: Absolute tolerance (float arrays only).
    """
    if baseline_output is None:
        return

    if student_output is None:
        pytest.fail(
            f"{_ERR}Your function returned None. "
            f"Make sure you have a 'return' statement and have replaced "
            f"NotImplementedError with your implementation.{reset}"
        )

    if isinstance(baseline_output, np.ndarray):
        if not isinstance(student_output, np.ndarray):
            pytest.fail(
                f"{_ERR}Expected a numpy.ndarray but got {type(student_output).__name__}.\n"
                f"Wrap your result with np.array(...) or use a NumPy creation function.{reset}"
            )

        if student_output.dtype != baseline_output.dtype:
            pytest.fail(
                f"{_ERR}Strict Dtype Enforcement failed!\n"
                f"Expected dtype {baseline_output.dtype!r}, but got {student_output.dtype!r}.\n"
                f"Please cast or construct your array with the exact dtype required by the baseline.{reset}"
            )

        is_inexact = np.issubdtype(baseline_output.dtype, np.inexact)
        try:
            if is_inexact:
                np.testing.assert_allclose(student_output, baseline_output, rtol=rtol, atol=atol)
            else:
                np.testing.assert_array_equal(student_output, baseline_output)
        except AssertionError as exc:
            pytest.fail(
                f"{_ERR}Output mismatch between your solution and the baseline:\n{exc}{reset}"
            )
    else:
        assert student_output == baseline_output, (
            f"{_ERR}Output mismatch: got {student_output!r}, expected {baseline_output!r}{reset}"
        )


def compare_and_benchmark(
    student_fn: Callable[[], Any],
    baseline_fn: Callable[[], Any],
    config: BenchmarkConfig | None = None,
    student_backend: ExecutionBackend | None = None,
    baseline_backend: ExecutionBackend | None = None,
) -> BenchmarkResult:
    """Compare correctness and measure performance of a student's solution.

    Steps:
        1. Run both functions (via their backends), measure peak memory via tracemalloc,
           and assert output equality and strict dtype.
        2. Time both over ``config.n_runs`` iterations.
        3. Fail if ``student_time / baseline_time > config.slowdown_threshold``.

    When ``student_backend`` or ``baseline_backend`` is provided, execution is
    routed through the full HPC lifecycle::

        backend.setup()    # allocate / H2D transfer
        backend.warmup()   # JIT / CUDA cold-start elimination
        result = backend.execute()
        backend.teardown() # free resources

    When either backend is ``None`` (the default), a :class:`NumpyBackend` is
    constructed transparently so the call signature remains backward-compatible
    with all existing ``test_XX.py`` files.

    Args:
        student_fn: Zero-argument callable wrapping the student's implementation.
            Used as the ``fn`` for a :class:`NumpyBackend` when no explicit
            ``student_backend`` is supplied.
        baseline_fn: Zero-argument callable wrapping the reference implementation.
            Used as the ``fn`` for a :class:`NumpyBackend` when no explicit
            ``baseline_backend`` is supplied.
        config: Benchmark configuration.  Uses :class:`BenchmarkConfig` defaults
            when ``None``.
        student_backend: Optional :class:`~forge_core.backends.base.ExecutionBackend`
            instance for the student's computation.  When ``None``, defaults to a
            :class:`~forge_core.backends.numpy_backend.NumpyBackend` wrapping
            ``student_fn``.
        baseline_backend: Optional :class:`~forge_core.backends.base.ExecutionBackend`
            instance for the baseline computation.  When ``None``, defaults to a
            :class:`~forge_core.backends.numpy_backend.NumpyBackend` wrapping
            ``baseline_fn``.

    Returns:
        :class:`BenchmarkResult` with timing and memory details.

    Raises:
        pytest.fail: On correctness mismatch, dtype mismatch, or performance threshold breach.
    """
    from forge_core.backends.numpy_backend import NumpyBackend  # local import avoids circular deps

    if config is None:
        config = BenchmarkConfig()

    # Resolve backends: fall back to NumpyBackend when the caller did not supply one.
    _student_backend = (
        student_backend if student_backend is not None else NumpyBackend(fn=student_fn)
    )
    _baseline_backend = (
        baseline_backend if baseline_backend is not None else NumpyBackend(fn=baseline_fn)
    )

    # ------------------------------------------------------------------
    # Fast Mode bypass: single execution, correctness only, no profiling.
    # Activated by setting TFORGE_FAST_MODE=1 in the subprocess environment
    # (via `tforge check --fast`) or directly in the calling process env.
    # ------------------------------------------------------------------
    if os.environ.get("TFORGE_FAST_MODE") == "1":
        with _baseline_backend:
            baseline_output = _baseline_backend.execute()
        with _student_backend:
            student_output = _student_backend.execute()

        _assert_outputs_equal(student_output, baseline_output, config.rtol, config.atol)

        _FAST_COLOR = c24["00d7ff"]
        print(f"  {_FAST_COLOR}[Fast Mode]{reset} Correctness check passed.")
        return BenchmarkResult(
            student_time_ms=0.0,
            baseline_time_ms=0.0,
            slowdown_ratio=0.0,
            passed_performance=True,
            student_peak_kb=0.0,
            baseline_peak_kb=0.0,
        )
    # ------------------------------------------------------------------
    # Full benchmark path.
    # ------------------------------------------------------------------

    with _baseline_backend:
        baseline_time_s, baseline_peak_kb, baseline_output = _time_function(
            _baseline_backend.execute, config.n_runs
        )

    with _student_backend:
        student_time_s, student_peak_kb, student_output = _time_function(
            _student_backend.execute, config.n_runs
        )

    _assert_outputs_equal(student_output, baseline_output, config.rtol, config.atol)

    student_ms = student_time_s * 1_000
    baseline_ms = baseline_time_s * 1_000
    ratio = student_ms / baseline_ms if baseline_ms > 0 else 1.0
    passed = True

    if ratio <= 1.05:
        tier_name = "Forge God / Chad"
        tier_color = _CHAD
    elif ratio <= 1.5:
        tier_name = "Optimized"
        tier_color = _OPT
    elif ratio <= 3.0:
        tier_name = "Sub-optimal"
        tier_color = _SUB
    elif ratio <= 10.0:
        tier_name = "Mid-wit"
        tier_color = _MID
    else:
        tier_name = "Soy dev"
        tier_color = _SOY

    tier_badge = f"{tier_color}[{tier_name}] ({ratio:.2f}x Baseline Speed){reset}"

    if config.check_performance and ratio > config.slowdown_threshold:
        passed = False
        pytest.fail(
            f"\n{_WARN}{'=' * 65}\n"
            f"⚠️  PERFORMANCE CHECK FAILED  {tier_badge}\n"
            f"{'=' * 65}{reset}\n"
            f"  Your solution : {_CYAN}{student_ms:>10.4f} ms{reset} | {_CYAN}{student_peak_kb:>8.2f} KB{reset}  (mean over {config.n_runs} runs)\n"
            f"  Baseline      : {_CYAN}{baseline_ms:>10.4f} ms{reset} | {_CYAN}{baseline_peak_kb:>8.2f} KB{reset}\n"
            f"  Slowdown      : {_ERR}{ratio:>10.2f}×{reset}  (threshold: {config.slowdown_threshold}×)\n"
            f"{_WARN}{'-' * 65}{reset}\n"
            f"  Your implementation is {_ERR}{ratio:.1f}× slower{reset} than the baseline.\n"
            f"  This usually means you are using a Python for-loop where a\n"
            f"  vectorized operation would suffice.\n\n"
            f"{_GOLD}  💡 Tip: Replace loops with NumPy array operations.\n"
            f"          e.g.  result = arr ** 2   (not:  [x**2 for x in arr]){reset}\n"
            f"{_WARN}{'=' * 65}{reset}\n"
        )

    if passed and config.check_performance and not config.silent:
        print(f"\n  {tier_badge}")

    return BenchmarkResult(
        student_time_ms=student_ms,
        baseline_time_ms=baseline_ms,
        slowdown_ratio=ratio,
        passed_performance=passed,
        student_peak_kb=student_peak_kb,
        baseline_peak_kb=baseline_peak_kb,
    )
