"""
tests/test_infrastructure.py
==============================
Unit tests for the TensorForge infrastructure.

Verifies the correctness and performance reporting of the
:mod:`forge_core.benchmark` module, the hint registry, and the
:mod:`forge_core.backends` plugin architecture.

Lesson paths follow Tier-Scoped Numbering (composite keys):

    basic/01_array_creation        (was 00_array_creation)
    basic/02_properties_reshaping  (was 01_properties_reshaping)
    basic/03_indexing_slicing      (was 02_indexing_slicing)
    intermediate/01_vectorized_math    (was 03_vectorized_math)
    intermediate/02_array_manipulation (was 04_array_manipulation)
    intermediate/03_boolean_logic      (was 05_boolean_logic)
    advanced/01_memory_layout          (was 06_memory_layout)

Run with:
    tforge check infra
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "arraysmith" / "basic" / "01_array_creation"))

from forge_core.ast_validator import ast_policy
from forge_core.backends import ExecutionBackend, NumpyBackend
from forge_core.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    compare_and_benchmark,
)
from main import run_generate, run_status

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_baseline(tier_lesson_path: str, class_name: str) -> Any:
    """Import a baseline class using Tier-Scoped composite path.

    Args:
        tier_lesson_path: Composite path ``"<tier>/<lesson_dir>"``.
        class_name: Baseline class name.

    Returns:
        type: The baseline class.
    """
    import importlib

    test_path    = str(ROOT / "tests" / "arraysmith" / tier_lesson_path)
    student_path = str(ROOT / "arraysmith" / tier_lesson_path)
    for name in ("_baseline", "student_code"):
        sys.modules.pop(name, None)
    for path in (test_path, student_path):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, student_path)
    sys.path.insert(0, test_path)
    mod = importlib.import_module("_baseline")
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# BenchmarkConfig defaults
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Verify BenchmarkConfig default values."""

    def test_default_threshold(self):
        assert BenchmarkConfig().slowdown_threshold == 5.0

    def test_default_n_runs(self):
        assert BenchmarkConfig().n_runs == 50

    def test_default_check_performance_enabled(self):
        assert BenchmarkConfig().check_performance is True

    def test_custom_values(self):
        cfg = BenchmarkConfig(slowdown_threshold=10.0, n_runs=200)
        assert cfg.slowdown_threshold == 10.0
        assert cfg.n_runs == 200


# ---------------------------------------------------------------------------
# compare_and_benchmark — happy path
# ---------------------------------------------------------------------------


class TestBenchmarkHappyPath:
    """compare_and_benchmark passes when student == baseline."""

    def test_returns_benchmark_result_namedtuple(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        result = compare_and_benchmark(
            student_fn=cls.create_integer_range,
            baseline_fn=cls.create_integer_range,
            config=BenchmarkConfig(n_runs=10, check_performance=False),
        )
        assert isinstance(result, BenchmarkResult)

    def test_slowdown_ratio_near_one_for_identical_fns(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        result = compare_and_benchmark(
            student_fn=cls.create_integer_range,
            baseline_fn=cls.create_integer_range,
            config=BenchmarkConfig(n_runs=20, check_performance=False),
        )
        assert result.slowdown_ratio < 5.0, (
            f"Identical fn ratio unexpectedly high: {result.slowdown_ratio}"
        )

    def test_passed_performance_true_within_threshold(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        result = compare_and_benchmark(
            student_fn=cls.create_integer_range,
            baseline_fn=cls.create_integer_range,
            config=BenchmarkConfig(n_runs=10, slowdown_threshold=100.0),
        )
        assert result.passed_performance is True

    def test_timing_fields_are_positive(self):
        cls = _load_baseline("basic/02_properties_reshaping", "PropertiesReshapingBaseline")
        arr = np.arange(12, dtype=np.int32)
        result = compare_and_benchmark(
            student_fn=lambda: cls.reshape_to_matrix(arr),
            baseline_fn=lambda: cls.reshape_to_matrix(arr),
            config=BenchmarkConfig(n_runs=10, check_performance=False),
        )
        assert result.student_time_ms >= 0.0
        assert result.baseline_time_ms >= 0.0

    @pytest.mark.parametrize(
        "student_s,expected_name,expected_hex",
        [
            (1.0, "Forge God / Chad", "ff00ff"),
            (1.2, "Optimized", "00ff87"),
            (2.0, "Sub-optimal", "ffd700"),
            (5.0, "Mid-wit", "ff8800"),
            (15.0, "Soy dev", "ff3333"),
        ],
    )
    def test_all_five_gamification_tiers(self, capsys, student_s, expected_name, expected_hex):
        from nguyenpanda.swan import c24
        expected_color = c24[expected_hex]

        with patch("forge_core.benchmark._time_function") as mock_time:
            mock_time.side_effect = [
                (1.0, 10.0, np.zeros(5, dtype=np.int8)),
                (student_s, 10.0, np.zeros(5, dtype=np.int8)),
            ]
            compare_and_benchmark(
                student_fn=lambda: np.zeros(5, dtype=np.int8),
                baseline_fn=lambda: np.zeros(5, dtype=np.int8),
                config=BenchmarkConfig(n_runs=1, check_performance=True, silent=False, slowdown_threshold=20.0),
            )
        captured = capsys.readouterr()
        assert expected_name in captured.out
        assert expected_color in captured.out

    def test_memory_profiling_captures_kb(self):
        result = compare_and_benchmark(
            student_fn=lambda: np.ones((500, 500), dtype=np.float64),
            baseline_fn=lambda: np.ones((500, 500), dtype=np.float64),
            config=BenchmarkConfig(n_runs=2, check_performance=False),
        )
        assert result.student_peak_kb > 0.0
        assert result.baseline_peak_kb > 0.0


# ---------------------------------------------------------------------------
# compare_and_benchmark — correctness failures
# ---------------------------------------------------------------------------


class TestBenchmarkCorrectnessFailures:
    """compare_and_benchmark fails correctly when outputs differ."""

    def test_wrong_values_fail(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        with pytest.raises(BaseException):
            compare_and_benchmark(
                student_fn=lambda: np.zeros(100, dtype=np.int8),
                baseline_fn=cls.create_integer_range,
                config=BenchmarkConfig(n_runs=5, check_performance=False),
            )

    def test_wrong_dtype_integer_caught_by_exact_comparison(self):
        """An off-by-one in an integer array must be caught."""
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")

        def student_off_by_one():
            arr = cls.create_integer_range().copy()
            arr[50] = 51
            return arr

        with pytest.raises(BaseException):
            compare_and_benchmark(
                student_fn=student_off_by_one,
                baseline_fn=cls.create_integer_range,
                config=BenchmarkConfig(n_runs=5, check_performance=False),
            )

    def test_none_return_fails_with_helpful_message(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        with pytest.raises(BaseException, match="None"):
            compare_and_benchmark(
                student_fn=lambda: None,
                baseline_fn=cls.create_integer_range,
                config=BenchmarkConfig(n_runs=5, check_performance=False),
            )

    def test_wrong_type_fails_with_helpful_message(self):
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")
        with pytest.raises(BaseException, match="ndarray"):
            compare_and_benchmark(
                student_fn=lambda: list(range(100)),
                baseline_fn=cls.create_integer_range,
                config=BenchmarkConfig(n_runs=5, check_performance=False),
            )

    def test_strict_dtype_mismatch_fails_validation(self):
        with pytest.raises(BaseException, match="Strict Dtype Enforcement failed!"):
            compare_and_benchmark(
                student_fn=lambda: np.zeros(10, dtype=np.float32),
                baseline_fn=lambda: np.zeros(10, dtype=np.float64),
                config=BenchmarkConfig(n_runs=2, check_performance=False),
            )


# ---------------------------------------------------------------------------
# compare_and_benchmark — performance failures
# ---------------------------------------------------------------------------


class TestBenchmarkPerformanceFailures:
    """compare_and_benchmark fails when student code is too slow."""

    def test_loop_implementation_fails_tight_threshold(self):
        """A Python loop must be flagged with an impossibly tight threshold."""
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")

        def loop_student():
            result = []
            for i in range(100):
                result.append(i)
            return np.array(result, dtype=np.int8)

        with pytest.raises(BaseException, match="PERFORMANCE"):
            compare_and_benchmark(
                student_fn=loop_student,
                baseline_fn=cls.create_integer_range,
                config=BenchmarkConfig(n_runs=50, slowdown_threshold=0.001),
            )

    def test_performance_check_skipped_when_disabled(self):
        """Slow code must not fail when check_performance=False."""
        cls = _load_baseline("basic/01_array_creation", "ArrayCreationBaseline")

        def slow_fn():
            import time
            time.sleep(0)
            return cls.create_integer_range()

        result = compare_and_benchmark(
            student_fn=slow_fn,
            baseline_fn=cls.create_integer_range,
            config=BenchmarkConfig(n_runs=5, check_performance=False),
        )
        assert result.passed_performance is True


# ---------------------------------------------------------------------------
# Backend plugin architecture
# ---------------------------------------------------------------------------


class TestBackendPluginArchitecture:
    """Verify the ExecutionBackend ABC and NumpyBackend implementation."""

    def test_numpy_backend_is_concrete(self):
        """NumpyBackend must instantiate without error."""
        backend = NumpyBackend(fn=lambda: np.arange(10, dtype=np.int64))
        assert isinstance(backend, ExecutionBackend)

    def test_numpy_backend_execute_returns_correct_result(self):
        expected = np.arange(10, dtype=np.int64)
        backend = NumpyBackend(fn=lambda: np.arange(10, dtype=np.int64))
        result = backend.execute()
        np.testing.assert_array_equal(result, expected)

    def test_numpy_backend_context_manager(self):
        """setup/warmup/teardown must not raise through context manager."""
        backend = NumpyBackend(fn=lambda: np.zeros(5, dtype=np.float64))
        with backend as b:
            result = b.execute()
        assert result.shape == (5,)

    def test_compare_and_benchmark_with_explicit_numpy_backend(self):
        """Passing explicit NumpyBackend instances must produce a valid result."""
        def fn():
            return np.arange(100, dtype=np.int8)
        result = compare_and_benchmark(
            student_fn=fn,
            baseline_fn=fn,
            config=BenchmarkConfig(n_runs=5, check_performance=False),
            student_backend=NumpyBackend(fn=fn),
            baseline_backend=NumpyBackend(fn=fn),
        )
        assert isinstance(result, BenchmarkResult)
        assert result.passed_performance is True

    def test_execution_backend_is_abstract(self):
        """ExecutionBackend must not be instantiable directly."""
        with pytest.raises(TypeError):
            ExecutionBackend()  # type: ignore[abstract]

    def test_custom_backend_subclass_accepted(self):
        """A minimal concrete backend must satisfy the ABC contract."""
        class MinimalBackend(ExecutionBackend):
            def setup(self): pass
            def warmup(self): pass
            def execute(self): return np.zeros(3, dtype=np.float32)
            def teardown(self): pass

        result = compare_and_benchmark(
            student_fn=lambda: np.zeros(3, dtype=np.float32),
            baseline_fn=lambda: np.zeros(3, dtype=np.float32),
            config=BenchmarkConfig(n_runs=3, check_performance=False),
            student_backend=MinimalBackend(),
            baseline_backend=MinimalBackend(),
        )
        assert result.passed_performance is True


# ---------------------------------------------------------------------------
# Hint registry
# ---------------------------------------------------------------------------


class TestHintRegistry:
    """Verify the hint auto-discovery registry is fully populated with composite keys."""

    @pytest.fixture(scope="class")
    @classmethod
    def registry(cls):
        from hint import HINTS_REGISTRY
        return HINTS_REGISTRY

    def test_all_modules_registered(self, registry):
        expected = {
            "basic/01_array_creation",
            "basic/02_properties_reshaping",
            "basic/03_indexing_slicing",
            "intermediate/01_vectorized_math",
            "intermediate/02_array_manipulation",
            "intermediate/03_boolean_logic",
            "advanced/01_memory_layout",
        }
        missing = expected - set(registry.keys())
        assert not missing, f"Missing modules in registry: {missing}"

    @pytest.mark.parametrize("module_key,class_name,method", [
        ("basic/01_array_creation",        "ArrayCreation",        "create_integer_range"),
        ("basic/01_array_creation",        "ArrayCreation",        "create_squared_range"),
        ("basic/02_properties_reshaping",  "PropertiesReshaping",  "reshape_to_matrix"),
        ("basic/02_properties_reshaping",  "PropertiesReshaping",  "flatten_and_cast"),
        ("basic/03_indexing_slicing",      "IndexingSlicing",      "filter_above_threshold"),
        ("basic/03_indexing_slicing",      "IndexingSlicing",      "gather_by_indices"),
        ("intermediate/01_vectorized_math",    "VectorizedMath",       "row_means"),
        ("intermediate/01_vectorized_math",    "VectorizedMath",       "normalize_columns"),
        ("intermediate/02_array_manipulation", "ArrayManipulation",    "stack_rows"),
        ("intermediate/02_array_manipulation", "ArrayManipulation",    "concatenate_side_by_side"),
        ("intermediate/03_boolean_logic",      "BooleanLogic",         "apply_discount"),
        ("intermediate/03_boolean_logic",      "BooleanLogic",         "identify_close_rows"),
        ("advanced/01_memory_layout",      "MemoryLayout",         "ensure_c_contiguous"),
        ("advanced/01_memory_layout",      "MemoryLayout",         "check_memory_share"),
        ("advanced/01_memory_layout",      "MemoryLayout",         "get_row_stride_bytes"),
    ])
    def test_method_registered_with_hint_text(
        self, registry, module_key, class_name, method
    ):
        assert module_key in registry, f"Module '{module_key}' not in registry"
        assert class_name in registry[module_key], (
            f"Class '{class_name}' missing from registry['{module_key}']"
        )
        assert method in registry[module_key][class_name], (
            f"Method '{method}' missing from registry['{module_key}']['{class_name}']"
        )
        hint = registry[module_key][class_name][method]
        assert hint.get("major", "").strip(), (
            f"Empty 'major' hint for {module_key}::{class_name}::{method}"
        )
        assert hint.get("minor", "").strip(), (
            f"Empty 'minor' hint for {module_key}::{class_name}::{method}"
        )


class TestASTValidator:
    """Verify AST static analysis policy enforcement."""

    def test_for_loop_violation_caught(self):
        def bad_fn():
            res = 0
            for i in range(10):
                res += i
            return res

        @ast_policy(max_for_loops=0, target=bad_fn)
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="for-loops"):
            run_test()

    def test_forbid_calls_violation_caught(self):
        def bad_fn():
            return np.vectorize(lambda x: x + 1)([1, 2, 3])

        @ast_policy(forbid_calls=["np.vectorize"], target=bad_fn)
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="vectorize"):
            run_test()

    def test_require_calls_violation_caught(self):
        def bad_fn():
            return np.zeros(5)

        @ast_policy(require_calls=["np.where"], target=bad_fn)
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="np.where"):
            run_test()

    def test_clean_stub_skips_ast_violation(self):
        def stub_fn():
            """A docstring."""
            raise NotImplementedError("Not implemented yet")

        @ast_policy(require_calls=["np.where"], target=stub_fn)
        def run_test():
            with pytest.raises(NotImplementedError):
                stub_fn()

        run_test()

    def test_custom_feedback_printed(self):
        def bad_fn():
            res = 0
            for i in range(5):
                res += i
            return res

        @ast_policy(
            max_for_loops=0,
            target=bad_fn,
            feedback={"max_for_loops": "Custom Smart AST Feedback: Do not use for loops!"},
        )
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="Custom Smart AST Feedback: Do not use for loops!"):
            run_test()

    def test_custom_feedback_forbid_calls(self):
        def bad_fn():
            return np.vectorize(lambda x: x + 1)([1, 2, 3])

        @ast_policy(
            forbid_calls=["np.vectorize"],
            target=bad_fn,
            feedback={"np.vectorize": "Custom Smart AST Feedback: Do not use vectorize!"},
        )
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="Custom Smart AST Feedback: Do not use vectorize!"):
            run_test()

    def test_custom_feedback_require_calls(self):
        def bad_fn():
            return np.zeros(5)

        @ast_policy(
            require_calls=["np.where"],
            target=bad_fn,
            feedback={"np.where": "Custom Smart AST Feedback: You must use np.where!"},
        )
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="Custom Smart AST Feedback: You must use np.where!"):
            run_test()

    def test_custom_feedback_forbid_imports(self):
        def bad_fn():
            import math
            return math.sqrt(4)

        @ast_policy(
            forbid_imports=["math"],
            target=bad_fn,
            feedback={"math": "Custom Smart AST Feedback: Do not import math!"},
        )
        def run_test():
            return bad_fn()

        with pytest.raises(BaseException, match="Custom Smart AST Feedback: Do not import math!"):
            run_test()


class TestCLIWorkflows:
    """Verify CLI workflow commands status and generate."""

    def test_run_status_output(self, capsys):
        res = run_status()
        assert res == 0
        captured = capsys.readouterr()
        assert "Module Progress Report" in captured.out
        assert "Curriculum Status:" in captured.out

    def test_run_generate_workflow(self, capsys):
        lesson_name = "99_test_lesson"
        module = "arraysmith"
        tier = "basic"
        student_dir = ROOT / module / tier / lesson_name
        test_dir = ROOT / "tests" / module / tier / lesson_name

        try:
            res = run_generate(module, tier, lesson_name)
            assert res == 0
            assert (student_dir / "INSTRUCTION.md").exists()
            assert (student_dir / "student_code.py").exists()
            assert (test_dir / "_baseline.py").exists()
            assert (test_dir / "test_99.py").exists()

            res_again = run_generate(module, tier, lesson_name)
            assert res_again == 1
        finally:
            if student_dir.exists():
                shutil.rmtree(student_dir)
            if test_dir.exists():
                shutil.rmtree(test_dir)
