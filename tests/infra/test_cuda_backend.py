"""
tests/infra/test_cuda_backend.py
================================
Unit tests for :class:`~forge_core.backends.cuda_backend.CudaJitBackend`.

Strategy
--------
All tests use :mod:`unittest.mock` to patch the PyTorch entry-points that
:class:`CudaJitBackend` depends on:

- ``torch.cuda.is_available``        — controls the GPU availability branch.
- ``forge_core.backends.cuda_backend._cpp_load`` — prevents real compilation.
- ``forge_core.backends.cuda_backend.Path``      — bypasses filesystem checks.
- ``torch.cuda.synchronize``         — verifies the mandatory device barrier.
- ``torch.cuda.empty_cache``         — verifies explicit VRAM reclamation.

No GPU hardware, no C++ toolchain, and no PyTorch installation are required.
The suite verifies behavioural contracts (call order, count, argument forwarding)
rather than numerical GPU correctness.

Run with:
    uv run tforge check infra
    pytest tests/infra/test_cuda_backend.py -v
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from forge_core.backends.cuda_backend import CudaJitBackend


@pytest.fixture(autouse=True)
def _clear_module_cache(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Clear the CudaJitBackend process-level module cache before each test.

    Prevents stale mock modules cached by one test from being returned by the
    ``_module_cache`` lookup in a subsequent test that shares the same
    ``(module_name, resolved_source)`` cache key.
    """
    from forge_core.backends.cuda_backend import CudaJitBackend

    CudaJitBackend._module_cache.clear()

    if "Stub" not in request.node.nodeid:
        with patch("forge_core.backends.cuda_backend._TORCH_AVAILABLE", True):
            yield
    else:
        yield


def _mock_path_cls(exists: bool = True) -> MagicMock:
    """Return a :class:`~unittest.mock.MagicMock` that replaces :class:`pathlib.Path`.

    The mock makes ``Path(source_path).exists()`` return *exists* and
    ``Path(source_path).resolve()`` return the path unchanged, so the
    :class:`~forge_core.backends.cuda_backend.CudaJitBackend` filesystem guard
    can be exercised in isolation without touching the real filesystem.

    Args:
        exists: Value returned by the mocked ``Path(...).exists()`` call.

    Returns:
        MagicMock: A callable mock that produces a path-like object.
    """
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = exists
    mock_path_instance.resolve.return_value = mock_path_instance
    mock_path_instance.__str__ = lambda self: getattr(self, "_source_repr", "<mock_path>")  # type: ignore[method-assign,misc,assignment]

    mock_path_cls = MagicMock(side_effect=lambda s: _make_path_instance(s, exists))
    return mock_path_cls


def _make_path_instance(source: str, exists: bool) -> MagicMock:
    """Build a single mocked :class:`pathlib.Path` instance for *source*."""
    instance = MagicMock()
    instance.exists.return_value = exists
    instance.resolve.return_value = instance
    instance.__str__ = lambda self: source  # type: ignore[method-assign,misc,assignment]
    return instance


def _build_backend_with_mocks(
    cuda_available: bool = True,
    kernel_return: Any = None,
) -> tuple[CudaJitBackend, MagicMock, MagicMock]:
    """Construct a :class:`CudaJitBackend` with all external calls mocked.

    Patches ``torch.cuda.is_available``, ``_cpp_load``, and :class:`pathlib.Path`
    so no filesystem access or real compilation occurs.

    Args:
        cuda_available: Value returned by the mocked ``torch.cuda.is_available``.
        kernel_return: Value returned by the mock kernel function.

    Returns:
        Tuple of ``(backend_instance, mock_kernel, mock_synchronize)``.
    """
    from forge_core.backends.cuda_backend import CudaJitBackend

    mock_kernel = MagicMock(return_value=kernel_return)
    mock_module = MagicMock()
    mock_module.my_forward = mock_kernel
    mock_load = MagicMock(return_value=mock_module)
    mock_sync = MagicMock()
    mock_empty_cache = MagicMock()

    with (
        patch(
            "forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=cuda_available
        ),
        patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
        patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
        patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", mock_sync),
        patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", mock_empty_cache),
    ):
        backend = CudaJitBackend(
            source_path="/fake/path/kernel.cu",
            module_name="fake_kernel",
            function_name="my_forward",
        )

    backend._mock_sync = mock_sync  # type: ignore[attr-defined]
    backend._mock_empty_cache = mock_empty_cache  # type: ignore[attr-defined]
    backend._mock_load = mock_load  # type: ignore[attr-defined]
    return backend, mock_kernel, mock_sync


def _make_path_instance_exists(source: str) -> MagicMock:
    """Build a mocked :class:`pathlib.Path` instance that reports the file as existing."""
    return _make_path_instance(source, exists=True)


def _make_path_instance_missing(source: str) -> MagicMock:
    """Build a mocked :class:`pathlib.Path` instance that reports the file as absent."""
    return _make_path_instance(source, exists=False)


class TestCudaJitBackendConstruction:
    """Verify that __init__ correctly delegates to the JIT compiler and extracts the kernel."""

    def test_load_called_with_correct_arguments(self) -> None:
        """cpp_extension.load must receive the module_name and resolved sources list."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_kernel = MagicMock(return_value=None)
        mock_module = MagicMock()
        mock_module.my_forward = mock_kernel
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
        ):
            CudaJitBackend(
                source_path="/fake/kernel.cu",
                module_name="test_mod",
                function_name="my_forward",
            )

        mock_load.assert_called_once_with(
            name="test_mod",
            sources=["/fake/kernel.cu"],
            verbose=False,
        )

    def test_asan_flags_injected_when_debug_env_set(self) -> None:
        """cpp_extension.load must receive AddressSanitizer flags when TFORGE_DEBUG_CPP=1."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_kernel = MagicMock(return_value=None)
        mock_module = MagicMock()
        mock_module.my_forward = mock_kernel
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
            patch.dict("os.environ", {"TFORGE_DEBUG_CPP": "1"}),
        ):
            CudaJitBackend(
                source_path="/fake/kernel.cu",
                module_name="test_asan_mod",
                function_name="my_forward",
            )

        mock_load.assert_called_once_with(
            name="test_asan_mod",
            sources=["/fake/kernel.cu"],
            extra_cflags=["-fsanitize=address", "-fno-omit-frame-pointer", "-g"],
            extra_ldflags=["-fsanitize=address"],
            verbose=False,
        )

    def test_kernel_attribute_bound_to_module_function(self) -> None:
        """self.kernel must reference the function extracted from the compiled module."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock()
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
        ):
            backend = CudaJitBackend(
                source_path="/fake/kernel.cu",
                module_name="test_mod",
                function_name="forward_pass",
            )

        assert backend.kernel is mock_module.forward_pass

    def test_module_attribute_stores_compiled_extension(self) -> None:
        """self.module must hold the return value of cpp_extension.load."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock()
        mock_module.fn = MagicMock()
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
        ):
            backend = CudaJitBackend("/src.cu", "mod", "fn")

        assert backend.module is mock_module


class TestCudaJitBackendNoCudaWarning:
    """Verify the graceful degradation path when no CUDA GPU is detected."""

    def test_user_warning_emitted_when_cuda_unavailable(self) -> None:
        """A UserWarning — not an exception — must be emitted when CUDA is absent."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock()
        mock_module.fn = MagicMock()
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=False),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            CudaJitBackend("/src.cu", "mod", "fn")

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1, (
            f"Expected exactly one UserWarning; got {len(user_warnings)}: {caught}"
        )

    def test_warning_message_mentions_cuda_unavailability(self) -> None:
        """The emitted warning must reference is_available() returning False."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock()
        mock_module.fn = MagicMock()
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=False),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            CudaJitBackend("/src.cu", "mod", "fn")

        assert caught, "No warnings were emitted"
        message = str(caught[0].message)
        assert "is_available" in message or "CUDA" in message, (
            f"Warning message did not reference CUDA availability: {message!r}"
        )

    def test_no_warning_when_cuda_available(self) -> None:
        """No UserWarning must be emitted when CUDA is available."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock()
        mock_module.fn = MagicMock()
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            CudaJitBackend("/src.cu", "mod", "fn")

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"Unexpected UserWarning when CUDA is available: {user_warnings}"
        )


class TestCudaJitBackendLifecycle:
    """Verify that setup/warmup/execute/teardown obey the HPC contract."""

    def test_setup_transfers_tensors_to_cuda(self) -> None:
        """setup() must call .cuda() on each supplied tensor argument."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)

        mock_tensor_a = MagicMock()
        mock_tensor_b = MagicMock()
        mock_tensor_a.cuda.return_value = mock_tensor_a
        mock_tensor_b.cuda.return_value = mock_tensor_b

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", MagicMock()),
        ):
            backend.setup(mock_tensor_a, mock_tensor_b)

        mock_tensor_a.cuda.assert_called_once()
        mock_tensor_b.cuda.assert_called_once()
        assert len(backend._device_args) == 2

    def test_warmup_calls_kernel_once_with_device_args(self) -> None:
        """warmup() must invoke self.kernel exactly once with self._device_args."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)

        sentinel_a = object()
        sentinel_b = object()
        backend._device_args = (sentinel_a, sentinel_b)

        backend.warmup()

        mock_kernel.assert_called_once_with(sentinel_a, sentinel_b)

    def test_execute_calls_kernel_then_synchronize(self) -> None:
        """execute() must call kernel before synchronize — order is mandatory."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("arg_0",)

        call_order: list[str] = []
        mock_kernel.side_effect = lambda *_, **__: call_order.append("kernel")
        mock_sync = MagicMock(side_effect=lambda: call_order.append("sync"))

        with patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", mock_sync):
            backend.execute()

        assert call_order == ["kernel", "sync"], f"Expected ['kernel', 'sync'], got {call_order}"

    def test_execute_returns_kernel_result(self) -> None:
        """execute() must return the value produced by self.kernel."""
        expected = MagicMock(name="gpu_tensor")
        backend, mock_kernel, _ = _build_backend_with_mocks(
            cuda_available=True, kernel_return=expected
        )
        backend._device_args = ("x",)

        with patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()):
            result = backend.execute()

        assert result is expected

    def test_teardown_clears_device_args_and_calls_empty_cache(self) -> None:
        """teardown() must empty _device_args and call torch.cuda.empty_cache()."""
        backend, _, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = (MagicMock(), MagicMock())

        mock_empty_cache = MagicMock()
        with patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", mock_empty_cache):
            backend.teardown()

        assert len(backend._device_args) == 0  # type: ignore[arg-type]
        mock_empty_cache.assert_called_once()

    def test_teardown_is_idempotent(self) -> None:
        """teardown() must not raise when called on an already-torn-down backend."""
        backend, _, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ()

        mock_empty_cache = MagicMock()
        with patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", mock_empty_cache):
            backend.teardown()
            backend.teardown()

        assert mock_empty_cache.call_count == 2

    def test_full_lifecycle_call_order(self) -> None:
        """setup → warmup → execute → teardown must produce the exact expected call sequence."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        call_log: list[str] = []

        mock_kernel = MagicMock(side_effect=lambda *_, **__: call_log.append("kernel"))
        mock_module = MagicMock()
        mock_module.fn = mock_kernel
        mock_load = MagicMock(return_value=mock_module)
        mock_sync = MagicMock(side_effect=lambda: call_log.append("sync"))
        mock_empty_cache = MagicMock(side_effect=lambda: call_log.append("empty_cache"))

        sentinel = MagicMock()
        sentinel.cuda = MagicMock(return_value=sentinel)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", mock_sync),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", mock_empty_cache),
        ):
            backend = CudaJitBackend("/src.cu", "mod", "fn")
            backend.setup(sentinel)
            backend.warmup()
            backend.execute()
            backend.teardown()

        assert call_log == ["kernel", "kernel", "sync", "empty_cache"], (
            f"Unexpected lifecycle call order: {call_log}"
        )

    def test_warmup_kernel_call_count_is_exactly_one(self) -> None:
        """warmup() must invoke the kernel exactly once — not zero or more."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        backend.warmup()

        assert mock_kernel.call_count == 1

    def test_execute_kernel_call_count_is_exactly_one(self) -> None:
        """execute() must invoke the kernel exactly once per call."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        with patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()):
            backend.execute()

        assert mock_kernel.call_count == 1

    def test_execute_synchronize_call_count_is_exactly_one(self) -> None:
        """execute() must call torch.cuda.synchronize() exactly once per call."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        mock_sync = MagicMock()
        with patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", mock_sync):
            backend.execute()

        mock_sync.assert_called_once()


class TestCudaJitBackendContextManager:
    """Verify that the ExecutionBackend context manager protocol works correctly."""

    def test_context_manager_calls_setup_and_warmup_on_enter(self) -> None:
        """__enter__ must invoke setup() and warmup() exactly once each."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        original_setup = backend.setup
        original_warmup = backend.warmup

        setup_calls: list[int] = []
        warmup_calls: list[int] = []

        def spy_setup(*args: Any, **kwargs: Any) -> None:
            setup_calls.append(1)
            return original_setup(*args, **kwargs)

        def spy_warmup() -> None:
            warmup_calls.append(1)
            return original_warmup()

        backend.setup = spy_setup  # type: ignore[method-assign]
        backend.warmup = spy_warmup  # type: ignore[method-assign]

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", MagicMock()),
            backend,
        ):
            pass

        assert len(setup_calls) == 1, f"setup called {len(setup_calls)} times, expected 1"
        assert len(warmup_calls) == 1, f"warmup called {len(warmup_calls)} times, expected 1"

    def test_context_manager_calls_teardown_on_exit(self) -> None:
        """__exit__ must call teardown() even when no exception is raised."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        teardown_calls: list[int] = []
        original_teardown = backend.teardown

        def spy_teardown() -> None:
            teardown_calls.append(1)
            return original_teardown()

        backend.teardown = spy_teardown  # type: ignore[method-assign]

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", MagicMock()),
            backend,
        ):
            pass

        assert len(teardown_calls) == 1, f"teardown called {len(teardown_calls)} times, expected 1"

    def test_context_manager_teardown_called_on_exception(self) -> None:
        """__exit__ must call teardown() even when the body raises an exception."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        teardown_calls: list[int] = []
        original_teardown = backend.teardown

        def spy_teardown() -> None:
            teardown_calls.append(1)
            return original_teardown()

        backend.teardown = spy_teardown  # type: ignore[method-assign]

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="deliberate test error"):
                with backend:
                    raise RuntimeError("deliberate test error")

        assert len(teardown_calls) == 1, (
            f"teardown was not called on exception; call count: {len(teardown_calls)}"
        )

    def test_context_manager_does_not_suppress_exceptions(self) -> None:
        """__exit__ must return False — exceptions must propagate to the caller."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)
        backend._device_args = ("x",)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()),
            patch("forge_core.backends.cuda_backend.torch.cuda.empty_cache", MagicMock()),
        ):
            with pytest.raises(ValueError, match="must propagate"):
                with backend:
                    raise ValueError("must propagate")


class TestCudaJitBackendStub:
    """Verify the RuntimeError guard when PyTorch is not installed."""

    def test_stub_raises_runtime_error_on_instantiation(self) -> None:
        """Instantiation must raise RuntimeError when torch is absent."""
        try:
            import torch  # noqa: F401

            from forge_core.backends.cuda_backend import CudaJitBackend as _Real

            with patch("forge_core.backends.cuda_backend._TORCH_AVAILABLE", False):
                with pytest.raises(RuntimeError, match="uv sync --extra torch"):
                    _Real.__new__(_Real).__init__("/src.cu", "mod", "fn")  # type: ignore[misc,call-arg]
        except ImportError:
            from forge_core.backends import CudaJitBackend

            with pytest.raises(RuntimeError, match="uv sync --extra torch"):
                CudaJitBackend("/src.cu", "mod", "fn")  # type: ignore[call-arg]

    def test_stub_error_message_contains_uv_command(self) -> None:
        """The RuntimeError message must contain the actionable uv install command."""
        try:
            import torch  # noqa: F401

            from forge_core.backends.cuda_backend import CudaJitBackend as _Real

            with patch("forge_core.backends.cuda_backend._TORCH_AVAILABLE", False):
                with pytest.raises(RuntimeError) as exc_info:
                    _Real.__new__(_Real).__init__("/src.cu", "mod", "fn")  # type: ignore[misc,call-arg]
                assert "uv sync --extra torch" in str(exc_info.value)
        except ImportError:
            from forge_core.backends import CudaJitBackend

            with pytest.raises(RuntimeError) as exc_info:
                CudaJitBackend("/src.cu", "mod", "fn")  # type: ignore[call-arg]
            assert "uv sync --extra torch" in str(exc_info.value)


class TestCudaJitBackendIsConcreteBackend:
    """Verify CudaJitBackend satisfies the ExecutionBackend ABC contract."""

    def test_cuda_jit_backend_is_subclass_of_execution_backend(self) -> None:
        """CudaJitBackend must be a subclass of ExecutionBackend."""
        from forge_core.backends.base import ExecutionBackend
        from forge_core.backends.cuda_backend import CudaJitBackend

        assert issubclass(CudaJitBackend, ExecutionBackend)

    def test_cuda_jit_backend_exported_from_package(self) -> None:
        """CudaJitBackend must be importable from the top-level backends package."""
        from forge_core.backends import CudaJitBackend  # noqa: F401

        assert CudaJitBackend is not None

    def test_cuda_jit_backend_in_dunder_all(self) -> None:
        """CudaJitBackend must appear in forge_core.backends.__all__."""
        import forge_core.backends as _backends

        assert "CudaJitBackend" in _backends.__all__


class TestCudaJitBackendSafetyAndCaching:
    """Verify safety checks, kwargs support, and caching behavior."""

    def test_file_not_found_error_raised_when_source_missing(self, tmp_path: Any) -> None:
        """Instantiating with a non-existent source file must raise FileNotFoundError."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        missing_path = str(tmp_path / "non_existent_kernel.cu")
        with patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True):
            with pytest.raises(FileNotFoundError, match="Source file not found"):
                CudaJitBackend(missing_path, "missing_mod", "fn")

    def test_descriptive_attribute_error_on_missing_symbol(self) -> None:
        """Missing kernel symbol must raise AttributeError listing available symbols."""
        from forge_core.backends.cuda_backend import CudaJitBackend

        mock_module = MagicMock(spec_set=["available_fn"])
        mock_load = MagicMock(return_value=mock_module)

        with (
            patch("forge_core.backends.cuda_backend.torch.cuda.is_available", return_value=True),
            patch("forge_core.backends.cuda_backend._cpp_load", mock_load),
            patch("forge_core.backends.cuda_backend.Path", side_effect=_make_path_instance_exists),
        ):
            with pytest.raises(AttributeError, match="Available public symbols"):
                CudaJitBackend("/fake/kernel.cu", "test_mod", "missing_fn")

    def test_setup_forwards_kwargs_to_kernel(self) -> None:
        """Keyword arguments passed to setup() must be forwarded to kernel execution."""
        backend, mock_kernel, _ = _build_backend_with_mocks(cuda_available=True)

        sentinel_kwarg = MagicMock()
        sentinel_kwarg.cuda.return_value = sentinel_kwarg

        with patch("forge_core.backends.cuda_backend.torch.cuda.synchronize", MagicMock()):
            backend.setup(alpha=sentinel_kwarg)
            backend.warmup()
            backend.execute()

        sentinel_kwarg.cuda.assert_called_once()
        mock_kernel.assert_called_with(alpha=sentinel_kwarg)
