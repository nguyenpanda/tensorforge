"""Unit tests for the TensorForgeBootstrapper OOP architecture and legacy wrapper functions."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from bootstrap import (
    TensorForgeBootstrapper,
    phase3_module_selection,
    phase4_dynamic_url_resolution,
    wipe_venv,
)


class TestTensorForgeBootstrapperInit:
    """Verify initialization and attribute assignment of TensorForgeBootstrapper."""

    def test_default_init_values(self) -> None:
        """Default instantiation must configure non-interactive mode false and all extras."""
        bootstrapper = TensorForgeBootstrapper()
        assert bootstrapper.target_path is None
        assert not bootstrapper.reconfigure
        assert not bootstrapper.non_interactive
        assert bootstrapper.choice == 4
        assert bootstrapper.extras == ["torch", "hpc", "dev"]

    def test_custom_init_values(self) -> None:
        """Custom instantiation must preserve argument overrides."""
        bootstrapper = TensorForgeBootstrapper(
            target_path="core", reconfigure=True, non_interactive=True
        )
        assert bootstrapper.target_path == "core"
        assert bootstrapper.reconfigure
        assert bootstrapper.non_interactive


class TestTensorForgeBootstrapperOSCheck:
    """Verify OS and Python version validation rules."""

    def test_windows_os_raises_system_exit(self) -> None:
        """Windows operating systems must be rejected with SystemExit."""
        bootstrapper = TensorForgeBootstrapper()
        with (
            patch("platform.system", return_value="Windows"),
            pytest.raises(SystemExit) as exc_info,
        ):
            bootstrapper.check_os_sanity()
        assert exc_info.value.code == 1

    def test_old_python_version_raises_system_exit(self) -> None:
        """Python runtime versions below 3.12 must raise SystemExit."""
        bootstrapper = TensorForgeBootstrapper()
        mock_ver = MagicMock(major=3, minor=11, micro=0)
        mock_ver.__lt__.side_effect = lambda other: other > (3, 11)
        with (
            patch("platform.system", return_value="Linux"),
            patch.object(sys, "version_info", mock_ver),
            pytest.raises(SystemExit) as exc_info,
        ):
            bootstrapper.check_os_sanity()
        assert exc_info.value.code == 1

    def test_valid_os_and_python_passes(self) -> None:
        """Supported OS and Python >= 3.12 must complete without raising."""
        bootstrapper = TensorForgeBootstrapper()
        mock_ver = MagicMock(major=3, minor=12, micro=1)
        mock_ver.__lt__.side_effect = lambda other: other > (3, 12)
        with (
            patch("platform.system", return_value="Darwin"),
            patch.object(sys, "version_info", mock_ver),
        ):
            bootstrapper.check_os_sanity()


class TestTensorForgeBootstrapperModuleSelection:
    """Verify learning path selection and extra dependency mapping."""

    @pytest.mark.parametrize(
        ("path_arg", "expected_choice", "expected_extras"),
        [
            ("core", 1, ["dev"]),
            ("dev", 1, ["dev"]),
            ("torch", 2, ["torch", "dev"]),
            ("hpc", 4, ["torch", "hpc", "dev"]),
            ("all", 4, ["torch", "hpc", "dev"]),
        ],
    )
    def test_target_path_mapping(
        self, path_arg: str, expected_choice: int, expected_extras: list[str]
    ) -> None:
        """Target path string arguments must map precisely to expected menu choice and extras."""
        bootstrapper = TensorForgeBootstrapper(target_path=path_arg, non_interactive=True)
        choice, extras, display = bootstrapper.select_modules()
        assert choice == expected_choice
        assert extras == expected_extras
        assert bootstrapper.choice == expected_choice
        assert bootstrapper.extras == expected_extras


class TestTensorForgeBootstrapperURLResolution:
    """Verify platform-aware dynamic PyTorch index URL generation."""

    def test_core_path_returns_empty_url(self) -> None:
        """Core learning path must return an empty index argument string."""
        bootstrapper = TensorForgeBootstrapper(target_path="core", non_interactive=True)
        bootstrapper.select_modules()
        url_arg = bootstrapper.resolve_dynamic_urls()
        assert url_arg == ""

    def test_darwin_arm64_returns_empty_url(self) -> None:
        """Apple Silicon macOS must use default PyPI index without extra URL arguments."""
        bootstrapper = TensorForgeBootstrapper(target_path="torch", non_interactive=True)
        bootstrapper.select_modules()
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            url_arg = bootstrapper.resolve_dynamic_urls()
        assert url_arg == ""


class TestBackwardCompatibleWrappers:
    """Verify legacy module-level wrapper functions delegate cleanly to bootstrapper instances."""

    def test_phase3_wrapper_delegation(self) -> None:
        """phase3_module_selection wrapper must return correct tuple."""
        choice, extras, display = phase3_module_selection("core", non_interactive=True)
        assert choice == 1
        assert extras == ["dev"]
        assert display == "dev"

    def test_phase4_wrapper_delegation(self) -> None:
        """phase4_dynamic_url_resolution wrapper must return string without raising."""
        res = phase4_dynamic_url_resolution(1, non_interactive=True)
        assert res == ""

    def test_wipe_venv_wrapper_delegation(self) -> None:
        """wipe_venv wrapper must invoke bootstrapper method without error when venv missing."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.path.islink", return_value=False),
        ):
            wipe_venv()
