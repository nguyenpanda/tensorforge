"""Unit tests for the decoupled CLIController workflows and routing utilities in forge_core."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forge_core.cli_controllers import (
    CLIController,
    _ensure_torch,
    _resolve_target,
    _to_pascal_case,
)


class TestResolveTarget:
    """Verify target path resolution across curriculum tiers and lessons."""

    def test_resolve_all_curricula(self) -> None:
        """Target 'all' must map directly to tests/ directory."""
        args = _resolve_target("all", None, None)
        assert args == ["tests/", "-v", "-s"]

    def test_resolve_infra_curriculum(self) -> None:
        """Target 'infra' must map directly to tests/infra/ directory."""
        args = _resolve_target("infra", None, None)
        assert args == ["tests/infra/", "-v", "-s"]

    def test_resolve_specific_lesson_with_method(self) -> None:
        """Targeting a specific lesson and method must append -k flag and relative lesson path."""
        args = _resolve_target("arraysmith", "basic", "01", method="create_integer_range")
        assert "tests/curriculum/arraysmith/basic/01_array_creation" in args[0]
        assert "-k" in args
        assert "create_integer_range" in args

    def test_resolve_nonexistent_curriculum_exits(self) -> None:
        """Requesting a nonexistent curriculum directory must terminate with exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _resolve_target("fake_curriculum", None, None)
        assert exc_info.value.code == 1


class TestEnsureTorch:
    """Verify missing torch dependency guards."""

    def test_missing_torch_exits(self) -> None:
        """When torch is uninstalled, command invocation must exit with status code 1."""
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(SystemExit) as exc_info,
        ):
            _ensure_torch()
        assert exc_info.value.code == 1

    def test_installed_torch_passes(self) -> None:
        """When torch spec is found, verification must pass silently without raising."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            _ensure_torch()


class TestToPascalCase:
    """Verify directory name conversion to PascalCase class names."""

    @pytest.mark.parametrize(
        ("dir_name", "expected_class_name"),
        [
            ("01_array_creation", "ArrayCreation"),
            ("02_properties_reshaping", "PropertiesReshaping"),
            ("03_indexing_slicing", "IndexingSlicing"),
            ("vectorized_math", "VectorizedMath"),
        ],
    )
    def test_pascal_case_conversion(self, dir_name: str, expected_class_name: str) -> None:
        """Numbered prefix must be stripped and underscore words capitalized."""
        assert _to_pascal_case(dir_name) == expected_class_name


class TestCLIControllerWorkflows:
    """Verify CLIController static methods delegate properly without errors."""

    def test_controller_has_all_workflow_methods(self) -> None:
        """CLIController must expose check, validate, lint, status, generate, and autocomplete."""
        assert hasattr(CLIController, "run_check")
        assert hasattr(CLIController, "run_validate")
        assert hasattr(CLIController, "run_lint")
        assert hasattr(CLIController, "run_status")
        assert hasattr(CLIController, "run_generate")
        assert hasattr(CLIController, "run_autocomplete")
