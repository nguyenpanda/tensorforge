"""
tests/infra/test_dataset_manager.py
=====================================
Unit tests for :class:`~forge_core.dataset_manager.DatasetManager`.
Verifies atomic downloads, checksum validation, and cache management.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge_core.dataset_manager import DatasetManager


@pytest.fixture
def temp_cache(tmp_path: Path) -> Path:
    """Provide a temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    return cache_dir


class TestDatasetManager:
    """Test suite for DatasetManager operations."""

    def test_init_default_cache(self) -> None:
        dm = DatasetManager()
        assert dm.cache_dir == Path.home() / ".cache" / "tensorforge" / "datasets"

    def test_init_custom_cache(self, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        assert dm.cache_dir == temp_cache.resolve()

    @patch("requests.get")
    def test_get_dataset_download_atomic(self, mock_get: MagicMock, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        content = b"fake dataset content"
        expected_sha = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(content))}
        mock_resp.iter_content.return_value = [content]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("os.replace", wraps=os.replace) as mock_replace:
            path = dm.get_dataset(
                "https://example.com/data.bin",
                expected_sha256=expected_sha,
            )

            assert path.exists()
            assert path.read_bytes() == content
            assert mock_replace.called

    @patch("requests.get")
    def test_get_dataset_cached(self, mock_get: MagicMock, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        content = b"existing data"
        expected_sha = hashlib.sha256(content).hexdigest()

        temp_cache.mkdir(parents=True, exist_ok=True)
        file_path = temp_cache / "data.bin"
        file_path.write_bytes(content)

        path = dm.get_dataset(
            "https://example.com/data.bin",
            expected_sha256=expected_sha,
        )

        assert path == file_path
        assert not mock_get.called

    @patch("requests.get")
    def test_get_dataset_checksum_failure(self, mock_get: MagicMock, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        content = b"corrupted data"

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(content))}
        mock_resp.iter_content.return_value = [content]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="SHA-256 checksum verification failed"):
            dm.get_dataset(
                "https://example.com/data.bin",
                expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            )

        # Ensure no .tmp files remain
        tmp_files = list(temp_cache.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_clear_cache_single_file(self, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        temp_cache.mkdir(parents=True, exist_ok=True)
        f1 = temp_cache / "f1.bin"
        f2 = temp_cache / "f2.bin"
        f1.write_bytes(b"1")
        f2.write_bytes(b"2")

        dm.clear_cache("f1.bin")
        assert not f1.exists()
        assert f2.exists()

    def test_clear_cache_all(self, temp_cache: Path) -> None:
        dm = DatasetManager(temp_cache)
        temp_cache.mkdir(parents=True, exist_ok=True)
        (temp_cache / "f1.bin").write_bytes(b"1")

        dm.clear_cache()
        assert not temp_cache.exists()
