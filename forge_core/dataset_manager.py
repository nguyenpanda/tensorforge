"""
dataset_manager.py — Centralized Dataset Manager for TensorForge
================================================================
Provides automated, atomic downloads and checksum verification for large
datasets required across LeetCode-style curriculum exercises.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Final

import requests
from nguyenpanda.swan import c24, reset
from tqdm import tqdm

_ERR: Final = c24["ff3333"]
_WARN: Final = c24["ff8800"]
_CYAN: Final = c24["00d7ff"]
_INFO: Final = c24["a8b1c2"]


class DatasetManager:
    """Manages downloading, verifying, and caching datasets for TensorForge.

    Attributes:
        cache_dir: The absolute Path where datasets are stored locally.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the DatasetManager.

        Args:
            cache_dir: Optional custom directory for caching downloaded files.
                Defaults to ~/.cache/tensorforge/datasets/.
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "tensorforge" / "datasets"
        else:
            self.cache_dir = Path(cache_dir).resolve()

    def get_dataset(
        self,
        url: str,
        filename: str | None = None,
        expected_sha256: str | None = None,
        force_download: bool = False,
    ) -> Path:
        """Retrieve a dataset from cache or download it atomically if missing.

        Downloads are written to a temporary UUID-tagged `.tmp` file and atomically
        renamed to the target filename upon completion and SHA-256 validation. This
        prevents corrupted partial downloads if interrupted.

        Args:
            url: The remote URL to download the dataset from.
            filename: Optional target filename. If None, derived from URL.
            expected_sha256: Optional hex string of the expected SHA-256 hash.
            force_download: If True, bypasses existing cache and re-downloads.

        Returns:
            Path: Absolute path to the cached dataset file.

        Raises:
            ValueError: If SHA-256 verification fails or URL is invalid.
            requests.RequestException: On network errors during download.
        """
        if not filename:
            derived = url.split("/")[-1].split("?")[0]
            if not derived or derived in (".", ".."):
                raise ValueError(f"Could not derive valid filename from URL: {url}")
            filename = derived

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.cache_dir / filename

        if final_path.exists() and not force_download:
            if expected_sha256 is None:
                return final_path
            if self.verify_checksum(final_path, expected_sha256):
                return final_path
            print(
                f"{_WARN}Checksum mismatch for existing file {filename}. Re-downloading...{reset}",
                file=sys.stderr,
            )

        tmp_filename = f"{filename}.tmp.{uuid.uuid4().hex}"
        tmp_path = self.cache_dir / tmp_filename

        print(f"{_CYAN}Downloading {filename}...{reset}", file=sys.stderr)
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            bytes_written = 0
            with (
                open(tmp_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    disable=None,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        pbar.update(len(chunk))

            if total_size > 0 and bytes_written != total_size:
                tmp_path.unlink(missing_ok=True)
                raise requests.RequestException(
                    f"Incomplete download for {filename}: expected {total_size} bytes, got {bytes_written} bytes."
                )

            if expected_sha256 is not None and not self.verify_checksum(tmp_path, expected_sha256):
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 checksum verification failed for {filename} downloaded from {url}."
                )

            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return final_path

    def verify_checksum(self, filepath: str | Path, expected_sha256: str) -> bool:
        """Verify the SHA-256 checksum of a local file.

        Args:
            filepath: Path to the local file to inspect.
            expected_sha256: The expected hexadecimal SHA-256 hash string.

        Returns:
            bool: True if the file's SHA-256 hash matches the expected hash.
        """
        path = Path(filepath)
        if not path.is_file():
            return False

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        return hasher.hexdigest().lower() == expected_sha256.lower()

    def clear_cache(self, filename: str | None = None) -> None:
        """Remove cached dataset files.

        Args:
            filename: If specified, deletes only that file from cache.
                If None, deletes the entire cache directory.
        """
        if filename is not None:
            target = self.cache_dir / filename
            if target.exists():
                target.unlink()
        elif self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
