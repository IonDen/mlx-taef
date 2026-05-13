"""Shared pytest fixtures for mlx-taef tests."""

from pathlib import Path

import pytest

CONVERTED_DIR = Path(__file__).parent / "converted"
REF_DIR = Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def converted_dir() -> Path:
    """Path to pre-converted MLX safetensors files (committed in repo)."""
    return CONVERTED_DIR


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    """Path to reference fixtures (PyTorch-decoded outputs, committed in repo)."""
    return REF_DIR
