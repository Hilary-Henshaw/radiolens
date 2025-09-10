"""Shared pytest fixtures for the radiolens test suite."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiolens.config import Settings

# NOTE: radiolens.core.detector imports Keras at module level.
# Do NOT import from it here — tests that need it must do so
# locally or use pytest.importorskip.


@pytest.fixture
def settings() -> Settings:
    """Return a lightweight Settings instance for unit tests.

    Uses 64x64 images and reduced hyperparams to keep tests fast.
    """
    return Settings(
        image_height=64,
        image_width=64,
        image_channels=3,
        batch_size=2,
        max_epochs=1,
        early_stop_patience=1,
        lr_reduce_patience=1,
        random_seed=0,
        bootstrap_resamples=20,
        bootstrap_ci_level=0.95,
        train_fraction=0.7,
        validation_fraction=0.2,
        test_fraction=0.1,
    )


@pytest.fixture
def synthetic_rgb_image() -> Image.Image:
    """Return a small synthetic RGB PIL Image (64x64)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def synthetic_pixel_array(settings: Settings) -> np.ndarray:
    """Return a float32 (H, W, 3) array with values in [0, 1]."""
    h = settings.image_height
    w = settings.image_width
    c = settings.image_channels
    rng = np.random.default_rng(0)
    return rng.random((h, w, c)).astype(np.float32)


@pytest.fixture
def jpeg_bytes(synthetic_rgb_image: Image.Image) -> bytes:
    """Return JPEG-encoded bytes of the synthetic image."""
    buf = io.BytesIO()
    synthetic_rgb_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def binary_labels_and_scores() -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_proba) arrays for metrics testing.

    y_true is 100 balanced binary labels.
    y_proba is a noisy version of y_true (realistic predictions).
    """
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.15, size=n)
    y_proba = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)
    return y_true.astype(int), y_proba.astype(np.float64)


@pytest.fixture
def balanced_dataset_dir(
    tmp_path: Path,
    settings: Settings,
) -> Path:
    """Create a minimal balanced image dataset in a temp directory.

    Structure::

        <tmp>/
            normal/
                img_000.png ... img_009.png   (10 images)
            pneumonia/
                img_000.png ... img_009.png   (10 images)
    """
    for class_name in ("normal", "pneumonia"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(10):
            arr = np.full(
                (64, 64, 3),
                fill_value=i * 10,
                dtype=np.uint8,
            )
            img = Image.fromarray(arr, mode="RGB")
            img.save(class_dir / f"img_{i:03d}.png")
    return tmp_path


@pytest.fixture
def imbalanced_dataset_dir(tmp_path: Path) -> Path:
    """Create an imbalanced dataset (30 normal, 10 pneumonia)."""
    (tmp_path / "normal").mkdir()
    (tmp_path / "pneumonia").mkdir()

    for i in range(30):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / "normal" / f"n_{i}.png")

    for i in range(10):
        arr = np.full((64, 64, 3), 200, dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / "pneumonia" / f"p_{i}.png")

    return tmp_path
