"""Unit tests for ImageNormalizer image preprocessing."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiolens.config import Settings
from radiolens.core.preprocessor import ImageNormalizer

# ----------------------------------------------------------------- from_pil


class TestFromPil:
    """Tests for ImageNormalizer.from_pil."""

    def test_output_shape_matches_settings(
        self,
        settings: Settings,
        synthetic_rgb_image: Image.Image,
    ) -> None:
        """Output array shape is (image_height, image_width, 3)."""
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(synthetic_rgb_image)
        h = settings.image_height
        w = settings.image_width
        assert result.shape == (h, w, 3)

    def test_output_dtype_is_float32(
        self,
        settings: Settings,
        synthetic_rgb_image: Image.Image,
    ) -> None:
        """Output array dtype is float32."""
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(synthetic_rgb_image)
        assert result.dtype == np.float32

    def test_values_between_zero_and_one(
        self,
        settings: Settings,
        synthetic_rgb_image: Image.Image,
    ) -> None:
        """All pixel values fall within [0.0, 1.0] after normalisation."""
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(synthetic_rgb_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_grayscale_image_converted_to_rgb(
        self,
        settings: Settings,
    ) -> None:
        """A grayscale (L mode) image is upcast to 3 channels."""
        gray_arr = np.full((64, 64), 128, dtype=np.uint8)
        gray_img = Image.fromarray(gray_arr, mode="L")
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(gray_img)
        assert result.shape[-1] == 3

    def test_rgba_image_converted_to_rgb(
        self,
        settings: Settings,
    ) -> None:
        """An RGBA image (4 channels) is stripped to 3 channels."""
        rng = np.random.default_rng(1)
        rgba_arr = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(rgba_img)
        assert result.shape[-1] == 3

    def test_non_square_image_resized_correctly(
        self,
        settings: Settings,
    ) -> None:
        """A 100x200 input is resized to (image_height, image_width, 3)."""
        wide_arr = np.zeros((100, 200, 3), dtype=np.uint8)
        wide_img = Image.fromarray(wide_arr, mode="RGB")
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_pil(wide_img)
        h = settings.image_height
        w = settings.image_width
        assert result.shape == (h, w, 3)


# --------------------------------------------------------------- from_bytes


class TestFromBytes:
    """Tests for ImageNormalizer.from_bytes."""

    def test_valid_jpeg_bytes_returns_array(
        self,
        settings: Settings,
        jpeg_bytes: bytes,
    ) -> None:
        """Valid JPEG bytes decode to a float32 (H, W, 3) array."""
        normalizer = ImageNormalizer(settings)
        result = normalizer.from_bytes(jpeg_bytes)
        h = settings.image_height
        w = settings.image_width
        assert result.shape == (h, w, 3)
        assert result.dtype == np.float32

    def test_invalid_bytes_raise_value_error(
        self,
        settings: Settings,
    ) -> None:
        """Bytes that are not a valid image raise ValueError."""
        normalizer = ImageNormalizer(settings)
        with pytest.raises(ValueError, match="Cannot decode"):
            normalizer.from_bytes(b"notanimage")

    def test_empty_bytes_raise_value_error(
        self,
        settings: Settings,
    ) -> None:
        """Empty bytes raise ValueError."""
        normalizer = ImageNormalizer(settings)
        with pytest.raises(ValueError):
            normalizer.from_bytes(b"")

    def test_valid_png_bytes_return_correct_dtype(
        self,
        settings: Settings,
    ) -> None:
        """PNG bytes also decode to float32 output."""
        arr = np.full((32, 32, 3), 100, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        normalizer = ImageNormalizer(settings)
        result = normalizer.from_bytes(png_bytes)
        assert result.dtype == np.float32


# --------------------------------------------------------------- from_path


class TestFromPath:
    """Tests for ImageNormalizer.from_path."""

    def test_valid_image_path_returns_array(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """A valid PNG on disk loads to the expected shape and dtype."""
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "test.png"
        Image.fromarray(arr, mode="RGB").save(img_path)

        normalizer = ImageNormalizer(settings)
        result = normalizer.from_path(img_path)
        h = settings.image_height
        w = settings.image_width
        assert result.shape == (h, w, 3)
        assert result.dtype == np.float32

    def test_missing_file_raises_file_not_found_error(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """A path pointing to a non-existent file raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist.jpg"
        normalizer = ImageNormalizer(settings)
        with pytest.raises(FileNotFoundError):
            normalizer.from_path(missing)

    def test_corrupted_file_raises_value_error(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """A file with garbage bytes raises ValueError."""
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"\x00\xff\xfe garbage not an image")
        normalizer = ImageNormalizer(settings)
        with pytest.raises(ValueError, match="Cannot decode"):
            normalizer.from_path(corrupt)

    def test_values_in_zero_one_range_from_path(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Pixel values are normalised to [0, 1] when loading from disk."""
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        img_path = tmp_path / "white.png"
        Image.fromarray(arr, mode="RGB").save(img_path)

        normalizer = ImageNormalizer(settings)
        result = normalizer.from_path(img_path)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


# -------------------------------------------------------------- target_size


class TestTargetSize:
    """Tests for ImageNormalizer.target_size property."""

    def test_target_size_matches_settings(
        self,
        settings: Settings,
    ) -> None:
        """target_size returns (image_height, image_width) from settings."""
        normalizer = ImageNormalizer(settings)
        expected = (settings.image_height, settings.image_width)
        assert normalizer.target_size == expected

    def test_target_size_is_tuple_of_ints(
        self,
        settings: Settings,
    ) -> None:
        """target_size contains Python int values."""
        normalizer = ImageNormalizer(settings)
        h, w = normalizer.target_size
        assert isinstance(h, int)
        assert isinstance(w, int)
