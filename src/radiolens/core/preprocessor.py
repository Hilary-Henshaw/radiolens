"""Image normalization and resizing for the radiolens inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
from PIL import Image, UnidentifiedImageError

from radiolens.config import Settings

log = structlog.get_logger(__name__)


class ImageNormalizer:
    """Converts raw images to float32 arrays suitable for ThoraxClassifier.

    All outputs are float32 NumPy arrays of shape ``(H, W, 3)`` with
    pixel values normalised to ``[0, 1]``.

    Args:
        settings: Runtime configuration instance.

    Example:
        >>> normalizer = ImageNormalizer(get_settings())
        >>> arr = normalizer.from_path(Path("xray.jpg"))
        >>> arr.shape
        (224, 224, 3)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------- Public API

    def from_pil(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to a normalised float32 array.

        The image is converted to RGB (discarding alpha or grayscale),
        resized to ``(height, width)``, and scaled to ``[0, 1]``.

        Args:
            image: Any PIL Image object.

        Returns:
            Float32 NumPy array of shape ``(H, W, 3)``.
        """
        rgb = image.convert("RGB")
        h, w = self._settings.image_height, self._settings.image_width
        resized = rgb.resize((w, h), Image.Resampling.BILINEAR)
        array = np.array(resized, dtype=np.float32) / 255.0
        log.debug(
            "image_normalised",
            source="pil",
            output_shape=array.shape,
        )
        return array

    def from_path(self, image_path: Path) -> np.ndarray:
        """Load an image from disk and return a normalised float32 array.

        Args:
            image_path: Path to a JPEG, PNG, or other PIL-supported file.

        Returns:
            Float32 NumPy array of shape ``(H, W, 3)``.

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            image = Image.open(image_path)
            array = self.from_pil(image)
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Cannot decode image at {image_path}: {exc}"
            ) from exc
        log.debug("image_loaded_from_path", path=str(image_path))
        return array

    def from_bytes(self, raw_bytes: bytes) -> np.ndarray:
        """Decode image bytes and return a normalised float32 array.

        Args:
            raw_bytes: Raw binary content of an image file.

        Returns:
            Float32 NumPy array of shape ``(H, W, 3)``.

        Raises:
            ValueError: If ``raw_bytes`` cannot be decoded as an image.
        """
        import io

        try:
            image = Image.open(io.BytesIO(raw_bytes))
            array = self.from_pil(image)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot decode image from bytes: {exc}") from exc
        log.debug(
            "image_decoded_from_bytes",
            byte_length=len(raw_bytes),
        )
        return array

    # ------------------------------------------------------- Properties

    @property
    def target_size(self) -> tuple[int, int]:
        """Return ``(height, width)`` resize target.

        Returns:
            Tuple of ``(image_height, image_width)`` from settings.
        """
        return (
            self._settings.image_height,
            self._settings.image_width,
        )
