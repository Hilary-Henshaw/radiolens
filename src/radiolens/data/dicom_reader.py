"""DICOM medical imaging file support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger(__name__)

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pydicom is required for DICOM support. "
        "Install it with: pip install pydicom"
    ) from exc


def read_dicom_as_pil(dicom_path: Path) -> Image.Image:
    """Load a DICOM file and return an RGB PIL Image.

    Applies window/level normalisation to map pixel values to ``[0, 255]``
    before converting to RGB PIL Image.

    Window centre and width are read from the DICOM tags when present;
    otherwise a min/max stretch is applied.

    Args:
        dicom_path: Path to the ``.dcm`` file.

    Returns:
        An RGB :class:`PIL.Image.Image`.

    Raises:
        FileNotFoundError: If ``dicom_path`` does not exist on disk.
        ValueError: If the file is not a valid DICOM dataset.
    """
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    try:
        dataset = pydicom.dcmread(str(dicom_path))
    except InvalidDicomError as exc:
        raise ValueError(
            f"Not a valid DICOM file: {dicom_path} — {exc}"
        ) from exc

    pixel_data: np.ndarray = dataset.pixel_array.astype(np.float32)

    # Apply window/level if tags are present
    if hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth"):
        _mv = pydicom.multival.MultiValue
        centre = float(
            dataset.WindowCenter
            if not isinstance(dataset.WindowCenter, _mv)
            else dataset.WindowCenter[0]
        )
        width = float(
            dataset.WindowWidth
            if not isinstance(dataset.WindowWidth, _mv)
            else dataset.WindowWidth[0]
        )
        low = centre - width / 2.0
        high = centre + width / 2.0
        pixel_data = np.clip(pixel_data, low, high)
        pixel_data = (pixel_data - low) / (high - low) * 255.0
    else:
        p_min = float(pixel_data.min())
        p_max = float(pixel_data.max())
        if p_max > p_min:
            pixel_data = (pixel_data - p_min) / (p_max - p_min) * 255.0
        else:
            pixel_data = np.zeros_like(pixel_data)

    uint8_array = pixel_data.clip(0, 255).astype(np.uint8)

    # Handle grayscale vs RGB
    if uint8_array.ndim == 2:
        pil_image = Image.fromarray(uint8_array, mode="L").convert("RGB")
    elif uint8_array.ndim == 3 and uint8_array.shape[2] == 3:
        pil_image = Image.fromarray(uint8_array, mode="RGB")
    elif uint8_array.ndim == 3 and uint8_array.shape[2] == 1:
        pil_image = Image.fromarray(uint8_array[:, :, 0], mode="L").convert(
            "RGB"
        )
    else:
        raise ValueError(
            f"Unexpected pixel array shape {uint8_array.shape} "
            f"in DICOM file: {dicom_path}"
        )

    log.info(
        "dicom_loaded",
        path=str(dicom_path),
        original_shape=dataset.pixel_array.shape,
    )
    return pil_image


def is_dicom_file(file_path: Path) -> bool:
    """Return ``True`` if ``file_path`` is a readable DICOM file.

    Attempts a lightweight header read without loading pixel data.

    Args:
        file_path: Path to the file to test.

    Returns:
        ``True`` if the file is a valid DICOM dataset, ``False`` otherwise.
    """
    if not file_path.exists():
        return False
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
        return True
    except (InvalidDicomError, Exception):
        return False
