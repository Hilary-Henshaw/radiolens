"""Keras ImageDataGenerator factories for training and validation."""

from __future__ import annotations

from pathlib import Path

import structlog

from radiolens.config import Settings

log = structlog.get_logger(__name__)

try:
    from keras.preprocessing.image import (
        DirectoryIterator,
        ImageDataGenerator,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow / Keras is required for radiolens. "
        "Install it with: pip install tensorflow"
    ) from exc


def build_training_flow(
    data_dir: Path,
    settings: Settings,
) -> DirectoryIterator:
    """Return an augmented DirectoryIterator for the training split.

    Applies rotation, width/height shift, zoom, horizontal flip,
    and brightness jitter as configured in ``settings``.

    Args:
        data_dir: Path to the training data directory.  Must contain
            one sub-folder per class.
        settings: Runtime configuration instance.

    Returns:
        A Keras :class:`DirectoryIterator` yielding ``(batch_X, batch_y)``
        pairs in binary mode.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {data_dir}"
        )

    s = settings
    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=s.aug_rotation_degrees,
        width_shift_range=s.aug_width_shift,
        height_shift_range=s.aug_height_shift,
        zoom_range=s.aug_zoom_range,
        horizontal_flip=s.aug_horizontal_flip,
        brightness_range=(s.aug_brightness_lower, s.aug_brightness_upper),
        fill_mode="nearest",
    )

    flow = generator.flow_from_directory(
        str(data_dir),
        target_size=(s.image_height, s.image_width),
        color_mode="rgb",
        class_mode="binary",
        batch_size=s.batch_size,
        shuffle=True,
        seed=s.random_seed,
    )

    log.info(
        "training_flow_created",
        data_dir=str(data_dir),
        samples=flow.samples,
        classes=flow.class_indices,
        batch_size=s.batch_size,
    )
    return flow


def build_validation_flow(
    data_dir: Path,
    settings: Settings,
) -> DirectoryIterator:
    """Return a non-augmented DirectoryIterator for validation or test data.

    Only applies rescaling to ``[0, 1]`` — no geometric augmentation.

    Args:
        data_dir: Path to the validation or test data directory.
        settings: Runtime configuration instance.

    Returns:
        A Keras :class:`DirectoryIterator` yielding ``(batch_X, batch_y)``
        pairs in binary mode.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Validation/test data directory not found: {data_dir}"
        )

    s = settings
    generator = ImageDataGenerator(rescale=1.0 / 255.0)

    flow = generator.flow_from_directory(
        str(data_dir),
        target_size=(s.image_height, s.image_width),
        color_mode="rgb",
        class_mode="binary",
        batch_size=s.batch_size,
        shuffle=False,
    )

    log.info(
        "validation_flow_created",
        data_dir=str(data_dir),
        samples=flow.samples,
        classes=flow.class_indices,
        batch_size=s.batch_size,
    )
    return flow
