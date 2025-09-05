"""Settings for the radiolens pneumonia detection system."""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger(__name__)


class Settings(BaseSettings):
    """All runtime configuration for radiolens.

    Values are read from environment variables (prefix RADIOLENS_)
    and from a .env file in the working directory.

    Example:
        >>> s = Settings()
        >>> s.image_height
        224
    """

    model_config = SettingsConfigDict(
        env_prefix="RADIOLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ Image
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 3

    # ---------------------------------------------------------------- Backbone
    backbone_name: str = "MobileNetV2"

    # -------------------------------------------------------------------- Head
    dense_layer_units: int = 128
    first_dropout_rate: float = 0.3
    second_dropout_rate: float = 0.2

    # --------------------------------------------------------------- Training
    batch_size: int = 32
    max_epochs: int = 25
    initial_learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-7
    early_stop_patience: int = 7
    lr_reduce_patience: int = 4
    lr_reduce_factor: float = 0.5
    random_seed: int = 42

    # ----------------------------------------------------------- Augmentation
    aug_rotation_degrees: float = 20.0
    aug_width_shift: float = 0.2
    aug_height_shift: float = 0.2
    aug_zoom_range: float = 0.2
    aug_horizontal_flip: bool = True
    aug_brightness_lower: float = 0.8
    aug_brightness_upper: float = 1.2

    # -------------------------------------------------------------------- Data
    train_fraction: float = 0.7
    validation_fraction: float = 0.2
    test_fraction: float = 0.1
    accepted_image_suffixes: list[str] = Field(
        default=[".jpeg", ".jpg", ".png"]
    )

    # --------------------------------------------------------------------- API
    api_host: str = "0.0.0.0"
    api_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("RADIOLENS_API_PORT", "PORT"),
    )
    max_upload_bytes: int = 10 * 1024 * 1024
    model_weights_path: Path = Path("model/best_weights.keras")
    cors_allow_origins: list[str] = Field(default=["*"])

    # ------------------------------------------------------------- Bootstrap
    bootstrap_resamples: int = 1000
    bootstrap_ci_level: float = 0.95

    # -------------------------------------------------------- Derived helpers
    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return (height, width, channels) as a tuple."""
        return (self.image_height, self.image_width, self.image_channels)

    @property
    def target_size(self) -> tuple[int, int]:
        """Return (height, width) for PIL resize operations."""
        return (self.image_height, self.image_width)

    # ------------------------------------------------------- Validation
    @model_validator(mode="after")
    def _check_split_fractions(self) -> Settings:
        """Verify that train + val + test fractions sum to 1.0."""
        total = (
            self.train_fraction + self.validation_fraction + self.test_fraction
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_fraction + validation_fraction + test_fraction "
                f"must equal 1.0, got {total:.6f}"
            )
        return self


# ------------------------------------------------------------------ Singleton

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the module-level Settings singleton.

    Creates the instance on first call and reuses it thereafter.

    Returns:
        The global Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        log.debug(
            "settings_initialised",
            backbone=_settings.backbone_name,
            image_shape=_settings.input_shape,
        )
    return _settings
