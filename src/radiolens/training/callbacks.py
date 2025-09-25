"""Custom Keras callbacks for structured logging."""

from __future__ import annotations

from pathlib import Path

import structlog

from radiolens.config import Settings

log = structlog.get_logger(__name__)

try:
    import keras
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow / Keras is required for radiolens. "
        "Install it with: pip install tensorflow"
    ) from exc


class StructuredLoggingCallback(keras.callbacks.Callback):
    """Log per-epoch metrics via structlog at INFO level.

    Each epoch emits a single structured log event containing the epoch
    number and all metric values reported by Keras.

    Example log event::

        {"event": "epoch_end", "epoch": 3, "loss": 0.21, "accuracy": 0.91}
    """

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, float] | None = None,
    ) -> None:
        """Emit a structured log entry with all epoch metrics.

        Args:
            epoch: Zero-based epoch index.
            logs: Dictionary of metric names to values for this epoch.
        """
        metrics = logs or {}
        log.info(
            "epoch_end",
            epoch=epoch + 1,
            **{k: round(float(v), 5) for k, v in metrics.items()},
        )


def build_standard_callbacks(
    settings: Settings,
    checkpoint_path: Path,
) -> list[keras.callbacks.Callback]:
    """Return the standard callback list for detector training.

    Builds and returns:

    1. :class:`keras.callbacks.ModelCheckpoint` — saves the best weights
       (by ``val_accuracy``) to ``checkpoint_path``.
    2. :class:`keras.callbacks.EarlyStopping` — stops training when
       ``val_accuracy`` has not improved for ``early_stop_patience`` epochs.
    3. :class:`keras.callbacks.ReduceLROnPlateau` — halves the learning rate
       when ``val_loss`` stagnates.
    4. :class:`StructuredLoggingCallback` — structured per-epoch logs.

    Args:
        settings: Runtime configuration instance supplying all patience,
            factor, and minimum LR values.
        checkpoint_path: Full file path where the best weights are saved.

    Returns:
        List of four :class:`keras.callbacks.Callback` instances.
    """
    s = settings

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=0,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=s.early_stop_patience,
        restore_best_weights=True,
        mode="max",
        verbose=0,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=s.lr_reduce_factor,
        patience=s.lr_reduce_patience,
        min_lr=s.minimum_learning_rate,
        mode="min",
        verbose=0,
    )

    structured_logger = StructuredLoggingCallback()

    return [checkpoint, early_stop, reduce_lr, structured_logger]
