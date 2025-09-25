"""DetectorTrainer: orchestrates the full training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from radiolens.config import Settings
from radiolens.core.detector import ThoraxClassifier
from radiolens.training.callbacks import build_standard_callbacks

log = structlog.get_logger(__name__)

try:
    from keras.preprocessing.image import DirectoryIterator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow / Keras is required for radiolens. "
        "Install it with: pip install tensorflow"
    ) from exc


@dataclass
class TrainingOutcome:
    """Summary of a completed training run.

    Attributes:
        final_val_accuracy: Validation accuracy at the best epoch.
        final_val_loss: Validation loss at the best epoch.
        best_epoch: One-based epoch number with the best val_accuracy.
        total_epochs_run: Total number of epochs completed.
        history: Full Keras history dictionary (metric name → list).
        checkpoint_path: Path where the best weights were saved.
    """

    final_val_accuracy: float
    final_val_loss: float
    best_epoch: int
    total_epochs_run: int
    history: dict[str, list[float]]
    checkpoint_path: Path


class DetectorTrainer:
    """Wraps the Keras fit() loop with callbacks, logging, and checkpointing.

    Args:
        classifier: Initialised
            :class:`~radiolens.core.detector.ThoraxClassifier`.
        settings: Runtime configuration instance.
        checkpoint_dir: Directory where model checkpoints are written.

    Example:
        >>> trainer = DetectorTrainer(clf, get_settings(), Path("./model"))
        >>> outcome = trainer.fit(train_flow, val_flow)
    """

    def __init__(
        self,
        classifier: ThoraxClassifier,
        settings: Settings,
        checkpoint_dir: Path,
    ) -> None:
        self._classifier = classifier
        self._settings = settings
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_flow: DirectoryIterator,
        val_flow: DirectoryIterator,
    ) -> TrainingOutcome:
        """Run the full training loop and return a :class:`TrainingOutcome`.

        Builds the model if it has not been built yet. Attaches
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and
        StructuredLoggingCallback. Logs start and end events.

        Args:
            train_flow: Augmented :class:`DirectoryIterator` for training.
            val_flow: Non-augmented :class:`DirectoryIterator` for validation.

        Returns:
            A :class:`TrainingOutcome` populated from the training history.

        Raises:
            RuntimeError: If the classifier's internal model cannot be built.
        """
        s = self._settings

        if self._classifier._model is None:
            self._classifier.build()

        checkpoint_path = self._checkpoint_dir / "best_weights.keras"
        callbacks = build_standard_callbacks(s, checkpoint_path)

        log.info(
            "training_started",
            max_epochs=s.max_epochs,
            batch_size=s.batch_size,
            train_samples=train_flow.samples,
            val_samples=val_flow.samples,
        )

        history = self._classifier._model.fit(  # type: ignore[union-attr]
            train_flow,
            validation_data=val_flow,
            epochs=s.max_epochs,
            callbacks=callbacks,
            verbose=0,
        )

        hist_dict: dict[str, list[float]] = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        total_epochs = len(hist_dict.get("loss", []))

        val_acc_series = hist_dict.get("val_accuracy", [])
        val_loss_series = hist_dict.get("val_loss", [])

        if val_acc_series:
            best_epoch_idx = int(
                max(
                    range(len(val_acc_series)),
                    key=lambda i: val_acc_series[i],
                )
            )
            final_val_accuracy = val_acc_series[best_epoch_idx]
            final_val_loss = (
                val_loss_series[best_epoch_idx] if val_loss_series else 0.0
            )
            best_epoch = best_epoch_idx + 1
        else:
            final_val_accuracy = 0.0
            final_val_loss = 0.0
            best_epoch = total_epochs

        outcome = TrainingOutcome(
            final_val_accuracy=final_val_accuracy,
            final_val_loss=final_val_loss,
            best_epoch=best_epoch,
            total_epochs_run=total_epochs,
            history=hist_dict,
            checkpoint_path=checkpoint_path,
        )

        log.info(
            "training_complete",
            best_epoch=best_epoch,
            total_epochs=total_epochs,
            val_accuracy=round(final_val_accuracy, 5),
            val_loss=round(final_val_loss, 5),
            checkpoint=str(checkpoint_path),
        )
        return outcome
