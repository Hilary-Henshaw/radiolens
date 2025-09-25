"""Smoke test: full training loop on synthetic miniature data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Keras/TF required; skip entire module gracefully if absent.
pytest.importorskip(
    "keras",
    reason="TensorFlow/Keras not installed — skipping smoke tests",
)

from radiolens.config import Settings  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a minimal 20-image dataset for a 1-epoch smoke test.

    Structure::

        <tmp>/
            train/normal/    (5 images)
            train/pneumonia/ (5 images)
            val/normal/      (3 images)
            val/pneumonia/   (3 images)
            test/normal/     (2 images)
            test/pneumonia/  (2 images)
    """
    splits: dict[str, int] = {"train": 5, "val": 3, "test": 2}
    classes: dict[str, int] = {"normal": 0, "pneumonia": 200}

    for split, count in splits.items():
        for class_name, fill_val in classes.items():
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            for i in range(count):
                arr = np.full(
                    (64, 64, 3),
                    fill_val,
                    dtype=np.uint8,
                )
                Image.fromarray(arr).save(folder / f"{i}.png")

    return tmp_path


@pytest.mark.slow
def test_training_completes_without_error(
    tiny_dataset: Path,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """A full 1-epoch training run finishes and produces a checkpoint."""
    from radiolens.core.detector import ThoraxClassifier
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    clf = ThoraxClassifier(settings)
    clf.build()
    train_flow = build_training_flow(tiny_dataset / "train", settings)
    val_flow = build_validation_flow(tiny_dataset / "val", settings)
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = DetectorTrainer(clf, settings, checkpoint_dir)
    outcome = trainer.fit(train_flow, val_flow)

    assert outcome.total_epochs_run >= 1
    assert 0.0 <= outcome.final_val_accuracy <= 1.0
    assert outcome.checkpoint_path.exists()


@pytest.mark.slow
def test_training_outcome_has_history_keys(
    tiny_dataset: Path,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """TrainingOutcome.history contains accuracy, loss and val variants."""
    from radiolens.core.detector import ThoraxClassifier
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    clf = ThoraxClassifier(settings)
    clf.build()
    train_flow = build_training_flow(tiny_dataset / "train", settings)
    val_flow = build_validation_flow(tiny_dataset / "val", settings)
    trainer = DetectorTrainer(clf, settings, tmp_path / "checkpoints")
    outcome = trainer.fit(train_flow, val_flow)

    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        assert key in outcome.history, f"Expected '{key}' in history keys"


@pytest.mark.slow
def test_model_can_infer_after_training(
    tiny_dataset: Path,
    settings: Settings,
    tmp_path: Path,
    synthetic_pixel_array: np.ndarray,
) -> None:
    """After training, the classifier can run inference without error."""
    from radiolens.core.detector import InferenceResult, ThoraxClassifier
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    clf = ThoraxClassifier(settings)
    clf.build()
    train_flow = build_training_flow(tiny_dataset / "train", settings)
    val_flow = build_validation_flow(tiny_dataset / "val", settings)
    trainer = DetectorTrainer(clf, settings, tmp_path / "checkpoints")
    trainer.fit(train_flow, val_flow)

    result = clf.run_inference(synthetic_pixel_array)
    assert isinstance(result, InferenceResult)
    assert result.label in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= result.probability <= 1.0


@pytest.mark.slow
def test_best_epoch_is_within_total_epochs(
    tiny_dataset: Path,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """best_epoch is a positive int <= total_epochs_run."""
    from radiolens.core.detector import ThoraxClassifier
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    clf = ThoraxClassifier(settings)
    clf.build()
    train_flow = build_training_flow(tiny_dataset / "train", settings)
    val_flow = build_validation_flow(tiny_dataset / "val", settings)
    trainer = DetectorTrainer(clf, settings, tmp_path / "checkpoints")
    outcome = trainer.fit(train_flow, val_flow)

    assert 1 <= outcome.best_epoch <= outcome.total_epochs_run


@pytest.mark.slow
def test_final_val_loss_is_positive(
    tiny_dataset: Path,
    settings: Settings,
    tmp_path: Path,
) -> None:
    """final_val_loss is a positive float value."""
    from radiolens.core.detector import ThoraxClassifier
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    clf = ThoraxClassifier(settings)
    clf.build()
    train_flow = build_training_flow(tiny_dataset / "train", settings)
    val_flow = build_validation_flow(tiny_dataset / "val", settings)
    trainer = DetectorTrainer(clf, settings, tmp_path / "checkpoints")
    outcome = trainer.fit(train_flow, val_flow)

    assert outcome.final_val_loss >= 0.0
