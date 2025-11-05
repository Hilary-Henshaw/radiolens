"""
training-walkthrough/train.py
==============================
End-to-end example: balance a raw dataset, train the model, and evaluate it.

Expected raw dataset structure:
    data/raw/
        NORMAL/    *.jpg  *.jpeg  *.png
        PNEUMONIA/ *.jpg  *.jpeg  *.png

The script will create:
    data/balanced/       — balanced, stratified splits
    runs/<experiment>/   — training artefacts and metrics

Usage
-----
    RADIOLENS_MODEL_WEIGHTS_PATH=runs/exp01/best_weights.keras \
        python examples/training-walkthrough/train.py \
        --raw-data data/raw \
        --output-dir runs/exp01

Requirements
------------
    pip install "radiolens[api]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radiolens.config import Settings, get_settings
from radiolens.core.detector import ThoraxClassifier
from radiolens.core.preprocessor import ImageNormalizer
from radiolens.data.balancer import RadiographBalancer
from radiolens.evaluation.metrics import compute_binary_metrics, report_to_dict


def balance_data(
    raw_dir: Path, balanced_dir: Path, settings: Settings
) -> None:
    """Undersample majority class and create train/val/test splits.

    Args:
        raw_dir: Root directory with NORMAL/ and PNEUMONIA/ subfolders.
        balanced_dir: Destination for balanced splits.
        settings: Runtime configuration.
    """
    print(f"Inspecting distribution in: {raw_dir}")
    balancer = RadiographBalancer(settings)
    dist = balancer.inspect_distribution(raw_dir)

    print(f"  normal:    {dist.class_counts['normal']}")
    print(f"  pneumonia: {dist.class_counts['pneumonia']}")
    print(f"  imbalance ratio: {dist.imbalance_ratio:.2f}x")
    print()

    print(f"Balancing and splitting → {balanced_dir}")
    summary = balancer.equalize_and_split(raw_dir, balanced_dir)
    total = summary.total_images
    print(f"  train: {sum(summary.train_counts.values())} images")
    print(f"  val:   {sum(summary.validation_counts.values())} images")
    print(f"  test:  {sum(summary.test_counts.values())} images")
    print(f"  total: {total} images\n")


def train_model(
    balanced_dir: Path,
    output_dir: Path,
    settings: Settings,
) -> Path:
    """Build and train ThoraxClassifier, returning the saved weights path.

    Uses Keras ImageDataGenerator for augmented training flows.

    Args:
        balanced_dir: Root of balanced dataset with train/val/test splits.
        output_dir: Directory to save the best checkpoint.
        settings: Runtime configuration.

    Returns:
        Path to the saved .keras weights file.
    """
    from radiolens.data.augmentor import (
        build_training_flow,
        build_validation_flow,
    )
    from radiolens.training.runner import DetectorTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "best_weights.keras"

    print("Building model...")
    classifier = ThoraxClassifier(settings)
    classifier.build()

    print("Creating data generators...")
    train_gen = build_training_flow(balanced_dir / "train", settings)
    val_gen = build_validation_flow(balanced_dir / "val", settings)

    print(f"Training for up to {settings.max_epochs} epochs "
          f"(early stop patience={settings.early_stop_patience})...\n")
    trainer = DetectorTrainer(classifier, settings, output_dir)
    outcome = trainer.fit(train_gen, val_gen)

    classifier.save(weights_path)
    print(f"Best epoch: {outcome.best_epoch}/{outcome.total_epochs_run}  "
          f"val_accuracy={outcome.final_val_accuracy:.4f}")
    print(f"\nWeights saved to: {weights_path}")
    return weights_path


def evaluate_model(
    balanced_dir: Path,
    weights_path: Path,
    output_dir: Path,
    settings: Settings,
) -> None:
    """Load trained weights and evaluate on the held-out test split.

    Args:
        balanced_dir: Root of balanced dataset.
        weights_path: Path to saved .keras weights.
        output_dir: Directory to write metrics JSON.
        settings: Runtime configuration.
    """
    print("\nEvaluating on test set...")
    classifier = ThoraxClassifier(settings)
    classifier.load_weights(weights_path)
    normalizer = ImageNormalizer(settings)

    test_dir = balanced_dir / "test"
    normal_files = sorted((test_dir / "normal").glob("*.jpg")) + \
                   sorted((test_dir / "normal").glob("*.jpeg")) + \
                   sorted((test_dir / "normal").glob("*.png"))
    pneumonia_files = sorted((test_dir / "pneumonia").glob("*.jpg")) + \
                      sorted((test_dir / "pneumonia").glob("*.jpeg")) + \
                      sorted((test_dir / "pneumonia").glob("*.png"))

    labels: list[int] = []
    probabilities: list[float] = []

    for path in normal_files:
        arr = normalizer.from_path(path)
        result = classifier.run_inference(arr)
        labels.append(0)
        probabilities.append(result.probability)

    for path in pneumonia_files:
        arr = normalizer.from_path(path)
        result = classifier.run_inference(arr)
        labels.append(1)
        probabilities.append(result.probability)

    y_true = np.array(labels, dtype=int)
    y_proba = np.array(probabilities, dtype=np.float32)

    report = compute_binary_metrics(y_true, y_proba)
    metrics = report_to_dict(report)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<28} {value:>10.4f}")
        else:
            print(f"  {key:<28} {value:>10}")

    out_path = output_dir / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to: {out_path}")


def main(raw_data: Path, output_dir: Path) -> None:
    """Run the full training walkthrough.

    Args:
        raw_data: Root directory with raw NORMAL/ and PNEUMONIA/ images.
        output_dir: Directory for all outputs.
    """
    settings = get_settings()
    balanced_dir = output_dir / "balanced"

    # Step 1: Balance and split
    balance_data(raw_data, balanced_dir, settings)

    # Step 2: Train
    weights_path = train_model(balanced_dir, output_dir, settings)

    # Step 3: Evaluate
    evaluate_model(balanced_dir, weights_path, output_dir, settings)

    print(f"\nDone. Artefacts in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balance, train, and evaluate radiolens."
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        required=True,
        help="Root directory with NORMAL/ and PNEUMONIA/ image subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/exp01"),
        help="Destination for balanced data and model artefacts.",
    )
    args = parser.parse_args()
    main(args.raw_data, args.output_dir)
