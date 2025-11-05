"""
batch-evaluation/evaluate.py
=============================
Demonstrates how to evaluate the model on an external test set and compute
the full clinical metric suite with bootstrap statistical testing.

Directory structure expected:
    data/external_test/
        normal/    *.jpg
        pneumonia/ *.jpg

Usage
-----
    RADIOLENS_MODEL_WEIGHTS_PATH=../../model/best_weights.keras \
        python examples/batch-evaluation/evaluate.py \
        --data-dir data/external_test \
        --output-dir results/

Requirements
------------
    pip install "radiolens[api]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radiolens.config import get_settings
from radiolens.core.detector import ThoraxClassifier
from radiolens.core.preprocessor import ImageNormalizer
from radiolens.evaluation.metrics import compute_binary_metrics, report_to_dict


def load_test_set(
    data_dir: Path,
    normalizer: ImageNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from normal/ and pneumonia/ subdirectories.

    Args:
        data_dir: Root directory containing normal/ and pneumonia/ folders.
        normalizer: ImageNormalizer instance for preprocessing.

    Returns:
        Tuple of (y_true, pixel_arrays) where y_true is a 1-D int array
        (0 = normal, 1 = pneumonia) and pixel_arrays has shape
        (N, 224, 224, 3).
    """
    normal_dir = data_dir / "normal"
    pneumonia_dir = data_dir / "pneumonia"

    if not normal_dir.is_dir():
        normal_dir = data_dir / "NORMAL"
    if not pneumonia_dir.is_dir():
        pneumonia_dir = data_dir / "PNEUMONIA"

    suffixes = {".jpg", ".jpeg", ".png"}

    normal_files = sorted(
        f for f in normal_dir.iterdir() if f.suffix.lower() in suffixes
    )
    pneumonia_files = sorted(
        f for f in pneumonia_dir.iterdir() if f.suffix.lower() in suffixes
    )

    n_normal, n_pneumonia = len(normal_files), len(pneumonia_files)
    print(f"Found {n_normal} normal, {n_pneumonia} pneumonia images.")

    arrays: list[np.ndarray] = []
    labels: list[int] = []

    for path in normal_files:
        arrays.append(normalizer.from_path(path))
        labels.append(0)

    for path in pneumonia_files:
        arrays.append(normalizer.from_path(path))
        labels.append(1)

    return np.array(labels, dtype=int), np.stack(arrays, axis=0)


def run_batch_inference(
    classifier: ThoraxClassifier,
    pixel_arrays: np.ndarray,
) -> np.ndarray:
    """Run inference on every image and return raw sigmoid probabilities.

    Args:
        classifier: Loaded ThoraxClassifier.
        pixel_arrays: Array of shape (N, H, W, C).

    Returns:
        1-D float32 array of probabilities, shape (N,).
    """
    probabilities: list[float] = []
    n = len(pixel_arrays)

    for i, arr in enumerate(pixel_arrays):
        result = classifier.run_inference(arr)
        probabilities.append(result.probability)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} images...")

    return np.array(probabilities, dtype=np.float32)


def main(data_dir: Path, output_dir: Path) -> None:
    """Evaluate the model and save metrics to output_dir.

    Args:
        data_dir: Test set root directory.
        output_dir: Directory for output JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    # Load model
    print(f"Loading weights from: {settings.model_weights_path}")
    classifier = ThoraxClassifier(settings)
    classifier.load_weights(Path(settings.model_weights_path))

    normalizer = ImageNormalizer(settings)

    # Load test data
    print(f"\nLoading test data from: {data_dir}")
    y_true, pixel_arrays = load_test_set(data_dir, normalizer)

    # Batch inference
    print(f"\nRunning inference on {len(y_true)} images...")
    y_proba = run_batch_inference(classifier, pixel_arrays)

    # Compute metrics
    print("\nComputing clinical metrics...")
    report = compute_binary_metrics(y_true, y_proba)
    metrics_dict = report_to_dict(report)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key:<28} {value:>10.4f}")
        else:
            print(f"  {key:<28} {value:>10}")

    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, indent=2))
    print(f"\nMetrics saved to: {metrics_path}")

    # Bootstrap generalisation test (requires internal set for comparison)
    # Uncomment and provide internal test arrays to run bootstrap testing:
    #
    # validator = BootstrapValidator(settings)
    # bootstrap = validator.assess_generalisation(
    #     y_true_internal, y_proba_internal, y_true, y_proba
    # )
    # print(f"\nBootstrap p-value: {bootstrap.p_value:.4f}")
    # print(
    #     f"95% CI for ΔAUC:   [{bootstrap.ci_lower:.4f}, "
    #     f"{bootstrap.ci_upper:.4f}]"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate radiolens on an external test set."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root of test set (contains normal/ and pneumonia/ subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/"),
        help="Directory for output JSON files (default: results/).",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
