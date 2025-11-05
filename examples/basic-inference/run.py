"""
basic-inference/run.py
======================
Demonstrates the minimum code needed to load the model and classify a single
chest X-ray image.

Usage
-----
    # Set the path to your weights file, then run:
    RADIOLENS_MODEL_WEIGHTS_PATH=../../model/best_weights.keras \
        python examples/basic-inference/run.py path/to/xray.jpg

Requirements
------------
    pip install "radiolens[api]"
"""

from __future__ import annotations

import sys
from pathlib import Path

from radiolens.config import get_settings
from radiolens.core.detector import ThoraxClassifier
from radiolens.core.preprocessor import ImageNormalizer


def classify_image(image_path: Path) -> None:
    """Load model, preprocess image, and print the classification result.

    Args:
        image_path: Path to a JPEG or PNG chest X-ray file.
    """
    settings = get_settings()

    # 1. Build the classifier and load saved weights
    classifier = ThoraxClassifier(settings)
    classifier.load_weights(Path(settings.model_weights_path))

    # 2. Preprocess: resize to 224×224, normalise to [0, 1] float32
    normalizer = ImageNormalizer(settings)
    pixel_array = normalizer.from_path(image_path)

    # 3. Run inference
    result = classifier.run_inference(pixel_array)

    # 4. Display results
    print(f"Image:         {image_path.name}")
    print(f"Diagnosis:     {result.label}")
    print(f"Probability:   {result.probability:.4f}")
    print(f"Confidence:    {result.confidence:.4f}")
    print(f"Certainty:     {result.certainty_tier}")
    print()
    print(
        "DISCLAIMER: This is a research tool. "
        "Do not use for clinical decisions."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    classify_image(Path(sys.argv[1]))
