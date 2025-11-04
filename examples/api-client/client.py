"""
api-client/client.py
====================
Shows how to interact with the radiolens REST API from Python.

Covers:
- POST /api/v1/classify   — single-image classification
- GET  /api/v1/health     — service health check
- GET  /api/v1/performance — published validation metrics

Usage
-----
    # 1. Start the API server in another terminal:
    #    RADIOLENS_MODEL_WEIGHTS_PATH=../../model/best_weights.keras \\
    #        uvicorn radiolens.api.server:app --port 8000

    # 2. Run this client:
    python examples/api-client/client.py path/to/xray.jpg

Requirements
------------
    pip install requests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000/api/v1"


def check_health() -> None:
    """Print the service health status."""
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    status_icon = "✓" if data["status"] == "healthy" else "✗"
    print(f"[{status_icon}] API status: {data['status']}  "
          f"(model_loaded={data['model_loaded']}, "
          f"uptime={data['uptime_seconds']:.1f}s, "
          f"version={data['version']})")


def classify_image(image_path: Path) -> dict:
    """Submit an image to the API and return the classification result.

    Args:
        image_path: Path to a JPEG, PNG, or DICOM file.

    Returns:
        Parsed JSON response dict.
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/classify",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )

    if resp.status_code == 400:
        print(f"[ERROR] Unsupported file type: {resp.json()['detail']}")
        sys.exit(1)
    if resp.status_code == 422:
        print(f"[ERROR] Cannot decode image: {resp.json()['detail']}")
        sys.exit(1)

    resp.raise_for_status()
    return resp.json()


def get_performance_stats() -> dict:
    """Fetch published cross-operator validation metrics.

    Returns:
        Parsed JSON response dict.
    """
    resp = requests.get(f"{BASE_URL}/performance", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main(image_path: Path) -> None:
    """Run a complete client demonstration.

    Args:
        image_path: Path to the image file to classify.
    """
    print("=== radiolens API Client Demo ===\n")

    # Health check
    check_health()
    print()

    # Classify
    print(f"Classifying: {image_path}")
    result = classify_image(image_path)
    print(f"  Label:         {result['label']}")
    print(f"  Probability:   {result['probability']:.4f}")
    print(f"  Confidence:    {result['confidence']:.4f}")
    print(f"  Certainty tier:{result['certainty_tier']}")
    print(f"  Model version: {result['model_version']}")
    print()

    # Performance stats
    print("Cross-operator validation performance:")
    stats = get_performance_stats()
    print(f"  Accuracy:    {stats['external_accuracy']:.1%}")
    print(f"  Sensitivity: {stats['external_sensitivity']:.1%}")
    print(f"  Specificity: {stats['external_specificity']:.1%}")
    print(f"  ROC-AUC:     {stats['external_roc_auc']:.1%}")
    print(f"  Bootstrap p: {stats['bootstrap_p_value']:.3f}")
    print()
    print("Full response JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    main(Path(sys.argv[1]))
