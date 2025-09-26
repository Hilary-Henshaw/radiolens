"""CLI: export a trained model to TensorFlow Lite for edge deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``export_weights``.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="export_weights",
        description=(
            "Convert a trained ThoraxClassifier model to TensorFlow Lite "
            "format with DEFAULT optimisations (float16 quantization).  "
            "Reports original vs compressed file sizes on completion."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        metavar="PATH",
        help="Source .keras or .h5 model file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        metavar="PATH",
        help="Destination .tflite file path.",
    )
    return parser


def _human_readable_size(n_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n_bytes: Raw byte count.

    Returns:
        String such as ``"3.14 MB"`` or ``"512.0 KB"``.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes = int(n_bytes / 1024.0)
    return f"{n_bytes:.1f} TB"


def main() -> None:
    """Entry point for ``export_weights``."""
    parser = _build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not model_path.exists():
        print(
            f"ERROR: Model file not found: {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import tensorflow as tf

        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path))

        print("Converting to TensorFlow Lite with DEFAULT optimisations…")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply float16 quantisation via DEFAULT optimisations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)

        original_bytes = model_path.stat().st_size
        compressed_bytes = output_path.stat().st_size
        compression_ratio = (
            original_bytes / compressed_bytes
            if compressed_bytes > 0
            else float("inf")
        )

        print("\n--- Export Summary ---")
        print(
            f"  {'Source model':>20s}: {_human_readable_size(original_bytes)}"
        )
        print(
            f"  {'TFLite model':>20s}: "
            f"{_human_readable_size(compressed_bytes)}"
        )
        print(f"  {'Compression ratio':>20s}: {compression_ratio:.2f}x")
        print(f"\nTFLite model saved to: {output_path}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(
            f"ERROR: TensorFlow is required for export: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
