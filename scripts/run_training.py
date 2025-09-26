"""CLI: train the ThoraxClassifier from a balanced dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``run_training``.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="run_training",
        description=(
            "Train the ThoraxClassifier on a balanced chest X-ray dataset "
            "and save the best weights to a checkpoint directory."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="PATH",
        help=(
            "Root of the balanced dataset.  Must contain train/ and val/ "
            "subdirectories produced by prepare_dataset."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./model",
        metavar="PATH",
        help="Directory where best weights are saved (default: ./model).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a .env file that overrides default settings.",
    )
    return parser


def _print_outcome(outcome: object) -> None:  # type: ignore[type-arg]
    """Print a human-readable TrainingOutcome summary.

    Args:
        outcome: A :class:`~radiolens.training.runner.TrainingOutcome`.
    """
    from radiolens.training.runner import TrainingOutcome

    assert isinstance(outcome, TrainingOutcome)
    print("\n--- Training Outcome ---")
    print(f"  {'Best epoch':>22s}: {outcome.best_epoch}")
    print(f"  {'Total epochs run':>22s}: {outcome.total_epochs_run}")
    print(f"  {'Val accuracy (best)':>22s}: {outcome.final_val_accuracy:.5f}")
    print(f"  {'Val loss (best)':>22s}: {outcome.final_val_loss:.5f}")
    print(f"  {'Checkpoint saved':>22s}: {outcome.checkpoint_path}")
    print()


def main() -> None:
    """Entry point for ``run_training``."""
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Allow an override .env file
    if args.config:
        import os

        os.environ.setdefault("RADIOLENS_ENV_FILE", args.config)

    try:
        from radiolens.config import get_settings
        from radiolens.core.detector import ThoraxClassifier
        from radiolens.data.augmentor import (
            build_training_flow,
            build_validation_flow,
        )
        from radiolens.training.runner import DetectorTrainer

        settings = get_settings()

        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

        for required in (train_dir, val_dir):
            if not required.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {required}"
                )

        print(f"Building model ({settings.backbone_name})…")
        classifier = ThoraxClassifier(settings)
        classifier.build()

        print(f"Loading training data from: {train_dir}")
        train_flow = build_training_flow(train_dir, settings)

        print(f"Loading validation data from: {val_dir}")
        val_flow = build_validation_flow(val_dir, settings)

        print(
            f"Training for up to {settings.max_epochs} epochs "
            f"(batch size {settings.batch_size})…"
        )
        trainer = DetectorTrainer(classifier, settings, checkpoint_dir)
        outcome = trainer.fit(train_flow, val_flow)

        _print_outcome(outcome)
        print("Training complete.")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
