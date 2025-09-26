"""CLI: inspect and balance a raw chest X-ray dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``prepare_dataset``.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description=(
            "Inspect and optionally balance a raw chest X-ray dataset "
            "using RadiographBalancer.  Outputs a stratified train/val/test "
            "directory structure suitable for ThoraxClassifier training."
        ),
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        metavar="PATH",
        help="Root directory of the raw dataset (contains class subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        help=(
            "Destination directory for the balanced dataset.  "
            "Ignored when --dry-run is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=("Inspect class distribution only; do not copy any files."),
    )
    return parser


def _print_distribution(dist: object) -> None:  # type: ignore[type-arg]
    """Print a ClassDistribution summary to stdout.

    Args:
        dist: A :class:`~radiolens.data.balancer.ClassDistribution` instance.
    """
    from radiolens.data.balancer import ClassDistribution

    assert isinstance(dist, ClassDistribution)
    print("\n--- Class Distribution ---")
    for cls, count in dist.class_counts.items():
        print(f"  {cls:>20s}: {count:>6d} images")
    print(f"  {'Total':>20s}: {dist.total_images:>6d} images")
    print(f"  {'Imbalance ratio':>20s}: {dist.imbalance_ratio:.3f}")
    print(f"  {'Majority class':>20s}: {dist.majority_class}")
    print(f"  {'Minority class':>20s}: {dist.minority_class}")
    print()


def _print_summary(summary: object) -> None:  # type: ignore[type-arg]
    """Print a PartitionSummary to stdout.

    Args:
        summary: A :class:`~radiolens.data.balancer.PartitionSummary` instance.
    """
    from radiolens.data.balancer import PartitionSummary

    assert isinstance(summary, PartitionSummary)
    print("\n--- Partition Summary ---")
    for split_name, counts in [
        ("Train", summary.train_counts),
        ("Validation", summary.validation_counts),
        ("Test", summary.test_counts),
    ]:
        total_split = sum(counts.values())
        class_str = ", ".join(
            f"{cls}: {n}" for cls, n in sorted(counts.items())
        )
        print(f"  {split_name:>12s}: {total_split:>5d}  [{class_str}]")
    print(f"  {'Grand total':>12s}: {summary.total_images:>5d}")
    print()


def main() -> None:
    """Entry point for ``prepare_dataset``."""
    parser = _build_parser()
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    try:
        # Import here to delay heavy dependency loading
        from radiolens.config import get_settings
        from radiolens.data.balancer import RadiographBalancer

        settings = get_settings()
        balancer = RadiographBalancer(settings)

        print(f"Inspecting: {source_dir}")
        dist = balancer.inspect_distribution(source_dir)
        _print_distribution(dist)

        if args.dry_run:
            print("--dry-run specified; skipping file copy.")
            return

        print(f"Balancing and splitting → {output_dir}")
        summary = balancer.equalize_and_split(source_dir, output_dir)
        _print_summary(summary)
        print("Done.")

    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
