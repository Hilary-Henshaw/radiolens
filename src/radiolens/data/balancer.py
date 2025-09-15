"""RadiographBalancer: equalise class distribution and partition datasets."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog

from radiolens.config import Settings

log = structlog.get_logger(__name__)


@dataclass
class ClassDistribution:
    """Summary of class counts in a raw dataset directory.

    Attributes:
        class_counts: Mapping of canonical class name to image count.
        total_images: Sum of all class counts.
        imbalance_ratio: majority_count / minority_count.
        majority_class: Canonical name of the over-represented class.
        minority_class: Canonical name of the under-represented class.
    """

    class_counts: dict[str, int]
    total_images: int
    imbalance_ratio: float
    majority_class: str
    minority_class: str


@dataclass
class PartitionSummary:
    """Counts of images written to each split directory.

    Attributes:
        train_counts: Per-class counts in the training split.
        validation_counts: Per-class counts in the validation split.
        test_counts: Per-class counts in the test split.
        total_images: Grand total across all splits.
    """

    train_counts: dict[str, int]
    validation_counts: dict[str, int]
    test_counts: dict[str, int]
    total_images: int


class RadiographBalancer:
    """Undersample majority class and create stratified train/val/test splits.

    Supported folder naming for each class:
    - Normal: ``normal`` or ``NORMAL``
    - Pneumonia: ``pneumonia`` or ``PNEUMONIA``

    Args:
        settings: Runtime configuration instance.

    Example:
        >>> balancer = RadiographBalancer(get_settings())
        >>> dist = balancer.inspect_distribution(Path("raw_data/train"))
        >>> summary = balancer.equalize_and_split(
        ...     Path("raw_data"), Path("balanced_data")
        ... )
    """

    NORMAL_ALIASES: ClassVar[frozenset[str]] = frozenset({"normal", "NORMAL"})
    PNEUMONIA_ALIASES: ClassVar[frozenset[str]] = frozenset(
        {"pneumonia", "PNEUMONIA"}
    )

    _CANONICAL_NORMAL: ClassVar[str] = "normal"
    _CANONICAL_PNEUMONIA: ClassVar[str] = "pneumonia"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --------------------------------------------------------- Inspection

    def inspect_distribution(self, source_dir: Path) -> ClassDistribution:
        """Count images per class in ``source_dir`` subfolders.

        Args:
            source_dir: Root directory containing class sub-directories.

        Returns:
            A :class:`ClassDistribution` describing the dataset.

        Raises:
            FileNotFoundError: If no valid class folders are found under
                ``source_dir``.
        """
        normal_files = self._collect_files(source_dir, self.NORMAL_ALIASES)
        pneumonia_files = self._collect_files(
            source_dir, self.PNEUMONIA_ALIASES
        )

        if not normal_files and not pneumonia_files:
            raise FileNotFoundError(
                f"No valid class folders (normal/NORMAL, "
                f"pneumonia/PNEUMONIA) found under: {source_dir}"
            )

        counts: dict[str, int] = {
            self._CANONICAL_NORMAL: len(normal_files),
            self._CANONICAL_PNEUMONIA: len(pneumonia_files),
        }
        total = sum(counts.values())

        minority_cls, majority_cls = sorted(counts, key=lambda k: counts[k])
        minority_count = counts[minority_cls]
        majority_count = counts[majority_cls]
        ratio = (
            majority_count / minority_count
            if minority_count > 0
            else float("inf")
        )

        dist = ClassDistribution(
            class_counts=counts,
            total_images=total,
            imbalance_ratio=ratio,
            majority_class=majority_cls,
            minority_class=minority_cls,
        )
        log.info(
            "distribution_inspected",
            normal=counts[self._CANONICAL_NORMAL],
            pneumonia=counts[self._CANONICAL_PNEUMONIA],
            imbalance_ratio=round(ratio, 3),
        )
        return dist

    # --------------------------------------------------------- Balancing

    def equalize_and_split(
        self,
        source_dir: Path,
        output_dir: Path,
    ) -> PartitionSummary:
        """Undersample majority class, then create stratified splits.

        Creates the following directory tree under ``output_dir``::

            output_dir/
              train/normal/   train/pneumonia/
              val/normal/     val/pneumonia/
              test/normal/    test/pneumonia/

        Args:
            source_dir: Root of the raw dataset (contains class folders).
            output_dir: Destination root for the balanced dataset.

        Returns:
            A :class:`PartitionSummary` with per-split counts.
        """
        s = self._settings
        rng = np.random.default_rng(s.random_seed)

        normal_files = self._collect_files(source_dir, self.NORMAL_ALIASES)
        pneumonia_files = self._collect_files(
            source_dir, self.PNEUMONIA_ALIASES
        )

        if not normal_files or not pneumonia_files:
            raise FileNotFoundError(
                f"Both normal and pneumonia class folders are required "
                f"under: {source_dir}"
            )

        log.info(
            "balancing_started",
            normal_total=len(normal_files),
            pneumonia_total=len(pneumonia_files),
        )

        min_count = min(len(normal_files), len(pneumonia_files))

        normal_files = [
            normal_files[i] for i in rng.permutation(len(normal_files))
        ]
        pneumonia_files = [
            pneumonia_files[i] for i in rng.permutation(len(pneumonia_files))
        ]

        normal_files = normal_files[:min_count]
        pneumonia_files = pneumonia_files[:min_count]

        log.info(
            "class_undersampled",
            samples_per_class=min_count,
        )

        splits = self._compute_split_indices(min_count, s)

        split_names = ("train", "val", "test")
        class_names = (
            self._CANONICAL_NORMAL,
            self._CANONICAL_PNEUMONIA,
        )
        all_files = {
            self._CANONICAL_NORMAL: normal_files,
            self._CANONICAL_PNEUMONIA: pneumonia_files,
        }

        train_counts: dict[str, int] = {}
        val_counts: dict[str, int] = {}
        test_counts: dict[str, int] = {}

        for cls in class_names:
            files = all_files[cls]
            train_end, val_end = splits
            partitions = {
                "train": files[:train_end],
                "val": files[train_end:val_end],
                "test": files[val_end:],
            }
            for split in split_names:
                dest_dir = output_dir / split / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in partitions[split]:
                    shutil.copy2(src, dest_dir / src.name)

            train_counts[cls] = len(partitions["train"])
            val_counts[cls] = len(partitions["val"])
            test_counts[cls] = len(partitions["test"])

            log.info(
                "class_split_written",
                cls=cls,
                train=train_counts[cls],
                val=val_counts[cls],
                test=test_counts[cls],
            )

        total = (
            sum(train_counts.values())
            + sum(val_counts.values())
            + sum(test_counts.values())
        )

        summary = PartitionSummary(
            train_counts=train_counts,
            validation_counts=val_counts,
            test_counts=test_counts,
            total_images=total,
        )
        log.info(
            "equalization_complete",
            total_images=total,
            output_dir=str(output_dir),
        )
        return summary

    # ---------------------------------------------------------- Helpers

    def _collect_files(
        self,
        root: Path,
        aliases: frozenset[str],
    ) -> list[Path]:
        """Return all accepted image files from matching subdirectories.

        Args:
            root: Directory to search.
            aliases: Set of acceptable folder names.

        Returns:
            Sorted list of matching file paths.
        """
        suffixes = set(self._settings.accepted_image_suffixes)
        files: list[Path] = []
        for alias in aliases:
            folder = root / alias
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.suffix.lower() in suffixes:
                        files.append(f)
        return sorted(files)

    @staticmethod
    def _compute_split_indices(n: int, settings: Settings) -> tuple[int, int]:
        """Compute (train_end, val_end) indices for splitting n samples.

        Args:
            n: Total number of samples per class.
            settings: Runtime settings with fraction fields.

        Returns:
            Tuple ``(train_end, val_end)`` where each is an integer index.
        """
        train_end = int(n * settings.train_fraction)
        val_end = train_end + int(n * settings.validation_fraction)
        return train_end, val_end
