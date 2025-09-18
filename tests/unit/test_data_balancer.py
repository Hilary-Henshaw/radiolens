"""Unit tests for RadiographBalancer dataset handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiolens.config import Settings
from radiolens.data.balancer import (
    ClassDistribution,
    PartitionSummary,
    RadiographBalancer,
)

# --------------------------------------------------- inspect_distribution


class TestInspectDistribution:
    """Tests for RadiographBalancer.inspect_distribution."""

    def test_returns_class_distribution_dataclass(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """inspect_distribution returns a ClassDistribution instance."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(balanced_dataset_dir)
        assert isinstance(result, ClassDistribution)

    def test_counts_correct_for_balanced_dataset(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """Balanced fixture has 10 normal and 10 pneumonia images."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(balanced_dataset_dir)
        assert result.class_counts["normal"] == 10
        assert result.class_counts["pneumonia"] == 10

    def test_counts_correct_for_imbalanced_dataset(
        self,
        imbalanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """Imbalanced fixture has 30 normal and 10 pneumonia images."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(imbalanced_dataset_dir)
        assert result.class_counts["normal"] == 30
        assert result.class_counts["pneumonia"] == 10

    def test_imbalance_ratio_is_one_for_balanced(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """A 1:1 dataset reports imbalance_ratio of 1.0."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(balanced_dataset_dir)
        assert result.imbalance_ratio == pytest.approx(1.0)

    def test_imbalance_ratio_is_three_for_3to1_dataset(
        self,
        imbalanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """30 normal / 10 pneumonia → imbalance_ratio=3.0."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(imbalanced_dataset_dir)
        assert result.imbalance_ratio == pytest.approx(3.0)

    def test_majority_class_identified_correctly(
        self,
        imbalanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """majority_class is 'normal' for the 30/10 fixture."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(imbalanced_dataset_dir)
        assert result.majority_class == "normal"

    def test_minority_class_identified_correctly(
        self,
        imbalanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """minority_class is 'pneumonia' for the 30/10 fixture."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(imbalanced_dataset_dir)
        assert result.minority_class == "pneumonia"

    def test_total_images_equals_sum_of_counts(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
    ) -> None:
        """total_images equals sum of individual class counts."""
        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(balanced_dataset_dir)
        expected = sum(result.class_counts.values())
        assert result.total_images == expected

    def test_raises_file_not_found_for_empty_directory(
        self,
        tmp_path: Path,
        settings: Settings,
    ) -> None:
        """An empty directory raises FileNotFoundError."""
        balancer = RadiographBalancer(settings)
        with pytest.raises(FileNotFoundError):
            balancer.inspect_distribution(tmp_path)

    def test_handles_uppercase_folder_names(
        self,
        tmp_path: Path,
        settings: Settings,
    ) -> None:
        """NORMAL and PNEUMONIA (uppercase) folder names are accepted."""
        (tmp_path / "NORMAL").mkdir()
        (tmp_path / "PNEUMONIA").mkdir()

        for i in range(5):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(tmp_path / "NORMAL" / f"n_{i}.png")
            img.save(tmp_path / "PNEUMONIA" / f"p_{i}.png")

        balancer = RadiographBalancer(settings)
        result = balancer.inspect_distribution(tmp_path)
        assert result.class_counts["normal"] == 5
        assert result.class_counts["pneumonia"] == 5


# -------------------------------------------------- equalize_and_split


class TestEqualizeAndSplit:
    """Tests for RadiographBalancer.equalize_and_split."""

    def test_creates_output_directory_structure(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Output contains train/, val/, and test/ subdirectories."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        balancer.equalize_and_split(balanced_dataset_dir, out_dir)

        assert (out_dir / "train").is_dir()
        assert (out_dir / "val").is_dir()
        assert (out_dir / "test").is_dir()

    def test_returns_partition_summary(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """equalize_and_split returns a PartitionSummary dataclass."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        result = balancer.equalize_and_split(balanced_dataset_dir, out_dir)
        assert isinstance(result, PartitionSummary)

    def test_output_is_balanced_after_equalization(
        self,
        imbalanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """After equalisation both classes have equal sample counts."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        summary = balancer.equalize_and_split(imbalanced_dataset_dir, out_dir)
        # Across all splits, normal == pneumonia counts
        total_normal = (
            summary.train_counts.get("normal", 0)
            + summary.validation_counts.get("normal", 0)
            + summary.test_counts.get("normal", 0)
        )
        total_pneumonia = (
            summary.train_counts.get("pneumonia", 0)
            + summary.validation_counts.get("pneumonia", 0)
            + summary.test_counts.get("pneumonia", 0)
        )
        assert total_normal == total_pneumonia

    def test_total_images_consistent_with_partition_summary(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """PartitionSummary.total_images matches sum of all split counts."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        summary = balancer.equalize_and_split(balanced_dataset_dir, out_dir)
        computed_total = (
            sum(summary.train_counts.values())
            + sum(summary.validation_counts.values())
            + sum(summary.test_counts.values())
        )
        assert summary.total_images == computed_total

    def test_split_fractions_approximately_correct(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Training set is approximately train_fraction of total."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        summary = balancer.equalize_and_split(balanced_dataset_dir, out_dir)
        train_total = sum(summary.train_counts.values())
        # Allow ±2 images tolerance for integer rounding
        expected_min = int(summary.total_images * settings.train_fraction - 2)
        expected_max = int(summary.total_images * settings.train_fraction + 2)
        assert expected_min <= train_total <= expected_max

    def test_split_directories_contain_class_subfolders(
        self,
        balanced_dataset_dir: Path,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Each split subdirectory contains normal/ and pneumonia/ folders."""
        out_dir = tmp_path / "output"
        balancer = RadiographBalancer(settings)
        balancer.equalize_and_split(balanced_dataset_dir, out_dir)

        for split in ("train", "val", "test"):
            assert (out_dir / split / "normal").is_dir()
            assert (out_dir / split / "pneumonia").is_dir()

    def test_images_are_copied_not_moved(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Source directory is unchanged after equalize_and_split."""
        # Build a standalone src dir (not a subdir of out_dir)
        # to avoid rglob cross-contamination.
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"

        for cls in ("normal", "pneumonia"):
            (src_dir / cls).mkdir(parents=True)
            for i in range(10):
                arr = np.zeros((64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(src_dir / cls / f"img_{i:03d}.png")

        original_count = sum(1 for _ in src_dir.rglob("*.png"))
        balancer = RadiographBalancer(settings)
        balancer.equalize_and_split(src_dir, out_dir)

        remaining_count = sum(1 for _ in src_dir.rglob("*.png"))
        assert remaining_count == original_count
