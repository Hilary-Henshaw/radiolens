"""Unit tests for ThoraxClassifier model construction and inference."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Keras/TF required; skip the entire module gracefully if absent.
pytest.importorskip(
    "keras",
    reason="TensorFlow/Keras not installed — skipping detector tests",
)

from radiolens.config import Settings  # noqa: E402
from radiolens.core.detector import (  # noqa: E402
    InferenceResult,
    ThoraxClassifier,
    _determine_certainty_tier,
)

# ----------------------------------------------- _determine_certainty_tier


class TestDetermineCertaintyTier:
    """Tests for the _determine_certainty_tier helper."""

    def test_high_tier_at_threshold(self) -> None:
        """Confidence of 0.85 (the HIGH boundary) → 'HIGH'."""
        assert _determine_certainty_tier(0.85) == "HIGH"

    def test_high_tier_above_threshold(self) -> None:
        """Confidence of 0.99 → 'HIGH'."""
        assert _determine_certainty_tier(0.99) == "HIGH"

    def test_moderate_tier(self) -> None:
        """Confidence of 0.75 → 'MODERATE'."""
        assert _determine_certainty_tier(0.75) == "MODERATE"

    def test_low_tier(self) -> None:
        """Confidence of 0.60 → 'LOW'."""
        assert _determine_certainty_tier(0.60) == "LOW"

    def test_boundary_moderate(self) -> None:
        """Exactly 0.70 (MODERATE lower bound) → 'MODERATE'."""
        assert _determine_certainty_tier(0.70) == "MODERATE"

    def test_just_below_moderate_is_low(self) -> None:
        """Confidence just below 0.70 → 'LOW'."""
        assert _determine_certainty_tier(0.699) == "LOW"

    def test_just_below_high_is_moderate(self) -> None:
        """Confidence just below 0.85 → 'MODERATE'."""
        assert _determine_certainty_tier(0.849) == "MODERATE"

    def test_all_tiers_are_valid_strings(self) -> None:
        """All returned values are members of the expected set."""
        valid = {"HIGH", "MODERATE", "LOW"}
        for confidence in [0.50, 0.65, 0.72, 0.85, 0.95, 1.0]:
            assert _determine_certainty_tier(confidence) in valid


# ----------------------------------------- ThoraxClassifier build (slow)


@pytest.mark.slow
class TestThoraxClassifierBuild:
    """Integration-level tests that build a real Keras model."""

    def test_build_creates_model(
        self,
        settings: Settings,
    ) -> None:
        """After build(), the internal _model attribute is not None."""
        clf = ThoraxClassifier(settings)
        clf.build()
        assert clf._model is not None

    def test_input_shape_matches_settings(
        self,
        settings: Settings,
    ) -> None:
        """input_shape property returns (H, W, C) from settings."""
        clf = ThoraxClassifier(settings)
        expected = (
            settings.image_height,
            settings.image_width,
            settings.image_channels,
        )
        assert clf.input_shape == expected

    def test_input_shape_before_build(
        self,
        settings: Settings,
    ) -> None:
        """input_shape property works even before build() is called."""
        clf = ThoraxClassifier(settings)
        h = settings.image_height
        w = settings.image_width
        c = settings.image_channels
        assert clf.input_shape == (h, w, c)

    def test_model_output_shape_is_one(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """Model predict on a single image returns shape (1, 1)."""
        clf = ThoraxClassifier(settings)
        clf.build()
        batch = np.expand_dims(synthetic_pixel_array, axis=0)
        output = clf._model.predict(batch, verbose=0)  # type: ignore
        assert output.shape == (1, 1)

    def test_save_creates_file(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """save() writes a .keras file to the given path."""
        clf = ThoraxClassifier(settings)
        clf.build()
        out_path = tmp_path / "model.keras"
        clf.save(out_path)
        assert out_path.exists()


# ----------------------------------------------- run_inference (slow)


@pytest.mark.slow
class TestRunInference:
    """Tests for ThoraxClassifier.run_inference with a real model."""

    def test_returns_inference_result(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """run_inference returns an InferenceResult dataclass."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        assert isinstance(result, InferenceResult)

    def test_label_is_pneumonia_or_normal(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """Predicted label is either 'PNEUMONIA' or 'NORMAL'."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        assert result.label in {"PNEUMONIA", "NORMAL"}

    def test_probability_in_zero_one_range(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """Raw sigmoid probability is in [0, 1]."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        assert 0.0 <= result.probability <= 1.0

    def test_confidence_gte_half(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """Confidence is max(p, 1-p) which is always >= 0.5."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        assert result.confidence >= 0.5

    def test_certainty_tier_is_valid_string(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """certainty_tier is one of the three valid tier strings."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        assert result.certainty_tier in {"HIGH", "MODERATE", "LOW"}

    def test_confidence_equals_max_of_probability_complement(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """confidence == max(probability, 1.0 - probability)."""
        clf = ThoraxClassifier(settings)
        clf.build()
        result = clf.run_inference(synthetic_pixel_array)
        expected = max(result.probability, 1.0 - result.probability)
        assert result.confidence == pytest.approx(expected)


# --------------------------------- run_inference validation (no slow mark)


class TestRunInferenceValidation:
    """Tests that validate input checks — model is mocked to skip TF."""

    def test_raises_runtime_error_if_not_built(
        self,
        settings: Settings,
    ) -> None:
        """run_inference raises RuntimeError if build() was never called."""
        clf = ThoraxClassifier(settings)
        # _model is None by default
        arr = np.zeros(
            (settings.image_height, settings.image_width, 3),
            dtype=np.float32,
        )
        with pytest.raises(RuntimeError, match="not initialised"):
            clf.run_inference(arr)

    def test_raises_value_error_for_wrong_shape(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """Wrong spatial shape raises ValueError."""
        clf = ThoraxClassifier(settings)
        clf._model = MagicMock()  # type: ignore[assignment]

        wrong_shape = np.zeros((32, 32, 3), dtype=np.float32)  # expected 64x64
        with pytest.raises(ValueError, match="shape"):
            clf.run_inference(wrong_shape)

    def test_raises_value_error_for_wrong_dtype(
        self,
        settings: Settings,
        synthetic_pixel_array: np.ndarray,
    ) -> None:
        """float64 dtype raises ValueError (float32 is required)."""
        clf = ThoraxClassifier(settings)
        clf._model = MagicMock()  # type: ignore[assignment]

        wrong_dtype = synthetic_pixel_array.astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            clf.run_inference(wrong_dtype)

    def test_raises_value_error_for_2d_input(
        self,
        settings: Settings,
    ) -> None:
        """A 2-D array (missing channel dim) raises ValueError."""
        clf = ThoraxClassifier(settings)
        clf._model = MagicMock()  # type: ignore[assignment]

        two_d = np.zeros(
            (settings.image_height, settings.image_width),
            dtype=np.float32,
        )
        with pytest.raises(ValueError):
            clf.run_inference(two_d)


# ------------------------------------------------------- load_weights


class TestLoadWeights:
    """Tests for ThoraxClassifier.load_weights."""

    def test_raises_file_not_found_for_missing_path(
        self,
        settings: Settings,
        tmp_path: Path,
    ) -> None:
        """load_weights raises FileNotFoundError for a missing file."""
        clf = ThoraxClassifier(settings)
        missing_path = tmp_path / "nonexistent.keras"
        with pytest.raises(FileNotFoundError):
            clf.load_weights(missing_path)
