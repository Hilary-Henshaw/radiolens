"""ThoraxClassifier: build, load, and run inference with MobileNetV2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from radiolens.config import Settings

log = structlog.get_logger(__name__)

# Lazy import to avoid mandatory TF import at package load time
try:
    import keras
    from keras import layers
    from keras.metrics import Precision, Recall
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow / Keras is required for radiolens. "
        "Install it with: pip install tensorflow"
    ) from exc

_HIGH_TIER_THRESHOLD: float = 0.85
_MODERATE_TIER_THRESHOLD: float = 0.70


def _determine_certainty_tier(confidence: float) -> str:
    """Map a confidence value to a human-readable certainty tier.

    Args:
        confidence: Value in [0.5, 1.0] representing max(prob, 1-prob).

    Returns:
        One of ``"HIGH"``, ``"MODERATE"``, or ``"LOW"``.
    """
    if confidence >= _HIGH_TIER_THRESHOLD:
        return "HIGH"
    if confidence >= _MODERATE_TIER_THRESHOLD:
        return "MODERATE"
    return "LOW"


@dataclass
class InferenceResult:
    """Output of a single-image forward pass through ThoraxClassifier.

    Attributes:
        label: Predicted class — ``"PNEUMONIA"`` or ``"NORMAL"``.
        probability: Raw sigmoid output in ``[0, 1]``.
        confidence: ``max(probability, 1 - probability)``.
        certainty_tier: ``"HIGH"`` ≥ 0.85, ``"MODERATE"`` ≥ 0.70,
            else ``"LOW"``.
    """

    label: str
    probability: float
    confidence: float
    certainty_tier: str


class ThoraxClassifier:
    """MobileNetV2-based binary classifier for pneumonia detection.

    Args:
        settings: Runtime configuration instance.

    Example:
        >>> clf = ThoraxClassifier(get_settings())
        >>> clf.build()
        >>> result = clf.run_inference(pixel_array)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: keras.Model | None = None

    # -------------------------------------------------------------- Build

    def build(self) -> None:
        """Construct MobileNetV2 backbone + classification head.

        Freezes all backbone weights and compiles with Adam,
        binary cross-entropy, accuracy, Precision, and Recall.
        Stores the compiled model in ``self._model``.
        """
        s = self._settings
        input_shape = (s.image_height, s.image_width, s.image_channels)

        base = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        base.trainable = False

        inputs = keras.Input(shape=input_shape)
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(s.first_dropout_rate)(x)
        x = layers.Dense(s.dense_layer_units, activation="relu")(x)
        x = layers.Dropout(s.second_dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=s.initial_learning_rate
            ),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )

        self._model = model
        log.info(
            "model_built",
            backbone=s.backbone_name,
            input_shape=input_shape,
            dense_units=s.dense_layer_units,
            trainable_params=model.count_params(),
        )

    # --------------------------------------------------------- Load weights

    def load_weights(self, weights_path: Path) -> None:
        """Load saved weights or a full saved model from disk.

        Builds the model first if ``self._model`` is ``None``.

        Args:
            weights_path: Path to a ``.keras`` or ``.h5`` file.

        Raises:
            FileNotFoundError: If ``weights_path`` does not exist.
        """
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {weights_path}"
            )

        suffix = weights_path.suffix.lower()

        if suffix == ".keras":
            self._model = keras.models.load_model(str(weights_path))
            log.info(
                "model_loaded",
                path=str(weights_path),
                format="keras",
            )
        elif suffix == ".h5":
            if self._model is None:
                self.build()
            self._model.load_weights(str(weights_path))  # type: ignore[union-attr]
            log.info(
                "model_loaded",
                path=str(weights_path),
                format="h5",
            )
        else:
            # Attempt generic load for unknown extensions
            if self._model is None:
                self.build()
            self._model.load_weights(str(weights_path))  # type: ignore[union-attr]
            log.warning(
                "model_loaded_unknown_extension",
                path=str(weights_path),
                suffix=suffix,
            )

    # ------------------------------------------------------------- Inference

    def run_inference(self, pixel_array: np.ndarray) -> InferenceResult:
        """Run single-image inference.

        Args:
            pixel_array: Float32 array of shape ``(H, W, 3)`` with
                values in ``[0, 1]``.

        Returns:
            An :class:`InferenceResult` with label, probability,
            confidence, and certainty tier.

        Raises:
            RuntimeError: If the model has not been built or loaded.
            ValueError: If ``pixel_array`` has an unexpected shape or dtype.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not initialised. Call build() or load_weights()."
            )

        s = self._settings
        expected_shape = (s.image_height, s.image_width, s.image_channels)

        if pixel_array.ndim != 3:
            raise ValueError(
                f"Expected 3-D array (H, W, C), got ndim={pixel_array.ndim}"
            )
        if pixel_array.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {pixel_array.shape}"
            )
        if pixel_array.dtype != np.float32:
            raise ValueError(
                f"Expected dtype float32, got {pixel_array.dtype}"
            )

        batch = np.expand_dims(pixel_array, axis=0)  # (1, H, W, C)
        raw_output: np.ndarray = self._model.predict(batch, verbose=0)
        probability = float(raw_output[0, 0])

        label = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        confidence = max(probability, 1.0 - probability)
        certainty_tier = _determine_certainty_tier(confidence)

        log.debug(
            "inference_complete",
            label=label,
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            certainty_tier=certainty_tier,
        )
        return InferenceResult(
            label=label,
            probability=probability,
            confidence=confidence,
            certainty_tier=certainty_tier,
        )

    # ----------------------------------------------------------- Properties

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return ``(H, W, C)`` from settings.

        Returns:
            Tuple of (height, width, channels).
        """
        s = self._settings
        return (s.image_height, s.image_width, s.image_channels)

    # ----------------------------------------------------------------- Save

    def save(self, output_path: Path) -> None:
        """Save the full model to ``output_path`` in native Keras format.

        Args:
            output_path: Destination file path (should end in ``.keras``).

        Raises:
            RuntimeError: If the model has not been built.
        """
        if self._model is None:
            raise RuntimeError(
                "Nothing to save — call build() or load_weights() first."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(output_path))
        log.info("model_saved", path=str(output_path))
