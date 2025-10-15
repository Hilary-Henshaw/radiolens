"""Pydantic v2 request/response contracts for the radiolens API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClassificationResponse(BaseModel):
    """Response body for the ``/classify`` endpoint.

    Attributes:
        label: Predicted class — ``"PNEUMONIA"`` or ``"NORMAL"``.
        probability: Raw sigmoid output in ``[0, 1]``.
        confidence: ``max(probability, 1 - probability)``.
        certainty_tier: ``"HIGH"``, ``"MODERATE"``, or ``"LOW"``.
        clinical_note: Mandatory disclaimer text.
        model_version: Semantic version of the deployed model.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    probability: float
    confidence: float
    certainty_tier: str
    clinical_note: str
    model_version: str


class HealthStatus(BaseModel):
    """Response body for the ``/health`` endpoint.

    Attributes:
        status: ``"healthy"`` when the model is loaded, ``"degraded"``
            otherwise.
        model_loaded: ``True`` when the classifier is ready.
        uptime_seconds: Seconds elapsed since the application started.
        version: Deployed application version string.
    """

    model_config = ConfigDict(frozen=True)

    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str


class ModelInfo(BaseModel):
    """Response body for the ``/info`` endpoint.

    Attributes:
        backbone: Name of the CNN backbone (e.g. ``"MobileNetV2"``).
        input_shape: Expected image dimensions ``[H, W, C]``.
        output_classes: List of output class labels.
        training_dataset: Description of the training data source.
        validation_approach: Description of the validation methodology.
        cross_operator_accuracy: Published cross-operator accuracy.
        cross_operator_sensitivity: Published cross-operator sensitivity.
        cross_operator_auc: Published cross-operator ROC-AUC.
    """

    model_config = ConfigDict(frozen=True)

    backbone: str
    input_shape: list[int]
    output_classes: list[str]
    training_dataset: str
    validation_approach: str
    cross_operator_accuracy: float
    cross_operator_sensitivity: float
    cross_operator_auc: float


class PerformanceStats(BaseModel):
    """Response body for the ``/performance`` endpoint.

    All metrics are sourced from the published cross-operator validation
    study.

    Attributes:
        internal_accuracy: Accuracy on the internal test set.
        internal_sensitivity: Sensitivity on the internal test set.
        internal_specificity: Specificity on the internal test set.
        internal_roc_auc: ROC-AUC on the internal test set.
        external_accuracy: Accuracy on the external (cross-operator) set.
        external_sensitivity: Sensitivity on the external set.
        external_specificity: Specificity on the external set.
        external_roc_auc: ROC-AUC on the external set.
        bootstrap_p_value: Two-sided bootstrap p-value for ΔAUC.
        bootstrap_ci_lower: Lower bound of 95 % CI for ΔAUC.
        bootstrap_ci_upper: Upper bound of 95 % CI for ΔAUC.
        n_external_samples: Number of samples in the external dataset.
    """

    model_config = ConfigDict(frozen=True)

    internal_accuracy: float
    internal_sensitivity: float
    internal_specificity: float
    internal_roc_auc: float
    external_accuracy: float
    external_sensitivity: float
    external_specificity: float
    external_roc_auc: float
    bootstrap_p_value: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    n_external_samples: int


class ErrorDetail(BaseModel):
    """Standard error response body.

    Attributes:
        error: Short error identifier.
        detail: Human-readable description of the error.
        status_code: HTTP status code.
    """

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str
    status_code: int
