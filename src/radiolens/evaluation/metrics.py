"""Clinical performance metrics for binary pneumonia classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from scipy.integrate import trapezoid as _trapezoid

log = structlog.get_logger(__name__)


@dataclass
class ClinicalMetricsReport:
    """Full suite of binary classification metrics for clinical evaluation.

    Attributes:
        accuracy: Overall fraction of correct predictions.
        sensitivity: True positive rate (recall).
        specificity: True negative rate.
        precision: Positive predictive value.
        negative_predictive_value: Negative predictive value.
        false_positive_rate: FP / (FP + TN).
        false_negative_rate: FN / (FN + TP).
        f1_score: Harmonic mean of precision and sensitivity.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the precision-recall curve.
        matthews_correlation: Matthews correlation coefficient.
        cohens_kappa: Cohen's kappa statistic.
        true_positives: Raw count of true positives.
        true_negatives: Raw count of true negatives.
        false_positives: Raw count of false positives.
        false_negatives: Raw count of false negatives.
        threshold: Decision threshold used to compute binary metrics.
        n_samples: Total number of samples evaluated.
    """

    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    matthews_correlation: float
    cohens_kappa: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    threshold: float
    n_samples: int


# -------------------------------------------------------------------- Helpers


def _validate_inputs(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> None:
    """Raise ValueError for mismatched or invalid inputs.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities.

    Raises:
        ValueError: If shapes mismatch or values are out of range.
    """
    if y_true.shape[0] != y_pred_proba.shape[0]:
        raise ValueError(
            f"y_true length {y_true.shape[0]} != "
            f"y_pred_proba length {y_pred_proba.shape[0]}"
        )
    if y_pred_proba.min() < 0.0 or y_pred_proba.max() > 1.0:
        raise ValueError(
            "y_pred_proba values must be in [0, 1]; "
            f"got min={y_pred_proba.min():.4f}, "
            f"max={y_pred_proba.max():.4f}"
        )


def _compute_confusion_elements(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) from binary label arrays.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).

    Returns:
        Tuple ``(true_positives, true_negatives, false_positives,
        false_negatives)``.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def _compute_sensitivity(tp: int, fn: int) -> float:
    """Compute sensitivity (recall / true positive rate).

    Args:
        tp: True positives.
        fn: False negatives.

    Returns:
        Sensitivity value in ``[0, 1]``.
    """
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def _compute_specificity(tn: int, fp: int) -> float:
    """Compute specificity (true negative rate).

    Args:
        tn: True negatives.
        fp: False positives.

    Returns:
        Specificity value in ``[0, 1]``.
    """
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def _compute_precision(tp: int, fp: int) -> float:
    """Compute precision (positive predictive value).

    Args:
        tp: True positives.
        fp: False positives.

    Returns:
        Precision value in ``[0, 1]``.
    """
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def _compute_npv(tn: int, fn: int) -> float:
    """Compute negative predictive value.

    Args:
        tn: True negatives.
        fn: False negatives.

    Returns:
        NPV value in ``[0, 1]``.
    """
    denom = tn + fn
    return tn / denom if denom > 0 else 0.0


def _compute_f1(precision: float, sensitivity: float) -> float:
    """Compute F1 score from precision and sensitivity.

    Args:
        precision: Positive predictive value.
        sensitivity: True positive rate.

    Returns:
        F1 score in ``[0, 1]``.
    """
    denom = precision + sensitivity
    return 2 * precision * sensitivity / denom if denom > 0 else 0.0


def _compute_matthews(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute the Matthews correlation coefficient (MCC).

    Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.

    Returns:
        MCC in ``[-1, 1]``.
    """
    numerator = float(tp * tn - fp * fn)
    denom_sq = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denom_sq**0.5) if denom_sq > 0 else 0.0


def _compute_cohens_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute Cohen's kappa statistic.

    Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Kappa in ``[-1, 1]``.
    """
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    p_observed = (tp + tn) / n
    p_pos = ((tp + fp) / n) * ((tp + fn) / n)
    p_neg = ((tn + fn) / n) * ((tn + fp) / n)
    p_expected = p_pos + p_neg
    denom = 1.0 - p_expected
    return (p_observed - p_expected) / denom if abs(denom) > 1e-12 else 0.0


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the area under the ROC curve via the trapezoidal rule.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability scores.

    Returns:
        ROC-AUC in ``[0, 1]``.
    """
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs: list[float] = []
    fprs: list[float] = []

    for t in np.concatenate(([thresholds[0] + 1e-8], thresholds)):
        y_pred = (y_score >= t).astype(int)
        tp, tn, fp, fn = _compute_confusion_elements(y_true, y_pred)
        tprs.append(_compute_sensitivity(tp, fn))
        fprs.append(1.0 - _compute_specificity(tn, fp))

    tprs_arr = np.array(tprs)
    fprs_arr = np.array(fprs)
    sorted_idx = np.argsort(fprs_arr)
    return float(_trapezoid(tprs_arr[sorted_idx], fprs_arr[sorted_idx]))


def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability scores.

    Returns:
        PR-AUC in ``[0, 1]``.
    """
    thresholds = np.sort(np.unique(y_score))[::-1]
    precisions: list[float] = []
    recalls: list[float] = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp, _tn, fp, fn = _compute_confusion_elements(y_true, y_pred)
        precisions.append(_compute_precision(tp, fp))
        recalls.append(_compute_sensitivity(tp, fn))

    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)
    sorted_idx = np.argsort(recalls_arr)
    return float(
        _trapezoid(precisions_arr[sorted_idx], recalls_arr[sorted_idx])
    )


# ------------------------------------------------------------------- Public


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> ClinicalMetricsReport:
    """Compute full clinical metric suite from model predictions.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Decision boundary for binarising probabilities.
        positive_label: Value of the positive class in ``y_true``.

    Returns:
        A fully populated :class:`ClinicalMetricsReport`.

    Raises:
        ValueError: If array lengths differ or probabilities are outside
            ``[0, 1]``.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_pred_proba, dtype=float)

    _validate_inputs(y_true_arr, y_score_arr)

    y_pred = (y_score_arr >= threshold).astype(int)
    tp, tn, fp, fn = _compute_confusion_elements(y_true_arr, y_pred)

    n = len(y_true_arr)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    sensitivity = _compute_sensitivity(tp, fn)
    specificity = _compute_specificity(tn, fp)
    precision = _compute_precision(tp, fp)
    npv = _compute_npv(tn, fn)
    fpr = 1.0 - specificity
    fnr = 1.0 - sensitivity
    f1 = _compute_f1(precision, sensitivity)
    mcc = _compute_matthews(tp, tn, fp, fn)
    kappa = _compute_cohens_kappa(tp, tn, fp, fn)
    roc_auc = _compute_roc_auc(y_true_arr, y_score_arr)
    pr_auc = _compute_pr_auc(y_true_arr, y_score_arr)

    report = ClinicalMetricsReport(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        negative_predictive_value=npv,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        f1_score=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        matthews_correlation=mcc,
        cohens_kappa=kappa,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        threshold=threshold,
        n_samples=n,
    )

    log.info(
        "metrics_computed",
        n_samples=n,
        accuracy=round(accuracy, 4),
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        roc_auc=round(roc_auc, 4),
    )
    return report


def report_to_dict(
    report: ClinicalMetricsReport,
) -> dict[str, float | int]:
    """Convert a :class:`ClinicalMetricsReport` to a JSON-serialisable dict.

    Args:
        report: The metrics report to serialise.

    Returns:
        Dictionary with all field names as keys.
    """
    return {
        "accuracy": report.accuracy,
        "sensitivity": report.sensitivity,
        "specificity": report.specificity,
        "precision": report.precision,
        "negative_predictive_value": report.negative_predictive_value,
        "false_positive_rate": report.false_positive_rate,
        "false_negative_rate": report.false_negative_rate,
        "f1_score": report.f1_score,
        "roc_auc": report.roc_auc,
        "pr_auc": report.pr_auc,
        "matthews_correlation": report.matthews_correlation,
        "cohens_kappa": report.cohens_kappa,
        "true_positives": report.true_positives,
        "true_negatives": report.true_negatives,
        "false_positives": report.false_positives,
        "false_negatives": report.false_negatives,
        "threshold": report.threshold,
        "n_samples": report.n_samples,
    }
