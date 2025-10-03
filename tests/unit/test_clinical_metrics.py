"""Unit tests for clinical performance metrics computation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radiolens.evaluation.metrics import (
    ClinicalMetricsReport,
    _compute_f1,
    _compute_matthews,
    _compute_npv,
    _compute_precision,
    _compute_sensitivity,
    _compute_specificity,
    compute_binary_metrics,
    report_to_dict,
)

# --------------------------------------------------------- Sensitivity helper


class TestSensitivityHelper:
    """Tests for _compute_sensitivity."""

    def test_perfect_recall_when_no_false_negatives(self) -> None:
        """tp=10, fn=0 yields sensitivity of 1.0."""
        assert _compute_sensitivity(10, 0) == pytest.approx(1.0)

    def test_zero_recall_when_all_false_negatives(self) -> None:
        """tp=0, fn=10 yields sensitivity of 0.0."""
        assert _compute_sensitivity(0, 10) == pytest.approx(0.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """tp=0, fn=0 (no positives) returns 0.0 safely."""
        assert _compute_sensitivity(0, 0) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """tp=3, fn=1 yields 3/(3+1)=0.75."""
        assert _compute_sensitivity(3, 1) == pytest.approx(0.75)


# -------------------------------------------------------- Specificity helper


class TestSpecificityHelper:
    """Tests for _compute_specificity."""

    def test_perfect_specificity(self) -> None:
        """tn=10, fp=0 yields specificity of 1.0."""
        assert _compute_specificity(10, 0) == pytest.approx(1.0)

    def test_zero_specificity(self) -> None:
        """tn=0, fp=10 yields specificity of 0.0."""
        assert _compute_specificity(0, 10) == pytest.approx(0.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """tn=0, fp=0 (no negatives) returns 0.0 safely."""
        assert _compute_specificity(0, 0) == pytest.approx(0.0)

    def test_partial_specificity(self) -> None:
        """tn=7, fp=3 yields 7/10=0.70."""
        assert _compute_specificity(7, 3) == pytest.approx(0.70)


# ---------------------------------------------------------- Precision helper


class TestPrecisionHelper:
    """Tests for _compute_precision."""

    def test_perfect_precision(self) -> None:
        """tp=10, fp=0 yields precision of 1.0."""
        assert _compute_precision(10, 0) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        """tp=0, fp=10 yields precision of 0.0."""
        assert _compute_precision(0, 10) == pytest.approx(0.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """tp=0, fp=0 (no predicted positives) returns 0.0 safely."""
        assert _compute_precision(0, 0) == pytest.approx(0.0)

    def test_partial_precision(self) -> None:
        """tp=4, fp=1 yields 4/5=0.80."""
        assert _compute_precision(4, 1) == pytest.approx(0.80)


# ------------------------------------------------------------- NPV helper


class TestNPVHelper:
    """Tests for _compute_npv."""

    def test_perfect_npv(self) -> None:
        """tn=10, fn=0 yields NPV of 1.0."""
        assert _compute_npv(10, 0) == pytest.approx(1.0)

    def test_zero_npv(self) -> None:
        """tn=0, fn=10 yields NPV of 0.0."""
        assert _compute_npv(0, 10) == pytest.approx(0.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """tn=0, fn=0 (no predicted negatives) returns 0.0 safely."""
        assert _compute_npv(0, 0) == pytest.approx(0.0)

    def test_partial_npv(self) -> None:
        """tn=9, fn=1 yields 9/10=0.90."""
        assert _compute_npv(9, 1) == pytest.approx(0.90)


# --------------------------------------------------------------- F1 helper


class TestF1Helper:
    """Tests for _compute_f1."""

    def test_perfect_f1(self) -> None:
        """precision=1.0, sensitivity=1.0 yields F1=1.0."""
        assert _compute_f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_f1_when_both_zero(self) -> None:
        """precision=0.0, sensitivity=0.0 yields F1=0.0 safely."""
        assert _compute_f1(0.0, 0.0) == pytest.approx(0.0)

    def test_harmonic_mean_formula(self) -> None:
        """precision=0.8, sensitivity=0.6 yields 2*0.8*0.6/(0.8+0.6)."""
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert _compute_f1(0.8, 0.6) == pytest.approx(expected)

    def test_f1_with_zero_precision(self) -> None:
        """precision=0.0, sensitivity=1.0 yields F1=0.0."""
        assert _compute_f1(0.0, 1.0) == pytest.approx(0.0)


# ----------------------------------------------- Matthews correlation helper


class TestMatthewsCorrelation:
    """Tests for _compute_matthews."""

    def test_perfect_mcc_is_one(self) -> None:
        """Perfect predictions (tp=10, tn=10, fp=0, fn=0) → MCC=1.0."""
        assert _compute_matthews(10, 10, 0, 0) == pytest.approx(1.0)

    def test_all_wrong_is_negative_one(self) -> None:
        """All incorrect (tp=0, tn=0, fp=10, fn=10) → MCC=-1.0."""
        assert _compute_matthews(0, 0, 10, 10) == pytest.approx(-1.0)

    def test_zero_denominator_returns_zero(self) -> None:
        """All zeros → MCC=0.0 safely (denom_sq=0)."""
        assert _compute_matthews(0, 0, 0, 0) == pytest.approx(0.0)

    def test_mcc_in_valid_range(self) -> None:
        """Arbitrary confusion values stay within [-1, 1]."""
        mcc = _compute_matthews(8, 6, 4, 2)
        assert -1.0 <= mcc <= 1.0


# ----------------------------------------- compute_binary_metrics (main API)


class TestComputeBinaryMetrics:
    """Tests for the public compute_binary_metrics function."""

    def test_returns_clinical_metrics_report_instance(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """compute_binary_metrics returns a ClinicalMetricsReport."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)
        assert isinstance(report, ClinicalMetricsReport)

    def test_perfect_classifier_achieves_unit_accuracy(self) -> None:
        """A classifier that always predicts the correct class → 1.0."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9])
        report = compute_binary_metrics(y_true, y_proba)
        assert report.accuracy == pytest.approx(1.0)

    def test_perfect_classifier_achieves_unit_sensitivity(self) -> None:
        """Perfect classifier yields sensitivity=1.0."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9])
        report = compute_binary_metrics(y_true, y_proba)
        assert report.sensitivity == pytest.approx(1.0)

    def test_perfect_classifier_achieves_unit_specificity(self) -> None:
        """Perfect classifier yields specificity=1.0."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9])
        report = compute_binary_metrics(y_true, y_proba)
        assert report.specificity == pytest.approx(1.0)

    def test_all_metrics_within_valid_ranges(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """All [0,1]-valued metrics stay in range; MCC/kappa in [-1,1]."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)

        bounded_0_1 = [
            report.accuracy,
            report.sensitivity,
            report.specificity,
            report.precision,
            report.negative_predictive_value,
            report.false_positive_rate,
            report.false_negative_rate,
            report.f1_score,
            report.roc_auc,
            report.pr_auc,
        ]
        for value in bounded_0_1:
            assert 0.0 <= value <= 1.0

        assert -1.0 <= report.matthews_correlation <= 1.0
        assert -1.0 <= report.cohens_kappa <= 1.0

    def test_n_samples_matches_input_length(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """n_samples field equals len(y_true)."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)
        assert report.n_samples == len(y_true)

    def test_confusion_elements_sum_to_n_samples(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """TP + TN + FP + FN must equal total sample count."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)
        total = (
            report.true_positives
            + report.true_negatives
            + report.false_positives
            + report.false_negatives
        )
        assert total == report.n_samples

    def test_threshold_field_preserved(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The chosen threshold is echoed back in the report."""
        y_true, y_proba = binary_labels_and_scores
        threshold = 0.3
        report = compute_binary_metrics(y_true, y_proba, threshold=threshold)
        assert report.threshold == pytest.approx(threshold)

    def test_custom_threshold_higher_reduces_positives(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A higher threshold classifies fewer samples as positive."""
        y_true, y_proba = binary_labels_and_scores
        low = compute_binary_metrics(y_true, y_proba, threshold=0.2)
        high = compute_binary_metrics(y_true, y_proba, threshold=0.8)
        positives_low = low.true_positives + low.false_positives
        positives_high = high.true_positives + high.false_positives
        assert positives_high <= positives_low

    def test_raises_value_error_on_mismatched_lengths(self) -> None:
        """Mismatched array lengths raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="length"):
            compute_binary_metrics(y_true, y_proba)

    def test_raises_value_error_on_probabilities_above_one(self) -> None:
        """Probabilities above 1.0 raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.1, 1.5, 0.3])
        with pytest.raises(ValueError):
            compute_binary_metrics(y_true, y_proba)

    def test_raises_value_error_on_probabilities_below_zero(self) -> None:
        """Probabilities below 0.0 raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_proba = np.array([-0.1, 0.9, 0.3])
        with pytest.raises(ValueError):
            compute_binary_metrics(y_true, y_proba)

    def test_roc_auc_near_one_for_separable_data(self) -> None:
        """Nearly separable data yields ROC-AUC close to 1.0."""
        rng = np.random.default_rng(7)
        y_true = np.repeat([0, 1], 50)
        y_proba = np.concatenate(
            [
                rng.uniform(0.0, 0.2, 50),
                rng.uniform(0.8, 1.0, 50),
            ]
        )
        report = compute_binary_metrics(y_true, y_proba)
        assert report.roc_auc >= 0.95

    def test_sensitivity_matches_direct_computation(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """sensitivity field agrees with manual TP/(TP+FN) formula."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)
        tp = report.true_positives
        fn = report.false_negatives
        expected = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert report.sensitivity == pytest.approx(expected)

    def test_specificity_matches_direct_computation(
        self,
        binary_labels_and_scores: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """specificity field agrees with manual TN/(TN+FP) formula."""
        y_true, y_proba = binary_labels_and_scores
        report = compute_binary_metrics(y_true, y_proba)
        tn = report.true_negatives
        fp = report.false_positives
        expected = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        assert report.specificity == pytest.approx(expected)


# ------------------------------------------------------ report_to_dict


class TestReportToDict:
    """Tests for report_to_dict serialisation helper."""

    def _make_report(self) -> ClinicalMetricsReport:
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        return compute_binary_metrics(y_true, y_proba)

    def test_returns_dict_type(self) -> None:
        """report_to_dict returns a plain dict."""
        result = report_to_dict(self._make_report())
        assert isinstance(result, dict)

    def test_contains_all_required_keys(self) -> None:
        """All expected metric keys are present in the output dict."""
        required_keys = {
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "negative_predictive_value",
            "false_positive_rate",
            "false_negative_rate",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "matthews_correlation",
            "cohens_kappa",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "threshold",
            "n_samples",
        }
        result = report_to_dict(self._make_report())
        assert required_keys.issubset(result.keys())

    def test_values_are_json_serialisable(self) -> None:
        """json.dumps should not raise on the returned dict."""
        result = report_to_dict(self._make_report())
        json.dumps(result)  # should not raise

    def test_integer_fields_are_python_ints(self) -> None:
        """TP, TN, FP, FN, n_samples should be plain Python ints."""
        result = report_to_dict(self._make_report())
        int_keys = [
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "n_samples",
        ]
        for key in int_keys:
            assert isinstance(result[key], int), (
                f"{key} should be int, got {type(result[key])}"
            )
