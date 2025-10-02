"""DiagnosticVisualizer: generate all evaluation plots to disk."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog

from radiolens.evaluation.metrics import ClinicalMetricsReport

log = structlog.get_logger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualisations. "
        "Install it with: pip install matplotlib"
    ) from exc


class DiagnosticVisualizer:
    """Save publication-quality diagnostic plots to an output directory.

    All plots are saved as PNG files using ``FIGURE_DPI`` resolution.
    The output directory is created on instantiation if it does not exist.

    Args:
        output_dir: Directory where plot files are written.

    Example:
        >>> viz = DiagnosticVisualizer(Path("results/plots"))
        >>> viz.roc_curve_plot(fpr, tpr, auc_score=0.988)
    """

    FIGURE_DPI: ClassVar[int] = 150
    FIGURE_SIZE: ClassVar[tuple[int, int]] = (8, 6)

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- Individual plots

    def confusion_matrix_plot(
        self,
        confusion: np.ndarray,
        class_labels: list[str],
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Save an annotated confusion matrix with percentage labels.

        Args:
            confusion: 2×2 NumPy array ``[[TN, FP], [FN, TP]]``.
            class_labels: List of two class names (negative, positive).
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        total = confusion.sum()
        im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(class_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_labels)

        thresh = confusion.max() / 2.0
        for row in range(confusion.shape[0]):
            for col in range(confusion.shape[1]):
                count = confusion[row, col]
                pct = 100.0 * count / total if total > 0 else 0.0
                color = "white" if count > thresh else "black"
                ax.text(
                    col,
                    row,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def roc_curve_plot(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        filename: str = "roc_curve.png",
    ) -> Path:
        """Save an ROC curve with AUC annotation.

        Args:
            fpr: False positive rate array.
            tpr: True positive rate array.
            auc_score: Scalar AUC value for legend annotation.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        ax.plot(
            fpr,
            tpr,
            color="steelblue",
            lw=2,
            label=f"ROC curve (AUC = {auc_score:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="grey",
            lw=1,
            linestyle="--",
            label="Random classifier",
        )
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Receiver Operating Characteristic Curve", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def precision_recall_plot(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        average_precision: float,
        filename: str = "precision_recall.png",
    ) -> Path:
        """Save a precision-recall curve.

        Args:
            precision: Precision values at each threshold.
            recall: Recall values at each threshold.
            average_precision: Scalar AP score for annotation.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        ax.step(
            recall,
            precision,
            color="darkorange",
            lw=2,
            where="post",
            label=f"AP = {average_precision:.3f}",
        )
        ax.fill_between(
            recall,
            precision,
            alpha=0.15,
            color="darkorange",
            step="post",
        )
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def calibration_plot(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        filename: str = "calibration.png",
    ) -> Path:
        """Save a reliability diagram (predicted probability vs actual rate).

        Args:
            y_true: Ground-truth binary labels.
            y_proba: Predicted positive-class probabilities.
            n_bins: Number of bins for the reliability diagram.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_midpoints: list[float] = []
        fraction_positives: list[float] = []

        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_midpoints.append(
                    float((bin_edges[i] + bin_edges[i + 1]) / 2)
                )
                fraction_positives.append(float(y_true[mask].mean()))

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="grey",
            lw=1,
            label="Perfect calibration",
        )
        ax.plot(
            bin_midpoints,
            fraction_positives,
            marker="o",
            color="steelblue",
            lw=2,
            label="Model",
        )
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=14)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def confidence_distribution_plot(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        filename: str = "confidence_distribution.png",
    ) -> Path:
        """Save overlapping confidence histograms for each class.

        Args:
            y_proba: Predicted positive-class probabilities.
            y_true: Ground-truth binary labels.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        bins = np.linspace(0, 1, 30)
        bins_list = bins.tolist()
        ax.hist(
            y_proba[y_true == 0],
            bins=bins_list,
            alpha=0.6,
            color="steelblue",
            label="Normal",
            density=True,
        )
        ax.hist(
            y_proba[y_true == 1],
            bins=bins_list,
            alpha=0.6,
            color="tomato",
            label="Pneumonia",
            density=True,
        )
        ax.axvline(
            0.5,
            color="black",
            linestyle="--",
            lw=1.5,
            label="Threshold (0.5)",
        )
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Confidence Score Distribution by Class", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def class_balance_plot(
        self,
        class_counts: dict[str, int],
        filename: str = "class_distribution.png",
    ) -> Path:
        """Save a side-by-side bar chart and pie chart for class distribution.

        Args:
            class_counts: Mapping of class name to sample count.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        labels = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ["steelblue", "tomato", "seagreen", "gold"][: len(labels)]

        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(12, 5), dpi=self.FIGURE_DPI
        )

        bars = ax_bar.bar(labels, counts, color=colors, edgecolor="white")
        for bar, count in zip(bars, counts, strict=False):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax_bar.set_ylabel("Sample Count", fontsize=12)
        ax_bar.set_title("Class Distribution (Bar)", fontsize=13)
        ax_bar.grid(axis="y", alpha=0.3)

        ax_pie.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax_pie.set_title("Class Distribution (Pie)", fontsize=13)

        fig.tight_layout()
        return self._save_figure(fig, filename)

    def performance_comparison_plot(
        self,
        internal_metrics: ClinicalMetricsReport,
        external_metrics: ClinicalMetricsReport,
        metric_names: list[str] | None = None,
        filename: str = "performance_comparison.png",
    ) -> Path:
        """Save a grouped bar chart comparing internal vs external validation.

        Args:
            internal_metrics: Metrics from the internal test set.
            external_metrics: Metrics from the cross-operator dataset.
            metric_names: Metric attribute names to compare.  Defaults to
                ``["accuracy", "sensitivity", "specificity", "roc_auc"]``.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        if metric_names is None:
            metric_names = [
                "accuracy",
                "sensitivity",
                "specificity",
                "roc_auc",
            ]

        internal_values = [getattr(internal_metrics, m) for m in metric_names]
        external_values = [getattr(external_metrics, m) for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        bars_int = ax.bar(
            x - width / 2,
            internal_values,
            width,
            label="Internal",
            color="steelblue",
            edgecolor="white",
        )
        bars_ext = ax.bar(
            x + width / 2,
            external_values,
            width,
            label="External",
            color="tomato",
            edgecolor="white",
        )

        for bar in (*bars_int, *bars_ext):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
        ax.set_ylim((0.0, 1.1))
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            "Internal vs External Validation Performance", fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def comprehensive_dashboard(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metrics: ClinicalMetricsReport,
        filename: str = "dashboard.png",
    ) -> Path:
        """Save a 2x3 subplot grid with all key diagnostic charts.

        The grid layout is::

            [Confusion Matrix] [ROC Curve]       [PR Curve]
            [Calibration]      [Conf. Dist.]     [Metrics Table]

        Args:
            y_true: Ground-truth binary labels.
            y_proba: Predicted positive-class probabilities.
            metrics: Pre-computed :class:`ClinicalMetricsReport`.
            filename: Output file name within ``output_dir``.

        Returns:
            Path to the saved PNG file.
        """
        from radiolens.evaluation.metrics import (
            _compute_confusion_elements,
        )

        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=self.FIGURE_DPI)

        # 1 — Confusion matrix
        confusion = np.array(
            [
                [metrics.true_negatives, metrics.false_positives],
                [metrics.false_negatives, metrics.true_positives],
            ]
        )
        ax = axes[0, 0]
        ax.imshow(confusion, cmap="Blues")
        labels = ["Normal", "Pneumonia"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
        total = confusion.sum()
        thresh = confusion.max() / 2
        for r in range(2):
            for c in range(2):
                cnt = confusion[r, c]
                pct = 100 * cnt / total if total else 0
                ax.text(
                    c,
                    r,
                    f"{cnt}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cnt > thresh else "black",
                    fontsize=9,
                )
        ax.set_title("Confusion Matrix", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # 2 — ROC curve (build from scratch)
        thresholds = np.sort(np.unique(y_proba))[::-1]
        tprs_list: list[float] = []
        fprs_list: list[float] = []
        for t in np.concatenate(([thresholds[0] + 1e-8], thresholds)):
            yp = (y_proba >= t).astype(int)
            tp, tn, fp, fn = _compute_confusion_elements(y_true, yp)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            tprs_list.append(sens)
            fprs_list.append(1.0 - spec)
        tprs_arr = np.array(tprs_list)
        fprs_arr = np.array(fprs_list)
        idx = np.argsort(fprs_arr)
        ax = axes[0, 1]
        ax.plot(
            fprs_arr[idx],
            tprs_arr[idx],
            "steelblue",
            lw=2,
            label=f"AUC={metrics.roc_auc:.3f}",
        )
        ax.plot([0, 1], [0, 1], "grey", lw=1, ls="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3 — PR curve
        prec_list: list[float] = []
        rec_list: list[float] = []
        for t in thresholds:
            yp = (y_proba >= t).astype(int)
            tp, _tn, fp, fn = _compute_confusion_elements(y_true, yp)
            prec_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            rec_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        prec_arr = np.array(prec_list)
        rec_arr = np.array(rec_list)
        idx2 = np.argsort(rec_arr)
        ax = axes[0, 2]
        ax.step(
            rec_arr[idx2],
            prec_arr[idx2],
            "darkorange",
            lw=2,
            where="post",
            label=f"AP={metrics.pr_auc:.3f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 4 — Calibration
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bm: list[float] = []
        fp_frac: list[float] = []
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bm.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
                fp_frac.append(float(y_true[mask].mean()))
        ax = axes[1, 0]
        ax.plot([0, 1], [0, 1], "grey", ls="--", lw=1, label="Perfect")
        ax.plot(bm, fp_frac, "o-", color="steelblue", lw=2, label="Model")
        ax.set_xlabel("Mean Predicted Prob.")
        ax.set_ylabel("Fraction Positive")
        ax.set_title("Calibration Plot", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 5 — Confidence distribution
        bins = np.linspace(0, 1, 25)
        bins_list = bins.tolist()
        ax = axes[1, 1]
        ax.hist(
            y_proba[y_true == 0],
            bins=bins_list,
            alpha=0.6,
            color="steelblue",
            label="Normal",
            density=True,
        )
        ax.hist(
            y_proba[y_true == 1],
            bins=bins_list,
            alpha=0.6,
            color="tomato",
            label="Pneumonia",
            density=True,
        )
        ax.axvline(0.5, color="black", ls="--", lw=1.5)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Confidence Distribution", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 6 — Metrics table
        ax = axes[1, 2]
        ax.axis("off")
        metric_rows = [
            ["Accuracy", f"{metrics.accuracy:.4f}"],
            ["Sensitivity", f"{metrics.sensitivity:.4f}"],
            ["Specificity", f"{metrics.specificity:.4f}"],
            ["Precision", f"{metrics.precision:.4f}"],
            ["NPV", f"{metrics.negative_predictive_value:.4f}"],
            ["F1 Score", f"{metrics.f1_score:.4f}"],
            ["ROC AUC", f"{metrics.roc_auc:.4f}"],
            ["PR AUC", f"{metrics.pr_auc:.4f}"],
            ["MCC", f"{metrics.matthews_correlation:.4f}"],
            ["Kappa", f"{metrics.cohens_kappa:.4f}"],
        ]
        table = ax.table(
            cellText=metric_rows,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        ax.set_title("Key Metrics", fontsize=11)

        fig.suptitle(
            "radiolens — Comprehensive Diagnostic Dashboard",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return self._save_figure(fig, filename)

    # --------------------------------------------------------- Private helpers

    def _save_figure(self, fig: Figure, filename: str) -> Path:
        """Save a matplotlib figure to the output directory and close it.

        Args:
            fig: The matplotlib Figure to save.
            filename: File name (relative to ``output_dir``).

        Returns:
            Absolute path to the saved file.
        """
        output_path = self._output_dir / filename
        fig.savefig(str(output_path), dpi=self.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("plot_saved", path=str(output_path))
        return output_path
