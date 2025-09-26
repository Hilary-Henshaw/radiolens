"""CLI: evaluate a trained model on internal and/or external datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``run_evaluation``.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="run_evaluation",
        description=(
            "Evaluate a trained ThoraxClassifier on a test dataset and "
            "optionally on a cross-operator external dataset. Saves all "
            "diagnostic plots and a JSON metrics report to --output-dir."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        metavar="PATH",
        help="Path to the saved .keras or .h5 model weights file.",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        metavar="PATH",
        help=(
            "Test data directory containing class subfolders "
            "(e.g. normal/, pneumonia/)."
        ),
    )
    parser.add_argument(
        "--cross-op-dir",
        default=None,
        metavar="PATH",
        help=(
            "Optional cross-operator dataset for external validation. "
            "Must have the same folder structure as --test-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        metavar="PATH",
        help=(
            "Directory for saved metrics JSON and plots (default: ./results)."
        ),
    )
    return parser


def _collect_predictions(
    data_dir: Path,
    classifier: object,
    normalizer: object,
    settings: object,
) -> tuple[list[int], list[float]]:
    """Walk a class-labelled directory and collect ground-truth + predictions.

    Args:
        data_dir: Root directory with ``normal/`` and ``pneumonia/``
            subfolders.
        classifier: Loaded :class:`~radiolens.core.detector.ThoraxClassifier`.
        normalizer: :class:`~radiolens.core.preprocessor.ImageNormalizer`.
        settings: :class:`~radiolens.config.Settings`.

    Returns:
        Tuple ``(y_true, y_proba)`` as Python lists.
    """
    from radiolens.config import Settings
    from radiolens.core.detector import ThoraxClassifier
    from radiolens.core.preprocessor import ImageNormalizer

    assert isinstance(classifier, ThoraxClassifier)
    assert isinstance(normalizer, ImageNormalizer)
    assert isinstance(settings, Settings)

    label_map = {"normal": 0, "NORMAL": 0, "pneumonia": 1, "PNEUMONIA": 1}
    y_true: list[int] = []
    y_proba: list[float] = []

    for cls_folder in sorted(data_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        label = label_map.get(cls_folder.name)
        if label is None:
            continue
        for img_path in sorted(cls_folder.iterdir()):
            if img_path.suffix.lower() not in settings.accepted_image_suffixes:
                continue
            try:
                arr = normalizer.from_path(img_path)
                result = classifier.run_inference(arr)
                y_true.append(label)
                y_proba.append(result.probability)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                print(
                    f"  Skipping {img_path.name}: {exc}",
                    file=sys.stderr,
                )

    return y_true, y_proba


def main() -> None:
    """Entry point for ``run_evaluation``."""
    parser = _build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    cross_op_dir = Path(args.cross_op_dir) if args.cross_op_dir else None

    try:
        import numpy as np

        from radiolens.config import get_settings
        from radiolens.core.detector import ThoraxClassifier
        from radiolens.core.preprocessor import ImageNormalizer
        from radiolens.evaluation.metrics import (
            compute_binary_metrics,
            report_to_dict,
        )
        from radiolens.evaluation.significance import BootstrapValidator
        from radiolens.evaluation.visualizer import DiagnosticVisualizer

        settings = get_settings()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading model from: {model_path}")
        classifier = ThoraxClassifier(settings)
        classifier.load_weights(model_path)
        normalizer = ImageNormalizer(settings)

        # ------------------------------------------------ Internal evaluation
        print(f"Collecting internal predictions from: {test_dir}")
        y_true_int, y_proba_int = _collect_predictions(
            test_dir, classifier, normalizer, settings
        )
        if not y_true_int:
            raise ValueError(f"No valid images found in: {test_dir}")

        y_true_int_arr = np.array(y_true_int)
        y_proba_int_arr = np.array(y_proba_int)

        print(
            f"  {len(y_true_int)} samples "
            f"(positives: {sum(y_true_int)}, "
            f"negatives: {len(y_true_int) - sum(y_true_int)})"
        )

        internal_report = compute_binary_metrics(
            y_true_int_arr, y_proba_int_arr
        )

        viz = DiagnosticVisualizer(output_dir)

        # Save internal plots
        confusion = np.array(
            [
                [
                    internal_report.true_negatives,
                    internal_report.false_positives,
                ],
                [
                    internal_report.false_negatives,
                    internal_report.true_positives,
                ],
            ]
        )
        viz.confusion_matrix_plot(
            confusion,
            class_labels=["Normal", "Pneumonia"],
            filename="internal_confusion_matrix.png",
        )

        thresholds = np.sort(np.unique(y_proba_int_arr))[::-1]
        tprs_list: list[float] = []
        fprs_list: list[float] = []
        prec_list: list[float] = []
        rec_list: list[float] = []

        from radiolens.evaluation.metrics import _compute_confusion_elements

        for t in thresholds:
            yp = (y_proba_int_arr >= t).astype(int)
            tp, tn, fp, fn = _compute_confusion_elements(y_true_int_arr, yp)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            tprs_list.append(sens)
            fprs_list.append(1.0 - spec)
            prec_list.append(prec)
            rec_list.append(sens)

        viz.roc_curve_plot(
            fpr=np.array(fprs_list),
            tpr=np.array(tprs_list),
            auc_score=internal_report.roc_auc,
            filename="internal_roc_curve.png",
        )
        viz.precision_recall_plot(
            precision=np.array(prec_list),
            recall=np.array(rec_list),
            average_precision=internal_report.pr_auc,
            filename="internal_pr_curve.png",
        )
        viz.calibration_plot(
            y_true=y_true_int_arr,
            y_proba=y_proba_int_arr,
            filename="internal_calibration.png",
        )
        viz.confidence_distribution_plot(
            y_proba=y_proba_int_arr,
            y_true=y_true_int_arr,
            filename="internal_confidence_dist.png",
        )
        viz.comprehensive_dashboard(
            y_true=y_true_int_arr,
            y_proba=y_proba_int_arr,
            metrics=internal_report,
            filename="internal_dashboard.png",
        )

        results: dict[str, object] = {
            "internal": report_to_dict(internal_report)
        }

        print("\n--- Internal Metrics ---")
        for k, v in report_to_dict(internal_report).items():
            print(f"  {k:>35s}: {v}")

        # ------------------------------------------- External (optional)
        if cross_op_dir is not None:
            print(f"\nCollecting external predictions from: {cross_op_dir}")
            y_true_ext, y_proba_ext = _collect_predictions(
                cross_op_dir, classifier, normalizer, settings
            )
            if not y_true_ext:
                raise ValueError(f"No valid images found in: {cross_op_dir}")

            y_true_ext_arr = np.array(y_true_ext)
            y_proba_ext_arr = np.array(y_proba_ext)

            external_report = compute_binary_metrics(
                y_true_ext_arr, y_proba_ext_arr
            )

            ext_confusion = np.array(
                [
                    [
                        external_report.true_negatives,
                        external_report.false_positives,
                    ],
                    [
                        external_report.false_negatives,
                        external_report.true_positives,
                    ],
                ]
            )
            viz.confusion_matrix_plot(
                ext_confusion,
                class_labels=["Normal", "Pneumonia"],
                filename="external_confusion_matrix.png",
            )
            viz.performance_comparison_plot(
                internal_metrics=internal_report,
                external_metrics=external_report,
                filename="performance_comparison.png",
            )

            validator = BootstrapValidator(settings)
            bootstrap_result = validator.assess_generalisation(
                y_true_int_arr,
                y_proba_int_arr,
                y_true_ext_arr,
                y_proba_ext_arr,
            )

            results["external"] = report_to_dict(external_report)
            results["bootstrap"] = {
                "mean_auc": bootstrap_result.mean_auc,
                "std_error": bootstrap_result.std_error,
                "ci_lower": bootstrap_result.ci_lower,
                "ci_upper": bootstrap_result.ci_upper,
                "p_value": bootstrap_result.p_value,
                "n_resamples": bootstrap_result.n_resamples,
                "ci_level": bootstrap_result.ci_level,
            }

            print("\n--- External Metrics ---")
            for k, v in report_to_dict(external_report).items():
                print(f"  {k:>35s}: {v}")

            print("\n--- Bootstrap Statistics ---")
            print(f"  {'mean_auc':>20s}: {bootstrap_result.mean_auc:.4f}")
            print(f"  {'std_error':>20s}: {bootstrap_result.std_error:.4f}")
            print(f"  {'ci_lower':>20s}: {bootstrap_result.ci_lower:.4f}")
            print(f"  {'ci_upper':>20s}: {bootstrap_result.ci_upper:.4f}")
            print(f"  {'p_value':>20s}: {bootstrap_result.p_value:.4f}")

        # ------------------------------------------- Save JSON report
        metrics_json_path = output_dir / "metrics.json"
        with metrics_json_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        print(f"\nMetrics saved to: {metrics_json_path}")
        print(f"Plots saved to:   {output_dir}")
        print("Evaluation complete.")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
