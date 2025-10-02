"""Bootstrap-based statistical significance testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from radiolens.config import Settings
from radiolens.evaluation.metrics import _compute_roc_auc

log = structlog.get_logger(__name__)


@dataclass
class BootstrapResult:
    """Result of a bootstrap generalisation assessment.

    Attributes:
        mean_auc: Mean AUC across all bootstrap resamples of the
            internal dataset.
        std_error: Standard deviation of bootstrap AUC estimates.
        ci_lower: Lower bound of the confidence interval for ΔAUC.
        ci_upper: Upper bound of the confidence interval for ΔAUC.
        p_value: Proportion of resamples where |ΔAUC| ≥ observed |ΔAUC|.
            Under H0: internal AUC == cross-operator AUC.
        n_resamples: Number of bootstrap resamples performed.
        ci_level: Confidence level used (e.g. 0.95).
    """

    mean_auc: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_resamples: int
    ci_level: float


class BootstrapValidator:
    """Assess generalisation via bootstrap resampling of ROC-AUC.

    Resamples both the internal and external datasets independently and
    builds a distribution of ΔAUC values.  The two-sided p-value is the
    fraction of resamples where |ΔAUC| ≥ observed |ΔAUC|.

    Args:
        settings: Runtime configuration instance supplying
            ``bootstrap_resamples`` and ``bootstrap_ci_level``.

    Example:
        >>> validator = BootstrapValidator(get_settings())
        >>> result = validator.assess_generalisation(
        ...     y_true_int, y_proba_int, y_true_ext, y_proba_ext
        ... )
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def assess_generalisation(
        self,
        y_true_internal: np.ndarray,
        y_proba_internal: np.ndarray,
        y_true_external: np.ndarray,
        y_proba_external: np.ndarray,
    ) -> BootstrapResult:
        """Bootstrap resample both datasets and compute the ΔAUC distribution.

        Uses ``settings.bootstrap_resamples`` iterations.  The random number
        generator is seeded with ``settings.random_seed`` for reproducibility.

        The confidence interval is computed over the bootstrap ΔAUC values at
        the ``bootstrap_ci_level`` coverage.

        Args:
            y_true_internal: Ground-truth labels for the internal test set.
            y_proba_internal: Predicted probabilities for the internal set.
            y_true_external: Ground-truth labels for the external dataset.
            y_proba_external: Predicted probabilities for the external dataset.

        Returns:
            A :class:`BootstrapResult` with AUC statistics and p-value.

        Raises:
            ValueError: If any input array is empty or lengths are mismatched.
        """
        s = self._settings

        if len(y_true_internal) != len(y_proba_internal):
            raise ValueError(
                "Internal y_true and y_proba must have the same length."
            )
        if len(y_true_external) != len(y_proba_external):
            raise ValueError(
                "External y_true and y_proba must have the same length."
            )
        if len(y_true_internal) == 0 or len(y_true_external) == 0:
            raise ValueError("Input arrays must not be empty.")

        rng = np.random.default_rng(s.random_seed)
        n_int = len(y_true_internal)
        n_ext = len(y_true_external)

        auc_int_obs = _compute_roc_auc(y_true_internal, y_proba_internal)
        auc_ext_obs = _compute_roc_auc(y_true_external, y_proba_external)
        observed_delta = abs(auc_int_obs - auc_ext_obs)

        bootstrap_auc_internal: list[float] = []
        delta_aucs: list[float] = []

        n_resamples = s.bootstrap_resamples

        for _ in range(n_resamples):
            idx_int = rng.integers(0, n_int, size=n_int)
            auc_int_b = _compute_roc_auc(
                y_true_internal[idx_int], y_proba_internal[idx_int]
            )

            idx_ext = rng.integers(0, n_ext, size=n_ext)
            auc_ext_b = _compute_roc_auc(
                y_true_external[idx_ext], y_proba_external[idx_ext]
            )

            bootstrap_auc_internal.append(auc_int_b)
            delta_aucs.append(abs(auc_int_b - auc_ext_b))

        auc_int_arr = np.array(bootstrap_auc_internal)
        delta_arr = np.array(delta_aucs)

        mean_auc = float(auc_int_arr.mean())
        std_error = float(auc_int_arr.std())

        alpha = 1.0 - s.bootstrap_ci_level
        ci_lower = float(np.percentile(delta_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(delta_arr, 100 * (1 - alpha / 2)))

        p_value = float(np.mean(delta_arr >= observed_delta))

        result = BootstrapResult(
            mean_auc=mean_auc,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_resamples=n_resamples,
            ci_level=s.bootstrap_ci_level,
        )

        log.info(
            "bootstrap_complete",
            n_resamples=n_resamples,
            mean_auc=round(mean_auc, 4),
            std_error=round(std_error, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            p_value=round(p_value, 4),
            observed_delta=round(observed_delta, 4),
        )
        return result
