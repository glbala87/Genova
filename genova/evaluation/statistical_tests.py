"""Statistical testing utilities for genomic model evaluation.

Provides bootstrap confidence intervals (BCa), paired comparison tests
(McNemar, paired t-test, Wilcoxon signed-rank), AUROC comparison
(DeLong), effect size measures (Cohen's d, Cliff's delta), and
multiple testing corrections (Bonferroni, Benjamini-Hochberg FDR).

Example::

    from genova.evaluation.statistical_tests import (
        bootstrap_ci,
        mcnemar_test,
        delong_test,
        cohens_d,
        fdr_correction,
    )

    ci_low, ci_high = bootstrap_ci(scores, labels, metric_fn=auroc)
    stat, p = mcnemar_test(preds_a, preds_b, labels)
    z, p = delong_test(scores_a, scores_b, labels)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (BCa)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    method: str = "bca",
) -> Tuple[float, float]:
    """Compute bootstrap confidence intervals for a metric.

    Supports percentile and BCa (bias-corrected and accelerated) methods.

    Args:
        scores: Predicted scores, shape ``(N,)``.
        labels: Ground truth labels, shape ``(N,)``.
        metric_fn: Callable ``(scores, labels) -> float``.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.
        method: ``"percentile"`` or ``"bca"``.

    Returns:
        Tuple of ``(ci_lower, ci_upper)``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    n = len(scores)
    rng = np.random.RandomState(seed)

    observed = metric_fn(scores, labels)

    # Bootstrap distribution
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[i] = metric_fn(scores[idx], labels[idx])

    if method == "percentile":
        alpha = (1.0 - confidence) / 2.0
        ci_lower = float(np.percentile(boot_stats, 100 * alpha))
        ci_upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
        return ci_lower, ci_upper

    # BCa method
    # Bias correction factor
    z0 = _norm_ppf(np.mean(boot_stats < observed))

    # Acceleration factor (jackknife)
    jackknife_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_scores = np.delete(scores, i)
        jack_labels = np.delete(labels, i)
        jackknife_stats[i] = metric_fn(jack_scores, jack_labels)

    jack_mean = jackknife_stats.mean()
    jack_diff = jack_mean - jackknife_stats
    numerator = (jack_diff ** 3).sum()
    denominator = 6.0 * ((jack_diff ** 2).sum()) ** 1.5
    a_hat = numerator / denominator if denominator != 0 else 0.0

    alpha = (1.0 - confidence) / 2.0
    z_alpha_low = _norm_ppf(alpha)
    z_alpha_high = _norm_ppf(1.0 - alpha)

    # Adjusted percentiles
    def _bca_percentile(z_alpha: float) -> float:
        num = z0 + z_alpha
        denom = 1.0 - a_hat * num
        if denom == 0:
            return z_alpha
        adjusted_z = z0 + num / denom
        return _norm_cdf(adjusted_z)

    p_lower = _bca_percentile(z_alpha_low)
    p_upper = _bca_percentile(z_alpha_high)

    ci_lower = float(np.percentile(boot_stats, 100 * max(0, min(p_lower, 1))))
    ci_upper = float(np.percentile(boot_stats, 100 * max(0, min(p_upper, 1))))

    return ci_lower, ci_upper


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (probit function).

    Uses the rational approximation from Abramowitz and Stegun.

    Args:
        p: Probability in ``(0, 1)``.

    Returns:
        Z-score corresponding to *p*.
    """
    p = max(1e-10, min(1.0 - 1e-10, p))
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    """Helper for _norm_ppf."""
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)


def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF.

    Args:
        z: Z-score.

    Returns:
        Probability.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """McNemar's test for paired binary classifier comparison.

    Tests whether two classifiers have the same error rate on paired
    samples.

    Args:
        predictions_a: Predicted probabilities from model A, shape ``(N,)``.
        predictions_b: Predicted probabilities from model B, shape ``(N,)``.
        labels: Binary ground truth labels, shape ``(N,)``.
        threshold: Decision threshold for converting probabilities to
            binary predictions.
        continuity_correction: Apply Edwards' continuity correction.

    Returns:
        Tuple of ``(chi2_statistic, p_value)``.
    """
    preds_a = (np.asarray(predictions_a).ravel() >= threshold).astype(int)
    preds_b = (np.asarray(predictions_b).ravel() >= threshold).astype(int)
    labels = np.asarray(labels).ravel().astype(int)

    correct_a = (preds_a == labels).astype(int)
    correct_b = (preds_b == labels).astype(int)

    # Contingency table
    # b = A correct, B wrong; c = A wrong, B correct
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())

    if b + c == 0:
        return 0.0, 1.0

    if continuity_correction:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    # 1 degree of freedom chi-squared p-value approximation
    p_value = _chi2_sf(chi2, df=1)

    return float(chi2), float(p_value)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function approximation.

    Args:
        x: Chi-squared statistic.
        df: Degrees of freedom.

    Returns:
        p-value (upper tail probability).
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return 2.0 * (1.0 - _norm_cdf(math.sqrt(x)))
    # General case: use Abramowitz & Stegun approximation
    k = df / 2.0
    return 1.0 - _regularized_gamma_p(k, x / 2.0)


def _regularized_gamma_p(a: float, x: float, max_iter: int = 200) -> float:
    """Regularized lower incomplete gamma function P(a, x).

    Uses the series expansion for small x.

    Args:
        a: Shape parameter.
        x: Upper limit.
        max_iter: Maximum iterations.

    Returns:
        P(a, x) value.
    """
    if x <= 0:
        return 0.0
    if x > a + 1:
        return 1.0 - _regularized_gamma_q(a, x, max_iter)

    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break

    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _regularized_gamma_q(a: float, x: float, max_iter: int = 200) -> float:
    """Regularized upper incomplete gamma function Q(a, x).

    Uses the continued fraction expansion.

    Args:
        a: Shape parameter.
        x: Lower limit.
        max_iter: Maximum iterations.

    Returns:
        Q(a, x) value.
    """
    f = 1e-30
    c = 1e-30
    d = 1.0 / (x + 1.0 - a)
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        bn = x + 2.0 * i + 1.0 - a
        d = an * d + bn
        if abs(d) < 1e-30:
            d = 1e-30
        c = bn + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break

    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


def paired_ttest(
    metrics_a: np.ndarray,
    metrics_b: np.ndarray,
) -> Tuple[float, float]:
    """Paired two-sided t-test for comparing metrics from two models.

    Appropriate when comparing per-fold or per-sample metrics from two
    models evaluated on the same data splits.

    Args:
        metrics_a: Per-fold/sample metric values from model A, shape ``(N,)``.
        metrics_b: Per-fold/sample metric values from model B, shape ``(N,)``.

    Returns:
        Tuple of ``(t_statistic, p_value)``.
    """
    a = np.asarray(metrics_a, dtype=np.float64).ravel()
    b = np.asarray(metrics_b, dtype=np.float64).ravel()

    if len(a) != len(b):
        raise ValueError(
            f"Arrays must have the same length: {len(a)} vs {len(b)}"
        )

    n = len(a)
    if n < 2:
        return 0.0, 1.0

    diffs = a - b
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)

    if std_diff == 0:
        return 0.0, 1.0 if mean_diff == 0 else 0.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # Two-sided p-value from t-distribution (approximation)
    p_value = 2.0 * _t_sf(abs(t_stat), df)

    return float(t_stat), float(p_value)


def _t_sf(t: float, df: int) -> float:
    """Survival function for the t-distribution.

    Uses a normal approximation for large df and the regularized
    incomplete beta function for smaller df.

    Args:
        t: t-statistic (positive).
        df: Degrees of freedom.

    Returns:
        Upper-tail probability.
    """
    if df > 100:
        return 1.0 - _norm_cdf(t)

    x = df / (df + t * t)
    return 0.5 * _regularized_beta(x, df / 2.0, 0.5)


def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses the continued fraction expansion.

    Args:
        x: Value in ``[0, 1]``.
        a: First shape parameter.
        b: Second shape parameter.
        max_iter: Maximum iterations.

    Returns:
        I_x(a, b) value.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry relation if needed
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a, max_iter)

    log_prefix = (
        a * math.log(x) + b * math.log(1.0 - x)
        - math.log(a)
        - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b)
    )
    prefix = math.exp(log_prefix)

    # Lentz's continued fraction
    f = 1e-30
    c = 1e-30
    d = 1.0 / (1.0 - (a + b) * x / (a + 1.0))
    if abs(d) < 1e-30:
        d = 1e-30
    h = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 1e-15:
            break

    return prefix * h


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_test(
    metrics_a: np.ndarray,
    metrics_b: np.ndarray,
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired metric comparison.

    Non-parametric alternative to the paired t-test.

    Args:
        metrics_a: Per-fold/sample metric values from model A, shape ``(N,)``.
        metrics_b: Per-fold/sample metric values from model B, shape ``(N,)``.

    Returns:
        Tuple of ``(W_statistic, p_value)``.
    """
    a = np.asarray(metrics_a, dtype=np.float64).ravel()
    b = np.asarray(metrics_b, dtype=np.float64).ravel()

    if len(a) != len(b):
        raise ValueError(
            f"Arrays must have same length: {len(a)} vs {len(b)}"
        )

    diffs = a - b

    # Remove zero differences
    nonzero_mask = diffs != 0
    diffs = diffs[nonzero_mask]
    n = len(diffs)

    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences
    abs_diffs = np.abs(diffs)
    order = np.argsort(abs_diffs)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Handle ties: average ranks
    sorted_abs = abs_diffs[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    # Signed ranks
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    W = min(w_plus, w_minus)

    # Normal approximation for n >= 10
    if n >= 10:
        mean_W = n * (n + 1) / 4.0
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if std_W == 0:
            return float(W), 1.0
        z = (W - mean_W) / std_W
        p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    else:
        # For very small n, use normal approximation as well
        # (exact tables would be ideal but add complexity)
        mean_W = n * (n + 1) / 4.0
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if std_W == 0:
            return float(W), 1.0
        z = (W - mean_W) / std_W
        p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return float(W), float(max(0.0, min(1.0, p_value)))


# ---------------------------------------------------------------------------
# DeLong test for AUROC comparison
# ---------------------------------------------------------------------------


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """DeLong test for comparing two AUROCs on the same dataset.

    Implements the DeLong et al. (1988) method for testing whether the
    AUROCs of two classifiers are significantly different.

    Args:
        scores_a: Predicted scores from model A, shape ``(N,)``.
        scores_b: Predicted scores from model B, shape ``(N,)``.
        labels: Binary ground truth labels (0 or 1), shape ``(N,)``.

    Returns:
        Tuple of ``(z_statistic, p_value)``.
    """
    labels = np.asarray(labels).ravel()
    scores_a = np.asarray(scores_a, dtype=np.float64).ravel()
    scores_b = np.asarray(scores_b, dtype=np.float64).ravel()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        logger.warning("DeLong test requires both classes; returning NaN.")
        return float("nan"), float("nan")

    def _placement_values(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute placement values for DeLong variance estimation."""
        v_pos = np.zeros(m, dtype=np.float64)
        v_neg = np.zeros(n, dtype=np.float64)

        for i, pi in enumerate(pos_idx):
            for j, nj in enumerate(neg_idx):
                if scores[pi] > scores[nj]:
                    v_pos[i] += 1.0
                    v_neg[j] += 1.0
                elif scores[pi] == scores[nj]:
                    v_pos[i] += 0.5
                    v_neg[j] += 0.5

        v_pos /= n
        v_neg /= m
        return v_pos, v_neg

    v10_a, v01_a = _placement_values(scores_a)
    v10_b, v01_b = _placement_values(scores_b)

    auc_a = float(v10_a.mean())
    auc_b = float(v10_b.mean())

    # Covariance matrix of (AUC_a, AUC_b)
    s10 = np.cov(np.stack([v10_a, v10_b])) / m if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.stack([v01_a, v01_b])) / n if n > 1 else np.zeros((2, 2))
    s = s10 + s01

    # Variance of difference
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return float(z), float(p_value)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation.

    Args:
        group_a: Values from group A, shape ``(N_a,)``.
        group_b: Values from group B, shape ``(N_b,)``.

    Returns:
        Cohen's d value.  Positive means group A > group B.
    """
    a = np.asarray(group_a, dtype=np.float64).ravel()
    b = np.asarray(group_b, dtype=np.float64).ravel()

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_diff = a.mean() - b.mean()

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_std = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    return float(mean_diff / pooled_std)


def cliffs_delta(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Compute Cliff's delta effect size.

    A non-parametric effect size measure based on the probability that a
    randomly selected observation from group A is larger than one from
    group B.

    Args:
        group_a: Values from group A, shape ``(N_a,)``.
        group_b: Values from group B, shape ``(N_b,)``.

    Returns:
        Cliff's delta in ``[-1, 1]``.  Positive means group A tends to
        be larger.
    """
    a = np.asarray(group_a, dtype=np.float64).ravel()
    b = np.asarray(group_b, dtype=np.float64).ravel()

    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0

    # Count dominance
    more = 0
    less = 0
    for ai in a:
        for bj in b:
            if ai > bj:
                more += 1
            elif ai < bj:
                less += 1

    return float((more - less) / (n_a * n_b))


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction for multiple comparisons.

    Args:
        p_values: Array of raw p-values, shape ``(M,)``.
        alpha: Family-wise error rate threshold.

    Returns:
        Tuple of ``(adjusted_p_values, rejected)`` where *rejected* is
        a boolean array indicating which hypotheses are rejected at
        level *alpha*.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p)

    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted < alpha

    return adjusted, rejected


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Controls the false discovery rate at level *alpha*.

    Args:
        p_values: Array of raw p-values, shape ``(M,)``.
        alpha: Desired FDR level.

    Returns:
        Tuple of ``(adjusted_p_values, rejected)`` where *rejected* is
        a boolean array indicating which hypotheses are rejected at
        FDR level *alpha*.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p)

    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort p-values
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH adjusted p-values
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[sorted_idx[-1]] = sorted_p[-1]

    # Work backwards to enforce monotonicity
    for i in range(m - 2, -1, -1):
        rank = i + 1  # 1-based rank
        bh_val = sorted_p[i] * m / rank
        adjusted[sorted_idx[i]] = min(bh_val, adjusted[sorted_idx[i + 1]])

    adjusted = np.minimum(adjusted, 1.0)
    rejected = adjusted < alpha

    return adjusted, rejected
