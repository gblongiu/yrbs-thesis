from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import calibration_slope_intercept


def _safe_metric_values(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    out = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
        "calibration_slope": np.nan,
        "calibration_intercept": np.nan,
    }
    if y.size == 0:
        return out

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out["brier"] = float(brier_score_loss(y, p))
    if np.unique(y).size >= 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        slope, intercept = calibration_slope_intercept(y, p)
        out["calibration_slope"] = float(slope)
        out["calibration_intercept"] = float(intercept)
    return out


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    if idx_pos.size == 0 or idx_neg.size == 0:
        return rng.integers(0, y.size, size=y.size, endpoint=False)

    s_pos = idx_pos[rng.integers(0, idx_pos.size, size=idx_pos.size, endpoint=False)]
    s_neg = idx_neg[rng.integers(0, idx_neg.size, size=idx_neg.size, endpoint=False)]
    s = np.concatenate([s_pos, s_neg])
    rng.shuffle(s)
    return s


def stratified_bootstrap_metric_draws(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if n_boot <= 0:
        return pd.DataFrame(columns=["iter", "roc_auc", "pr_auc", "brier", "calibration_slope", "calibration_intercept"])
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_boot):
        idx = _stratified_bootstrap_indices(y, rng)
        m = _safe_metric_values(y[idx], p[idx])
        rows.append({"iter": i, **m})
    return pd.DataFrame(rows)


def summarize_bootstrap_ci(
    draws: pd.DataFrame,
    *,
    alpha: float = 0.05,
    metrics: Iterable[str] = ("roc_auc", "pr_auc", "brier", "calibration_slope", "calibration_intercept"),
) -> Dict[str, Tuple[float, float]]:
    if draws.empty:
        return {m: (np.nan, np.nan) for m in metrics}
    lo = float(100.0 * (alpha / 2.0))
    hi = float(100.0 * (1.0 - alpha / 2.0))
    out: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        vals = draws[m].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            out[m] = (np.nan, np.nan)
        else:
            out[m] = (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))
    return out
