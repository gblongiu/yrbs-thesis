from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib must be configured before importing pyplot.
_mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CALIBRATION_FINAL_STRATEGY,
    CALIBRATION_HOLDOUT_SIZE,
    CV_FOLDS,
    DATASET_VERSION,
    EXPERIMENT_NAMESPACE,
    EXPOSURE_COLS,
    FEATURES_BASELINE,
    FEATURES_FULL,
    MIN_GROUP_N,
    MIN_GROUP_NEG,
    MIN_GROUP_POS,
    POSITIVE_LABEL,
    PROCESSED_DIR,
    RANDOM_SEEDS,
    TARGET_COL,
    TEST_SIZE,
)
from src.evaluation.metrics import compute_binary_metrics  # noqa: E402
from src.evaluation.subgroup_adequacy import evaluate_subgroup_adequacy  # noqa: E402
from src.evaluation.bootstrap import (  # noqa: E402
    stratified_bootstrap_metric_draws,
    summarize_bootstrap_ci,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "no_vcs"
    except Exception:
        return "no_vcs"


def package_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def deterministic_run_id(seed: int, model: str, featureset: str, calibration: str) -> str:
    return f"{EXPERIMENT_NAMESPACE}_seed{seed}_{model}_{featureset}_{calibration}"


def assert_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in parquet: {missing}")


def assert_binary_target(y: pd.Series) -> None:
    if y.isna().any():
        raise SystemExit(f"Target column {TARGET_COL} contains missing values; expected fully observed after Week 2.")
    vals = set(y.unique().tolist())
    if not vals.issubset({0, 1}):
        raise SystemExit(f"Target column {TARGET_COL} must be binary {{0,1}}; observed values: {sorted(vals)}")


def build_preprocessor(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    # Treat listed categorical columns as categorical even if numeric-like.
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers = []
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))
    if num_cols:
        transformers.append(("num", numeric, num_cols))

    if not transformers:
        raise SystemExit("No features selected: feature set is empty.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_estimator(model: str, preprocessor: ColumnTransformer, seed: int) -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def _safe_clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)


def _platt_feature(y_prob: np.ndarray) -> np.ndarray:
    p = _safe_clip_probs(y_prob)
    logit = np.log(p / (1.0 - p))
    return logit.reshape(-1, 1)


def fit_posthoc_calibrator(method: str, y_prob_raw: np.ndarray, y_true: np.ndarray) -> Optional[dict]:
    if method == "none":
        return None
    y_true = np.asarray(y_true, dtype=int)
    y_prob_raw = np.asarray(y_prob_raw, dtype=float)
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    if method == "platt":
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
        model.fit(_platt_feature(y_prob_raw), y_true)
        return {"method": "platt", "model": model}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(_safe_clip_probs(y_prob_raw), y_true)
        return {"method": "isotonic", "model": model}
    raise ValueError(f"Unknown calibration method: {method}")


def apply_posthoc_calibrator(calibrator: Optional[dict], y_prob_raw: np.ndarray) -> np.ndarray:
    y_prob_raw = np.asarray(y_prob_raw, dtype=float)
    if calibrator is None:
        return y_prob_raw.copy()
    method = calibrator.get("method")
    model = calibrator.get("model")
    if method == "platt":
        y_prob = model.predict_proba(_platt_feature(y_prob_raw))[:, 1]
    elif method == "isotonic":
        y_prob = model.predict(_safe_clip_probs(y_prob_raw))
    else:
        raise ValueError(f"Unsupported calibrator method: {method}")
    return _safe_clip_probs(np.asarray(y_prob, dtype=float))


def predict_proba_positive(estimator: object, X: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba output has unexpected shape.")
    return proba[:, POSITIVE_LABEL]


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sha256_np_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def fold_probabilities_with_optional_calibration(
    *,
    estimator_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    calibration_method: str,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rows: List[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        base = estimator_factory()
        base.fit(X_tr, y_tr)
        y_prob_raw = predict_proba_positive(base, X_va)

        calibrator = fit_posthoc_calibrator(calibration_method, y_prob_raw, y_va.to_numpy(dtype=int))
        y_prob_cal = apply_posthoc_calibrator(calibrator, y_prob_raw)

        m_uncal = compute_binary_metrics(y_va.to_numpy(), y_prob_raw)
        m_cal = compute_binary_metrics(y_va.to_numpy(), y_prob_cal)
        rows.append(
            {
                "fold": fold,
                **{f"uncalibrated_{k}": float(v) for k, v in m_uncal.items()},
                **{f"calibrated_{k}": float(v) for k, v in m_cal.items()},
            }
        )

    return pd.DataFrame(rows).sort_values("fold", kind="mergesort").reset_index(drop=True)


def aggregate_fold_metrics(
    fold_df: pd.DataFrame, *, calibration_method: str
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    metric_cols = [c for c in fold_df.columns if c != "fold"]
    means = {f"{c}_mean": float(fold_df[c].mean()) for c in metric_cols}
    stds = {f"{c}_std": float(fold_df[c].std(ddof=1)) for c in metric_cols}

    legacy: Dict[str, float] = {}
    for m in ["roc_auc", "pr_auc", "brier", "calibration_slope", "calibration_intercept"]:
        base_key = f"uncalibrated_{m}"
        cal_key = f"calibrated_{m}"
        chosen = cal_key if calibration_method != "none" else base_key
        legacy[f"{m}_mean"] = means[f"{chosen}_mean"]
        legacy[f"{m}_std"] = stds[f"{chosen}_std"]
    return means, stds, legacy


def fit_cv_stacking_calibrator(
    *,
    estimator_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    calibration_method: str,
) -> Optional[dict]:
    if calibration_method == "none":
        return None
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(len(X_train), np.nan, dtype=float)
    y_arr = y_train.to_numpy(dtype=int)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        base = estimator_factory()
        base.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[va_idx] = predict_proba_positive(base, X_train.iloc[va_idx])
    if np.isnan(oof).any():
        raise RuntimeError("OOF probabilities contain NaN; cannot fit post-hoc calibrator.")
    return fit_posthoc_calibrator(calibration_method, oof, y_arr)


def fit_train_split_calibrator(
    *,
    estimator_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    calibration_method: str,
) -> Optional[dict]:
    if calibration_method == "none":
        return None
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=CALIBRATION_HOLDOUT_SIZE, random_state=seed + 17)
    tr_pos, cal_pos = next(splitter.split(X_train, y_train))
    base = estimator_factory()
    base.fit(X_train.iloc[tr_pos], y_train.iloc[tr_pos])
    y_prob_calfit = predict_proba_positive(base, X_train.iloc[cal_pos])
    return fit_posthoc_calibrator(calibration_method, y_prob_calfit, y_train.iloc[cal_pos].to_numpy(dtype=int))


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, int]:
    y_pred = (np.asarray(y_prob, dtype=float) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true, dtype=int), y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_calibration_from_curve_csv(csv_path: Path, out_path: Path, title: str, label: str) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal")
    ax.plot(df["mean_predicted"], df["fraction_positive"], marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compute_test_artifacts(
    *,
    y_test: pd.Series,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    out_tables: Path,
    out_figures: Path,
    model: str,
    featureset: str,
    seed: int,
    calibration_method: str,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    y_true = y_test.to_numpy(dtype=int)
    y_prob_uncal = _safe_clip_probs(y_prob_uncal)
    y_prob_cal = _safe_clip_probs(y_prob_cal)

    metrics_uncal = compute_binary_metrics(y_true, y_prob_uncal)
    metrics_cal = compute_binary_metrics(y_true, y_prob_cal)

    active_prob = y_prob_cal if calibration_method != "none" else y_prob_uncal
    cm = confusion_counts(y_true, active_prob)

    for label, arr in [("uncalibrated", y_prob_uncal), ("calibrated", y_prob_cal)]:
        frac_pos, mean_pred = calibration_curve(y_true, arr, n_bins=10, strategy="quantile")
        cal_df = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
        cal_csv = (
            out_tables
            / f"calibration_curve_test_{label}_{model}_{featureset}_seed{seed}_{calibration_method}.csv"
        )
        cal_df.to_csv(
            cal_csv,
            index=False,
        )
        plot_calibration_from_curve_csv(
            cal_csv,
            out_figures / f"calibration_test_{label}_{model}_{featureset}_seed{seed}_{calibration_method}.png",
            f"Calibration Curve (Test): {model}/{featureset}/{label}/seed={seed}",
            label,
        )

    # Keep legacy file names for downstream compatibility; reflect active probabilities.
    frac_pos_active, mean_pred_active = calibration_curve(y_true, active_prob, n_bins=10, strategy="quantile")
    legacy_cal_df = pd.DataFrame({"mean_predicted": mean_pred_active, "fraction_positive": frac_pos_active})
    legacy_csv = out_tables / f"calibration_curve_{model}_{featureset}_seed{seed}.csv"
    legacy_cal_df.to_csv(legacy_csv, index=False)
    plot_calibration_from_curve_csv(
        legacy_csv,
        out_figures / f"calibration_{model}_{featureset}_seed{seed}.png",
        f"Calibration Curve (Test): {model}/{featureset}/seed={seed}",
        "active",
    )

    # ROC curve (active probability stream for stable downstream filenames).
    fpr, tpr, _ = roc_curve(y_true, active_prob)
    roc_auc = roc_auc_score(y_true, active_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title(f"ROC Curve (Test): {model} / {featureset} / seed={seed}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_figures / f"roc_curve_test_{model}_{featureset}_seed{seed}.png", dpi=300)
    plt.close(fig)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, active_prob)
    pr_auc = average_precision_score(y_true, active_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"AP={pr_auc:.3f}")
    ax.set_title(f"PR Curve (Test): {model} / {featureset} / seed={seed}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_figures / f"pr_curve_test_{model}_{featureset}_seed{seed}.png", dpi=300)
    plt.close(fig)

    # Confusion matrix plot at threshold 0.5 on active probabilities.
    y_pred = (active_prob >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix @0.5 (Test): {model}/{featureset}")
    fig.tight_layout()
    fig.savefig(out_figures / f"confusion_matrix_test_{model}_{featureset}_seed{seed}.png", dpi=300)
    plt.close(fig)

    return metrics_uncal, cm, metrics_cal


def save_logreg_coefficients(
    pipeline: Pipeline, out_tables: Path, featureset: str, seed: int, run_id: str
) -> Path:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    coefs = model.coef_.ravel()
    if len(feature_names) != len(coefs):
        raise RuntimeError("Coefficient length does not match feature names.")

    cat_cols: List[str] = []
    num_cols: List[str] = []
    for name, _trans, cols in pre.transformers_:
        if name == "cat":
            cat_cols = list(cols)
        elif name == "num":
            num_cols = list(cols)

    rows = []
    for name, coef in zip(feature_names, coefs):
        transformed = ""
        orig_var = ""
        level = ""

        if name.startswith("num__"):
            transformed = "num"
            orig_var = name[len("num__") :]
            level = ""
        elif name.startswith("cat__"):
            transformed = "cat"
            remainder = name[len("cat__") :]
            # Determine the originating variable by matching known categorical feature names.
            match = next((c for c in cat_cols if remainder == c or remainder.startswith(c + "_")), None)
            orig_var = match or remainder
            if match and remainder.startswith(match + "_"):
                level = remainder[len(match) + 1 :]
            else:
                level = ""
        else:
            transformed = "unknown"
            orig_var = name
            level = ""

        rows.append(
            {
                "dataset_version": DATASET_VERSION,
                "run_id": run_id,
                "seed": seed,
                "featureset": featureset,
                "feature_name": name,
                "transform": transformed,
                "orig_variable": orig_var,
                "level": level,
                "coefficient": float(coef),
                "odds_ratio": float(np.exp(coef)),
            }
        )

    df = pd.DataFrame(rows)
    out_path = out_tables / f"logreg_coefficients_{featureset}_seed{seed}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_predictions_and_error_slices(
    df_all: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: pd.Series,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    active_prob: np.ndarray,
    calibration_method: str,
    model: str,
    featureset: str,
    seed: int,
    out_tables: Path,
    min_group_n: int,
    min_group_pos: int,
    min_group_neg: int,
) -> Tuple[Path, Path, Dict[str, List[str]]]:
    out_tables.mkdir(parents=True, exist_ok=True)

    y_true_arr = y_true.to_numpy(dtype=int)
    y_prob_uncal = np.asarray(y_prob_uncal, dtype=float)
    y_prob_cal = np.asarray(y_prob_cal, dtype=float)
    active_prob = np.asarray(active_prob, dtype=float)
    y_pred = (active_prob >= 0.5).astype(int)

    test_rows = df_all.loc[test_idx, :].copy()
    preds = pd.DataFrame(
        {
            "row_index": test_rows.index.astype(int),
            "y_true": y_true_arr,
            "y_prob": active_prob,
            "y_prob_uncalibrated": y_prob_uncal,
            "y_prob_calibrated": y_prob_cal,
            "calibration_method": calibration_method,
            "y_pred_0p5": y_pred,
        }
    )

    # Optional feature columns for downstream subgroup analysis.
    for col in ["q1", "q2", "q3", "raceeth"] + EXPOSURE_COLS:
        if col in test_rows.columns:
            preds[col] = test_rows[col].astype("string")

    preds_path = out_tables / f"preds_test_{model}_{featureset}_seed{seed}.csv"
    preds.to_csv(preds_path, index=False)

    # Error slices by raceeth and q1 (test set only), only groups with n >= MIN_GROUP_N.
    slices = []
    omitted: Dict[str, List[str]] = {"raceeth": [], "q1": []}
    for group_col in ["raceeth", "q1"]:
        if group_col not in test_rows.columns:
            continue
        gvals = test_rows[group_col].astype("string").fillna("<NA>")
        tmp = pd.DataFrame(
            {
                "group": gvals,
                "y_true": y_true_arr,
                "y_pred": y_pred,
                "y_prob": active_prob,
            }
        )
        for g, gdf in tmp.groupby("group", sort=False):
            adequacy = evaluate_subgroup_adequacy(
                gdf["y_true"].to_numpy(dtype=int),
                min_group_n=min_group_n,
                min_group_pos=min_group_pos,
                min_group_neg=min_group_neg,
            )
            if not adequacy.adequate:
                omitted[group_col].append(str(g))
                slices.append(
                    {
                        "grouping": group_col,
                        "group_value": str(g),
                        "n": adequacy.n,
                        "n_pos": adequacy.n_pos,
                        "n_neg": adequacy.n_neg,
                        "event_rate": adequacy.event_rate,
                        "adequacy_flag": False,
                        "reason": adequacy.reason,
                        "error_rate_0p5": np.nan,
                        "y_true_rate": np.nan,
                        "y_prob_mean": np.nan,
                    }
                )
                continue
            err = float((gdf["y_true"] != gdf["y_pred"]).mean())
            pos_rate = adequacy.event_rate
            mean_prob = float(gdf["y_prob"].mean())
            slices.append(
                {
                    "grouping": group_col,
                    "group_value": str(g),
                    "n": adequacy.n,
                    "n_pos": adequacy.n_pos,
                    "n_neg": adequacy.n_neg,
                    "event_rate": adequacy.event_rate,
                    "adequacy_flag": True,
                    "reason": "",
                    "error_rate_0p5": round(err, 6),
                    "y_true_rate": round(pos_rate, 6),
                    "y_prob_mean": round(mean_prob, 6),
                }
            )

    error_df = pd.DataFrame(slices)
    if not error_df.empty:
        error_df = error_df.sort_values(["grouping", "group_value"], kind="mergesort").reset_index(drop=True)
    error_path = out_tables / f"error_slices_{model}_{featureset}_seed{seed}.csv"
    error_df.to_csv(error_path, index=False)

    return preds_path, error_path, omitted


def write_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 modeling + evaluation (unweighted predictive protocol).")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for holdout split and CV folds.")
    parser.add_argument(
        "--allow-any-seed",
        action="store_true",
        help="Allow seeds not listed in src/config.py RANDOM_SEEDS (not recommended).",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional dev mode: head(n) rows deterministically.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory (default: outputs/).")
    parser.add_argument("--model", choices=["logreg", "hgb", "both"], default="both")
    parser.add_argument("--features", choices=["baseline", "full"], default="full")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument(
        "--calibration-final-strategy",
        choices=["cv_stacking", "train_split"],
        default=CALIBRATION_FINAL_STRATEGY,
        help="How to fit the final post-hoc calibrator on the training split.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id; otherwise deterministic.")
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of stratified bootstrap resamples for test-set confidence intervals.",
    )
    args = parser.parse_args()

    if (not args.allow_any_seed) and (args.seed not in RANDOM_SEEDS):
        raise SystemExit(f"--seed must be one of {RANDOM_SEEDS} unless --allow-any-seed is provided.")

    if args.nrows is not None and args.nrows <= 0:
        raise SystemExit("--nrows must be a positive integer.")
    if args.calibration_final_strategy not in {"cv_stacking", "train_split"}:
        raise SystemExit("--calibration-final-strategy must be one of: cv_stacking, train_split.")
    if args.n_boot < 0:
        raise SystemExit("--n_boot must be >= 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    parquet_path = PROCESSED_DIR / "yrbs_2023_modeling.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Modeling input not found: {parquet_path}. Run scripts/01_build_dataset.py first.")

    df = pd.read_parquet(parquet_path)
    if args.nrows is not None:
        df = df.head(args.nrows).copy()

    # Feature-set selection locked to config lists.
    featureset = args.features
    feature_cols = FEATURES_FULL if featureset == "full" else FEATURES_BASELINE

    required = [TARGET_COL] + list(feature_cols)
    assert_required_columns(df, required)

    y = df[TARGET_COL].astype(int)
    assert_binary_target(y)

    # Categorical covariates are always treated as categorical for preprocessing.
    categorical_covariates = [c for c in FEATURES_BASELINE if c in feature_cols]

    X = df[list(feature_cols)].copy()
    for c in categorical_covariates:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            # Preserve category-ness by mapping numeric codes to strings (and keeping missing as np.nan).
            def _to_cat(v):
                if pd.isna(v):
                    return np.nan
                fv = float(v)
                if fv.is_integer():
                    return f"cat_{int(fv)}"
                return f"cat_{fv}"

            X[c] = s.map(_to_cat).astype(object)
        else:
            # Keep as object/string-like with missing as np.nan for sklearn imputers.
            X[c] = s.astype(object)

    for c in [col for col in feature_cols if col not in categorical_covariates]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Defensive: eliminate pd.NA anywhere (e.g., nullable integers) to keep sklearn imputers happy.
    X = X.replace({pd.NA: np.nan})

    # Frozen holdout split (unweighted predictive evaluation)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=args.seed)
    (train_pos, test_pos) = next(splitter.split(X, y))
    row_index = df.index.to_numpy()
    train_idx = row_index[train_pos]
    test_idx = row_index[test_pos]

    outdir = args.outdir
    out_metrics = outdir / "metrics"
    out_tables = outdir / "tables"
    out_figures = outdir / "figures"
    out_models = outdir / "models"
    out_splits = outdir / "splits"
    out_logs = outdir / "logs"

    for d in [out_metrics, out_tables, out_figures, out_models, out_splits, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    save_npz(out_splits / f"holdout_seed{args.seed}.npz", train_idx=train_idx, test_idx=test_idx)

    # CV fold assignments for the training portion only (saved as train_idx + fold_id).
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=args.seed)
    fold_id = np.full(len(train_idx), fill_value=-1, dtype=int)
    for f, (_, va) in enumerate(skf.split(X_train, y_train)):
        fold_id[va] = f
    if (fold_id < 0).any():
        raise RuntimeError("Failed to assign all training rows to CV folds.")
    save_npz(out_splits / f"cvfolds_seed{args.seed}.npz", train_idx=train_idx, fold_id=fold_id)

    parquet_sha = sha256_file(parquet_path)
    requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements_sha = sha256_file(requirements_path) if requirements_path.exists() else ""
    git_commit = resolve_git_commit(PROJECT_ROOT)

    n_boot_effective = args.n_boot
    if args.nrows is not None and args.n_boot > 200:
        n_boot_effective = 200

    pkg_versions = package_versions(
        ["pandas", "numpy", "pyarrow", "scikit-learn", "matplotlib", "joblib", "statsmodels"]
    )

    models_to_run = ["logreg", "hgb"] if args.model == "both" else [args.model]

    summary_test_rows = []
    summary_cv_rows = []
    primary_ci_rows = []

    for model_name in models_to_run:
        run_id = args.run_id or deterministic_run_id(args.seed, model_name, featureset, args.calibration)
        if args.run_id and args.model == "both":
            # Avoid collisions when a single --run-id is provided with --model both.
            run_id = f"{args.run_id}_{model_name}"

        def estimator_factory():
            pre = build_preprocessor(list(feature_cols), categorical_covariates)
            return build_estimator(model_name, preprocessor=pre, seed=args.seed)

        # CV metrics on training portion only.
        cv_fold_df = fold_probabilities_with_optional_calibration(
            estimator_factory=estimator_factory,
            X_train=X_train,
            y_train=y_train,
            seed=args.seed,
            calibration_method=args.calibration,
        )
        cv_means, cv_stds, cv_legacy = aggregate_fold_metrics(cv_fold_df, calibration_method=args.calibration)
        cv_fold_path = out_metrics / f"metrics_cv_folds_seed{args.seed}_{model_name}_{featureset}_{args.calibration}.csv"
        cv_fold_df.to_csv(cv_fold_path, index=False)

        cv_row = {
            "dataset_version": DATASET_VERSION,
            "run_id": run_id,
            "seed": args.seed,
            "model": model_name,
            "featureset": featureset,
            "calibration_method": args.calibration,
            "n_train": int(len(train_idx)),
            "n_test": int(len(train_idx)),  # each training row is evaluated exactly once via CV validation
            "cv_folds": CV_FOLDS,
            "calibrator_training_strategy": args.calibration_final_strategy if args.calibration != "none" else "none",
            **cv_legacy,
            **cv_means,
            **cv_stds,
        }
        cv_path = out_metrics / f"metrics_cv_seed{args.seed}_{model_name}_{featureset}_{args.calibration}.csv"
        write_metrics_row(cv_path, cv_row)
        summary_cv_rows.append(cv_row)

        # Fit final base estimator on full training portion.
        final_est = estimator_factory()
        final_est.fit(X_train, y_train)

        # Fit post-hoc calibrator on training data only (never on test).
        calibrator: Optional[dict]
        if args.calibration == "none":
            calibrator = None
            calibrator_strategy = "none"
        elif args.calibration_final_strategy == "cv_stacking":
            calibrator = fit_cv_stacking_calibrator(
                estimator_factory=estimator_factory,
                X_train=X_train,
                y_train=y_train,
                seed=args.seed,
                calibration_method=args.calibration,
            )
            calibrator_strategy = "cv_stacking_oof"
        else:
            calibrator = fit_train_split_calibrator(
                estimator_factory=estimator_factory,
                X_train=X_train,
                y_train=y_train,
                seed=args.seed,
                calibration_method=args.calibration,
            )
            calibrator_strategy = "train_split_holdout"

        # Save model artifact (joblib)
        model_path = out_models / f"{model_name}_{featureset}_seed{args.seed}.joblib"
        joblib.dump(final_est, model_path)
        calibrator_path: Optional[Path] = None
        if calibrator is not None:
            calibrator_path = out_models / f"{model_name}_{featureset}_seed{args.seed}.calibrator.joblib"
            joblib.dump(calibrator, calibrator_path)

        y_prob_uncal = predict_proba_positive(final_est, X_test)
        y_prob_cal = apply_posthoc_calibrator(calibrator, y_prob_uncal)
        active_prob = y_prob_cal if args.calibration != "none" else y_prob_uncal

        test_metrics_uncal, cm, test_metrics_cal = compute_test_artifacts(
            y_test=y_test,
            y_prob_uncal=y_prob_uncal,
            y_prob_cal=y_prob_cal,
            out_tables=out_tables,
            out_figures=out_figures,
            model=model_name,
            featureset=featureset,
            seed=args.seed,
            calibration_method=args.calibration,
        )
        selected_test_metrics = test_metrics_cal if args.calibration != "none" else test_metrics_uncal

        test_row = {
            "dataset_version": DATASET_VERSION,
            "run_id": run_id,
            "seed": args.seed,
            "model": model_name,
            "featureset": featureset,
            "calibration_method": args.calibration,
            "calibrator_training_strategy": calibrator_strategy,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            **selected_test_metrics,
            **{f"uncalibrated_{k}": float(v) for k, v in test_metrics_uncal.items()},
            **{f"calibrated_{k}": float(v) for k, v in test_metrics_cal.items()},
            **cm,
        }
        test_path = out_metrics / f"metrics_test_seed{args.seed}_{model_name}_{featureset}_{args.calibration}.csv"
        write_metrics_row(test_path, test_row)
        summary_test_rows.append(test_row)

        # Stratified bootstrap CIs on held-out test metrics for uncalibrated and calibrated probabilities.
        boot_uncal = stratified_bootstrap_metric_draws(
            y_true=y_test.to_numpy(dtype=int),
            y_prob=y_prob_uncal,
            n_boot=n_boot_effective,
            seed=args.seed + 101,
        )
        boot_uncal["stream"] = "uncalibrated"

        boot_cal = stratified_bootstrap_metric_draws(
            y_true=y_test.to_numpy(dtype=int),
            y_prob=y_prob_cal,
            n_boot=n_boot_effective,
            seed=args.seed + 202,
        )
        boot_cal["stream"] = "calibrated"

        boot_draws = pd.concat([boot_uncal, boot_cal], ignore_index=True)
        boot_draws.insert(0, "model", model_name)
        boot_draws.insert(1, "featureset", featureset)
        boot_draws.insert(2, "calibration_method", args.calibration)
        boot_draws.insert(3, "seed", args.seed)
        boot_draws.to_csv(
            out_tables / f"bootstrap_draws_test_seed{args.seed}_{model_name}_{featureset}_{args.calibration}.csv",
            index=False,
        )

        ci_uncal = summarize_bootstrap_ci(boot_uncal)
        ci_cal = summarize_bootstrap_ci(boot_cal)
        for stream, point, ci in [
            ("uncalibrated", test_metrics_uncal, ci_uncal),
            ("calibrated", test_metrics_cal, ci_cal),
        ]:
            primary_ci_rows.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "run_id": run_id,
                    "seed": args.seed,
                    "model": model_name,
                    "featureset": featureset,
                    "calibration_method": args.calibration,
                    "calibrator_training_strategy": calibrator_strategy,
                    "stream": stream,
                    "n_test": int(len(test_idx)),
                    "n_boot": int(n_boot_effective),
                    "roc_auc": float(point["roc_auc"]),
                    "roc_auc_ci95_low": ci["roc_auc"][0],
                    "roc_auc_ci95_high": ci["roc_auc"][1],
                    "pr_auc": float(point["pr_auc"]),
                    "pr_auc_ci95_low": ci["pr_auc"][0],
                    "pr_auc_ci95_high": ci["pr_auc"][1],
                    "brier": float(point["brier"]),
                    "brier_ci95_low": ci["brier"][0],
                    "brier_ci95_high": ci["brier"][1],
                    "calibration_slope": float(point["calibration_slope"]),
                    "calibration_slope_ci95_low": ci["calibration_slope"][0],
                    "calibration_slope_ci95_high": ci["calibration_slope"][1],
                    "calibration_intercept": float(point["calibration_intercept"]),
                    "calibration_intercept_ci95_low": ci["calibration_intercept"][0],
                    "calibration_intercept_ci95_high": ci["calibration_intercept"][1],
                    "ci_method": "stratified_bootstrap_percentile",
                }
            )

        # Predictions + minimal error analysis
        preds_path, error_path, omitted_groups = save_predictions_and_error_slices(
            df_all=df,
            test_idx=test_idx,
            y_true=y_test,
            y_prob_uncal=y_prob_uncal,
            y_prob_cal=y_prob_cal,
            active_prob=active_prob,
            calibration_method=args.calibration,
            model=model_name,
            featureset=featureset,
            seed=args.seed,
            out_tables=out_tables,
            min_group_n=MIN_GROUP_N,
            min_group_pos=MIN_GROUP_POS,
            min_group_neg=MIN_GROUP_NEG,
        )

        # Logistic regression coefficients
        coef_path = None
        if model_name == "logreg" and isinstance(final_est, Pipeline):
            coef_path = str(save_logreg_coefficients(final_est, out_tables, featureset, args.seed, run_id))

        feature_cols_hash = hashlib.sha256("|".join(feature_cols).encode("utf-8")).hexdigest()
        train_idx_hash = sha256_np_array(train_idx.astype(np.int64))
        test_idx_hash = sha256_np_array(test_idx.astype(np.int64))
        y_train_hash = sha256_np_array(y_train.to_numpy(dtype=np.int64))
        y_test_hash = sha256_np_array(y_test.to_numpy(dtype=np.int64))

        # Model metadata JSON
        meta = {
            "dataset_version": DATASET_VERSION,
            "experiment_namespace": EXPERIMENT_NAMESPACE,
            "run_id": run_id,
            "seed": args.seed,
            "model": model_name,
            "featureset": featureset,
            "feature_cols": list(feature_cols),
            "calibration_method": args.calibration,
            "calibrator_training_strategy": calibrator_strategy,
            "calibration_cv_mode": "fold_validation_fit_and_eval",
            "calibration_training_data_scope": "train_only",
            "target_col": TARGET_COL,
            "positive_label": POSITIVE_LABEL,
            "validation_protocol": {"test_size": TEST_SIZE, "cv_folds": CV_FOLDS, "random_seed": args.seed},
            "inputs": {
                "parquet_path": str(parquet_path),
                "parquet_sha256": parquet_sha,
                "nrows": args.nrows,
                "requirements_path": str(requirements_path),
                "requirements_sha256": requirements_sha,
                "feature_cols_sha256": feature_cols_hash,
                "train_index_sha256": train_idx_hash,
                "test_index_sha256": test_idx_hash,
                "y_train_sha256": y_train_hash,
                "y_test_sha256": y_test_hash,
            },
            "artifacts": {
                "model_joblib": str(model_path),
                "calibrator_joblib": str(calibrator_path) if calibrator_path is not None else None,
                "metrics_cv_csv": str(cv_path),
                "metrics_cv_folds_csv": str(cv_fold_path),
                "metrics_test_csv": str(test_path),
                "primary_metrics_with_ci_csv": str(out_tables / "primary_metrics_with_ci.csv"),
                "bootstrap_draws_test_csv": str(
                    out_tables / f"bootstrap_draws_test_seed{args.seed}_{model_name}_{featureset}_{args.calibration}.csv"
                ),
                "preds_test_csv": str(preds_path),
                "error_slices_csv": str(error_path),
                "holdout_split_npz": str(out_splits / f"holdout_seed{args.seed}.npz"),
                "cvfolds_npz": str(out_splits / f"cvfolds_seed{args.seed}.npz"),
                "logreg_coefficients_csv": coef_path,
                "calibration_curve_uncalibrated_csv": str(
                    out_tables
                    / f"calibration_curve_test_uncalibrated_{model_name}_{featureset}_seed{args.seed}_{args.calibration}.csv"
                ),
                "calibration_curve_calibrated_csv": str(
                    out_tables
                    / f"calibration_curve_test_calibrated_{model_name}_{featureset}_seed{args.seed}_{args.calibration}.csv"
                ),
            },
            "error_slices": {
                "min_group_n": MIN_GROUP_N,
                "min_group_pos": MIN_GROUP_POS,
                "min_group_neg": MIN_GROUP_NEG,
                "omitted_groups": omitted_groups,
            },
            "runtime": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "argv": sys.argv,
                "packages": pkg_versions,
                "git_commit": git_commit,
                "package_lock_sha256": requirements_sha,
                "n_boot_requested": int(args.n_boot),
                "n_boot_effective": int(n_boot_effective),
            },
            "scope_notes": [
                "Predictive evaluation is unweighted (no survey design adjustment in metrics).",
                "Weighted prevalence summaries are handled separately in the EDA pipeline.",
                "Results are predictive associations only; no causal interpretation.",
            ],
        }

        meta_path = out_models / f"{model_name}_{featureset}_seed{args.seed}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        # Compact run-level log pointer (optional but useful)
        (out_logs / f"run_{run_id}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    # Thesis-ready summary tables (one row per model/featureset for this run)
    results_test = pd.DataFrame(summary_test_rows)
    results_cv = pd.DataFrame(summary_cv_rows)
    results_test.to_csv(out_tables / "results_summary_test.csv", index=False)
    results_cv.to_csv(out_tables / "results_summary_cv.csv", index=False)
    pd.DataFrame(primary_ci_rows).to_csv(out_tables / "primary_metrics_with_ci.csv", index=False)

    print(f"Wrote modeling artifacts to {outdir}/")


if __name__ == "__main__":
    main()
