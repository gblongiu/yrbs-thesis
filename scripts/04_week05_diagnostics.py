from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CV_FOLDS, FEATURES_BASELINE, FEATURES_FULL, PROCESSED_DIR, TARGET_COL


def build_preprocessor(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    transformers = []
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))
    if num_cols:
        transformers.append(("num", numeric, num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_hgb_pipeline(
    *, feature_cols: List[str], categorical_cols: List[str], seed: int, tuned_params: Optional[Dict[str, object]] = None
) -> Pipeline:
    pre = build_preprocessor(feature_cols, categorical_cols)
    params: Dict[str, object] = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 400,
        "early_stopping": True,
        "random_state": seed,
    }
    if tuned_params:
        params.update(tuned_params)
    params["random_state"] = seed
    model = HistGradientBoostingClassifier(**params)
    return Pipeline(steps=[("preprocessor", pre), ("model", model)])


def prepare_features(df: pd.DataFrame, feature_cols: List[str], categorical_covariates: List[str]) -> pd.DataFrame:
    X = df[list(feature_cols)].copy()
    for c in categorical_covariates:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):

            def _to_cat(v):
                if pd.isna(v):
                    return np.nan
                fv = float(v)
                if fv.is_integer():
                    return f"cat_{int(fv)}"
                return f"cat_{fv}"

            X[c] = s.map(_to_cat).astype(object)
        else:
            X[c] = s.astype(object)

    for c in [col for col in feature_cols if col not in categorical_covariates]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace({pd.NA: np.nan})
    return X


def make_calibration_overlay(
    *,
    out_tables: Path,
    out_figures: Path,
    seed: int,
    featureset: str,
    calibration: str,
    baseline_model: str,
    model: str,
) -> bool:
    baseline_curve = out_tables / f"calibration_curve_test_uncalibrated_{baseline_model}_{featureset}_seed{seed}_{calibration}.csv"
    model_curve = out_tables / f"calibration_curve_test_uncalibrated_{model}_{featureset}_seed{seed}_{calibration}.csv"

    fallback_used = False
    if baseline_curve.exists() and model_curve.exists():
        baseline_df = pd.read_csv(baseline_curve)
        model_df = pd.read_csv(model_curve)
    else:
        fallback_used = True
        baseline_preds = out_tables / f"preds_test_{baseline_model}_{featureset}_seed{seed}.csv"
        model_preds = out_tables / f"preds_test_{model}_{featureset}_seed{seed}.csv"
        if not baseline_preds.exists() or not model_preds.exists():
            raise SystemExit(
                "Calibration overlay fallback failed: expected prediction tables were not found. "
                f"Missing baseline={baseline_preds.exists()} model={model_preds.exists()}"
            )
        bdf = pd.read_csv(baseline_preds)
        mdf = pd.read_csv(model_preds)
        b_frac, b_mean = calibration_curve(bdf["y_true"], bdf["y_prob"], n_bins=10, strategy="quantile")
        m_frac, m_mean = calibration_curve(mdf["y_true"], mdf["y_prob"], n_bins=10, strategy="quantile")
        baseline_df = pd.DataFrame({"mean_predicted": b_mean, "fraction_positive": b_frac})
        model_df = pd.DataFrame({"mean_predicted": m_mean, "fraction_positive": m_frac})

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal")
    ax.plot(baseline_df["mean_predicted"], baseline_df["fraction_positive"], marker="o", linewidth=2, label=baseline_model)
    ax.plot(model_df["mean_predicted"], model_df["fraction_positive"], marker="o", linewidth=2, label=model)
    ax.set_title(f"Week 5 Calibration Comparison (Test), seed={seed}, features={featureset}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_figures / f"week05_calibration_comparison_seed{seed}.png", dpi=300)
    plt.close(fig)

    return fallback_used


def compute_feature_stability(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    fold_id: np.ndarray,
    feature_cols: List[str],
    categorical_covariates: List[str],
    tuned_params: Dict[str, object],
    seed: int,
    perm_repeats: int,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    unique_folds = sorted(np.unique(fold_id).tolist())
    if len(unique_folds) != CV_FOLDS:
        raise SystemExit(f"Expected {CV_FOLDS} folds in frozen artifact, found {len(unique_folds)}.")

    for fold in unique_folds:
        va_mask = fold_id == fold
        tr_idx = train_idx[~va_mask]
        va_idx = train_idx[va_mask]
        X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
        X_va, y_va = X.loc[va_idx], y.loc[va_idx]

        pipe = build_hgb_pipeline(
            feature_cols=feature_cols,
            categorical_cols=categorical_covariates,
            seed=seed,
            tuned_params=tuned_params,
        )
        pipe.fit(X_tr, y_tr)
        perm = permutation_importance(
            estimator=pipe,
            X=X_va,
            y=y_va,
            scoring="roc_auc",
            n_repeats=perm_repeats,
            random_state=seed + int(fold) + 1000,
            n_jobs=1,
        )

        feature_names = X_va.columns.tolist()
        imp_means = pd.Series(perm.importances_mean, index=feature_names)
        rank_series = imp_means.rank(ascending=False, method="min")
        imp_stds = pd.Series(perm.importances_std, index=feature_names)
        for fname in feature_names:
            rows.append(
                {
                    "fold": int(fold),
                    "feature_name": str(fname),
                    "importance_mean": float(imp_means.loc[fname]),
                    "importance_std": float(imp_stds.loc[fname]),
                    "rank": float(rank_series.loc[fname]),
                    "n_repeats": int(perm_repeats),
                }
            )

    by_fold_df = pd.DataFrame(rows).sort_values(["fold", "rank", "feature_name"], kind="mergesort").reset_index(drop=True)

    summary_df = (
        by_fold_df.groupby("feature_name", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            mean_rank=("rank", "mean"),
        )
        .sort_values(["importance_mean", "mean_rank"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )

    rank_pivot = by_fold_df.pivot(index="feature_name", columns="fold", values="rank")
    mean_rank_series = rank_pivot.mean(axis=1)
    spearman_vals = []
    for fold in rank_pivot.columns:
        spearman_vals.append(float(rank_pivot[fold].corr(mean_rank_series, method="spearman")))
    mean_spearman = float(np.nanmean(spearman_vals)) if spearman_vals else np.nan

    top_sets = {}
    for fold in unique_folds:
        subset = by_fold_df[by_fold_df["fold"] == fold].sort_values("rank", kind="mergesort")
        top_sets[fold] = set(subset.head(top_k)["feature_name"].tolist())
    jaccard_vals = []
    for a, b in combinations(unique_folds, 2):
        inter = len(top_sets[a].intersection(top_sets[b]))
        union = len(top_sets[a].union(top_sets[b]))
        jaccard_vals.append(float(inter / union) if union else np.nan)
    mean_jaccard = float(np.nanmean(jaccard_vals)) if jaccard_vals else np.nan

    summary_df["mean_spearman_rank_vs_mean"] = mean_spearman
    summary_df["mean_pairwise_jaccard_topk"] = mean_jaccard
    summary_df["top_k"] = int(top_k)

    return by_fold_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 diagnostics for calibration comparison and feature stability.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", choices=["baseline", "full"], required=True)
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"], required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="hgb")
    parser.add_argument("--baseline-model", type=str, default="logreg")
    parser.add_argument("--perm-repeats", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_metrics = args.outdir / "metrics"
    out_tables = args.outdir / "tables"
    out_figures = args.outdir / "figures"
    out_tuning = args.outdir / "tuning"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = out_metrics / f"metrics_test_seed{args.seed}_{args.baseline_model}_{args.features}_{args.calibration}.csv"
    model_metrics_path = out_metrics / f"metrics_test_seed{args.seed}_{args.model}_{args.features}_{args.calibration}.csv"
    if not baseline_metrics_path.exists() or not model_metrics_path.exists():
        raise SystemExit(
            "Missing required metrics files for calibration comparison. "
            f"baseline_exists={baseline_metrics_path.exists()} model_exists={model_metrics_path.exists()}"
        )

    baseline_metrics = pd.read_csv(baseline_metrics_path).iloc[0]
    model_metrics = pd.read_csv(model_metrics_path).iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "model": args.baseline_model,
                "roc_auc": float(baseline_metrics["roc_auc"]),
                "pr_auc": float(baseline_metrics["pr_auc"]),
                "brier": float(baseline_metrics["brier"]),
                "calibration_slope": float(baseline_metrics["calibration_slope"]),
                "calibration_intercept": float(baseline_metrics["calibration_intercept"]),
            },
            {
                "model": args.model,
                "roc_auc": float(model_metrics["roc_auc"]),
                "pr_auc": float(model_metrics["pr_auc"]),
                "brier": float(model_metrics["brier"]),
                "calibration_slope": float(model_metrics["calibration_slope"]),
                "calibration_intercept": float(model_metrics["calibration_intercept"]),
            },
        ]
    )
    comparison_path = out_tables / f"week05_calibration_comparison_seed{args.seed}.csv"
    comparison_df.to_csv(comparison_path, index=False)

    fallback_used = make_calibration_overlay(
        out_tables=out_tables,
        out_figures=out_figures,
        seed=args.seed,
        featureset=args.features,
        calibration=args.calibration,
        baseline_model=args.baseline_model,
        model=args.model,
    )

    tuning_params_path = out_tuning / f"hgb_seed{args.seed}_{args.features}_best_params.json"
    if not tuning_params_path.exists():
        raise SystemExit(f"Missing tuned parameter file: {tuning_params_path}")
    tuning_payload = json.loads(tuning_params_path.read_text(encoding="utf-8"))
    tuned_params = tuning_payload.get("best_params", {})

    feature_cols = FEATURES_BASELINE if args.features == "baseline" else FEATURES_FULL
    categorical_covariates = [c for c in FEATURES_BASELINE if c in feature_cols]

    parquet_path = PROCESSED_DIR / "yrbs_2023_modeling.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Modeling parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    X = prepare_features(df, list(feature_cols), categorical_covariates)
    y = df[TARGET_COL].astype(int)

    holdout_path = args.outdir / "splits" / f"holdout_seed{args.seed}.npz"
    cvfolds_path = args.outdir / "splits" / f"cvfolds_seed{args.seed}.npz"
    if not holdout_path.exists() or not cvfolds_path.exists():
        raise SystemExit(
            "Missing frozen split artifacts required for stability diagnostics. "
            f"holdout_exists={holdout_path.exists()} cvfolds_exists={cvfolds_path.exists()}"
        )
    holdout_npz = np.load(holdout_path)
    cv_npz = np.load(cvfolds_path)
    train_idx = holdout_npz["train_idx"]
    cv_train_idx = cv_npz["train_idx"]
    if not np.array_equal(train_idx, cv_train_idx):
        raise SystemExit("Frozen artifacts inconsistent: holdout train_idx differs from cvfolds train_idx.")
    fold_id = cv_npz["fold_id"]

    by_fold_df, summary_df = compute_feature_stability(
        X=X,
        y=y,
        train_idx=train_idx,
        fold_id=fold_id,
        feature_cols=list(feature_cols),
        categorical_covariates=categorical_covariates,
        tuned_params=tuned_params,
        seed=args.seed,
        perm_repeats=args.perm_repeats,
        top_k=args.top_k,
    )

    by_fold_path = out_tables / f"hgb_seed{args.seed}_{args.features}_perm_importance_by_fold.csv"
    summary_path = out_tables / f"hgb_seed{args.seed}_{args.features}_perm_importance_summary.csv"
    by_fold_df.to_csv(by_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    top_n = min(20, len(summary_df))
    if top_n > 0:
        plot_df = summary_df.head(top_n).iloc[::-1].copy()
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.28)))
        ax.barh(plot_df["feature_name"], plot_df["importance_mean"], xerr=plot_df["importance_std"].fillna(0.0))
        ax.set_xlabel("Permutation importance mean (validation ROC AUC drop)")
        ax.set_ylabel("Feature")
        ax.set_title(f"HGB Feature Importance Stability, seed={args.seed}, features={args.features}")
        fig.tight_layout()
        fig.savefig(out_figures / f"hgb_seed{args.seed}_{args.features}_importance_stability.png", dpi=300)
        plt.close(fig)

    print(
        json.dumps(
            {
                "calibration_comparison_csv": str(comparison_path),
                "calibration_overlay_fallback_used": fallback_used,
                "perm_importance_by_fold_csv": str(by_fold_path),
                "perm_importance_summary_csv": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
