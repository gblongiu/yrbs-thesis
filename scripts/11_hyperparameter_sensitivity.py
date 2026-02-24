from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from week06 import (
    CV_FOLDS,
    PROJECT_ROOT,
    TARGET_COL,
    aggregate_cv_metrics,
    build_estimator,
    build_preprocessor,
    derive_feature_sets,
    prepare_feature_frame,
    require_paths,
    resolved_hgb_params,
    run_cv_metrics_with_leakage_safe_calibration,
)


def _load_tuned_params(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", {})
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Invalid tuned parameter payload in {path}")
    return params


def _to_positive_float(value: object) -> float:
    return float(value)


def _to_positive_int(value: object) -> int:
    out = int(round(float(value)))
    return max(1, out)


def _make_configs(base: Dict[str, object]) -> List[Tuple[str, Dict[str, object]]]:
    base_learning_rate = _to_positive_float(base.get("learning_rate", 0.05))
    base_max_depth_raw = base.get("max_depth", 6)
    base_min_samples_leaf = _to_positive_int(base.get("min_samples_leaf", 20))

    if base_max_depth_raw is None:
        base_max_depth = 6
    else:
        base_max_depth = _to_positive_int(base_max_depth_raw)

    configs: List[Tuple[str, Dict[str, object]]] = []

    configs.append(("baseline_tuned", dict(base)))

    lr_minus = dict(base)
    lr_minus["learning_rate"] = max(1e-6, base_learning_rate * 0.8)
    configs.append(("learning_rate_minus20", lr_minus))

    lr_plus = dict(base)
    lr_plus["learning_rate"] = max(1e-6, base_learning_rate * 1.2)
    configs.append(("learning_rate_plus20", lr_plus))

    md_minus = dict(base)
    md_minus["max_depth"] = max(1, base_max_depth - 1)
    configs.append(("max_depth_minus1", md_minus))

    md_plus = dict(base)
    md_plus["max_depth"] = base_max_depth + 1
    configs.append(("max_depth_plus1", md_plus))

    leaf_minus = dict(base)
    leaf_minus["min_samples_leaf"] = max(1, int(round(base_min_samples_leaf * 0.8)))
    configs.append(("min_samples_leaf_minus20", leaf_minus))

    leaf_plus = dict(base)
    leaf_plus["min_samples_leaf"] = max(1, int(round(base_min_samples_leaf * 1.2)))
    configs.append(("min_samples_leaf_plus20", leaf_plus))

    if len(configs) != 7:
        raise RuntimeError("Expected 7 hyperparameter sensitivity configurations")

    return configs


def _fold_consistency_flag(values: np.ndarray) -> bool:
    signs = np.sign(np.asarray(values, dtype=float))
    nonzero = signs[signs != 0]
    if nonzero.size == 0:
        return True
    return bool(np.all(nonzero == nonzero[0]))


def _build_extended_perm_summary(by_fold_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "fold",
        "feature_name",
        "importance_mean_neg_brier_score",
        "importance_mean_roc_auc",
        "rank_neg_brier_score",
    ]
    missing = [c for c in required_cols if c not in by_fold_df.columns]
    if missing:
        raise RuntimeError(f"Permutation by-fold file missing required columns: {missing}")

    grouped = by_fold_df.groupby("feature_name", as_index=False)
    summary = grouped.agg(
        mean_importance_neg_brier_score=("importance_mean_neg_brier_score", "mean"),
        sd_importance_neg_brier_score=("importance_mean_neg_brier_score", "std"),
        mean_importance_roc_auc=("importance_mean_roc_auc", "mean"),
        sd_importance_roc_auc=("importance_mean_roc_auc", "std"),
        mean_rank_neg_brier_score=("rank_neg_brier_score", "mean"),
        sd_rank_neg_brier_score=("rank_neg_brier_score", "std"),
        folds_present=("fold", "nunique"),
    )

    cv_values: List[float] = []
    consistency_values: List[bool] = []
    for _, row in summary.iterrows():
        feature = str(row["feature_name"])
        vals = by_fold_df.loc[
            by_fold_df["feature_name"] == feature,
            "importance_mean_neg_brier_score",
        ].to_numpy(dtype=float)
        mean_val = float(row["mean_importance_neg_brier_score"])
        sd_val = float(row["sd_importance_neg_brier_score"]) if pd.notna(row["sd_importance_neg_brier_score"]) else np.nan

        if np.isfinite(mean_val) and abs(mean_val) > 0 and np.isfinite(sd_val):
            cv_values.append(float(sd_val / abs(mean_val)))
        else:
            cv_values.append(np.nan)

        consistency_values.append(_fold_consistency_flag(vals))

    summary["coefficient_of_variation_neg_brier_score"] = cv_values
    summary["fold_consistency_flag"] = consistency_values
    summary["fold_consistency_all_folds"] = summary["folds_present"] == int(by_fold_df["fold"].nunique())

    summary = summary.sort_values(
        ["mean_importance_neg_brier_score", "mean_rank_neg_brier_score"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    for bully_feature in ["x_qn24", "x_qn25"]:
        if bully_feature not in set(summary["feature_name"].astype(str).tolist()):
            raise RuntimeError(f"Missing required bullying feature in extended permutation summary: {bully_feature}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 HGB hyperparameter sensitivity and permutation extension")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.seed != 2026:
        raise RuntimeError("Frozen protocol seed must remain 2026")

    modeling_path = PROJECT_ROOT / "data" / "processed" / "yrbs_2023_modeling.parquet"
    holdout_path = args.outdir / "splits" / "holdout_seed2026.npz"
    cvfolds_path = args.outdir / "splits" / "cvfolds_seed2026.npz"
    tuned_path = args.outdir / "tuning" / "hgb_seed2026_baseline_best_params.json"
    manifest_path = PROJECT_ROOT / "docs" / "status_reports" / "report_03" / "week06_run_manifest.json"
    perm_by_fold_path = args.outdir / "tables" / "hgb_seed2026_full_perm_importance_by_fold.csv"

    require_paths(
        [
            modeling_path,
            holdout_path,
            cvfolds_path,
            tuned_path,
            manifest_path,
            perm_by_fold_path,
        ]
    )

    df = pd.read_parquet(modeling_path)
    holdout = np.load(holdout_path)
    cvfolds = np.load(cvfolds_path)
    train_idx = holdout["train_idx"]
    cv_train_idx = cvfolds["train_idx"]
    fold_id = cvfolds["fold_id"]

    if not np.array_equal(train_idx, cv_train_idx):
        raise RuntimeError("Frozen split mismatch: holdout train_idx differs from cvfolds train_idx")
    if len(fold_id) != len(train_idx):
        raise RuntimeError("Frozen fold mismatch: fold_id length does not match train_idx length")

    y = df[TARGET_COL].astype(int)
    feature_sets = derive_feature_sets(df.columns.tolist())
    cat_covars = [c for c in ["q1", "q2", "q3", "raceeth"] if c in feature_sets.full_features]
    x_full = prepare_feature_frame(df, feature_sets.full_features, cat_covars)

    x_train = x_full.loc[train_idx]
    y_train = y.loc[train_idx]

    tuned_params = _load_tuned_params(tuned_path)
    configs = _make_configs(tuned_params)

    rows: List[Dict[str, object]] = []

    for config_name, config_params in configs:

        def estimator_factory() -> object:
            pre = build_preprocessor(feature_cols=feature_sets.full_features, categorical_cols=cat_covars)
            return build_estimator(
                model_name="hgb",
                preprocessor=pre,
                seed=args.seed,
                tuned_hgb_params=config_params,
            )

        fold_df = run_cv_metrics_with_leakage_safe_calibration(
            estimator_factory=estimator_factory,
            x_train=x_train,
            y_train=y_train,
            fold_id=fold_id,
            calibration_method="none",
            seed=args.seed,
        )
        cv_metrics = aggregate_cv_metrics(fold_df)
        resolved = resolved_hgb_params(config_params, args.seed)

        rows.append(
            {
                "config_name": config_name,
                "seed": args.seed,
                "cv_folds": int(CV_FOLDS),
                "brier_mean": float(cv_metrics["brier"]),
                "calibration_slope_mean": float(cv_metrics["calibration_slope"]),
                "roc_auc_mean": float(cv_metrics["roc_auc"]),
                "pr_auc_mean": float(cv_metrics["pr_auc"]),
                "calibration_intercept_mean": float(cv_metrics["calibration_intercept"]),
                "params_json": json.dumps(resolved, sort_keys=True),
            }
        )

    sensitivity_df = pd.DataFrame(rows)
    if len(sensitivity_df) != 7:
        raise RuntimeError("Hyperparameter sensitivity output must contain exactly 7 rows")

    out_sensitivity = args.outdir / "tables" / "hgb_hyperparameter_sensitivity_seed2026.csv"
    out_sensitivity.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(out_sensitivity, index=False)

    by_fold_df = pd.read_csv(perm_by_fold_path)
    extended_summary = _build_extended_perm_summary(by_fold_df)
    out_perm_ext = args.outdir / "tables" / "hgb_seed2026_full_perm_importance_summary_extended.csv"
    extended_summary.to_csv(out_perm_ext, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "sensitivity_table": str(out_sensitivity),
                "perm_summary_extended": str(out_perm_ext),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
