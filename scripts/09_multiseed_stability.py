from __future__ import annotations

"""
Week 7 multiseed stability analysis.

Notes:
- This script keeps the Week 4 holdout split fixed with seed 2026.
- It varies only CV fold construction on the fixed training partition.
- Across-seed standard deviations are repeated on every output row by design
  so each row is self-contained for downstream audit joins.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    run_cv_metrics_with_leakage_safe_calibration,
)


def _load_tuned_params(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", {})
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Invalid tuned parameter payload in {path}")
    return params


def _compute_fold_id_from_train_only(y_train: pd.Series, cv_folds: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_id = np.full(len(y_train), fill_value=-1, dtype=int)
    x_dummy = np.zeros((len(y_train), 1), dtype=float)

    for fold_idx, (_, va_pos) in enumerate(skf.split(x_dummy, y_train.to_numpy(dtype=int)), start=0):
        fold_id[va_pos] = fold_idx

    if (fold_id < 0).any():
        raise RuntimeError("Failed to assign all training rows to folds")
    return fold_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 multi-seed stability for HGB full none")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.seed != 2026:
        raise RuntimeError("Frozen protocol seed must remain 2026")

    modeling_path = PROJECT_ROOT / "data" / "processed" / "yrbs_2023_modeling.parquet"
    holdout_path = args.outdir / "splits" / "holdout_seed2026.npz"
    cvfolds_path = args.outdir / "splits" / "cvfolds_seed2026.npz"
    tuned_path = args.outdir / "tuning" / "hgb_seed2026_baseline_best_params.json"
    manifest_path = PROJECT_ROOT / "docs" / "status_reports" / "report_03" / "week06_run_manifest.json"
    perm_by_fold_path = args.outdir / "tables" / "hgb_seed2026_full_perm_importance_by_fold.csv"

    # Fail-fast guard for all declared Week 7 prerequisites.
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
    train_idx = holdout["train_idx"]

    y = df[TARGET_COL].astype(int)

    feature_sets = derive_feature_sets(df.columns.tolist())
    cat_covars = [c for c in ["q1", "q2", "q3", "raceeth"] if c in feature_sets.full_features]
    x_full = prepare_feature_frame(df, feature_sets.full_features, cat_covars)

    x_train = x_full.loc[train_idx]
    y_train = y.loc[train_idx]

    tuned_params = _load_tuned_params(tuned_path)

    def estimator_factory() -> object:
        pre = build_preprocessor(feature_cols=feature_sets.full_features, categorical_cols=cat_covars)
        # Keep model random state fixed to isolate partition sensitivity from fold construction.
        return build_estimator(
            model_name="hgb",
            preprocessor=pre,
            seed=args.model_seed,
            tuned_hgb_params=tuned_params,
        )

    seed_list = [2026, 2027, 2028, 2029]
    rows: List[Dict[str, object]] = []

    for fold_seed in seed_list:
        fold_id = _compute_fold_id_from_train_only(y_train=y_train, cv_folds=CV_FOLDS, seed=fold_seed)
        fold_df = run_cv_metrics_with_leakage_safe_calibration(
            estimator_factory=estimator_factory,
            x_train=x_train,
            y_train=y_train,
            fold_id=fold_id,
            calibration_method="none",
            seed=args.seed,
        )
        cv_metrics = aggregate_cv_metrics(fold_df)

        rows.append(
            {
                "seed": int(fold_seed),
                "roc_auc_mean": float(cv_metrics["roc_auc"]),
                "pr_auc_mean": float(cv_metrics["pr_auc"]),
                "brier_mean": float(cv_metrics["brier"]),
                "slope_mean": float(cv_metrics["calibration_slope"]),
                "n_train": int(len(train_idx)),
                "cv_folds": int(CV_FOLDS),
                "test_split_seed": 2026,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("seed", kind="mergesort").reset_index(drop=True)

    # Repeated per-row by convention so each row contains both point and dispersion context.
    out_df["roc_auc_std_across_seeds"] = float(out_df["roc_auc_mean"].std(ddof=1))
    out_df["pr_auc_std_across_seeds"] = float(out_df["pr_auc_mean"].std(ddof=1))
    out_df["brier_std_across_seeds"] = float(out_df["brier_mean"].std(ddof=1))
    out_df["slope_std_across_seeds"] = float(out_df["slope_mean"].std(ddof=1))

    out_path = args.outdir / "tables" / "multiseed_stability_seed2026_2029.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(json.dumps({"status": "ok", "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
