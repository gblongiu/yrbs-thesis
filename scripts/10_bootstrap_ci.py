from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from src.evaluation.bootstrap import stratified_bootstrap_metric_draws, summarize_bootstrap_ci
from src.evaluation.metrics import compute_binary_metrics
from week06 import (
    PROJECT_ROOT,
    TARGET_COL,
    build_estimator,
    build_preprocessor,
    clip_probs,
    derive_feature_sets,
    prepare_feature_frame,
    require_paths,
)


def _load_tuned_params(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", {})
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Invalid tuned parameter payload in {path}")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 held-out bootstrap CIs for HGB full none")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-seed", type=int, default=72026)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.seed != 2026:
        raise RuntimeError("Frozen protocol seed must remain 2026")
    if args.n_boot < 1000:
        raise RuntimeError("n_boot must be at least 1000")

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
    train_idx = holdout["train_idx"]
    test_idx = holdout["test_idx"]

    y = df[TARGET_COL].astype(int)

    feature_sets = derive_feature_sets(df.columns.tolist())
    cat_covars = [c for c in ["q1", "q2", "q3", "raceeth"] if c in feature_sets.full_features]
    x_full = prepare_feature_frame(df, feature_sets.full_features, cat_covars)

    x_train = x_full.loc[train_idx]
    y_train = y.loc[train_idx]
    x_test = x_full.loc[test_idx]
    y_test = y.loc[test_idx]

    tuned_params = _load_tuned_params(tuned_path)
    pre = build_preprocessor(feature_cols=feature_sets.full_features, categorical_cols=cat_covars)
    model = build_estimator(
        model_name="hgb",
        preprocessor=pre,
        seed=args.seed,
        tuned_hgb_params=tuned_params,
    )
    model.fit(x_train, y_train)
    y_prob = clip_probs(model.predict_proba(x_test)[:, 1])

    preds_path = args.outdir / "preds" / "preds_test_hgb_full_none_seed2026.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame(
        {
            "row_index": test_idx.astype(int),
            "y_true": y_test.to_numpy(dtype=int),
            "y_prob": y_prob,
            "seed": args.seed,
            "sample_scope": "heldout_test_only",
        }
    )
    preds_df.to_csv(preds_path, index=False)

    # Stratified bootstrap on held-out test rows only.
    draws = stratified_bootstrap_metric_draws(
        y_true=y_test.to_numpy(dtype=int),
        y_prob=y_prob,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
    )

    ci = summarize_bootstrap_ci(draws, metrics=("roc_auc", "brier", "calibration_slope"))
    point = compute_binary_metrics(y_test.to_numpy(dtype=int), y_prob)

    out_rows: List[Dict[str, object]] = []
    for metric_name in ["roc_auc", "brier", "calibration_slope"]:
        out_rows.append(
            {
                "metric": metric_name,
                "mean": float(point[metric_name]),
                "lower_95": float(ci[metric_name][0]),
                "upper_95": float(ci[metric_name][1]),
                "bootstrap_seed": int(args.bootstrap_seed),
                "n_boot": int(args.n_boot),
                "sample_scope": "heldout_test_only",
                "stratified": True,
            }
        )

    out_path = args.outdir / "tables" / "heldout_bootstrap_ci_seed2026.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "predictions": str(preds_path),
                "ci_table": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
