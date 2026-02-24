from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from src.config import MIN_GROUP_EVENTRATE, MIN_GROUP_N, MIN_GROUP_NEG, MIN_GROUP_POS
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


def _calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = clip_probs(np.asarray(y_prob, dtype=float))
    logit = np.log(probs / (1.0 - probs)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, np.asarray(y_true, dtype=int))
    return float(model.coef_[0][0])


def _threshold_ok(n_pos: int, n_neg: int) -> bool:
    return n_pos >= int(MIN_GROUP_POS) and n_neg >= int(MIN_GROUP_NEG)


def compute_subgroup_metrics_row(
    *,
    subgroup_var: str,
    subgroup_value: str,
    y_group: np.ndarray,
    p_group: np.ndarray,
    overall_brier: float,
    seed: int,
) -> Dict[str, object]:
    y_group = np.asarray(y_group, dtype=int)
    p_group = clip_probs(np.asarray(p_group, dtype=float))

    n = int(y_group.size)
    n_pos = int(np.sum(y_group == 1))
    n_neg = int(np.sum(y_group == 0))

    threshold_ok = _threshold_ok(n_pos=n_pos, n_neg=n_neg)

    if threshold_ok:
        roc_auc = float(roc_auc_score(y_group, p_group))
        roc_auc_defined_flag = True
    else:
        roc_auc = np.nan
        roc_auc_defined_flag = False

    brier = float(brier_score_loss(y_group, p_group))

    if threshold_ok:
        try:
            calibration_slope = float(_calibration_slope(y_group, p_group))
            slope_defined_flag = True
        except Exception:
            calibration_slope = np.nan
            slope_defined_flag = False
    else:
        calibration_slope = np.nan
        slope_defined_flag = False

    low_slope_flag = bool(slope_defined_flag and np.isfinite(calibration_slope) and calibration_slope < 0.8)
    high_error_flag = bool(brier > (1.15 * overall_brier))

    return {
        "subgroup_var": subgroup_var,
        "subgroup_value": str(subgroup_value),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "roc_auc": roc_auc,
        "brier": brier,
        "calibration_slope": calibration_slope,
        "roc_auc_defined_flag": roc_auc_defined_flag,
        "slope_defined_flag": slope_defined_flag,
        "low_slope_flag": low_slope_flag,
        "high_error_flag": high_error_flag,
        "seed": seed,
        "overall_brier": overall_brier,
        "MIN_GROUP_POS": int(MIN_GROUP_POS),
        "MIN_GROUP_NEG": int(MIN_GROUP_NEG),
        "MIN_GROUP_N": int(MIN_GROUP_N),
        "MIN_GROUP_EVENTRATE": MIN_GROUP_EVENTRATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 held-out subgroup performance audit for HGB full none")
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

    y_prob_test = clip_probs(model.predict_proba(x_test)[:, 1])
    y_true_test = y_test.to_numpy(dtype=int)
    overall_brier = float(brier_score_loss(y_true_test, y_prob_test))

    test_rows = df.loc[test_idx, :].copy()
    rows: List[Dict[str, object]] = []

    for subgroup_var in ["raceeth", "q2"]:
        if subgroup_var not in test_rows.columns:
            raise RuntimeError(f"Missing subgroup column in test rows: {subgroup_var}")

        subgroup_values = test_rows[subgroup_var].astype("string").fillna("<NA>").to_numpy()

        temp_df = pd.DataFrame(
            {
                "subgroup_value": subgroup_values,
                "y_true": y_true_test,
                "y_prob": y_prob_test,
            }
        )

        for subgroup_value, group_df in temp_df.groupby("subgroup_value", sort=True):
            rows.append(
                compute_subgroup_metrics_row(
                    subgroup_var=subgroup_var,
                    subgroup_value=str(subgroup_value),
                    y_group=group_df["y_true"].to_numpy(dtype=int),
                    p_group=group_df["y_prob"].to_numpy(dtype=float),
                    overall_brier=overall_brier,
                    seed=args.seed,
                )
            )

    out_df = pd.DataFrame(rows).sort_values(["subgroup_var", "subgroup_value"], kind="mergesort").reset_index(drop=True)
    out_path = args.outdir / "tables" / "subgroup_performance_seed2026.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(json.dumps({"status": "ok", "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
