from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from week06 import (
    CV_FOLDS,
    PROJECT_ROOT,
    REQUIRED_METRIC_COLS,
    TARGET_COL,
    TEST_SIZE,
    aggregate_cv_metrics,
    build_estimator,
    build_preprocessor,
    build_week06_figures,
    build_week06_tables,
    derive_feature_sets,
    prepare_feature_frame,
    recompute_split_and_fold_artifacts,
    require_paths,
    run_cv_metrics_with_leakage_safe_calibration,
    run_permutation_importance_stability,
    run_test_metrics_with_train_only_calibration,
    snapshot_hashes,
    validate_metric_fields,
    verify_frozen_split_and_fold_equality,
    verify_hashes_unchanged,
    write_row_csv,
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def run_single_configuration(
    *,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: np.ndarray,
    seed: int,
    model_name: str,
    feature_cols: List[str],
    featureset_name: str,
    calibration_method: str,
    tuned_hgb_params: Dict[str, object],
    out_cv_path: Path,
    out_test_path: Path,
) -> Tuple[Path, Path]:
    y = df[TARGET_COL].astype(int)
    categorical_covariates = [c for c in ["q1", "q2", "q3", "raceeth"] if c in feature_cols]
    x = prepare_feature_frame(df, feature_cols, categorical_covariates)

    x_train = x.loc[train_idx]
    y_train = y.loc[train_idx]
    x_test = x.loc[test_idx]
    y_test = y.loc[test_idx]

    def estimator_factory():
        pre = build_preprocessor(feature_cols=feature_cols, categorical_cols=categorical_covariates)
        return build_estimator(
            model_name=model_name,
            preprocessor=pre,
            seed=seed,
            tuned_hgb_params=tuned_hgb_params if model_name == "hgb" else None,
        )

    fold_df = run_cv_metrics_with_leakage_safe_calibration(
        estimator_factory=estimator_factory,
        x_train=x_train,
        y_train=y_train,
        fold_id=fold_id,
        calibration_method=calibration_method,
        seed=seed,
    )
    cv_metrics = aggregate_cv_metrics(fold_df)
    cv_row = {
        "split_scope": "cv_mean",
        "seed": seed,
        "model": model_name,
        "featureset": featureset_name,
        "calibration_method": calibration_method,
        "n_train": int(len(train_idx)),
        "cv_folds": int(CV_FOLDS),
        **{k: cv_metrics[k] for k in REQUIRED_METRIC_COLS},
        **{f"{k}_std": cv_metrics[f"{k}_std"] for k in REQUIRED_METRIC_COLS},
    }
    write_row_csv(out_cv_path, cv_row)
    validate_metric_fields(out_cv_path)

    test_metrics = run_test_metrics_with_train_only_calibration(
        estimator_factory=estimator_factory,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        fold_id=fold_id,
        calibration_method=calibration_method,
    )
    test_row = {
        "split_scope": "heldout_test",
        "seed": seed,
        "model": model_name,
        "featureset": featureset_name,
        "calibration_method": calibration_method,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **{k: test_metrics[k] for k in REQUIRED_METRIC_COLS},
    }
    write_row_csv(out_test_path, test_row)
    validate_metric_fields(out_test_path)

    return out_cv_path, out_test_path


def append_logs(
    *,
    experiment_log_path: Path,
    decisions_log_path: Path,
    seed: int,
    feature_sets,
    equality_triggered: bool,
    commands_used: List[str],
) -> None:
    date_str = datetime.now(timezone.utc).date().isoformat()

    exp_lines = []
    exp_lines.append("")
    exp_lines.append(f"## (PERFORMED) | {date_str} | Week 6 | Full-feature, ablation, and calibration analysis")
    exp_lines.append("- Commands used:")
    for cmd in commands_used:
        exp_lines.append(f"  - `{cmd}`")
    exp_lines.append("- Feature definitions:")
    exp_lines.append(f"  - `baseline_features = {feature_sets.baseline_features}`")
    exp_lines.append(f"  - `full_features = {feature_sets.full_features}`")
    exp_lines.append(f"  - `full_minus_bullying_features = {feature_sets.full_minus_bullying_features}`")
    exp_lines.append("- Calibration protocol:")
    exp_lines.append("  - CV: calibrator fit on training-fold probabilities only.")
    exp_lines.append("  - Held-out: calibrator fit on train-only OOF probabilities.")
    exp_lines.append("- Tradeoff analysis:")
    exp_lines.append("  - Week 6 tables report ranking and calibration deltas jointly for full-feature, ablation, and calibration sensitivity comparisons.")
    if equality_triggered:
        exp_lines.append("- Equality case: `full_minus_bullying_features` equals `baseline_features` under current modeling-table scope.")
    exp_lines.append("- Deviations: None")
    with experiment_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(exp_lines) + "\n")

    dec_lines = []
    dec_lines.append("")
    dec_lines.append(f"## D014 (PERFORMED) | {date_str} | Week 6")
    dec_lines.append("- Decision: Execute Week 6 full-feature comparison, bullying-block ablation, and calibration sensitivity under frozen seed 2026 protocol.")
    dec_lines.append("- Rationale: The locked analysis plan requires these comparisons under unchanged split, fold, and baseline-reference artifacts.")
    if equality_triggered:
        dec_lines.append("- Equality condition: `full_minus_bullying_features` equals `baseline_features` due current modeling-table predictor scope.")
        dec_lines.append("- Handling policy: Continue execution and document this as a structural limitation, not a protocol violation.")
    dec_lines.append("- Evidence:")
    dec_lines.append("  - `outputs/tables/week06_full_feature_comparison_seed2026.csv`")
    dec_lines.append("  - `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`")
    dec_lines.append("  - `outputs/tables/week06_calibration_sensitivity_seed2026.csv`")
    dec_lines.append("- Deviations: None")
    with decisions_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(dec_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 6 full-feature, ablation, and calibration analysis.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-perm-repeats", type=int, default=30)
    args = parser.parse_args()

    if args.seed != 2026:
        raise RuntimeError("Week 6 frozen protocol requires seed 2026.")

    out_metrics = args.outdir / "metrics"
    out_tables = args.outdir / "tables"
    out_figures = args.outdir / "figures"

    modeling_path = PROJECT_ROOT / "data" / "processed" / "yrbs_2023_modeling.parquet"
    holdout_path = args.outdir / "splits" / "holdout_seed2026.npz"
    cvfolds_path = args.outdir / "splits" / "cvfolds_seed2026.npz"
    frozen_metric_cv_logreg = out_metrics / "metrics_cv_seed2026_logreg_baseline_none.csv"
    frozen_metric_test_logreg = out_metrics / "metrics_test_seed2026_logreg_baseline_none.csv"
    frozen_metric_cv_hgb = out_metrics / "metrics_cv_seed2026_hgb_baseline_none.csv"
    frozen_metric_test_hgb = out_metrics / "metrics_test_seed2026_hgb_baseline_none.csv"
    tuned_path = args.outdir / "tuning" / "hgb_seed2026_baseline_best_params.json"

    required_paths = [
        modeling_path,
        holdout_path,
        cvfolds_path,
        frozen_metric_cv_logreg,
        frozen_metric_test_logreg,
        frozen_metric_cv_hgb,
        frozen_metric_test_hgb,
        tuned_path,
    ]
    require_paths(required_paths)

    pre_hashes = snapshot_hashes(required_paths)

    df = pd.read_parquet(modeling_path)
    rec_train_idx, rec_test_idx, rec_fold_id = recompute_split_and_fold_artifacts(
        df=df,
        target_col=TARGET_COL,
        seed=args.seed,
        test_size=TEST_SIZE,
        cv_folds=CV_FOLDS,
    )
    train_idx, test_idx, fold_id = verify_frozen_split_and_fold_equality(
        holdout_npz=holdout_path,
        cvfolds_npz=cvfolds_path,
        recomputed_train_idx=rec_train_idx,
        recomputed_test_idx=rec_test_idx,
        recomputed_fold_id=rec_fold_id,
        atol=1e-12,
    )

    feature_sets = derive_feature_sets(df.columns.tolist())

    tuned_payload = json.loads(tuned_path.read_text(encoding="utf-8"))
    tuned_best_params = tuned_payload.get("best_params", {})
    if not isinstance(tuned_best_params, dict) or not tuned_best_params:
        raise RuntimeError("Could not load Week 5 tuned HGB parameters from JSON.")

    metrics_paths: Dict[Tuple[str, str, str, str], Path] = {}

    run_specs = [
        (
            "logreg",
            feature_sets.full_features,
            "full",
            "none",
            out_metrics / "metrics_cv_seed2026_logreg_full_none.csv",
            out_metrics / "metrics_test_seed2026_logreg_full_none.csv",
        ),
        (
            "hgb",
            feature_sets.full_features,
            "full",
            "none",
            out_metrics / "metrics_cv_seed2026_hgb_full_none.csv",
            out_metrics / "metrics_test_seed2026_hgb_full_none.csv",
        ),
        (
            "hgb",
            feature_sets.full_minus_bullying_features,
            "full_minus_bullying",
            "none",
            out_metrics / "metrics_cv_seed2026_hgb_full_minus_bullying.csv",
            out_metrics / "metrics_test_seed2026_hgb_full_minus_bullying.csv",
        ),
        (
            "hgb",
            feature_sets.full_features,
            "full",
            "platt",
            out_metrics / "metrics_cv_seed2026_hgb_full_platt.csv",
            out_metrics / "metrics_test_seed2026_hgb_full_platt.csv",
        ),
        (
            "hgb",
            feature_sets.full_features,
            "full",
            "isotonic",
            out_metrics / "metrics_cv_seed2026_hgb_full_isotonic.csv",
            out_metrics / "metrics_test_seed2026_hgb_full_isotonic.csv",
        ),
    ]

    for model_name, feature_cols, featureset_name, calibration_method, cv_out, test_out in run_specs:
        run_single_configuration(
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_id=fold_id,
            seed=args.seed,
            model_name=model_name,
            feature_cols=feature_cols,
            featureset_name=featureset_name,
            calibration_method=calibration_method,
            tuned_hgb_params=tuned_best_params,
            out_cv_path=cv_out,
            out_test_path=test_out,
        )
        metrics_paths[("cv", model_name, featureset_name, calibration_method)] = cv_out
        metrics_paths[("test", model_name, featureset_name, calibration_method)] = test_out

    baseline_paths = {
        "cv": frozen_metric_cv_logreg,
        "test": frozen_metric_test_logreg,
    }

    full_table_path = out_tables / "week06_full_feature_comparison_seed2026.csv"
    ablation_table_path = out_tables / "week06_bullying_ablation_comparison_seed2026.csv"
    calibration_table_path = out_tables / "week06_calibration_sensitivity_seed2026.csv"

    full_df, ablation_df, calibration_df = build_week06_tables(
        metrics_paths=metrics_paths,
        baseline_paths=baseline_paths,
        out_full_table=full_table_path,
        out_ablation_table=ablation_table_path,
        out_cal_table=calibration_table_path,
    )

    full_fig_path = out_figures / "week06_full_feature_comparison_seed2026.png"
    ablation_fig_path = out_figures / "week06_bullying_ablation_comparison_seed2026.png"
    cal_fig_path = out_figures / "week06_calibration_sensitivity_seed2026.png"
    build_week06_figures(
        full_df=full_df,
        ablation_df=ablation_df,
        cal_df=calibration_df,
        out_full_fig=full_fig_path,
        out_ablation_fig=ablation_fig_path,
        out_cal_fig=cal_fig_path,
    )

    # Permutation-importance stability for HGB full none.
    y = df[TARGET_COL].astype(int)
    cat_covars = [c for c in ["q1", "q2", "q3", "raceeth"] if c in feature_sets.full_features]
    x_full = prepare_feature_frame(df, feature_sets.full_features, cat_covars)
    x_train_full = x_full.loc[train_idx]
    y_train = y.loc[train_idx]

    def hgb_full_factory():
        pre = build_preprocessor(feature_cols=feature_sets.full_features, categorical_cols=cat_covars)
        return build_estimator(
            model_name="hgb",
            preprocessor=pre,
            seed=args.seed,
            tuned_hgb_params=tuned_best_params,
        )

    by_fold_df, summary_df = run_permutation_importance_stability(
        estimator_factory=hgb_full_factory,
        x_train=x_train_full,
        y_train=y_train,
        fold_id=fold_id,
        n_repeats=args.n_perm_repeats,
        seed=args.seed,
    )
    by_fold_path = out_tables / "hgb_seed2026_full_perm_importance_by_fold.csv"
    summary_path = out_tables / "hgb_seed2026_full_perm_importance_summary.csv"
    by_fold_df.to_csv(by_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    commands = [
        f".venv/bin/python scripts/07_week06_pipeline.py --seed {args.seed}",
    ]
    append_logs(
        experiment_log_path=PROJECT_ROOT / "docs" / "experiment_log.md",
        decisions_log_path=PROJECT_ROOT / "docs" / "decisions_log.md",
        seed=args.seed,
        feature_sets=feature_sets,
        equality_triggered=feature_sets.equals_baseline,
        commands_used=commands,
    )

    post_hashes = snapshot_hashes(required_paths)
    verify_hashes_unchanged(pre_hashes, post_hashes)

    print(
        json.dumps(
            {
                "status": "ok",
                "tables": sorted([_rel(full_table_path), _rel(ablation_table_path), _rel(calibration_table_path)]),
                "figures": sorted([_rel(full_fig_path), _rel(ablation_fig_path), _rel(cal_fig_path)]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
