# Experiment Log (Performed vs Planned)

Status checkpoint (recorded 2026-02-24):
- Weeks 1 to 6 deliverables are complete.
- Selected Week 7 robustness and governance artifacts are complete ahead of schedule.
- Week 8 is the next milestone for draft integration.

## PERFORMED (Weeks 1-5)

## (PERFORMED) | 2026-02-09 | Week 1 | Environment validation
- What ran: `python3 scripts/00_validate_environment.py`
- Outputs:
  - `outputs/logs/environment_check.json`

## (PERFORMED) | 2026-02-09 | Week 1 | Schema audit and variable inventory
- What ran: `python3 scripts/00_schema_audit.py`
- Inputs:
  - `data/raw/YRBS_2023_MH_subset.xlsx`
- Outputs:
  - `outputs/tables/schema.csv`
  - `outputs/tables/missingness_summary.csv`
  - `outputs/tables/value_counts_QN24.csv`
  - `outputs/tables/value_counts_QN25.csv`
  - `outputs/tables/value_counts_QN26.csv`

## (PERFORMED) | 2026-02-09 | Week 2 | Build analysis-ready modeling table
- What ran: `python3 scripts/01_build_dataset.py`
- Outputs:
  - `data/processed/yrbs_2023_modeling.parquet`
  - `outputs/tables/modeling_table_audit.csv`
  - `outputs/tables/missingness_modeling.csv`
  - `outputs/logs/decisions.json`

## (PERFORMED) | 2026-02-09 | Week 3 | EDA tables and figures
- What ran: `python3 scripts/02_eda.py --outdir outputs`
- Outputs:
  - `outputs/tables/missingness_eda.csv`
  - `outputs/tables/unweighted_prevalence_overall.csv`
  - `outputs/tables/weighted_prevalence_overall.csv`
  - `outputs/tables/weighted_prevalence_by_q1.csv`
  - `outputs/tables/weighted_prevalence_by_q2.csv`
  - `outputs/tables/weighted_prevalence_by_q3.csv`
  - `outputs/tables/weighted_prevalence_by_raceeth.csv`
  - `outputs/figures/missingness_bar.png`
  - `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
  - `outputs/figures/qn26_prevalence_by_q1.png`
  - `outputs/figures/qn26_prevalence_by_raceeth.png`
  - `outputs/logs/eda_run_metadata.json`

## (PERFORMED) | 2026-02-09 | Week 4 | Baseline logistic model under frozen split protocol
- What ran: `python3 scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`
- Outputs:
  - `outputs/splits/holdout_seed2026.npz`
  - `outputs/splits/cvfolds_seed2026.npz`
  - `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
  - `docs/modeling_report.md`

## (PERFORMED) | 2026-02-18 | Week 5 | HGB tuning on frozen protocol training partition
- Date: `2026-02-18`
- Run ID: `week05_models_v1_seed2026_hgb_baseline_none`
- Git commit at run start: `d4cce49`
- Command used:
  - `.venv/bin/python scripts/03_train_models.py --model hgb --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs --run-id week05_models_v1_seed2026_hgb_baseline_none --tune_hgb 1 --hgb_search_iter 12 --save_cv_preds 1 --enforce_frozen_artifacts 1 --week5_artifacts_only 1`
- Primary artifacts produced:
  - `outputs/tuning/hgb_seed2026_baseline_search_results.csv`
  - `outputs/tuning/hgb_seed2026_baseline_best_params.json`
- Interpretation:
  - The tuning search selected a conservative boosted configuration with `learning_rate=0.01`, `max_depth=5`, `max_iter=600`, `min_samples_leaf=120`, `max_leaf_nodes=15`, and `l2_regularization=0.01`.
  - The best cross-validation ROC AUC for the search was `0.650538`, which is near the Week 4 baseline ROC AUC mean and indicates limited rank-order gain but stable optimization under the frozen protocol.

## (PERFORMED) | 2026-02-18 | Week 5 | Final tuned HGB evaluation and diagnostics
- Date: `2026-02-18`
- Run ID: `week05_models_v1_seed2026_hgb_baseline_none`
- Git commit at run start: `d4cce49`
- Commands used:
  - `.venv/bin/python scripts/03_train_models.py --model hgb --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs --run-id week05_models_v1_seed2026_hgb_baseline_none --tune_hgb 1 --hgb_search_iter 12 --save_cv_preds 1 --enforce_frozen_artifacts 1 --week5_artifacts_only 1`
  - `.venv/bin/python scripts/04_week05_diagnostics.py --model hgb --baseline-model logreg --features baseline --seed 2026 --calibration none --outdir outputs`
- Primary artifacts produced:
  - `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`
  - `outputs/metrics/metrics_cv_folds_seed2026_hgb_baseline_none.csv`
  - `outputs/tables/week05_calibration_comparison_seed2026.csv`
  - `outputs/figures/week05_calibration_comparison_seed2026.png`
  - `outputs/tables/hgb_seed2026_baseline_perm_importance_by_fold.csv`
  - `outputs/tables/hgb_seed2026_baseline_perm_importance_summary.csv`
  - `outputs/figures/hgb_seed2026_baseline_importance_stability.png`
- Interpretation:
  - On held-out test data, HGB improved Brier (`0.225224` vs `0.231985`) and slightly improved ROC AUC (`0.650200` vs `0.649822`) relative to the Week 4 baseline, while PR AUC decreased (`0.537872` vs `0.547476`).
  - Calibration intercept moved from `-0.363811` to `0.036123`, while calibration slope increased from `0.987174` to `1.071831`, indicating less systematic underprediction but steeper probability scaling.

## PLANNED (Weeks 8-10)

- Week 8: first full paper draft and reproducibility appendix.
- Week 9: revision and presentation plan.
- Week 10: final submission bundle.

Planned anchor document:
- `docs/ablation_report.md`

## (PERFORMED) | 2026-02-22 | Week 6 | Full-feature, ablation, and calibration package
- Commands used:
  - `.venv/bin/python scripts/07_week06_pipeline.py --seed 2026`
  - `.venv/bin/python scripts/08_week06_report_package.py --seed 2026`
  - `.venv/bin/pytest -q`
- Feature definitions:
  - `baseline_features = ['q1', 'q2', 'q3', 'raceeth']`
  - `full_features = ['x_qn24', 'x_qn25', 'q1', 'q2', 'q3', 'raceeth']`
  - `full_minus_bullying_features = ['q1', 'q2', 'q3', 'raceeth']`
- Calibration protocol:
  - CV: calibrator fit on training-fold probabilities only.
  - Held-out: calibrator fit on train-only OOF probabilities.
- Tradeoff analysis:
  - Week 6 tables report ranking and calibration deltas jointly for full-feature, ablation, and calibration sensitivity comparisons.
- Equality case: `full_minus_bullying_features` equals `baseline_features` under current modeling-table scope.
- Deviations: None

## (PERFORMED) | 2026-02-24 | Week 7 | Selected robustness and governance tasks completed ahead of schedule
- Commands used:
  - `.venv/bin/python scripts/09_multiseed_stability.py`
  - `.venv/bin/python scripts/10_bootstrap_ci.py`
  - `.venv/bin/python scripts/11_hyperparameter_sensitivity.py`
  - `.venv/bin/python scripts/12_subgroup_audit.py`
  - `.venv/bin/python scripts/13_upgrade_integrity_check.py`
- Ahead-of-schedule artifact evidence:
  - `outputs/tables/multiseed_stability_seed2026_2029.csv`
  - `outputs/tables/heldout_bootstrap_ci_seed2026.csv`
  - `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`
  - `outputs/tables/subgroup_performance_seed2026.csv`
  - `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`
  - `outputs/preds/preds_test_hgb_full_none_seed2026.csv`
  - `outputs/audits/week07_upgrade_integrity_audit.md`
  - `model_selection_framework.md`
  - `deployment_and_use_constraints.md`
- Frozen protocol continuity:
  - Week 6 core outputs remain unchanged.
  - Frozen Week 4 and Week 5 artifacts remain unchanged.
  - Integrity evidence: `outputs/audits/week07_upgrade_integrity_audit.md`, `docs/status_reports/report_03/protocol_lock_confirmation.md`.
