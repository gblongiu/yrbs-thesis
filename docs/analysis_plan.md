# Analysis Plan

This file records the implemented final analysis sequence for the thesis repository.

## Locked Research Question
Do bullying on school property (`QN24`) and electronic bullying (`QN25`) add incremental predictive value for persistent sadness or hopelessness (`QN26`) beyond the locked demographic baseline (`q1`, `q2`, `q3`, `raceeth`) under a fixed validation protocol?

## Frozen Validation Protocol
- Seed: `2026`
- Held-out split artifact: `outputs/splits/holdout_seed2026.npz`
- Cross-validation fold artifact: `outputs/splits/cvfolds_seed2026.npz`
- Frozen baseline reference metrics:
  - `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`
- Frozen tuned HGB parameter file: `outputs/tuning/hgb_seed2026_baseline_best_params.json`

## Implemented Feature Sets
- `baseline_features = ['q1', 'q2', 'q3', 'raceeth']`
- `full_features = ['x_qn24', 'x_qn25', 'q1', 'q2', 'q3', 'raceeth']`
- `full_minus_bullying_features = ['q1', 'q2', 'q3', 'raceeth']`

Under the final modeling-table scope, `full_minus_bullying_features` equals the locked baseline covariate set. That equality is treated as a structural limitation of the retained predictor set, not as a protocol failure.

## Final Workflow
1. Validate the local environment and raw workbook presence.
2. Audit the source workbook schema and key raw-value distributions.
3. Build the analysis-ready modeling table.
4. Produce descriptive EDA tables and figures.
5. Fit the frozen logistic and HGB baseline references.
6. Run the Week 6 full-feature comparison, bullying-block ablation, and calibration sensitivity package.
7. Run the Week 7 robustness summaries that remain in scope for the final thesis.
8. Build the curated submission pack from canonical repository sources.

## Retained Scientific Outputs
- Dataset construction: `outputs/tables/modeling_table_audit.csv`, `outputs/tables/missingness_modeling.csv`
- EDA: `outputs/tables/missingness_eda.csv`, `outputs/tables/weighted_prevalence_overall.csv`, `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
- Week 4-5 baseline references: baseline metric CSVs, `outputs/tables/week05_calibration_comparison_seed2026.csv`, `outputs/tables/hgb_seed2026_baseline_perm_importance_summary.csv`
- Week 6 comparison package:
  - `outputs/tables/week06_full_feature_comparison_seed2026.csv`
  - `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`
  - `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
- Week 7 robustness package:
  - `outputs/tables/multiseed_stability_seed2026_2029.csv`
  - `outputs/tables/heldout_bootstrap_ci_seed2026.csv`
  - `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`
  - `outputs/tables/subgroup_performance_seed2026.csv`
  - `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`

## Interpretation Boundary
- Weighted estimates are descriptive context only.
- Predictive conclusions come from unweighted held-out metrics under the frozen protocol.
- The final selection of `hgb_full_platt` is a calibration choice layered on top of the retained full HGB candidate family; some Week 7 robustness summaries therefore continue to describe the uncalibrated full HGB comparison object.
- Findings are reported as predictive associations, not causal effects.
