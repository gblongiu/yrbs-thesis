# Model Selection Framework

## Scope
This framework defines the Week 7 model selection rule for the reproducible YRBS thesis pipeline.
It applies to the fixed Week 4 holdout split with seed 2026 and to additive Week 7 robustness outputs.

## Candidate Models
- HGB full none: `outputs/metrics/metrics_test_seed2026_hgb_full_none.csv`
- HGB full platt: `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
- HGB full isotonic: `outputs/metrics/metrics_test_seed2026_hgb_full_isotonic.csv`
- Logistic full none: `outputs/metrics/metrics_test_seed2026_logreg_full_none.csv`

## Primary and Secondary Criteria
1. Primary objective: lowest held-out Brier score.
2. Secondary objective: calibration slope closest to 1 on held-out data.
3. Tertiary objective: held-out ROC AUC as ranking tiebreaker.

Evidence paths:
- `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- `outputs/tables/week06_full_feature_comparison_seed2026.csv`

## Stability and Robustness Constraints
A model is eligible for selection only if all constraints pass.

1. Multiseed stability constraint
- Requirement: `brier_std_across_seeds < 0.005`.
- Evidence: `outputs/tables/multiseed_stability_seed2026_2029.csv`.

2. Hyperparameter local stability constraint
- Requirement: perturbation analysis does not change decision ranking by Brier.
- Evidence: `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`.

3. Subgroup slope constraint
- Requirement: no flagged subgroup with defined slope below 0.8.
- Evidence: `outputs/tables/subgroup_performance_seed2026.csv`.

4. Subgroup error constraint
- Requirement: investigate and document any subgroup with `high_error_flag = True` before selection finalization.
- Evidence: `outputs/tables/subgroup_performance_seed2026.csv`.

## Interpretability and Importance Stability
- Inspect stability for bullying features using fold-level permutation importance with negative Brier scoring.
- Evidence:
  - `outputs/tables/hgb_seed2026_full_perm_importance_by_fold.csv`
  - `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`

## Structural Limitation Statement
Under the current modeling table scope, `full_minus_bullying_features` equals baseline covariates.
This limits the ablation control to the currently available predictor set and should be interpreted as a scope-bound comparison.
Evidence:
- `docs/status_reports/report_03/feature_set_definitions.md`
- `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`

## Decision Record Requirement
Any final selection decision must cite all required evidence artifacts and must use non-causal language.
Decision write-up target paths:
- `docs/decisions_log.md`
- `docs/status_reports/report_03/Project_Status_Report_03_Submission.md`
