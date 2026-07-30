# Modeling Report

This report summarizes the retained final modeling results from Weeks 4 through 7.

## Feature Set Construction
- Baseline covariates: `q1`, `q2`, `q3`, `raceeth`
- Full feature set: baseline covariates plus `x_qn24` and `x_qn25`
- `full_features` are derived from the modeling table after excluding targets, survey-design fields, identifiers, and known post-event leakage columns.
- Under the retained predictor scope, `full_minus_bullying_features` equals the locked baseline covariate set. The Week 6 ablation should therefore be read as the incremental bullying-block comparison within the final thesis predictor boundary.

## Frozen Validation Protocol
- Seed: `2026`
- Held-out split artifact: `outputs/splits/holdout_seed2026.npz`
- Cross-validation fold artifact: `outputs/splits/cvfolds_seed2026.npz`
- Frozen tuned HGB parameter artifact: `outputs/tuning/hgb_seed2026_baseline_best_params.json`

## Week 4 Baseline Logistic Reference
- Held-out ROC AUC: `0.649822`
- Held-out PR AUC: `0.547476`
- Held-out Brier: `0.231985`
- Held-out calibration slope: `0.987174`
- Held-out calibration intercept: `-0.363811`

Core evidence:
- `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`

## Week 5 Tuned HGB Baseline Comparator
- Held-out ROC AUC: `0.650200`
- Held-out PR AUC: `0.537872`
- Held-out Brier: `0.225224`
- Held-out calibration slope: `1.071831`
- Held-out calibration intercept: `0.036123`

Best retained tuning parameters:
- `learning_rate = 0.01`
- `max_depth = 5`
- `max_iter = 600`
- `min_samples_leaf = 120`
- `max_leaf_nodes = 15`
- `l2_regularization = 0.01`

Core evidence:
- `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`
- `outputs/tuning/hgb_seed2026_baseline_best_params.json`
- `outputs/tables/week05_calibration_comparison_seed2026.csv`
- `outputs/tables/hgb_seed2026_baseline_perm_importance_summary.csv`

## Week 6 Full-Feature Comparison
Adding the bullying block improved held-out performance relative to the frozen logistic baseline:
- ROC AUC delta: `0.066235`
- PR AUC delta: `0.096596`
- Brier delta: `-0.025807`

Bullying-block ablation relative to the non-bullying comparator:
- ROC AUC delta: `0.065857`
- PR AUC delta: `0.106201`
- Brier delta: `-0.019046`

Calibration sensitivity on held-out data for the HGB full candidate:
- None: Brier `0.206178`, slope `1.046301`
- Platt: Brier `0.206122`, slope `1.005455`
- Isotonic: Brier `0.206546`, slope `0.980255`

Core evidence:
- `outputs/tables/week06_full_feature_comparison_seed2026.csv`
- `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`
- `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
- `outputs/figures/week06_calibration_sensitivity_seed2026.png`

## Week 7 Model Selection And Robustness
Selected final headline model: `hgb_full_platt`

Ranking rule:
1. Lowest held-out Brier score
2. Calibration slope closest to `1`
3. Held-out ROC AUC as the tie-breaker

Held-out metrics for the selected model:
- ROC AUC: `0.716057`
- PR AUC: `0.644073`
- Brier: `0.206122`
- Calibration slope: `1.005455`
- Calibration intercept: `0.014371`

Retained robustness summaries:
- `outputs/tables/multiseed_stability_seed2026_2029.csv`
- `outputs/tables/heldout_bootstrap_ci_seed2026.csv`
- `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`
- `outputs/tables/subgroup_performance_seed2026.csv`
- `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`

Robustness boundary:
- The multiseed, bootstrap, hyperparameter, subgroup, and permutation-importance summaries describe the retained full HGB candidate family under the frozen split protocol.
- The final calibration choice is reported separately through the Week 6 calibration-sensitivity package and the selected held-out metrics file for `hgb_full_platt`.

## Interpretation Boundary
All reported results are predictive associations under the fixed protocol. The study does not claim causal effects and does not present the selected model as deployment-ready.
