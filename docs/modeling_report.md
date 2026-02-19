# Modeling Report (Weeks 4-5)

Status: Week 4 and Week 5 modeling deliverables are performed and complete as of 2026-02-18.

## Scope
This report covers:
- Week 4 baseline logistic model under frozen protocol
- Week 5 tuned boosted model comparison under the same protocol
- early calibration checks and fold-level stability diagnostics requested for Week 5

## Shared Validation Protocol
- Seed: `2026`
- Test split policy: frozen holdout artifact in `outputs/splits/holdout_seed2026.npz`
- CV policy: frozen fold artifact in `outputs/splits/cvfolds_seed2026.npz`
- Feature scope in Week 5: `baseline`
- Calibration mode in Week 5 primary run: `none`

## Week 4 Baseline Reference
Command:
- `python3 scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`

Core artifacts:
- `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`

Held-out baseline metrics:
- ROC AUC: `0.649822`
- PR AUC: `0.547476`
- Brier: `0.231985`
- Calibration slope: `0.987174`
- Calibration intercept: `-0.363811`

## Week 5 Tuned HGB Run
Run ID:
- `week05_models_v1_seed2026_hgb_baseline_none`

Commands:
- `.venv/bin/python scripts/03_train_models.py --model hgb --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs --run-id week05_models_v1_seed2026_hgb_baseline_none --tune_hgb 1 --hgb_search_iter 12 --save_cv_preds 1 --enforce_frozen_artifacts 1 --week5_artifacts_only 1`
- `.venv/bin/python scripts/04_week05_diagnostics.py --model hgb --baseline-model logreg --features baseline --seed 2026 --calibration none --outdir outputs`

Tuning method:
- RandomizedSearchCV on training partition only with frozen seed and CV policy
- Scoring metric: ROC AUC
- Search iterations: `12`

Best tuned parameters:
- `learning_rate`: `0.01`
- `max_depth`: `5`
- `max_iter`: `600`
- `min_samples_leaf`: `120`
- `max_leaf_nodes`: `15`
- `l2_regularization`: `0.01`
- Best CV ROC AUC from search: `0.650538`

Primary Week 5 artifacts:
- `outputs/tuning/hgb_seed2026_baseline_search_results.csv`
- `outputs/tuning/hgb_seed2026_baseline_best_params.json`
- `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`
- `outputs/tables/week05_calibration_comparison_seed2026.csv`
- `outputs/figures/week05_calibration_comparison_seed2026.png`
- `outputs/tables/hgb_seed2026_baseline_perm_importance_summary.csv`

## Week 5 Comparison vs Baseline

### Cross-validation summary
| Model | ROC AUC mean | PR AUC mean | Brier mean | Calibration slope mean | Calibration intercept mean |
| --- | --- | --- | --- | --- | --- |
| logreg baseline | 0.651014 | 0.534537 | 0.232424 | 0.981693 | -0.371663 |
| tuned hgb | 0.650538 | 0.534563 | 0.225156 | 1.047686 | 0.015911 |

### Held-out test summary
| Model | ROC AUC | PR AUC | Brier | Calibration slope | Calibration intercept |
| --- | --- | --- | --- | --- | --- |
| logreg baseline | 0.649822 | 0.547476 | 0.231985 | 0.987174 | -0.363811 |
| tuned hgb | 0.650200 | 0.537872 | 0.225224 | 1.071831 | 0.036123 |

### Calibration interpretation
- Intercept changed from `-0.363811` to `0.036123`, which indicates reduced systematic underprediction on the held-out test set.
- Slope changed from `0.987174` to `1.071831`, which indicates steeper probability scaling for HGB.
- Brier improved from `0.231985` to `0.225224`, so probability error decreased even with the slope increase.

### Early calibration checks
- Comparison table: `outputs/tables/week05_calibration_comparison_seed2026.csv`
- Comparison figure: `outputs/figures/week05_calibration_comparison_seed2026.png`
- Overlay was built from the existing calibration curve CSV outputs produced by `scripts/03_train_models.py`, preserving the same calibration binning method used in the Week 4 plotting workflow.

## Feature Importance Stability (Week 5)
- Mean Spearman rank stability vs mean rank: `1.0`
- Mean pairwise Jaccard overlap for top-k: `1.0`
- Top-k used: `10`

Per-fold top feature set confirmation from `outputs/tables/hgb_seed2026_baseline_perm_importance_by_fold.csv`:
- fold 0: `q2`, `raceeth`, `q1`, `q3`
- fold 1: `q2`, `raceeth`, `q1`, `q3`
- fold 2: `q2`, `raceeth`, `q1`, `q3`
- fold 3: `q2`, `raceeth`, `q1`, `q3`
- fold 4: `q2`, `raceeth`, `q1`, `q3`
- all fold top-k sets match: `True`

Interpretation:
- The raw-input feature ranking pattern is fully stable across folds in this Week 5 run, with `q2` strongest and `raceeth` second.

Limitation:
- Permutation importance in this implementation is computed at the raw input feature level, not at transformed one-hot feature level.

## Week 5 Completion Statement
Week 5 deliverables are met:
- tuned boosted model run under frozen protocol
- baseline vs boosted calibration comparison artifacts
- fold-level feature importance stability diagnostics
- updated documentation and status-report package
