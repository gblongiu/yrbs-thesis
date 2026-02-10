# Modeling Report (Week 4 Baseline)

Status: PERFORMED and complete for Week 4 scope (as of 2026-02-09).

## Scope
This report is limited to the proposal-defined Week 4 baseline deliverable:
- baseline logistic model;
- frozen split protocol;
- cross-validation and held-out test metrics.

No Week 5+ boosted, ablation, or interpretability completion claims are made in this report.

## Run Configuration
- Command:
  - `.venv/bin/python scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`
- Seed: `2026`
- Model: `logreg`
- Feature set: `baseline`
- Calibration method: `none`

## Survey design and weights policy (Week 4)
- Policy: weighted analyses are used in EDA for descriptive prevalence to reflect the YRBS sampling design.
- Policy: Week 4 baseline predictive modeling is fit and evaluated unweighted as the primary analysis.
- Policy: weighted-fit sensitivity is planned future work and is not implemented in Week 4.

## Core Week 4 Artifacts
- `outputs/splits/holdout_seed2026.npz`
- `outputs/splits/cvfolds_seed2026.npz`
- `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`

## Metrics Summary

### Cross-validation (training folds)
Source: `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`

- ROC AUC mean: `0.651014`
- PR AUC mean: `0.534537`
- Brier mean: `0.232424`
- Calibration slope mean: `0.981693`
- Calibration intercept mean: `-0.371663`

### Held-out test set
Source: `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`

- ROC AUC: `0.649822`
- PR AUC: `0.547476`
- Brier: `0.231985`
- Calibration slope: `0.987174`
- Calibration intercept: `-0.363811`
- Train rows: `15890`
- Test rows: `3973`

## Week 4 Completion Statement
Week 4 proposal deliverables are met:
- baseline model trained and evaluated;
- frozen split artifacts persisted;
- baseline metric report documented.
