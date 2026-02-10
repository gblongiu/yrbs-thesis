# Decisions Log (Performed vs Planned)

This register tracks Week 1-4 implemented decisions and future planned decisions.

## PERFORMED Decisions (Weeks 1-4)

## D001 (PERFORMED) | 2026-01-16 | Week 1
- Decision: Primary target locked to `QN26` / `y_qn26`.
- Rationale: Keep one headline endpoint and avoid scope diffusion.
- Evidence:
  - `src/config.py`
  - `docs/data_dictionary.md`
  - `scripts/01_build_dataset.py`

## D002 (PERFORMED) | 2026-01-16 | Week 1
- Decision: Primary exposures locked to `QN24` and `QN25` (`x_qn24`, `x_qn25`).
- Rationale: Align features to thesis question and preserve interpretability.
- Evidence:
  - `src/config.py`
  - `scripts/01_build_dataset.py`
  - `docs/covariates_proposal.md`

## D003 (PERFORMED) | 2026-01-16 | Week 1
- Decision: Baseline covariates locked to `q1`, `q2`, `q3`, `raceeth`.
- Rationale: Demographic adjustment with manageable complexity.
- Evidence:
  - `src/config.py`
  - `docs/covariates_proposal.md`
  - `docs/data_dictionary.md`

## D004 (PERFORMED) | 2026-01-30 | Week 2
- Decision: Recode binary YRBS items into `{0,1,NA}` with explicit special-code handling.
- Rationale: Remove coding ambiguity and preserve auditability.
- Evidence:
  - `src/data/coding.py`
  - `scripts/01_build_dataset.py`
  - `outputs/logs/decisions.json`

## D005 (PERFORMED) | 2026-01-30 | Week 2
- Decision: Drop rows with missing primary outcome `y_qn26`.
- Rationale: Outcome-missing rows cannot support supervised training and evaluation.
- Evidence:
  - `scripts/01_build_dataset.py`
  - `outputs/tables/modeling_table_audit.csv`
  - `docs/data_dictionary.md`

## D006 (PERFORMED) | 2026-01-30 | Week 2
- Decision: Preserve design fields (`weight`, `stratum`, `psu`) without imputation.
- Rationale: Keep survey-design context available while excluding it from predictive features.
- Evidence:
  - `src/config.py`
  - `scripts/01_build_dataset.py`
  - `docs/data_dictionary.md`

## D007 (PERFORMED) | 2026-02-06 | Week 3
- Decision: Keep weighted descriptive context separate from unweighted predictive metrics.
- Rationale: Prevent scope and interpretation drift.
- Evidence:
  - `scripts/02_eda.py`
  - `scripts/03_train_models.py`
  - `docs/eda_report.md`
  - `docs/modeling_report.md`

## D008 (PERFORMED) | 2026-02-06 | Week 3
- Decision: Use approximate weight-only confidence intervals for descriptive prevalence.
- Rationale: Provide context while avoiding unsupported design-based variance claims.
- Evidence:
  - `scripts/02_eda.py`
  - `docs/eda_report.md`

## D009 (PERFORMED) | 2026-02-06 | Week 4
- Decision: Freeze protocol constants (`TEST_SIZE`, `CV_FOLDS`, `RANDOM_SEEDS`).
- Rationale: Ensure reproducible split and evaluation behavior.
- Evidence:
  - `src/config.py`
  - `scripts/03_train_models.py`
  - `outputs/splits/holdout_seed2026.npz`
  - `outputs/splits/cvfolds_seed2026.npz`

## D010 (PERFORMED) | 2026-02-09 | Week 4
- Decision: Complete Week 4 baseline deliverables with logistic baseline (`logreg`, `baseline`, seed `2026`).
- Rationale: Meet proposal Week 4 requirements without claiming Week 5+ completion.
- Evidence:
  - `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
  - `docs/modeling_report.md`

## D011 (PERFORMED) | 2026-02-09 | Week 4
- Decision: Weights policy (EDA vs modeling).
- Rationale: Week 4 uses survey weights in EDA because descriptive prevalence should reflect the YRBS sampling design. The baseline predictive workflow remains unweighted so that model fitting and held-out evaluation are directly comparable under one fixed protocol and consistent with the Week 4 baseline objective. This separation avoids mixing descriptive weighting decisions into the primary predictive benchmark before robustness checks are defined. Weighted-fit modeling is deferred to planned sensitivity work so it can be evaluated explicitly as a robustness branch rather than merged into the primary Week 4 analysis.
- Evidence:
  - `docs/eda_report.md`
  - `docs/modeling_report.md`
  - `docs/experiment_log.md`
  - `outputs/tables/weighted_prevalence_overall.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`

## PLANNED Decisions (Week 5+)
- Planned: boosted-model tuning and comparison policy (Week 5).
- Planned: ablation and interpretability decision package (Week 6).
- Planned: sensitivity and claim-audit policy for submission readiness (Week 8+).

Planned document anchor:
- `docs/ablation_report.md`