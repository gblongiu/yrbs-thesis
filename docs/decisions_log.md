# Decisions Log (Performed vs Planned)

This register tracks implemented decisions through Week 7 ahead-of-schedule work and forward planned decisions.

## PERFORMED Decisions (Weeks 1-5)

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

## D012 (PERFORMED) | 2026-02-18 | Week 5
- Decision: Week 5 scope is baseline feature set only for tuned boosted model comparison.
- Rationale: Baseline-only scope keeps the frozen protocol comparison direct against Week 4 and keeps Week 5 effort focused on tuning, calibration checks, and stability diagnostics without mixing in Week 6 ablation objectives.
- Evidence:
  - `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`
  - `outputs/tables/week05_calibration_comparison_seed2026.csv`

## D013 (PERFORMED) | 2026-02-18 | Week 5
- Decision: Defer alternative calibration methods and broader baseline variants to Week 6+.
- Rationale: Week 5 primary comparison holds calibration mode at `none` to isolate model-family change from calibration-method effects. Platt and isotonic were considered and deferred. Alternative baseline variants and larger tuning budgets were also considered but deferred to avoid timeline risk and to preserve interpretability of the Week 5 comparison.
- Week 6 plan:
  - run full-feature comparison under the same frozen protocol
  - run bullying-block ablation after full-feature results
  - assess post-hoc calibration methods (`platt`, `isotonic`) as follow-up sensitivity
- Evidence:
  - `outputs/tuning/hgb_seed2026_baseline_search_results.csv`
  - `outputs/tuning/hgb_seed2026_baseline_best_params.json`
  - `docs/experiment_log.md`

## PLANNED Decisions (Week 8+)
- Planned: sensitivity and claim-audit policy for submission readiness (Week 8+).
- Planned: final paper integration and submission package controls (Week 8 to Week 10).

Planned document anchor:
- `docs/ablation_report.md`

## D014 (PERFORMED) | 2026-02-22 | Week 6
- Decision: Execute Week 6 full-feature comparison, bullying-block ablation, and calibration sensitivity under frozen seed 2026 protocol.
- Rationale: Proposal Week 6 milestone and Status Report 02 forward plan require these analyses under unchanged validation artifacts.
- Equality condition: `full_minus_bullying_features` equals `baseline_features` due current modeling-table predictor scope.
- Handling policy: Continue execution and document this as a structural limitation, not a protocol violation.
- Evidence:
  - `docs/status_reports/report_03/feature_set_definitions.md`
  - `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`
  - `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- Deviations: None

## D015 (PERFORMED) | 2026-02-24 | Week 7 Ahead of Schedule
- Decision: Treat selected Week 7 robustness and governance tasks as completed ahead of schedule while preserving Week 6 submission scope.
- Rationale: Artifacts were produced additively with frozen protocol continuity and no modification to frozen Week 4 and Week 5 artifacts.
- Evidence:
  - `scripts/09_multiseed_stability.py`
  - `scripts/10_bootstrap_ci.py`
  - `scripts/11_hyperparameter_sensitivity.py`
  - `scripts/12_subgroup_audit.py`
  - `scripts/13_upgrade_integrity_check.py`
  - `outputs/tables/multiseed_stability_seed2026_2029.csv`
  - `outputs/tables/heldout_bootstrap_ci_seed2026.csv`
  - `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`
  - `outputs/tables/subgroup_performance_seed2026.csv`
  - `outputs/audits/week07_upgrade_integrity_audit.md`
  - `docs/status_reports/report_03/protocol_lock_confirmation.md`
- Scope statement: Week 6 outputs remain unchanged and frozen checks still pass.

## D016 (PERFORMED) | 2026-03-02 | Week 7 Reporting Lock
- Decision: Use `hgb_full_platt` as the final model choice for Week 7 reporting.
- Selection criteria source: `model_selection_framework.md`.
- Rationale: The held-out candidate comparison places `hgb_full_platt` first by lowest Brier, then slope proximity to 1, with ROC AUC used as tertiary tie break.
- Evidence:
  - `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_isotonic.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_full_none.csv`
  - `docs/status_reports/report_04/week07_model_selection_decision.md`
  - `docs/status_reports/report_04/Project_Status_Report_04_Submission.md`

## D017 (PERFORMED) | 2026-03-02 | Week 7 Scope Protection
- Decision: Keep Week 7 as a documentation-first week and not run new model experiments.
- Rationale: This protects Week 8 manuscript drafting time and preserves frozen protocol continuity while formalizing evidence and decisions for the report.
- Evidence:
  - `docs/experiment_log.md`
  - `docs/status_reports/report_04/Project_Status_Report_04_Submission.md`
  - `docs/status_reports/report_04/week07_run_manifest.json`
  - `docs/manuscript_skeleton.md`
  - `docs/figures_tables_inventory.md`

## D018 (PERFORMED) | 2026-03-02 | Week 7 Reporting Package Strategy
- Decision: Build the Week 7 reporting package from existing Week 6 and Week 7 artifacts, then add only minimal documentation files needed for rubric coverage and Week 8 drafting.
- Rationale: This keeps scope controlled, improves narrative coherence, and avoids redundant regeneration work when evidence artifacts already exist.
- Evidence:
  - `docs/status_reports/report_04/Project_Status_Report_04_Submission.md`
  - `docs/status_reports/report_04/week07_model_selection_decision.md`
  - `docs/status_reports/report_04/week07_alignment_with_feedback.md`
  - `docs/status_reports/report_04/week07_docs_audit.md`
  - `docs/status_reports/report_04/week07_run_manifest.json`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_isotonic.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_full_none.csv`
