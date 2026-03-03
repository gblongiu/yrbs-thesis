# Manuscript Skeleton

## Title Placeholder
- Predicting Persistent Sadness or Hopelessness in the 2023 National YRBS Using Bullying Exposure Indicators

## Introduction
### Background Placeholder
- Summarize adolescent mental health burden and relevance of YRBS 2023 context.
- Cite descriptive context from existing prevalence artifacts when needed.

### Research Question Placeholder
- Primary question: what is the predictive contribution of QN24 and QN25 for QN26 under fixed demographic adjustment.

### Study Objective Placeholder
- Build and compare candidate predictive models under a frozen protocol.
- Prioritize probability quality and calibration alongside ranking metrics.

## Methods
### Data
- Dataset: CDC 2023 National YRBS modeling table.
- Source policy and local-only handling: `data/README.md`.
- Modeling input reference: `data/processed/yrbs_2023_modeling.parquet`.

### Outcomes
- Primary outcome: QN26 coded as `y_qn26`.
- Configuration source: `src/config.py`.

### Predictors
- Core exposures: QN24 and QN25 coded as `x_qn24` and `x_qn25`.
- Baseline covariates: `q1`, `q2`, `q3`, `raceeth`.
- Feature set references: `docs/status_reports/report_03/feature_set_definitions.md` and `src/config.py`.

### Split Protocol
- Frozen held-out split and CV fold artifacts with seed 2026:
  - `outputs/splits/holdout_seed2026.npz`
  - `outputs/splits/cvfolds_seed2026.npz`
- Protocol continuity reference: `docs/status_reports/report_03/protocol_lock_confirmation.md`.

### Metrics
- ROC AUC
- PR AUC
- Brier
- Calibration slope
- Calibration intercept
- Metric implementation reference: `src/evaluation/metrics.py`.

### Calibration Approach
- Compare none, Platt, and isotonic calibration within the fixed protocol.
- Core evidence table: `outputs/tables/week06_calibration_sensitivity_seed2026.csv`.

## Results
### Insert Plan for Tables and Figures
- Insert Table R1: held-out candidate model comparison built from `docs/status_reports/report_04/Project_Status_Report_04_Submission.md` and metric CSV sources in `outputs/metrics`.
- Insert Table R2: calibration sensitivity from `outputs/tables/week06_calibration_sensitivity_seed2026.csv`.
- Insert Figure R1: calibration sensitivity figure from `outputs/figures/week06_calibration_sensitivity_seed2026.png`.
- Insert Table R3: bullying predictive contribution from `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`.
- Insert Table R4: robustness and governance summary from `outputs/tables/multiseed_stability_seed2026_2029.csv`, `outputs/tables/heldout_bootstrap_ci_seed2026.csv`, `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`, and `outputs/tables/subgroup_performance_seed2026.csv`.
- Insert Table R5: stable importance summary from `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`.

### Candidate Model Comparison
- Insert held-out candidate comparison table from `docs/status_reports/report_04/Project_Status_Report_04_Submission.md`.
- Primary held-out metric files:
  - `outputs/metrics/metrics_test_seed2026_hgb_full_platt.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_none.csv`
  - `outputs/metrics/metrics_test_seed2026_hgb_full_isotonic.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_full_none.csv`

### Model Selection Decision
- Decision narrative source: `docs/status_reports/report_04/week07_model_selection_decision.md`.
- Criteria source: `model_selection_framework.md`.

### Calibration and Probability Quality
- Calibration sensitivity summary table: `outputs/tables/week06_calibration_sensitivity_seed2026.csv`.
- Calibration sensitivity figure: `outputs/figures/week06_calibration_sensitivity_seed2026.png`.
- Bootstrap uncertainty table: `outputs/tables/heldout_bootstrap_ci_seed2026.csv`.

### Bullying Predictive Contribution
- Bullying-block comparison table: `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`.
- Full-feature comparison table: `outputs/tables/week06_full_feature_comparison_seed2026.csv`.

### Robustness and Governance
- Multiseed stability table: `outputs/tables/multiseed_stability_seed2026_2029.csv`.
- Hyperparameter sensitivity table: `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`.
- Subgroup audit table: `outputs/tables/subgroup_performance_seed2026.csv`.
- Importance stability table: `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv`.
- Integrity audit: `outputs/audits/week07_upgrade_integrity_audit.md`.

## Limitations and Ethics
### Limitations Placeholder
- Cross-sectional survey design limits interpretation to predictive association.
- Current covariate breadth is limited by modeling table scope.
- Structural note for ablation contrast: `docs/status_reports/report_03/feature_set_definitions.md`.

All results are interpreted as predictive associations under a fixed protocol. This project does not claim causal mechanisms, intervention impact, or person-level targeting suitability.

### Ethics Placeholder
- Non-causal interpretation boundary and intended use constraints:
  - `deployment_and_use_constraints.md`
  - `risk_register.md`
- No individual targeting and no clinical deployment claims.

## Reproducibility Appendix Placeholder
- Run provenance and evidence map:
  - `docs/status_reports/report_03/week06_run_manifest.json`
  - `docs/status_reports/report_04/week07_run_manifest.json`
  - `traceability_matrix.csv`
