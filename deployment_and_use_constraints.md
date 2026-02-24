# Deployment and Use Constraints

## Intended Use
This model package is intended for research-stage probability estimation of persistent sadness or hopelessness in the 2023 National YRBS dataset under the frozen protocol.

## Prohibited Use
- No clinical diagnosis.
- No individual-level intervention assignment.
- No causal claims about bullying exposures.
- No deployment without subgroup and calibration monitoring.

## Non-Causal Interpretation Rule
All model outputs represent predictive associations under observed covariates and a fixed validation protocol.
Causal language is out of scope.
Evidence:
- `docs/status_reports/report_03/contract_extracts.md`
- `docs/status_reports/report_03/Project_Status_Report_03_Submission.md`

## Data and Scope Limitations
- Current modeling table has limited covariate breadth.
- Ablation control set equals baseline covariates under current scope.
Evidence:
- `data/processed/yrbs_2023_modeling.parquet`
- `docs/status_reports/report_03/feature_set_definitions.md`

## Calibration Requirement
Any deployment-adjacent use requires calibration verification on current target population data before use.
Evidence:
- `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- `outputs/tables/heldout_bootstrap_ci_seed2026.csv`

## Subgroup Monitoring Requirement
Any deployment-adjacent use requires periodic subgroup audit checks for `raceeth` and `q2` with documented flags.
Evidence:
- `outputs/tables/subgroup_performance_seed2026.csv`

## Reproducibility Requirement
All reported claims must trace to reproducible script outputs with deterministic seeds and artifact checks.
Evidence:
- `outputs/audits/week07_upgrade_integrity_audit.md`
- `docs/status_reports/report_03/week06_run_manifest.json`

## Drift and Recalibration Trigger
Recalibration review is required if subgroup error flags increase, if multiseed stability worsens beyond threshold, or if held-out Brier degrades on refresh data.
Evidence targets:
- `outputs/tables/multiseed_stability_seed2026_2029.csv`
- `outputs/tables/subgroup_performance_seed2026.csv`
