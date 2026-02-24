# Risk Register

| risk_id | risk | likelihood | impact | mitigation | evidence_artifact |
| --- | --- | --- | --- | --- | --- |
| R-001 | Data leakage across folds or split boundaries | Medium | High | Enforce frozen split checks and fold-only training for CV and calibrator fitting | `scripts/13_upgrade_integrity_check.py`, `outputs/splits/holdout_seed2026.npz` |
| R-002 | Distribution drift relative to frozen 2023 data context | Medium | High | Monitor held-out Brier and subgroup flags on refresh data and trigger recalibration review | `deployment_and_use_constraints.md`, `outputs/tables/subgroup_performance_seed2026.csv` |
| R-003 | Subgroup instability with undefined or weak calibration | High | High | Apply subgroup gating thresholds and flag-based decision constraints | `outputs/tables/subgroup_performance_seed2026.csv`, `model_selection_framework.md` |
| R-004 | Overfitting to one fold realization | Medium | Medium | Evaluate multiseed fold sensitivity and enforce Brier standard deviation threshold | `outputs/tables/multiseed_stability_seed2026_2029.csv`, `model_selection_framework.md` |
| R-005 | Hyperparameter brittleness near tuned point | Medium | Medium | Run one-factor perturbation sensitivity and confirm stable ranking | `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv` |
| R-006 | Interpretability volatility for bullying feature importance | Medium | Medium | Track fold-level and summary permutation stability with coefficient of variation and sign consistency | `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv` |
| R-007 | Governance misuse through causal interpretation | Medium | High | Enforce non-causal language constraints and intended-use boundaries in report and deployment guidance | `deployment_and_use_constraints.md`, `docs/status_reports/report_03/Project_Status_Report_03_Submission.md` |
