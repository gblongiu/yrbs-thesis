# Reproducibility Checklist

Status checkpoint (as of 2026-02-24):
- Weeks 1 to 6 reproducibility requirements are complete and locked.
- Selected Week 7 robustness and governance artifacts are complete ahead of schedule.
- Frozen Week 4 and Week 5 artifacts remain unchanged.

## Week 1-6 Completed Checks
- [x] Environment validation captured: `outputs/logs/environment_check.json`
- [x] Schema audit artifacts generated from code: `outputs/tables/schema.csv`, `outputs/tables/missingness_summary.csv`
- [x] Modeling table build is reproducible: `data/processed/yrbs_2023_modeling.parquet`
- [x] Frozen train/test and CV split artifacts exist: `outputs/splits/holdout_seed2026.npz`, `outputs/splits/cvfolds_seed2026.npz`
- [x] Week 4 baseline metrics are present and frozen: `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`, `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
- [x] Week 5 tuned HGB metrics and tuning artifacts are present and frozen: `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`, `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`, `outputs/tuning/hgb_seed2026_baseline_best_params.json`
- [x] Week 6 full-feature, ablation, and calibration sensitivity artifacts are reproducible: `outputs/tables/week06_full_feature_comparison_seed2026.csv`, `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`, `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- [x] Week 6 stability and interpretability artifacts are reproducible: `outputs/tables/hgb_seed2026_full_perm_importance_by_fold.csv`, `outputs/tables/hgb_seed2026_full_perm_importance_summary.csv`, `outputs/tables/week06_feature_importance_change_tracking_seed2026.csv`
- [x] Week 6 protocol lock and rubric checks are documented: `docs/status_reports/report_03/protocol_lock_confirmation.md`, `docs/status_reports/report_03/rubric_audit.md`

## Ahead-of-Schedule Week 7 Artifacts
- [x] Multi-seed stability artifact exists with fixed Week 4 holdout split: `scripts/09_multiseed_stability.py`, `outputs/tables/multiseed_stability_seed2026_2029.csv`
- [x] Held-out bootstrap CI artifact exists with prediction trace file: `scripts/10_bootstrap_ci.py`, `outputs/preds/preds_test_hgb_full_none_seed2026.csv`, `outputs/tables/heldout_bootstrap_ci_seed2026.csv`
- [x] Hyperparameter sensitivity and subgroup audit artifacts exist: `scripts/11_hyperparameter_sensitivity.py`, `scripts/12_subgroup_audit.py`, `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`, `outputs/tables/subgroup_performance_seed2026.csv`
- [x] Governance and integrity artifacts exist: `model_selection_framework.md`, `deployment_and_use_constraints.md`, `traceability_matrix.csv`, `qa_checklist.md`, `risk_register.md`, `outputs/audits/week07_upgrade_integrity_audit.md`

## Week 8 Next Checks
- [ ] Integrate Week 6 and ahead-of-schedule Week 7 evidence into full paper draft artifacts.
- [ ] Finalize submission-ready reproducibility appendix and evidence mapping.
