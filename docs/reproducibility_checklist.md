# Reproducibility Checklist

Status checkpoint (as of 2026-02-18):
- Weeks 1-5 reproducibility requirements are complete.
- Week 6+ checks are planned and not yet executed.

## Week 1-5 Completed Checks
- [x] Environment validation captured: `outputs/logs/environment_check.json`
- [x] Schema audit artifacts generated from code: `outputs/tables/schema.csv`, `outputs/tables/missingness_summary.csv`
- [x] Modeling table build is reproducible: `data/processed/yrbs_2023_modeling.parquet`
- [x] Data dictionary aligned to current modeling table: `docs/data_dictionary.md`
- [x] Missingness and prevalence EDA generated from script: `scripts/02_eda.py`
- [x] Frozen train/test and CV split artifacts exist: `outputs/splits/holdout_seed2026.npz`, `outputs/splits/cvfolds_seed2026.npz`
- [x] Week 4 baseline metrics recorded: `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`, `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
- [x] Week 5 tuned HGB metrics recorded: `outputs/metrics/metrics_cv_seed2026_hgb_baseline_none.csv`, `outputs/metrics/metrics_test_seed2026_hgb_baseline_none.csv`, `outputs/metrics/metrics_cv_folds_seed2026_hgb_baseline_none.csv`
- [x] Week 5 tuning artifacts recorded: `outputs/tuning/hgb_seed2026_baseline_search_results.csv`, `outputs/tuning/hgb_seed2026_baseline_best_params.json`
- [x] Week 5 diagnostics generated from script: `python scripts/04_week05_diagnostics.py --model hgb --baseline-model logreg --features baseline --seed 2026 --calibration none --outdir outputs`
- [x] Week 5 calibration comparison artifacts exist: `outputs/tables/week05_calibration_comparison_seed2026.csv`, `outputs/figures/week05_calibration_comparison_seed2026.png`
- [x] Week 5 stability artifacts exist: `outputs/tables/hgb_seed2026_baseline_perm_importance_by_fold.csv`, `outputs/tables/hgb_seed2026_baseline_perm_importance_summary.csv`, `outputs/figures/hgb_seed2026_baseline_importance_stability.png`
- [x] Experiment tracking updated through Week 5: `docs/experiment_log.md`
- [x] Decisions tracking updated through Week 5: `docs/decisions_log.md`

## Week 6+ Planned Checks (Not Yet Executed)
- [ ] PLANNED: full-feature comparison and bullying-block ablation reproducibility package.
- [ ] PLANNED: final model selection and robustness package.
- [ ] PLANNED: Week 10 submission bundle and manifest hashing.
