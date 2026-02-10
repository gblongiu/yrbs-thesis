# Reproducibility Checklist

Status checkpoint (as of 2026-02-09):
- Weeks 1-4 reproducibility requirements are complete.
- Week 5+ checks are planned and not yet executed.

## Week 1-4 Completed Checks
- [x] Environment validation captured: `outputs/logs/environment_check.json`
- [x] Schema audit artifacts generated from code: `outputs/tables/schema.csv`, `outputs/tables/missingness_summary.csv`
- [x] Modeling table build is reproducible: `data/processed/yrbs_2023_modeling.parquet`
- [x] Data dictionary aligned to current modeling table: `docs/data_dictionary.md`
- [x] Missingness and prevalence EDA generated from script: `scripts/02_eda.py`
- [x] Frozen train/test and CV split artifacts exist: `outputs/splits/holdout_seed2026.npz`, `outputs/splits/cvfolds_seed2026.npz`
- [x] Week 4 baseline metrics recorded: `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`, `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
- [x] Experiment tracking is current through Week 4: `docs/experiment_log.md`
- [x] Audit summary recorded: `docs/week04_audit_report.md`

## Week 5+ Planned Checks (Not Yet Executed)
- [ ] PLANNED: boosted-model comparison reproducibility package.
- [ ] PLANNED: ablation reproducibility and interpretability package.
- [ ] PLANNED: Week 10 submission bundle and manifest hashing.
