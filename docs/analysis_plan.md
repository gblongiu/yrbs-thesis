# Analysis Plan

Status checkpoint (as of 2026-02-09):
- Weeks 1-4 deliverables are complete.
- Weeks 5-10 items below are planned and not yet executed.

## Week 1-4 Completed Workflow
1. Validate environment and raw-file availability with `scripts/00_validate_environment.py`.
2. Run schema audit and variable inventory with `scripts/00_schema_audit.py`.
3. Build the analysis-ready modeling table with `scripts/01_build_dataset.py`.
4. Run EDA tables and figures with `scripts/02_eda.py --outdir outputs`.
5. Train the Week 4 baseline model under the frozen protocol with `scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`.

## Week 1-4 Verified Outputs
- Week 2:
  - `data/processed/yrbs_2023_modeling.parquet`
  - `outputs/tables/modeling_table_audit.csv`
  - `outputs/tables/missingness_modeling.csv`
  - `outputs/logs/decisions.json`
- Week 3:
  - `outputs/tables/missingness_eda.csv`
  - `outputs/tables/weighted_prevalence_overall.csv`
  - `outputs/tables/weighted_prevalence_by_q1.csv`
  - `outputs/tables/weighted_prevalence_by_q2.csv`
  - `outputs/tables/weighted_prevalence_by_q3.csv`
  - `outputs/tables/weighted_prevalence_by_raceeth.csv`
  - `outputs/logs/eda_run_metadata.json`
  - `outputs/figures/missingness_bar.png`
  - `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
  - `outputs/figures/qn26_prevalence_by_q1.png`
  - `outputs/figures/qn26_prevalence_by_raceeth.png`
- Week 4:
  - `outputs/splits/holdout_seed2026.npz`
  - `outputs/splits/cvfolds_seed2026.npz`
  - `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
  - `docs/modeling_report.md`

## Planned Week 5-10 Work (Not Executed Yet)
1. Week 5: tuned boosted-model comparison.
2. Week 6: ablation and interpretability package.
3. Week 7: final model selection and calibration-focused figures and tables.
4. Week 8: first full paper draft and reproducibility appendix.
5. Week 9: revision pass and presentation outline.
6. Week 10: final package and submission bundle.

Planned future output path (not generated yet):
- PLANNED: `outputs/submission/week10_submission_bundle_v1/`
