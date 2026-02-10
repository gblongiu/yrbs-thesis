# Experiment Log (Performed vs Planned)

Status checkpoint (recorded 2026-02-09):
- Weeks 1-4 deliverables complete.
- Weeks 5-10 planned and not executed.

## PERFORMED (Weeks 1-4)

## (PERFORMED) | 2026-02-09 | Week 1 | Environment validation
- What ran: `python3 scripts/00_validate_environment.py`
- Outputs:
  - `outputs/logs/environment_check.json`

## (PERFORMED) | 2026-02-09 | Week 1 | Schema audit and variable inventory
- What ran: `python3 scripts/00_schema_audit.py`
- Inputs:
  - `data/raw/YRBS_2023_MH_subset.xlsx`
- Outputs:
  - `outputs/tables/schema.csv`
  - `outputs/tables/missingness_summary.csv`
  - `outputs/tables/value_counts_QN24.csv`
  - `outputs/tables/value_counts_QN25.csv`
  - `outputs/tables/value_counts_QN26.csv`

## (PERFORMED) | 2026-02-09 | Week 2 | Build analysis-ready modeling table
- What ran: `python3 scripts/01_build_dataset.py`
- Outputs:
  - `data/processed/yrbs_2023_modeling.parquet`
  - `outputs/tables/modeling_table_audit.csv`
  - `outputs/tables/missingness_modeling.csv`
  - `outputs/logs/decisions.json`

## (PERFORMED) | 2026-02-09 | Week 3 | EDA tables and figures
- What ran: `python3 scripts/02_eda.py --outdir outputs`
- Outputs:
  - `outputs/tables/missingness_eda.csv`
  - `outputs/tables/unweighted_prevalence_overall.csv`
  - `outputs/tables/weighted_prevalence_overall.csv`
  - `outputs/tables/weighted_prevalence_by_q1.csv`
  - `outputs/tables/weighted_prevalence_by_q2.csv`
  - `outputs/tables/weighted_prevalence_by_q3.csv`
  - `outputs/tables/weighted_prevalence_by_raceeth.csv`
  - `outputs/figures/missingness_bar.png`
  - `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
  - `outputs/figures/qn26_prevalence_by_q1.png`
  - `outputs/figures/qn26_prevalence_by_raceeth.png`
  - `outputs/logs/eda_run_metadata.json`

## (PERFORMED) | 2026-02-09 | Week 4 | Baseline logistic model under frozen split protocol
- What ran: `python3 scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`
- Outputs:
  - `outputs/splits/holdout_seed2026.npz`
  - `outputs/splits/cvfolds_seed2026.npz`
  - `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
  - `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
  - `docs/modeling_report.md`

## PLANNED (Weeks 5-10, not executed)

- Week 5: boosted-model tuning and metric comparison.
- Week 6: ablation and interpretability package.
- Week 7: final model selection package.
- Week 8: first full paper draft and reproducibility appendix.
- Week 9: revision and presentation plan.
- Week 10: final submission bundle.
- Planned: weighted-fit sensitivity check for baseline logistic model. Not executed in Week 4.

Planned anchor document:
- `docs/ablation_report.md`