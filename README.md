# Predicting Youth Mental Health Risk from Bullying Exposure in the 2023 YRBS

This repository contains the analysis pipeline, documentation, and curated evidence artifacts for an INFO-I 492 senior thesis using 2023 YRBS microdata.

## Research Focus
Primary question:
- What is the incremental predictive value of bullying exposure indicators `QN24` and `QN25` for predicting `QN26` (persistent sadness/hopelessness), after demographic adjustment, under a fixed out-of-sample protocol?

Scope guardrails:
- Predictive associations only
- No causal claims
- No individual screening or diagnostic use

## Project Status
Implemented now (Weeks 1-4):
- Week 1: question/scope/reproducibility setup complete
- Week 2: analysis-ready dataset and data dictionary complete
- Week 3: descriptive EDA tables and figures complete
- Week 4: baseline logistic model under frozen validation protocol complete

Planned later (Weeks 5-10):
- Sensitivity, calibration, and extension analyses are documented as planned-only and not executed in current scope.

## Data
Primary local inputs:
- `data/raw/YRBS_2023_MH_subset.xlsx`
- `data/raw/YRBS_2023_Combined_MH_subset.xlsx` (context/trend file)

Core local analysis-ready table:
- `data/processed/yrbs_2023_modeling.parquet`

## Week 1-4 Pipeline Commands
Run from repository root.

1. Create environment and install dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

2. Validate environment and audit schema.
```bash
python3 scripts/00_validate_environment.py
python3 scripts/00_schema_audit.py
```

3. Build modeling table (Week 2).
```bash
python3 scripts/01_build_dataset.py
```

4. Run EDA (Week 3).
```bash
python3 scripts/02_eda.py --outdir outputs
```

5. Run Week 4 baseline model.
```bash
python3 scripts/03_train_models.py \
  --model logreg \
  --features baseline \
  --seed 2026 \
  --calibration none \
  --n_boot 0 \
  --outdir outputs
```

## Week 4 Required Outputs
- `outputs/splits/holdout_seed2026.npz`
- `outputs/splits/cvfolds_seed2026.npz`
- `outputs/metrics/metrics_cv_seed2026_logreg_baseline_none.csv`
- `outputs/metrics/metrics_test_seed2026_logreg_baseline_none.csv`
- `docs/modeling_report.md`

## Dependency Locking (Optional)
- `requirements.txt` remains the direct dependency list for normal setup.
- For exact local reproducibility, an optional lock snapshot can be generated from a known-good environment:
```bash
python3 -m pip freeze > requirements-lock.txt
```
- The exact environment can then be recreated with:
```bash
python3 -m pip install -r requirements-lock.txt
```

## Key Documentation
- `docs/project_plan.md`
- `docs/analysis_plan.md`
- `docs/covariates_proposal.md`
- `docs/data_dictionary.md`
- `docs/eda_report.md`
- `docs/decisions_log.md`
- `docs/experiment_log.md`
- `docs/modeling_report.md`
- `docs/week04_audit_report.md`
- `docs/ablation_report.md` (planned placeholder)

## Repository Layout
- `data/` local datasets and staged artifacts
- `docs/` planning, logs, and reports
- `outputs/` generated artifacts (tables/figures/metrics/models/splits/logs)
- `scripts/` executable pipeline entrypoints
- `src/` reusable project code
- `tests/` smoke tests
