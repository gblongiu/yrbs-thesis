# Predicting Youth Mental Health Risk from Bullying Exposure in the 2023 YRBS

## Current Status
This repository contains the code, documentation, and curated evidence artifacts for my INFO-I 492 senior thesis using 2023 YRBS microdata.

- Week 1: question/scope/reproducibility setup complete
- Week 2: analysis-ready dataset + data dictionary complete
- Week 3: descriptive EDA tables/figures complete
- Week 4: baseline model under frozen validation protocol complete

Weeks 5-10 work is planned, not executed.

## Research Focus
Primary question:
- What is the incremental predictive value of bullying exposure indicators `QN24` and `QN25` for predicting `QN26` (persistent sadness/hopelessness), after demographic adjustment, under a fixed out-of-sample protocol?

Scope guardrails:
- Predictive associations only
- No causal claims
- No individual screening/diagnostic use

## Data
Primary local inputs:
- `data/raw/YRBS_2023_MH_subset.xlsx`
- `data/raw/YRBS_2023_Combined_MH_subset.xlsx` (context/trend file)

Core analysis-ready table:
- `data/processed/yrbs_2023_modeling.parquet`

## Week 1-4 Pipeline Commands
Run from repository root.

1. Create environment and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Validate environment and audit schema
```bash
python scripts/00_validate_environment.py
python scripts/00_schema_audit.py
```

3. Build modeling table (Week 2)
```bash
python scripts/01_build_dataset.py
```

4. Run EDA (Week 3)
```bash
python scripts/02_eda.py --outdir outputs
```

5. Run Week 4 baseline model
```bash
python scripts/03_train_models.py \
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
- `docs/ablation_report.md` (PLANNED placeholder)

## Repository Layout
- `data/` local datasets and processed table
- `docs/` planning, logs, reports
- `outputs/` generated artifacts (tables/figures/metrics/models/splits/logs)
- `scripts/` executable pipeline entrypoints
- `src/` reusable project code
- `tests/` smoke tests
