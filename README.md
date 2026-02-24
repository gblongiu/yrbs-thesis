# Predicting Youth Mental Health Risk from Bullying Exposure in the 2023 YRBS

This repository contains the analysis pipeline, documentation, and curated evidence artifacts tracked in Git for an INFO-I 492 senior thesis using 2023 YRBS microdata.

## Research Focus
Primary question:
- What is the incremental predictive value of bullying exposure indicators `QN24` and `QN25` for predicting `QN26` (persistent sadness/hopelessness), after demographic adjustment, under a fixed out-of-sample protocol?

Scope guardrails:
- Predictive associations only
- No causal claims
- No individual screening or diagnostic use

## Current Project Status
- Weeks 1 to 6 are complete under the frozen validation protocol, including full-feature comparison, bullying ablation, calibration sensitivity, and Week 6 stability diagnostics. Evidence: `outputs/tables/week06_full_feature_comparison_seed2026.csv`, `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`, `outputs/tables/week06_calibration_sensitivity_seed2026.csv`, `outputs/tables/hgb_seed2026_full_perm_importance_summary.csv`.
- Selected Week 7 robustness and governance tasks are complete ahead of schedule with additive outputs only. Evidence: `scripts/09_multiseed_stability.py`, `scripts/10_bootstrap_ci.py`, `scripts/11_hyperparameter_sensitivity.py`, `scripts/12_subgroup_audit.py`, `outputs/tables/multiseed_stability_seed2026_2029.csv`, `outputs/tables/heldout_bootstrap_ci_seed2026.csv`, `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv`, `outputs/tables/subgroup_performance_seed2026.csv`, `outputs/audits/week07_upgrade_integrity_audit.md`.
- Next milestone is Week 8 paper draft integration and synthesis of reproducibility and governance evidence.

## Week 6 Reproduction
Run from repository root.

```
.venv/bin/python scripts/07_week06_pipeline.py --seed 2026
.venv/bin/python scripts/08_week06_report_package.py --seed 2026
```

## Data
Primary local inputs (not committed):
- data/raw/YRBS_2023_MH_subset.xlsx
- data/raw/YRBS_2023_Combined_MH_subset.xlsx (context and trend file)

Core local analysis-ready table (not committed):
- data/processed/yrbs_2023_modeling.parquet

## Week 1-5 Core Pipeline Commands
Run from repository root.

1. Create environment and install dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt

2. Validate environment and audit schema.

    python3 scripts/00_validate_environment.py
    python3 scripts/00_schema_audit.py

3. Build modeling table.

    python3 scripts/01_build_dataset.py

4. Run EDA (writes local outputs to outputs/).

    python3 scripts/02_eda.py --outdir outputs

5. Run Week 4 baseline model (writes local outputs to outputs/).

    python3 scripts/03_train_models.py \
      --model logreg \
      --features baseline \
      --seed 2026 \
      --calibration none \
      --n_boot 0 \
      --outdir outputs

6. Run Week 5 tuned HGB model under frozen protocol (writes local outputs to outputs/).

    python3 scripts/03_train_models.py \
      --model hgb \
      --features baseline \
      --seed 2026 \
      --calibration none \
      --n_boot 0 \
      --outdir outputs \
      --run-id week05_models_v1_seed2026_hgb_baseline_none \
      --tune_hgb 1 \
      --hgb_search_iter 12 \
      --save_cv_preds 1 \
      --enforce_frozen_artifacts 1 \
      --week5_artifacts_only 1

7. Run Week 5 diagnostics (writes local outputs to outputs/).

    python3 scripts/04_week05_diagnostics.py \
      --model hgb \
      --baseline-model logreg \
      --features baseline \
      --seed 2026 \
      --calibration none \
      --outdir outputs

## Navigation for Reviewers
This repository uses two artifact areas.

Tracked evidence available on GitHub:
- Week 05 diagnostics tables: reports/week05/
  - reports/week05/week05_calibration_comparison_seed2026.csv
  - reports/week05/hgb_seed2026_baseline_perm_importance_by_fold.csv
  - reports/week05/hgb_seed2026_baseline_perm_importance_summary.csv

Project narrative and protocol documentation:
- Experiment log: docs/experiment_log.md
- Decisions log: docs/decisions_log.md
- Modeling report: docs/modeling_report.md
- Reproducibility checklist: docs/reproducibility_checklist.md

If present, status report submissions:
- Status reports: docs/status_reports/

Local generated outputs not committed:
- The pipeline writes generated artifacts to outputs/ (tables, figures, metrics, tuning, models, splits, logs).
- These are generally ignored by .gitignore to keep the repository lightweight and to avoid committing volatile artifacts.
- If a specific output is required for grading, it is copied into reports/<weekXX>/ and committed.

## Key Documentation
- docs/project_plan.md
- docs/analysis_plan.md
- docs/covariates_proposal.md
- docs/data_dictionary.md
- docs/eda_report.md
- docs/decisions_log.md
- docs/experiment_log.md
- docs/modeling_report.md
- docs/reproducibility_checklist.md

## Repository Layout
- data/   local datasets and staged artifacts (raw and processed data are not committed)
- docs/   plans, logs, reports, and submission artifacts
- reports/ curated evidence artifacts committed for review and grading
- outputs/ generated artifacts produced locally by scripts (generally not committed)
- scripts/ executable pipeline entrypoints
- src/    reusable project code
- tests/  smoke tests
