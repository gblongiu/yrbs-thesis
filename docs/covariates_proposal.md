# Baseline Covariate Proposal (YRBS 2023 Mental Health Subset)

This document records the Week 1-2 covariate lock for:
**Predicting Youth Mental Health Risk from Bullying Exposure in the 2023 YRBS**.

Status checkpoint (as of 2026-02-09):
- Week 1-2 scope complete.
- This file is active reference, not future planning.

## Evidence Inputs
- `scripts/00_schema_audit.py`
- `outputs/tables/schema.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/value_counts_QN24.csv`
- `outputs/tables/value_counts_QN25.csv`
- `outputs/tables/value_counts_QN26.csv`

## Locked Configuration (Implemented in `src/config.py`)
- `TARGET_PRIMARY = "QN26"`
- `BULLYING_EXPOSURES = ["QN24", "QN25"]`
- `BASELINE_COVARIATES = ["q1", "q2", "q3", "raceeth"]`
- `SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]`

## Covariate Rationale
- `q1`: core demographic control for age-related risk context.
- `q2`: core demographic control for sex-related prevalence differences.
- `q3`: schooling-stage demographic control.
- `raceeth`: demographic subgroup context and fairness-aware reporting.

## Exclusions for Headline Week 1-4 Scope
- Row-identifier or non-informative fields such as `record`, `orig_rec`, and `site`.
- Broad behavioral fields outside the primary question to avoid scope creep.
- Alternative outcomes `QN27` to `QN30` as predictors for the primary target.

## Missingness and Design Handling
- Drop missing `QN26` rows in supervised modeling table creation.
- Keep predictor missingness and handle in modeling pipeline.
- Preserve `weight`, `stratum`, and `psu` for descriptive weighting context.

## Forward Constraint
Week 5+ feature expansion is allowed only as explicitly PLANNED work and must not overwrite this baseline covariate lock.
