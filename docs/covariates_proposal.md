# Baseline Covariate Lock Record

This document preserves the Week 1-2 covariate lock that carried through to the finished thesis.

## Locked Configuration
- `TARGET_PRIMARY = "QN26"`
- `BULLYING_EXPOSURES = ["QN24", "QN25"]`
- `BASELINE_COVARIATES = ["q1", "q2", "q3", "raceeth"]`
- `SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]`

## Rationale
- `q1`: demographic age context
- `q2`: demographic sex context
- `q3`: school-stage context
- `raceeth`: demographic subgroup context used in the main comparison and subgroup review

## Scope Boundary
- Behavioral fields outside the thesis question were not promoted into the locked headline comparison.
- Secondary outcomes `QN27` to `QN30` were retained as columns for continuity but not used as predictors for the primary target.
- Survey-design fields were preserved for weighted descriptives only and never used as predictive features.

## Supporting Evidence
- `outputs/tables/schema.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/value_counts_QN24.csv`
- `outputs/tables/value_counts_QN25.csv`
- `outputs/tables/value_counts_QN26.csv`
