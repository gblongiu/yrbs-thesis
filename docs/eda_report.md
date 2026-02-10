# EDA Report

Status checkpoint (as of 2026-02-09):
- Week 3 descriptive EDA deliverables are complete.
- This file reports descriptive context only.

## Purpose and Scope
This EDA provides:
- weighted and unweighted prevalence context;
- missingness and value-distribution checks for the modeling table.

This EDA does not include predictive model fitting or test-set performance claims.

## Inputs
- `data/processed/yrbs_2023_modeling.parquet`
- `docs/data_dictionary.md`
- `scripts/02_eda.py`

## Weighted Prevalence Definition
`weighted_prevalence = sum(weight * indicator) / sum(weight)`

Confidence intervals are approximate and weight-only (not full design-based variance estimation).

## Outputs Inventory

### Tables (`outputs/tables/`)
- `outputs/tables/missingness_eda.csv`
- `outputs/tables/unweighted_prevalence_overall.csv`
- `outputs/tables/weighted_prevalence_overall.csv`
- `outputs/tables/weighted_prevalence_by_q1.csv`
- `outputs/tables/weighted_prevalence_by_q2.csv`
- `outputs/tables/weighted_prevalence_by_q3.csv`
- `outputs/tables/weighted_prevalence_by_raceeth.csv`
- `outputs/tables/value_counts_y_qn26.csv`
- `outputs/tables/value_counts_x_qn24.csv`
- `outputs/tables/value_counts_x_qn25.csv`
- `outputs/tables/value_counts_q1.csv`
- `outputs/tables/value_counts_q2.csv`
- `outputs/tables/value_counts_q3.csv`
- `outputs/tables/value_counts_raceeth.csv`
- `outputs/tables/value_counts_weight.csv`
- `outputs/tables/value_counts_stratum.csv`
- `outputs/tables/value_counts_psu.csv`

### Figures (`outputs/figures/`)
- `outputs/figures/missingness_bar.png`
- `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
- `outputs/figures/qn26_prevalence_by_raceeth.png`
- `outputs/figures/qn26_prevalence_by_q1.png`

### Logs (`outputs/logs/`)
- `outputs/logs/eda_run_metadata.json`

## Interpretation Guardrail
These outputs are descriptive context for thesis background and data quality checks. They should not be interpreted as causal evidence.
