# EDA Report

This report summarizes the descriptive exploratory analysis used to check the modeling table and provide thesis context before model fitting.

## Scope
- Missingness checks for the retained modeling columns
- Weighted and unweighted prevalence summaries
- Value-distribution checks for both raw key variables and analysis-ready columns

The EDA does not include model fitting or causal interpretation.

## Inputs
- `data/processed/yrbs_2023_modeling.parquet`
- `scripts/00_schema_audit.py`
- `scripts/02_eda.py`
- `docs/data_dictionary.md`

## Retained Outputs
### Tables
- `outputs/tables/schema.csv`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/missingness_eda.csv`
- `outputs/tables/unweighted_prevalence_overall.csv`
- `outputs/tables/weighted_prevalence_overall.csv`
- `outputs/tables/weighted_prevalence_by_q1.csv`
- `outputs/tables/weighted_prevalence_by_q2.csv`
- `outputs/tables/weighted_prevalence_by_q3.csv`
- `outputs/tables/weighted_prevalence_by_raceeth.csv`
- `outputs/tables/value_counts_QN24.csv`
- `outputs/tables/value_counts_QN25.csv`
- `outputs/tables/value_counts_QN26.csv`
- `outputs/tables/value_counts_y_qn26.csv`
- `outputs/tables/value_counts_x_qn24.csv`
- `outputs/tables/value_counts_x_qn25.csv`
- `outputs/tables/value_counts_q1.csv`
- `outputs/tables/value_counts_q2.csv`
- `outputs/tables/value_counts_q3.csv`
- `outputs/tables/value_counts_raceeth.csv`

### Figures
- `outputs/figures/missingness_bar.png`
- `outputs/figures/prevalence_overall_weighted_vs_unweighted.png`
- `outputs/figures/qn26_prevalence_by_q1.png`
- `outputs/figures/qn26_prevalence_by_raceeth.png`

## Interpretation Guardrail
Weighted prevalence outputs are descriptive population context only. They do not replace the unweighted predictive evaluation used for the main thesis comparison.
