# Data Dictionary (Analysis-Ready Modeling Table)

This data dictionary documents the analysis-ready table produced by `scripts/01_build_dataset.py`
from `data/raw/YRBS_2023_MH_subset.xlsx`.

Status checkpoint (as of 2026-02-09):
- Week 1-2 dictionary requirements are complete.
- Secondary-outcome modeling remains planned future work.

## Variable Table

| analysis_name | source_column | role | type | coding | missingness_handling | notes |
| --- | --- | --- | --- | --- | --- | --- |
| y_qn26 | QN26 | target | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Drop rows with missing `y_qn26`. | Primary target for Week 1-4 deliverables. |
| y_qn27 | QN27 | target_secondary | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA in modeling table. | PLANNED for Week 9 appendix-scope analysis; not executed yet. |
| y_qn28 | QN28 | target_secondary | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA in modeling table. | PLANNED for Week 9 appendix-scope analysis; not executed yet. |
| y_qn29 | QN29 | target_secondary | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA in modeling table. | PLANNED for Week 9 appendix-scope analysis; not executed yet. |
| y_qn30 | QN30 | target_secondary | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA in modeling table. | PLANNED for Week 9 appendix-scope analysis; not executed yet. |
| x_qn24 | QN24 | exposure | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA. | Primary exposure for thesis comparison. |
| x_qn25 | QN25 | exposure | binary | Recode: 1 -> 1, 2 -> 0, special missing codes -> NA. | Keep NA. | Primary exposure for thesis comparison. |
| q1 | q1 | covariate | categorical | Stored as categorical labels (`cat_1` ...). | Keep NA. | Baseline covariate. |
| q2 | q2 | covariate | categorical | Stored as categorical labels (`cat_1`, `cat_2`). | Keep NA. | Baseline covariate. |
| q3 | q3 | covariate | categorical | Stored as categorical labels (`cat_1` ...). | Keep NA. | Baseline covariate. |
| raceeth | raceeth | covariate | categorical | Stored as categorical labels (`cat_1` ...). | Keep NA. | Baseline covariate. |
| weight | weight | design | design | Preserved as supplied. | Keep as-is. | Weighted descriptive context only. |
| stratum | stratum | design | design | Preserved as supplied. | Keep as-is. | Weighted descriptive context only. |
| psu | psu | design | design | Preserved as supplied. | Keep as-is. | Weighted descriptive context only. |

## Modeling Table Column Order
`y_qn26`, `y_qn27`, `y_qn28`, `y_qn29`, `y_qn30`, `x_qn24`, `x_qn25`, `q1`, `q2`, `q3`, `raceeth`, `weight`, `stratum`, `psu`

## Audit Artifacts
- `data/processed/yrbs_2023_modeling.parquet`
- `outputs/tables/missingness_modeling.csv`
- `outputs/tables/modeling_table_audit.csv`
- `outputs/logs/decisions.json`
