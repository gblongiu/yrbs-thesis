# Data Dictionary

This dictionary describes the analysis-ready modeling table produced by `scripts/01_build_dataset.py` from the local workbook `data/raw/YRBS_2023_MH_subset.xlsx`.

| analysis_name | source_column | role | type | coding | missingness handling | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `y_qn26` | `QN26` | primary target | binary | `1 -> 1`, `2 -> 0`, special missing codes -> `NA` | rows missing `y_qn26` are dropped | main thesis outcome |
| `y_qn27` | `QN27` | retained secondary outcome | binary | same binary recode | retained as `NA` if missing | kept for continuity with the original proposal record; not part of the primary comparison |
| `y_qn28` | `QN28` | retained secondary outcome | binary | same binary recode | retained as `NA` if missing | kept for continuity with the original proposal record; not part of the primary comparison |
| `y_qn29` | `QN29` | retained secondary outcome | binary | same binary recode | retained as `NA` if missing | kept for continuity with the original proposal record; not part of the primary comparison |
| `y_qn30` | `QN30` | retained secondary outcome | binary | same binary recode | retained as `NA` if missing | kept for continuity with the original proposal record; not part of the primary comparison |
| `x_qn24` | `QN24` | exposure | binary | same binary recode | retained as `NA` if missing | bullying on school property |
| `x_qn25` | `QN25` | exposure | binary | same binary recode | retained as `NA` if missing | electronic bullying |
| `q1` | `q1` | baseline covariate | categorical | stored as `cat_*` labels | retained as `NA` if missing | demographic baseline |
| `q2` | `q2` | baseline covariate | categorical | stored as `cat_*` labels | retained as `NA` if missing | demographic baseline |
| `q3` | `q3` | baseline covariate | categorical | stored as `cat_*` labels | retained as `NA` if missing | demographic baseline |
| `raceeth` | `raceeth` | baseline covariate | categorical | stored as `cat_*` labels | retained as `NA` if missing | demographic baseline |
| `weight` | `weight` | survey design | numeric | preserved as supplied | kept unchanged | descriptive weighting only |
| `stratum` | `stratum` | survey design | numeric | preserved as supplied | kept unchanged | descriptive weighting only |
| `psu` | `psu` | survey design | numeric | preserved as supplied | kept unchanged | descriptive weighting only |

## Column Order
`y_qn26`, `y_qn27`, `y_qn28`, `y_qn29`, `y_qn30`, `x_qn24`, `x_qn25`, `q1`, `q2`, `q3`, `raceeth`, `weight`, `stratum`, `psu`

## Supporting Artifacts
- `outputs/tables/modeling_table_audit.csv`
- `outputs/tables/missingness_modeling.csv`
- Dataset-build coding decisions are recorded in the local rerun log written by `scripts/01_build_dataset.py`; they are not retained as a canonical submission artifact.
