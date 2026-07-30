# Data Directory Policy

The `data/` tree is for local working inputs and regenerated intermediate state. It is not the place for the polished thesis deliverables that support the final submission pack.

## Expected Local Contents
- `data/raw/YRBS_2023_MH_subset.xlsx`: primary local workbook used by the thesis pipeline
- `data/processed/yrbs_2023_modeling.parquet`: regenerated analysis-ready parquet written by `scripts/01_build_dataset.py`
- `data/interim/` and `data/transcripts/`: optional scratch space only

## Tracking Policy
- Raw workbooks stay local and are ignored by Git.
- The processed parquet is regenerated locally and is not treated as a curated repository artifact.
- Curated scientific outputs live under `outputs/`.
- The submission-pack build script may copy the local workbook into `Draft_Thesis_Submission_GabrielLong/Code/data/raw/` when it exists, so the supplementary package can remain runnable without turning the live repository into a raw-data mirror.
