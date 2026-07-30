# Reproducibility Checklist

- Confirm the local raw workbook exists at `data/raw/YRBS_2023_MH_subset.xlsx`.
- Install the Python dependencies from `requirements.txt`.
- Regenerate the modeling parquet with `python3 scripts/01_build_dataset.py`.
- Run the canonical execution order listed in the top-level `README.md`.
- Verify that the frozen split artifacts remain present in `outputs/splits/`.
- Verify that the tuned HGB parameter file remains present in `outputs/tuning/hgb_seed2026_baseline_best_params.json`.
- Confirm the retained comparison metric CSVs, Week 6 comparison tables, and Week 7 robustness tables exist under `outputs/`.
- Run the smoke tests:
  - `pytest -q tests/test_build_dataset_smoke.py`
  - `pytest -q tests/test_eda_smoke.py`
  - `pytest -q tests/test_week07_integrity_and_subgroup.py`
- Rebuild the curated submission package with `python3 scripts/20_build_submission_pack.py`.
- Confirm that the submission directory and zip were refreshed from canonical sources.
