# yrbs-thesis

Final thesis repository for Gabriel Long's INFO-I 492 project on whether bullying exposure variables improve prediction of persistent sadness or hopelessness in the 2023 national Youth Risk Behavior Survey.

**Final manuscript:** [Read the complete thesis PDF](manuscript/Long_Gabriel_INFOI492_ThesisPaper_FINAL.pdf).

## Thesis Boundary
- Primary outcome: `QN26` / `y_qn26`
- Locked baseline covariates: `q1`, `q2`, `q3`, `raceeth`
- Bullying block: `QN24`, `QN25` / `x_qn24`, `x_qn25`
- Evaluation frame: frozen held-out split plus frozen cross-validation folds under seed `2026`

This repository is the canonical source of truth for the finished thesis workflow. It supports a bounded predictive analysis only. It does not make causal claims, does not add new datasets or outcomes, and is not a deployment package or screening tool.

## Key Results
- Selected model: Platt-calibrated histogram gradient boosting with the full predictor set (`hgb_full_platt`).
- Held-out performance: ROC AUC `0.716057`, PR AUC `0.644073`, Brier score `0.206122`, and calibration slope `1.005455`.
- Adding the bullying block versus the non-bullying HGB comparator improved held-out ROC AUC by `0.065857` and PR AUC by `0.106201`.
- The same ablation reduced held-out Brier score by `0.019046`; these results are predictive, not causal or deployment-ready.

## What Is In Scope Here
- Analysis-ready dataset construction from the local YRBS workbook
- Descriptive EDA with weighted prevalence context
- Baseline versus bullying-augmented predictive comparison
- Calibration review, model selection, and Week 7 robustness summaries
- Final presentation source materials in `presentation/`
- Reproducible build logic for the curated submission package

## Repository Layout
- `data/`: local working input policy and regenerated intermediate state
- `docs/`: final technical documentation, methods notes, proposal-scope audit, and reproducibility guidance
- `manuscript/`: final thesis manuscript in PDF format
- `outputs/`: curated aggregated metrics, tables, figures, split artifacts, and tuned-parameter JSON files
- `presentation/`: final slide outline, speaker notes, and slide asset manifest
- `scripts/`: pipeline entrypoints and the submission-pack build script
- `src/`: reusable data, modeling, evaluation, and reporting code
- `tests/`: smoke tests for the retained pipeline and subgroup edge cases

The week-numbered script names are preserved from the course workflow chronology, but the repository contents reflect the final thesis state rather than interim draft checkpoints.

## Canonical Execution Order
Run from the repository root after installing `requirements.txt`.

1. `python3 scripts/00_validate_environment.py`
2. `python3 scripts/00_schema_audit.py`
3. `python3 scripts/01_build_dataset.py`
4. `python3 scripts/02_eda.py --outdir outputs`
5. `python3 scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`
6. `python3 scripts/03_train_models.py --model hgb --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs --run-id week05_models_v1_seed2026_hgb_baseline_none --tune_hgb 1 --hgb_search_iter 12 --save_cv_preds 1 --enforce_frozen_artifacts 1 --week5_artifacts_only 1`
7. `python3 scripts/04_week05_diagnostics.py --model hgb --baseline-model logreg --features baseline --seed 2026 --calibration none --outdir outputs`
8. `python3 scripts/07_week06_pipeline.py --seed 2026 --outdir outputs`
9. Optional Week 7 robustness summaries:
   `python3 scripts/09_multiseed_stability.py --outdir outputs`
   `python3 scripts/10_bootstrap_ci.py --outdir outputs`
   `python3 scripts/11_hyperparameter_sensitivity.py --outdir outputs`
   `python3 scripts/12_subgroup_audit.py --outdir outputs`
10. Build the curated submission materials:
    `python3 scripts/20_build_submission_pack.py`

## Final Scientific Outputs
- Frozen protocol artifacts in `outputs/splits/`
- Held-out and cross-validation metric CSVs in `outputs/metrics/`
- Comparison, calibration, missingness, prevalence, interpretability, and robustness tables in `outputs/tables/`
- Curated thesis figures in `outputs/figures/`
- Final presentation materials in `presentation/`

Key result summaries are described in:
- `docs/modeling_report.md`
- `docs/analysis_plan.md`
- `docs/proposal_scope_alignment.md`

## Local Data Policy
- Expected raw workbook: `data/raw/YRBS_2023_MH_subset.xlsx`
- Regenerated parquet: `data/processed/yrbs_2023_modeling.parquet`

The canonical repository keeps raw data and regenerated parquet local-only. The submission-pack build script can copy the local workbook and current processed parquet into the packaged `Draft_Thesis_Submission_GabrielLong/Code/data/` tree when those files are present so the supplementary submission remains runnable without turning the live repository into a raw-data mirror.

## Validation
The retained smoke tests are:
- `pytest -q tests/test_build_dataset_smoke.py`
- `pytest -q tests/test_eda_smoke.py`
- `pytest -q tests/test_week07_integrity_and_subgroup.py`

See `docs/reproducibility_checklist.md` for the final rerun checklist and `docs/proposal_scope_alignment.md` for the concise proposal-to-final scope alignment note.
