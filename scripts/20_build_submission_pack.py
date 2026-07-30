from __future__ import annotations

"""Build the curated final submission pack for the YRBS thesis project."""

import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = PROJECT_ROOT / "Draft_Thesis_Submission_GabrielLong"
ZIP_PATH = PROJECT_ROOT / "Draft_Thesis_Submission_GabrielLong.zip"

DESKTOP_THESIS_ROOT = Path("/Users/gabriellong/Desktop/Senior Thesis")
TOP_LEVEL_DOCS = [
    "analysis_plan.md",
    "modeling_report.md",
    "data_dictionary.md",
    "reproducibility_checklist.md",
    "proposal_scope_alignment.md",
]


def _latest_matching(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, suffixes: set[str] | None = None) -> None:
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        rel = path.relative_to(src)
        _copy_file(path, dst / rel)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _build_top_level_readme(manuscript_path: Path | None, deck_path: Path | None) -> str:
    manuscript_line = (
        f"- `Manuscript/{manuscript_path.name}`: current thesis manuscript copied from the local thesis workspace"
        if manuscript_path is not None
        else "- `Manuscript/README.md`: note explaining that the current manuscript file was not found during packaging"
    )
    deck_line = (
        f"- `Presentation/{deck_path.name}`: current narrated presentation deck copied from the local thesis workspace"
        if deck_path is not None
        else "- `Presentation/README.md`: note explaining that the narrated presentation deck was not found during packaging"
    )
    return f"""
# Draft Thesis Submission Pack

This folder is the curated supplementary submission package for the final YRBS thesis project. It is built from the canonical `yrbs-thesis` repository plus the current manuscript and narrated presentation files when they are available locally.

## Package Contents
- `Code/`: runnable thesis code, canonical project docs, retained outputs, tests, and local input policy
- `Documentation/`: grader-facing copies of the key thesis and reproducibility notes
- `Graphs/`: reviewer-friendly copies of the final thesis figures
- `Presentation/`: final presentation source notes plus the current narrated presentation deck when available
- `Manuscript/`: current thesis paper file when available

## Included Thesis-Facing Materials
{manuscript_line}
{deck_line}

## Review Order
1. Read the manuscript in `Manuscript/`.
2. Review the deck and slide materials in `Presentation/`.
3. Use `Documentation/` for the thesis and reproducibility notes most likely to matter during grading.
4. Review the final figures in `Graphs/`.
5. Use `Code/README.md` and `Code/docs/` for the full runnable record.

## Packaging Note
`Code/outputs/` preserves the canonical repository structure needed for reproducibility. `Graphs/` duplicates the final figure PNGs only so a reviewer can browse the visuals without navigating the code tree.
"""


def _build_code_readme(raw_workbook_present: bool, parquet_present: bool) -> str:
    raw_note = (
        "- `data/raw/YRBS_2023_MH_subset.xlsx` is included for direct reruns."
        if raw_workbook_present
        else "- The raw workbook was not found locally during packaging, so `data/raw/` contains only the policy note."
    )
    parquet_note = (
        "- `data/processed/yrbs_2023_modeling.parquet` is included as the current analysis-ready dataset snapshot."
        if parquet_present
        else "- The analysis-ready parquet was not found locally during packaging; rerun `scripts/01_build_dataset.py` to regenerate it."
    )
    return f"""
# Code Package

This directory is the runnable thesis code subset taken from the canonical repository.

## Execution Root
Run commands from this directory.

## Contents
- `scripts/`: retained pipeline entrypoints, including the submission-pack builder
- `src/`: reusable data, evaluation, modeling, and reporting modules
- `tests/`: retained smoke tests
- `docs/`: canonical project documentation
- `outputs/`: curated metrics, figures, tables, split artifacts, and tuning parameters
- `presentation/`: presentation markdown sources copied from the live repository
- `data/`: local input policy and, when available, the raw YRBS workbook
- `requirements.txt`: Python dependency list

## Data
{raw_note}
{parquet_note}

## Suggested Rerun Order
1. `python3 scripts/00_validate_environment.py`
2. `python3 scripts/00_schema_audit.py`
3. `python3 scripts/01_build_dataset.py`
4. `python3 scripts/02_eda.py --outdir outputs`
5. `python3 scripts/03_train_models.py --model logreg --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs`
6. `python3 scripts/03_train_models.py --model hgb --features baseline --seed 2026 --calibration none --n_boot 0 --outdir outputs --run-id week05_models_v1_seed2026_hgb_baseline_none --tune_hgb 1 --hgb_search_iter 12 --save_cv_preds 1 --enforce_frozen_artifacts 1 --week5_artifacts_only 1`
7. `python3 scripts/04_week05_diagnostics.py --model hgb --baseline-model logreg --features baseline --seed 2026 --calibration none --outdir outputs`
8. `python3 scripts/07_week06_pipeline.py --seed 2026 --outdir outputs`
9. Optional Week 7 summaries: `scripts/09_multiseed_stability.py`, `scripts/10_bootstrap_ci.py`, `scripts/11_hyperparameter_sensitivity.py`, and `scripts/12_subgroup_audit.py`
"""


def _build_graphs_readme() -> str:
    return """
# Graphs

This folder contains reviewer-friendly copies of the final thesis figures. The same PNG files also remain under `Code/outputs/figures/` so the canonical code tree stays self-consistent for reruns and documentation references.
"""


def _build_documentation_readme() -> str:
    return """
# Documentation

This folder contains a small top-level copy set of the thesis and reproducibility notes most useful for grading and package navigation. The full canonical documentation set remains under `Code/docs/`.
"""


def _build_presentation_readme(deck_path: Path | None) -> str:
    deck_note = (
        f"- `{deck_path.name}`: current narrated presentation deck copied from the local thesis workspace"
        if deck_path is not None
        else "- Current narrated presentation deck was not found locally during packaging"
    )
    return f"""
# Presentation Materials

- `slide_outline.md`: final slide sequence
- `speaker_notes.md`: final speaking notes
- `slide_asset_manifest.md`: mapping from slides to supporting thesis artifacts
{deck_note}
"""


def _build_manuscript_readme(manuscript_path: Path | None) -> str:
    if manuscript_path is None:
        return """
# Manuscript

The current thesis manuscript file was not found in the expected local thesis workspace during packaging.
"""
    return f"""
# Manuscript

This folder contains the current thesis manuscript copied from the local thesis workspace:
- `{manuscript_path.name}`
"""


def _zip_pack() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACK_ROOT.rglob("*")):
            if path.is_dir() or path.name == ".DS_Store":
                continue
            zf.write(path, arcname=path.relative_to(PROJECT_ROOT))


def main() -> None:
    if PACK_ROOT.exists():
        shutil.rmtree(PACK_ROOT)

    manuscript_path = _latest_matching(
        DESKTOP_THESIS_ROOT,
        "Module */Long_Gabriel_INFOI492_ThesisPaper*.docx",
    )
    deck_path = _latest_matching(
        DESKTOP_THESIS_ROOT,
        "Module */*Narrated_Presentation*.pptx",
    )
    raw_workbook = PROJECT_ROOT / "data" / "raw" / "YRBS_2023_MH_subset.xlsx"
    processed_parquet = PROJECT_ROOT / "data" / "processed" / "yrbs_2023_modeling.parquet"

    code_root = PACK_ROOT / "Code"
    documentation_root = PACK_ROOT / "Documentation"
    graphs_root = PACK_ROOT / "Graphs"
    presentation_root = PACK_ROOT / "Presentation"
    manuscript_root = PACK_ROOT / "Manuscript"

    _copy_file(PROJECT_ROOT / "requirements.txt", code_root / "requirements.txt")
    _copy_file(PROJECT_ROOT / "data" / "README.md", code_root / "data" / "README.md")

    if raw_workbook.exists():
        _copy_file(raw_workbook, code_root / "data" / "raw" / raw_workbook.name)
    if processed_parquet.exists():
        _copy_file(processed_parquet, code_root / "data" / "processed" / processed_parquet.name)

    _copy_tree(PROJECT_ROOT / "scripts", code_root / "scripts", suffixes={".py"})
    _copy_tree(PROJECT_ROOT / "src", code_root / "src", suffixes={".py"})
    _copy_tree(PROJECT_ROOT / "tests", code_root / "tests", suffixes={".py"})
    _copy_tree(PROJECT_ROOT / "docs", code_root / "docs", suffixes={".md"})
    _copy_tree(PROJECT_ROOT / "presentation", code_root / "presentation", suffixes={".md"})
    _copy_tree(PROJECT_ROOT / "outputs" / "figures", code_root / "outputs" / "figures", suffixes={".png"})
    _copy_tree(PROJECT_ROOT / "outputs" / "metrics", code_root / "outputs" / "metrics", suffixes={".csv"})
    _copy_tree(PROJECT_ROOT / "outputs" / "splits", code_root / "outputs" / "splits", suffixes={".npz"})
    _copy_tree(PROJECT_ROOT / "outputs" / "tables", code_root / "outputs" / "tables", suffixes={".csv"})
    _copy_tree(PROJECT_ROOT / "outputs" / "tuning", code_root / "outputs" / "tuning", suffixes={".json"})

    _copy_tree(PROJECT_ROOT / "outputs" / "figures", graphs_root, suffixes={".png"})
    _copy_tree(PROJECT_ROOT / "presentation", presentation_root, suffixes={".md"})
    for name in TOP_LEVEL_DOCS:
        _copy_file(PROJECT_ROOT / "docs" / name, documentation_root / name)

    if manuscript_path is not None and manuscript_path.exists():
        _copy_file(manuscript_path, manuscript_root / manuscript_path.name)
    if deck_path is not None and deck_path.exists():
        _copy_file(deck_path, presentation_root / deck_path.name)

    _write_text(PACK_ROOT / "README.md", _build_top_level_readme(manuscript_path, deck_path))
    _write_text(code_root / "README.md", _build_code_readme(raw_workbook.exists(), processed_parquet.exists()))
    _write_text(documentation_root / "README.md", _build_documentation_readme())
    _write_text(graphs_root / "README.md", _build_graphs_readme())
    _write_text(presentation_root / "README.md", _build_presentation_readme(deck_path))
    _write_text(manuscript_root / "README.md", _build_manuscript_readme(manuscript_path))

    _zip_pack()

    print(f"Built {PACK_ROOT}")
    print(f"Built {ZIP_PATH}")


if __name__ == "__main__":
    main()
