# Data Directory Policy

This directory stores local datasets and staged artifacts used by the Week 1-4 pipeline.

## Expected Local Layout
- `data/raw_sources/`: original download bundles and archives (local-only)
- `data/raw/`: extracted or manually prepared source files (local-only)
- `data/interim/`: temporary transformation artifacts (local-only)
- `data/processed/`: analysis-ready tables generated locally (local-only)
- `data/transcripts/`: advisor/course transcript materials (local-only)

## Tracking Policy
- Raw, restricted, or otherwise sensitive inputs are local-only and must not be committed.
- Processed tables under `data/processed/` are also ignored by default.
- Curated grading evidence is stored under `outputs/` and selectively tracked via `.gitignore` allowlist rules.
