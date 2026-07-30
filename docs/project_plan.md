# Project Plan

This file records the final technical scope of the repository after the thesis reached submission-ready status on 2026-03-20.

## Final Study Scope
- Build a modeling table for `QN26` from the 2023 YRBS mental-health subset.
- Compare the locked demographic baseline (`q1`, `q2`, `q3`, `raceeth`) against the bullying-augmented feature set that adds `QN24` and `QN25`.
- Evaluate all retained comparisons under the frozen held-out split and frozen cross-validation folds.
- Report predictive performance, calibration, interpretability, and subgroup robustness using curated aggregated outputs.

## Proposal Commitments Carried Into The Final Repository
| Proposal area | Final status |
| --- | --- |
| Analysis-ready dataset and data dictionary | Complete |
| EDA with prevalence and missingness outputs | Complete |
| Baseline versus bullying-augmented predictive comparison | Complete |
| Calibration review and held-out model selection | Complete |
| Robustness summaries across seeds, hyperparameters, and subgroups | Complete |
| Final presentation source materials | Complete in `presentation/` |
| Curated supplementary submission package | Complete via `scripts/20_build_submission_pack.py` |

## Repository Responsibilities
- Keep the scientific pipeline runnable from local workbook to curated outputs.
- Preserve the frozen protocol artifacts that anchor the final reported comparisons.
- Maintain the canonical project documentation that explains scope, coding choices, and reproducibility.
- Build the curated submission package from canonical sources rather than maintaining a second hand-edited project tree.

## Explicit Exclusions
- New outcome expansion, new dataset ingestion, or causal re-framing
- Deployment, monitoring, dashboards, or application layers
- Placeholder extensions, abandoned stubs, and duplicate package scaffolding
