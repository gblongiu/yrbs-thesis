# Decisions Log

## D001 | 2026-01-16
- Locked the primary outcome to `QN26` / `y_qn26`.

## D002 | 2026-01-16
- Locked the primary exposures to `QN24` and `QN25`.

## D003 | 2026-01-16
- Locked the baseline covariates to `q1`, `q2`, `q3`, and `raceeth`.

## D004 | 2026-01-30
- Recode binary YRBS items to `{0, 1, NA}` with explicit handling for special missing codes.

## D005 | 2026-01-30
- Drop rows missing the primary outcome `y_qn26` when building the modeling table.

## D006 | 2026-02-06
- Keep weighted descriptive summaries separate from unweighted predictive evaluation.

## D007 | 2026-02-09
- Freeze the validation protocol around seed `2026`, the stored holdout split, and the stored fold assignments.

## D008 | 2026-02-18
- Keep Week 5 on the baseline feature set so the tuned HGB comparison stays directly comparable to the Week 4 logistic reference.

## D009 | 2026-02-18
- Defer broader feature expansion and alternative calibration methods until Week 6.

## D010 | 2026-02-22
- Run Week 6 as a full-feature comparison, bullying-block ablation, and calibration sensitivity package under the frozen protocol.
- Derive `full_features` programmatically from the modeling table after excluding targets, survey-design fields, identifiers, and known post-event leakage columns.

## D011 | 2026-02-22
- Treat the equality of `full_minus_bullying_features` and `baseline_features` as a structural limitation of the current predictor set, not as a protocol failure.

## D012 | 2026-03-02
- Select `hgb_full_platt` as the final held-out headline model using this ranking rule: lowest held-out Brier score, calibration slope closest to `1`, and held-out ROC AUC as the tie-breaker.
- Candidate set: `hgb_full_none`, `hgb_full_platt`, `hgb_full_isotonic`, `logreg_full_none`.

## D013 | 2026-03-12
- Keep the live repository lean by retaining summary tables, figures, decision notes, and reproducibility aids rather than manuscript binaries, status-report mirrors, and duplicate package scaffolding.

## D014 | 2026-03-20
- Keep the thesis centered on `QN26`; do not promote secondary outcomes into new headline analyses.

## D015 | 2026-03-20
- Represent boosted-model interpretability with retained permutation-importance stability outputs instead of adding a new late-stage explainer stack.

## D016 | 2026-03-20
- Build the final submission materials from canonical repository sources with a scripted export rather than maintaining a second hand-edited thesis package tree.

## D014 (PERFORMED) | 2026-03-20 | Week 6
- Decision: Execute Week 6 full-feature comparison, bullying-block ablation, and calibration sensitivity under frozen seed 2026 protocol.
- Rationale: The locked analysis plan requires these comparisons under unchanged split, fold, and baseline-reference artifacts.
- Equality condition: `full_minus_bullying_features` equals `baseline_features` due current modeling-table predictor scope.
- Handling policy: Continue execution and document this as a structural limitation, not a protocol violation.
- Evidence:
  - `outputs/tables/week06_full_feature_comparison_seed2026.csv`
  - `outputs/tables/week06_bullying_ablation_comparison_seed2026.csv`
  - `outputs/tables/week06_calibration_sensitivity_seed2026.csv`
- Deviations: None
