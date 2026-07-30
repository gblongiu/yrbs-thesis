# Experiment Log

## 2026-01-16 | Week 1
- Locked the primary outcome (`QN26`), the bullying exposures (`QN24`, `QN25`), and the baseline covariates (`q1`, `q2`, `q3`, `raceeth`).

## 2026-01-30 | Week 2
- Built the analysis-ready modeling table from the local YRBS subset workbook.
- Preserved survey design fields for descriptive weighting only.

## 2026-02-06 | Week 3
- Completed descriptive EDA tables and figures for missingness and prevalence context.

## 2026-02-09 | Week 4
- Trained the frozen logistic baseline under seed `2026`.
- Saved the held-out split and cross-validation fold artifacts used in later comparisons.

## 2026-02-18 | Week 5
- Tuned the histogram gradient boosting baseline comparator.
- Generated Week 5 calibration and permutation-importance summaries.

## 2026-02-22 | Week 6
- Ran the full-feature comparison, bullying-block ablation, and calibration sensitivity analyses.
- Documented the equality case where the non-bullying comparator equals the baseline covariate set.

## 2026-02-24 | Week 7
- Generated robustness summaries for multiseed stability, bootstrap intervals, hyperparameter sensitivity, and subgroup review.

## 2026-03-20 | Final Repository Pass
- Regenerated the missing Week 6 metric files so the retained comparison tables point to real source artifacts.
- Refreshed canonical documentation, presentation materials, and submission-pack build logic for the final thesis state.
