# Thesis Presentation Slide Outline

## Slide 1. Title And Question
- Thesis title
- Author, course, advisor
- Core question: do `QN24` and `QN25` add predictive value for `QN26` beyond the locked demographic baseline?

## Slide 2. Problem And Motivation
- Adolescent mental-health risk remains common in the 2023 YRBS sample.
- Bullying exposure is substantively important and plausible as a predictive signal.
- The thesis focus is incremental predictive value, not causal explanation.

## Slide 3. Literature Framing And Gap
- Prior literature links bullying exposure with depressive symptoms and suicidality risk.
- The project gap is a disciplined predictive comparison under a frozen held-out evaluation frame.
- Emphasis stays on calibration and reproducibility rather than narrative overreach.

## Slide 4. Data Source And Study Boundary
- 2023 national YRBS mental-health subset
- `20,103` raw rows; `19,863` modeling rows after dropping missing `QN26`
- Survey weights retained for descriptive context only
- No new datasets, no linkage, no individual targeting

## Slide 5. Outcome And Predictor Specification
- Outcome: `QN26` persistent sadness or hopelessness
- Locked baseline: `q1`, `q2`, `q3`, `raceeth`
- Bullying block: `x_qn24`, `x_qn25`
- Secondary outcomes retained in the dataset but excluded from the headline comparison

## Slide 6. Frozen Evaluation Protocol
- Frozen seed `2026`
- Stored held-out split and stored cross-validation folds
- Baseline logistic reference and tuned HGB comparison
- Metrics: ROC AUC, PR AUC, Brier, calibration slope, calibration intercept

## Slide 7. Baseline Versus Bullying-Augmented Comparison
- Week 4-5 baseline reference performance
- Week 6 full-feature results
- Bullying-block ablation result as the main incremental evidence

## Slide 8. Calibration And Model-Selection Logic
- Compare none, Platt, and isotonic calibration on the full HGB candidate
- Selection rule: lowest held-out Brier, slope closest to `1`, ROC AUC as tie-breaker
- Final reported model: `hgb_full_platt`

## Slide 9. Robustness Checks
- Multiseed stability
- Held-out bootstrap intervals
- Local hyperparameter sensitivity
- Permutation-importance stability for the full HGB candidate

## Slide 10. Subgroup Review
- Review by `q2` and `raceeth`
- Report only groups that clear adequacy thresholds
- Treat subgroup differences as descriptive risk flags, not fairness claims or deployment guidance

## Slide 11. Limitations And Governance
- Cross-sectional public survey data
- Predictive, not causal
- Weighted descriptives separated from unweighted predictive evaluation
- No screening, intervention, or individual monitoring use

## Slide 12. Conclusion And Future Work
- Bullying variables add modest but consistent incremental predictive value under the locked protocol
- Gains are modest but consistent across the retained robustness checks
- Future work stays outside this thesis scope: external validation, larger subgroup samples, and prospective designs
