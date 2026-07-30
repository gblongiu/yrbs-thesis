# Speaker Notes

## Slide 1. Title And Question
This thesis asks a narrow predictive question: in the 2023 national YRBS, do the bullying exposure variables `QN24` and `QN25` improve prediction of persistent sadness or hopelessness, `QN26`, beyond a locked demographic baseline? The project is intentionally bounded to reproducible predictive comparison under a fixed held-out evaluation protocol.

## Slide 2. Problem And Motivation
The motivation is practical but limited. Persistent sadness or hopelessness remains common in this dataset, and bullying exposure is a plausible signal that may improve population-level risk stratification. The thesis does not attempt diagnosis, intervention targeting, or causal explanation. It asks whether these two variables add measurable predictive information.

## Slide 3. Literature Framing And Gap
The literature already documents strong associations between bullying exposure and adverse mental-health outcomes. The gap here is methodological rather than conceptual: this project tests incremental predictive value under a frozen validation protocol, with explicit attention to calibration, reproducibility, and scope control. That keeps the contribution empirical and bounded.

## Slide 4. Data Source And Study Boundary
The modeling data come from the 2023 national YRBS mental-health subset. The raw workbook contains `20,103` rows, and the final modeling table contains `19,863` rows after dropping records missing the primary outcome. Survey weights, strata, and PSUs are retained for descriptive prevalence work, but predictive training and held-out evaluation remain unweighted by design.

## Slide 5. Outcome And Predictor Specification
The outcome is `QN26`, coded as `y_qn26`. The locked demographic baseline contains `q1`, `q2`, `q3`, and `raceeth`. The bullying block adds `x_qn24` and `x_qn25`. Secondary outcomes `QN27` through `QN30` remain in the modeling table for continuity with the broader project record, but they are not part of the final thesis comparison.

## Slide 6. Frozen Evaluation Protocol
All headline comparisons use the stored held-out split and stored cross-validation folds tied to seed `2026`. The main model families are logistic regression for the linear reference and histogram gradient boosting for the nonlinear candidate. The evaluation focuses on ROC AUC, PR AUC, Brier score, calibration slope, and calibration intercept, so ranking performance and probability quality are reviewed together.

## Slide 7. Baseline Versus Bullying-Augmented Comparison
The baseline logistic reference reaches held-out ROC AUC `0.6498` and Brier `0.2320`. The full HGB candidate without post-hoc calibration reaches held-out ROC AUC `0.7161` and Brier `0.2062`. Relative to the frozen logistic baseline, the full-feature comparison improves ROC AUC by `0.0662`, PR AUC by `0.0966`, and lowers Brier by `0.0258`. The bullying-block ablation is the key result: adding the bullying block raises ROC AUC by `0.0659`, raises PR AUC by `0.1062`, and lowers Brier by `0.0190`.

## Slide 8. Calibration And Model-Selection Logic
For the full HGB candidate, calibration is compared under no post-hoc adjustment, Platt scaling, and isotonic regression. Platt scaling produces the best final tradeoff on held-out data, with Brier `0.2061` and calibration slope `1.0055`, slightly improving on the uncalibrated version while preserving the ROC AUC of `0.7161`. That is why the final reported configuration is `hgb_full_platt`.

## Slide 9. Robustness Checks
The retained robustness checks support stability rather than novelty. Across seeds `2026` through `2029`, the standard deviations are very small: roughly `0.00042` for ROC AUC, `0.00111` for PR AUC, `0.00005` for Brier, and `0.00459` for calibration slope. The held-out bootstrap interval for ROC AUC is approximately `0.699` to `0.733`, and local hyperparameter changes around the tuned HGB baseline do not overturn the ranking. In the permutation-importance summary, `x_qn25` ranks second and `x_qn24` ranks third behind `q2`, which is consistent with the central thesis claim.

## Slide 10. Subgroup Review
The subgroup review stays descriptive and thresholded. Only `q2` and `raceeth` groups with adequate sample size are interpreted for ROC AUC and calibration slope. Performance varies across groups, but the audit does not trigger any retained low-slope or high-error flags among adequately sized cells. Small groups are still visible in the table, but their undefined ROC AUC or slope values are treated as adequacy limits rather than substantive findings.

## Slide 11. Limitations And Governance
The project uses one public, cross-sectional survey subset and therefore cannot support causal claims. Weighted descriptive context is intentionally kept separate from unweighted predictive evaluation. The work does not attempt individual screening, intervention targeting, or operational deployment. Those governance boundaries are not side notes; they are part of the thesis design.

## Slide 12. Conclusion And Future Work
The conclusion is narrow: under the locked demographic baseline and frozen held-out protocol, the bullying variables add modest but consistent incremental predictive value for `QN26`. The gains are large enough to register clearly in the retained comparison tables, but they do not change the study's non-causal interpretation. Reasonable next steps lie outside this thesis scope: external validation, stronger subgroup sample sizes, and prospective study designs.
