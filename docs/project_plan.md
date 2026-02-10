# Project Plan

Status checkpoint (as of 2026-02-09): Weeks 1-4 complete, Weeks 5-10 planned.

## Objectives
- Create an analysis-ready modeling table with documented coding decisions and a variable dictionary keyed to exact YRBS columns.
- Train interpretable baseline models and compare them to boosted models under a fixed validation protocol.
- Estimate the incremental predictive value of bullying exposure variables using ablation comparisons.
- Assess probability calibration and provide defensible error analysis.
- Evaluate robustness across available demographic subgroups and document limitations.

## Deliverables
- Cleaned analysis dataset and data dictionary.
- EDA report with prevalence tables, missingness profile, and visualizations.
- Modeling report with cross-validated metrics and held-out test evaluation.
- Interpretability outputs for baseline and boosted models.
- Subgroup robustness report for performance and calibration.
- Final thesis paper and narrated presentation with reproducible appendix.

## Milestone Timeline (10 Weeks)
| Week | Tasks | End-of-week outcome |
| --- | --- | --- |
| Week 1 | Lock research question, primary target, core exposures, baseline covariates, and metrics. Finalize repo structure and reproducibility checklist. | Problem statement and evaluation plan are frozen. |
| Week 2 | Import YRBS 2023 data, standardize coding, finalize missingness strategy. Build modeling table and data dictionary. | Analysis-ready dataset and variable dictionary complete. |
| Week 3 | EDA with weighted prevalence summaries for QN24 to QN30 and key covariates. Missingness profile and core plots. | EDA section complete with figures and tables. |
| Week 4 | Train baseline models with stratified cross-validation. Establish frozen test split protocol and baseline metric report. | Baseline results and validation framework complete. |
| Week 5 | Train and tune boosted models within cross-validation. Maintain experiment log and compare ROC AUC, PR AUC, and Brier score. | Competitive models and transparent experiment record. |
| Week 6 | Ablation study to measure incremental value of QN24 and QN25. Begin interpretability outputs. | Evidence of bullying exposure contribution and initial explanations. |
| Week 7 | Finalize model selection. Produce test evaluation, calibration plots, and polished figures and tables. | Final results package complete with calibrated outputs. |
| Week 8 | Write the first complete paper draft. Integrate figures, tables, and reproducibility appendix. | Complete first draft ready for revision. |
| Week 9 | Revise for clarity and rigor. Strengthen limitations and ethics. Draft narrated presentation outline. | Near-final paper and presentation plan complete. |
| Week 10 | Finalize paper and references. Record narrated presentation. Package code and artifacts for submission. | All deliverables complete and backed up. |

## Risk Controls
- Keep trend analysis as descriptive context only and confine full trend tables to an appendix.
- Lock primary outcome and core exposures by Week 2.
- Choose final model based on performance, calibration, and interpretability.
- Require improvements to replicate across folds and split seeds.
- Treat new datasets or outcomes as stretch goals after Week 8 deliverables are complete.
