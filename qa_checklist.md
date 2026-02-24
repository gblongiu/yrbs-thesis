# QA Checklist

## Week 7 Script Execution
- [ ] Run `scripts/09_multiseed_stability.py` and confirm 4 seed rows.
- [ ] Run `scripts/10_bootstrap_ci.py` and confirm held-out-only sample scope fields.
- [ ] Run `scripts/11_hyperparameter_sensitivity.py` and confirm 7 configuration rows.
- [ ] Run `scripts/12_subgroup_audit.py` and confirm both subgroup dimensions are present.
- [ ] Run `scripts/14_week07_report_upgrade.py` and confirm report files are regenerated.
- [ ] Run `scripts/13_upgrade_integrity_check.py` and confirm audit result PASS.

## Schema and Content Checks
- [ ] `outputs/tables/multiseed_stability_seed2026_2029.csv` has required columns and repeated across-seed standard deviation values.
- [ ] `outputs/tables/heldout_bootstrap_ci_seed2026.csv` has exactly metrics `roc_auc`, `brier`, `calibration_slope`.
- [ ] `outputs/tables/hgb_hyperparameter_sensitivity_seed2026.csv` includes baseline plus six perturbations.
- [ ] `outputs/tables/hgb_seed2026_full_perm_importance_summary_extended.csv` includes `x_qn24` and `x_qn25`.
- [ ] `outputs/tables/subgroup_performance_seed2026.csv` includes undefined flags and subgroup counts.

## Integrity and Immutability Checks
- [ ] Frozen Week 4 and Week 5 hashes remain unchanged.
- [ ] Week 6 manifest hash checks pass with whitelist exclusions only.
- [ ] Report section order matches required six-section template.
- [ ] Report markdown and docx contain no semicolons and no em dash characters.

## Testing
- [ ] Run `.venv/bin/pytest -q` and confirm all tests pass.
