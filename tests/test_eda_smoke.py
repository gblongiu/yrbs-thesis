import subprocess
import sys
from pathlib import Path


def test_eda_smoke():
    repo_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "02_eda.py"),
        "--nrows",
        "2000",
        "--outdir",
        "outputs",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    required_paths = [
        "outputs/tables/missingness_eda.csv",
        "outputs/tables/unweighted_prevalence_overall.csv",
        "outputs/tables/weighted_prevalence_overall.csv",
        "outputs/tables/weighted_prevalence_by_q1.csv",
        "outputs/tables/weighted_prevalence_by_q2.csv",
        "outputs/tables/weighted_prevalence_by_q3.csv",
        "outputs/tables/weighted_prevalence_by_raceeth.csv",
        "outputs/tables/value_counts_y_qn26.csv",
        "outputs/tables/value_counts_x_qn24.csv",
        "outputs/tables/value_counts_x_qn25.csv",
        "outputs/tables/value_counts_q1.csv",
        "outputs/tables/value_counts_q2.csv",
        "outputs/tables/value_counts_q3.csv",
        "outputs/tables/value_counts_raceeth.csv",
        "outputs/tables/value_counts_weight.csv",
        "outputs/tables/value_counts_stratum.csv",
        "outputs/tables/value_counts_psu.csv",
        "outputs/figures/missingness_bar.png",
        "outputs/figures/prevalence_overall_weighted_vs_unweighted.png",
        "outputs/figures/qn26_prevalence_by_raceeth.png",
        "outputs/figures/qn26_prevalence_by_q1.png",
        "outputs/logs/eda_run_metadata.json",
    ]

    for rel in required_paths:
        assert (repo_root / rel).exists(), f"Missing expected EDA artifact: {rel}"

