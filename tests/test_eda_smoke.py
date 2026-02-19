import hashlib
import os
import subprocess
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eda_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    metadata_path = repo_root / "outputs/logs/eda_run_metadata.json"
    pre_hash = _sha256_file(metadata_path) if metadata_path.exists() else None

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "02_eda.py"),
        "--nrows",
        "2000",
        "--outdir",
        "outputs",
    ]
    env = os.environ.copy()
    env["YRBS_THESIS_DISABLE_RUN_METADATA"] = "1"
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)

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
    ]

    for rel in required_paths:
        assert (repo_root / rel).exists(), f"Missing expected EDA artifact: {rel}"

    post_hash = _sha256_file(metadata_path) if metadata_path.exists() else None
    if pre_hash is None:
        assert not metadata_path.exists(), "Metadata file was created despite suppression."
    else:
        assert post_hash == pre_hash, "Metadata file changed despite suppression."
