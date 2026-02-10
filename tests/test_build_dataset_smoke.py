import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_dataset_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]

    out_parquet = tmp_path / "yrbs_2023_modeling.parquet"
    audit_csv = tmp_path / "modeling_table_audit.csv"
    missingness_csv = tmp_path / "missingness_modeling.csv"
    decisions_json = tmp_path / "decisions.json"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "01_build_dataset.py"),
        "--nrows",
        "500",
        "--out-parquet",
        str(out_parquet),
        "--audit-csv",
        str(audit_csv),
        "--missingness-csv",
        str(missingness_csv),
        "--decisions-json",
        str(decisions_json),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    assert out_parquet.exists()
    df = pd.read_parquet(out_parquet)

    expected_cols = [
        "y_qn26",
        "y_qn27",
        "y_qn28",
        "y_qn29",
        "y_qn30",
        "x_qn24",
        "x_qn25",
        "q1",
        "q2",
        "q3",
        "raceeth",
        "weight",
        "stratum",
        "psu",
    ]
    assert df.columns.tolist() == expected_cols

    # Target should be binary and non-missing after drop-missing-outcome filter.
    assert set(df["y_qn26"].dropna().unique().tolist()) <= {0, 1}
    assert df["y_qn26"].isna().sum() == 0

    # Exposures should be binary with possible NA.
    assert set(df["x_qn24"].dropna().unique().tolist()) <= {0, 1}
    assert set(df["x_qn25"].dropna().unique().tolist()) <= {0, 1}

    # Secondary outcomes should be binary with possible NA (rows are not filtered on these outcomes).
    for col in ["y_qn27", "y_qn28", "y_qn29", "y_qn30"]:
        assert set(df[col].dropna().unique().tolist()) <= {0, 1}

    assert audit_csv.exists()
    assert missingness_csv.exists()
    assert decisions_json.exists()

    payload = json.loads(decisions_json.read_text(encoding="utf-8"))
    assert payload["columns"]["target_primary"] == "QN26"
    assert payload["columns"]["secondary_targets"] == ["QN27", "QN28", "QN29", "QN30"]
    assert payload["columns"]["bullying_exposures"] == ["QN24", "QN25"]
