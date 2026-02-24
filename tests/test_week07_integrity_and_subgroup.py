from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_module(path: Path, name: str):
    scripts_dir = str(path.resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def test_subgroup_undefined_handling_does_not_crash() -> None:
    module = _load_module(Path("scripts/12_subgroup_audit.py"), "week07_subgroup")

    y_group = np.array([0, 0, 0, 0, 0], dtype=int)
    p_group = np.array([0.1, 0.2, 0.15, 0.08, 0.05], dtype=float)

    row = module.compute_subgroup_metrics_row(
        subgroup_var="q2",
        subgroup_value="cat_1",
        y_group=y_group,
        p_group=p_group,
        overall_brier=0.20,
        seed=2026,
    )

    assert row["n"] == 5
    assert row["n_pos"] == 0
    assert row["n_neg"] == 5
    assert pd.isna(row["roc_auc"])
    assert row["roc_auc_defined_flag"] is False
    assert pd.isna(row["calibration_slope"])
    assert row["slope_defined_flag"] is False


def test_manifest_immutability_whitelist_is_enforced(tmp_path: Path) -> None:
    module = _load_module(Path("scripts/13_upgrade_integrity_check.py"), "week07_integrity")

    whitelist_rel = "docs/status_reports/report_03/Project_Status_Report_03_Submission.md"
    non_whitelist_rel = "outputs/tables/week06_full_feature_comparison_seed2026.csv"

    whitelist_file = tmp_path / whitelist_rel
    non_whitelist_file = tmp_path / non_whitelist_rel
    whitelist_file.parent.mkdir(parents=True, exist_ok=True)
    non_whitelist_file.parent.mkdir(parents=True, exist_ok=True)

    whitelist_file.write_text("new whitelist content", encoding="utf-8")
    non_whitelist_file.write_text("new non whitelist content", encoding="utf-8")

    manifest_payload = {
        "new_files": [
            {
                "path": whitelist_rel,
                "sha256": _sha256_bytes(b"old whitelist content"),
            },
            {
                "path": non_whitelist_rel,
                "sha256": _sha256_bytes(b"old non whitelist content"),
            },
        ]
    }
    manifest_path = tmp_path / "docs/status_reports/report_03/week06_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    failures = module.check_week06_manifest_immutability(
        project_root=tmp_path,
        manifest_path=manifest_path,
        whitelist=sorted(module.WEEK6_HASH_WHITELIST),
    )

    assert any(non_whitelist_rel in msg for msg in failures)
    assert not any(whitelist_rel in msg for msg in failures)


def test_schema_validation_catches_missing_columns() -> None:
    module = _load_module(Path("scripts/13_upgrade_integrity_check.py"), "week07_integrity_schema")
    df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(RuntimeError):
        module.ensure_required_columns(df, ["a", "b", "c"], "test_table")
