from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
