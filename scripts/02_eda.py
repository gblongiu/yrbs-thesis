from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, Tuple

_mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR  # noqa: E402


REQUIRED_COLUMNS = [
    "y_qn26",
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

VALUE_COUNTS_COLS = [
    "y_qn26",
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


def _stringify_value(v) -> str:
    if pd.isna(v):
        return "<NA>"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    return str(v)


def _write_value_counts(series: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = series.map(_stringify_value)
    vc = labels.value_counts(dropna=False)
    counts = vc.rename_axis("value").reset_index(name="count")
    counts["proportion"] = (counts["count"] / len(series)).round(6)
    counts = counts.sort_values(["count", "value"], ascending=[False, True], kind="mergesort")
    counts.to_csv(out_path, index=False)


def _safe_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in name)


def _weighted_prevalence_with_ci(
    indicator: np.ndarray, weights: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float, str]:
    """Return (weighted_n, weighted_rate, ci_low, ci_high, method).

    CI is approximate (weight-only), not design-based.
    """

    if indicator.size == 0:
        return 0.0, np.nan, np.nan, np.nan, "empty"

    w = weights.astype(float)
    x = indicator.astype(float)

    w_sum = float(w.sum())
    if w_sum <= 0:
        return w_sum, np.nan, np.nan, np.nan, "nonpositive_weight_sum"

    mean = float(np.sum(w * x) / w_sum)

    # Preferred path: statsmodels' DescrStatsW t-based CI on the weighted mean.
    try:
        from statsmodels.stats.weightstats import DescrStatsW  # type: ignore

        ds = DescrStatsW(x, weights=w, ddof=0)
        ci_low, ci_high = ds.tconfint_mean(alpha=alpha)
        method = "statsmodels_DescrStatsW_tconfint_mean"
        return w_sum, mean, float(ci_low), float(ci_high), method
    except Exception:
        # Fallback: normal approximation using an effective sample size.
        # n_eff = (sum(w)^2) / sum(w^2)
        w2_sum = float(np.sum(w * w))
        n_eff = (w_sum * w_sum / w2_sum) if w2_sum > 0 else np.nan
        var_w = float(np.sum(w * (x - mean) ** 2) / w_sum)
        se = float(np.sqrt(var_w / n_eff)) if (n_eff and n_eff > 0) else np.nan

        z = 1.959963984540054  # ~N(0,1) 97.5% quantile
        ci_low = mean - z * se
        ci_high = mean + z * se
        ci_low = float(max(0.0, ci_low))
        ci_high = float(min(1.0, ci_high))
        method = "normal_approx_n_eff"
        return w_sum, mean, ci_low, ci_high, method


def _compute_missingness_eda(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns.astype(str).tolist():
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = round((n_missing / n) * 100.0, 6) if n else np.nan
        n_unique = int(s.nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n": n,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_unique": n_unique,
            }
        )
    return pd.DataFrame(rows)


def _binary_prevalence_unweighted(series: pd.Series) -> Tuple[int, int, float]:
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return 0, 0, np.nan

    values = set(non_missing.unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Expected binary {{0,1}} values but observed: {sorted(values)}")

    n = int(len(non_missing))
    positive_n = int(non_missing.eq(1).sum())
    rate = round(positive_n / n, 6) if n else np.nan
    return n, positive_n, rate


def _group_sort_key(value: str) -> tuple:
    if value == "<NA>":
        return (2, 0, value)
    if value.startswith("cat_"):
        suffix = value[4:]
        try:
            return (0, int(suffix), value)
        except Exception:
            return (1, 0, value)
    return (1, 0, value)


def _weighted_prevalence_by_group(
    df: pd.DataFrame, group_col: str, weight_col: str, target_col: str, exposure_cols: Iterable[str]
) -> tuple[pd.DataFrame, str]:
    # Include an explicit <NA> group for transparency.
    group_vals = df[group_col].map(_stringify_value)
    tmp = df.copy()
    tmp["_group"] = group_vals

    out_rows = []
    ci_method_used = None

    for g, gdf in tmp.groupby("_group", sort=False):
        w = gdf[weight_col]
        y = gdf[target_col]

        mask = y.notna() & w.notna()
        indicator = y.loc[mask].to_numpy(dtype=float)
        weights = w.loc[mask].to_numpy(dtype=float)

        weighted_n, weighted_rate, ci_low, ci_high, ci_method = _weighted_prevalence_with_ci(indicator, weights)
        ci_method_used = ci_method_used or ci_method

        row = {
            "group_value": g,
            "n": int(len(gdf)),
            "weighted_n": round(weighted_n, 6),
            "y_qn26_weighted_rate": round(weighted_rate, 6) if not pd.isna(weighted_rate) else np.nan,
            "y_qn26_ci95_low_approx": round(ci_low, 6) if not pd.isna(ci_low) else np.nan,
            "y_qn26_ci95_high_approx": round(ci_high, 6) if not pd.isna(ci_high) else np.nan,
            "y_qn26_missing_rate": round(float(gdf[target_col].isna().mean()), 6),
        }

        for ex in exposure_cols:
            row[f"{ex}_missing_rate"] = round(float(gdf[ex].isna().mean()), 6)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out = out.sort_values("group_value", key=lambda s: s.map(_group_sort_key), kind="mergesort")
    return out, (ci_method_used or "unknown")


def _write_run_metadata(logs_dir: Path, payload: dict) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "eda_run_metadata.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis EDA: descriptive missingness + weighted prevalence tables.")
    parser.add_argument("--nrows", type=int, default=None, help="Use only the first N rows (deterministic head).")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory (default: outputs/).")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (reserved for future sampling).")
    args = parser.parse_args()

    in_path = PROCESSED_DIR / "yrbs_2023_modeling.parquet"
    if not in_path.exists():
        raise SystemExit(f"Modeling table not found: {in_path}. Run scripts/01_build_dataset.py first.")

    df = pd.read_parquet(in_path)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns in modeling table: {missing_required}")

    if args.nrows is not None:
        if args.nrows <= 0:
            raise SystemExit("--nrows must be a positive integer.")
        df = df.head(args.nrows).copy()

    outdir = args.outdir
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    logs_dir = outdir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Run metadata
    packages = {}
    for pkg in ["pandas", "numpy", "pyarrow", "matplotlib", "statsmodels", "scikit-learn"]:
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "input_parquet": str(in_path),
        "nrows": args.nrows,
        "seed": args.seed,
        "outdir": str(outdir),
        "notes": [
            "EDA is descriptive context only (no predictive modeling).",
            "Weighted prevalence uses weight-only estimation; CIs are approximate and not design-based.",
        ],
    }

    # Missingness table
    miss = _compute_missingness_eda(df)
    miss.to_csv(tables_dir / "missingness_eda.csv", index=False)

    # Value counts
    for col in VALUE_COUNTS_COLS:
        _write_value_counts(df[col], tables_dir / f"value_counts_{_safe_filename(col)}.csv")

    # Unweighted prevalence overall
    prevalence_vars = ["y_qn26", "x_qn24", "x_qn25"]
    for qn in [27, 28, 29, 30]:
        col = f"y_qn{qn}"
        if col in df.columns:
            prevalence_vars.append(col)

    unweighted_rows = []
    for col in prevalence_vars:
        n, positive_n, rate = _binary_prevalence_unweighted(df[col])
        unweighted_rows.append({"variable": col, "n": n, "positive_n": positive_n, "rate": rate})
    unweighted_prev = pd.DataFrame(unweighted_rows)
    unweighted_prev.to_csv(tables_dir / "unweighted_prevalence_overall.csv", index=False)

    # Weighted prevalence overall (weight-only; approximate CI)
    weight_col = "weight"
    weighted_rows = []
    ci_methods = set()
    for col in prevalence_vars:
        s = df[col]
        w = df[weight_col]
        mask = s.notna() & w.notna()
        non_missing = s.loc[mask]

        values = set(non_missing.unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Expected binary {{0,1}} values in {col} but observed: {sorted(values)}")

        indicator = non_missing.to_numpy(dtype=float)
        weights = w.loc[mask].to_numpy(dtype=float)
        weighted_n, weighted_rate, ci_low, ci_high, ci_method = _weighted_prevalence_with_ci(indicator, weights)
        ci_methods.add(ci_method)

        weighted_rows.append(
            {
                "variable": col,
                "weighted_n": round(weighted_n, 6),
                "weighted_rate": round(weighted_rate, 6) if not pd.isna(weighted_rate) else np.nan,
                "ci95_low_approx": round(ci_low, 6) if not pd.isna(ci_low) else np.nan,
                "ci95_high_approx": round(ci_high, 6) if not pd.isna(ci_high) else np.nan,
            }
        )

    weighted_prev = pd.DataFrame(weighted_rows)
    weighted_prev.to_csv(tables_dir / "weighted_prevalence_overall.csv", index=False)

    # Weighted prevalence of y_qn26 by baseline covariate groups
    exposure_cols = ["x_qn24", "x_qn25"]
    group_outputs = {
        "q1": "weighted_prevalence_by_q1.csv",
        "q2": "weighted_prevalence_by_q2.csv",
        "q3": "weighted_prevalence_by_q3.csv",
        "raceeth": "weighted_prevalence_by_raceeth.csv",
    }

    for group_col, filename in group_outputs.items():
        table, _ci_method = _weighted_prevalence_by_group(
            df=df,
            group_col=group_col,
            weight_col=weight_col,
            target_col="y_qn26",
            exposure_cols=exposure_cols,
        )
        table.to_csv(tables_dir / filename, index=False)

    # Figures (matplotlib only; no seaborn)
    # Missingness bar
    miss_sorted = miss.sort_values(["pct_missing", "column"], ascending=[False, True], kind="mergesort")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(miss_sorted["column"], miss_sorted["pct_missing"])
    ax.set_title("Missingness by Column (Modeling Table)")
    ax.set_ylabel("Percent missing")
    ax.set_xlabel("Column")
    ax.set_ylim(0, max(1.0, float(miss_sorted["pct_missing"].max()) * 1.05))
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "missingness_bar.png", dpi=300)
    plt.close(fig)

    # Overall prevalence: weighted vs unweighted
    plot_vars = ["y_qn26", "x_qn24", "x_qn25"]
    uw = unweighted_prev.set_index("variable").loc[plot_vars]
    ww = weighted_prev.set_index("variable").loc[plot_vars]

    x = np.arange(len(plot_vars))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, uw["rate"], width, label="Unweighted")
    ax.bar(x + width / 2, ww["weighted_rate"], width, label="Weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_vars)
    ax.set_ylabel("Prevalence (rate)")
    ax.set_title("Overall Prevalence: Weighted vs Unweighted")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "prevalence_overall_weighted_vs_unweighted.png", dpi=300)
    plt.close(fig)

    # QN26 prevalence by raceeth (weighted)
    raceeth_tbl = pd.read_csv(tables_dir / "weighted_prevalence_by_raceeth.csv", keep_default_na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        raceeth_tbl["group_value"],
        raceeth_tbl["y_qn26_weighted_rate"],
        yerr=[
            raceeth_tbl["y_qn26_weighted_rate"] - raceeth_tbl["y_qn26_ci95_low_approx"],
            raceeth_tbl["y_qn26_ci95_high_approx"] - raceeth_tbl["y_qn26_weighted_rate"],
        ],
        capsize=3,
    )
    ax.set_title("Weighted Prevalence of QN26 by raceeth (Approx. 95% CI)")
    ax.set_ylabel("Weighted prevalence (rate)")
    ax.set_xlabel("raceeth (coded categories)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "qn26_prevalence_by_raceeth.png", dpi=300)
    plt.close(fig)

    # QN26 prevalence by q1 (weighted)
    q1_tbl = pd.read_csv(tables_dir / "weighted_prevalence_by_q1.csv", keep_default_na=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        q1_tbl["group_value"],
        q1_tbl["y_qn26_weighted_rate"],
        yerr=[
            q1_tbl["y_qn26_weighted_rate"] - q1_tbl["y_qn26_ci95_low_approx"],
            q1_tbl["y_qn26_ci95_high_approx"] - q1_tbl["y_qn26_weighted_rate"],
        ],
        capsize=3,
    )
    ax.set_title("Weighted Prevalence of QN26 by q1 (Approx. 95% CI)")
    ax.set_ylabel("Weighted prevalence (rate)")
    ax.set_xlabel("q1 (coded categories)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "qn26_prevalence_by_q1.png", dpi=300)
    plt.close(fig)

    run_meta["ci_methods_observed"] = sorted(ci_methods)
    _write_run_metadata(logs_dir, run_meta)

    print(f"Wrote EDA artifacts to {outdir}/")


if __name__ == "__main__":
    main()
