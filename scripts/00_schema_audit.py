from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_FILE_2023, TABLES_DIR  # noqa: E402


REQUIRED_COLUMNS = ["QN24", "QN25", "QN26", "raceeth", "q1"]
VALUE_COUNTS_COLUMNS = ["QN24", "QN25", "QN26", "raceeth", "q1"]
DESIGN_FIELD_CANDIDATES = ["weight", "stratum", "psu"]


def stringify_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        fv = float(value)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    return str(value)


def first_k_distinct_examples(series: pd.Series, k: int = 5) -> list[str]:
    seen: set[str] = set()
    examples: list[str] = []
    for v in series.to_numpy(copy=False):
        if pd.isna(v):
            continue
        sv = stringify_value(v)
        if sv in seen:
            continue
        seen.add(sv)
        examples.append(sv)
        if len(examples) >= k:
            break
    return examples


def is_numeric_series(series: pd.Series) -> bool:
    # pandas treats many coded survey columns as float because of NaNs.
    return pd.api.types.is_numeric_dtype(series)


def schema_notes(column: str, n_total: int, n_missing: int, n_unique: int) -> str:
    notes: list[str] = []
    missing_rate = (n_missing / n_total) if n_total else 0.0

    if n_missing == n_total:
        notes.append("all_missing")
        return ";".join(notes)

    if n_unique == 1:
        notes.append("constant")
    if missing_rate >= 0.95:
        notes.append("extreme_sparsity")
    if missing_rate >= 0.50:
        notes.append("high_missingness")
    if n_unique == n_total and n_total > 0:
        notes.append("unique_per_row")
    if n_total > 0 and (n_unique > 0.10 * n_total or n_unique >= 1000):
        notes.append("high_cardinality")

    col_lower = column.lower()
    if ("unique_per_row" in notes) and (
        col_lower.endswith("id") or col_lower in {"record", "orig_rec"} or "record" in col_lower
    ):
        notes.append("possible_identifier")

    return ";".join(notes)


def build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df)
    rows: list[dict] = []

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        missing_rate = (n_missing / n_total) if n_total else np.nan
        n_unique = int(s.nunique(dropna=True))

        dtype_inf = infer_dtype(s, skipna=True)
        examples = first_k_distinct_examples(s, k=5)

        min_val = ""
        max_val = ""
        if is_numeric_series(s):
            non_null = s.dropna()
            if len(non_null) > 0:
                min_val = non_null.min()
                max_val = non_null.max()

        rows.append(
            {
                "column": col,
                "dtype_inferred": dtype_inf,
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_rate": round(float(missing_rate), 6) if n_total else np.nan,
                "n_unique": n_unique,
                "example_values": json.dumps(examples, ensure_ascii=True),
                "min": min_val,
                "max": max_val,
                "notes": schema_notes(col, n_total=n_total, n_missing=n_missing, n_unique=n_unique),
            }
        )

    return pd.DataFrame(rows)


def write_value_counts(series: pd.Series, col_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if (ch.isalnum() or ch in {"_", "-"} or ch == ".") else "_" for ch in col_name)
    out_path = out_dir / f"value_counts_{safe_name}.csv"

    labels = series.map(stringify_value)
    vc = labels.value_counts(dropna=False)
    counts = vc.rename_axis("value").reset_index(name="count")
    counts["proportion"] = (counts["count"] / len(series)).round(6)

    counts = counts.sort_values(["count", "value"], ascending=[False, True], kind="mergesort")
    counts.to_csv(out_path, index=False)


def detect_design_fields(columns: list[str]) -> list[str]:
    lower_to_actual = {c.lower(): c for c in columns}
    found: list[str] = []
    for cand in DESIGN_FIELD_CANDIDATES:
        if cand in lower_to_actual:
            found.append(lower_to_actual[cand])
    return found


def write_missingness_summary(schema: pd.DataFrame, df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    total_cells = int(n_rows * n_cols)
    missing_cells = int(df.isna().sum().sum())
    overall_rate = round(missing_cells / total_cells, 6) if total_cells else np.nan

    col_missing = schema[["column", "n_total", "n_missing", "missing_rate"]].copy()
    col_missing = col_missing.sort_values(["missing_rate", "column"], ascending=[False, True], kind="mergesort")
    top30 = col_missing.head(30).copy()

    design_fields = detect_design_fields(df.columns.tolist())
    explicit = [c for c in REQUIRED_COLUMNS if c in df.columns] + design_fields
    explicit_rows = schema[schema["column"].isin(explicit)][["column", "n_total", "n_missing", "missing_rate"]].copy()
    explicit_rows = explicit_rows.set_index("column").loc[explicit].reset_index()

    overall_row = pd.DataFrame(
        [
            {
                "column": "__overall__",
                "n_total": total_cells,
                "n_missing": missing_cells,
                "missing_rate": overall_rate,
            }
        ]
    )

    combined = pd.concat([overall_row, explicit_rows, top30], ignore_index=True)
    combined = combined.drop_duplicates(subset=["column"], keep="first")
    combined.to_csv(out_path, index=False)


def write_qn26_prevalence(df: pd.DataFrame, group_col: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = "QN26"
    if target not in df.columns:
        raise SystemExit(f"Required column missing: {target}")
    if group_col not in df.columns:
        raise SystemExit(f"Required column missing: {group_col}")

    tmp = pd.DataFrame(
        {
            group_col: df[group_col].map(stringify_value),
            target: df[target],
        }
    )

    # Treat only {1,2} as valid responses for YRBS dichotomized QN variables.
    valid = tmp[target].isin([1, 2])
    tmp["_valid"] = valid
    tmp["_is_1"] = tmp[target].eq(1)
    tmp["_is_2"] = tmp[target].eq(2)

    grouped = tmp.groupby(group_col, sort=False)
    stats = grouped.agg(
        n=(target, "size"),
        n_valid=("_valid", "sum"),
        n_1=("_is_1", "sum"),
        n_2=("_is_2", "sum"),
    ).reset_index()

    stats["qn26_missing_rate"] = ((stats["n"] - stats["n_valid"]) / stats["n"]).round(6)

    # Default qn26_rate uses the common convention (1=yes), but we also compute the alternative (2=yes).
    stats["qn26_rate"] = np.where(stats["n_valid"] > 0, (stats["n_1"] / stats["n_valid"]).round(6), np.nan)
    stats["qn26_rate_assuming_2_yes"] = np.where(
        stats["n_valid"] > 0, (stats["n_2"] / stats["n_valid"]).round(6), np.nan
    )

    # Deterministic ordering: numeric groups ascending, then string; <NA> last.
    stats["_is_na_group"] = stats[group_col].eq("<NA>")
    stats["_group_num"] = pd.to_numeric(stats[group_col], errors="coerce")
    stats = stats.sort_values(
        ["_is_na_group", "_group_num", group_col],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    )

    stats = stats[[group_col, "n", "qn26_rate", "qn26_rate_assuming_2_yes", "qn26_missing_rate"]]
    stats.to_csv(out_path, index=False)


def main() -> None:
    if not RAW_FILE_2023.exists():
        raise SystemExit(f"Input file not found: {RAW_FILE_2023}")

    df = pd.read_excel(RAW_FILE_2023)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        available = ", ".join(df.columns.astype(str).tolist())
        raise SystemExit(
            "Missing required columns: "
            + ", ".join(missing_required)
            + f"\nAvailable columns ({len(df.columns)}): {available}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    schema = build_schema_table(df)
    schema.to_csv(TABLES_DIR / "schema.csv", index=False)

    write_missingness_summary(schema, df, TABLES_DIR / "missingness_summary.csv")

    design_fields = detect_design_fields(df.columns.tolist())
    for col in VALUE_COUNTS_COLUMNS + design_fields:
        write_value_counts(df[col], col, TABLES_DIR)

    write_qn26_prevalence(df, group_col="raceeth", out_path=TABLES_DIR / "qn26_prevalence_by_raceeth.csv")
    write_qn26_prevalence(df, group_col="q1", out_path=TABLES_DIR / "qn26_prevalence_by_q1.csv")

    print("Wrote schema audit outputs to outputs/tables/")


if __name__ == "__main__":
    main()
