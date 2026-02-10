from typing import Iterable, Optional
import pandas as pd
from .validate import assert_required_columns


def build_modeling_table(
    df: pd.DataFrame,
    target: str,
    exposures: Iterable[str],
    covariates: Iterable[str],
    design_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    # TODO: Apply coding rules and missingness strategy once finalized.
    required = [target] + list(exposures)
    assert_required_columns(df, required)

    design_cols = list(design_cols or [])
    columns = [target] + list(exposures) + list(covariates) + design_cols
    columns = [c for c in columns if c in df.columns]
    return df[columns].copy()
