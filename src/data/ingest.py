from pathlib import Path
import pandas as pd


def load_yrbs_raw(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)
