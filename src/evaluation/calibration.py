import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_curve_df(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
