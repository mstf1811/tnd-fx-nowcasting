import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

FX_COLS = ["dlogEUR", "dlogGBP", "dlogJPY"]


def log_return(series):
    return np.log(series / series.shift(1))


def build_features(basket):
    """Add log-returns, the lagged fixing, and the fixing-to-interbank spread."""
    df = basket.dropna(subset=["EUR", "GBP", "JPY", "Fixing", "IBrate"]).copy()
    df["Fixing_lag1"] = df["Fixing"].shift(1)
    df["dlogFix"] = log_return(df["Fixing"])
    df["dlogEUR"] = log_return(df["EUR"])
    df["dlogGBP"] = log_return(df["GBP"])
    df["dlogJPY"] = log_return(df["JPY"])
    df["spread"] = df["IBrate"] - df["Fixing"]          # market minus official, in TND
    df = df.dropna(subset=["dlogFix"] + FX_COLS + ["spread"])
    return df


def adf_table(df, cols):
    """ADF statistic, p-value and critical values for each series."""
    rows = []
    for c in cols:
        stat, p, _, _, crit, _ = adfuller(df[c].dropna(), autolag="AIC")
        rows.append({"Series": c, "ADF": stat, "p-value": p,
                     "1%": crit["1%"], "5%": crit["5%"], "10%": crit["10%"]})
    return pd.DataFrame(rows).set_index("Series").round(4)


def bidask_spread(fixing):
    """Absolute and relative bid/ask spread of the fixing, used as a liquidity proxy."""
    df = fixing.copy()
    df["ba_spread"] = df["Ask"] - df["Bid"]
    df["ba_spread_rel"] = df["ba_spread"] / df["Mid"]
    return df
