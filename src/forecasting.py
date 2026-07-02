import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def split_index(n, frac=0.8):
    return int(frac * n)


def directional_accuracy(actual, pred):
    a = np.sign(actual.diff().dropna())
    p = np.sign(pred.diff().dropna())
    common = a.index.intersection(p.index)
    return 100.0 * (a.loc[common] == p.loc[common]).mean()


def evaluate(actual, pred, mape=True):
    """Error metrics on aligned actual/predicted series. MAPE only for positive levels."""
    a, p = actual.align(pred, join="inner")
    out = {"RMSE": np.sqrt(mean_squared_error(a, p)),
           "MAE": mean_absolute_error(a, p),
           "Corr": a.corr(p),
           "DirAcc(%)": directional_accuracy(a, p)}
    if mape:
        out["MAPE(%)"] = mean_absolute_percentage_error(a, p) * 100
    return out


def random_walk(series):
    """Naive nowcast: today equals yesterday."""
    return series.shift(1)


def reconstruct_level(level_lag1, dlog_pred):
    return level_lag1 * np.exp(dlog_pred)


def performance_table(results, order=None):
    tbl = pd.DataFrame(results).T
    cols = ["RMSE", "MAE", "MAPE(%)", "Corr", "DirAcc(%)"]
    tbl = tbl[[c for c in cols if c in tbl.columns]]
    if order:
        tbl = tbl.loc[order]
    return tbl.round(4)


def basket_sensitivity(df, weights, test_idx, bump=0.5):
    """How the implied USD/TND nowcast moves when each basket weight is bumped +/- bump.

    Reports the change in test-window MAPE relative to the fitted weights, so we can see
    which leg the valuation leans on and how fragile the implied level is.
    """
    fx = df.loc[test_idx, ["dlogEUR", "dlogGBP", "dlogJPY"]].values
    lag1 = df.loc[test_idx, "Fixing_lag1"].values
    actual = df.loc[test_idx, "Fixing"].values
    base_w = np.asarray(weights, dtype=float)

    def mape(w):
        pred = lag1 * np.exp(fx.dot(w))
        return mean_absolute_percentage_error(actual, pred) * 100

    rows = []
    for j, name in enumerate(["w_EUR", "w_GBP", "w_JPY"]):
        for sign in (-1, 1):
            w = base_w.copy()
            w[j] = w[j] * (1 + sign * bump)
            rows.append({"weight": name, "change": f"{sign * bump:+.0%}",
                         "value": round(w[j], 4), "MAPE(%)": round(mape(w), 4)})
    out = pd.DataFrame(rows)
    out.attrs["base_mape"] = round(mape(base_w), 4)
    return out
