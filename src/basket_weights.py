import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import FX_COLS

# Three ways of estimating the basket weights, in the order we tried them:
#   1. constant weights by OLS  (the benchmark)
#   2. online weights by SGD with partial_fit  (meant to drift, but stays flat)
#   3. weights produced by inverting the fixing equation, then learned by a Random Forest
# The dynamic version that actually works (Kalman) lives in models.py.


def ols_weights(train):
    """Constant basket weights: regress the fixing log-return on the three majors."""
    X = train[FX_COLS].values
    y = train["dlogFix"].values
    return LinearRegression().fit(X, y)


def sgd_online_weights(df, eta0=1e-3, seed=0):
    """Update the weights one day at a time and record their path. Returns (path, final)."""
    sx = StandardScaler().fit(df[FX_COLS].values)
    sy = StandardScaler().fit(df[["dlogFix"]].values)
    Xs = sx.transform(df[FX_COLS].values)
    ys = sy.transform(df[["dlogFix"]].values).ravel()

    model = SGDRegressor(penalty=None, learning_rate="constant", eta0=eta0, random_state=seed)
    coefs = np.zeros((len(df), 3))
    for t in range(len(df)):
        model.partial_fit(Xs[t:t + 1], ys[t:t + 1])
        coefs[t] = model.coef_
    # rescale standardised coefficients back to log-return units for interpretation
    scale = sy.scale_[0] / sx.scale_
    path = pd.DataFrame(coefs * scale, index=df.index, columns=["w_EUR", "w_GBP", "w_JPY"])
    return path, path.iloc[-1].values


def rf_target_weights(df, window=5, seed=0):
    """Invert the fixing equation over a rolling window to get target weights, then learn them.

    For each day we solve, in a least-squares sense, the weights that reproduce the last
    `window` daily fixing moves from the basket moves, and train a Random Forest to map the
    recent FX returns onto those weights. Returns (rmse, target, predicted) on the test split.
    """
    r = df[FX_COLS].values
    yfix = df["dlogFix"].values
    n = len(df)
    target = np.full((n, 3), np.nan)
    feat = np.full((n, 3 * window), np.nan)
    for t in range(window, n):
        A = r[t - window:t]
        w, *_ = np.linalg.lstsq(A, yfix[t - window:t], rcond=None)
        target[t] = w
        feat[t] = r[t - window:t].ravel()

    keep = ~np.isnan(target).any(axis=1)
    X, Y = feat[keep], target[keep]
    split = int(0.8 * len(X))
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X[:split], Y[:split])
    pred = rf.predict(X[split:])
    rmse = np.sqrt(mean_squared_error(Y[split:], pred))
    return rmse, Y[split:], pred
