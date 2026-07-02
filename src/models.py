import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ----------------------------------------------------------------------------
# Kalman filter for time-varying basket weights
# ----------------------------------------------------------------------------
# State: the basket weights, optionally plus a constant intercept term. The
# weights follow a random walk; the observation is the fixing log-return. We
# keep the one-step-ahead prediction (made from the prior state, before seeing
# the current observation) so the evaluation is genuinely out-of-sample.

def kalman_filter(X, y, q, r, intercept=True, q_intercept=None, p0=0.1):
    n, k = X.shape
    dim = k + 1 if intercept else k
    q_int = q * 100 if q_intercept is None else q_intercept
    Q = np.diag([q] * k + [q_int]) if intercept else np.diag([q] * k)
    rt = np.full(n, r, dtype=float) if np.isscalar(r) else np.asarray(r, dtype=float)

    theta = np.zeros(dim)
    P = np.eye(dim) * p0
    states = np.zeros((n, dim))
    pred = np.zeros(n)
    for t in range(n):
        P = P + Q                                    # predict step
        H = (np.append(X[t], 1.0) if intercept else X[t]).reshape(1, dim)
        pred[t] = float(H.dot(theta))                # one-step-ahead, prior state only
        S = float(H.dot(P).dot(H.T)) + rt[t]
        K = P.dot(H.T) / S
        theta = theta + K.flatten() * (y[t] - H.dot(theta))
        P = P - K.dot(H).dot(P)
        states[t] = theta
    return states, pred


def kalman_grid_search(X, y, q_grid, r_grid, val, intercept=True, obs_var=None):
    """Pick (q, r) by the one-step RMSE of the fixing log-return on a validation slice."""
    best, best_rmse = None, np.inf
    for q in q_grid:
        for r in r_grid:
            rr = obs_var if obs_var is not None else r
            _, pred = kalman_filter(X, y, q, rr, intercept=intercept)
            rmse = np.sqrt(mean_squared_error(y[val], pred[val]))
            if rmse < best_rmse:
                best_rmse, best = rmse, (q, r)
    return best


# ----------------------------------------------------------------------------
# GARCH volatility
# ----------------------------------------------------------------------------

# daily FX log-returns are ~1e-3; arch estimates better on values of order 1-1000,
# so we fit on returns * SCALE and divide the resulting volatility back out.
SCALE = 1000.0


def garch_select(returns, specs=((1, 1), (1, 2), (2, 1), (2, 2)), dist="normal"):
    """Fit a few GARCH(p,q) specs and rank them by AIC/BIC. Returns (table, fits, best)."""
    scaled = returns.dropna().values * SCALE
    rows, fits = [], {}
    for p, q in specs:
        res = arch_model(scaled, mean="Constant", vol="GARCH", p=p, q=q, dist=dist,
                         rescale=False).fit(disp="off")
        fits[(p, q)] = res
        rows.append({"spec": f"GARCH({p},{q})", "AIC": res.aic, "BIC": res.bic,
                     "logL": res.loglikelihood})
    table = pd.DataFrame(rows).set_index("spec").round(2)
    best = min(fits, key=lambda k: fits[k].bic)
    return table, fits, best


def garch_conditional_vol(returns, p=1, q=1):
    """Conditional volatility from a GARCH(p,q) fit, back on the original return scale."""
    scaled = returns.dropna()
    res = arch_model(scaled.values * SCALE, mean="Constant", vol="GARCH", p=p, q=q,
                     rescale=False).fit(disp="off")
    return pd.Series(res.conditional_volatility / SCALE, index=scaled.index)


def garch_obs_variance(X, y, split, p=1, q=1):
    """Time-varying observation variance R_t from a GARCH(1,1) on the OLS residuals.

    Both the OLS fit and the GARCH parameters are estimated on the training span
    only; the fitted GARCH is then applied to the full residual series with those
    fixed coefficients, so R_t over the test span carries no out-of-sample info."""
    ols = LinearRegression().fit(X[:split], y[:split])
    resid = (y - ols.predict(X)) * SCALE
    params = arch_model(resid[:split], mean="Zero", vol="GARCH", p=p, q=q,
                        rescale=False).fit(disp="off").params
    res = arch_model(resid, mean="Zero", vol="GARCH", p=p, q=q,
                     rescale=False).fix(params)
    return (res.conditional_volatility / SCALE) ** 2


# ----------------------------------------------------------------------------
# Spread between the official fixing and the interbank rate
# ----------------------------------------------------------------------------

def spread_ar_onestep(spread, split, order=(1, 0, 0), exog=None):
    """One-step-ahead forecast of the spread with an AR / ARIMAX model.

    Estimated once on the training span; in-sample one-step values are kept for
    the train part and the state is rolled forward through the test part with
    fixed coefficients. Returns predictions aligned to the full index.
    """
    y = spread.reset_index(drop=True)
    ex = None if exog is None else exog.reset_index(drop=True)
    res = ARIMA(y[:split], order=order,
                exog=None if ex is None else ex[:split]).fit()
    fitted = list(res.fittedvalues)
    preds = []
    for t in range(split, len(y)):
        step_exog = None if ex is None else ex.iloc[[t]]
        preds.append(float(res.forecast(1, exog=step_exog).iloc[0]))
        res = res.append(y[t:t + 1], exog=None if ex is None else ex.iloc[[t]], refit=False)
    return pd.Series(fitted + preds, index=spread.index)


def spread_rf_onestep(df, split, lags=3, seed=0):
    """Random-forest spread nowcast from lagged spread and recent spread volatility."""
    feat = pd.DataFrame(index=df.index)
    for l in range(1, lags + 1):
        feat[f"spread_lag{l}"] = df["spread"].shift(l)
    feat["spread_vol5"] = df["spread"].rolling(5).std().shift(1)
    feat = feat.dropna()
    y = df["spread"].loc[feat.index]
    cut = df.index[split]
    tr, te = feat.index < cut, feat.index >= cut
    rf = RandomForestRegressor(n_estimators=300, random_state=seed)
    rf.fit(feat[tr].values, y[tr].values)
    pred = pd.Series(rf.predict(feat[te].values), index=feat.index[te])
    imp = pd.Series(rf.feature_importances_, index=feat.columns)
    return pred, imp


# ----------------------------------------------------------------------------
# Neural-network residual correction (evaluated strictly out-of-sample)
# ----------------------------------------------------------------------------

def residual_mlp(resid, feat, split, hidden=(32, 16), seed=0):
    """Learn the interbank residual from its own lags on the training span only,
    then predict the untouched test span. Returns test-aligned residual predictions."""
    data = feat.join(resid.rename("resid")).dropna()
    cut = resid.index[split]
    tr, te = data.index < cut, data.index >= cut
    cols = [c for c in data.columns if c != "resid"]
    model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000, random_state=seed)
    model.fit(data.loc[tr, cols].values, data.loc[tr, "resid"].values)
    pred = model.predict(data.loc[te, cols].values)
    return pd.Series(pred, index=data.index[te])
