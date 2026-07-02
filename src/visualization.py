import matplotlib
matplotlib.use("Agg")                      # save to file, no interactive window
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "axes.titlesize": 11})


def _save(fig, outdir, name):
    fig.tight_layout()
    path = outdir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_levels(df, outdir):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    df[["EUR", "GBP"]].plot(ax=ax[0]); ax[0].set_title("EUR/USD and GBP/USD")
    df[["Fixing", "IBrate"]].plot(ax=ax[1]); ax[1].set_title("USD/TND: BCT fixing vs interbank")
    return _save(fig, outdir, "levels.png")


def plot_log_returns(df, outdir):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    df["dlogFix"].plot(ax=ax, lw=0.7)
    ax.set_title("USD/TND fixing log-returns"); ax.set_xlabel("")
    return _save(fig, outdir, "log_returns.png")


def plot_acf_pacf(series, outdir, name="acf_pacf.png"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    plot_acf(series.dropna(), ax=ax[0], lags=20, title="ACF")
    plot_pacf(series.dropna(), ax=ax[1], lags=20, title="PACF", method="ywm")
    return _save(fig, outdir, name)


def plot_fit(actual, pred, title, outdir, name):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    actual.plot(ax=ax, label="actual", lw=1)
    pred.plot(ax=ax, label="nowcast", lw=1, alpha=0.8)
    ax.set_title(title); ax.legend(); ax.set_xlabel("")
    return _save(fig, outdir, name)


def plot_weights(states, outdir):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    states[["w_EUR", "w_GBP", "w_JPY"]].plot(ax=ax, lw=1)
    ax.set_title("Kalman-filtered basket weights"); ax.set_xlabel("")
    return _save(fig, outdir, "kalman_weights.png")


def plot_residual_diag(actual, pred, outdir, name="residual_diag.png"):
    a, p = actual.align(pred, join="inner")
    resid = a - p
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(resid, bins=30, edgecolor="k", alpha=0.7); ax[0].set_title("Residuals")
    ax[1].scatter(a, p, s=8, alpha=0.5)
    lo, hi = a.min(), a.max()
    ax[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[1].set_title("Actual vs nowcast"); ax[1].set_xlabel("actual"); ax[1].set_ylabel("nowcast")
    return _save(fig, outdir, name)


def plot_divergence(df, outdir, window=60):
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    df[["Fixing", "IBrate"]].plot(ax=ax[0]); ax[0].set_title("BCT fixing vs interbank (USD/TND)")
    df["spread"].plot(ax=ax[1], color="darkred", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("Spread (interbank - fixing)")
    roll = df["dlogFix"].rolling(window).corr(np.log(df["IBrate"]).diff())
    roll.plot(ax=ax[2], color="purple", lw=0.9)
    ax[2].set_title(f"{window}-day rolling correlation of fixing vs interbank returns")
    ax[2].set_xlabel("")
    return _save(fig, outdir, "divergence.png")


def plot_bidask(fixing_ba, outdir):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    (fixing_ba["ba_spread_rel"] * 1e4).plot(ax=ax, lw=0.7, color="teal")
    ax.set_title("Fixing bid/ask spread (basis points) - liquidity proxy"); ax.set_xlabel("")
    return _save(fig, outdir, "bidask_spread.png")


def plot_garch_vol(vol, outdir):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    (vol * 100).plot(ax=ax, color="black", lw=0.8)
    ax.set_title("GARCH(1,1) conditional volatility of fixing log-returns (%)"); ax.set_xlabel("")
    return _save(fig, outdir, "garch_vol.png")


def plot_sensitivity(sens, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = sens["weight"] + " " + sens["change"]
    ax.bar(labels, sens["MAPE(%)"], color="steelblue", alpha=0.85)
    base = sens.attrs.get("base_mape")
    if base is not None:
        ax.axhline(base, color="red", ls="--", lw=1, label=f"fitted weights ({base}%)")
        ax.legend()
    ax.set_ylabel("test MAPE (%)"); ax.set_title("Basket-weight sensitivity")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, outdir, "sensitivity.png")
