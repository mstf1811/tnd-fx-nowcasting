import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import data_loader as dl
import preprocessing as pp
import basket_weights as bw
import models as md
import forecasting as fc
import visualization as viz


def main():
    # 1) load and build features
    basket = dl.load_basket()
    fixing = dl.load_fixing()
    df = pp.build_features(basket)
    n = len(df)
    split = fc.split_index(n)
    train, test_idx = df.iloc[:split], df.index[split:]
    X, y = df[pp.FX_COLS].values, df["dlogFix"].values
    metrics = {"n_obs": n, "train": split, "test": n - split,
               "test_start": str(test_idx[0].date()), "test_end": str(test_idx[-1].date())}

    # 2) stationarity
    adf = pp.adf_table(df, pp.FX_COLS + ["dlogFix"])
    adf.to_csv(RESULTS / "adf.csv")

    # 3) constant-weight benchmark (OLS) and the two dynamic-weight attempts
    ols = bw.ols_weights(train)
    ols_fix = fc.reconstruct_level(df.loc[test_idx, "Fixing_lag1"],
                                   pd.Series(ols.predict(df.loc[test_idx, pp.FX_COLS].values),
                                             index=test_idx))
    metrics["ols_weights"] = dict(zip(["EUR", "GBP", "JPY"], np.round(ols.coef_, 4).tolist()))
    metrics["ols_fixing"] = {k: round(v, 4) for k, v in
                             fc.evaluate(df.loc[test_idx, "Fixing"], ols_fix).items()}
    sgd_path, sgd_final = bw.sgd_online_weights(df)
    rf_rmse, rf_y, rf_p = bw.rf_target_weights(df)
    metrics["sgd_final_weights"] = np.round(sgd_final, 4).tolist()
    metrics["rf_target_weight_std"] = round(float(np.nanstd(rf_y)), 4)   # how unstable the targets are

    # 4) Kalman time-varying weights -> fixing nowcast
    val = slice(split // 2, split)
    q, r = md.kalman_grid_search(X, y, [1e-6, 5e-6, 1e-5, 5e-5], [1e-4, 1e-3, 1e-2], val)
    states, dlog_pred = md.kalman_filter(X, y, q, r)
    wpath = pd.DataFrame(states[:, :3], index=df.index, columns=["w_EUR", "w_GBP", "w_JPY"])
    fix_pred = fc.reconstruct_level(df["Fixing_lag1"], pd.Series(dlog_pred, index=df.index))
    metrics["kalman_qr"] = [q, r]
    metrics["kalman_fixing"] = {k: round(v, 4) for k, v in
                                fc.evaluate(df.loc[test_idx, "Fixing"], fix_pred.loc[test_idx]).items()}

    # 5) GARCH volatility (model selection) + KF with GARCH observation noise
    gtable, gfits, gbest = md.garch_select(df["dlogFix"])
    gtable.to_csv(RESULTS / "garch_selection.csv")
    gvol = md.garch_conditional_vol(df["dlogFix"], *gbest)
    obs_var = md.garch_obs_variance(X, y, split)
    _, dlog_pred_g = md.kalman_filter(X, y, q, obs_var)
    fix_pred_g = fc.reconstruct_level(df["Fixing_lag1"], pd.Series(dlog_pred_g, index=df.index))
    metrics["garch_best"] = f"GARCH{gbest}"
    metrics["kalman_garch_fixing"] = {k: round(v, 4) for k, v in
                                      fc.evaluate(df.loc[test_idx, "Fixing"], fix_pred_g.loc[test_idx]).items()}

    # 6) spread models (interbank minus fixing). MAPE is meaningless here (spread crosses 0).
    sp_ar = md.spread_ar_onestep(df["spread"], split, order=(1, 0, 0))
    sp_x = md.spread_ar_onestep(df["spread"], split, order=(1, 0, 1), exog=df[pp.FX_COLS])
    sp_rf, sp_imp = md.spread_rf_onestep(df, split)
    for nm, sp in [("ar", sp_ar), ("arimax", sp_x)]:
        metrics[f"spread_{nm}"] = {k: round(v, 5) for k, v in
                                   fc.evaluate(df.loc[test_idx, "spread"], sp.loc[test_idx], mape=False).items()}
    metrics["spread_rf"] = {k: round(v, 5) for k, v in
                            fc.evaluate(df["spread"].loc[sp_rf.index], sp_rf, mape=False).items()}
    metrics["spread_rf_importance"] = sp_imp.round(3).to_dict()

    # 7) interbank nowcast = fixing nowcast + spread nowcast (ARIMAX is the best spread model)
    ib_base = fix_pred + sp_x
    metrics["interbank_base"] = {k: round(v, 4) for k, v in
                                 fc.evaluate(df.loc[test_idx, "IBrate"], ib_base.loc[test_idx]).items()}

    # 8) neural-network residual correction, trained on train residuals, scored on test only
    resid = df["IBrate"] - ib_base
    feat = pd.DataFrame(index=df.index)
    for l in range(1, 4):
        feat[f"resid_lag{l}"] = resid.shift(l)
    feat["spread_vol5"] = df["spread"].rolling(5).std().shift(1)
    resid_pred = md.residual_mlp(resid, feat, split)
    ib_hybrid = ib_base.loc[resid_pred.index] + resid_pred
    metrics["interbank_hybrid"] = {k: round(v, 4) for k, v in
                                   fc.evaluate(df["IBrate"].loc[resid_pred.index], ib_hybrid).items()}

    # 9) naive benchmarks
    rw_fix = fc.random_walk(df["Fixing"])
    rw_ib = fc.random_walk(df["IBrate"])
    metrics["rw_fixing"] = {k: round(v, 4) for k, v in
                            fc.evaluate(df.loc[test_idx, "Fixing"], rw_fix.loc[test_idx]).items()}
    metrics["rw_interbank"] = {k: round(v, 4) for k, v in
                               fc.evaluate(df.loc[test_idx, "IBrate"], rw_ib.loc[test_idx]).items()}

    # 10) fixing vs interbank divergence
    ib_ret = np.log(df["IBrate"]).diff()
    metrics["divergence"] = {
        "mean_spread": round(df["spread"].mean(), 5),
        "std_spread": round(df["spread"].std(), 5),
        "max_spread": round(df["spread"].max(), 5),
        "pct_days_positive": round(100 * (df["spread"] > 0).mean(), 1),
        "return_corr": round(df["dlogFix"].corr(ib_ret), 4),
    }
    fixing_ba = pp.bidask_spread(fixing)
    metrics["mean_bidask_bp"] = round(fixing_ba["ba_spread_rel"].mean() * 1e4, 2)

    # 11) basket-weight sensitivity on the implied fixing
    sens = fc.basket_sensitivity(df, ols.coef_, test_idx, bump=0.5)
    sens.to_csv(RESULTS / "sensitivity.csv", index=False)
    metrics["sensitivity_base_mape"] = sens.attrs["base_mape"]

    # 12) figures
    viz.plot_levels(df, RESULTS)
    viz.plot_log_returns(df, RESULTS)
    viz.plot_acf_pacf(df["dlogFix"], RESULTS)
    viz.plot_fit(df.loc[test_idx, "Fixing"], ols_fix, "OLS constant-weight fixing nowcast", RESULTS, "ols_fit.png")
    viz.plot_weights(wpath, RESULTS)
    viz.plot_fit(df.loc[test_idx, "Fixing"], fix_pred.loc[test_idx], "Fixing: actual vs Kalman nowcast", RESULTS, "fixing_nowcast.png")
    viz.plot_fit(df.loc[test_idx, "IBrate"], ib_base.loc[test_idx], "Interbank: actual vs nowcast", RESULTS, "interbank_nowcast.png")
    viz.plot_residual_diag(df.loc[test_idx, "IBrate"], ib_base.loc[test_idx], RESULTS)
    viz.plot_divergence(df, RESULTS)
    viz.plot_bidask(fixing_ba, RESULTS)
    viz.plot_garch_vol(gvol, RESULTS)
    viz.plot_sensitivity(sens, RESULTS)

    # 13) save metrics and print a short summary
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    levels = {"OLS fixing": metrics["ols_fixing"], "Kalman fixing": metrics["kalman_fixing"],
              "RW fixing": metrics["rw_fixing"], "Interbank base": metrics["interbank_base"],
              "Interbank hybrid": metrics["interbank_hybrid"], "RW interbank": metrics["rw_interbank"]}
    print(fc.performance_table(levels).to_string())
    print("\nGARCH selection (best by BIC: %s):" % metrics["garch_best"])
    print(gtable.to_string())
    print("\nDivergence:", metrics["divergence"], "| mean bid/ask (bp):", metrics["mean_bidask_bp"])
    print("Figures and metrics written to", RESULTS)
    return metrics


def live_nowcast():
    # Optional real-time leg: pull spot majors from Alpha Vantage and apply the last weights.
    # Off by default; set ALPHAVANTAGE_KEY to enable. Kept out of the reproducible pipeline.
    import requests, time
    key = os.environ.get("ALPHAVANTAGE_KEY")
    if not key:
        print("Set ALPHAVANTAGE_KEY to run the live nowcast.")
        return
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    rates = {}
    for p in pairs:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function": "CURRENCY_EXCHANGE_RATE",
                                 "from_currency": p[:3], "to_currency": p[3:], "apikey": key})
        rates[p] = float(r.json()["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        time.sleep(12)
    print("Live majors:", rates)


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_nowcast()
    else:
        main()
