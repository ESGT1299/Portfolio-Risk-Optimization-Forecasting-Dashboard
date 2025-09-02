import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR, TICKERS
from forecasting import arima_forecast, prophet_forecast, lstm_forecast
from risk import summarize, portfolio_series
from optimization import min_variance, max_sharpe, efficient_frontier, annualize_return, annualize_vol
from garch import garch_vol_forecast, garch_simulate
from monte_carlo import simulate_correlated_paths

prices = pd.read_csv(DATA_DIR / "prices_clean.csv", index_col=0, parse_dates=True)
pct_returns = pd.read_csv(DATA_DIR / "pct_returns.csv", index_col=0, parse_dates=True)

# --- CSV DOWNLOADS FOR TABLEAU ---
# Sidebar Downloads (Professional Naming)
with st.sidebar.expander("📥 Download Data (CSV)", expanded=False):
    file_labels = {
        "prices.csv": "📊 Raw Prices (Uncleaned)",
        "prices_clean.csv": "📊 Cleaned Prices",
        "pct_returns.csv": "📈 Daily % Returns",
        "log_returns.csv": "📉 Log Returns",
        "risk_metrics.csv": "⚠️ Risk Metrics",
        "correlation_matrix.csv": "🔗 Correlation Matrix",
        "efficient_frontier.csv": "🎯 Efficient Frontier Points",
        "optimal_weights.csv": "📊 Optimal Portfolio Weights",
        "mc_percentiles.csv": "🎲 Monte Carlo Percentiles",
        "forecast_all.csv": "🔮 Forecasts (ARIMA / Prophet / LSTM)",
        "garch_vol_forecasts.csv": "📉 GARCH Volatility Forecasts",
        "realized_vol_21d.csv": "📈 21-Day Realized Volatility",
    }

    cols = st.columns(2)  # 2-column layout for compact view
    for i, (fname, label) in enumerate(file_labels.items()):
        fpath = DATA_DIR / fname
        if fpath.exists():
            with cols[i % 2]:
                st.download_button(
                    label=label,
                    data=fpath.read_bytes(),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                )


st.title("Advanced Portfolio Forecasting, Risk & Optimization")

# Sidebar: asset selection + weights
st.sidebar.header("Assets & Weights")
cols = st.sidebar.columns(2)
selected = []
weights = []
for i, t in enumerate(TICKERS):
    use = cols[i % 2].checkbox(t, value=True, key=f"use_{t}")
    w = cols[i % 2].number_input(f"{t} weight", min_value=0.0, max_value=1.0, value=1.0/len(TICKERS), step=0.01, key=f"w_{t}")
    if use:
        selected.append(t)
        weights.append(w)

if len(selected) == 0:
    st.warning("Select at least one asset in the sidebar.")
    st.stop()

weights = np.array(weights)
weights = weights / weights.sum()

tabs = st.tabs(["Forecast", "Risk", "Optimization", "Monte Carlo"])

# -------------------- Forecast Tab --------------------
with tabs[0]:
    st.subheader("Single-Asset Forecast")
    asset = st.selectbox("Asset for forecast", selected)
    model = st.selectbox("Model", ["ARIMA", "Prophet", "LSTM"])
    horizon = st.slider("Forecast horizon (days)", 7, 120, 30)

    series = prices[asset].dropna()
    if model == "ARIMA":
        fc = arima_forecast(series, steps=horizon)
    elif model == "Prophet":
        try:
            fc = prophet_forecast(series, steps=horizon)
        except Exception as e:
            st.error(f"Prophet error: {e}")
            st.stop()
    else:
        fc = lstm_forecast(series, steps=horizon, lookback=60, epochs=15)

    hist = series.copy()
    all_series = pd.concat([hist, fc])
    st.line_chart(all_series)

    st.caption("This forecast is univariate per asset. Use the Monte Carlo tab for multi-asset, correlated scenarios.")

# -------------------- Risk Tab --------------------
with tabs[1]:
    st.subheader("Portfolio Risk Metrics")
    sel_returns = pct_returns[selected].dropna()
    port = portfolio_series(sel_returns, weights)

    var_level = st.slider("VaR level (%)", 90, 99, 95) / 100.0
    metrics = summarize(port, var_level)

    pretty = {
        "mean_daily": f"{metrics['mean_daily']:.4%}",
        "vol_daily": f"{metrics['vol_daily']:.4%}",
        "sharpe": f"{metrics['sharpe']:.2f}",
        "sortino": f"{metrics['sortino']:.2f}" if metrics['sortino'] == metrics['sortino'] else "NaN",
        "max_drawdown": f"{metrics['max_drawdown']:.2%}",
        "VaR": f"{metrics['VaR']:.2%}",
        "CVaR": f"{metrics['CVaR']:.2%}"
    }
    st.write(pretty)

    st.markdown("**Histogram of portfolio daily returns**")
    fig, ax = plt.subplots()
    ax.hist(port, bins=50, alpha=0.7)
    ax.set_title("Portfolio Return Distribution")
    st.pyplot(fig)

    st.markdown("**GARCH(1,1) Volatility Forecast (Single Asset)**")
    asset_g = st.selectbox("Asset for GARCH", selected, key="garch_asset")
    horizon_g = st.slider("GARCH horizon (days)", 5, 90, 30, key="garch_h")
    g_df = garch_vol_forecast(pct_returns[asset_g].dropna(), horizon=horizon_g)
    st.line_chart(g_df["sigma"])
    st.caption("Estimated conditional daily volatility (sigma) from a GARCH(1,1) model.")

# -------------------- Optimization Tab --------------------
with tabs[2]:
    st.subheader("Efficient Frontier & Optimal Portfolios")

    R = pct_returns[selected].dropna()
    mean_vec = R.mean().values            # daily mean
    cov = R.cov().values                  # daily cov

    ef = efficient_frontier(mean_vec, cov, n_points=40)
    minw = min_variance(mean_vec, cov)
    maxw = max_sharpe(mean_vec, cov, rf=0.0)

    # Plot EF (annualized)
    ef_ret_annual = [annualize_return(r) for r in ef["ret"]]
    ef_vol_annual = [annualize_vol(v) for v in ef["vol"]]

    fig, ax = plt.subplots()
    ax.plot(ef_vol_annual, ef_ret_annual, "-o", markersize=3, alpha=0.7)
    # min-var & max-sharpe points
    r_min, v_min, _ = (np.dot(minw, mean_vec), np.sqrt(minw @ cov @ minw), 0)
    r_max, v_max, _ = (np.dot(maxw, mean_vec), np.sqrt(maxw @ cov @ maxw), 0)
    ax.scatter(annualize_vol(v_min), annualize_return(r_min), c="red", s=80, label="Min Variance")
    ax.scatter(annualize_vol(v_max), annualize_return(r_max), c="green", s=80, label="Max Sharpe")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Min-Variance Weights**")
    st.write({a: float(w) for a, w in zip(selected, minw.round(4))})
    st.markdown("**Max-Sharpe Weights**")
    st.write({a: float(w) for a, w in zip(selected, maxw.round(4))})

# -------------------- Monte Carlo Tab --------------------
with tabs[3]:
    st.subheader("Multi-Asset Monte Carlo Simulation (Correlated)")

    steps = st.slider("Horizon (days)", 20, 252, 60)
    n_sims = st.slider("Simulations", 200, 5000, 1000, step=100)

    last_prices = prices[selected].dropna().iloc[-1]
    R = pct_returns[selected].dropna()

    sim = simulate_correlated_paths(last_prices, R, steps=steps, n_sims=n_sims)
    # Build portfolio path using the chosen weights
    # compute portfolio value starting from $1 scaled by weights across assets
    # Use equally scaled initial $ so that total initial portfolio value is 1
    init_val = 1.0
    w = weights

    # Approximate portfolio paths by summing weighted asset prices normalized by start
    # Normalize each asset path by its initial price to get index-like growth
    import numpy as np
    asset_paths = []
    for i, a in enumerate(selected):
        p = sim["paths"][a]   # shape (steps, sims)
        p_norm = p / p[0, :]
        asset_paths.append(w[i] * p_norm)
    portfolio_paths = np.sum(asset_paths, axis=0) * init_val  # shape (steps, sims)

    # Fan chart (percentiles)
    pctls = [1, 5, 25, 50, 75, 95, 99]
    bands = np.percentile(portfolio_paths, pctls, axis=1)  # shape (len(pctls), steps)

    fig, ax = plt.subplots()
    x = np.arange(steps)
    ax.plot(x, bands[pctls.index(50)], label="Median")
    ax.fill_between(x, bands[pctls.index(25)], bands[pctls.index(75)], alpha=0.3, label="IQR (25–75)")
    ax.fill_between(x, bands[pctls.index(5)], bands[pctls.index(95)], alpha=0.15, label="5–95%")
    ax.set_title("Portfolio Value (Simulated) — Fan Chart")
    ax.set_xlabel("Day")
    ax.set_ylabel("Value (normalized to 1 at t=0)")
    ax.legend()
    st.pyplot(fig)

    # VaR/CVaR on terminal distribution of returns
    terminal = portfolio_paths[-1, :]  # end value
    term_ret = terminal / portfolio_paths[0, :] - 1.0
    var_level = st.slider("VaR level (%)", 90, 99, 95, key="mc_var") / 100.0
    q = np.percentile(term_ret, (1 - var_level) * 100)
    tail = term_ret[term_ret <= q]
    cvar = tail.mean() if len(tail) else np.nan
    st.write({"Terminal VaR": f"{q:.2%}", "Terminal CVaR": f"{cvar:.2%}"})
