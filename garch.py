# garch.py (fixed for Python < 3.10)
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional  # <-- add this

def garch_vol_forecast(returns_series: pd.Series, horizon: int = 30, dist: str = "normal") -> pd.DataFrame:
    """
    Fit GARCH(1,1) to daily returns (decimal), forecast conditional volatility for 'horizon' days.
    Returns a DataFrame with columns: 'sigma' (daily vol forecast), 'mu' (mean return).
    """
    r_pct = returns_series.dropna() * 100.0
    am = arch_model(r_pct, vol="GARCH", p=1, q=1, mean="constant", dist=dist)
    res = am.fit(disp="off")
    f = res.forecast(horizon=horizon, reindex=False)
    sigma = np.sqrt(f.variance.values[-1]) / 100.0   # back to decimal
    mu = np.repeat((res.params.get("mu", 0.0) / 100.0), horizon)
    idx = pd.date_range(returns_series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"mu": mu, "sigma": sigma}, index=idx)

def garch_simulate(spot: float, mu: np.ndarray, sigma: np.ndarray, n_sims: int = 1000, seed: Optional[int] = 42) -> np.ndarray:
    """
    Simulate price paths using daily mu and sigma arrays (decimal).
    Geometric-ish update: P_{t+1} = P_t * (1 + mu_t + sigma_t * eps), eps ~ N(0,1).
    Returns array of shape (horizon, n_sims).
    """
    if seed is not None:
        np.random.seed(seed)
    horizon = len(mu)
    paths = np.zeros((horizon, n_sims))
    paths[0, :] = spot * (1 + mu[0] + sigma[0] * np.random.randn(n_sims))
    for t in range(1, horizon):
        eps = np.random.randn(n_sims)
        paths[t, :] = paths[t-1, :] * (1 + mu[t] + sigma[t] * eps)
    return paths

# --- CSV EXPORTS FOR TABLEAU
if __name__ == "__main__":
    import argparse
    from config import DATA_DIR, TICKERS
    import pandas as pd

    returns = pd.read_csv(DATA_DIR / "pct_returns.csv", index_col=0, parse_dates=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--dist", type=str, default="normal")
    args = parser.parse_args()

    # Forward GARCH vol forecasts (tall)
    rows = []
    for asset in TICKERS:
        try:
            r = returns[asset].dropna()
            f = garch_vol_forecast(r, horizon=args.horizon, dist=args.dist)
            f = f.reset_index().rename(columns={"index": "ds"})
            f["asset"] = asset
            rows.append(f[["ds", "asset", "mu", "sigma"]])
        except Exception as e:
            print(f"[WARN] {asset}: {e}")
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(DATA_DIR / "garch_vol_forecasts.csv", index=False)
        print("[OK] Saved: garch_vol_forecasts.csv")

    # Realized 21-day daily vol (for overlay)
    rv = []
    for asset in TICKERS:
        if asset in returns.columns:
            s = returns[asset].dropna()
            realized = s.rolling(21).std().reset_index()
            realized.columns = ["ds", "sigma_realized"]
            realized["asset"] = asset
            rv.append(realized[["ds", "asset", "sigma_realized"]])
    if rv:
        pd.concat(rv, ignore_index=True).to_csv(DATA_DIR / "realized_vol_21d.csv", index=False)
        print("[OK] Saved: realized_vol_21d.csv")
