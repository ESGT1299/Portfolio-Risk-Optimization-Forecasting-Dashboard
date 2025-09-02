# monte_carlo.py  (fixed for Python < 3.10)
import numpy as np
import pandas as pd
from typing import Optional  # <-- add this

def simulate_correlated_paths(
    last_prices: pd.Series,
    returns_df: pd.DataFrame,
    steps: int = 60,
    n_sims: int = 1000,
    seed: Optional[int] = 42  # <-- use Optional[int]
) -> dict:
    """
    Historical mean/cov Monte Carlo for multiple assets with Cholesky correlation.
    Returns dict with 'paths' (dict of ticker->array[horizon, sims]).
    """
    if seed is not None:
        np.random.seed(seed)

    assets = list(last_prices.index)
    mu = returns_df[assets].mean().values      # daily mean (decimal)
    cov = returns_df[assets].cov().values      # daily covariance

    # Numerical stabilizer on diagonal in case cov is near-singular
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(assets)))

    # initialize paths
    paths = {a: np.zeros((steps, n_sims)) for a in assets}
    # first step from last price with one random draw
    for i, a in enumerate(assets):
        paths[a][0, :] = last_prices[a] * (1 + mu[i] + np.sqrt(cov[i, i]) * np.random.randn(n_sims))

    for t in range(1, steps):
        # correlated returns via Cholesky
        z = np.random.randn(len(assets), n_sims)
        corr_draws = L @ z  # shape (n_assets, n_sims)
        for i, a in enumerate(assets):
            r = mu[i] + corr_draws[i, :]
            paths[a][t, :] = paths[a][t-1, :] * (1 + r)

    return {"paths": paths}

# --- CSV EXPORT FOR TABLEAU (append at end of file) ---
if __name__ == "__main__":
    import pandas as pd
    from config import DATA_DIR

    # Inputs: last prices & daily returns (already created by preprocessing.py)
    prices = pd.read_csv(DATA_DIR / "prices_clean.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(DATA_DIR / "pct_returns.csv", index_col=0, parse_dates=True)
    last_prices = prices.iloc[-1]

    sim = simulate_correlated_paths(last_prices, returns, steps=60, n_sims=5000, seed=42)
    # Build equal-weight portfolio paths
    assets = list(last_prices.index)
    import numpy as np
    w = np.ones(len(assets)) / len(assets)
    asset_paths = []
    for i, a in enumerate(assets):
        p = sim["paths"][a]         # shape (steps, sims)
        p_norm = p / p[0, :]
        asset_paths.append(w[i] * p_norm)
    portfolio_paths = np.sum(asset_paths, axis=0)  # (steps, sims)

    pctls = [5, 25, 50, 75, 95]
    bands = np.percentile(portfolio_paths, pctls, axis=1)  # (len(pctls), steps)

    out = pd.DataFrame({
        "day": np.arange(portfolio_paths.shape[0]),
        **{f"p{p:02d}": bands[i] for i, p in enumerate(pctls)}
    })
    out.to_csv(DATA_DIR / "mc_percentiles.csv", index=False)
    print("[OK] Saved: mc_percentiles.csv")
