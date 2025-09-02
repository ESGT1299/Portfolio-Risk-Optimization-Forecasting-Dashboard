import numpy as np
import pandas as pd
import scipy.optimize as sco

def annualize_return(mean_daily: float, periods: int = 252) -> float:
    return (1 + mean_daily) ** periods - 1

def annualize_vol(daily_vol: float, periods: int = 252) -> float:
    return daily_vol * np.sqrt(periods)

def portfolio_stats(mean_vec: np.ndarray, cov: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    ret = np.dot(w, mean_vec)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sharpe = ret / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def min_variance(mean_vec: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mean_vec)
    def vol(w): return portfolio_stats(mean_vec, cov, w)[1]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bnds = tuple((0, 1) for _ in range(n))
    w0 = np.ones(n) / n
    res = sco.minimize(vol, w0, bounds=bnds, constraints=cons)
    return res.x

def max_sharpe(mean_vec: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    n = len(mean_vec)
    def neg_sharpe(w):
        r, v, _ = portfolio_stats(mean_vec - rf/252, cov, w)
        return - (r / v) if v > 0 else 1e9
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bnds = tuple((0, 1) for _ in range(n))
    w0 = np.ones(n) / n
    res = sco.minimize(neg_sharpe, w0, bounds=bnds, constraints=cons)
    return res.x

def efficient_frontier(mean_vec: np.ndarray, cov: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    n = len(mean_vec)
    w0 = np.ones(n) / n
    ret_min, _, _ = portfolio_stats(mean_vec, cov, min_variance(mean_vec, cov))
    ret_max = float(mean_vec.max()) * 1.2  # a bit above max asset mean

    targets = np.linspace(ret_min, ret_max, n_points)
    ef = []
    bnds = tuple((0, 1) for _ in range(n))
    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_vec) - t},
        ]
        def vol(w): return portfolio_stats(mean_vec, cov, w)[1]
        res = sco.minimize(vol, w0, bounds=bnds, constraints=cons)
        if res.success:
            r, v, s = portfolio_stats(mean_vec, cov, res.x)
            ef.append((target, v, s, res.x))
        else:
            ef.append((np.nan, np.nan, np.nan, np.full(n, np.nan)))
    df = pd.DataFrame(ef, columns=["ret", "vol", "sharpe", "weights"])
    return df

# --- CSV EXPORTS FOR TABLEAU (append at end of file) ---
if __name__ == "__main__":
    import pandas as pd
    from config import DATA_DIR

    # Load daily % returns
    R = pd.read_csv(DATA_DIR / "pct_returns.csv", index_col=0, parse_dates=True)
    mean_vec = R.mean().values      # daily means
    cov = R.cov().values            # daily cov

    # Efficient frontier sample
    ef = efficient_frontier(mean_vec, cov, n_points=60)
    ef.to_csv(DATA_DIR / "efficient_frontier.csv", index=False)

    # Optimal portfolios
    w_min = min_variance(mean_vec, cov)
    w_max = max_sharpe(mean_vec, cov, rf=0.0)

    tickers = list(R.columns)
    opt = pd.DataFrame([
        {"portfolio": "min_variance", **{t: w for t, w in zip(tickers, w_min)}},
        {"portfolio": "max_sharpe",   **{t: w for t, w in zip(tickers, w_max)}},
    ])
    opt.to_csv(DATA_DIR / "optimal_weights.csv", index=False)
    print("[OK] Saved: efficient_frontier.csv, optimal_weights.csv")
