import numpy as np
import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd.min()

def sharpe_ratio(series: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = series - rf/periods
    return np.sqrt(periods) * excess.mean() / excess.std(ddof=0)

def sortino_ratio(series: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = series - rf/periods
    downside = excess[excess < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return np.sqrt(periods) * excess.mean() / denom

def var_cvar(series: pd.Series, level: float = 0.95) -> tuple[float, float]:
    q = np.percentile(series, (1 - level) * 100)
    tail = series[series <= q]
    cvar = tail.mean() if len(tail) else np.nan
    return q, cvar

def portfolio_series(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.array(weights).reshape(-1, 1)
    return (returns_df @ w).squeeze()

def summarize(series: pd.Series, var_level: float = 0.95, rf: float = 0.0) -> dict:
    var95, cvar95 = var_cvar(series, var_level)
    return {
        "mean_daily": series.mean(),
        "vol_daily": series.std(ddof=0),
        "sharpe": sharpe_ratio(series, rf=rf),
        "sortino": sortino_ratio(series, rf=rf),
        "max_drawdown": max_drawdown(series),
        "VaR": var95,
        "CVaR": cvar95
    }

# --- CSV EXPORTS FOR TABLEAU (append at end of file) ---
if __name__ == "__main__":
    import pandas as pd
    from config import DATA_DIR

    # Load daily % returns produced in preprocessing.py
    R = pd.read_csv(DATA_DIR / "pct_returns.csv", index_col=0, parse_dates=True)

    # Per-asset KPIs
    rows = []
    for col in R.columns:
        s = R[col].dropna()
        rows.append({
            "asset": col,
            **summarize(s, var_level=0.95, rf=0.0)  # daily stats
        })
    risk_df = pd.DataFrame(rows).set_index("asset")
    risk_df.to_csv(DATA_DIR / "risk_metrics.csv")

    # Correlation matrix
    R.corr().to_csv(DATA_DIR / "correlation_matrix.csv")
    print("[OK] Saved: risk_metrics.csv, correlation_matrix.csv")
