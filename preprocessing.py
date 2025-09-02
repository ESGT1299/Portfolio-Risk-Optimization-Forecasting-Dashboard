import numpy as np
import pandas as pd
from config import DATA_DIR

prices = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
prices = prices.dropna(how="all")
prices = prices.fillna(method="ffill")

log_returns = np.log(prices / prices.shift(1)).dropna()
pct_returns = prices.pct_change().dropna()

prices.to_csv(DATA_DIR / "prices_clean.csv")
log_returns.to_csv(DATA_DIR / "log_returns.csv")
pct_returns.to_csv(DATA_DIR / "pct_returns.csv")
print("Saved cleaned prices, log_returns, pct_returns.")
