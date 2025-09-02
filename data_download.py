import pandas as pd
import yfinance as yf
from config import DATA_DIR, TICKERS, START_DATE, END_DATE

DATA_DIR.mkdir(exist_ok=True)

df = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
df = df.dropna(how="all")
df.to_csv(DATA_DIR / "prices.csv")
for t in TICKERS:
    s = df[[t]].dropna()
    s.to_csv(DATA_DIR / f"{t}.csv")
print("Downloaded and saved:", list(df.columns))
