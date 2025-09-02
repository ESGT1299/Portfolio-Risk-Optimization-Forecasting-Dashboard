from pathlib import Path

DATA_DIR = Path("data")

TICKERS = [
    "SPY",       # S&P 500 ETF (equity)
    "QQQ",       # Nasdaq-100 ETF (equity)
    "GLD",       # Gold ETF (commodity)
    "BND",       # Total Bond Market ETF (bonds)
    "BTC-USD"    # Bitcoin (crypto)
]

START_DATE = "2010-01-01"
END_DATE = None  # latest
