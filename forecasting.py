import numpy as np
import pandas as pd
from pmdarima import auto_arima

def arima_forecast(price_series: pd.Series, steps: int = 30) -> pd.Series:
    model = auto_arima(price_series, seasonal=False, suppress_warnings=True)
    fc = model.predict(n_periods=steps)
    idx = pd.date_range(price_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.Series(fc, index=idx, name="ARIMA")

def prophet_forecast(price_series: pd.Series, steps: int = 30) -> pd.Series:
    from prophet import Prophet
    df = price_series.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq="D")
    f = m.predict(future).set_index("ds")["yhat"].iloc[-steps:]
    f.name = "Prophet"
    return f

def lstm_forecast(price_series: pd.Series, steps: int = 30, lookback: int = 60, epochs: int = 15) -> pd.Series:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    values = price_series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_window = scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(steps):
        p = model.predict(last_window, verbose=0)[0, 0]
        preds.append(p)
        last_window = np.concatenate([last_window[:, 1:, :], p.reshape(1, 1, 1)], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    idx = pd.date_range(price_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.Series(preds, index=idx, name="LSTM")


# --- CSV EXPORTS FOR TABLEAU 
if __name__ == "__main__":
    import argparse
    from config import DATA_DIR, TICKERS
    import pandas as pd

    prices = pd.read_csv(DATA_DIR / "prices_clean.csv", index_col=0, parse_dates=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--models", nargs="+", default=["ARIMA", "Prophet", "LSTM"])
    args = parser.parse_args()

    rows = []
    for asset in TICKERS:
        s = prices[asset].dropna()

        # history
        hist = s.reset_index()
        hist.columns = ["ds", "value"]  # regardless of original index name
        hist["asset"] = asset
        hist["model"] = "HISTORY"
        hist["type"] = "history"
        rows.append(hist[["ds", "asset", "model", "type", "value"]])

        # forecasts
        for m in args.models:
            try:
                if m == "ARIMA":
                    fc = arima_forecast(s, steps=args.steps)
                elif m == "Prophet":
                    fc = prophet_forecast(s, steps=args.steps)
                elif m == "LSTM":
                    fc = lstm_forecast(s, steps=args.steps, lookback=60, epochs=15)
                df = fc.reset_index().rename(columns={"index": "ds", 0: "value"})
                df.columns = ["ds", "value"]
                df["asset"] = asset
                df["model"] = m
                df["type"] = "forecast"
                rows.append(df[["ds", "asset", "model", "type", "value"]])
            except Exception as e:
                print(f"[WARN] {asset} {m}: {e}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA_DIR / "forecast_all.csv", index=False)
    print("[OK] Saved: forecast_all.csv")
