"""Educational walk-forward evaluation, never an investment recommendation.
Input: adjusted total-return simple returns, chronological rows, fixed universe.
Fees: bps per dollar bought OR sold; initial funding charged, no exit liquidation.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def validate(returns):
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex required")
    if returns.empty or returns.shape[1] < 1 or returns.columns.has_duplicates:
        raise ValueError("nonempty unique assets required")
    if returns.index.hasnans or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("dates must be unique, ordered, and nonmissing")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("finite decimal returns strictly greater than -100% required")


def target_weights(training, strategy):
    n = training.shape[1]
    if strategy in ("equal_weight", "buy_hold"):
        return np.ones(n) / n
    if strategy == "inverse_vol":
        vol = training.std(ddof=1).to_numpy()
        # Constant synthetic assets do not get infinite leverage.
        if (vol <= 1e-12).any():
            return np.ones(n) / n
        scores = 1.0 / vol
        return scores / scores.sum()
    raise ValueError("unknown strategy")


def backtest(returns, lookback=60, rebalance=20, fee_bps=10., strategy="inverse_vol"):
    validate(returns)
    if not isinstance(lookback, int) or lookback < 2 or len(returns) <= lookback:
        raise ValueError("lookback >= 2 and at least one out-of-sample row required")
    if not isinstance(rebalance, int) or rebalance < 1:
        raise ValueError("positive integer rebalance required")
    if not np.isfinite(fee_bps) or not 0 <= fee_bps < 5000:
        raise ValueError("fee_bps must be in [0, 5000)")
    if strategy not in ("inverse_vol", "equal_weight", "buy_hold"):
        raise ValueError("unknown strategy")
    wealth, weights, rows = 1., np.zeros(returns.shape[1]), []
    for i in range(lookback, len(returns)):
        first = i == lookback
        do_trade = first or (strategy != "buy_hold" and (i-lookback) % rebalance == 0)
        turnover = 0.
        if do_trade:
            # The test-period return at i is NEVER visible to allocation.
            target = target_weights(returns.iloc[i-lookback:i], strategy)
            turnover = float(np.abs(target - weights).sum())
            weights = target
        cost_fraction = turnover * fee_bps / 10000.
        before = wealth
        fee_paid = before * cost_fraction
        asset_returns = returns.iloc[i].to_numpy(dtype=float)
        gross_return = float(weights @ asset_returns)
        wealth = before * (1-cost_fraction) * (1+gross_return)
        row = {"date": returns.index[i], "training_end": returns.index[i-1] if do_trade else pd.NaT,
               "rebalance": do_trade, "turnover": turnover, "fee_paid": fee_paid,
               "gross_return": gross_return, "net_return": wealth/before-1, "wealth": wealth}
        row.update({"weight_"+str(c): float(w) for c,w in zip(returns.columns, weights)})
        rows.append(row)
        # Holdings drift between trades, rather than secretly rebalancing daily.
        weights = weights * (1+asset_returns) / (1+gross_return)
    return pd.DataFrame(rows).set_index("date")


def synthetic_returns(seed=42, observations=260):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, .006, (observations, 1))
    values = common + rng.normal(0, [.005,.012,.02], (observations, 3))
    return pd.DataFrame(values, index=pd.bdate_range("2024-01-01", periods=observations),
                        columns=["SYNTH_A", "SYNTH_B", "SYNTH_C"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Do not overwrite an earlier evidence run accidentally.
    args.output.mkdir(parents=True, exist_ok=False)
    returns = synthetic_returns()
    returns.to_csv(args.output / "synthetic_returns.csv")
    for strategy in ("inverse_vol", "equal_weight", "buy_hold"):
        backtest(returns, strategy=strategy).to_csv(args.output / (strategy + ".csv"))
    (args.output / "NOTICE.txt").write_text(
        "SYNTHETIC DEMO ONLY. Not historical market performance or an investment claim. "
        "Seed 42; 60-row lookback; 20-row rebalance; 10 bps per traded dollar. "
        "No terminal liquidation, taxes, spread, impact or financing costs modeled.\n")
    print("Wrote labeled synthetic evidence for three strategies.")

if __name__ == "__main__": main()
