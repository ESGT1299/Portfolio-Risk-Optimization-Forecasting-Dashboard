# Walk-forward evaluation lab (synthetic demo)

This addition is an independent portfolio engineering exercise, not a strategy
recommendation, employment claim or claim of profitable historical performance.
It does not download prices, place trades, or connect financial accounts.

## Reproduce

Install NumPy and pandas, then run:

```powershell
python -m unittest test_walk_forward -v
python walk_forward.py --output synthetic_evidence_run1
```

The new output directory contains labeled synthetic returns, three daily audit
traces and a notice. All strategies use exactly the same out-of-sample dates.
A fixed seed makes the illustration repeatable, not representative of markets.

## Method

- 60 prior observations per training window; allocation sees no current or future return.
- Long-only inverse-volatility allocation versus equal-weight rebalanced and equal-weight buy-and-hold baselines.
- Rebalance every 20 observations; weights drift with holdings between trades.
- Costs of 10 basis points per dollar bought **or** sold: L1 weight change, including initial purchases from cash.
- Fees reduce wealth before that period's return. This is a proportional turnover approximation, not an exact broker execution model.
- Audit traces include training end, trade indicator, opening weights, turnover, fee paid, gross/net return and wealth.
- Zero-volatility windows fall back to equal weights. Invalid data fails rather than being silently filled.

## Limits and next validation

No terminal liquidation charge, tax, bid/ask spread, market impact, borrowing,
currency conversion or liquidity constraint. The fixed universe excludes
survivorship/delisting effects. Periods are observation counts, not exchange
calendar months. Same-close availability of prior observations is assumed;
real deployment must allow data/processing latency. Missing data is rejected.
For real research, supply licensed, point-in-time adjusted total-return data,
freeze the universe and evaluation protocol, document execution timing and
compare robustness across windows and costs before making any conclusions.
These results must never be marketed as realized or expected investment returns.
