import unittest
import numpy as np
import pandas as pd
from walk_forward import backtest, synthetic_returns

class WalkForwardTests(unittest.TestCase):
    def setUp(self): self.data = synthetic_returns(observations=100)
    def test_no_lookahead(self):
        original=backtest(self.data, lookback=20)
        changed=self.data.copy(); changed.iloc[20:] += .1
        other=backtest(changed, lookback=20)
        np.testing.assert_allclose(original.filter(like="weight_").iloc[0], other.filter(like="weight_").iloc[0])
        changed=self.data.copy(); changed.iloc[80:] -= .1
        pd.testing.assert_frame_equal(original.iloc[:60], backtest(changed, lookback=20).iloc[:60])
    def test_training_precedes_trade(self):
        out=backtest(self.data, lookback=20)
        trades=out[out.rebalance]
        self.assertTrue((trades.training_end < trades.index).all())
    def test_initial_fee(self):
        data=self.data*0
        out=backtest(data, lookback=20, fee_bps=10)
        self.assertAlmostEqual(out.iloc[0].wealth, .999)
        self.assertAlmostEqual(out.fee_paid.sum(), .001)
    def test_known_buy_hold(self):
        data=pd.DataFrame([[0,0],[0,0],[.1,0],[0,.2]], index=pd.date_range("2024-01-01",periods=4))
        out=backtest(data, lookback=2, fee_bps=0, strategy="buy_hold")
        self.assertAlmostEqual(out.iloc[-1].wealth, 1.15)
        self.assertAlmostEqual(out.iloc[1].weight_0, .55/1.05)
        self.assertEqual(out.rebalance.sum(), 1)
    def test_two_sided_turnover(self):
        data=pd.DataFrame([[0,0],[0,0],[1.,0],[0,0]], index=pd.date_range("2024-01-01",periods=4))
        out=backtest(data, lookback=2, rebalance=1, fee_bps=0, strategy="equal_weight")
        self.assertAlmostEqual(out.iloc[1].turnover, 1/3)
    def test_fees_reduce_wealth(self):
        a=backtest(self.data,lookback=20,fee_bps=0)
        b=backtest(self.data,lookback=20,fee_bps=10)
        self.assertTrue((b.wealth <= a.wealth).all())
    def test_weights_valid(self):
        out=backtest(self.data,lookback=20)
        weights=out.filter(like="weight_")
        np.testing.assert_allclose(weights.sum(axis=1),1)
        self.assertTrue((weights>=0).all().all())
    def test_reproducible(self):
        pd.testing.assert_frame_equal(synthetic_returns(), synthetic_returns())
    def test_invalid_returns(self):
        for value in (np.nan, np.inf, -1, -2):
            bad=self.data.copy(); bad.iloc[0,0]=value
            with self.assertRaises(ValueError): backtest(bad)
    def test_bad_dates(self):
        with self.assertRaises(ValueError): backtest(self.data.iloc[::-1])
        bad=self.data.copy(); bad.index=[bad.index[0]]*len(bad)
        with self.assertRaises(ValueError): backtest(bad)
    def test_bad_configuration(self):
        for kwargs in ({"fee_bps":-1},{"lookback":1},{"rebalance":0},{"strategy":"magic"}):
            with self.assertRaises(ValueError): backtest(self.data,**kwargs)

if __name__ == "__main__": unittest.main()
