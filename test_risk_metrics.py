"""Analytical regression cases; synthetic returns, not investment results."""
import unittest
import numpy as np
import pandas as pd
from risk import max_drawdown, sortino_ratio

class RiskMetricTests(unittest.TestCase):
    def test_initial_loss_is_not_hidden(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-.1, 0])), -.1)
    def test_monotonic_gains(self):
        self.assertEqual(max_drawdown(pd.Series([.1, .2])), 0)
    def test_recovery_then_new_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([-.1, 1/9, -.2])), -.2)
    def test_complete_loss(self):
        self.assertEqual(max_drawdown(pd.Series([-1., 0.])), -1)
    def test_known_downside_deviation(self):
        # excess mean=.005; target downside deviation=sqrt(.0001/2)
        expected=np.sqrt(252)*.005/np.sqrt(.0001/2)
        self.assertAlmostEqual(sortino_ratio(pd.Series([.02, -.01])), expected)
    def test_constant_losses_have_defined_sortino(self):
        self.assertAlmostEqual(sortino_ratio(pd.Series([-.01, -.01])), -np.sqrt(252))
    def test_no_downside_is_undefined(self):
        self.assertTrue(np.isnan(sortino_ratio(pd.Series([.01, .02]))))
    def test_invalid_input(self):
        for values in ([], [np.nan], [np.inf], [-1.1]):
            for function in (max_drawdown, sortino_ratio):
                with self.subTest(values=values, function=function):
                    with self.assertRaises(ValueError): function(pd.Series(values))
    def test_invalid_periods(self):
        with self.assertRaises(ValueError): sortino_ratio(pd.Series([.01]), periods=0)

if __name__ == '__main__': unittest.main()
