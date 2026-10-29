"""Read-only checks of fixed ADF, matching, and raw-price feature reproduction.

No full universe selection or backtest is run, and no caches are overwritten.
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import prepare

TOLERANCE = 1e-10


class PreparationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = prepare.load_data(prepare.PRICE_FILE)

    def test_40_fixed_adf_samples(self):
        hist = self.prices.loc[:'2016-12-30'].tail(504)
        hist = hist.loc[:, hist.notna().all() & (hist > 0).all()].sort_index(axis=1)
        a = np.log(hist).to_numpy()
        rng = np.random.default_rng(240307998)
        for _ in range(40):
            i, j = rng.choice(a.shape[1], 2, replace=False)
            xx = np.column_stack([np.ones(len(a)), a[:, i]])
            beta = np.linalg.lstsq(xx, a[:, j], rcond=None)[0]
            residual = a[:, j]-xx@beta
            fast = prepare.adf_fixed1_batch(residual[None, :])[0]
            reference = adfuller(residual, maxlag=1, regression='c', autolag=None)
            self.assertLess(abs(fast-reference[0]), TOLERANCE)
            self.assertLess(abs(prepare.adf_pvalues(np.array([fast]))[0]-reference[1]), TOLERANCE)

    def test_small_matching_global_choice(self):
        # Greedy AB+CD scores11; global AC+BD scores17.
        edges = pd.DataFrame({'x': ['A', 'A', 'B', 'C'], 'y': ['B', 'C', 'D', 'D'],
                              't_stat': [-10., -9., -8., -1.], 'p_value': [.001, .002, .003, .5]})
        selected, _ = prepare.select_matching_and_baseline(edges)
        chosen = {tuple(v) for v in selected.loc[selected.is_M, ['x', 'y']].to_numpy()}
        self.assertEqual(chosen, {('A', 'C'), ('B', 'D')})
        self.assertEqual(float(-selected.loc[selected.is_M, 't_stat'].sum()), 17.)

    def test_three_raw_price_feature_months(self):
        for month in ['2017-01', '2018-01', '2023-05']:
            with self.subTest(month=month):
                selected = pd.read_parquet(prepare.SELECTION_DIR/f'{month}.parquet')
                flags = prepare.selection_flags(selected)
                calculated = prepare.calculate_features(flags, self.prices, month).merge(
                    flags, on=['x', 'y'], validate='many_to_one')
                calculated = calculated.sort_values(['date', 'x', 'y']).reset_index(drop=True)
                cached = pd.read_parquet(prepare.FEATURE_DIR/f'{month}.parquet', columns=prepare.FEATURE_COLUMNS)
                cached = cached.sort_values(['date', 'x', 'y']).reset_index(drop=True)
                self.assertEqual(len(calculated), len(cached))
                for column in ['date', 'signal_date', 'x', 'y', 'xi', 'yi', 'is_M', 'is_B', 'is_C1', 'is_C2', 'is_C4', 'is_Cinf']:
                    np.testing.assert_array_equal(calculated[column].to_numpy(), cached[column].to_numpy())
                for column in ['beta', 'q_fit', 'z_fit']:
                    np.testing.assert_allclose(calculated[column].to_numpy(), cached[column].to_numpy(),
                                               rtol=0, atol=TOLERANCE, equal_nan=True)
                self.assertTrue((cached.is_M == cached.is_C1).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
