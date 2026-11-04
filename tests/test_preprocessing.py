"""Check time-series transformations at country and missing-data boundaries."""

import unittest

import numpy as np
import pandas as pd

from covidml.preprocessing import CovidDataPreprocessor, TargetGenerator, TimeSeriesImputer


class PreprocessingTests(unittest.TestCase):
    def test_cohort_selection_and_policy_features(self):
        frame = pd.DataFrame(
            {
                "iso_code": ["A", "A", "A", "B", "B", "B"],
                "time_step": [1, 2, 3, 1, 2, 3],
                "new_tests_smoothed_per_thousand": [1, 1, 1, 1, np.nan, np.nan],
                "Full lockdown": [0, 1, 0, 0, 0, 1],
            }
        )
        transformer = CovidDataPreprocessor()
        result = transformer.fit_transform(frame)
        self.assertEqual(transformer.final_cohort_, ["A"])
        self.assertEqual(result["iso_code"].tolist(), ["A"] * 3)
        self.assertEqual(result["days_since_last_harsh_measure"].tolist(), [999, 0, 1])
        self.assertEqual(result["active_harsh_measures_90d"].tolist(), [0, 1, 1])
        self.assertNotIn("harsh_measure_today", frame.columns)

    def test_targets_do_not_cross_country_boundaries(self):
        frame = pd.DataFrame(
            {
                "iso_code": ["A", "A", "A", "B", "B", "B"],
                "new_cases_smoothed_per_million": [100, 125, 100, 50, 25, 50],
            },
            index=[10, 11, 12, 20, 21, 22],
        )
        result = TargetGenerator(target_shift=1).fit_transform(frame)
        self.assertEqual(result.index.tolist(), [10, 11, 20, 21])
        self.assertEqual(result["target_surge"].tolist(), [1, 0, 0, 1])
        self.assertNotIn("cases_14d_future", result)

    def test_imputer_does_not_fill_from_another_country(self):
        frame = pd.DataFrame(
            {"iso_code": ["A", "A", "B", "B"], "value": [2, np.nan, np.nan, 3]}
        )
        result = TimeSeriesImputer(features=["value"]).fit_transform(frame)
        self.assertEqual(result["value"].tolist(), [2, 2, 0, 3])
        self.assertTrue(pd.isna(frame.loc[1, "value"]))
