"""Reusable feature engineering, target generation, and imputation."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CovidDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_shift=14, surge_threshold=1.25, completeness_threshold=0.80):
        self.target_shift = target_shift
        self.surge_threshold = surge_threshold
        self.completeness_threshold = completeness_threshold
        self.final_cohort_ = []
        self.picked_measures_ = [
            "Partial lockdown", "Full lockdown", "Isolation and quarantine policies",
            "State of emergency declared", "Curfews"
        ]
        self.features_ = [
            "new_cases_smoothed_per_million", "total_cases_per_million",
            "new_deaths_smoothed_per_million", "total_deaths_per_million",
            "new_tests_smoothed_per_thousand", "total_tests_per_thousand", "positive_rate",
            "reproduction_rate", "stringency_index", "population_density", "aged_65_older",
            "gdp_per_capita", "extreme_poverty", "life_expectancy", "cardiovasc_death_rate",
            "diabetes_prevalence", "hospital_beds_per_thousand",
            "days_since_last_harsh_measure", "active_harsh_measures_90d"
        ]

    def fit(self, X, y=None):
        testing_completeness = (
            X.groupby("iso_code")["new_tests_smoothed_per_thousand"]
            .apply(lambda x: x.notna().mean())
            .reset_index(name="testing_reporting_rate")
        )
        self.final_cohort_ = testing_completeness[
            testing_completeness["testing_reporting_rate"] >= self.completeness_threshold
        ]["iso_code"].tolist()
        return self

    def transform(self, X, y=None):
        df = X[X['iso_code'].isin(self.final_cohort_)].sort_values(["iso_code", "time_step"]).copy()

        for col in self.picked_measures_:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

        # Feature Engineering: Temporal policy variables
        df["harsh_measure_today"] = df[self.picked_measures_].max(axis=1)
        df["last_trigger_step"] = df["time_step"].where(df["harsh_measure_today"] == 1)
        df["last_trigger_step"] = df.groupby("iso_code")["last_trigger_step"].ffill()
        df["days_since_last_harsh_measure"] = (df["time_step"] - df["last_trigger_step"]).fillna(999)

        df["active_harsh_measures_90d"] = df.groupby('iso_code')['harsh_measure_today'].transform(
            lambda x: x.rolling(window=90, min_periods=1).sum()
        )
        return df


class TargetGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, target_shift=14, surge_threshold=1.25):
        self.target_shift = target_shift
        self.surge_threshold = surge_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df["cases_14d_future"] = df.groupby("iso_code")["new_cases_smoothed_per_million"].shift(-self.target_shift)
        df["target_surge"] = np.where(
            df["cases_14d_future"] >= (df["new_cases_smoothed_per_million"] * self.surge_threshold), 1, 0
        )

        df = df.dropna(subset=['cases_14d_future', 'new_cases_smoothed_per_million']).drop(columns=['cases_14d_future'])
        return df


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        impute_cols = [col for col in self.features if col in df.columns]
        df[impute_cols] = df.groupby('iso_code')[impute_cols].ffill().fillna(0)
        return df
