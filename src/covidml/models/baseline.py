"""Persistence baseline based on each country's recent case growth."""

import pandas as pd


def persistence_predictions(
    data: pd.DataFrame,
    lookback: int = 14,
    surge_threshold: float = 1.25,
) -> pd.Series:
    """
    Predict a surge when cases rose by the threshold over the lookback.

    Rows must be ordered by time within each country. Rows without enough
    history predict zero.
    """
    past_cases = data.groupby("iso_code")["new_cases_smoothed_per_million"].shift(
        lookback
    )
    return (
        data["new_cases_smoothed_per_million"] >= past_cases * surge_threshold
    ).astype(int)
