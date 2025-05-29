import numpy as np
import pandas as pd
import pytest

def map_and_aggregate_to_resampled_index(event_series, resampled_index):
    """
    Map event dates (peaks/troughs) to the closest resampled index and aggregate if needed.
    Returns a Series with unique index and no NaNs.
    """
    event_map = {}
    for date, price in event_series.items():
        closest_date = resampled_index[np.abs(resampled_index - date).argmin()]
        if closest_date in event_map:
            event_map[closest_date].append(price)
        else:
            event_map[closest_date] = [price]
    # Aggregate (mean)
    result = pd.Series({d: np.mean(prices) for d, prices in event_map.items()})
    return result.dropna()

def test_no_duplicate_dates():
    # Simulate a resampled index (e.g., weekly)
    resampled_index = pd.date_range('2023-01-01', periods=10, freq='W')
    # Simulate peaks/troughs that map to the same resampled date
    event_dates = [resampled_index[2] - pd.Timedelta(days=1), resampled_index[2] + pd.Timedelta(days=1)]
    event_prices = [100, 110]
    event_series = pd.Series(event_prices, index=event_dates)
    mapped = map_and_aggregate_to_resampled_index(event_series, resampled_index)
    # Should have only one entry for the duplicate week
    assert len(mapped) == 1
    assert mapped.index[0] == resampled_index[2]
    assert mapped.iloc[0] == 105  # mean of 100 and 110

def test_no_nan_and_matching_lengths():
    resampled_index = pd.date_range('2023-01-01', periods=5, freq='W')
    # Some event dates are far from resampled index, some are close
    event_dates = [resampled_index[0], resampled_index[1] + pd.Timedelta(days=2), resampled_index[3]]
    event_prices = [50, 60, 70]
    event_series = pd.Series(event_prices, index=event_dates)
    mapped = map_and_aggregate_to_resampled_index(event_series, resampled_index)
    # No NaNs, index and values same length
    assert not mapped.isnull().any()
    assert len(mapped.index) == len(mapped.values)
    assert mapped.index.is_unique

def test_empty_event_series():
    resampled_index = pd.date_range('2023-01-01', periods=5, freq='W')
    event_series = pd.Series([], dtype=float)
    mapped = map_and_aggregate_to_resampled_index(event_series, resampled_index)
    assert mapped.empty

def test_large_randomized_events():
    np.random.seed(42)
    resampled_index = pd.date_range('2023-01-01', periods=50, freq='W')
    # 200 random event dates within the range
    event_dates = pd.to_datetime(np.random.choice(resampled_index, size=200, replace=True)) + pd.to_timedelta(np.random.randint(-3, 4, size=200), unit='D')
    event_prices = np.random.uniform(50, 150, size=200)
    event_series = pd.Series(event_prices, index=event_dates)
    mapped = map_and_aggregate_to_resampled_index(event_series, resampled_index)
    # All indices should be in resampled_index, no NaNs, unique
    assert set(mapped.index).issubset(set(resampled_index))
    assert not mapped.isnull().any()
    assert mapped.index.is_unique
    assert len(mapped.index) <= len(resampled_index)

if __name__ == '__main__':
    pytest.main([__file__]) 