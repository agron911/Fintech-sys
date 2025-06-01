import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple

def detect_peaks_troughs_enhanced(df: pd.DataFrame, column: str = 'close', 
                                window: int = 5, min_prominence_pct: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced peak and trough detection with adaptive parameters and volume validation.
    Pure function: no plotting, no I/O.
    """
    if len(df) < window * 2:
        return np.array([]), np.array([])
    price_series = df[column].fillna(method='ffill').fillna(method='bfill')
    volatility = price_series.pct_change().rolling(window=20).std().fillna(0)
    avg_volatility = volatility.mean()
    distance = max(5, int(10 * (1 + avg_volatility * 10)))
    prominence = price_series.mean() * min_prominence_pct / 100
    smoothed = price_series.rolling(window=window, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    peaks, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)
    if 'volume' in df.columns and len(df['volume'].dropna()) > 0:
        avg_volume = df['volume'].mean()
        volume_threshold = avg_volume * 0.7
        peaks = np.array([p for p in peaks if p < len(df) and df['volume'].iloc[p] >= volume_threshold])
        troughs = np.array([t for t in troughs if t < len(df) and df['volume'].iloc[t] >= volume_threshold])
    validated_peaks = []
    for peak in peaks:
        left_bound = max(0, peak - window)
        right_bound = min(len(df), peak + window + 1)
        local_max = df[column].iloc[left_bound:right_bound].max()
        if abs(df[column].iloc[peak] - local_max) < local_max * 0.001:
            validated_peaks.append(peak)
    validated_troughs = []
    for trough in troughs:
        left_bound = max(0, trough - window)
        right_bound = min(len(df), trough + window + 1)
        local_min = df[column].iloc[left_bound:right_bound].min()
        if abs(df[column].iloc[trough] - local_min) < local_min * 0.001:
            validated_troughs.append(trough)
    return np.array(validated_peaks), np.array(validated_troughs) 