import numpy as np
import pandas as pd
from typing import Dict, Any

def analyze_diagonal_wave_alternation(df: pd.DataFrame, wave_points: np.ndarray, column: str = 'close') -> Dict[str, Any]:
    """
    Analyze wave alternation characteristics in diagonals.
    """
    try:
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]
        wave_durations = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        wave_magnitudes = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        if len(wave_durations) >= 4 and len(wave_magnitudes) >= 4:
            wave_2_duration = wave_durations[1]
            wave_4_duration = wave_durations[3]
            wave_2_magnitude = wave_magnitudes[1]
            wave_4_magnitude = wave_magnitudes[3]
            duration_alternation = abs(wave_2_duration - wave_4_duration) > min(wave_2_duration, wave_4_duration) * 0.3
            magnitude_alternation = abs(wave_2_magnitude - wave_4_magnitude) > min(wave_2_magnitude, wave_4_magnitude) * 0.2
            wave_alternation = duration_alternation or magnitude_alternation
        else:
            wave_alternation = True
        return {
            "wave_alternation": wave_alternation,
            "wave_durations": wave_durations,
            "wave_magnitudes": wave_magnitudes
        }
    except Exception as e:
        return {"wave_alternation": False, "reason": f"alternation_analysis_error_{str(e)}"}

def validate_alternation_principle(df: pd.DataFrame, wave_points: np.ndarray, column: str = 'close') -> float:
    """
    Validate the alternation principle between waves 2 and 4.
    If Wave 2 is sharp and short, Wave 4 should be flat and long (and vice versa).
    """
    if len(wave_points) < 5:
        return 0.5
    try:
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]
        wave_2_magnitude = abs(prices[2] - prices[1])
        wave_4_magnitude = abs(prices[4] - prices[3])
        wave_2_duration = (dates[2] - dates[1]).days
        wave_4_duration = (dates[4] - dates[3]).days
        wave_1_magnitude = abs(prices[1] - prices[0])
        wave_3_magnitude = abs(prices[3] - prices[2])
        wave_2_retracement = wave_2_magnitude / wave_1_magnitude if wave_1_magnitude != 0 else 0
        wave_4_retracement = wave_4_magnitude / wave_3_magnitude if wave_3_magnitude != 0 else 0
        confidence = 0.0
        deep_shallow_alternation = abs(wave_2_retracement - wave_4_retracement) > 0.2
        if deep_shallow_alternation:
            confidence += 0.3
        if wave_2_duration > 0 and wave_4_duration > 0:
            duration_ratio = max(wave_2_duration, wave_4_duration) / min(wave_2_duration, wave_4_duration)
            if duration_ratio > 1.5:
                confidence += 0.3
        wave_2_steepness = wave_2_magnitude / max(wave_2_duration, 1)
        wave_4_steepness = wave_4_magnitude / max(wave_4_duration, 1)
        if wave_2_steepness > 0 and wave_4_steepness > 0:
            steepness_ratio = max(wave_2_steepness, wave_4_steepness) / min(wave_2_steepness, wave_4_steepness)
            if steepness_ratio > 2.0:
                confidence += 0.4
        return min(confidence, 1.0)
    except Exception:
        return 0.5 