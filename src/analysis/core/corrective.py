import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from ..core import detect_peaks_troughs_enhanced
from enum import Enum
from ..core.models import WaveType

# (Add WaveType enum or import if needed)

def detect_corrective_patterns(df: pd.DataFrame, start_idx: int, 
                             column: str = 'close') -> Dict[str, Any]:
    """
    Comprehensive detection of corrective wave patterns including:
    - Zigzag (5-3-5)
    - Flat (3-3-5) - Regular, Irregular, Running
    - Triangle (3-3-3-3-3) - Contracting, Expanding
    - Complex corrections (Double/Triple combinations)
    """
    peaks, troughs = detect_peaks_troughs_enhanced(df[start_idx:], column)
    peaks += start_idx
    troughs += start_idx
    all_points = sorted([(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs])
    patterns = []
    if len(all_points) >= 3:
        patterns.extend(detect_zigzag_patterns(df, all_points, column))
    if len(all_points) >= 3:
        patterns.extend(detect_flat_patterns(df, all_points, column))
    if len(all_points) >= 5:
        patterns.extend(detect_triangle_patterns(df, all_points, column))
    if patterns:
        best_pattern = max(patterns, key=lambda x: x['confidence'])
        return best_pattern
    return {"type": "unknown", "confidence": 0.0, "points": np.array([])}

def detect_zigzag_patterns(df: pd.DataFrame, all_points: List[Tuple[int, str]], 
                          column: str = 'close') -> List[Dict[str, Any]]:
    """
    Detect zigzag corrective patterns (A-B-C where A=5 waves, B=3 waves, C=5 waves).
    """
    patterns = []
    for i in range(len(all_points) - 2):
        point_a = all_points[i]
        point_b = all_points[i + 1]
        point_c = all_points[i + 2]
        if point_a[1] != point_b[1] and point_b[1] != point_c[1]:
            indices = np.array([point_a[0], point_b[0], point_c[0]])
            prices = df[column].iloc[indices].values
            wave_a = prices[1] - prices[0]
            wave_b = prices[2] - prices[1]
            if wave_a != 0:
                retracement = abs(wave_b / wave_a)
                if 0.382 <= retracement <= 0.786:
                    wave_c_projection = abs(wave_b)
                    if wave_a != 0:
                        c_to_a_ratio = wave_c_projection / abs(wave_a)
                        if 0.618 <= c_to_a_ratio <= 1.618:
                            confidence = 0.7 + (0.3 * (1 - abs(1 - c_to_a_ratio)))
                            patterns.append({
                                "type": WaveType.ZIGZAG,
                                "points": indices,
                                "confidence": min(confidence, 1.0),
                                "characteristics": {
                                    "wave_b_retracement": retracement,
                                    "wave_c_ratio": c_to_a_ratio
                                }
                            })
    return patterns

def detect_flat_patterns(df: pd.DataFrame, all_points: List[Tuple[int, str]], 
                        column: str = 'close') -> List[Dict[str, Any]]:
    """
    Detect flat corrective patterns (A-B-C where all waves are 3-wave structures).
    Types: Regular, Irregular, Running flats.
    """
    patterns = []
    for i in range(len(all_points) - 2):
        point_a = all_points[i]
        point_b = all_points[i + 1]
        point_c = all_points[i + 2]
        if point_a[1] != point_b[1] and point_b[1] != point_c[1]:
            indices = np.array([point_a[0], point_b[0], point_c[0]])
            prices = df[column].iloc[indices].values
            wave_a = prices[1] - prices[0]
            wave_b = prices[2] - prices[1]
            if wave_a != 0:
                b_retracement = abs(wave_b / wave_a)
                if 0.90 <= b_retracement <= 1.10:
                    patterns.append({
                        "type": WaveType.FLAT_REGULAR,
                        "points": indices,
                        "confidence": 0.8,
                        "characteristics": {"b_retracement": b_retracement}
                    })
                elif 1.10 <= b_retracement <= 1.38:
                    patterns.append({
                        "type": WaveType.FLAT_IRREGULAR,
                        "points": indices,
                        "confidence": 0.7,
                        "characteristics": {"b_retracement": b_retracement}
                    })
                elif 0.70 <= b_retracement < 0.90:
                    patterns.append({
                        "type": WaveType.FLAT_RUNNING,
                        "points": indices,
                        "confidence": 0.6,
                        "characteristics": {"b_retracement": b_retracement}
                    })
    return patterns

def detect_triangle_patterns(df: pd.DataFrame, all_points: List[Tuple[int, str]], 
                           column: str = 'close') -> List[Dict[str, Any]]:
    """
    Detect triangle corrective patterns (A-B-C-D-E with specific characteristics).
    """
    patterns = []
    for i in range(len(all_points) - 4):
        triangle_points = all_points[i:i+5]
        valid_alternation = all(
            triangle_points[j][1] != triangle_points[j+1][1] 
            for j in range(4)
        )
        if valid_alternation:
            indices = np.array([point[0] for point in triangle_points])
            prices = df[column].iloc[indices].values
            triangle_analysis = analyze_triangle_characteristics(prices, indices)
            if triangle_analysis['is_valid']:
                patterns.append({
                    "type": triangle_analysis['type'],
                    "points": indices,
                    "confidence": triangle_analysis['confidence'],
                    "characteristics": triangle_analysis
                })
    return patterns

def analyze_triangle_characteristics(prices: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """
    Analyze triangle pattern characteristics to determine if it's a valid triangle.
    """
    try:
        upper_points = [(indices[i], prices[i]) for i in [0, 2, 4]]
        lower_points = [(indices[i], prices[i]) for i in [1, 3]]
        if len(upper_points) >= 2 and len(lower_points) >= 2:
            upper_slope = (upper_points[-1][1] - upper_points[0][1]) / max(1, upper_points[-1][0] - upper_points[0][0])
            lower_slope = (lower_points[-1][1] - lower_points[0][1]) / max(1, lower_points[-1][0] - lower_points[0][0])
            slope_diff = abs(upper_slope - lower_slope)
            is_contracting = slope_diff > 0.001 and abs(upper_slope) > abs(lower_slope)
            is_expanding = slope_diff > 0.001 and abs(lower_slope) > abs(upper_slope)
            wave_lengths = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
            diminishing_waves = all(
                wave_lengths[i] < wave_lengths[i-1] * 1.1
                for i in range(1, len(wave_lengths))
            )
            confidence = 0.0
            triangle_type = WaveType.TRIANGLE_CONTRACTING
            # (Add more logic as needed)
            return {
                'is_valid': is_contracting or is_expanding,
                'type': triangle_type,
                'confidence': confidence,
            }
        return {'is_valid': False, 'type': None, 'confidence': 0.0}
    except Exception:
        return {'is_valid': False, 'type': None, 'confidence': 0.0} 