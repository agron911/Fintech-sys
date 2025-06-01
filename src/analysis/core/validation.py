import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from .fib_utils import validate_fibonacci_relationships, FibonacciRatios
from .trendlines import analyze_diagonal_trend_lines
from .volume import analyze_diagonal_volume_pattern, validate_volume_patterns
from .alternation import analyze_diagonal_wave_alternation, validate_alternation_principle

def validate_impulse_wave_rules(df: pd.DataFrame, wave_points: np.ndarray, 
                              column: str = 'close') -> Tuple[bool, float, Dict[str, Any]]:
    """
    Comprehensive validation of Elliott Wave impulse rules with proper error handling.
    Pure function: no plotting, no I/O.
    """
    if len(wave_points) < 5:
        return False, 0.0, {"error": "insufficient_points"}
    prices = df[column].iloc[wave_points].values
    dates = df.index[wave_points]
    if not all(dates[i] < dates[i+1] for i in range(len(dates)-1)):
        return False, 0.0, {"error": "non_chronological"}
    waves = {
        1: prices[1] - prices[0],
        2: prices[2] - prices[1],
        3: prices[3] - prices[2],
        4: prices[4] - prices[3],
        5: prices[5] - prices[4] if len(prices) > 5 else 0
    }
    validation_results = {}
    confidence = 0.5
    if waves[1] != 0:
        wave_2_retracement = abs(waves[2] / waves[1])
        validation_results['wave_2_retracement'] = wave_2_retracement
        if wave_2_retracement > 1.0:
            return False, 0.0, {"error": "wave_2_100_percent_retracement", "retracement": wave_2_retracement}
        if 0.382 <= wave_2_retracement <= 0.618:
            confidence += 0.2
        elif 0.236 <= wave_2_retracement <= 0.786:
            confidence += 0.1
    wave_lengths = [abs(waves[1]), abs(waves[3])]
    if len(prices) > 5:
        wave_lengths.append(abs(waves[5]))
    validation_results['wave_lengths'] = wave_lengths
    if len(wave_lengths) >= 2 and abs(waves[3]) == min(wave_lengths) and len(wave_lengths) > 1:
        return False, 0.0, {"error": "wave_3_shortest", "lengths": wave_lengths}
    if abs(waves[3]) == max(wave_lengths):
        confidence += 0.2
    wave_1_territory = validate_wave_4_overlap(df, wave_points, column)
    validation_results.update(wave_1_territory)
    if not wave_1_territory['valid']:
        return False, 0.0, {"error": "wave_4_overlap_violation", "details": wave_1_territory}
    if wave_1_territory.get('is_diagonal', False):
        confidence += 0.1
    else:
        confidence += 0.2
    direction_valid, direction_confidence = validate_wave_directions(waves)
    validation_results['directions'] = direction_valid
    if not direction_valid:
        return False, 0.0, {"error": "direction_violation"}
    confidence += direction_confidence
    fib_confidence = validate_fibonacci_relationships(waves, dates)
    validation_results['fibonacci'] = fib_confidence
    confidence += fib_confidence * 0.3
    if 'volume' in df.columns:
        volume_confidence = validate_volume_patterns(df, wave_points)
        validation_results['volume'] = volume_confidence
        confidence += volume_confidence * 0.2
    alternation_confidence = validate_alternation_principle(df, wave_points, column)
    validation_results['alternation'] = alternation_confidence
    confidence += alternation_confidence * 0.15
    return True, min(1.0, confidence), validation_results

def validate_wave_4_overlap(df: pd.DataFrame, wave_points: np.ndarray, 
                          column: str = 'close') -> Dict[str, Any]:
    if len(wave_points) < 5:
        return {"valid": True, "reason": "insufficient_waves"}
    prices = df[column].iloc[wave_points].values
    wave_1_start = prices[0]
    wave_1_end = prices[1]
    wave_1_high = max(wave_1_start, wave_1_end)
    wave_1_low = min(wave_1_start, wave_1_end)
    wave_4_end = prices[4]
    has_overlap = wave_1_low <= wave_4_end <= wave_1_high
    result = {
        "valid": True,
        "has_overlap": has_overlap,
        "wave_1_territory": (wave_1_low, wave_1_high),
        "wave_4_end": wave_4_end,
        "is_diagonal": False
    }
    if has_overlap:
        diagonal_validation = validate_diagonal_triangle(df, wave_points, column)
        result.update(diagonal_validation)
        if diagonal_validation['is_valid_diagonal']:
            result["valid"] = True
            result["is_diagonal"] = True
            result["reason"] = "valid_diagonal_triangle"
        else:
            result["valid"] = False
            result["reason"] = "invalid_overlap_not_diagonal"
    else:
        result["reason"] = "no_overlap_clean_impulse"
    return result

def validate_diagonal_triangle(df: pd.DataFrame, wave_points: np.ndarray, 
                             column: str = 'close') -> Dict[str, Any]:
    if len(wave_points) < 5:
        return {"is_valid_diagonal": False, "reason": "insufficient_points"}
    prices = df[column].iloc[wave_points].values
    indices = wave_points
    validation_result = {
        "is_valid_diagonal": False,
        "trend_lines_converge": False,
        "volume_diminishes": False,
        "wave_alternation": False,
        "confidence": 0.0
    }
    try:
        trend_analysis = analyze_diagonal_trend_lines(prices, indices)
        validation_result.update(trend_analysis)
        if not trend_analysis['valid_trend_lines']:
            validation_result["reason"] = "invalid_trend_lines"
            return validation_result
        if 'volume' in df.columns:
            volume_analysis = analyze_diagonal_volume_pattern(df, wave_points)
            validation_result.update(volume_analysis)
        else:
            validation_result["volume_diminishes"] = True
        alternation_analysis = analyze_diagonal_wave_alternation(df, wave_points, column)
        validation_result.update(alternation_analysis)
        validation_score = 0
        if trend_analysis['valid_trend_lines']: validation_score += 0.4
        if validation_result['volume_diminishes']: validation_score += 0.3
        if validation_result['wave_alternation']: validation_score += 0.3
        validation_result["confidence"] = validation_score
        validation_result["is_valid_diagonal"] = validation_score >= 0.6
        if validation_result["is_valid_diagonal"]:
            validation_result["reason"] = "valid_diagonal_pattern"
        else:
            validation_result["reason"] = f"insufficient_diagonal_characteristics_{validation_score:.2f}"
    except Exception as e:
        validation_result["reason"] = f"analysis_error_{str(e)}"
    return validation_result

def validate_wave_directions(waves: Dict[int, float]) -> Tuple[bool, float]:
    try:
        trend_up = waves[1] > 0
        confidence = 0.0
        if (waves[3] > 0) == trend_up:
            confidence += 0.3
        else:
            return False, 0.0
        if (waves[2] > 0) != trend_up:
            confidence += 0.2
        else:
            return False, 0.0
        if (waves[4] > 0) != trend_up:
            confidence += 0.2
        else:
            return False, 0.0
        if waves[5] != 0:
            if (waves[5] > 0) == trend_up:
                confidence += 0.3
            else:
                return False, 0.0
        return True, confidence
    except Exception:
        return False, 0.0

# Add all helper functions here (validate_wave_4_overlap, validate_wave_directions, etc.) as pure functions from elliott_wave.py 