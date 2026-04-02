"""
Elliott Wave Time-Based Fibonacci Relationships
===============================================

This module implements time-based Fibonacci analysis for Elliott Wave patterns.
While price-based Fibonacci ratios are well-known, Elliott Wave Theory also
recognizes important time relationships between waves.

Time Relationships:
------------------
1. Wave Duration Equality:
   - Wave 1 duration ≈ Wave 5 duration (common)
   - Wave 2 duration ≈ Wave 4 duration (alternation)

2. Fibonacci Time Ratios:
   - Wave 3 duration = 1.618 × Wave 1 duration (most common)
   - Wave 5 duration = 0.618 × Wave 3 duration
   - Correction duration = 0.382-0.618 × prior impulse duration

3. Time Clusters:
   - Multiple time projections converging at same point
   - Higher probability reversal zones

Functions
---------
validate_fibonacci_time_relationships : Main time validation
calculate_wave_durations : Calculate duration for each wave
check_time_equality : Wave 1 ≈ Wave 5, Wave 2 ≈ Wave 4
check_fibonacci_time_ratios : Validate Fibonacci time extensions
project_time_targets : Project future reversal time targets

References
----------
Frost & Prechter (2005), "Elliott Wave Principle", Time Relationships
Prechter (2017), Fibonacci time projections and clusters

Examples
--------
>>> time_analysis = validate_fibonacci_time_relationships(df, wave_points)
>>> if time_analysis['wave_3_time_extension']['is_fibonacci']:
...     print(f"Wave 3 has Fibonacci time extension: {time_analysis['wave_3_ratio']:.3f}")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Standard Fibonacci ratios
FIBONACCI_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618]
FIBONACCI_TOLERANCE = 0.15  # 15% tolerance for ratio matching


def validate_fibonacci_time_relationships(df: pd.DataFrame, wave_points: np.ndarray,
                                         column: str = 'close') -> Dict[str, Any]:
    """
    Validate time-based Fibonacci relationships between Elliott Waves.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with DatetimeIndex or integer index
    wave_points : np.ndarray
        Array of wave point indices [start, w1, w2, w3, w4, w5]
    column : str, default 'close'
        Price column (not used for time analysis but kept for consistency)

    Returns
    -------
    Dict[str, Any]
        Time relationship validation results with confidence scores
    """
    if len(wave_points) < 6:
        return {
            'valid': False,
            'confidence': 0.0,
            'error': 'insufficient_points'
        }

    try:
        # Calculate wave durations
        durations = calculate_wave_durations(df, wave_points)

        # Check time equality (Wave 1 ≈ Wave 5, Wave 2 ≈ Wave 4)
        equality_result = check_time_equality(durations)

        # Check Fibonacci time ratios
        fibonacci_result = check_fibonacci_time_ratios(durations)

        # Calculate overall confidence
        confidence = (
            equality_result.get('confidence', 0.0) * 0.4 +
            fibonacci_result.get('confidence', 0.0) * 0.6
        )

        # Project time targets for potential Wave 6 (or end of pattern)
        time_targets = project_time_targets(df, wave_points, durations)

        return {
            'valid': True,
            'confidence': min(confidence, 0.95),
            'durations': durations,
            'time_equality': equality_result,
            'fibonacci_ratios': fibonacci_result,
            'time_targets': time_targets
        }

    except Exception as e:
        logger.error(f"Error validating time relationships: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


def calculate_wave_durations(df: pd.DataFrame, wave_points: np.ndarray) -> Dict[str, Any]:
    """
    Calculate duration (in bars/periods) for each wave.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave point indices

    Returns
    -------
    Dict[str, Any]
        Wave durations in bars and time-based units if available
    """
    try:
        durations = {}

        # Calculate duration in bars for each wave
        for i in range(min(5, len(wave_points) - 1)):
            wave_num = i + 1
            bars = wave_points[i + 1] - wave_points[i]
            durations[f'wave_{wave_num}'] = bars

        # Try to calculate actual time duration if DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            time_durations = {}
            for i in range(min(5, len(wave_points) - 1)):
                wave_num = i + 1
                start_time = df.index[wave_points[i]]
                end_time = df.index[wave_points[i + 1]]
                time_delta = end_time - start_time

                time_durations[f'wave_{wave_num}'] = {
                    'timedelta': time_delta,
                    'days': time_delta.days,
                    'hours': time_delta.total_seconds() / 3600
                }

            durations['time_based'] = time_durations

        # Calculate total impulse duration
        total_bars = wave_points[5] - wave_points[0] if len(wave_points) > 5 else 0
        durations['total_impulse'] = total_bars

        return durations

    except Exception as e:
        logger.error(f"Error calculating wave durations: {e}", exc_info=True)
        return {}


def check_time_equality(durations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for time equality between waves.

    Common Equalities:
    - Wave 1 ≈ Wave 5 (impulse waves)
    - Wave 2 ≈ Wave 4 (corrective waves, alternation)

    Parameters
    ----------
    durations : Dict[str, Any]
        Wave durations from calculate_wave_durations()

    Returns
    -------
    Dict[str, Any]
        Time equality validation results
    """
    try:
        wave_1 = durations.get('wave_1', 0)
        wave_2 = durations.get('wave_2', 0)
        wave_3 = durations.get('wave_3', 0)
        wave_4 = durations.get('wave_4', 0)
        wave_5 = durations.get('wave_5', 0)

        confidence = 0.0
        results = {}

        # Check Wave 1 ≈ Wave 5 (tolerance: ±20%)
        if wave_1 > 0 and wave_5 > 0:
            ratio_1_5 = wave_5 / wave_1
            is_equal_1_5 = 0.8 <= ratio_1_5 <= 1.2

            results['wave_1_5_equality'] = {
                'is_equal': is_equal_1_5,
                'ratio': ratio_1_5,
                'wave_1_bars': wave_1,
                'wave_5_bars': wave_5
            }

            if is_equal_1_5:
                confidence += 0.4
            elif 0.7 <= ratio_1_5 <= 1.4:
                confidence += 0.2  # Close but not exact

        # Check Wave 2 ≈ Wave 4 (tolerance: ±25%, more lenient due to alternation)
        if wave_2 > 0 and wave_4 > 0:
            ratio_2_4 = wave_4 / wave_2
            is_equal_2_4 = 0.75 <= ratio_2_4 <= 1.25

            results['wave_2_4_equality'] = {
                'is_equal': is_equal_2_4,
                'ratio': ratio_2_4,
                'wave_2_bars': wave_2,
                'wave_4_bars': wave_4
            }

            if is_equal_2_4:
                confidence += 0.3
            elif 0.6 <= ratio_2_4 <= 1.5:
                confidence += 0.15

        # Check Wave 3 not equal to Wave 1 or 5 (should be different, often longer)
        if wave_1 > 0 and wave_3 > 0:
            ratio_3_1 = wave_3 / wave_1
            wave_3_longer = ratio_3_1 > 1.2

            results['wave_3_distinct'] = {
                'is_longer_than_1': wave_3_longer,
                'ratio_to_wave_1': ratio_3_1
            }

            if wave_3_longer:
                confidence += 0.1

        results['confidence'] = confidence

        return results

    except Exception as e:
        logger.error(f"Error checking time equality: {e}", exc_info=True)
        return {'confidence': 0.0, 'error': str(e)}


def check_fibonacci_time_ratios(durations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Fibonacci time ratios between waves.

    Common Fibonacci Time Relationships:
    - Wave 3 = 1.618 × Wave 1 (most common extension)
    - Wave 5 = 0.618 × Wave 3 (common retracement)
    - Wave 5 = 1.0 × Wave 1 (equality)
    - Wave 2/4 = 0.382-0.618 × prior impulse

    Parameters
    ----------
    durations : Dict[str, Any]
        Wave durations

    Returns
    -------
    Dict[str, Any]
        Fibonacci time ratio validation
    """
    try:
        wave_1 = durations.get('wave_1', 0)
        wave_2 = durations.get('wave_2', 0)
        wave_3 = durations.get('wave_3', 0)
        wave_4 = durations.get('wave_4', 0)
        wave_5 = durations.get('wave_5', 0)

        confidence = 0.0
        results = {}

        # Check Wave 3 / Wave 1 ratio (expect 1.618 or 1.0)
        if wave_1 > 0 and wave_3 > 0:
            ratio_3_1 = wave_3 / wave_1
            fib_match_3_1 = find_nearest_fibonacci_ratio(ratio_3_1)

            results['wave_3_to_1'] = {
                'ratio': ratio_3_1,
                'nearest_fibonacci': fib_match_3_1['ratio'],
                'is_fibonacci': fib_match_3_1['is_match'],
                'deviation': fib_match_3_1['deviation']
            }

            if fib_match_3_1['is_match']:
                # Higher confidence for 1.618 (golden ratio)
                if abs(fib_match_3_1['ratio'] - 1.618) < 0.1:
                    confidence += 0.35
                elif abs(fib_match_3_1['ratio'] - 1.0) < 0.1:
                    confidence += 0.25
                else:
                    confidence += 0.20

        # Check Wave 5 / Wave 3 ratio (expect 0.618 or 0.382)
        if wave_3 > 0 and wave_5 > 0:
            ratio_5_3 = wave_5 / wave_3
            fib_match_5_3 = find_nearest_fibonacci_ratio(ratio_5_3)

            results['wave_5_to_3'] = {
                'ratio': ratio_5_3,
                'nearest_fibonacci': fib_match_5_3['ratio'],
                'is_fibonacci': fib_match_5_3['is_match'],
                'deviation': fib_match_5_3['deviation']
            }

            if fib_match_5_3['is_match']:
                confidence += 0.25

        # Check Wave 5 / Wave 1 ratio (expect 1.0, 0.618, or 1.618)
        if wave_1 > 0 and wave_5 > 0:
            ratio_5_1 = wave_5 / wave_1
            fib_match_5_1 = find_nearest_fibonacci_ratio(ratio_5_1)

            results['wave_5_to_1'] = {
                'ratio': ratio_5_1,
                'nearest_fibonacci': fib_match_5_1['ratio'],
                'is_fibonacci': fib_match_5_1['is_match'],
                'deviation': fib_match_5_1['deviation']
            }

            if fib_match_5_1['is_match']:
                confidence += 0.20

        # Check corrective waves (Wave 2, 4) vs impulse
        if wave_1 > 0 and wave_2 > 0:
            ratio_2_1 = wave_2 / wave_1
            fib_match_2_1 = find_nearest_fibonacci_ratio(ratio_2_1)

            results['wave_2_to_1'] = {
                'ratio': ratio_2_1,
                'nearest_fibonacci': fib_match_2_1['ratio'],
                'is_fibonacci': fib_match_2_1['is_match']
            }

            # Corrections typically 0.382-0.786 of prior impulse
            if 0.3 <= ratio_2_1 <= 0.9 and fib_match_2_1['is_match']:
                confidence += 0.15

        if wave_3 > 0 and wave_4 > 0:
            ratio_4_3 = wave_4 / wave_3
            fib_match_4_3 = find_nearest_fibonacci_ratio(ratio_4_3)

            results['wave_4_to_3'] = {
                'ratio': ratio_4_3,
                'nearest_fibonacci': fib_match_4_3['ratio'],
                'is_fibonacci': fib_match_4_3['is_match']
            }

            if 0.3 <= ratio_4_3 <= 0.9 and fib_match_4_3['is_match']:
                confidence += 0.15

        results['confidence'] = min(confidence, 1.0)

        return results

    except Exception as e:
        logger.error(f"Error checking Fibonacci time ratios: {e}", exc_info=True)
        return {'confidence': 0.0, 'error': str(e)}


def find_nearest_fibonacci_ratio(ratio: float, tolerance: float = FIBONACCI_TOLERANCE) -> Dict[str, Any]:
    """
    Find the nearest Fibonacci ratio to given value.

    Parameters
    ----------
    ratio : float
        Ratio to check
    tolerance : float
        Percentage tolerance for matching (default 15%)

    Returns
    -------
    Dict[str, Any]
        Nearest Fibonacci ratio and match status
    """
    try:
        if ratio <= 0:
            return {'is_match': False, 'ratio': 0, 'deviation': float('inf')}

        # Find nearest Fibonacci ratio
        deviations = [abs(ratio - fib) / fib for fib in FIBONACCI_RATIOS]
        min_deviation_idx = np.argmin(deviations)
        min_deviation = deviations[min_deviation_idx]
        nearest_fib = FIBONACCI_RATIOS[min_deviation_idx]

        is_match = min_deviation <= tolerance

        return {
            'is_match': is_match,
            'ratio': nearest_fib,
            'deviation': min_deviation,
            'actual_ratio': ratio
        }

    except Exception as e:
        logger.debug(f"Error finding Fibonacci ratio: {e}")
        return {'is_match': False, 'ratio': 0, 'deviation': float('inf')}


def project_time_targets(df: pd.DataFrame, wave_points: np.ndarray,
                        durations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project future time targets based on Fibonacci relationships.

    Uses completed waves to project when the pattern might complete
    or when next reversal might occur.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave point indices
    durations : Dict[str, Any]
        Wave durations

    Returns
    -------
    Dict[str, Any]
        Projected time targets with confidence
    """
    try:
        wave_1 = durations.get('wave_1', 0)
        wave_3 = durations.get('wave_3', 0)
        wave_5 = durations.get('wave_5', 0)

        targets = []

        # If Wave 5 is complete, project potential correction duration
        if len(wave_points) >= 6 and wave_1 > 0 and wave_3 > 0 and wave_5 > 0:
            total_impulse = durations.get('total_impulse', 0)

            # Correction typically 38.2%-61.8% of impulse
            correction_low = int(total_impulse * 0.382)
            correction_mid = int(total_impulse * 0.5)
            correction_high = int(total_impulse * 0.618)

            targets.append({
                'target_type': 'correction_duration',
                'min_bars': correction_low,
                'typical_bars': correction_mid,
                'max_bars': correction_high,
                'confidence': 0.6
            })

            # Project index positions
            wave_5_end = wave_points[5]
            targets.append({
                'target_type': 'correction_completion_range',
                'min_index': wave_5_end + correction_low,
                'typical_index': wave_5_end + correction_mid,
                'max_index': wave_5_end + correction_high,
                'confidence': 0.6
            })

        # If only waves 1-3 complete, project Wave 5 duration
        elif len(wave_points) >= 4 and wave_1 > 0 and wave_3 > 0:
            # Wave 5 often equals Wave 1 (1.0) or Wave 3 * 0.618
            wave_5_target_equality = wave_1
            wave_5_target_fibonacci = int(wave_3 * 0.618)

            targets.append({
                'target_type': 'wave_5_duration',
                'equality_projection': wave_5_target_equality,
                'fibonacci_projection': wave_5_target_fibonacci,
                'confidence': 0.5,
                'note': 'Wave 5 projection based on Wave 1 equality or Wave 3 * 0.618'
            })

        return {
            'projections': targets,
            'total_targets': len(targets)
        }

    except Exception as e:
        logger.error(f"Error projecting time targets: {e}", exc_info=True)
        return {'projections': [], 'error': str(e)}


def identify_time_clusters(time_targets: List[int], tolerance_bars: int = 3) -> List[Dict[str, Any]]:
    """
    Identify time clusters where multiple projections converge.

    Time clusters indicate higher probability reversal zones.

    Parameters
    ----------
    time_targets : List[int]
        List of projected target indices
    tolerance_bars : int
        Number of bars tolerance for clustering

    Returns
    -------
    List[Dict[str, Any]]
        Time clusters with confidence scores
    """
    try:
        if len(time_targets) < 2:
            return []

        # Sort targets
        sorted_targets = sorted(time_targets)

        clusters = []
        current_cluster = [sorted_targets[0]]

        for i in range(1, len(sorted_targets)):
            # Check if within tolerance of current cluster
            if sorted_targets[i] - current_cluster[-1] <= tolerance_bars:
                current_cluster.append(sorted_targets[i])
            else:
                # Save current cluster if it has multiple targets
                if len(current_cluster) >= 2:
                    cluster_center = int(np.mean(current_cluster))
                    cluster_confidence = min(0.3 + len(current_cluster) * 0.15, 0.9)

                    clusters.append({
                        'center_index': cluster_center,
                        'target_count': len(current_cluster),
                        'confidence': cluster_confidence,
                        'targets': current_cluster
                    })

                # Start new cluster
                current_cluster = [sorted_targets[i]]

        # Check last cluster
        if len(current_cluster) >= 2:
            cluster_center = int(np.mean(current_cluster))
            cluster_confidence = min(0.3 + len(current_cluster) * 0.15, 0.9)

            clusters.append({
                'center_index': cluster_center,
                'target_count': len(current_cluster),
                'confidence': cluster_confidence,
                'targets': current_cluster
            })

        return clusters

    except Exception as e:
        logger.debug(f"Error identifying time clusters: {e}")
        return []
