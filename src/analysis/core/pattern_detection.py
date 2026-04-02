"""
Complex Elliott Wave Pattern Detection
=======================================

This module provides detection for complex Elliott Wave patterns beyond
basic impulse/corrective structures, including:
- Diagonal triangles (leading and ending)
- Truncated fifth waves
- Extended waves
- Wave equality relationships

Functions
---------
detect_complex_elliott_patterns : Main entry point for complex pattern detection
check_diagonal_triangle_pattern : Detect diagonal triangle patterns
check_truncated_fifth_wave : Detect truncated fifth wave patterns

Examples
--------
>>> import pandas as pd
>>> from pattern_detection import detect_complex_elliott_patterns
>>> patterns = detect_complex_elliott_patterns(df, wave_data, column='close')
>>> if patterns['diagonal_triangle']:
...     print(f"Diagonal type: {patterns['diagonal_triangle']['type']}")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from .utils import get_pattern_direction
from .validation import WaveEqualityChecker

logger = logging.getLogger(__name__)


def detect_complex_elliott_patterns(df: pd.DataFrame, wave_data: Dict[str, Any],
                                  column: str = 'close') -> Dict[str, Any]:
    """
    Detect complex Elliott Wave patterns beyond basic impulse/corrective.

    Integrates with existing wave detection to identify special patterns
    such as diagonal triangles, truncated fifths, extended waves, and
    wave equality relationships.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC price data and datetime index
    wave_data : Dict[str, Any]
        Dictionary containing basic wave detection results with 'impulse_wave' key
    column : str, default 'close'
        Column name for price data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing detected complex patterns:
        - diagonal_triangle: Diagonal triangle pattern info or None
        - truncated_fifth: Truncated fifth wave info or None
        - extended_wave: Extended wave info or None
        - wave_equality: Wave equality relationships or None
        - pattern_confidence: Overall confidence score [0, 1]

    Notes
    -----
    This function should be called after basic wave detection to enhance
    results with complex pattern recognition.
    """
    patterns = {
        'diagonal_triangle': None,
        'truncated_fifth': None,
        'extended_wave': None,
        'wave_equality': None,
        'pattern_confidence': 0.0
    }

    impulse_wave = wave_data.get('impulse_wave', np.array([]))

    if len(impulse_wave) < 3:
        logger.debug("Insufficient wave points for complex pattern detection")
        return patterns

    try:
        # Check for diagonal triangle (leading or ending)
        diagonal_result = check_diagonal_triangle_pattern(df, impulse_wave, column)
        if diagonal_result['is_diagonal']:
            patterns['diagonal_triangle'] = diagonal_result
            patterns['pattern_confidence'] = max(
                patterns['pattern_confidence'],
                diagonal_result['confidence']
            )
            logger.info(f"Diagonal triangle detected: {diagonal_result['type']}, "
                       f"confidence={diagonal_result['confidence']:.2f}")

        # Check for truncated 5th wave
        if len(impulse_wave) >= 5:
            truncation_result = check_truncated_fifth_wave(df, impulse_wave, column)
            if truncation_result['is_truncated']:
                patterns['truncated_fifth'] = truncation_result
                patterns['pattern_confidence'] = max(
                    patterns['pattern_confidence'],
                    truncation_result['confidence']
                )
                logger.info(f"Truncated fifth wave detected, "
                           f"confidence={truncation_result['confidence']:.2f}")

        # Calculate wave lengths for equality and extension checks
        prices = df[column].iloc[impulse_wave].values
        waves = {i+1: prices[i+1] - prices[i] for i in range(min(len(prices)-1, 5))}

        # Use WaveEqualityChecker for wave equality and extension analysis
        equality_checker = WaveEqualityChecker(tolerance=0.1)

        # Check for extended waves
        extension_result = equality_checker.check_wave_extensions(waves)
        if extension_result['has_extension']:
            patterns['extended_wave'] = extension_result
            patterns['pattern_confidence'] = max(
                patterns['pattern_confidence'],
                extension_result['confidence']
            )
            logger.info(f"Extended wave detected: Wave {extension_result.get('extended_wave')}, "
                       f"confidence={extension_result['confidence']:.2f}")

        # Check for wave equality relationships
        equality_result = equality_checker.check_wave_equality(waves)
        if equality_result['has_equality']:
            patterns['wave_equality'] = equality_result
            logger.debug(f"Wave equality relationships found: {equality_result}")

    except Exception as e:
        logger.error(f"Error in complex pattern detection: {e}", exc_info=True)
        return patterns

    return patterns


def check_diagonal_triangle_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                                  column: str = 'close') -> Dict[str, Any]:
    """
    Check for diagonal triangle pattern characteristics.

    Diagonal triangles are wedge-shaped patterns where trendlines converge.
    They can be leading (wave 1 or A) or ending (wave 5 or C) diagonals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    wave_points : np.ndarray
        Array of wave point indices
    column : str, default 'close'
        Column name for price data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - is_diagonal: bool, True if diagonal pattern detected
        - confidence: float [0, 1], confidence in detection
        - type: str, 'leading' or 'ending' diagonal type
        - characteristics: Dict with pattern characteristics

    Notes
    -----
    Diagonal triangles have converging trendlines and typically show
    declining volume in the direction of the trend.
    """
    result = {
        'is_diagonal': False,
        'confidence': 0.0,
        'type': None,
        'characteristics': {}
    }

    if len(wave_points) < 5:
        return result

    try:
        prices = df[column].iloc[wave_points].values

        # Calculate wave lengths
        waves = {i+1: prices[i+1] - prices[i] for i in range(min(len(prices)-1, 5))}

        # Check for converging trend lines
        trend_lines_converge = True
        for i in range(1, 4):
            if i+2 < len(prices):
                # Calculate slopes between consecutive wave peaks/troughs
                slope1 = (prices[i+1] - prices[i]) / (wave_points[i+1] - wave_points[i])
                slope2 = (prices[i+2] - prices[i+1]) / (wave_points[i+2] - wave_points[i+1])

                # Trendlines should be converging (slopes decreasing in magnitude)
                if abs(slope1) <= abs(slope2):
                    trend_lines_converge = False
                    break

        if trend_lines_converge:
            result['is_diagonal'] = True
            result['confidence'] = 0.7
            # Determine diagonal type based on net direction
            result['type'] = 'ending' if prices[-1] > prices[0] else 'leading'
            result['characteristics']['trend_lines_converge'] = True

    except (KeyError, IndexError) as e:
        logger.warning(f"Error checking diagonal triangle pattern: {e}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in diagonal triangle check: {e}", exc_info=True)
        return result

    return result


def check_truncated_fifth_wave(df: pd.DataFrame, wave_points: np.ndarray,
                             column: str = 'close') -> Dict[str, Any]:
    """
    Check for truncated fifth wave pattern.

    A truncated fifth wave occurs when wave 5 fails to move beyond the
    end of wave 3, indicating strong reversal momentum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    wave_points : np.ndarray
        Array of wave point indices
    column : str, default 'close'
        Column name for price data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - is_truncated: bool, True if truncated fifth detected
        - confidence: float [0, 1], confidence in detection
        - characteristics: Dict with pattern characteristics

    Notes
    -----
    Truncated fifths are rare and typically signal strong reversals.
    Confidence increases if wave 5 is significantly shorter (< 0.618 * wave 3).
    """
    result = {
        'is_truncated': False,
        'confidence': 0.0,
        'characteristics': {}
    }

    if len(wave_points) < 6:  # Need all 5 waves (6 points)
        return result

    try:
        prices = df[column].iloc[wave_points].values

        # Calculate wave lengths
        waves = {i+1: prices[i+1] - prices[i] for i in range(min(len(prices)-1, 5))}

        # Check if wave 5 is shorter than wave 3
        if abs(waves.get(5, 0)) < abs(waves.get(3, 0)):
            result['is_truncated'] = True
            result['confidence'] = 0.6
            result['characteristics']['wave5_shorter_than_wave3'] = True

            # Additional confidence if wave 5 is significantly shorter
            # 0.618 is Fibonacci ratio indicating significant truncation
            if abs(waves[5]) < abs(waves[3]) * 0.618:
                result['confidence'] = 0.8
                result['characteristics']['significant_truncation'] = True

    except (KeyError, IndexError) as e:
        logger.warning(f"Error checking truncated fifth wave: {e}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error in truncated fifth check: {e}", exc_info=True)
        return result

    return result


def check_wave_equality_simple(wave_lengths: list, tolerance: float = 0.1) -> bool:
    """
    Check if any waves are approximately equal in length.

    Parameters
    ----------
    wave_lengths : list
        List of wave lengths to compare
    tolerance : float, default 0.1
        Tolerance for equality (0.1 = 10%)

    Returns
    -------
    bool
        True if any two waves are approximately equal

    Notes
    -----
    Wave equality is common in Elliott Wave patterns, especially
    between waves 1 and 5, or alternating corrective waves.
    """
    if len(wave_lengths) < 2:
        return False

    for i in range(len(wave_lengths)):
        for j in range(i + 1, len(wave_lengths)):
            if wave_lengths[j] == 0:
                continue

            ratio = wave_lengths[i] / wave_lengths[j]
            if abs(1 - ratio) <= tolerance:
                return True

    return False
