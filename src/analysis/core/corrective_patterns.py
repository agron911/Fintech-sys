"""
Elliott Wave Corrective Pattern Analysis
=========================================

This module implements detailed analysis of corrective wave patterns according
to Elliott Wave Theory.

Corrective Patterns Classified:
- Zigzag (5-3-5): Sharp corrections, often retrace 50-78.6%
- Flat (3-3-5): Sideways corrections, three waves of similar magnitude
- Triangle (3-3-3-3-3): Contracting/expanding triangles in wave 4 or B
- Complex Corrections: Double/triple zigzags, W-X-Y or W-X-Y-X-Z patterns

Pattern Types:
--------------
1. Simple Corrections:
   - Zigzag (5-3-5)
   - Flat (3-3-5): Regular, Expanded, Running
   - Triangle (3-3-3-3-3): Contracting, Expanding, Ascending, Descending

2. Complex Corrections:
   - Double Zigzag (5-3-5-X-5-3-5)
   - Triple Zigzag (5-3-5-X-5-3-5-X-5-3-5)
   - Double Three (3-3-5-X-3-3-5)
   - Triple Three (W-X-Y-X-Z)

Functions
---------
classify_corrective_pattern : Main pattern classification
detect_zigzag_pattern : Identifies zigzag corrections (5-3-5)
detect_flat_pattern : Identifies flat corrections (3-3-5)
detect_triangle_pattern : Identifies triangle patterns
detect_complex_correction : Identifies double/triple zigzags, W-X-Y patterns
classify_triangle_subtype : Ascending, descending, contracting, expanding

References
----------
Frost & Prechter (2005), "Elliott Wave Principle", Chapter 2: Corrective Waves
Prechter (2017), "The Elliott Wave Theorist", Complex Corrections

Examples
--------
>>> pattern = classify_corrective_pattern(df, wave_points, column='close')
>>> if pattern['type'] == 'double_zigzag':
...     print(f"Complex correction detected: {pattern['structure']}")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CorrectivePatternType(Enum):
    """Enumeration of corrective pattern types."""
    ZIGZAG = "zigzag"
    FLAT = "flat"
    TRIANGLE = "triangle"
    DOUBLE_ZIGZAG = "double_zigzag"
    TRIPLE_ZIGZAG = "triple_zigzag"
    DOUBLE_THREE = "double_three"
    TRIPLE_THREE = "triple_three"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class FlatType(Enum):
    """Flat correction subtypes."""
    REGULAR = "regular"      # Wave B ~90-100% of A
    EXPANDED = "expanded"    # Wave B >100% of A
    RUNNING = "running"      # Wave C fails to go beyond A


class TriangleType(Enum):
    """Triangle pattern subtypes."""
    CONTRACTING = "contracting"  # Upper/lower boundaries converge
    EXPANDING = "expanding"      # Upper/lower boundaries diverge
    ASCENDING = "ascending"      # Flat top, rising bottom
    DESCENDING = "descending"    # Flat bottom, declining top
    SYMMETRICAL = "symmetrical"  # Contracting with equal slopes


def classify_corrective_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                                column: str = 'close') -> Dict[str, Any]:
    """
    Classify corrective wave pattern according to Elliott Wave Theory.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLCV columns
    wave_points : np.ndarray
        Array of wave point indices (minimum 3 points for simple correction)
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Pattern classification with structure, confidence, and details
    """
    if len(wave_points) < 3:
        return {
            'type': CorrectivePatternType.UNKNOWN.value,
            'confidence': 0.0,
            'error': 'insufficient_points'
        }

    try:
        # Check for complex corrections first (need more points)
        if len(wave_points) >= 7:
            complex_result = detect_complex_correction(df, wave_points, column)
            if complex_result['valid'] and complex_result['confidence'] > 0.5:
                return complex_result

        # Check simple patterns
        if len(wave_points) >= 4:
            # Check zigzag (5-3-5 structure)
            zigzag_result = detect_zigzag_pattern(df, wave_points, column)
            if zigzag_result['valid'] and zigzag_result['confidence'] > 0.5:
                return zigzag_result

            # Check flat (3-3-5 structure)
            flat_result = detect_flat_pattern(df, wave_points, column)
            if flat_result['valid'] and flat_result['confidence'] > 0.5:
                return flat_result

        # Check triangle (needs 5 waves: A-B-C-D-E)
        if len(wave_points) >= 6:
            triangle_result = detect_triangle_pattern(df, wave_points, column)
            if triangle_result['valid'] and triangle_result['confidence'] > 0.5:
                return triangle_result

        # Default to unknown pattern
        return {
            'type': CorrectivePatternType.UNKNOWN.value,
            'confidence': 0.0,
            'structure': 'undetermined'
        }

    except Exception as e:
        logger.error(f"Error classifying corrective pattern: {e}", exc_info=True)
        return {'type': CorrectivePatternType.UNKNOWN.value, 'confidence': 0.0, 'error': str(e)}


def detect_zigzag_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                          column: str = 'close') -> Dict[str, Any]:
    """
    Detect zigzag pattern (5-3-5 structure).

    Zigzag Characteristics:
    - Wave A: 5-wave impulse
    - Wave B: 3-wave correction (typically 38.2%-61.8% retracement)
    - Wave C: 5-wave impulse (often 1.0-1.618x length of A)

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave points [start, A_end, B_end, C_end]
    column : str
        Price column

    Returns
    -------
    Dict[str, Any]
        Zigzag validation results
    """
    try:
        if len(wave_points) < 4:
            return {'valid': False, 'confidence': 0.0}

        prices = df[column].values

        # Calculate wave lengths
        wave_a_len = abs(prices[wave_points[1]] - prices[wave_points[0]])
        wave_b_len = abs(prices[wave_points[2]] - prices[wave_points[1]])
        wave_c_len = abs(prices[wave_points[3]] - prices[wave_points[2]])

        # Wave B retracement (should be 38.2%-78.6% of A)
        wave_b_retracement = wave_b_len / wave_a_len if wave_a_len > 0 else 0

        # Wave C/A ratio (often 1.0, 1.272, or 1.618)
        wave_c_ratio = wave_c_len / wave_a_len if wave_a_len > 0 else 0

        # Validation checks
        confidence = 0.0

        # Wave B retracement in typical range (38.2%-78.6%)
        if 0.382 <= wave_b_retracement <= 0.786:
            confidence += 0.35
        elif 0.20 <= wave_b_retracement <= 0.90:
            confidence += 0.15  # Outside ideal but acceptable

        # Wave C near Fibonacci extension
        if 0.9 <= wave_c_ratio <= 1.1:  # ~1.0x
            confidence += 0.30
        elif 1.2 <= wave_c_ratio <= 1.35:  # ~1.272x
            confidence += 0.25
        elif 1.5 <= wave_c_ratio <= 1.75:  # ~1.618x
            confidence += 0.30
        elif 0.618 <= wave_c_ratio <= 2.618:
            confidence += 0.10  # Within broader Fibonacci range

        # Wave C extends beyond Wave A endpoint
        is_bullish_correction = prices[wave_points[0]] > prices[wave_points[1]]
        if is_bullish_correction:
            extends_properly = prices[wave_points[3]] < prices[wave_points[1]]
        else:
            extends_properly = prices[wave_points[3]] > prices[wave_points[1]]

        if extends_properly:
            confidence += 0.20

        is_valid = confidence >= 0.4

        return {
            'valid': is_valid,
            'type': CorrectivePatternType.ZIGZAG.value,
            'structure': '5-3-5',
            'confidence': min(confidence, 0.95),
            'wave_b_retracement': wave_b_retracement,
            'wave_c_ratio': wave_c_ratio,
            'extends_properly': extends_properly
        }

    except Exception as e:
        logger.error(f"Error detecting zigzag pattern: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


def detect_flat_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                       column: str = 'close') -> Dict[str, Any]:
    """
    Detect flat pattern (3-3-5 structure).

    Flat Characteristics:
    - Wave A: 3-wave correction
    - Wave B: 3-wave correction (90-138% retracement of A)
    - Wave C: 5-wave impulse (typically terminates near A endpoint)

    Types:
    - Regular Flat: B = 90-100% of A, C ≈ A endpoint
    - Expanded Flat: B > 100% of A, C extends beyond A
    - Running Flat: B > 100% of A, C fails to reach A endpoint

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave points [start, A_end, B_end, C_end]
    column : str
        Price column

    Returns
    -------
    Dict[str, Any]
        Flat validation results with subtype
    """
    try:
        if len(wave_points) < 4:
            return {'valid': False, 'confidence': 0.0}

        prices = df[column].values

        # Calculate wave lengths
        wave_a_len = abs(prices[wave_points[1]] - prices[wave_points[0]])
        wave_b_len = abs(prices[wave_points[2]] - prices[wave_points[1]])
        wave_c_len = abs(prices[wave_points[3]] - prices[wave_points[2]])

        # Wave B retracement (should be 90-138% for flat)
        wave_b_retracement = wave_b_len / wave_a_len if wave_a_len > 0 else 0

        # Classify flat subtype
        flat_subtype = None
        confidence = 0.0

        if 0.90 <= wave_b_retracement <= 1.00:
            flat_subtype = FlatType.REGULAR.value
            confidence += 0.35
        elif 1.00 < wave_b_retracement <= 1.38:
            flat_subtype = FlatType.EXPANDED.value
            confidence += 0.30
        elif wave_b_retracement > 1.38:
            flat_subtype = FlatType.RUNNING.value
            confidence += 0.20
        elif 0.70 <= wave_b_retracement < 0.90:
            confidence += 0.10  # Weak flat

        # Check Wave C termination
        is_bullish_correction = prices[wave_points[0]] > prices[wave_points[1]]

        # Wave C should move in same direction as Wave A
        if is_bullish_correction:
            c_moves_correct_direction = prices[wave_points[3]] < prices[wave_points[2]]
        else:
            c_moves_correct_direction = prices[wave_points[3]] > prices[wave_points[2]]

        if c_moves_correct_direction:
            confidence += 0.20

        # Wave C length (typically 100-165% of A for regular/expanded)
        wave_c_ratio = wave_c_len / wave_a_len if wave_a_len > 0 else 0
        if 0.9 <= wave_c_ratio <= 1.65:
            confidence += 0.25

        # Check if three waves have similar magnitude (characteristic of flats)
        avg_wave_len = (wave_a_len + wave_b_len + wave_c_len) / 3
        all_similar = all(
            0.5 <= (wave_len / avg_wave_len) <= 1.5
            for wave_len in [wave_a_len, wave_b_len, wave_c_len]
        )
        if all_similar:
            confidence += 0.15

        is_valid = confidence >= 0.4

        return {
            'valid': is_valid,
            'type': CorrectivePatternType.FLAT.value,
            'subtype': flat_subtype,
            'structure': '3-3-5',
            'confidence': min(confidence, 0.95),
            'wave_b_retracement': wave_b_retracement,
            'wave_c_ratio': wave_c_ratio,
            'waves_similar_magnitude': all_similar
        }

    except Exception as e:
        logger.error(f"Error detecting flat pattern: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


def detect_triangle_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                           column: str = 'close') -> Dict[str, Any]:
    """
    Detect triangle pattern (3-3-3-3-3 structure).

    Triangle Characteristics:
    - Five overlapping waves (A-B-C-D-E)
    - Each wave subdivides into 3
    - Typically appears in wave 4 or wave B
    - Usually followed by sharp move (thrust)

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave points [start, A, B, C, D, E]
    column : str
        Price column

    Returns
    -------
    Dict[str, Any]
        Triangle validation with subtype
    """
    try:
        if len(wave_points) < 6:
            return {'valid': False, 'confidence': 0.0}

        prices = df[column].values

        # Extract triangle wave endpoints
        triangle_prices = prices[wave_points[:6]]

        # Separate upper and lower boundaries
        peaks = []
        troughs = []

        for i in range(len(triangle_prices)):
            if i % 2 == 1:  # Odd indices (A, C, E or B, D)
                if i < len(triangle_prices) - 1:
                    if triangle_prices[i] > triangle_prices[i-1] and triangle_prices[i] > triangle_prices[i+1]:
                        peaks.append(triangle_prices[i])
                    else:
                        troughs.append(triangle_prices[i])

        # Classify triangle subtype
        triangle_subtype = classify_triangle_subtype(wave_points, prices)

        # Validation checks
        confidence = 0.0

        # Each wave should be smaller than previous (converging)
        wave_lengths = []
        for i in range(5):
            wave_len = abs(triangle_prices[i+1] - triangle_prices[i])
            wave_lengths.append(wave_len)

        # Check if waves generally decrease in size (converging pattern)
        is_converging = True
        for i in range(len(wave_lengths) - 1):
            if wave_lengths[i+1] > wave_lengths[i] * 1.2:
                is_converging = False
                break

        if is_converging:
            confidence += 0.40

        # Waves should overlap (characteristic of triangles)
        has_overlaps = False
        for i in range(len(triangle_prices) - 3):
            # Check if wave i+2 overlaps wave i
            wave_i_range = (min(triangle_prices[i], triangle_prices[i+1]),
                          max(triangle_prices[i], triangle_prices[i+1]))
            wave_i2_range = (min(triangle_prices[i+2], triangle_prices[i+3]),
                            max(triangle_prices[i+2], triangle_prices[i+3]))

            # Check for overlap
            if (wave_i2_range[0] <= wave_i_range[1] and
                wave_i2_range[1] >= wave_i_range[0]):
                has_overlaps = True
                break

        if has_overlaps:
            confidence += 0.30

        # Triangle subtype gives additional confidence
        if triangle_subtype['type'] != 'unknown':
            confidence += 0.25

        is_valid = confidence >= 0.4

        return {
            'valid': is_valid,
            'type': CorrectivePatternType.TRIANGLE.value,
            'subtype': triangle_subtype['type'],
            'structure': '3-3-3-3-3',
            'confidence': min(confidence, 0.95),
            'is_converging': is_converging,
            'has_overlaps': has_overlaps,
            'subtype_details': triangle_subtype
        }

    except Exception as e:
        logger.error(f"Error detecting triangle pattern: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


def classify_triangle_subtype(wave_points: np.ndarray,
                              prices: np.ndarray) -> Dict[str, Any]:
    """
    Classify triangle into subtype: contracting, expanding, ascending, descending.

    Parameters
    ----------
    wave_points : np.ndarray
        Triangle wave points (at least 6 points)
    prices : np.ndarray
        Full price array

    Returns
    -------
    Dict[str, Any]
        Triangle subtype classification
    """
    try:
        if len(wave_points) < 6:
            return {'type': 'unknown', 'confidence': 0.0}

        triangle_prices = prices[wave_points[:6]]

        # Identify peaks (upper boundary) and troughs (lower boundary)
        peaks_idx = [1, 3]  # Waves B and D typically
        troughs_idx = [0, 2, 4]  # Waves A, C, E typically

        # Adjust based on direction
        if triangle_prices[1] < triangle_prices[0]:
            peaks_idx, troughs_idx = troughs_idx, peaks_idx

        peaks = [triangle_prices[i] for i in peaks_idx if i < len(triangle_prices)]
        troughs = [triangle_prices[i] for i in troughs_idx if i < len(triangle_prices)]

        if len(peaks) < 2 or len(troughs) < 2:
            return {'type': 'unknown', 'confidence': 0.0}

        # Analyze boundary trends
        upper_slope = peaks[-1] - peaks[0]  # Positive = rising, Negative = falling
        lower_slope = troughs[-1] - troughs[0]

        # Calculate slopes more precisely
        upper_range = max(peaks) - min(peaks)
        lower_range = max(troughs) - min(troughs)
        avg_range = (upper_range + lower_range) / 2

        # Classify based on boundary behavior
        tolerance = avg_range * 0.15  # 15% tolerance

        # Ascending: flat top, rising bottom
        if abs(upper_slope) < tolerance and lower_slope > tolerance:
            return {
                'type': TriangleType.ASCENDING.value,
                'confidence': 0.8,
                'upper_boundary': 'flat',
                'lower_boundary': 'rising'
            }

        # Descending: declining top, flat bottom
        elif upper_slope < -tolerance and abs(lower_slope) < tolerance:
            return {
                'type': TriangleType.DESCENDING.value,
                'confidence': 0.8,
                'upper_boundary': 'declining',
                'lower_boundary': 'flat'
            }

        # Contracting: converging boundaries
        elif upper_slope < -tolerance and lower_slope > tolerance:
            return {
                'type': TriangleType.CONTRACTING.value,
                'confidence': 0.9,
                'upper_boundary': 'declining',
                'lower_boundary': 'rising'
            }

        # Expanding: diverging boundaries
        elif upper_slope > tolerance and lower_slope < -tolerance:
            return {
                'type': TriangleType.EXPANDING.value,
                'confidence': 0.7,
                'upper_boundary': 'rising',
                'lower_boundary': 'declining'
            }

        # Symmetrical contracting (both slopes toward center)
        elif abs(upper_slope + lower_slope) < tolerance:
            return {
                'type': TriangleType.SYMMETRICAL.value,
                'confidence': 0.85,
                'upper_boundary': 'declining',
                'lower_boundary': 'rising'
            }

        else:
            return {'type': 'unknown', 'confidence': 0.0}

    except Exception as e:
        logger.debug(f"Error classifying triangle subtype: {e}")
        return {'type': 'unknown', 'confidence': 0.0}


def detect_complex_correction(df: pd.DataFrame, wave_points: np.ndarray,
                              column: str = 'close') -> Dict[str, Any]:
    """
    Detect complex corrections: double/triple zigzags, W-X-Y, W-X-Y-X-Z patterns.

    Complex Correction Types:
    - Double Zigzag: 5-3-5-X-5-3-5 (two zigzags connected by X wave)
    - Triple Zigzag: Three zigzags connected by X waves
    - Double Three: Two corrective patterns (flat/zigzag) connected by X
    - Triple Three: W-X-Y-X-Z (three corrective patterns)

    The X wave is a connector (usually a zigzag or flat).

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave points (needs at least 7 for double pattern)
    column : str
        Price column

    Returns
    -------
    Dict[str, Any]
        Complex correction classification
    """
    try:
        if len(wave_points) < 7:
            return {'valid': False, 'confidence': 0.0}

        prices = df[column].values

        # Try to identify double zigzag pattern (most common complex correction)
        # Structure: W(5-3-5) - X(3) - Y(5-3-5)

        # For double zigzag, we need at least 7 points
        # [0=start, 1=W.A, 2=W.B, 3=W.C, 4=X, 5=Y.A, 6=Y.B, 7=Y.C]

        if len(wave_points) >= 8:
            # Validate W as zigzag
            w_points = wave_points[:4]
            w_result = detect_zigzag_pattern(df, w_points, column)

            # Validate Y as zigzag
            y_start_idx = 3  # After W.C
            y_points = np.array([wave_points[y_start_idx]] + list(wave_points[y_start_idx+1:y_start_idx+4]))
            if len(y_points) >= 4:
                y_result = detect_zigzag_pattern(df, y_points, column)

                # Check X wave (connector)
                x_len = abs(prices[wave_points[4]] - prices[wave_points[3]])
                w_len = abs(prices[wave_points[3]] - prices[wave_points[0]])
                x_ratio = x_len / w_len if w_len > 0 else 0

                # X wave should be smaller (typically 61.8%-80% of W)
                x_is_small = 0.3 <= x_ratio <= 0.9

                # Both W and Y should be valid zigzags
                if (w_result.get('valid', False) and
                    y_result.get('valid', False) and
                    x_is_small):

                    confidence = (
                        w_result.get('confidence', 0.0) * 0.40 +
                        y_result.get('confidence', 0.0) * 0.40 +
                        (0.20 if x_is_small else 0.0)
                    )

                    return {
                        'valid': True,
                        'type': CorrectivePatternType.DOUBLE_ZIGZAG.value,
                        'structure': 'W-X-Y (5-3-5-X-5-3-5)',
                        'confidence': min(confidence, 0.95),
                        'w_pattern': w_result,
                        'y_pattern': y_result,
                        'x_ratio': x_ratio
                    }

        # Check for triple zigzag if enough points
        if len(wave_points) >= 11:
            # Structure: W-X-Y-X-Z
            # This is less common but possible

            # For now, return as complex pattern if we detect multiple zigzags
            return {
                'valid': True,
                'type': CorrectivePatternType.COMPLEX.value,
                'structure': 'Multiple corrective patterns',
                'confidence': 0.5,
                'note': 'Possible triple zigzag or complex correction'
            }

        # Check for double three (two corrective patterns of any type)
        if len(wave_points) >= 7:
            # First pattern (could be flat or zigzag)
            first_points = wave_points[:4]
            first_zigzag = detect_zigzag_pattern(df, first_points, column)
            first_flat = detect_flat_pattern(df, first_points, column)

            first_pattern = first_zigzag if first_zigzag['confidence'] > first_flat['confidence'] else first_flat

            if first_pattern.get('valid', False):
                return {
                    'valid': True,
                    'type': CorrectivePatternType.DOUBLE_THREE.value,
                    'structure': 'Two corrective patterns connected by X',
                    'confidence': first_pattern['confidence'] * 0.7,
                    'first_pattern': first_pattern,
                    'note': 'Partial double three pattern detected'
                }

        return {'valid': False, 'confidence': 0.0}

    except Exception as e:
        logger.error(f"Error detecting complex correction: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


# Backward compatibility wrapper
def detect_corrective_patterns(df: pd.DataFrame, start_idx: int,
                               column: str = 'close') -> Dict[str, Any]:
    """
    Backward compatibility wrapper for classify_corrective_pattern.

    This function exists for compatibility with older code that used the
    previous corrective.py module.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with OHLCV columns
    start_idx : int
        Starting index (currently unused, kept for compatibility)
    column : str, default 'close'
        Price column name

    Returns
    -------
    dict
        Pattern detection results with 'type' and 'confidence' keys
    """
    try:
        # Try to detect peaks/troughs if not provided
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        peaks, troughs = detect_peaks_troughs_enhanced(df)

        # Combine peaks and troughs and sort
        if len(peaks) > 0 and len(troughs) > 0:
            all_points = np.sort(np.concatenate([peaks, troughs]))
        elif len(peaks) > 0:
            all_points = peaks
        elif len(troughs) > 0:
            all_points = troughs
        else:
            return {'type': 'unknown', 'confidence': 0.0}

        # Use at least 3 points for pattern detection
        if len(all_points) < 3:
            return {'type': 'unknown', 'confidence': 0.0}

        # Take first few points for pattern detection
        wave_points = all_points[:min(7, len(all_points))]

        result = classify_corrective_pattern(df, wave_points, column)

        # Convert to old format
        return {
            'type': result.get('type', 'unknown').upper() if result.get('type') else 'unknown',
            'confidence': result.get('confidence', 0.0),
            'subtype': result.get('subtype', None),
            'structure': result.get('structure', None)
        }

    except Exception as e:
        logger.error(f"Error in detect_corrective_patterns: {e}", exc_info=True)
        return {'type': 'unknown', 'confidence': 0.0}