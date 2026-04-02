"""
Enhanced Elliott Wave Validation - Critical Rules
=================================================

This module implements strict Elliott Wave validation rules with MANDATORY
enforcement of core principles that should NEVER be violated in true impulse waves.

Critical Fixes:
1. Wave 4 overlap rule - MANDATORY enforcement (not optional)
2. Wave 3 length rule - Enhanced validation (cannot be shortest)
3. Wave subdivision validation - Check 5-wave and 3-wave structure
4. Alternation principle - Sharp vs sideways classification
5. Channel analysis - Parallel channel validation
6. Multi-degree hierarchy - Track wave relationships

References:
- Frost & Prechter, "Elliott Wave Principle" (2005)
- Rule #2: Wave 3 is never the shortest
- Rule #3: Wave 4 never overlaps Wave 1 price territory

Examples
--------
>>> from elliott_validation_enhanced import validate_impulse_wave_strict
>>> is_valid, confidence, details = validate_impulse_wave_strict(df, wave_points)
>>> if not is_valid:
...     print(f"Mandatory rule violation: {details['error']}")
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CorrectionType(Enum):
    """Classification of corrective wave structure"""
    SHARP = "sharp"          # Zigzag - steep, doesn't reach prior high
    SIDEWAYS = "sideways"    # Flat, Triangle - moves back to/beyond prior high
    UNDEFINED = "undefined"


# ==============================================================================
# CRITICAL FIX #1: MANDATORY Wave 4 Overlap Rule
# ==============================================================================

def validate_wave_4_overlap_mandatory(prices: np.ndarray,
                                     is_diagonal: bool = False) -> Dict[str, Any]:
    """
    MANDATORY Wave 4 overlap validation - Rule #3.

    In impulse waves, Wave 4 must NEVER overlap Wave 1 price territory.
    This is not optional - any overlap immediately invalidates the impulse.

    Exception: Diagonal triangles allow overlap.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices at wave points [0, 1, 2, 3, 4, 5]
    is_diagonal : bool, default False
        True if pattern is identified as diagonal triangle

    Returns
    -------
    Dict[str, Any]
        - valid: bool, True only if no overlap exists
        - has_overlap: bool
        - confidence: float, 0.0 if overlap, 1.0 if clean
        - error: str, mandatory rule violation message if applicable
        - overlap_amount: float, price overlap amount
        - overlap_percentage: float, overlap as % of Wave 1 range

    Notes
    -----
    This is Elliott Wave Rule #3 and has NO exceptions in true impulse waves.
    Even minor overlap (1%) invalidates the impulse pattern.

    References
    ----------
    Frost & Prechter (2005), p. 22: "Wave 4 may not overlap wave 1"
    """
    if len(prices) < 5:
        return {
            'valid': False,
            'confidence': 0.0,
            'error': 'insufficient_points',
            'has_overlap': False
        }

    try:
        # Calculate Wave 1 price territory
        wave_1_high = max(prices[0], prices[1])
        wave_1_low = min(prices[0], prices[1])
        wave_1_range = wave_1_high - wave_1_low

        # Check Wave 4 endpoint
        wave_4_end = prices[4]

        # Also check Wave 4's full range (conservative check)
        wave_4_low = min(prices[3], prices[4])
        wave_4_high = max(prices[3], prices[4])

        # Detect overlap
        has_overlap = (wave_4_low < wave_1_high) and (wave_4_high > wave_1_low)

        # Calculate overlap amount
        if has_overlap:
            overlap_start = max(wave_4_low, wave_1_low)
            overlap_end = min(wave_4_high, wave_1_high)
            overlap_amount = overlap_end - overlap_start
            overlap_pct = (overlap_amount / wave_1_range * 100) if wave_1_range > 0 else 0
        else:
            overlap_amount = 0
            overlap_pct = 0

        # CRITICAL: In impulse waves, ANY overlap is invalid
        if has_overlap and not is_diagonal:
            logger.warning(
                f"MANDATORY RULE VIOLATION: Wave 4 overlaps Wave 1 "
                f"({overlap_pct:.1f}% overlap, ${overlap_amount:.2f})"
            )

            return {
                'valid': False,
                'confidence': 0.0,
                'has_overlap': True,
                'error': 'MANDATORY_RULE_VIOLATION: Wave 4 overlaps Wave 1',
                'overlap_amount': overlap_amount,
                'overlap_percentage': overlap_pct,
                'wave_1_range': (wave_1_low, wave_1_high),
                'wave_4_end': wave_4_end,
                'is_diagonal': is_diagonal
            }

        # Diagonal triangles: Overlap is expected and acceptable
        if has_overlap and is_diagonal:
            logger.info(
                f"Wave 4 overlap detected but acceptable for diagonal triangle "
                f"({overlap_pct:.1f}% overlap)"
            )
            return {
                'valid': True,
                'confidence': 0.7,  # Lower confidence for diagonals
                'has_overlap': True,
                'overlap_amount': overlap_amount,
                'overlap_percentage': overlap_pct,
                'is_diagonal': True,
                'note': 'Overlap acceptable in diagonal triangles'
            }

        # No overlap: Pattern is valid
        logger.debug("Wave 4 overlap check passed - no overlap detected")
        return {
            'valid': True,
            'confidence': 1.0,
            'has_overlap': False,
            'overlap_amount': 0,
            'overlap_percentage': 0,
            'wave_1_range': (wave_1_low, wave_1_high),
            'wave_4_end': wave_4_end,
            'is_diagonal': is_diagonal
        }

    except Exception as e:
        logger.error(f"Error in Wave 4 overlap validation: {e}", exc_info=True)
        return {
            'valid': False,
            'confidence': 0.0,
            'error': f'validation_error: {str(e)}',
            'has_overlap': False
        }


# ==============================================================================
# CRITICAL FIX #2: Enhanced Wave 3 Length Validation
# ==============================================================================

def validate_wave_3_length_strict(waves: Dict[int, float]) -> Dict[str, Any]:
    """
    MANDATORY Wave 3 length validation - Rule #2.

    Wave 3 cannot be the shortest among waves 1, 3, and 5.
    Often, Wave 3 is the longest (most common scenario).

    Parameters
    ----------
    waves : Dict[int, float]
        Dictionary of wave numbers to wave lengths

    Returns
    -------
    Dict[str, Any]
        - valid: bool, True if Wave 3 is not shortest
        - confidence: float, higher if Wave 3 is longest
        - wave_3_length: float
        - wave_1_length: float
        - wave_5_length: float
        - wave_3_is_longest: bool
        - wave_3_is_shortest: bool
        - error: str, if mandatory rule violated

    Notes
    -----
    This is Elliott Wave Rule #2 and has NO exceptions.
    Wave 3 being the longest provides strongest confirmation (90% confidence).
    Wave 3 being middle length is acceptable (60% confidence).
    Wave 3 being shortest immediately invalidates the pattern.

    References
    ----------
    Frost & Prechter (2005), p. 22: "Wave 3 is never the shortest wave"
    """
    try:
        wave_1_len = abs(waves.get(1, 0))
        wave_3_len = abs(waves.get(3, 0))
        wave_5_len = abs(waves.get(5, 0))

        # Get all valid impulse wave lengths
        lengths = []
        if wave_1_len > 0:
            lengths.append(('Wave 1', wave_1_len))
        if wave_3_len > 0:
            lengths.append(('Wave 3', wave_3_len))
        if wave_5_len > 0:
            lengths.append(('Wave 5', wave_5_len))

        if len(lengths) < 2:
            return {
                'valid': False,
                'confidence': 0.0,
                'error': 'insufficient_waves',
                'wave_3_length': wave_3_len,
                'wave_1_length': wave_1_len,
                'wave_5_length': wave_5_len
            }

        # Sort lengths
        sorted_lengths = sorted(lengths, key=lambda x: x[1])
        shortest_wave = sorted_lengths[0][0]
        longest_wave = sorted_lengths[-1][0]

        wave_3_is_shortest = (shortest_wave == 'Wave 3')
        wave_3_is_longest = (longest_wave == 'Wave 3')

        # MANDATORY: Wave 3 cannot be shortest
        if wave_3_is_shortest:
            logger.warning(
                f"MANDATORY RULE VIOLATION: Wave 3 is shortest "
                f"(W1={wave_1_len:.2f}, W3={wave_3_len:.2f}, W5={wave_5_len:.2f})"
            )

            return {
                'valid': False,
                'confidence': 0.0,
                'error': 'MANDATORY_RULE_VIOLATION: Wave 3 is shortest',
                'wave_3_length': wave_3_len,
                'wave_1_length': wave_1_len,
                'wave_5_length': wave_5_len,
                'wave_3_is_shortest': True,
                'wave_3_is_longest': False,
                'shortest_wave': shortest_wave,
                'longest_wave': longest_wave
            }

        # Calculate confidence based on Wave 3 position
        if wave_3_is_longest:
            # Most common and ideal scenario
            confidence = 0.9
            note = "Wave 3 is longest (ideal Elliott pattern)"
            logger.debug(note)
        else:
            # Wave 3 is middle length (acceptable but less ideal)
            confidence = 0.6
            note = "Wave 3 is middle length (acceptable)"
            logger.debug(note)

        # Bonus confidence if Wave 3 is significantly longer
        if wave_3_is_longest:
            length_ratio = wave_3_len / max(wave_1_len, wave_5_len, 0.0001)
            if length_ratio > 1.618:  # Fibonacci extension
                confidence = min(confidence + 0.05, 0.95)
                note += f" with {length_ratio:.2f}x extension"

        return {
            'valid': True,
            'confidence': confidence,
            'wave_3_length': wave_3_len,
            'wave_1_length': wave_1_len,
            'wave_5_length': wave_5_len,
            'wave_3_is_shortest': False,
            'wave_3_is_longest': wave_3_is_longest,
            'shortest_wave': shortest_wave,
            'longest_wave': longest_wave,
            'length_ratio': wave_3_len / max(wave_1_len, wave_5_len, 0.0001),
            'note': note
        }

    except Exception as e:
        logger.error(f"Error in Wave 3 length validation: {e}", exc_info=True)
        return {
            'valid': False,
            'confidence': 0.0,
            'error': f'validation_error: {str(e)}'
        }


# ==============================================================================
# NEW FEATURE #3: Wave Subdivision Validation
# ==============================================================================

def validate_wave_subdivisions(df: pd.DataFrame, wave_points: np.ndarray,
                              column: str = 'close') -> Dict[str, Any]:
    """
    Validate that waves subdivide according to Elliott Wave principle.

    Elliott Rule: In an impulse:
    - Waves 1, 3, 5 (motive) must subdivide into 5 waves
    - Waves 2, 4 (corrective) must subdivide into 3 waves

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Array of wave point indices
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Subdivision analysis for each wave with validation

    Notes
    -----
    This is a guideline rather than a strict rule, as subdivision
    analysis depends on data resolution and timeframe.
    Confidence boost if subdivisions match expected structure.
    """
    from .peaks import detect_peaks_troughs_enhanced

    subdivisions = {}
    subdivision_confidence = 0.0

    try:
        for wave_num in range(1, 6):
            if wave_num >= len(wave_points):
                break

            start_idx = wave_points[wave_num - 1]
            end_idx = wave_points[wave_num]

            # Extract sub-wave data
            wave_df = df.iloc[start_idx:end_idx + 1]

            if len(wave_df) < 5:  # Not enough data for subdivision analysis
                continue

            # Detect sub-waves
            try:
                sub_peaks, sub_troughs = detect_peaks_troughs_enhanced(
                    wave_df, column
                )
                sub_wave_count = len(sub_peaks) + len(sub_troughs)
            except Exception as e:
                logger.debug(f"Could not detect sub-waves for Wave {wave_num}: {e}")
                sub_wave_count = 0

            # Determine expected subdivision
            if wave_num in [1, 3, 5]:  # Motive waves
                expected = 5
                wave_type = 'motive'
            else:  # Corrective waves (2, 4)
                expected = 3
                wave_type = 'corrective'

            # Validate subdivision count (with tolerance)
            is_valid = (expected - 1) <= sub_wave_count <= (expected + 2)

            subdivisions[f'wave_{wave_num}'] = {
                'expected_count': expected,
                'actual_count': sub_wave_count,
                'wave_type': wave_type,
                'valid': is_valid,
                'confidence': 0.2 if is_valid else 0.0
            }

            if is_valid:
                subdivision_confidence += 0.1  # Small boost per valid subdivision

        overall_valid = len([s for s in subdivisions.values() if s['valid']]) >= 3

        logger.debug(
            f"Subdivision validation: {len([s for s in subdivisions.values() if s['valid']])} "
            f"of {len(subdivisions)} waves have correct subdivision"
        )

        return {
            'valid': overall_valid,
            'confidence': min(subdivision_confidence, 0.3),  # Cap at 30%
            'subdivisions': subdivisions,
            'note': 'Subdivision analysis - guideline validation'
        }

    except Exception as e:
        logger.error(f"Error in subdivision validation: {e}", exc_info=True)
        return {
            'valid': True,  # Don't fail on subdivision error
            'confidence': 0.0,
            'error': str(e),
            'note': 'Subdivision analysis failed - skipped'
        }


# ==============================================================================
# NEW FEATURE #4: Enhanced Alternation with Sharp/Sideways Classification
# ==============================================================================

def classify_correction_type(df: pd.DataFrame, wave_start: int, wave_end: int,
                            prior_wave_end: int, column: str = 'close') -> CorrectionType:
    """
    Classify if correction is sharp (zigzag) or sideways (flat/triangle).

    Sharp corrections: Steep decline/rise, doesn't reach prior wave's end level
    Sideways corrections: Moves back to or beyond prior wave's end level

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_start : int
        Starting index of corrective wave
    wave_end : int
        Ending index of corrective wave
    prior_wave_end : int
        Ending index of prior wave
    column : str, default 'close'
        Price column name

    Returns
    -------
    CorrectionType
        SHARP or SIDEWAYS classification
    """
    try:
        wave_price_start = df[column].iloc[wave_start]
        wave_price_end = df[column].iloc[wave_end]
        prior_price_end = df[column].iloc[prior_wave_end]

        # Get wave range
        wave_segment = df[column].iloc[wave_start:wave_end + 1]
        wave_high = wave_segment.max()
        wave_low = wave_segment.min()

        # Determine if correction reaches back to prior wave level
        tolerance = 0.05  # 5% tolerance

        if wave_price_start > prior_price_end:  # Correction after uptrend
            # Check if correction low reaches or exceeds prior wave end
            if wave_low <= prior_price_end * (1 + tolerance):
                return CorrectionType.SIDEWAYS
        else:  # Correction after downtrend
            # Check if correction high reaches or exceeds prior wave end
            if wave_high >= prior_price_end * (1 - tolerance):
                return CorrectionType.SIDEWAYS

        return CorrectionType.SHARP

    except Exception as e:
        logger.debug(f"Error classifying correction type: {e}")
        return CorrectionType.UNDEFINED


def validate_alternation_principle_enhanced(df: pd.DataFrame, wave_points: np.ndarray,
                                           column: str = 'close') -> Dict[str, Any]:
    """
    Validate alternation principle with sharp vs sideways classification.

    Elliott Guideline: If Wave 2 is sharp (zigzag), Wave 4 should be sideways
    (flat/triangle), and vice versa.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave point indices
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Alternation analysis with correction type classification

    Notes
    -----
    This is a guideline rather than a mandatory rule.
    Provides confidence boost if alternation is present.
    """
    if len(wave_points) < 5:
        return {
            'valid': True,
            'confidence': 0.0,
            'note': 'Insufficient points for alternation analysis'
        }

    try:
        # Classify Wave 2 type
        wave2_type = classify_correction_type(
            df, wave_points[1], wave_points[2], wave_points[1], column
        )

        # Classify Wave 4 type
        wave4_type = classify_correction_type(
            df, wave_points[3], wave_points[4], wave_points[3], column
        )

        # Check alternation
        has_alternation = (
            wave2_type != wave4_type and
            wave2_type != CorrectionType.UNDEFINED and
            wave4_type != CorrectionType.UNDEFINED
        )

        confidence = 0.3 if has_alternation else 0.1

        logger.debug(
            f"Alternation analysis: Wave 2 is {wave2_type.value}, "
            f"Wave 4 is {wave4_type.value}, "
            f"alternation={'present' if has_alternation else 'absent'}"
        )

        return {
            'valid': True,  # Alternation is guideline, not mandatory
            'confidence': confidence,
            'has_alternation': has_alternation,
            'wave_2_type': wave2_type.value,
            'wave_4_type': wave4_type.value,
            'note': (
                'Alternation present - Wave 2 and 4 differ in type'
                if has_alternation else
                'Alternation absent - Wave 2 and 4 are similar'
            )
        }

    except Exception as e:
        logger.error(f"Error in alternation validation: {e}", exc_info=True)
        return {
            'valid': True,
            'confidence': 0.0,
            'error': str(e),
            'note': 'Alternation analysis failed'
        }


# ==============================================================================
# NEW FEATURE #5: Parallel Channel Analysis
# ==============================================================================

def calculate_parallel_channel(wave_points: np.ndarray,
                              prices: np.ndarray) -> Dict[str, Any]:
    """
    Calculate parallel channel for Elliott Wave pattern.

    Base line: Connects wave 0 (start) to wave 2 (end of Wave 2)
    Parallel line: Through wave 1 peak, parallel to base line

    Parameters
    ----------
    wave_points : np.ndarray
        Wave point indices
    prices : np.ndarray
        Prices at wave points

    Returns
    -------
    Dict[str, Any]
        Channel parameters (slopes, intercepts)
    """
    try:
        if len(wave_points) < 3 or len(prices) < 3:
            return {'valid': False, 'error': 'insufficient_points'}

        # Base trendline (through wave 0 and wave 2)
        x0, y0 = 0, prices[0]
        x2, y2 = wave_points[2] - wave_points[0], prices[2]

        # Calculate slope
        base_slope = (y2 - y0) / (x2 - x0) if x2 != x0 else 0
        base_intercept = y0

        # Parallel line through wave 1 peak
        x1, y1 = wave_points[1] - wave_points[0], prices[1]
        parallel_intercept = y1 - base_slope * x1

        return {
            'valid': True,
            'base_slope': base_slope,
            'base_intercept': base_intercept,
            'parallel_intercept': parallel_intercept,
            'channel_width': abs(parallel_intercept - base_intercept)
        }

    except Exception as e:
        logger.error(f"Error calculating parallel channel: {e}", exc_info=True)
        return {'valid': False, 'error': str(e)}


def validate_wave_5_channel_termination(wave_points: np.ndarray, prices: np.ndarray,
                                       tolerance: float = 0.10) -> Dict[str, Any]:
    """
    Check if Wave 5 terminates near the upper channel line.

    Elliott Guideline: Wave 5 often terminates at or near the upper channel.

    Parameters
    ----------
    wave_points : np.ndarray
        Wave point indices
    prices : np.ndarray
        Prices at wave points
    tolerance : float, default 0.10
        Tolerance for channel deviation (10%)

    Returns
    -------
    Dict[str, Any]
        Channel termination analysis
    """
    if len(wave_points) < 6 or len(prices) < 6:
        return {
            'valid': True,
            'confidence': 0.0,
            'note': 'Insufficient points for channel analysis'
        }

    try:
        channel = calculate_parallel_channel(wave_points, prices)

        if not channel['valid']:
            return {
                'valid': True,
                'confidence': 0.0,
                'note': 'Could not calculate channel'
            }

        # Calculate expected price at Wave 5's position
        x5 = wave_points[5] - wave_points[0]
        expected_price = channel['base_slope'] * x5 + channel['parallel_intercept']
        actual_price = prices[5]

        # Calculate deviation
        if expected_price == 0:
            deviation = 1.0
        else:
            deviation = abs(actual_price - expected_price) / abs(expected_price)

        terminates_at_channel = deviation <= tolerance
        confidence = max(0, 1 - deviation / tolerance) * 0.2  # Max 20% bonus

        logger.debug(
            f"Channel analysis: Wave 5 at {actual_price:.2f}, "
            f"channel expects {expected_price:.2f}, "
            f"deviation {deviation*100:.1f}%"
        )

        return {
            'valid': True,
            'confidence': confidence,
            'terminates_at_channel': terminates_at_channel,
            'deviation': deviation,
            'expected_price': expected_price,
            'actual_price': actual_price,
            'note': (
                'Wave 5 terminates near channel' if terminates_at_channel
                else f'Wave 5 deviates {deviation*100:.1f}% from channel'
            )
        }

    except Exception as e:
        logger.error(f"Error in channel termination validation: {e}", exc_info=True)
        return {
            'valid': True,
            'confidence': 0.0,
            'error': str(e)
        }


# ==============================================================================
# MAIN VALIDATION FUNCTION - Enhanced with All Fixes
# ==============================================================================

def validate_impulse_wave_strict(df: pd.DataFrame, wave_points: np.ndarray,
                                column: str = 'close',
                                is_diagonal: bool = False) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Comprehensive Elliott Wave validation with MANDATORY rule enforcement.

    This function implements strict Elliott Wave validation including:
    1. MANDATORY Wave 4 overlap check (Rule #3)
    2. MANDATORY Wave 3 length check (Rule #2)
    3. Wave subdivision validation (guideline)
    4. Enhanced alternation principle
    5. Parallel channel analysis

    Parameters
    ----------
    df : pd.DataFrame
        Price data with datetime index
    wave_points : np.ndarray
        Array of wave point indices
    column : str, default 'close'
        Price column name
    is_diagonal : bool, default False
        True if pattern is diagonal triangle (allows overlap)

    Returns
    -------
    Tuple[bool, float, Dict[str, Any]]
        - is_valid: bool, True if all MANDATORY rules pass
        - confidence: float [0, 1], overall confidence score
        - details: Dict with detailed validation results

    Notes
    -----
    Mandatory rules (must pass):
    - Wave 4 cannot overlap Wave 1 (except diagonals)
    - Wave 3 cannot be shortest

    Guidelines (boost confidence):
    - Wave subdivisions match expected structure
    - Alternation between Wave 2 and 4
    - Wave 5 terminates near channel

    Examples
    --------
    >>> is_valid, conf, details = validate_impulse_wave_strict(df, wave_points)
    >>> if not is_valid:
    ...     print(f"Failed: {details['mandatory_failures']}")
    """
    validation_results = {
        'mandatory_failures': [],
        'guideline_results': {},
        'is_diagonal': is_diagonal
    }

    if len(wave_points) < 6:
        return False, 0.0, {
            **validation_results,
            'error': 'insufficient_points',
            'note': 'Need at least 6 points for full impulse wave (start + 5 waves)'
        }

    try:
        prices = df[column].iloc[wave_points].values
        waves = {i+1: prices[i+1] - prices[i] for i in range(min(len(prices)-1, 5))}

        confidence = 0.0

        # ===== MANDATORY RULE #1: Wave 4 Overlap =====
        overlap_result = validate_wave_4_overlap_mandatory(prices, is_diagonal)
        validation_results['wave_4_overlap'] = overlap_result

        if not overlap_result['valid']:
            validation_results['mandatory_failures'].append('wave_4_overlap')
            logger.warning(f"FAILED: {overlap_result['error']}")
            return False, 0.0, validation_results

        confidence += overlap_result['confidence'] * 0.25  # 25% weight

        # ===== MANDATORY RULE #2: Wave 3 Length =====
        wave3_result = validate_wave_3_length_strict(waves)
        validation_results['wave_3_length'] = wave3_result

        if not wave3_result['valid']:
            validation_results['mandatory_failures'].append('wave_3_shortest')
            logger.warning(f"FAILED: {wave3_result['error']}")
            return False, 0.0, validation_results

        confidence += wave3_result['confidence'] * 0.25  # 25% weight

        # ===== GUIDELINE: Wave Subdivisions =====
        subdivision_result = validate_wave_subdivisions(df, wave_points, column)
        validation_results['guideline_results']['subdivisions'] = subdivision_result
        confidence += subdivision_result['confidence'] * 0.15  # 15% weight

        # ===== GUIDELINE: Alternation Principle =====
        alternation_result = validate_alternation_principle_enhanced(
            df, wave_points, column
        )
        validation_results['guideline_results']['alternation'] = alternation_result
        confidence += alternation_result['confidence'] * 0.15  # 15% weight

        # ===== GUIDELINE: Channel Analysis =====
        channel_result = validate_wave_5_channel_termination(wave_points, prices)
        validation_results['guideline_results']['channel'] = channel_result
        confidence += channel_result['confidence'] * 0.20  # 20% weight

        # Final confidence calculation
        final_confidence = min(confidence, 0.95)  # Cap at 95%
        is_valid = final_confidence >= 0.4  # Minimum threshold

        validation_results.update({
            'final_confidence': final_confidence,
            'is_valid': is_valid,
            'mandatory_rules_passed': len(validation_results['mandatory_failures']) == 0,
            'note': (
                'All mandatory Elliott Wave rules passed' if is_valid
                else 'Pattern fails Elliott Wave mandatory rules'
            )
        })

        logger.info(
            f"Elliott Wave validation complete: valid={is_valid}, "
            f"confidence={final_confidence:.2f}, "
            f"failures={validation_results['mandatory_failures']}"
        )

        return is_valid, final_confidence, validation_results

    except Exception as e:
        logger.error(f"Error in strict impulse validation: {e}", exc_info=True)
        return False, 0.0, {
            **validation_results,
            'error': f'validation_error: {str(e)}'
        }
