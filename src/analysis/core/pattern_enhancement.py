"""
Elliott Wave Pattern Enhancement and Volume Analysis
====================================================

This module provides enhancement functions for Elliott Wave patterns and
specialized volume analysis for diagonal triangles and other patterns.

Functions
---------
enhance_wave_detection_with_complex_patterns : Enhance wave detection with complex patterns
analyze_diagonal_subwaves : Analyze 3-wave substructure in diagonals
analyze_diagonal_volume_pattern : Analyze volume patterns in diagonal triangles
analyze_trendline_convergence : Analyze trendline convergence and project apex
project_diagonal_completion : Project completion of diagonal pattern
determine_wave_position : Determine if pattern is leading or ending diagonal

Examples
--------
>>> from pattern_enhancement import enhance_wave_detection_with_complex_patterns
>>> enhanced = enhance_wave_detection_with_complex_patterns(df, wave_data, column='close')
>>> if enhanced.get('wave_type') == 'diagonal_ending':
...     print(f"Reversal warning: {enhanced.get('reversal_probability')}")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

from .pattern_detection import detect_complex_elliott_patterns

logger = logging.getLogger(__name__)


def enhance_wave_detection_with_complex_patterns(df: pd.DataFrame,
                                               wave_data: Dict[str, Any],
                                               column: str = 'close') -> Dict[str, Any]:
    """
    Enhance existing wave detection with complex pattern recognition.

    Call this after regular wave detection to boost confidence and add
    pattern-specific information.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    wave_data : Dict[str, Any]
        Basic wave detection results
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Enhanced wave data with additional fields:
        - complex_patterns: Dict of detected complex patterns
        - confidence: Boosted confidence score
        - wave_type: Enhanced wave type (may include 'diagonal_leading/ending')
        - special_rules: Notes about special pattern rules
        - reversal_warning: bool, true if truncated fifth detected
        - reversal_probability: float, probability of reversal
        - extended_wave: int, which wave is extended
        - extension_ratio: float, extension ratio

    Notes
    -----
    Pattern-specific enhancements:
    - Diagonal triangles: Add wave_type suffix and special overlap rules
    - Truncated fifths: Add reversal warnings
    - Extended waves: Identify which wave and extension ratio
    Confidence is boosted by up to 20% based on pattern detection.

    Examples
    --------
    >>> wave_data = {'impulse_wave': np.array([0, 5, 10, 20, 25, 40]),
    ...              'confidence': 0.6}
    >>> enhanced = enhance_wave_detection_with_complex_patterns(df, wave_data)
    >>> print(f"Enhanced confidence: {enhanced['confidence']:.2f}")
    """
    try:
        # Detect complex patterns
        complex_patterns = detect_complex_elliott_patterns(df, wave_data, column)

        # Add to wave data
        wave_data['complex_patterns'] = complex_patterns

        # Adjust confidence based on pattern recognition
        if complex_patterns['pattern_confidence'] > 0:
            # Boost overall confidence if special patterns detected
            original_confidence = wave_data.get('confidence', 0.0)
            pattern_boost = complex_patterns['pattern_confidence'] * 0.2
            wave_data['confidence'] = min(original_confidence + pattern_boost, 0.95)

            logger.info(f"Enhanced confidence from {original_confidence:.2f} to "
                       f"{wave_data['confidence']:.2f}")

            # Add pattern-specific information
            if complex_patterns['diagonal_triangle']:
                diagonal = complex_patterns['diagonal_triangle']
                wave_data['wave_type'] = f"diagonal_{diagonal['type']}"
                wave_data['special_rules'] = 'Wave 4 overlap allowed in diagonal'
                logger.info(f"Identified {diagonal['type']} diagonal triangle")

            if complex_patterns['truncated_fifth']:
                wave_data['reversal_warning'] = True
                wave_data['reversal_probability'] = complex_patterns['truncated_fifth']['confidence']
                logger.warning(f"Truncated fifth detected - reversal probability: "
                             f"{wave_data['reversal_probability']:.2f}")

            if complex_patterns['extended_wave']:
                extended_num = complex_patterns['extended_wave']['extended_wave']
                wave_data['extended_wave'] = extended_num
                wave_data['extension_ratio'] = complex_patterns['extended_wave']['extension_ratio']
                logger.info(f"Wave {extended_num} is extended with ratio "
                          f"{wave_data['extension_ratio']:.2f}")

    except Exception as e:
        logger.error(f"Error enhancing wave detection: {e}", exc_info=True)

    return wave_data


def analyze_trendline_convergence(upper_points: List[Tuple[int, float]],
                                lower_points: List[Tuple[int, float]]) -> Dict[str, Any]:
    """
    Analyze if trendlines are converging and project apex.

    Parameters
    ----------
    upper_points : List[Tuple[int, float]]
        List of (index, price) tuples for upper trendline
    lower_points : List[Tuple[int, float]]
        List of (index, price) tuples for lower trendline

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - converging: bool, True if trendlines converge
        - angle: float, angle between lines in degrees
        - upper_slope: float, slope of upper trendline
        - lower_slope: float, slope of lower trendline
        - apex_projection: Tuple[int, float] or None, (index, price) of apex

    Notes
    -----
    Convergence indicates diagonal triangle pattern.
    Apex projection useful for completion timing estimates.
    """
    if len(upper_points) < 2 or len(lower_points) < 2:
        return {'converging': False}

    try:
        # Calculate slopes
        upper_slope = (
            (upper_points[-1][1] - upper_points[0][1]) /
            (upper_points[-1][0] - upper_points[0][0])
        )
        lower_slope = (
            (lower_points[-1][1] - lower_points[0][1]) /
            (lower_points[-1][0] - lower_points[0][0])
        )

        # Check if converging (slopes getting closer)
        converging = abs(upper_slope - lower_slope) > 0.001

        # Calculate angle between lines
        angle = np.arctan(abs(upper_slope - lower_slope)) * 180 / np.pi

        # Project apex (intersection point)
        apex_projection = None
        if converging and upper_slope != lower_slope:
            # Line equations: y = mx + b
            # Find b for each line
            b_upper = upper_points[0][1] - upper_slope * upper_points[0][0]
            b_lower = lower_points[0][1] - lower_slope * lower_points[0][0]

            # Find intersection
            x_intersect = (b_lower - b_upper) / (upper_slope - lower_slope)
            y_intersect = upper_slope * x_intersect + b_upper

            apex_projection = (int(x_intersect), y_intersect)

        return {
            'converging': converging,
            'angle': angle,
            'upper_slope': upper_slope,
            'lower_slope': lower_slope,
            'apex_projection': apex_projection
        }

    except ZeroDivisionError:
        logger.warning("Zero division in trendline convergence calculation")
        return {'converging': False}
    except Exception as e:
        logger.error(f"Error analyzing trendline convergence: {e}", exc_info=True)
        return {'converging': False}


def analyze_diagonal_subwaves(df: pd.DataFrame, wave_points: np.ndarray,
                            column: str = 'close') -> Dict[str, Any]:
    """
    Analyze if waves have 3-wave substructure (characteristic of diagonals).

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
        Dictionary containing:
        - is_three_wave_structure: bool
        - wave_structures: Dict with structure info for each wave

    Notes
    -----
    In diagonal triangles, all waves have 3-wave (corrective) substructure,
    unlike regular impulse patterns where waves 1, 3, 5 have 5-wave structure.
    This is a key distinguishing feature.
    """
    result = {
        'is_three_wave_structure': False,
        'wave_structures': {}
    }

    try:
        # Check each wave for 3-wave structure
        three_wave_count = 0
        for i in range(len(wave_points) - 1):
            start_idx = wave_points[i]
            end_idx = wave_points[i + 1]

            # Look for ABC pattern within this wave
            segment = df[column].iloc[start_idx:end_idx+1]

            # Simplified check - in practice would be more sophisticated
            if len(segment) >= 3:
                # Check if middle portion retraces
                mid_point = len(segment) // 2

                if i % 2 == 0:  # Impulse wave (1, 3, 5)
                    # Should still have 3-wave structure in diagonal
                    if segment.iloc[mid_point] < segment.iloc[-1]:
                        three_wave_count += 1
                else:  # Corrective wave (2, 4)
                    if segment.iloc[mid_point] > segment.iloc[0]:
                        three_wave_count += 1

        # Diagonals should have mostly 3-wave structures
        if three_wave_count >= 3:
            result['is_three_wave_structure'] = True
            logger.debug(f"Detected 3-wave structure in {three_wave_count} waves")

    except Exception as e:
        logger.warning(f"Error analyzing diagonal subwaves: {e}")

    return result


def project_diagonal_completion(df: pd.DataFrame, wave_points: np.ndarray,
                              prices: np.ndarray, convergence_data: Dict[str, Any],
                              column: str = 'close') -> Dict[str, Any]:
    """
    Project the completion of a diagonal pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Price data
    wave_points : np.ndarray
        Wave point indices
    prices : np.ndarray
        Price values at wave points
    convergence_data : Dict[str, Any]
        Trendline convergence analysis results
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - target_price: float or None, projected completion price
        - target_date: pd.Timestamp or None, projected completion date
        - confidence: float [0, 1], projection confidence

    Notes
    -----
    Diagonals typically complete at 70-80% of the way to apex.
    Uses trendline projections for price targets.
    """
    projection = {
        'target_price': None,
        'target_date': None,
        'confidence': 0.0
    }

    try:
        if convergence_data.get('apex_projection'):
            apex_x, apex_y = convergence_data['apex_projection']
            current_idx = len(df) - 1

            # Diagonals typically complete at 70-80% of the way to apex
            completion_ratio = 0.75

            # Calculate target
            distance_to_apex = apex_x - current_idx
            target_idx = current_idx + int(distance_to_apex * completion_ratio)

            # Price projection based on trendline
            current_position = len(wave_points)
            if current_position % 2 == 0:  # Next is a down wave
                # Use lower trendline
                slope = convergence_data['lower_slope']
            else:  # Next is an up wave
                # Use upper trendline
                slope = convergence_data['upper_slope']

            # Project price
            price_change = slope * (target_idx - current_idx)
            target_price = df[column].iloc[-1] + price_change

            projection['target_price'] = target_price
            projection['target_date'] = (
                df.index[-1] + pd.Timedelta(days=target_idx - current_idx)
            )
            projection['confidence'] = 0.7

            logger.info(f"Projected diagonal completion: price={target_price:.2f}, "
                       f"date={projection['target_date']}")

    except Exception as e:
        logger.warning(f"Error projecting diagonal completion: {e}")

    return projection


def determine_wave_position(df: pd.DataFrame, wave_start_idx: int) -> str:
    """
    Determine if pattern is in wave 1/A position (leading) or wave 5/C position (ending).

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'close' column
    wave_start_idx : int
        Starting index of wave pattern

    Returns
    -------
    str
        'leading' or 'ending'

    Notes
    -----
    Uses simple heuristic: strong trend before pattern suggests ending diagonal,
    weak/no trend suggests leading diagonal.
    Requires at least 20 bars of history for reliable determination.
    """
    # Look at broader context before this pattern
    lookback = min(wave_start_idx, 100)

    if lookback < 20:
        return "leading"  # Not enough history, assume leading

    try:
        # Simple heuristic - if there's a strong trend before this, it's likely ending
        prior_trend = df['close'].iloc[wave_start_idx-lookback:wave_start_idx]
        trend_strength = (prior_trend.iloc[-1] - prior_trend.iloc[0]) / prior_trend.iloc[0]

        if abs(trend_strength) > 0.5:  # Strong prior trend (>50% move)
            return "ending"
        else:
            return "leading"

    except (KeyError, IndexError) as e:
        logger.warning(f"Error determining wave position: {e}")
        return "leading"


def analyze_diagonal_volume_pattern(df: pd.DataFrame,
                                  wave_points: np.ndarray) -> Dict[str, Any]:
    """
    Analyze volume patterns in diagonal triangles.

    Diagonals typically show declining volume in the direction of the trend
    and increasing volume in the corrective waves.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'volume' column
    wave_points : np.ndarray
        Wave point indices

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - valid: bool, whether volume pattern is valid for diagonal
        - volume_trend: str or None, description of trend
        - confidence: float [0, 1], confidence in volume pattern
        - characteristics: Dict with detailed volume metrics

    Notes
    -----
    Valid diagonal volume patterns show:
    1. Declining volume in impulse waves (1, 3, 5)
    2. Increasing volume in corrective waves (2, 4)
    3. Overall declining volume pattern

    Returns {'valid': True} if no volume data available (skip validation).
    """
    result = {
        'valid': False,
        'volume_trend': None,
        'confidence': 0.0,
        'characteristics': {}
    }

    if 'volume' not in df.columns or len(wave_points) < 5:
        return {'valid': True}  # Skip volume analysis if no volume data

    try:
        # Calculate average volume for each wave
        wave_volumes = []
        for i in range(len(wave_points) - 1):
            start_idx = wave_points[i]
            end_idx = wave_points[i + 1]
            wave_volume = df['volume'].iloc[start_idx:end_idx+1].mean()
            wave_volumes.append(wave_volume)

        if len(wave_volumes) < 4:
            return {'valid': True}

        # Check volume pattern in impulse waves (1, 3, 5)
        impulse_volumes = [wave_volumes[i] for i in [0, 2, 4] if i < len(wave_volumes)]
        corrective_volumes = [wave_volumes[i] for i in [1, 3] if i < len(wave_volumes)]

        # Calculate volume trends
        impulse_trend = np.polyfit(range(len(impulse_volumes)), impulse_volumes, 1)[0]
        corrective_trend = np.polyfit(range(len(corrective_volumes)), corrective_volumes, 1)[0]

        # Diagonals typically show:
        # 1. Declining volume in impulse waves
        # 2. Increasing volume in corrective waves
        # 3. Overall declining volume pattern

        characteristics = {
            'impulse_volume_trend': 'declining' if impulse_trend < 0 else 'increasing',
            'corrective_volume_trend': 'increasing' if corrective_trend > 0 else 'declining',
            'impulse_volumes': impulse_volumes,
            'corrective_volumes': corrective_volumes
        }

        # Calculate confidence based on volume patterns
        confidence = 0.0

        # Check if impulse waves show declining volume
        if impulse_trend < 0:
            confidence += 0.4
            characteristics['impulse_volume_valid'] = True
        else:
            characteristics['impulse_volume_valid'] = False

        # Check if corrective waves show increasing volume
        if corrective_trend > 0:
            confidence += 0.3
            characteristics['corrective_volume_valid'] = True
        else:
            characteristics['corrective_volume_valid'] = False

        # Check overall volume pattern
        if len(wave_volumes) >= 3:
            overall_trend = np.polyfit(range(len(wave_volumes)), wave_volumes, 1)[0]
            if overall_trend < 0:
                confidence += 0.3
                characteristics['overall_volume_valid'] = True
            else:
                characteristics['overall_volume_valid'] = False

        result['valid'] = confidence >= 0.6
        result['confidence'] = confidence
        result['characteristics'] = characteristics

        if result['valid']:
            logger.info(f"Valid diagonal volume pattern detected, confidence={confidence:.2f}")
        else:
            logger.debug(f"Diagonal volume pattern not confirmed, confidence={confidence:.2f}")

    except Exception as e:
        logger.warning(f"Error analyzing diagonal volume pattern: {e}")
        return {'valid': True}  # Don't invalidate pattern due to volume error

    return result
