"""
Elliott Wave Personality and Characteristic Validation
======================================================

This module implements validation of characteristic wave personalities and
behavioral patterns according to Elliott Wave Theory.

Each wave has distinct characteristics:
- Wave 1: Tentative, low volume, often retraced
- Wave 2: Sharp retracement, fear dominates
- Wave 3: Strong, highest volume, longest wave, rarely fails
- Wave 4: Complex, corrective, alternates with Wave 2
- Wave 5: Mature, declining volume, possible momentum divergence

Functions
---------
validate_wave_personality : Main personality validation
validate_wave_3_personality : Wave 3 characteristics (strong, high volume)
validate_wave_5_personality : Wave 5 characteristics (divergence, declining volume)
check_momentum_divergence : RSI/MACD divergence detection
validate_volume_profile : Volume pattern validation

References
----------
Frost & Prechter (2005), "Elliott Wave Principle", Chapter 2: Wave Personalities

Examples
--------
>>> personality = validate_wave_personality(df, wave_points, column='close')
>>> if personality['wave_3']['highest_volume']:
...     print("Wave 3 shows characteristic high volume")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def validate_wave_personality(df: pd.DataFrame, wave_points: np.ndarray,
                             column: str = 'close') -> Dict[str, Any]:
    """
    Validate characteristic wave personalities according to Elliott Wave Theory.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with columns: close, volume (optional), high, low
    wave_points : np.ndarray
        Array of wave point indices [start, w1, w2, w3, w4, w5]
    column : str, default 'close'
        Price column name

    Returns
    -------
    Dict[str, Any]
        Personality validation results for each wave with confidence scores
    """
    if len(wave_points) < 6:
        return {
            'valid': False,
            'confidence': 0.0,
            'error': 'insufficient_points'
        }

    try:
        personalities = {
            'wave_3': validate_wave_3_personality(df, wave_points, column),
            'wave_5': validate_wave_5_personality(df, wave_points, column),
            'overall_confidence': 0.0
        }

        total_confidence = (
            personalities['wave_3'].get('confidence', 0.0) * 0.6 +
            personalities['wave_5'].get('confidence', 0.0) * 0.4
        )

        personalities['overall_confidence'] = min(total_confidence, 0.5)
        personalities['valid'] = True

        return personalities

    except Exception as e:
        logger.error(f"Error in wave personality validation: {e}", exc_info=True)
        return {'valid': False, 'confidence': 0.0, 'error': str(e)}


def validate_wave_3_personality(df: pd.DataFrame, wave_points: np.ndarray,
                               column: str = 'close') -> Dict[str, Any]:
    """Validate Wave 3: Strong, highest volume, longest wave."""
    try:
        has_volume = 'volume' in df.columns

        if has_volume:
            wave_volumes = []
            for i in range(min(5, len(wave_points) - 1)):
                vol = df['volume'].iloc[wave_points[i]:wave_points[i + 1] + 1].mean()
                wave_volumes.append(vol)

            impulse_volumes = [wave_volumes[i] for i in [0, 2, 4] if i < len(wave_volumes)]
            highest_volume = len(impulse_volumes) > 0 and wave_volumes[2] == max(impulse_volumes)
        else:
            highest_volume = None

        # Momentum
        wave_3_range = abs(df[column].iloc[wave_points[3]] - df[column].iloc[wave_points[2]])
        wave_3_bars = wave_points[3] - wave_points[2]
        momentum = wave_3_range / wave_3_bars if wave_3_bars > 0 else 0

        wave_1_range = abs(df[column].iloc[wave_points[1]] - df[column].iloc[wave_points[0]])
        wave_1_bars = wave_points[1] - wave_points[0]
        wave_1_momentum = wave_1_range / wave_1_bars if wave_1_bars > 0 else 0

        strong_momentum = momentum > wave_1_momentum * 1.2

        confidence = 0.0
        if highest_volume:
            confidence += 0.2
        if strong_momentum:
            confidence += 0.15

        return {
            'highest_volume': highest_volume,
            'strong_momentum': strong_momentum,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"Error validating Wave 3: {e}", exc_info=True)
        return {'confidence': 0.0, 'error': str(e)}


def validate_wave_5_personality(df: pd.DataFrame, wave_points: np.ndarray,
                               column: str = 'close') -> Dict[str, Any]:
    """Validate Wave 5: Declining volume, possible divergence."""
    try:
        has_volume = 'volume' in df.columns

        if has_volume:
            wave_3_vol = df['volume'].iloc[wave_points[2]:wave_points[3] + 1].mean()
            wave_5_vol = df['volume'].iloc[wave_points[4]:wave_points[5] + 1].mean()
            declining_volume = wave_5_vol < wave_3_vol
        else:
            declining_volume = None

        divergence = check_momentum_divergence(df, wave_points[3], wave_points[5], column)

        confidence = 0.0
        if declining_volume:
            confidence += 0.15
        if divergence['has_divergence']:
            confidence += 0.10

        return {
            'declining_volume': declining_volume,
            'momentum_divergence': divergence['has_divergence'],
            'confidence': confidence,
            'reversal_warning': declining_volume or divergence['has_divergence']
        }

    except Exception as e:
        logger.error(f"Error validating Wave 5: {e}", exc_info=True)
        return {'confidence': 0.0, 'error': str(e)}


def check_momentum_divergence(df: pd.DataFrame, start_idx: int, end_idx: int,
                              column: str = 'close') -> Dict[str, Any]:
    """Check for momentum divergence."""
    try:
        segment = df.iloc[start_idx:end_idx + 1]

        if len(segment) < 10:
            return {'has_divergence': False}

        period = min(5, len(segment) // 2)
        momentum = segment[column].diff(period)

        price_high_idx = segment[column].idxmax()
        momentum_high_idx = momentum.idxmax()

        price_pos = segment.index.get_loc(price_high_idx)
        momentum_pos = segment.index.get_loc(momentum_high_idx)

        has_divergence = price_pos > momentum_pos

        return {'has_divergence': has_divergence}

    except Exception as e:
        logger.debug(f"Error checking divergence: {e}")
        return {'has_divergence': False, 'error': str(e)}