"""
Pattern Adapter: Bridges Elliott Wave analysis output to strategy input format.

The analysis engine (impulse.py) outputs wave data with indices, confidence scores,
and cross-timeframe relationships. The strategies expect a different format with
price-based position data, alignment scores, and Fibonacci levels.

This adapter transforms one format to the other so the backtester can actually
generate trades.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps
_momentum = None
def _get_momentum():
    global _momentum
    if _momentum is None:
        from src.analysis.core.momentum_indicators import compute_momentum_composite
        _momentum = compute_momentum_composite
    return _momentum


def adapt_wave_data_to_strategy_input(df: pd.DataFrame,
                                      wave_data: Dict[str, Any],
                                      column: str = 'close') -> Dict[str, Any]:
    """
    Transform Elliott Wave analysis output into strategy-compatible format.

    Args:
        df: Price DataFrame with OHLCV data
        wave_data: Output from find_elliott_wave_pattern_enhanced()
        column: Price column name

    Returns:
        Dict compatible with strategy.generate_signals() pattern_analysis parameter
    """
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    confidence = wave_data.get('confidence', 0.0)
    relationships = wave_data.get('pattern_relationships', {})

    # Extract alignment score from cross-timeframe analysis
    alignment_score = relationships.get('alignment_score', 0.5)

    # Build current_position from wave points
    current_position = _build_current_position(df, impulse_wave, column)

    # Build best_pattern in the format expected by composite confidence filter
    best_pattern = _build_best_pattern(wave_data)

    # Compute broad trend context using SMA50/SMA200
    trend_context = _compute_trend_context(df, column)

    # Penalize confidence for counter-trend patterns
    adjusted_confidence = confidence
    if current_position and trend_context['trend'] != 'neutral':
        pattern_dir = current_position.get('trend_direction', 'up')
        if trend_context['trend'] == 'bullish' and pattern_dir == 'down':
            # Bearish pattern in a bullish trend — likely a correction, not a trade setup
            adjusted_confidence *= 0.4
        elif trend_context['trend'] == 'bearish' and pattern_dir == 'up':
            # Bullish pattern in a bearish trend — likely a bear rally
            adjusted_confidence *= 0.5

    # Compute momentum/velocity signals
    try:
        momentum_composite = _get_momentum()(df, column)
    except Exception:
        momentum_composite = {
            'composite_score': 0, 'entry_ok': True, 'exit_warning': False,
            'confidence_adjustment': 1.0, 'regime': 'UNKNOWN', 'stop_multiplier': 1.0,
        }

    # Apply momentum-based confidence adjustment
    adjusted_confidence *= momentum_composite.get('confidence_adjustment', 1.0)
    adjusted_confidence = min(adjusted_confidence, 1.0)

    return {
        'alignment_score': alignment_score,
        'current_position': current_position,
        'best_pattern': best_pattern,
        'confidence': adjusted_confidence,
        'raw_confidence': confidence,
        'trend_context': trend_context,
        'momentum': momentum_composite,
        'wave_data': wave_data,  # preserve original for advanced use
    }


def _compute_trend_context(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Determine broad market trend using SMA crossover.

    Returns dict with:
        trend: 'bullish', 'bearish', or 'neutral'
        sma50: current SMA50 value
        sma200: current SMA200 value
        price_vs_sma200: ratio of current price to SMA200
    """
    prices = df[column]
    result = {'trend': 'neutral', 'sma50': None, 'sma200': None, 'price_vs_sma200': 1.0}

    if len(prices) < 50:
        return result

    sma50 = prices.rolling(50).mean().iloc[-1]
    result['sma50'] = float(sma50)

    if len(prices) >= 200:
        sma200 = prices.rolling(200).mean().iloc[-1]
        result['sma200'] = float(sma200)
        current_price = float(prices.iloc[-1])
        result['price_vs_sma200'] = current_price / sma200 if sma200 > 0 else 1.0

        if sma50 > sma200 and current_price > sma200:
            result['trend'] = 'bullish'
        elif sma50 < sma200 and current_price < sma200:
            result['trend'] = 'bearish'
    else:
        # With only SMA50, compare price position
        current_price = float(prices.iloc[-1])
        if current_price > sma50 * 1.05:
            result['trend'] = 'bullish'
        elif current_price < sma50 * 0.95:
            result['trend'] = 'bearish'

    return result


def _build_current_position(df: pd.DataFrame,
                            impulse_wave: np.ndarray,
                            column: str) -> Optional[Dict[str, Any]]:
    """
    Determine current wave position and extract price levels.

    Maps the impulse_wave index array to actual prices and determines
    which wave the current price is most likely in.

    Wave point mapping (5-wave impulse):
        impulse_wave[0] = Wave 1 start
        impulse_wave[1] = Wave 1 end / Wave 2 start
        impulse_wave[2] = Wave 2 end / Wave 3 start
        impulse_wave[3] = Wave 3 end / Wave 4 start
        impulse_wave[4] = Wave 4 end / Wave 5 start
        impulse_wave[5] = Wave 5 end (if present)
    """
    if len(impulse_wave) < 3:
        return None

    # Clamp indices to valid range
    max_idx = len(df) - 1
    wave_indices = np.clip(impulse_wave, 0, max_idx).astype(int)
    wave_prices = df[column].iloc[wave_indices].values
    current_price = df[column].iloc[-1]
    current_idx = len(df) - 1

    # Determine trend direction (up or down impulse)
    trend_up = wave_prices[-1] > wave_prices[0]

    # Base position info
    position = {
        'wave_indices': wave_indices.tolist(),
        'wave_prices': wave_prices.tolist(),
        'trend_direction': 'up' if trend_up else 'down',
    }

    # Extract key price levels depending on how many wave points we have
    n_points = len(wave_indices)

    if n_points >= 2:
        # Wave 1 data
        position['wave_1_start'] = float(wave_prices[0])
        position['wave_1_end'] = float(wave_prices[1])
        position['wave_1_high'] = float(max(wave_prices[0], wave_prices[1]))
        position['wave_1_low'] = float(min(wave_prices[0], wave_prices[1]))
        position['wave_1_range'] = abs(float(wave_prices[1] - wave_prices[0]))

        # Previous wave high/low for Fibonacci calculations
        position['previous_wave_high'] = float(max(wave_prices[:2]))
        position['previous_wave_low'] = float(min(wave_prices[:2]))

    if n_points >= 3:
        position['wave_2_end'] = float(wave_prices[2])
        position['wave_2_low'] = float(wave_prices[2]) if trend_up else float(wave_prices[2])

    if n_points >= 4:
        position['wave_3_end'] = float(wave_prices[3])
        position['wave_3_high'] = float(wave_prices[3]) if trend_up else float(wave_prices[1])

    if n_points >= 5:
        position['wave_4_end'] = float(wave_prices[4])

    if n_points >= 6:
        position['wave_5_end'] = float(wave_prices[5])

    # Determine current wave number based on where current price/index sits
    position['wave_number'] = _determine_current_wave(
        current_idx, current_price, wave_indices, wave_prices, trend_up
    )

    # Compute impulse range metrics for FibonacciMeanReversion strategy
    if n_points >= 4:
        if trend_up:
            position['impulse_high'] = float(max(wave_prices[:4]))
            position['impulse_low'] = float(min(wave_prices[:4]))
        else:
            position['impulse_high'] = float(max(wave_prices[:4]))
            position['impulse_low'] = float(min(wave_prices[:4]))

        # Average volume during impulse waves (1 and 3)
        try:
            vol_col = 'volume' if 'volume' in df.columns else 'Volume'
            if vol_col in df.columns:
                w1_vol = df[vol_col].iloc[wave_indices[0]:wave_indices[1]+1].mean()
                w3_start = wave_indices[2] if n_points >= 3 else wave_indices[1]
                w3_end = wave_indices[3] if n_points >= 4 else wave_indices[2]
                w3_vol = df[vol_col].iloc[w3_start:w3_end+1].mean()
                position['impulse_avg_volume'] = float((w1_vol + w3_vol) / 2)
        except Exception:
            pass

    return position


def _determine_current_wave(current_idx: int,
                            current_price: float,
                            wave_indices: np.ndarray,
                            wave_prices: np.ndarray,
                            trend_up: bool) -> int:
    """
    Determine which wave the current bar is most likely in.

    Returns wave number (1-5 for impulse, or 6 for post-impulse/correction).
    """
    n_points = len(wave_indices)
    last_wave_idx = wave_indices[-1]

    # If current index is beyond the last detected wave point,
    # we're in or past the final detected wave
    if current_idx > last_wave_idx:
        days_beyond = current_idx - last_wave_idx
        if n_points >= 6:
            # Complete 5-wave structure detected, likely in correction
            return 6  # Post-impulse / correction phase
        elif n_points == 5:
            return 5  # In Wave 5
        elif n_points == 4:
            return 5 if days_beyond > 5 else 4  # Transitioning from 4 to 5
        elif n_points == 3:
            return 4 if days_beyond > 5 else 3
        elif n_points == 2:
            return 3 if days_beyond > 5 else 2
        else:
            return 1

    # Current index is within the detected wave structure
    for i in range(n_points - 1):
        if wave_indices[i] <= current_idx <= wave_indices[i + 1]:
            return i + 1  # Wave number (1-indexed)

    return n_points  # Default to last wave


def _build_best_pattern(wave_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build best_pattern dict in the format expected by the composite
    confidence filter in MultiTimeframeAlignmentStrategy.

    The filter expects:
        best_pattern['composite_confidence'] -> float
        best_pattern['base_pattern']['pattern']['points'] -> list/array
    """
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    confidence = wave_data.get('confidence', 0.0)

    # Check if wave_data already has structured composite patterns
    multiple_patterns = wave_data.get('multiple_patterns', [])
    if multiple_patterns and isinstance(multiple_patterns[0], dict):
        # Use the first (highest confidence) composite pattern
        first = multiple_patterns[0]
        composite_conf = first.get('composite_confidence', confidence)
        base_pattern = first.get('base_pattern', {})
        pattern = base_pattern.get('pattern', {})
        points = pattern.get('points', impulse_wave)

        return {
            'composite_confidence': composite_conf,
            'base_pattern': {
                'pattern': {
                    'points': points,
                    'confidence': pattern.get('confidence', confidence),
                    'wave_type': pattern.get('wave_type',
                                             wave_data.get('wave_type', 'unknown')),
                }
            }
        }

    # Fallback: construct from flat wave_data
    return {
        'composite_confidence': confidence,
        'base_pattern': {
            'pattern': {
                'points': impulse_wave,
                'confidence': confidence,
                'wave_type': wave_data.get('wave_type', 'unknown'),
            }
        }
    }


def build_bar_by_bar_positions(df: pd.DataFrame,
                               wave_data: Dict[str, Any],
                               column: str = 'close') -> pd.Series:
    """
    Build a Series mapping each bar index to its wave number.

    This enables strategies to know which wave each bar belongs to,
    rather than relying on a single static 'current_position'.

    Returns:
        pd.Series with index matching df.index, values are wave numbers (1-5, 6=correction)
    """
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    wave_positions = pd.Series(0, index=df.index, dtype=int)

    if len(impulse_wave) < 2:
        return wave_positions

    max_idx = len(df) - 1
    wave_indices = np.clip(impulse_wave, 0, max_idx).astype(int)

    # Assign wave numbers to ranges
    for i in range(len(wave_indices) - 1):
        start = wave_indices[i]
        end = wave_indices[i + 1]
        wave_num = i + 1  # Wave 1, 2, 3, 4, 5
        wave_positions.iloc[start:end + 1] = wave_num

    # Beyond last wave point
    last_idx = wave_indices[-1]
    if last_idx < max_idx:
        if len(wave_indices) >= 6:
            wave_positions.iloc[last_idx + 1:] = 6  # Correction
        else:
            wave_positions.iloc[last_idx + 1:] = len(wave_indices)

    return wave_positions
