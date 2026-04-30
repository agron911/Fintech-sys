"""
Market Regime Detection

Classifies market conditions to adapt trading strategy parameters:
- Trend strength via ADX
- Volatility regime via ATR percentile
- Combined regime for strategy selection
"""
import numpy as np
import pandas as pd
from typing import Dict


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    high = df['high'] if 'high' in df.columns else df['close']
    low = df['low'] if 'low' in df.columns else df['close']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period).mean()
    return adx


def detect_regime(df: pd.DataFrame, atr_period: int = 14, adx_period: int = 14) -> Dict:
    """
    Detect current market regime.

    Returns:
        Dict with:
        - trend_strength: 'strong' (ADX>25), 'moderate' (15-25), 'weak' (<15)
        - volatility: 'high', 'normal', 'low' based on ATR percentile
        - regime: combined label
        - adx: current ADX value
        - atr_pct: current ATR as % of price
        - strategy_hint: suggested strategy approach
    """
    if len(df) < max(atr_period, adx_period) * 3:
        return {
            'trend_strength': 'unknown',
            'volatility': 'unknown',
            'regime': 'insufficient_data',
            'adx': 0, 'atr_pct': 0,
            'strategy_hint': 'default',
        }

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close

    # ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.ewm(span=atr_period, min_periods=atr_period).mean()

    current_atr = atr.iloc[-1]
    current_price = close.iloc[-1]
    atr_pct = current_atr / current_price if current_price > 0 else 0

    # ATR percentile over last 100 bars
    lookback = min(100, len(atr))
    atr_percentile = (atr.iloc[-lookback:] < current_atr).mean()

    # Volatility regime
    if atr_percentile > 0.8:
        volatility = 'high'
    elif atr_percentile < 0.2:
        volatility = 'low'
    else:
        volatility = 'normal'

    # ADX
    adx = compute_adx(df, period=adx_period)
    current_adx = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0

    if current_adx > 25:
        trend_strength = 'strong'
    elif current_adx > 15:
        trend_strength = 'moderate'
    else:
        trend_strength = 'weak'

    # Combined regime and strategy hint
    if trend_strength == 'strong' and volatility != 'high':
        regime = 'trending'
        strategy_hint = 'breakout'
    elif trend_strength == 'weak':
        regime = 'ranging'
        strategy_hint = 'mean_reversion'
    elif volatility == 'high':
        regime = 'volatile'
        strategy_hint = 'reduce_size'
    else:
        regime = 'normal'
        strategy_hint = 'default'

    return {
        'trend_strength': trend_strength,
        'volatility': volatility,
        'regime': regime,
        'adx': float(current_adx),
        'atr_pct': float(atr_pct),
        'atr_percentile': float(atr_percentile),
        'strategy_hint': strategy_hint,
    }
