"""
Momentum & Velocity Indicators for Elliott Wave Confirmation.

These indicators measure the SPEED and STRENGTH of price moves,
complementing Elliott Wave's shape-based analysis:

- RSI: Overbought/oversold + divergence detection
- MACD: Momentum direction and acceleration
- ADX: Trend strength (strong vs choppy)
- ATR Regime: Volatility state (calm/normal/explosive)
- Rate of Change: Direct velocity measurement

Each returns a standardized signal dict that can gate entries/exits.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# RSI — Relative Strength Index
# ──────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average of gains/losses."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def rsi_signal(df: pd.DataFrame, column: str = 'close',
               period: int = 14) -> Dict[str, Any]:
    """
    Generate RSI-based signals.

    Returns:
        zone: 'oversold' (<30), 'overbought' (>70), 'neutral'
        value: current RSI value
        divergence: 'bullish_div', 'bearish_div', or None
        strength: -1.0 to +1.0 (negative = bearish, positive = bullish)
    """
    prices = df[column]
    rsi = compute_rsi(prices, period)
    current_rsi = float(rsi.iloc[-1])

    # Zone
    if current_rsi < 30:
        zone = 'oversold'
    elif current_rsi > 70:
        zone = 'overbought'
    else:
        zone = 'neutral'

    # Divergence detection (last 20 bars)
    divergence = _detect_rsi_divergence(prices, rsi, lookback=20)

    # Strength: map RSI to -1..+1 (50 = 0, 30 = -1, 70 = +1)
    strength = (current_rsi - 50) / 20
    strength = max(-1.0, min(1.0, strength))

    return {
        'indicator': 'rsi',
        'value': round(current_rsi, 1),
        'zone': zone,
        'divergence': divergence,
        'strength': round(strength, 3),
    }


def _detect_rsi_divergence(prices: pd.Series, rsi: pd.Series,
                           lookback: int = 20) -> Optional[str]:
    """
    Detect RSI divergence in the last N bars.

    Bullish divergence: price makes lower low, RSI makes higher low
    Bearish divergence: price makes higher high, RSI makes lower high
    """
    if len(prices) < lookback + 5:
        return None

    recent_prices = prices.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]
    prior_prices = prices.iloc[-lookback*2:-lookback] if len(prices) >= lookback*2 else prices.iloc[:lookback]
    prior_rsi = rsi.iloc[-lookback*2:-lookback] if len(rsi) >= lookback*2 else rsi.iloc[:lookback]

    if len(prior_prices) < 5:
        return None

    # Bullish divergence: price lower low, RSI higher low
    if (recent_prices.min() < prior_prices.min() and
            recent_rsi.min() > prior_rsi.min()):
        return 'bullish_div'

    # Bearish divergence: price higher high, RSI lower high
    if (recent_prices.max() > prior_prices.max() and
            recent_rsi.max() < prior_rsi.max()):
        return 'bearish_div'

    return None


# ──────────────────────────────────────────────────────────────────────
# MACD — Moving Average Convergence Divergence
# ──────────────────────────────────────────────────────────────────────

def compute_macd(prices: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_signal(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Generate MACD-based signals.

    Returns:
        crossover: 'bullish_cross', 'bearish_cross', or None (within last 3 bars)
        histogram_direction: 'expanding_up', 'expanding_down', 'contracting'
        momentum: 'accelerating', 'decelerating', 'flat'
        strength: -1.0 to +1.0
    """
    prices = df[column]
    if len(prices) < 30:
        return {'indicator': 'macd', 'crossover': None,
                'histogram_direction': 'flat', 'momentum': 'flat', 'strength': 0}

    macd_line, signal_line, histogram = compute_macd(prices)

    current_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    # Crossover detection (last 3 bars)
    crossover = None
    for i in range(-3, 0):
        if i-1 >= -len(macd_line):
            m_now = float(macd_line.iloc[i])
            m_prev = float(macd_line.iloc[i-1])
            s_now = float(signal_line.iloc[i])
            s_prev = float(signal_line.iloc[i-1])
            if m_prev <= s_prev and m_now > s_now:
                crossover = 'bullish_cross'
            elif m_prev >= s_prev and m_now < s_now:
                crossover = 'bearish_cross'

    # Histogram direction
    if current_hist > 0 and current_hist > prev_hist:
        hist_dir = 'expanding_up'
    elif current_hist < 0 and current_hist < prev_hist:
        hist_dir = 'expanding_down'
    else:
        hist_dir = 'contracting'

    # Momentum acceleration
    if len(histogram) >= 3:
        h1 = float(histogram.iloc[-1])
        h2 = float(histogram.iloc[-2])
        h3 = float(histogram.iloc[-3])
        accel = (h1 - h2) - (h2 - h3)
        if abs(accel) < 0.001 * abs(prices.iloc[-1]):
            momentum = 'flat'
        elif (h1 > 0 and accel > 0) or (h1 < 0 and accel < 0):
            momentum = 'accelerating'
        else:
            momentum = 'decelerating'
    else:
        momentum = 'flat'

    # Strength: normalize histogram relative to price
    price_scale = float(prices.iloc[-1]) * 0.02  # 2% of price as reference
    strength = current_hist / price_scale if price_scale > 0 else 0
    strength = max(-1.0, min(1.0, strength))

    return {
        'indicator': 'macd',
        'crossover': crossover,
        'histogram_direction': hist_dir,
        'momentum': momentum,
        'strength': round(strength, 3),
        'histogram': round(current_hist, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# ADX — Average Directional Index (Trend Strength)
# ──────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX (trend strength indicator)."""
    high = df['high'] if 'high' in df.columns else df['close']
    low = df['low'] if 'low' in df.columns else df['close']
    close = df['close']

    # True Range
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                        index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                         index=df.index)

    # Smoothed averages
    atr = tr.ewm(span=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() /
                     atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() /
                      atr.replace(0, np.nan))

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period).mean()

    return adx.fillna(0)


def adx_signal(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """
    Generate ADX-based trend strength signals.

    Returns:
        value: current ADX value
        regime: 'strong_trend' (>25), 'weak_trend' (15-25), 'no_trend' (<15)
        trending: bool — is the market trending enough to trade?
        strength: 0.0 to 1.0
    """
    if len(df) < period * 2:
        return {'indicator': 'adx', 'value': 0, 'regime': 'no_trend',
                'trending': False, 'strength': 0}

    adx = compute_adx(df, period)
    current_adx = float(adx.iloc[-1])

    if current_adx > 25:
        regime = 'strong_trend'
    elif current_adx > 15:
        regime = 'weak_trend'
    else:
        regime = 'no_trend'

    return {
        'indicator': 'adx',
        'value': round(current_adx, 1),
        'regime': regime,
        'trending': current_adx > 20,
        'strength': round(min(current_adx / 40, 1.0), 3),
    }


# ──────────────────────────────────────────────────────────────────────
# ATR Regime — Volatility State Classification
# ──────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df['high'] if 'high' in df.columns else df['close']
    low = df['low'] if 'low' in df.columns else df['close']
    close = df['close']

    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)

    return tr.ewm(span=period, min_periods=period).mean()


def atr_regime_signal(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """
    Classify current volatility regime using ATR percentile.

    Returns:
        atr_pct: ATR as percentage of price
        regime: 'calm' (<1%), 'normal' (1-3%), 'volatile' (3-5%), 'explosive' (>5%)
        percentile: where current ATR sits vs last 100 bars
        stop_multiplier: suggested stop distance multiplier
    """
    if len(df) < period + 10:
        return {'indicator': 'atr_regime', 'atr_pct': 0, 'regime': 'normal',
                'percentile': 50, 'stop_multiplier': 1.0}

    atr = compute_atr(df, period)
    current_atr = float(atr.iloc[-1])
    current_price = float(df['close'].iloc[-1])
    atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0

    # Percentile vs recent history
    lookback = min(100, len(atr) - 1)
    recent_atrs = atr.iloc[-lookback:]
    percentile = float((recent_atrs < current_atr).mean() * 100)

    # Regime classification
    if atr_pct < 1.0:
        regime = 'calm'
        stop_mult = 0.7  # Tighter stops in calm markets
    elif atr_pct < 3.0:
        regime = 'normal'
        stop_mult = 1.0
    elif atr_pct < 5.0:
        regime = 'volatile'
        stop_mult = 1.5  # Wider stops to avoid noise
    else:
        regime = 'explosive'
        stop_mult = 2.0  # Much wider stops, reduce size

    return {
        'indicator': 'atr_regime',
        'atr_pct': round(atr_pct, 2),
        'atr_value': round(current_atr, 4),
        'regime': regime,
        'percentile': round(percentile, 0),
        'stop_multiplier': stop_mult,
    }


# ──────────────────────────────────────────────────────────────────────
# Rate of Change — Direct Velocity Measurement
# ──────────────────────────────────────────────────────────────────────

def compute_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change: percentage change over N periods."""
    return ((prices / prices.shift(period)) - 1) * 100


def velocity_signal(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Multi-timeframe velocity measurement.

    Returns:
        velocity_5d: 5-day rate of change
        velocity_20d: 20-day rate of change
        acceleration: is velocity increasing or decreasing?
        speed_regime: 'crash' / 'fast_down' / 'slow_down' / 'flat' /
                      'slow_up' / 'fast_up' / 'melt_up'
    """
    prices = df[column]
    if len(prices) < 25:
        return {'indicator': 'velocity', 'velocity_5d': 0, 'velocity_20d': 0,
                'acceleration': 'flat', 'speed_regime': 'flat'}

    roc_5 = float(compute_roc(prices, 5).iloc[-1])
    roc_20 = float(compute_roc(prices, 20).iloc[-1])

    # Acceleration: compare recent vs prior 5-day ROC
    roc_5_series = compute_roc(prices, 5)
    if len(roc_5_series) >= 6:
        accel = float(roc_5_series.iloc[-1]) - float(roc_5_series.iloc[-6])
        if accel > 1:
            acceleration = 'accelerating_up'
        elif accel < -1:
            acceleration = 'accelerating_down'
        else:
            acceleration = 'flat'
    else:
        acceleration = 'flat'

    # Speed regime classification
    if roc_20 < -15:
        speed_regime = 'crash'
    elif roc_20 < -5:
        speed_regime = 'fast_down'
    elif roc_20 < -1:
        speed_regime = 'slow_down'
    elif roc_20 < 1:
        speed_regime = 'flat'
    elif roc_20 < 5:
        speed_regime = 'slow_up'
    elif roc_20 < 15:
        speed_regime = 'fast_up'
    else:
        speed_regime = 'melt_up'

    return {
        'indicator': 'velocity',
        'velocity_5d': round(roc_5, 2),
        'velocity_20d': round(roc_20, 2),
        'acceleration': acceleration,
        'speed_regime': speed_regime,
    }


# ──────────────────────────────────────────────────────────────────────
# Composite Momentum Score — Unified Signal
# ──────────────────────────────────────────────────────────────────────

def compute_momentum_composite(df: pd.DataFrame,
                                column: str = 'close') -> Dict[str, Any]:
    """
    Compute all momentum indicators and produce a unified assessment.

    Returns a dict with:
        - Individual indicator signals (rsi, macd, adx, atr_regime, velocity)
        - composite_score: -1.0 (max bearish) to +1.0 (max bullish)
        - entry_ok: bool — momentum confirms a BUY entry
        - exit_warning: bool — momentum suggests taking profits or exiting
        - regime: overall market regime description
        - confidence_adjustment: multiply pattern confidence by this (0.5 - 1.3)
    """
    rsi = rsi_signal(df, column)
    macd = macd_signal(df, column)
    adx = adx_signal(df)
    atr = atr_regime_signal(df)
    vel = velocity_signal(df, column)

    # Composite scoring
    score = 0.0
    weights_used = 0.0

    # RSI contribution (weight: 0.25)
    rsi_score = rsi['strength']
    if rsi['divergence'] == 'bullish_div':
        rsi_score += 0.3
    elif rsi['divergence'] == 'bearish_div':
        rsi_score -= 0.3
    score += rsi_score * 0.25
    weights_used += 0.25

    # MACD contribution (weight: 0.30)
    macd_score = macd['strength']
    if macd['crossover'] == 'bullish_cross':
        macd_score += 0.4
    elif macd['crossover'] == 'bearish_cross':
        macd_score -= 0.4
    if macd['momentum'] == 'accelerating':
        macd_score *= 1.2
    elif macd['momentum'] == 'decelerating':
        macd_score *= 0.7
    score += max(-1, min(1, macd_score)) * 0.30
    weights_used += 0.30

    # ADX contribution (weight: 0.15) — trend strength, not direction
    # Strong trend = trust the wave pattern more
    adx_contribution = adx['strength'] * 0.5  # Always positive
    score += adx_contribution * 0.15
    weights_used += 0.15

    # Velocity contribution (weight: 0.30)
    vel_5 = vel['velocity_5d']
    vel_20 = vel['velocity_20d']
    vel_score = 0
    if vel_5 > 0 and vel_20 > 0:
        vel_score = min(vel_20 / 10, 1.0)  # Cap at 1
    elif vel_5 < 0 and vel_20 < 0:
        vel_score = max(vel_20 / 10, -1.0)
    else:
        vel_score = vel_5 / 20  # Conflicting short/long term
    score += vel_score * 0.30
    weights_used += 0.30

    composite = score / weights_used if weights_used > 0 else 0
    composite = max(-1.0, min(1.0, composite))

    # Decision gates
    entry_ok = (
        composite > 0.1 and
        rsi['zone'] != 'overbought' and
        macd['crossover'] != 'bearish_cross' and
        vel['speed_regime'] not in ('crash', 'fast_down') and
        adx['trending']
    )

    exit_warning = (
        composite < -0.1 or
        rsi['divergence'] == 'bearish_div' or
        rsi['zone'] == 'overbought' or
        macd['crossover'] == 'bearish_cross' or
        vel['speed_regime'] in ('crash', 'fast_down') or
        (vel['acceleration'] == 'accelerating_down' and vel['velocity_5d'] < -3)
    )

    # Confidence adjustment factor
    if composite > 0.3 and adx['trending']:
        conf_adj = 1.2  # Strong momentum confirms pattern
    elif composite > 0.1:
        conf_adj = 1.0  # Neutral
    elif composite > -0.1:
        conf_adj = 0.8  # Weak momentum
    elif vel['speed_regime'] in ('crash', 'fast_down'):
        conf_adj = 0.3  # Crash — heavily discount any buy signal
    else:
        conf_adj = 0.6  # Bearish momentum

    # Regime description
    if vel['speed_regime'] == 'crash':
        regime = 'CRASH — avoid all entries'
    elif vel['speed_regime'] == 'melt_up':
        regime = 'MELT-UP — trail stops tight, expect reversal'
    elif adx['regime'] == 'no_trend':
        regime = 'CHOPPY — reduce size, widen stops'
    elif adx['regime'] == 'strong_trend' and composite > 0.2:
        regime = 'STRONG BULLISH — ideal for wave trading'
    elif adx['regime'] == 'strong_trend' and composite < -0.2:
        regime = 'STRONG BEARISH — only short or stay flat'
    elif atr['regime'] == 'explosive':
        regime = 'HIGH VOLATILITY — reduce size, widen stops'
    else:
        regime = 'NORMAL — standard parameters'

    return {
        'rsi': rsi,
        'macd': macd,
        'adx': adx,
        'atr_regime': atr,
        'velocity': vel,
        'composite_score': round(composite, 3),
        'entry_ok': entry_ok,
        'exit_warning': exit_warning,
        'confidence_adjustment': conf_adj,
        'regime': regime,
        'stop_multiplier': atr['stop_multiplier'],
    }
