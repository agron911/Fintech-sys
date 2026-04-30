"""
Signal scoring and action classification for stock analysis.

Extracted from scripts/what_to_buy.py so both CLI and GUI can share
the same analysis → score → classify pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_best_impulse_wave
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input
from src.analysis.core.volume import validate_volume_patterns
from src.analysis.core.wave_personality import validate_wave_personality

# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------
SCORING_CONFIG = {
    'min_confidence': 0.20,
    'wave_c_rsi_max': 40,
    'wave_scores': {1: 15, 2: 25, 3: 30, 4: 18, 5: 5},
    'trend_bullish_pts': 15,
    'pattern_up_pts': 10,
    'momentum_confirmed_pts': 15,
    'no_exit_warning_pts': 5,
    'confidence_max_pts': 10,
    'confidence_scale': 20,
    'rr_tiers': [(3.0, 15), (2.0, 12), (1.5, 8), (1.0, 4)],
    'rsi_oversold_threshold': 35,
    'rsi_oversold_pts': 5,
    'rsi_overbought_threshold': 75,
    'rsi_overbought_pts': -5,
    'macd_bullish_pts': 5,
    'rsi_bullish_div_pts': 5,
    'strong_buy_score': 75,
    'buy_score': 65,
    'buy_dip_score': 55,
    'watch_score': 45,
    'watch_min_rr': 2.0,
    'watch_min_conf': 0.25,
    # Tier 1: Momentum quality factors
    'momentum_quality_max_pts': 10,
    'macd_expanding_pts': 5,
    'macd_contracting_pts': -3,
    'vel_accel_up_pts': 5,
    'vel_decel_pts': -3,
    'strong_trend_pts': 5,
    'no_trend_pts': -10,
    'calm_atr_pts': 3,
    'explosive_atr_pts': -5,
    'sma200_overextend_pts': -5,
    'sma200_overextend_threshold': 1.30,
    'regime_reduce_pts': -5,
    'regime_weak_volatile_pts': -8,
    # Tier 2: Volume + personality factors
    'volume_max_pts': 10,
    'volume_disconfirm_pts': -5,
    'volume_disconfirm_threshold': 0.2,
    'w3_personality_pts': 5,
    'w5_exhaustion_pts': -10,
    'personality_max_pts': 5,
    'vol_surge_pts': 3,
    'vol_surge_threshold': 1.5,
    'vol_low_pts': -3,
    'vol_low_threshold': 0.5,
}


def analyze_stock(symbol: str, df: pd.DataFrame) -> dict | None:
    """Run full analysis on one stock. Returns result dict or None."""
    if len(df) < 60:
        return None

    if len(df) > 500:
        df = df.iloc[-500:]

    peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
    if len(peaks) < 3 or len(troughs) < 3:
        return None

    best = find_best_impulse_wave(df, peaks, troughs, column='close')
    if best.get('wave_type') in ('no_candidates', 'no_pattern'):
        return None

    wave_data = {
        'impulse_wave': best.get('wave_points', np.array([])),
        'confidence': best.get('confidence', 0),
        'wave_type': best.get('wave_type', 'unknown'),
        'pattern_relationships': {},
        'multiple_patterns': [],
    }

    pa = adapt_wave_data_to_strategy_input(df, wave_data, column='close')
    tc = pa.get('trend_context', {})
    pos = pa.get('current_position', {})
    mom = pa.get('momentum', {})

    if not pos:
        return None

    price = float(df['close'].iloc[-1])
    wave = pos.get('wave_number', 0)
    wave_type = best.get('wave_type', 'unknown')
    trend = tc.get('trend', 'neutral')
    pdir = pos.get('trend_direction', 'unknown')
    conf = pa.get('confidence', 0)

    entry_ok = mom.get('entry_ok', False)
    exit_warn = mom.get('exit_warning', False)
    composite = mom.get('composite_score', 0)
    regime = mom.get('regime', '?')
    stop_mult = mom.get('stop_multiplier', 1.0)

    rsi = mom.get('rsi', {})
    macd = mom.get('macd', {})
    adx = mom.get('adx', {})
    vel = mom.get('velocity', {})

    # Tier 1: Deep momentum fields (already computed by pattern_adapter)
    atr_data = mom.get('atr_regime', {})
    atr_pct = atr_data.get('atr_pct', 0)
    atr_value = atr_data.get('atr_value', 0)
    atr_regime_name = atr_data.get('regime', 'normal')
    atr_percentile = atr_data.get('percentile', 50)

    macd_hist_dir = macd.get('histogram_direction', 'flat')
    macd_strength = macd.get('strength', 0)
    vel_accel = vel.get('acceleration', 'flat')
    rsi_strength = rsi.get('strength', 0)
    adx_strength = adx.get('strength', 0)
    adx_regime_name = adx.get('regime', 'unknown')

    price_vs_sma200 = tc.get('price_vs_sma200', 1.0)
    sma50_val = tc.get('sma50')
    sma200_val = tc.get('sma200')

    regime_data = pa.get('regime', {})
    regime_strategy = regime_data.get('strategy_hint', 'default')
    regime_trend_str = regime_data.get('trend_strength', 'unknown')
    regime_vol = regime_data.get('volatility', 'unknown')

    # Tier 2: Volume confirmation
    wave_pts = best.get('wave_points', np.array([]))
    vol_score = validate_volume_patterns(df, wave_pts)

    # Wave personality validation
    personality = validate_wave_personality(df, wave_pts, column='close')
    w3_highest_vol = (personality.get('wave_3', {}).get('highest_volume', False)
                      if personality.get('valid') else False)
    w5_reversal_warn = (personality.get('wave_5', {}).get('reversal_warning', False)
                        if personality.get('valid') else False)
    personality_conf = (personality.get('overall_confidence', 0)
                        if personality.get('valid') else 0)

    # Current volume vs average
    if 'volume' in df.columns and len(df) >= 60:
        recent_vol = float(df['volume'].iloc[-5:].mean())
        avg_vol = float(df['volume'].iloc[-60:].mean())
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    w2_low = pos.get('wave_2_low', price * 0.95)
    w1_range = pos.get('wave_1_range', price * 0.05)
    w2_end = pos.get('wave_2_end', price)

    impulse_high = pos.get('impulse_high', None)
    impulse_low = pos.get('impulse_low', None)
    fib_retrace = None
    if impulse_high and impulse_low and impulse_high > impulse_low:
        imp_range = impulse_high - impulse_low
        fib_382 = impulse_high - imp_range * 0.382
        fib_500 = impulse_high - imp_range * 0.500
        fib_618 = impulse_high - imp_range * 0.618
        for level, name in [(fib_382, '38.2%'), (fib_500, '50%'), (fib_618, '61.8%')]:
            if abs(price - level) / price < 0.03:
                fib_retrace = name
                break

    stop = w2_low * 0.98 * stop_mult + price * (1 - stop_mult) if w2_low else price * 0.95
    target1 = w2_end + w1_range * 1.618 if w1_range > 0 else price * 1.10
    target2 = w2_end + w1_range * 2.618 if w1_range > 0 else price * 1.20
    risk = abs(price - stop)
    reward = abs(target1 - price)
    rr = reward / risk if risk > 0 else 0

    # Tier 3: ATR trailing stop
    trailing_stop = price - atr_value * 2.5 if atr_value > 0 else stop

    # Position sizing hint
    wave_size_map = {1: 0.6, 2: 0.7, 3: 1.0, 4: 0.3, 5: 0.5}
    size_mult = wave_size_map.get(wave, 0.5)
    wave_size_notes = {
        1: 'Wave 1: establishing trend',
        2: 'Wave 2: high prob entry',
        3: 'Wave 3: full size',
        4: 'Wave 4: reduce size',
        5: 'Wave 5: exhaustion risk',
    }
    size_note = wave_size_notes.get(wave, 'Unknown wave')

    p6m = float(df['close'].iloc[-min(126, len(df))])
    mom_6m = (price / p6m - 1) * 100

    return {
        'symbol': symbol,
        'price': price,
        'wave': wave,
        'wave_type': wave_type,
        'trend': trend,
        'pdir': pdir,
        'conf': conf,
        'entry_ok': entry_ok,
        'exit_warn': exit_warn,
        'composite': composite,
        'regime': regime,
        'rsi': rsi.get('value', 50),
        'rsi_zone': rsi.get('zone', '?'),
        'rsi_div': rsi.get('divergence', None),
        'macd_cross': macd.get('crossover', None),
        'macd_mom': macd.get('momentum', '?'),
        'adx': adx.get('value', 0),
        'adx_trending': adx.get('trending', False),
        'vel_5d': vel.get('velocity_5d', 0),
        'vel_20d': vel.get('velocity_20d', 0),
        'speed': vel.get('speed_regime', '?'),
        'mom_6m': mom_6m,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'rr': rr,
        'fib_retrace': fib_retrace,
        # Tier 1: Deep momentum
        'macd_hist_dir': macd_hist_dir,
        'macd_strength': macd_strength,
        'vel_accel': vel_accel,
        'rsi_strength': rsi_strength,
        'adx_strength': adx_strength,
        'adx_regime': adx_regime_name,
        'atr_pct': atr_pct,
        'atr_value': atr_value,
        'atr_regime': atr_regime_name,
        'atr_percentile': atr_percentile,
        'price_vs_sma200': price_vs_sma200,
        'sma50': sma50_val,
        'sma200': sma200_val,
        'regime_strategy': regime_strategy,
        'regime_trend_str': regime_trend_str,
        'regime_vol': regime_vol,
        # Tier 2: Volume + personality
        'volume_score': vol_score,
        'vol_ratio': vol_ratio,
        'w3_highest_vol': w3_highest_vol,
        'w5_reversal_warn': w5_reversal_warn,
        'personality_conf': personality_conf,
        # Tier 3: Position sizing + trailing stop
        'trailing_stop': trailing_stop,
        'size_mult': size_mult,
        'size_note': size_note,
    }


def classify_action(r: dict) -> tuple[str, str]:
    """Classify into BUY / WATCH / HOLD / AVOID. Returns (action, reason).

    Also sets r['score'] as a side effect.
    """
    cfg = SCORING_CONFIG
    wave = r['wave']
    trend = r['trend']
    pdir = r['pdir']
    entry_ok = r['entry_ok']
    exit_warn = r['exit_warn']
    conf = r['conf']

    if trend == 'bearish':
        r['score'] = 0
        return 'AVOID', 'Bearish trend'
    if exit_warn and wave >= 5:
        r['score'] = 0
        return 'EXIT', f'Wave {wave} + exit warning'
    if conf < cfg['min_confidence']:
        r['score'] = 0
        return 'SKIP', f'Low confidence ({conf * 100:.0f}%)'

    fib_retrace = r.get('fib_retrace')
    if (fib_retrace and wave in [5, 6] and trend == 'bullish'
            and r['rsi'] < cfg['wave_c_rsi_max']
            and r.get('macd_cross') == 'bullish'
            and conf >= cfg['min_confidence']):
        r['score'] = 70
        return 'BUY CORRECTION', f'Wave C complete at {fib_retrace} Fib retracement'

    if pdir == 'down' and trend == 'bullish':
        r['score'] = 0
        return 'AVOID', 'Counter-trend pattern'

    score = 0
    factors = []

    wave_pts = cfg['wave_scores'].get(wave, 0)
    score += wave_pts
    if wave_pts > 0:
        factors.append(f'Wave {wave} (+{wave_pts})')

    if trend == 'bullish':
        pts = cfg['trend_bullish_pts']
        score += pts
        factors.append(f'Bullish trend (+{pts})')

    if pdir == 'up':
        pts = cfg['pattern_up_pts']
        score += pts
        factors.append(f'Pattern up (+{pts})')

    if entry_ok:
        pts = cfg['momentum_confirmed_pts']
        score += pts
        factors.append(f'Momentum confirmed (+{pts})')

    if not exit_warn:
        score += cfg['no_exit_warning_pts']

    conf_pts = min(cfg['confidence_max_pts'], int(conf * cfg['confidence_scale']))
    score += conf_pts

    rr = r['rr']
    rr_pts = 0
    for min_rr, pts in cfg['rr_tiers']:
        if rr >= min_rr:
            rr_pts = pts
            break
    score += rr_pts
    if rr_pts > 0:
        factors.append(f'R:R {rr:.1f} (+{rr_pts})')

    rsi_val = r['rsi']
    if rsi_val < cfg['rsi_oversold_threshold']:
        score += cfg['rsi_oversold_pts']
        factors.append(f'Oversold RSI (+{cfg["rsi_oversold_pts"]})')
    elif rsi_val > cfg['rsi_overbought_threshold']:
        score += cfg['rsi_overbought_pts']
        factors.append(f'Overbought RSI ({cfg["rsi_overbought_pts"]})')

    if r.get('macd_cross') == 'bullish':
        pts = cfg['macd_bullish_pts']
        score += pts
        factors.append(f'MACD bullish (+{pts})')

    if r.get('rsi_div') == 'bullish_divergence':
        pts = cfg['rsi_bullish_div_pts']
        score += pts
        factors.append(f'RSI bullish div (+{pts})')

    # --- Tier 1: Momentum quality factors ---
    composite_val = r.get('composite', 0)
    mom_quality_pts = int(max(0, composite_val) * cfg['momentum_quality_max_pts'])
    if mom_quality_pts > 0:
        score += mom_quality_pts
        factors.append(f'Momentum quality (+{mom_quality_pts})')

    macd_hist = r.get('macd_hist_dir', 'flat')
    if macd_hist == 'expanding_up':
        pts = cfg['macd_expanding_pts']
        score += pts
        factors.append(f'MACD expanding (+{pts})')
    elif macd_hist in ('contracting', 'expanding_down'):
        score += cfg['macd_contracting_pts']

    vel_acc = r.get('vel_accel', 'flat')
    if vel_acc == 'accelerating_up':
        pts = cfg['vel_accel_up_pts']
        score += pts
        factors.append(f'Vel accelerating (+{pts})')
    elif vel_acc == 'decelerating':
        score += cfg['vel_decel_pts']

    adx_reg = r.get('adx_regime', 'unknown')
    if adx_reg == 'strong_trend':
        pts = cfg['strong_trend_pts']
        score += pts
        factors.append(f'Strong trend (+{pts})')
    elif adx_reg == 'no_trend':
        score += cfg['no_trend_pts']
        factors.append(f'No trend ({cfg["no_trend_pts"]})')

    atr_reg = r.get('atr_regime', 'normal')
    if atr_reg == 'calm':
        score += cfg['calm_atr_pts']
    elif atr_reg == 'explosive':
        score += cfg['explosive_atr_pts']
        factors.append(f'Explosive vol ({cfg["explosive_atr_pts"]})')

    if r.get('price_vs_sma200', 1.0) > cfg['sma200_overextend_threshold']:
        score += cfg['sma200_overextend_pts']
        factors.append(f'SMA200 overextended ({cfg["sma200_overextend_pts"]})')

    if r.get('regime_strategy') == 'reduce_size':
        score += cfg['regime_reduce_pts']

    if r.get('regime_trend_str') == 'weak' and r.get('regime_vol') == 'high':
        score += cfg['regime_weak_volatile_pts']
        factors.append(f'Weak trend + volatile ({cfg["regime_weak_volatile_pts"]})')

    # --- Tier 2: Volume + personality ---
    vol_s = r.get('volume_score', 0.5)
    volume_pts = int(vol_s * cfg['volume_max_pts'])
    score += volume_pts
    if volume_pts >= 5:
        factors.append(f'Volume confirmed (+{volume_pts})')
    if vol_s < cfg['volume_disconfirm_threshold']:
        score += cfg['volume_disconfirm_pts']
        factors.append(f'Volume disconfirm ({cfg["volume_disconfirm_pts"]})')

    if wave == 3 and r.get('w3_highest_vol'):
        pts = cfg['w3_personality_pts']
        score += pts
        factors.append(f'W3 personality (+{pts})')

    if r.get('w5_reversal_warn'):
        pts = cfg['w5_exhaustion_pts']
        score += pts
        factors.append(f'W5 exhaustion ({pts})')

    p_conf = r.get('personality_conf', 0)
    p_pts = int(p_conf * cfg['personality_max_pts'])
    score += p_pts

    vr = r.get('vol_ratio', 1.0)
    if vr > cfg['vol_surge_threshold']:
        pts = cfg['vol_surge_pts']
        score += pts
        factors.append(f'Volume surge (+{pts})')
    elif vr < cfg['vol_low_threshold']:
        score += cfg['vol_low_pts']
        factors.append(f'Low volume ({cfg["vol_low_pts"]})')

    r['score'] = score
    top_factors = ', '.join(factors[:3])

    if score >= cfg['strong_buy_score'] and wave == 3:
        return 'STRONG BUY', f'Score {score}/100: {top_factors}'
    if score >= cfg['buy_score']:
        return 'BUY', f'Score {score}/100: {top_factors}'
    if score >= cfg['buy_dip_score'] and wave == 4 and trend == 'bullish':
        return 'BUY DIP', f'Score {score}/100: Wave 4 pullback'
    if score >= cfg['watch_score']:
        missing = []
        if not entry_ok:
            missing.append('momentum')
        if rr < cfg['watch_min_rr']:
            missing.append('risk/reward')
        if conf < cfg['watch_min_conf']:
            missing.append('confidence')
        return 'WATCH', f'Score {score}/100 — missing: {", ".join(missing) if missing else "timing"}'

    if wave in [5, 6] and trend == 'bullish' and entry_ok and not exit_warn:
        return 'WATCH', 'Bullish trend, watching for new Wave 1'

    if wave in [5, 6]:
        return 'HOLD', f'Wave {wave} — wait for new impulse'

    return 'WAIT', f'Score {score}/100 — no clear setup'


def apply_market_regime(results: list[dict]) -> tuple[str, str, float]:
    """Apply market regime overlay. Mutates results in place.

    Returns (regime, message, exit_pct).
    """
    exit_count = sum(1 for r in results if r.get('action') in ('EXIT', 'AVOID'))
    exit_pct = exit_count / len(results) * 100 if results else 0

    if exit_pct >= 30:
        regime = 'CAUTION'
        msg = f'{exit_pct:.0f}% EXIT/AVOID — market overheated, raising BUY thresholds'
        for r in results:
            if r.get('action') in ('BUY', 'BUY DIP', 'BUY CORRECTION'):
                if r.get('score', 0) < 75:
                    r['action'] = 'WATCH'
                    r['reason'] = f'[REGIME: Caution] {r["reason"]}'
    elif exit_pct >= 15:
        regime = 'MIXED'
        msg = f'{exit_pct:.0f}% EXIT/AVOID — mixed market, only high-conviction entries'
        for r in results:
            if r.get('action') in ('BUY DIP', 'BUY CORRECTION'):
                if r.get('score', 0) < 65:
                    r['action'] = 'WATCH'
                    r['reason'] = f'[REGIME: Mixed] {r["reason"]}'
    else:
        regime = 'FAVORABLE'
        msg = f'{exit_pct:.0f}% EXIT/AVOID — market conditions favorable'

    return regime, msg, exit_pct
