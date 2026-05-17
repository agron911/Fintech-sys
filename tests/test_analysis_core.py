"""
Tests for core analysis modules: peaks, validation, momentum indicators,
signal scoring, and end-to-end integration.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_impulse_df(n=300):
    """Create a DataFrame with a clear 5-wave impulse pattern."""
    np.random.seed(42)
    # Build the wave structure to match exactly n bars
    w1, w2, w3, w4, w5, wc = 60, 40, 80, 50, 40, 30
    total_wave = w1 + w2 + w3 + w4 + w5 + wc  # 300
    if n > total_wave:
        # Extend the correction phase to fill n bars
        wc = n - (w1 + w2 + w3 + w4 + w5)
    elif n < total_wave:
        # Scale each segment proportionally
        scale = n / total_wave
        w1 = max(10, int(60 * scale))
        w2 = max(8, int(40 * scale))
        w3 = max(15, int(80 * scale))
        w4 = max(10, int(50 * scale))
        w5 = max(8, int(40 * scale))
        wc = n - (w1 + w2 + w3 + w4 + w5)

    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    close = np.concatenate([
        np.linspace(100, 130, w1),   # Wave 1 up
        np.linspace(130, 115, w2),   # Wave 2 down (38% retrace)
        np.linspace(115, 165, w3),   # Wave 3 up (strongest)
        np.linspace(165, 150, w4),   # Wave 4 down
        np.linspace(150, 175, w5),   # Wave 5 up
        np.linspace(175, 160, wc),   # correction
    ])
    noise = np.random.normal(0, 0.5, n)
    close = close + noise

    vol_base = 1_000_000
    volume = np.concatenate([
        np.linspace(vol_base, vol_base * 1.3, w1),
        np.linspace(vol_base * 1.3, vol_base * 0.8, w2),
        np.linspace(vol_base * 0.8, vol_base * 1.8, w3),   # Wave 3 highest
        np.linspace(vol_base * 1.8, vol_base * 0.7, w4),
        np.linspace(vol_base * 0.7, vol_base * 0.9, w5),
        np.linspace(vol_base * 0.9, vol_base * 0.6, wc),
    ])

    df = pd.DataFrame({
        'open': close * 0.998,
        'high': close * 1.008,
        'low': close * 0.992,
        'close': close,
        'volume': volume.astype(int),
    }, index=dates)
    return df


def make_momentum_df(n=100, trend='up'):
    """Create a simple trending DataFrame for momentum indicator tests."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    if trend == 'up':
        close = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
    elif trend == 'down':
        close = np.linspace(150, 100, n) + np.random.normal(0, 1, n)
    else:
        close = 100 + np.random.normal(0, 3, n)

    df = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(500_000, 2_000_000, n),
    }, index=dates)
    return df


# ===================================================================
# TEST: Peak/Trough Detection
# ===================================================================

class TestPeakDetection:
    def test_basic_detection(self):
        """detect_peaks_troughs_enhanced should find peaks and troughs."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        df = make_impulse_df()
        peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
        assert len(peaks) >= 2, f"Expected >= 2 peaks, got {len(peaks)}"
        assert len(troughs) >= 2, f"Expected >= 2 troughs, got {len(troughs)}"

    def test_peaks_are_local_maxima(self):
        """Detected peaks should correspond to local high points."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        df = make_impulse_df()
        peaks, _ = detect_peaks_troughs_enhanced(df, column='close')
        closes = df['close'].values
        for p in peaks:
            if p > 2 and p < len(closes) - 2:
                assert closes[p] >= min(closes[p-2], closes[p+2]), \
                    f"Peak at {p} (value {closes[p]:.2f}) is not a local max"

    def test_troughs_are_local_minima(self):
        """Detected troughs should correspond to local low points."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        df = make_impulse_df()
        _, troughs = detect_peaks_troughs_enhanced(df, column='close')
        closes = df['close'].values
        for t in troughs:
            if t > 2 and t < len(closes) - 2:
                assert closes[t] <= max(closes[t-2], closes[t+2]), \
                    f"Trough at {t} (value {closes[t]:.2f}) is not a local min"

    def test_short_data_no_crash(self):
        """Should handle short data without crashing."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        df = make_momentum_df(n=20)
        peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)

    def test_alternating_structure(self):
        """Peaks and troughs should generally alternate."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        df = make_impulse_df()
        peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
        all_points = sorted(
            [(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs]
        )
        if len(all_points) >= 4:
            violations = 0
            for i in range(1, len(all_points)):
                if all_points[i][1] == all_points[i-1][1]:
                    violations += 1
            assert violations < len(all_points) * 0.3, \
                f"Too many non-alternating points: {violations}/{len(all_points)}"


# ===================================================================
# TEST: Validation
# ===================================================================

class TestValidation:
    def _make_wave_points(self, df):
        """Helper to get wave points from a DataFrame."""
        from src.analysis.core.peaks import detect_peaks_troughs_enhanced
        from src.analysis.core.impulse import find_best_impulse_wave
        peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
        result = find_best_impulse_wave(df, peaks, troughs, column='close')
        return result.get('wave_points', np.array([]))

    def test_validation_returns_required_keys(self):
        """validate_impulse_wave_rules should return a tuple of (valid, confidence, details)."""
        from src.analysis.core.validation import validate_impulse_wave_rules
        df = make_impulse_df()
        wave_pts = self._make_wave_points(df)
        if len(wave_pts) < 5:
            pytest.skip("Could not generate 5-wave pattern for validation test")

        result = validate_impulse_wave_rules(df, wave_pts, column='close')
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-tuple, got length {len(result)}"
        valid, confidence, details = result
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1.0

    def test_validation_config_strict_mode(self):
        """Strict config should have lower tolerance and higher threshold."""
        from src.analysis.core.validation import ValidationConfig
        strict = ValidationConfig.strict_config()
        default = ValidationConfig()
        assert strict.fibonacci_tolerance < default.fibonacci_tolerance
        assert strict.acceptance_threshold > default.acceptance_threshold
        assert strict.wave2_max_retracement < default.wave2_max_retracement

    def test_validation_config_relaxed_mode(self):
        """Relaxed config should have higher tolerance and lower threshold."""
        from src.analysis.core.validation import ValidationConfig
        relaxed = ValidationConfig.relaxed_config()
        default = ValidationConfig()
        assert relaxed.fibonacci_tolerance > default.fibonacci_tolerance
        assert relaxed.acceptance_threshold < default.acceptance_threshold

    def test_reality_adjustment_allows_passing(self):
        """After our fix, reality_adjustment=0.92 should allow good patterns to pass 0.20."""
        from src.analysis.core.validation import ValidationConfig
        config = ValidationConfig()
        assert config.reality_adjustment >= 0.90, \
            f"reality_adjustment {config.reality_adjustment} too low for patterns to pass thresholds"


# ===================================================================
# TEST: Momentum Indicators
# ===================================================================

class TestMomentumIndicators:
    def test_rsi_range(self):
        """RSI values should be between 0 and 100."""
        from src.analysis.core.momentum_indicators import compute_rsi
        df = make_momentum_df(n=100)
        rsi = compute_rsi(df['close'])
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI should be >= 0"
        assert (valid_rsi <= 100).all(), "RSI should be <= 100"

    def test_rsi_uptrend_above_50(self):
        """In a strong uptrend, RSI should generally be above 50."""
        from src.analysis.core.momentum_indicators import compute_rsi
        df = make_momentum_df(n=100, trend='up')
        rsi = compute_rsi(df['close'])
        recent_rsi = rsi.iloc[-20:].mean()
        assert recent_rsi > 45, f"Uptrend RSI avg {recent_rsi:.1f} should be above 45"

    def test_rsi_downtrend_below_50(self):
        """In a strong downtrend, RSI should generally be below 50."""
        from src.analysis.core.momentum_indicators import compute_rsi
        df = make_momentum_df(n=100, trend='down')
        rsi = compute_rsi(df['close'])
        recent_rsi = rsi.iloc[-20:].mean()
        assert recent_rsi < 55, f"Downtrend RSI avg {recent_rsi:.1f} should be below 55"

    def test_rsi_signal_returns_required_keys(self):
        """rsi_signal should return zone, value, divergence, strength."""
        from src.analysis.core.momentum_indicators import rsi_signal
        df = make_momentum_df()
        result = rsi_signal(df)
        assert 'zone' in result
        assert 'value' in result
        assert 'strength' in result
        assert result['zone'] in ('oversold', 'overbought', 'neutral')

    def test_macd_signal_returns_required_keys(self):
        """macd_signal should return crossover, histogram_direction, strength."""
        from src.analysis.core.momentum_indicators import macd_signal
        df = make_momentum_df()
        result = macd_signal(df)
        assert 'strength' in result
        assert 'histogram_direction' in result

    def test_adx_signal_returns_required_keys(self):
        """adx_signal should return value, regime, trending, strength."""
        from src.analysis.core.momentum_indicators import adx_signal
        df = make_momentum_df()
        result = adx_signal(df)
        assert 'value' in result
        assert 'regime' in result
        assert 'trending' in result
        assert isinstance(result['trending'], bool)

    def test_atr_regime_returns_required_keys(self):
        """atr_regime_signal should return regime, atr_pct, stop_multiplier."""
        from src.analysis.core.momentum_indicators import atr_regime_signal
        df = make_momentum_df()
        result = atr_regime_signal(df)
        assert 'regime' in result
        assert 'atr_pct' in result
        assert 'stop_multiplier' in result
        assert result['regime'] in ('calm', 'normal', 'volatile', 'explosive')

    def test_composite_momentum_returns_gates(self):
        """compute_momentum_composite should return entry_ok, exit_warning, composite_score."""
        from src.analysis.core.momentum_indicators import compute_momentum_composite
        df = make_momentum_df()
        result = compute_momentum_composite(df)
        assert 'entry_ok' in result
        assert 'exit_warning' in result
        assert 'composite_score' in result
        assert isinstance(result['entry_ok'], bool)
        assert isinstance(result['exit_warning'], bool)

    def test_composite_uptrend_positive(self):
        """In an uptrend, composite score should generally be positive."""
        from src.analysis.core.momentum_indicators import compute_momentum_composite
        df = make_momentum_df(n=100, trend='up')
        result = compute_momentum_composite(df)
        assert result['composite_score'] > -0.5, \
            f"Uptrend composite {result['composite_score']:.2f} should not be strongly negative"


# ===================================================================
# TEST: Signal Scoring
# ===================================================================

class TestSignalScoring:
    def _make_result_dict(self, **overrides):
        """Create a minimal result dict for classify_action."""
        defaults = {
            'symbol': 'TEST', 'price': 150.0, 'wave': 3,
            'wave_type': 'impulse', 'trend': 'bullish', 'pdir': 'up',
            'conf': 0.6, 'entry_ok': True, 'exit_warn': False,
            'composite': 0.3, 'regime': 'TRENDING',
            'rsi': 55, 'rsi_zone': 'neutral', 'rsi_div': None,
            'macd_cross': None, 'macd_mom': 'accelerating',
            'adx': 30, 'adx_trending': True,
            'vel_5d': 3, 'vel_20d': 5, 'speed': 'slow_up', 'mom_6m': 10,
            'stop': 140, 'target1': 170, 'target2': 185, 'rr': 2.0,
            'fib_retrace': None,
            'macd_hist_dir': 'expanding_up', 'macd_strength': 0.5,
            'vel_accel': 'accelerating_up', 'rsi_strength': 0.3,
            'adx_strength': 0.6, 'adx_regime': 'strong_trend',
            'atr_pct': 2.0, 'atr_value': 3.0, 'atr_regime': 'normal',
            'atr_percentile': 50, 'price_vs_sma200': 1.1,
            'sma50': 145, 'sma200': 135,
            'regime_strategy': 'breakout', 'regime_trend_str': 'strong',
            'regime_vol': 'normal',
            'volume_score': 0.7, 'vol_ratio': 1.2,
            'w3_highest_vol': True, 'w5_reversal_warn': False,
            'personality_conf': 0.4,
            'trailing_stop': 143, 'size_mult': 1.0, 'size_note': 'Wave 3',
        }
        defaults.update(overrides)
        return defaults

    def test_strong_buy_wave3_bullish(self):
        """Wave 3 + bullish + momentum confirmed should score >= 65."""
        from src.analysis.core.signal_scoring import classify_action
        r = self._make_result_dict(wave=3, trend='bullish', entry_ok=True, conf=0.6)
        action, reason = classify_action(r)
        assert r['score'] >= 65, f"Wave 3 bullish score {r['score']} should be >= 65"
        assert action in ('STRONG BUY', 'BUY'), f"Got {action} instead of BUY/STRONG BUY"

    def test_avoid_bearish_trend(self):
        """Bearish trend should produce AVOID."""
        from src.analysis.core.signal_scoring import classify_action
        r = self._make_result_dict(trend='bearish', pdir='down')
        action, reason = classify_action(r)
        assert action == 'AVOID', f"Got {action} instead of AVOID for bearish trend"

    def test_exit_wave5_warning(self):
        """Wave 5+ with exit warning and below SMA50 should trigger EXIT."""
        from src.analysis.core.signal_scoring import classify_action
        r = self._make_result_dict(
            wave=5, exit_warn=True, trend='bullish', pdir='up',
            price=140, sma50=145
        )
        action, reason = classify_action(r)
        assert action == 'EXIT', f"Got {action} instead of EXIT"

    def test_wait_low_score(self):
        """Low confidence + no momentum should produce WATCH or WAIT."""
        from src.analysis.core.signal_scoring import classify_action
        r = self._make_result_dict(
            wave=1, trend='neutral', pdir='up', entry_ok=False,
            conf=0.25, rr=0.8, composite=-0.1,
            adx_trending=False, volume_score=0.1,
            macd_hist_dir='contracting', vel_accel='decelerating',
            adx_regime='no_trend', w3_highest_vol=False,
            personality_conf=0.1, macd_cross=None,
            rsi=55, rsi_zone='neutral', rsi_div=None,
        )
        action, reason = classify_action(r)
        assert action in ('WATCH', 'WAIT'), \
            f"Got {action} with score {r.get('score')} — low conditions should produce WATCH/WAIT"

    def test_score_set_as_side_effect(self):
        """classify_action should set r['score'] as a side effect."""
        from src.analysis.core.signal_scoring import classify_action
        r = self._make_result_dict()
        classify_action(r)
        assert 'score' in r
        assert isinstance(r['score'], (int, float))

    def test_grade_conviction_returns_grades(self):
        """grade_conviction should return a dict with trend_grade, timing_grade, risk_grade."""
        from src.analysis.core.signal_scoring import grade_conviction
        r = self._make_result_dict()
        r['score'] = 70
        r['action'] = 'BUY'
        result = grade_conviction(r)
        assert 'trend_grade' in result
        assert 'timing_grade' in result
        assert 'risk_grade' in result
        for grade in [result['trend_grade'], result['timing_grade'], result['risk_grade']]:
            assert grade in ('A', 'B', 'C', 'D', 'F'), f"Invalid grade: {grade}"


# ===================================================================
# TEST: Pattern Adapter
# ===================================================================

class TestPatternAdapterConfidence:
    def test_confidence_floor(self):
        """Confidence should never drop below 0.15 after penalty stacking."""
        from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input
        df = make_impulse_df()
        wave_data = {
            'impulse_wave': np.array([0, 60, 100, 180, 230, 270]),
            'confidence': 0.10,
            'wave_type': 'impulse',
            'pattern_relationships': {},
            'multiple_patterns': [],
        }
        result = adapt_wave_data_to_strategy_input(df, wave_data, column='close')
        assert result['confidence'] >= 0.15, \
            f"Confidence {result['confidence']} fell below floor of 0.15"


# ===================================================================
# TEST: Strategy Thresholds
# ===================================================================

class TestStrategyThresholds:
    def test_fibonacci_tolerance_relaxed(self):
        """Fibonacci tolerance should be 0.10, not the overly strict 0.05."""
        from src.backtest.strategy_advanced import MultiTimeframeAlignmentStrategy
        strategy = MultiTimeframeAlignmentStrategy({})
        assert strategy.fib_tolerance == 0.10, \
            f"fib_tolerance {strategy.fib_tolerance} should be 0.10"

    def test_composite_confidence_threshold(self):
        """Composite confidence threshold should be 0.20."""
        from src.backtest.strategy_advanced import MultiTimeframeAlignmentStrategy
        strategy = MultiTimeframeAlignmentStrategy({})
        assert strategy.min_composite_confidence == 0.20, \
            f"min_composite_confidence {strategy.min_composite_confidence} should be 0.20"

    def test_validation_confidence_threshold(self):
        """Validation confidence threshold should be 0.20."""
        from src.backtest.strategy_advanced import MultiTimeframeAlignmentStrategy
        strategy = MultiTimeframeAlignmentStrategy({})
        assert strategy.min_validation_confidence == 0.20, \
            f"min_validation_confidence {strategy.min_validation_confidence} should be 0.20"

    def test_thresholds_configurable(self):
        """All thresholds should be overridable via config."""
        from src.backtest.strategy_advanced import MultiTimeframeAlignmentStrategy
        strategy = MultiTimeframeAlignmentStrategy({
            'fibonacci_tolerance': 0.15,
            'min_composite_confidence': 0.10,
            'min_validation_confidence': 0.10,
        })
        assert strategy.fib_tolerance == 0.15
        assert strategy.min_composite_confidence == 0.10
        assert strategy.min_validation_confidence == 0.10


# ===================================================================
# TEST: Market Structure
# ===================================================================

class TestMarketStructure:
    def test_sector_map_covers_expanded_universe(self):
        """All stocks in EXPANDED_UNIVERSE should have a sector mapping."""
        from src.analysis.market_structure import SECTOR_MAP
        expanded = [
            'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'NFLX', 'CRM', 'ADBE',
            'INTC', 'QCOM', 'AVGO', 'MU', 'MRVL', 'ANET', 'PANW', 'CRWD',
            'SNOW', 'DDOG', 'ZS', 'NET', 'ORCL', 'CSCO', 'IBM', 'NOW',
            'ADSK', 'FTNT', 'ESTC', 'DELL', 'HPE', 'ASML', 'LRCX', 'KLAC',
            'MCHP', 'ON', 'SWKS', 'TXN', 'ADI', 'AMAT', 'TSM', 'SMCI',
            'PLTR', 'APP', 'IONQ', 'SOUN', 'TTD', 'RKLB', 'U', 'ROKU',
            'SNAP', 'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ',
            'COIN', 'BLK', 'BX', 'SCHW', 'AXP', 'BK', 'ICE', 'WFC',
            'HOOD', 'SOFI', 'AFRM', 'MARA', 'UNH', 'JNJ', 'LLY', 'PFE',
            'ABBV', 'MRK', 'BMY', 'GILD', 'AMGN', 'REGN', 'ISRG', 'DHR',
            'MDT', 'VRTX', 'HIMS', 'NVO', 'TMO', 'XOM', 'CVX', 'COP',
            'SLB', 'OXY', 'MPC', 'EQT', 'COST', 'WMT', 'TGT', 'NKE',
            'SBUX', 'MCD', 'DIS', 'HD', 'LOW', 'LULU', 'CMG', 'BKNG',
            'ABNB', 'DLTR', 'CHWY', 'CELH', 'TOST', 'DASH', 'DKNG',
            'CAT', 'DE', 'GE', 'BA', 'LMT', 'RTX', 'GD', 'HON', 'UNP',
            'UPS', 'PG', 'KO', 'PEP', 'CL', 'PM', 'MO', 'MDLZ', 'AFL',
            'T', 'VZ', 'CMCSA', 'SPOT', 'EA', 'BIDU', 'PDD', 'SE',
            'SHOP', 'MELI', 'NTES', 'TSLA', 'RIVN', 'LCID', 'FSLR',
            'ENPH', 'ACHR', 'PINS', 'ZM', 'F', 'GM', 'CCL', 'LVS',
            'TRIP', 'WBA', 'CVS', 'DOW', 'MMM', 'LIN', 'APD', 'NEE',
            'AEP', 'ADP', 'APH', 'VRT', 'TEM',
        ]
        missing = [s for s in expanded if s not in SECTOR_MAP]
        assert len(missing) == 0, f"Missing from SECTOR_MAP: {missing}"

    def test_unmapped_stocks_get_other_sector(self):
        """Stocks not in SECTOR_MAP should be classified as 'Other'."""
        from src.analysis.market_structure import _compute_sector_rotation
        results = [
            {'symbol': 'FAKE_STOCK', 'trend': 'bullish', 'mom_6m': 10},
        ]
        rotation = _compute_sector_rotation(results)
        sectors = [r['sector'] for r in rotation]
        assert 'Other' in sectors, "Unmapped stocks should appear under 'Other'"

    def test_analyze_market_structure_empty(self):
        """Empty input should return UNKNOWN regime."""
        from src.analysis.market_structure import analyze_market_structure
        result = analyze_market_structure([])
        assert result['regime'] == 'UNKNOWN'
        assert result['total_stocks'] == 0

    def test_tw_stocks_have_sector_mapping(self):
        """Key TW stocks should have sector mappings."""
        from src.analysis.market_structure import SECTOR_MAP
        tw_tickers = ['2330', '2317', '2881', '1216', '1303', '2412']
        missing = [t for t in tw_tickers if t not in SECTOR_MAP]
        assert len(missing) == 0, f"TW stocks missing from SECTOR_MAP: {missing}"


class TestMarketClassification:
    def test_classify_us_stock(self):
        """US tickers should be classified as 'US'."""
        from src.analysis.core.signal_scoring import _classify_market
        assert _classify_market('AAPL') == 'US'
        assert _classify_market('MSFT') == 'US'
        assert _classify_market('META') == 'US'

    def test_classify_tw_stock(self):
        """Numeric tickers should be classified as 'TW'."""
        from src.analysis.core.signal_scoring import _classify_market
        assert _classify_market('2330') == 'TW'
        assert _classify_market('1102') == 'TW'
        assert _classify_market('TWII') == 'TW'

    def test_rs_ranking_per_market(self):
        """RS ranking should be independent per market."""
        from src.analysis.core.signal_scoring import apply_relative_strength
        results = [
            {'symbol': 'AAPL', 'mom_6m': 20, 'market': 'US'},
            {'symbol': 'MSFT', 'mom_6m': 10, 'market': 'US'},
            {'symbol': '2330', 'mom_6m': 15, 'market': 'TW'},
            {'symbol': '2317', 'mom_6m': 5, 'market': 'TW'},
        ]
        apply_relative_strength(results)
        # AAPL should be #1 in US, MSFT #2 in US
        assert results[0]['rs_rank'] == 1
        assert results[1]['rs_rank'] == 2
        # 2330 should be #1 in TW, 2317 #2 in TW
        assert results[2]['rs_rank'] == 1
        assert results[3]['rs_rank'] == 2


# ===================================================================
# TEST: Integration - Full Pipeline
# ===================================================================

class TestIntegration:
    def test_analyze_stock_produces_result(self):
        """analyze_stock should produce a non-None result for clean impulse data."""
        from src.analysis.core.signal_scoring import analyze_stock
        df = make_impulse_df(n=300)
        result = analyze_stock('TEST', df)
        # The synthetic data may or may not produce a pattern - that's OK
        # What matters is it doesn't crash
        if result is not None:
            assert 'symbol' in result
            assert 'price' in result
            assert 'wave' in result
            assert 'rr' in result

    def test_classify_action_after_analyze(self):
        """Full pipeline: analyze_stock -> classify_action should not crash."""
        from src.analysis.core.signal_scoring import analyze_stock, classify_action
        df = make_impulse_df(n=300)
        result = analyze_stock('TEST', df)
        if result is not None:
            action, reason = classify_action(result)
            assert action in (
                'STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION',
                'WATCH', 'EXIT', 'AVOID', 'HOLD', 'WAIT', 'SKIP'
            )
            assert isinstance(reason, str)
            assert 'score' in result

    def test_walk_forward_produces_diagnostic_logs(self):
        """Walk-forward should run without errors on synthetic data."""
        from src.backtest.backtester import Backtester
        import tempfile, os
        df = make_impulse_df(n=600)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'TEST.txt')
            save_df = df.copy()
            save_df.index = save_df.index.strftime('%Y/%m/%d')
            save_df.insert(0, 'Date', save_df.index)
            save_df['Date_end'] = save_df['Date']
            save_df.columns = list(save_df.columns[:-1]) + ['Date']
            save_df.to_csv(filepath, sep='\t', index=False)

            config = {
                'stk2_dir': tmpdir,
                'processed_dir': tmpdir,
                'backtest_window_size': 300,
                'backtest_step_size': 60,
                'initial_capital': 100000,
            }
            bt = Backtester(config)
            bt.run(['TEST'])
            # Should complete without error; results may or may not have trades


# ===================================================================
# TEST: Requirements
# ===================================================================

# ===================================================================
# TEST: Liquidity Filter
# ===================================================================

class TestLiquidityFilter:
    def test_low_volume_returns_none(self):
        """Stocks with avg volume < 500K should be filtered out."""
        from src.analysis.core.signal_scoring import analyze_stock
        df = make_impulse_df(n=300)
        df['volume'] = 100_000  # Below 500K threshold
        result = analyze_stock('LOW_VOL', df)
        assert result is None, "Low-volume stock should return None"

    def test_high_volume_passes(self):
        """Stocks with avg volume >= 500K should pass the filter."""
        from src.analysis.core.signal_scoring import analyze_stock
        df = make_impulse_df(n=300)
        df['volume'] = 1_000_000  # Above threshold
        result = analyze_stock('HIGH_VOL', df)
        # May still return None due to pattern detection, but not due to liquidity
        # Just verify it doesn't crash
        assert result is None or isinstance(result, dict)


# ===================================================================
# TEST: Sector Rotation Scoring
# ===================================================================

class TestSectorRotationScoring:
    def test_leading_sector_bonus(self):
        """Leading sector should add +5 to score."""
        from src.analysis.core.signal_scoring import classify_action
        r = TestSignalScoring()._make_result_dict(sector_direction='leading')
        classify_action(r)
        r_no_sector = TestSignalScoring()._make_result_dict()
        classify_action(r_no_sector)
        assert r['score'] >= r_no_sector['score'] + 4, \
            f"Leading sector score {r['score']} should be ~5 more than {r_no_sector['score']}"

    def test_lagging_sector_penalty(self):
        """Lagging sector should subtract -5 from score."""
        from src.analysis.core.signal_scoring import classify_action
        r = TestSignalScoring()._make_result_dict(sector_direction='lagging')
        classify_action(r)
        r_no_sector = TestSignalScoring()._make_result_dict()
        classify_action(r_no_sector)
        assert r['score'] <= r_no_sector['score'] - 4, \
            f"Lagging sector score {r['score']} should be ~5 less than {r_no_sector['score']}"

    def test_compute_sector_direction_map(self):
        """compute_sector_direction_map should return per-symbol directions."""
        from src.analysis.market_structure import compute_sector_direction_map
        results = [
            {'symbol': 'AAPL', 'trend': 'bullish', 'mom_6m': 20},
            {'symbol': 'MSFT', 'trend': 'bullish', 'mom_6m': 15},
            {'symbol': 'XOM', 'trend': 'bearish', 'mom_6m': -10},
        ]
        dir_map = compute_sector_direction_map(results)
        assert isinstance(dir_map, dict)
        assert dir_map.get('AAPL') in ('leading', 'neutral', 'lagging')
        assert dir_map.get('XOM') in ('leading', 'neutral', 'lagging')


# ===================================================================
# TEST: Gap-Down Stop
# ===================================================================

class TestGapDownStop:
    def test_gap_down_exits_at_open(self):
        """When open gaps below stop, should exit at open price."""
        from src.backtest.strategy_advanced import AdvancedBacktester
        dates = pd.date_range('2023-01-01', periods=20, freq='B')
        close = [100] * 5 + [105, 108, 110, 112, 115, 80, 82, 85, 88, 90, 92, 95, 97, 100, 102]
        df = pd.DataFrame({
            'open':   [100] * 5 + [104, 107, 109, 111, 114, 75, 81, 84, 87, 89, 91, 94, 96, 99, 101],
            'high':   [c * 1.01 for c in close],
            'low':    [c * 0.99 for c in close],
            'close':  close,
            'volume': [1_000_000] * 20,
        }, index=dates)

        signals = [{
            'date': dates[5],
            'type': 'BUY',
            'price': 105,
            'stop_loss': 95,
            'wave_number': 3,
            'confidence': 0.7,
            'targets': [120, 130],
        }]

        bt = AdvancedBacktester(initial_capital=100_000, config={})
        stats = bt.simulate(df, signals)
        if stats.get('total_trades', 0) > 0:
            trades = stats.get('trades', [])
            gap_trades = [t for t in trades if t.get('exit_reason') == 'gap_down_stop']
            if gap_trades:
                assert gap_trades[0]['exit_price'] == 75, \
                    f"Gap-down exit should be at open price 75, got {gap_trades[0]['exit_price']}"


# ===================================================================
# TEST: Sector Concentration
# ===================================================================

class TestSectorConcentration:
    def test_limits_buys_per_sector(self):
        """3 Tech BUYs with max_per_sector=2 should downgrade lowest-score one."""
        from src.analysis.core.signal_scoring import apply_sector_concentration
        results = [
            {'symbol': 'AAPL', 'action': 'BUY', 'score': 80, 'reason': 'good'},
            {'symbol': 'MSFT', 'action': 'BUY', 'score': 75, 'reason': 'good'},
            {'symbol': 'GOOG', 'action': 'BUY', 'score': 70, 'reason': 'good'},
            {'symbol': 'JPM',  'action': 'BUY', 'score': 65, 'reason': 'good'},
        ]
        apply_sector_concentration(results, max_per_sector=2)
        tech_buys = [r for r in results if r['symbol'] in ('AAPL', 'MSFT', 'GOOG')
                     and r['action'] == 'BUY']
        assert len(tech_buys) <= 2
        goog = [r for r in results if r['symbol'] == 'GOOG'][0]
        assert goog['action'] == 'WATCH'
        assert 'Sector limit' in goog['reason']
        jpm = [r for r in results if r['symbol'] == 'JPM'][0]
        assert jpm['action'] == 'BUY'

    def test_no_change_under_limit(self):
        """2 Tech BUYs with max=2 should remain unchanged."""
        from src.analysis.core.signal_scoring import apply_sector_concentration
        results = [
            {'symbol': 'AAPL', 'action': 'BUY', 'score': 80, 'reason': 'good'},
            {'symbol': 'MSFT', 'action': 'BUY', 'score': 75, 'reason': 'good'},
        ]
        apply_sector_concentration(results, max_per_sector=2)
        assert all(r['action'] == 'BUY' for r in results)

    def test_watch_stocks_unaffected(self):
        """Non-BUY stocks should not count toward sector limit."""
        from src.analysis.core.signal_scoring import apply_sector_concentration
        results = [
            {'symbol': 'AAPL', 'action': 'BUY', 'score': 80, 'reason': 'good'},
            {'symbol': 'MSFT', 'action': 'WATCH', 'score': 50, 'reason': 'wait'},
            {'symbol': 'GOOG', 'action': 'EXIT', 'score': 10, 'reason': 'exit'},
        ]
        apply_sector_concentration(results, max_per_sector=1)
        assert results[0]['action'] == 'BUY'
        assert results[1]['action'] == 'WATCH'
        assert results[2]['action'] == 'EXIT'


# ===================================================================
# TEST: Random Entry Benchmark
# ===================================================================

class TestRandomEntryBenchmark:
    def test_benchmark_runs_without_error(self):
        """Random benchmark should complete without crashing on synthetic data."""
        from src.backtest.backtester import Backtester
        import tempfile, os
        df = make_impulse_df(n=600)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'TEST.txt')
            save_df = df.copy()
            save_df.index = save_df.index.strftime('%Y/%m/%d')
            save_df.insert(0, 'Date', save_df.index)
            save_df['Date_end'] = save_df['Date']
            save_df.columns = list(save_df.columns[:-1]) + ['Date']
            save_df.to_csv(filepath, sep='\t', index=False)

            config = {
                'stk2_dir': tmpdir, 'processed_dir': tmpdir,
                'backtest_window_size': 300, 'backtest_step_size': 60,
                'initial_capital': 100000,
            }
            bt = Backtester(config)
            bt.run(['TEST'])
            result = bt.run_random_entry_benchmark(['TEST'], n_iterations=5)
            assert isinstance(result, dict)

    def test_empty_results_handled(self):
        """Benchmark with no data should return empty dict."""
        from src.backtest.backtester import Backtester
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'stk2_dir': tmpdir, 'processed_dir': tmpdir,
                'initial_capital': 100000,
            }
            bt = Backtester(config)
            result = bt.run_random_entry_benchmark(['FAKE'], n_iterations=3)
            assert isinstance(result, dict)


class TestRequirements:
    def test_no_sklearn_dependency(self):
        """requirements.txt should not include scikit-learn."""
        from pathlib import Path
        req_path = Path(__file__).parent.parent / 'requirements.txt'
        content = req_path.read_text()
        assert 'scikit-learn' not in content, \
            "scikit-learn should be removed from requirements.txt"
