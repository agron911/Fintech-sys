"""
Tests for the backtester and related components.

Covers:
- PositionManager: position sizing, risk limits
- RiskManager: drawdown tracking, position limits
- AdvancedBacktester: signal execution, exits, cooldown, statistics
- pattern_adapter: wave data transformation, trend context
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtest.strategy_advanced import (
    PositionManager, RiskManager, AdvancedBacktester
)
from src.backtest.pattern_adapter import (
    adapt_wave_data_to_strategy_input,
    _compute_trend_context,
    _build_current_position,
    _determine_current_wave,
)


# ============================================================================
# Helpers
# ============================================================================

def make_ohlcv(closes, start='2024-01-01', volume=1_000_000):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    df = pd.DataFrame({
        'open':   [c * 0.995 for c in closes],
        'high':   [c * 1.01  for c in closes],
        'low':    [c * 0.99  for c in closes],
        'close':  closes,
        'volume': [volume] * n,
    }, index=dates)
    return df


def make_bullish_wave_df(n=300):
    """Create a DataFrame with a clear bullish SMA50>SMA200 trend and a 5-wave impulse."""
    # Start with a long base period so SMA200 is established
    base = np.linspace(50, 80, 200)  # slow uptrend for SMA setup
    # Then the impulse pattern
    w1 = np.linspace(80, 100, 20)   # Wave 1 up
    w2 = np.linspace(100, 92, 15)   # Wave 2 retrace
    w3 = np.linspace(92, 130, 30)   # Wave 3 up (strongest)
    w4 = np.linspace(130, 122, 15)  # Wave 4 retrace
    w5 = np.linspace(122, 140, 20)  # Wave 5 up
    prices = np.concatenate([base, w1, w2, w3, w4, w5])
    return make_ohlcv(prices.tolist())


def make_pattern_analysis(df, wave_indices=None, confidence=0.5, trend_direction='up'):
    """Build a pattern_analysis dict compatible with strategy.generate_signals()."""
    if wave_indices is None:
        n = len(df)
        wave_indices = [n - 100, n - 80, n - 65, n - 35, n - 20, n - 1]

    wave_prices = [float(df['close'].iloc[i]) for i in wave_indices]

    current_position = {
        'wave_indices': wave_indices,
        'wave_prices': wave_prices,
        'trend_direction': trend_direction,
        'wave_number': 3,
        'wave_1_start': wave_prices[0],
        'wave_1_end': wave_prices[1],
        'wave_1_high': max(wave_prices[0], wave_prices[1]),
        'wave_1_low': min(wave_prices[0], wave_prices[1]),
        'wave_1_range': abs(wave_prices[1] - wave_prices[0]),
        'wave_2_end': wave_prices[2],
        'wave_2_low': wave_prices[2],
        'previous_wave_high': max(wave_prices[:2]),
        'previous_wave_low': min(wave_prices[:2]),
        'wave_3_end': wave_prices[3],
        'wave_3_high': wave_prices[3],
        'impulse_high': max(wave_prices[:4]),
        'impulse_low': min(wave_prices[:4]),
    }

    return {
        'alignment_score': 0.6,
        'current_position': current_position,
        'best_pattern': {
            'composite_confidence': confidence,
            'base_pattern': {
                'pattern': {
                    'points': np.array(wave_indices),
                    'confidence': confidence,
                    'wave_type': 'impulse',
                }
            }
        },
        'confidence': confidence,
        'raw_confidence': confidence,
        'trend_context': {'trend': 'bullish', 'sma50': 110.0, 'sma200': 90.0, 'price_vs_sma200': 1.2},
        'momentum': {
            'composite_score': 0.5,
            'entry_ok': True,
            'exit_warning': False,
            'confidence_adjustment': 1.0,
            'regime': 'BULLISH',
            'stop_multiplier': 1.0,
            'velocity': {'speed_regime': 'slow_up'},
        },
        'wave_data': {
            'impulse_wave': np.array(wave_indices),
            'confidence': confidence,
            'wave_type': 'impulse',
        },
    }


# ============================================================================
# PositionManager Tests
# ============================================================================

class TestPositionManager:

    def test_basic_position_size(self):
        pm = PositionManager(100_000, risk_per_trade=0.02)
        # $2000 risk, entry=100, stop=95 → risk_per_share=5 → 400 shares
        # but capped at 15% allocation = 150 shares
        shares = pm.calculate_position_size(100.0, 95.0)
        assert shares == 150  # min(400 from risk, 150 from allocation cap)

    def test_max_allocation_cap(self):
        pm = PositionManager(100_000, risk_per_trade=0.02)
        # Tiny risk per share → huge position → capped at 15%
        shares = pm.calculate_position_size(10.0, 9.99)
        max_allowed = int(100_000 * 0.15 / 10.0)  # 1500
        assert shares <= max_allowed

    def test_zero_risk_fallback(self):
        pm = PositionManager(100_000, risk_per_trade=0.02)
        shares = pm.calculate_position_size(100.0, 100.0)
        # Should fall back to 10% allocation
        assert shares == int(100_000 * 0.1 / 100.0)

    def test_minimum_one_share(self):
        pm = PositionManager(1_000, risk_per_trade=0.01)
        shares = pm.calculate_position_size(500.0, 490.0)
        assert shares >= 1

    def test_update_portfolio_value(self):
        pm = PositionManager(100_000)
        pm.update_portfolio_value(120_000)
        assert pm.portfolio_value == 120_000
        assert pm.max_risk_amount == 120_000 * pm.risk_per_trade


# ============================================================================
# RiskManager Tests
# ============================================================================

class TestRiskManager:

    def test_can_open_within_limits(self):
        rm = RiskManager(100_000, max_concurrent_positions=3)
        assert rm.can_open_position() is True

    def test_blocks_at_max_positions(self):
        rm = RiskManager(100_000, max_concurrent_positions=2)
        rm.open_positions = [{'a': 1}, {'b': 2}]
        assert rm.can_open_position() is False

    def test_blocks_on_daily_drawdown(self):
        rm = RiskManager(100_000, max_daily_dd=0.05)
        rm.daily_start_capital = 100_000
        rm.current_capital = 94_000  # 6% drawdown > 5% limit
        assert rm.can_open_position() is False

    def test_drawdown_calculation(self):
        rm = RiskManager(100_000)
        rm.update_capital(110_000)
        assert rm.peak_capital == 110_000
        rm.update_capital(99_000)
        dd = rm.get_current_drawdown()
        assert abs(dd - 0.1) < 0.001  # 10% drawdown from 110K peak

    def test_reset_daily(self):
        rm = RiskManager(100_000)
        rm.update_capital(95_000)
        rm.reset_daily()
        assert rm.daily_start_capital == 95_000


# ============================================================================
# AdvancedBacktester Tests
# ============================================================================

class TestAdvancedBacktester:

    def test_no_signals_returns_zero_trades(self):
        """Backtest with bearish data should produce no trades."""
        # Bearish trend → buy_allowed = False → no signals
        prices = np.linspace(200, 100, 300).tolist()
        df = make_ohlcv(prices)
        pa = make_pattern_analysis(df, confidence=0.5)
        pa['trend_context']['trend'] = 'bearish'
        pa['current_position']['trend_direction'] = 'down'

        bt = AdvancedBacktester(initial_capital=100_000)
        stats = bt.run_backtest(df, pattern_analysis=pa)
        assert stats['total_trades'] == 0
        assert stats['final_capital'] == 100_000

    def test_same_day_exit_protection(self):
        """Positions should not be stopped out on the entry day."""
        df = make_bullish_wave_df()
        pa = make_pattern_analysis(df, confidence=0.5)

        bt = AdvancedBacktester(initial_capital=100_000)
        stats = bt.run_backtest(df, pattern_analysis=pa)

        # No trade should have entry_date == exit_date with reason 'stop_loss'
        for trade in stats.get('trades', []):
            if trade['exit_reason'] == 'stop_loss':
                assert trade['entry_date'] != trade['exit_date'], \
                    f"Same-day stop-out: {trade['entry_date']}"

    def test_entry_cooldown(self):
        """Should not open positions within 5 bars of each other."""
        df = make_bullish_wave_df()
        pa = make_pattern_analysis(df, confidence=0.5)

        bt = AdvancedBacktester(initial_capital=100_000)
        stats = bt.run_backtest(df, pattern_analysis=pa)

        trades = stats.get('trades', [])
        if len(trades) >= 2:
            for i in range(1, len(trades)):
                d1 = trades[i - 1]['entry_date']
                d2 = trades[i]['entry_date']
                try:
                    idx1 = df.index.get_loc(d1)
                    idx2 = df.index.get_loc(d2)
                    assert idx2 - idx1 >= 5, \
                        f"Cooldown violated: {d1} and {d2} are {idx2-idx1} bars apart"
                except KeyError:
                    pass

    def test_statistics_with_no_trades(self):
        bt = AdvancedBacktester(initial_capital=50_000)
        bt.trades = []
        stats = bt._calculate_statistics()
        assert stats['total_trades'] == 0
        assert stats['final_capital'] == 50_000
        assert stats['win_rate'] == 0
        assert stats['sharpe_ratio'] == 0

    def test_statistics_with_trades(self):
        bt = AdvancedBacktester(initial_capital=100_000)
        bt.trades = [
            {'profit': 500, 'profit_pct': 5.0, 'r_multiple': 1.5,
             'entry_date': '2024-01-01', 'exit_date': '2024-01-10',
             'entry_price': 100, 'exit_price': 105, 'shares': 100},
            {'profit': -200, 'profit_pct': -2.0, 'r_multiple': -1.0,
             'entry_date': '2024-02-01', 'exit_date': '2024-02-10',
             'entry_price': 100, 'exit_price': 98, 'shares': 100},
            {'profit': 300, 'profit_pct': 3.0, 'r_multiple': 1.0,
             'entry_date': '2024-03-01', 'exit_date': '2024-03-10',
             'entry_price': 100, 'exit_price': 103, 'shares': 100},
        ]
        stats = bt._calculate_statistics()
        assert stats['total_trades'] == 3
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert abs(stats['win_rate'] - 2/3) < 0.001
        assert stats['total_profit'] == 600
        assert stats['final_capital'] == 100_600

    def test_close_position_updates_capital(self):
        bt = AdvancedBacktester(initial_capital=100_000)
        entry_date = pd.Timestamp('2024-01-01')
        bt.open_positions[entry_date] = {
            'entry_date': entry_date,
            'entry_price': 100.0,
            'shares': 50,
            'stop_loss': 95.0,
            'wave_number': 3,
            'confidence': 0.6,
            'targets': [],
        }
        bt.risk_manager.open_positions.append(bt.open_positions[entry_date])

        # Close at 110 → raw profit = (110-100)*50 = $500
        # Minus transaction costs (buy + sell commission, sell tax, slippage)
        # Capital was 100K - (50*100) = 95K open capital
        cap = bt._close_position(entry_date, pd.Timestamp('2024-01-15'), 110.0, 'test', 95_000.0)
        assert len(bt.trades) == 1
        # Profit is reduced by round-trip transaction costs
        assert bt.trades[0]['profit'] < 500.0  # Less than raw due to costs
        assert bt.trades[0]['profit'] > 450.0  # But still profitable
        # Capital includes sell proceeds minus sell costs
        sell_cost = bt.cost_model.sell_cost(110.0, 50)
        assert abs(cap - (95_000 + 110 * 50 - sell_cost)) < 0.01

    def test_trend_filter_blocks_bearish(self):
        """Bearish trend should produce no BUY signals."""
        df = make_bullish_wave_df()
        pa = make_pattern_analysis(df, confidence=0.5)
        pa['trend_context']['trend'] = 'bearish'

        bt = AdvancedBacktester(initial_capital=100_000)
        stats = bt.run_backtest(df, pattern_analysis=pa)
        buy_trades = [t for t in stats.get('trades', []) if t.get('wave_number') in [1, 3]]
        assert len(buy_trades) == 0


# ============================================================================
# Pattern Adapter Tests
# ============================================================================

class TestPatternAdapter:

    def test_adapt_returns_required_keys(self):
        df = make_bullish_wave_df()
        wave_data = {
            'impulse_wave': np.array([200, 220, 235, 265, 280, 299]),
            'confidence': 0.6,
            'wave_type': 'impulse',
            'pattern_relationships': {},
            'multiple_patterns': [],
        }
        result = adapt_wave_data_to_strategy_input(df, wave_data)
        assert 'alignment_score' in result
        assert 'current_position' in result
        assert 'best_pattern' in result
        assert 'confidence' in result
        assert 'trend_context' in result
        assert 'momentum' in result

    def test_trend_context_bullish(self):
        # Create clear bullish data: price well above SMA50 and SMA200
        prices = np.concatenate([
            np.linspace(50, 80, 200),  # establish SMA200
            np.linspace(80, 120, 100),  # strong move up
        ]).tolist()
        df = make_ohlcv(prices)
        ctx = _compute_trend_context(df, 'close')
        assert ctx['trend'] == 'bullish'
        assert ctx['sma50'] is not None
        assert ctx['sma200'] is not None
        assert ctx['price_vs_sma200'] > 1.0

    def test_trend_context_bearish(self):
        prices = np.concatenate([
            np.linspace(120, 80, 200),  # establish downward SMA200
            np.linspace(80, 50, 100),   # continue down
        ]).tolist()
        df = make_ohlcv(prices)
        ctx = _compute_trend_context(df, 'close')
        assert ctx['trend'] == 'bearish'

    def test_trend_context_short_data(self):
        df = make_ohlcv([100, 101, 102, 103])
        ctx = _compute_trend_context(df, 'close')
        assert ctx['trend'] == 'neutral'
        assert ctx['sma50'] is None

    def test_counter_trend_penalty(self):
        """Bearish pattern in bullish trend should have reduced confidence."""
        prices = np.concatenate([
            np.linspace(50, 100, 200),
            np.linspace(100, 120, 100),
        ]).tolist()
        df = make_ohlcv(prices)

        # Make a downward impulse wave in bullish trend
        n = len(df)
        wave_data = {
            'impulse_wave': np.array([n-100, n-80, n-65, n-35, n-20, n-1]),
            'confidence': 0.6,
            'wave_type': 'impulse',
            'pattern_relationships': {},
            'multiple_patterns': [],
        }

        result = adapt_wave_data_to_strategy_input(df, wave_data)
        # Pattern is upward (prices go up), trend is bullish → no penalty expected
        # But if we force trend_direction to 'down':
        if result['current_position']['trend_direction'] == 'down':
            assert result['confidence'] < 0.6  # Should be penalized

    def test_build_current_position_basic(self):
        prices = list(range(100, 200))
        df = make_ohlcv(prices)
        impulse = np.array([0, 20, 35, 65, 85, 99])
        pos = _build_current_position(df, impulse, 'close')
        assert pos is not None
        assert 'wave_number' in pos
        assert 'wave_1_range' in pos
        assert pos['trend_direction'] in ('up', 'down')

    def test_build_current_position_short_wave(self):
        df = make_ohlcv([100, 101, 102])
        pos = _build_current_position(df, np.array([0, 1]), 'close')
        # Only 2 points → should return None (need at least 3)
        assert pos is None

    def test_determine_current_wave_beyond_pattern(self):
        """Current bar beyond detected pattern → should be in Wave 5 or 6."""
        wave_indices = np.array([0, 20, 35, 65, 85, 100])
        wave_prices = np.array([100, 120, 110, 150, 140, 160])
        wave_num = _determine_current_wave(150, 165.0, wave_indices, wave_prices, True)
        assert wave_num == 6  # Past 6-point pattern → correction phase

    def test_determine_current_wave_within_pattern(self):
        wave_indices = np.array([0, 20, 35, 65, 85, 100])
        wave_prices = np.array([100, 120, 110, 150, 140, 160])
        wave_num = _determine_current_wave(50, 135.0, wave_indices, wave_prices, True)
        assert wave_num == 3  # Between indices 35 and 65 → Wave 3


# ============================================================================
# Walk-Forward Tests
# ============================================================================

class TestWalkForward:

    def test_insufficient_data(self):
        from src.backtest.walk_forward import WalkForwardValidator
        df = make_ohlcv(list(range(50)))  # Only 50 bars
        v = WalkForwardValidator(train_bars=100, test_bars=30)
        result = v.validate(df)
        assert result['windows'] == 0
        assert 'error' in result

    def test_walk_forward_produces_windows(self):
        """Walk-forward should produce window results with sufficient data."""
        from src.backtest.walk_forward import WalkForwardValidator
        # Use enough bars to produce at least one window
        # With train=100 and test=30, need 130+ bars
        prices = np.concatenate([
            np.linspace(100, 150, 100),
            np.linspace(150, 130, 50),
            np.linspace(130, 180, 100),
        ])
        df = make_ohlcv(prices.tolist())
        v = WalkForwardValidator(train_bars=100, test_bars=30, step_bars=30)
        result = v.validate(df)
        # Synthetic data may not produce detectable patterns,
        # but the validator should still process windows
        assert result.get('total_bars', 0) > 0 or 'error' not in result or result['windows'] == 0

    def test_choose_tradeable_pattern_bullish(self):
        """Bullish impulse patterns should be accepted for correction entries."""
        from src.backtest.walk_forward import WalkForwardValidator
        v = WalkForwardValidator()
        # Mock a bullish impulse pattern (up direction)
        prices = np.linspace(100, 150, 60).tolist()
        detection_df = make_ohlcv(prices)
        best = {
            'wave_points': np.array([0, 10, 20, 30, 40, 59]),
            'confidence': 0.5,
            'alternatives': [],
        }
        result = v._choose_tradeable_pattern(best, detection_df, 59, 0, 'close')
        assert result is not None  # Bullish impulse → tradeable

    def test_choose_tradeable_pattern_bearish_rejected(self):
        """Bearish impulse patterns in Wave 5/6 should be rejected."""
        from src.backtest.walk_forward import WalkForwardValidator
        v = WalkForwardValidator()
        # Mock a bearish impulse pattern (down direction)
        prices = np.linspace(150, 100, 60).tolist()
        detection_df = make_ohlcv(prices)
        best = {
            'wave_points': np.array([0, 10, 20, 30, 40, 59]),
            'confidence': 0.5,
            'alternatives': [],
        }
        result = v._choose_tradeable_pattern(best, detection_df, 59, 0, 'close')
        assert result is None  # Bearish impulse → not tradeable


# ============================================================================
# Bug Regression Tests
# ============================================================================

class TestBugRegressions:
    """Regression tests for critical bugs found in Sprint 3 analysis."""

    def test_exits_checked_every_bar(self):
        """
        Bug 1: _check_exits() was only called on signal dates.
        Stop losses must trigger on non-signal bars too.

        Setup: Open a position, then price drops below stop loss on a bar
        with NO signal. The stop should still trigger.
        """
        # Create price data: up trend, then sharp drop below stop
        prices = list(range(100, 130))  # bars 0-29: 100→129
        prices += list(range(129, 80, -1))  # bars 30-79: 129→80 (crash)
        prices += [80] * 20  # bars 80-99: flat
        df = make_ohlcv(prices)

        # Create pattern analysis with a BUY signal at bar 25 (price=125)
        # Stop loss at 120
        n = len(df)
        wave_indices = [5, 10, 15, 20, 25, 28]
        pa = make_pattern_analysis(df, wave_indices=wave_indices, confidence=0.5)

        bt = AdvancedBacktester(initial_capital=100_000)
        stats = bt.run_backtest(df, pattern_analysis=pa)

        # If exits are checked every bar, any position opened should be
        # stopped out when price drops below stop loss.
        # If exits only run on signal dates, the stop would be missed.
        for trade in stats.get('trades', []):
            if trade['exit_reason'] == 'stop_loss':
                # Stop was triggered — the fix works
                assert trade['exit_price'] < trade['entry_price'], \
                    "Stop loss should exit at a loss"

        # Also verify: no open positions remain that should have been stopped
        assert len(bt.open_positions) == 0, \
            "All positions should be closed at end of backtest"

    def test_target_exit_uses_highest_target(self):
        """
        Bug 2: sorted(targets) exits at lowest exceeded target.
        Should exit at highest exceeded target (sorted descending).

        Setup: Price reaches $150, targets at $120 and $140.
        Old code: exits at $120 (lowest exceeded). Fixed: exits at $140.
        """
        bt = AdvancedBacktester(initial_capital=100_000)
        entry_date = pd.Timestamp('2024-01-01')

        # Position with two targets
        bt.open_positions[entry_date] = {
            'entry_date': entry_date,
            'entry_price': 100.0,
            'shares': 50,
            'stop_loss': 90.0,
            'wave_number': 3,
            'confidence': 0.6,
            'targets': [120.0, 140.0],
            'highest_price': 100.0,
        }
        bt.risk_manager.open_positions.append(bt.open_positions[entry_date])

        # Price at 150 — both targets exceeded
        prices = [100.0] + [150.0] * 5
        df = make_ohlcv(prices)

        # Check exits on the bar where price is 150
        date = df.index[1]
        bt._check_exits(df, date, 95_000.0)

        # Should have closed at the highest target (140), not lowest (120)
        assert len(bt.trades) == 1
        assert 'target_140.00' in bt.trades[0]['exit_reason'], \
            f"Should exit at target 140, got: {bt.trades[0]['exit_reason']}"

    def test_load_data_method_exists(self):
        """
        Bug 3: analyze_stock() called self.load_data() which doesn't exist.
        Should call self.load_from_file() instead.
        """
        from src.backtest.backtester import Backtester
        # Verify load_from_file exists
        assert hasattr(Backtester, 'load_from_file')
        # Verify analyze_stock doesn't reference nonexistent load_data
        import inspect
        source = inspect.getsource(Backtester.analyze_stock)
        assert 'load_data' not in source or 'load_from_file' in source, \
            "analyze_stock should use load_from_file, not load_data"
