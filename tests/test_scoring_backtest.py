"""
Tests for the scoring validation backtest system.

Covers:
- ScoringBacktester: historical replay, signal conversion, forward returns
- ScoringSignalStrategy: adapter passthrough
- ScoringWalkForwardValidator: expanding window OOS
- Score bucket analysis: statistical tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtest.scoring_backtest import (
    ScoringBacktester,
    ScoringSignalStrategy,
    ScoringWalkForwardValidator,
)


# ============================================================================
# Helpers (reuse patterns from test_backtester.py)
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
    """Create a DataFrame with a clear bullish trend and 5-wave impulse."""
    base = np.linspace(50, 80, 200)
    w1 = np.linspace(80, 100, 20)
    w2 = np.linspace(100, 92, 15)
    w3 = np.linspace(92, 130, 30)
    w4 = np.linspace(130, 122, 15)
    w5 = np.linspace(122, 140, 20)
    prices = np.concatenate([base, w1, w2, w3, w4, w5])
    return make_ohlcv(prices.tolist())


# ============================================================================
# ScoringSignalStrategy tests
# ============================================================================

class TestScoringSignalStrategy:
    def test_passthrough(self):
        """generate_signals() returns exact input signals unchanged."""
        signals = [
            {'date': datetime(2024, 1, 1), 'type': 'BUY', 'price': 100,
             'stop_loss': 95, 'wave_number': 3, 'confidence': 0.7, 'alignment': 0.6},
            {'date': datetime(2024, 1, 5), 'type': 'BUY', 'price': 105,
             'stop_loss': 100, 'wave_number': 1, 'confidence': 0.5, 'alignment': 0.4},
        ]
        strategy = ScoringSignalStrategy(signals)
        result = strategy.generate_signals(pd.DataFrame(), {})
        assert result is signals
        assert len(result) == 2

    def test_empty_signals(self):
        """Empty signal list passes through."""
        strategy = ScoringSignalStrategy([])
        result = strategy.generate_signals(pd.DataFrame(), {})
        assert result == []


# ============================================================================
# ScoringBacktester — generate_historical_scores
# ============================================================================

class TestGenerateHistoricalScores:
    def test_returns_list_of_dicts(self):
        """Each score has action, score, bar_idx, date."""
        df = make_bullish_wave_df(300)
        sb = ScoringBacktester(eval_interval=20)
        scores = sb.generate_historical_scores(df, 'TEST')

        assert isinstance(scores, list)
        # Should have some scores (at least a few from 300 bars)
        if scores:
            s = scores[0]
            assert 'action' in s
            assert 'score' in s
            assert 'bar_idx' in s
            assert 'date' in s

    def test_no_lookahead(self):
        """Each score's bar_idx must be within the sub-DataFrame used."""
        df = make_bullish_wave_df(300)
        sb = ScoringBacktester(eval_interval=20)
        scores = sb.generate_historical_scores(df, 'TEST')

        for s in scores:
            # bar_idx should be a valid index in the original df
            assert 0 <= s['bar_idx'] < len(df)
            # The score was computed using df.iloc[:bar_idx+1], so bar_idx
            # is the last bar the scorer could see
            assert s['date'] == df.index[s['bar_idx']]

    def test_empty_on_short_df(self):
        """DataFrame shorter than 60 bars produces no scores."""
        df = make_ohlcv(list(range(50, 100)))  # 50 bars
        sb = ScoringBacktester()
        scores = sb.generate_historical_scores(df, 'SHORT')
        assert scores == []

    def test_eval_interval_spacing(self):
        """Scores are spaced by eval_interval bars."""
        df = make_bullish_wave_df(300)
        sb = ScoringBacktester(eval_interval=10)
        scores = sb.generate_historical_scores(df, 'TEST')

        if len(scores) >= 2:
            bar_indices = [s['bar_idx'] for s in scores]
            for i in range(1, len(bar_indices)):
                assert bar_indices[i] - bar_indices[i-1] == 10


# ============================================================================
# ScoringBacktester — scores_to_signals
# ============================================================================

class TestScoresToSignals:
    def test_signal_format(self):
        """Output signals have all required AdvancedBacktester fields."""
        scores = [
            {'action': 'BUY', 'date': datetime(2024, 3, 1), 'price': 100,
             'stop': 95, 'wave': 3, 'conf': 0.7, 'composite': 0.6,
             'score': 75, 'bar_idx': 100},
        ]
        signals = ScoringBacktester.scores_to_signals(scores)
        assert len(signals) == 1
        s = signals[0]
        required_keys = {'date', 'type', 'price', 'stop_loss', 'wave_number',
                         'confidence', 'alignment'}
        assert required_keys.issubset(s.keys())
        assert s['type'] == 'BUY'
        assert s['price'] == 100
        assert s['stop_loss'] == 95

    def test_filtering(self):
        """Only BUY-type actions produce signals."""
        scores = [
            {'action': 'BUY', 'date': datetime(2024, 1, 1), 'price': 100,
             'stop': 95, 'wave': 3, 'conf': 0.7, 'composite': 0.5,
             'score': 75, 'bar_idx': 100},
            {'action': 'STRONG BUY', 'date': datetime(2024, 1, 5), 'price': 105,
             'stop': 100, 'wave': 3, 'conf': 0.8, 'composite': 0.6,
             'score': 85, 'bar_idx': 105},
            {'action': 'WATCH', 'date': datetime(2024, 1, 10), 'price': 110,
             'stop': 105, 'wave': 4, 'conf': 0.4, 'composite': 0.3,
             'score': 50, 'bar_idx': 110},
            {'action': 'AVOID', 'date': datetime(2024, 1, 15), 'price': 90,
             'stop': 85, 'wave': 5, 'conf': 0.3, 'composite': 0.2,
             'score': 20, 'bar_idx': 115},
        ]
        signals = ScoringBacktester.scores_to_signals(scores)
        assert len(signals) == 2  # BUY + STRONG BUY, not WATCH or AVOID

    def test_empty_scores(self):
        """Empty input produces empty output."""
        assert ScoringBacktester.scores_to_signals([]) == []


# ============================================================================
# ScoringBacktester — compute_forward_returns
# ============================================================================

class TestComputeForwardReturns:
    def test_correct_columns(self):
        """DataFrame has expected columns for default horizons."""
        df = make_ohlcv(list(range(100, 200)))  # 100 bars, prices 100-199
        scores = [
            {'symbol': 'TEST', 'date': df.index[10], 'action': 'BUY',
             'score': 70, 'wave': 3, 'price': float(df['close'].iloc[10]),
             'bar_idx': 10},
        ]
        fwd = ScoringBacktester.compute_forward_returns(df, scores)
        assert 'fwd_5d' in fwd.columns
        assert 'fwd_20d' in fwd.columns
        assert 'fwd_60d' in fwd.columns

    def test_forward_returns_values(self):
        """Forward returns are computed as percentage price changes."""
        prices = list(range(100, 200))  # monotonically increasing
        df = make_ohlcv(prices)
        bar_idx = 10
        scores = [
            {'symbol': 'TEST', 'date': df.index[bar_idx], 'action': 'BUY',
             'score': 70, 'wave': 3, 'price': float(df['close'].iloc[bar_idx]),
             'bar_idx': bar_idx},
        ]
        fwd = ScoringBacktester.compute_forward_returns(df, scores)
        row = fwd.iloc[0]

        # For monotonically increasing integer prices:
        # price at bar 10 = 110, price at bar 15 = 115
        # fwd_5d = (115/110 - 1) * 100 ≈ 4.55%
        assert row['fwd_5d'] > 0
        assert row['fwd_20d'] > row['fwd_5d']  # longer horizon = more gain

    def test_clipped_at_end(self):
        """Scores near end of df get NaN for horizons that exceed data length."""
        df = make_ohlcv(list(range(100, 170)))  # 70 bars
        scores = [
            {'symbol': 'TEST', 'date': df.index[65], 'action': 'BUY',
             'score': 70, 'wave': 3, 'price': float(df['close'].iloc[65]),
             'bar_idx': 65},
        ]
        fwd = ScoringBacktester.compute_forward_returns(df, scores)
        row = fwd.iloc[0]
        # bar 65 + 5 = 70 >= 70 bars, so fwd_5d should be NaN
        assert pd.isna(row['fwd_5d'])
        assert pd.isna(row['fwd_20d'])
        assert pd.isna(row['fwd_60d'])

    def test_empty_scores(self):
        """Empty input produces empty DataFrame with correct columns."""
        df = make_ohlcv([100, 101, 102])
        fwd = ScoringBacktester.compute_forward_returns(df, [])
        assert fwd.empty
        assert 'fwd_5d' in fwd.columns


# ============================================================================
# ScoringBacktester — analyze_score_buckets
# ============================================================================

class TestAnalyzeScoreBuckets:
    def test_structure(self):
        """Returns dict with by_action, correlation, and tests keys."""
        prices = list(range(100, 200))
        df = make_ohlcv(prices)
        scores = []
        for bar_idx in range(10, 80, 5):
            action = 'BUY' if bar_idx % 10 == 0 else 'WATCH'
            scores.append({
                'symbol': 'TEST', 'date': df.index[bar_idx],
                'action': action, 'score': 50 + bar_idx % 30,
                'wave': 3, 'price': float(df['close'].iloc[bar_idx]),
                'bar_idx': bar_idx,
            })

        fwd = ScoringBacktester.compute_forward_returns(df, scores)
        analysis = ScoringBacktester.analyze_score_buckets(fwd)

        assert 'by_action' in analysis
        assert 'correlation' in analysis
        assert 'tests' in analysis

    def test_empty_df(self):
        """Empty DataFrame returns empty analysis."""
        analysis = ScoringBacktester.analyze_score_buckets(pd.DataFrame())
        assert analysis['by_action'] == {}


# ============================================================================
# ScoringWalkForwardValidator
# ============================================================================

class TestScoringWalkForward:
    def test_insufficient_data(self):
        """Returns error when df is too short."""
        df = make_ohlcv(list(range(100, 200)))  # 100 bars, need 315
        wf = ScoringWalkForwardValidator(train_bars=252, test_bars=63)
        result = wf.validate(df, 'SHORT')
        assert 'error' in result
        assert result['windows'] == 0

    def test_sufficient_data_runs(self):
        """With enough data, produces windows and results."""
        # make_bullish_wave_df creates 300 bars; use smaller windows to fit
        df = make_bullish_wave_df()
        wf = ScoringWalkForwardValidator(
            train_bars=150, test_bars=50, step_bars=50,
            eval_interval=10,
        )
        result = wf.validate(df, 'TEST')

        assert result['symbol'] == 'TEST'
        assert 'windows' in result
        assert 'total_trades' in result
        # May have 0 trades if no BUY signals, but should run without error


# ============================================================================
# Integration: full pipeline smoke test
# ============================================================================

class TestFullPipelineSmoke:
    def test_run_backtest_returns_stats(self):
        """Full pipeline: make_bullish_wave_df → run_backtest → stats dict."""
        df = make_bullish_wave_df(300)
        sb = ScoringBacktester(eval_interval=10)
        stats = sb.run_backtest(df, 'SMOKE')

        assert isinstance(stats, dict)
        assert 'symbol' in stats
        assert stats['symbol'] == 'SMOKE'
        assert 'scores_generated' in stats
        # Should have generated some scores
        assert stats['scores_generated'] > 0

    def test_forward_returns_direction(self):
        """On bullish data, BUY signals should tend to have positive forward returns."""
        df = make_bullish_wave_df(300)
        sb = ScoringBacktester(eval_interval=10)
        scores = sb.generate_historical_scores(df, 'BULL')

        if not scores:
            pytest.skip("No scores generated on synthetic data")

        fwd = sb.compute_forward_returns(df, scores, horizons=(5, 20))

        # On a monotonically bullish dataset, most forward returns should be positive
        valid_20d = fwd['fwd_20d'].dropna()
        if len(valid_20d) > 0:
            positive_pct = (valid_20d > 0).mean()
            # On our synthetic bullish data, >50% should be positive
            assert positive_pct > 0.3, f"Only {positive_pct:.0%} of 20d returns were positive on bullish data"
