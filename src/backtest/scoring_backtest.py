"""
Scoring Validation Backtest System

Bridges the signal scoring pipeline (signal_scoring.py) with the trade
execution engine (strategy_advanced.py) to empirically validate whether
the scoring system's BUY/WATCH/AVOID classifications predict actual
forward returns.

Three validation modes:
1. Forward returns analysis — do BUY stocks outperform WATCH/AVOID?
2. Trade simulation — realistic P&L with costs, stops, position sizing
3. Walk-forward OOS — expanding window out-of-sample validation

Usage:
    from src.backtest.scoring_backtest import ScoringBacktester
    sb = ScoringBacktester()
    stats = sb.run_backtest(df, 'AAPL')
    fwd = sb.compute_forward_returns(df, scores)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging

from src.analysis.core.signal_scoring import analyze_stock, classify_action
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_best_impulse_wave
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input
from src.backtest.strategy_advanced import AdvancedBacktester
from src.backtest.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class ScoringSignalStrategy:
    """Strategy adapter that returns pre-computed signals from the scoring pipeline.

    Implements the same interface as MultiTimeframeAlignmentStrategy so it can
    be injected into AdvancedBacktester without modifying it.
    """

    def __init__(self, signals: List[Dict]):
        self.signals = signals

    def generate_signals(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[Dict]:
        return self.signals


class ScoringBacktester:
    """Validates the signal scoring pipeline via historical replay and trade simulation."""

    def __init__(self, initial_capital: float = 100_000, eval_interval: int = 5,
                 config: Optional[Dict] = None):
        """
        Args:
            initial_capital: Starting capital for trade simulation
            eval_interval: Re-score every N bars (default 5 ~ weekly)
            config: Passed through to AdvancedBacktester
        """
        self.initial_capital = initial_capital
        self.eval_interval = eval_interval
        self.config = config or {}

    def generate_historical_scores(self, df: pd.DataFrame, symbol: str,
                                   eval_interval: Optional[int] = None) -> List[Dict]:
        """Replay the scoring pipeline at historical points without future data leakage.

        Args:
            df: Full OHLCV DataFrame
            symbol: Stock symbol
            eval_interval: Override default eval interval

        Returns:
            List of scoring result dicts, each with 'action', 'score', 'bar_idx', 'date' appended
        """
        interval = eval_interval or self.eval_interval
        scores = []

        for bar_idx in range(60, len(df), interval):
            sub_df = df.iloc[:bar_idx + 1]
            try:
                result = analyze_stock(symbol, sub_df)
                if result is None:
                    continue
                action, reason = classify_action(result)
                result['action'] = action
                result['reason'] = reason
                result['bar_idx'] = bar_idx
                result['date'] = df.index[bar_idx]
                scores.append(result)
            except Exception as e:
                logger.debug(f"Scoring failed at bar {bar_idx} for {symbol}: {e}")
                continue

        return scores

    @staticmethod
    def scores_to_signals(scores: List[Dict],
                          actions: Tuple[str, ...] = ('BUY', 'STRONG BUY', 'BUY DIP', 'BUY CORRECTION')
                          ) -> List[Dict]:
        """Convert scoring results to AdvancedBacktester signal format.

        Args:
            scores: Output from generate_historical_scores()
            actions: Which action types to convert to BUY signals

        Returns:
            List of signal dicts compatible with AdvancedBacktester
        """
        signals = []
        for s in scores:
            if s.get('action') not in actions:
                continue
            signals.append({
                'date': s['date'],
                'type': 'BUY',
                'price': s['price'],
                'stop_loss': s['stop'],
                'wave_number': s.get('wave', 3),
                'confidence': s.get('conf', 0.5),
                'alignment': s.get('composite', 0.5),
            })
        return signals

    def build_pattern_analysis(self, df: pd.DataFrame, bar_idx: int) -> Dict:
        """Build pattern_analysis dict at a specific bar for AdvancedBacktester.

        Args:
            df: Full DataFrame
            bar_idx: Bar index to analyze up to

        Returns:
            pattern_analysis dict compatible with AdvancedBacktester
        """
        sub_df = df.iloc[:bar_idx + 1]
        if len(sub_df) < 60:
            return {}

        try:
            peaks, troughs = detect_peaks_troughs_enhanced(sub_df, column='close')
            if len(peaks) < 3 or len(troughs) < 3:
                return {}

            best = find_best_impulse_wave(sub_df, peaks, troughs, column='close')
            if best.get('wave_type') in ('no_candidates', 'no_pattern'):
                return {}

            wave_data = {
                'impulse_wave': best.get('wave_points', np.array([])),
                'confidence': best.get('confidence', 0),
                'wave_type': best.get('wave_type', 'unknown'),
                'pattern_relationships': {},
                'multiple_patterns': [],
            }
            return adapt_wave_data_to_strategy_input(sub_df, wave_data, column='close')
        except Exception as e:
            logger.debug(f"Pattern analysis failed at bar {bar_idx}: {e}")
            return {}

    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run full scoring-based backtest on a single stock.

        1. Generate historical scores
        2. Convert BUY scores to signals
        3. Build pattern analysis for trade execution context
        4. Feed into AdvancedBacktester

        Args:
            df: Full OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Stats dict from AdvancedBacktester plus scoring metadata
        """
        scores = self.generate_historical_scores(df, symbol)
        if not scores:
            return {
                'symbol': symbol,
                'error': 'no_scores',
                'total_trades': 0,
                'scores_generated': 0,
            }

        signals = self.scores_to_signals(scores)
        if not signals:
            return {
                'symbol': symbol,
                'error': 'no_buy_signals',
                'total_trades': 0,
                'scores_generated': len(scores),
                'action_counts': self._count_actions(scores),
            }

        # Build pattern analysis from the most recent BUY signal's bar position
        buy_scores = [s for s in scores if s.get('action') in ('BUY', 'STRONG BUY', 'BUY DIP', 'BUY CORRECTION')]
        last_buy_bar = buy_scores[-1]['bar_idx'] if buy_scores else len(df) - 1
        pa = self.build_pattern_analysis(df, last_buy_bar)
        if not pa:
            pa = self.build_pattern_analysis(df, len(df) - 1)
        if not pa:
            # Minimal fallback so AdvancedBacktester doesn't crash
            pa = {
                'alignment_score': 0.5,
                'current_position': {'wave_number': 3, 'trend_direction': 'up'},
                'trend_context': {'trend': 'neutral'},
                'momentum': {},
                'confidence': 0.5,
            }

        # Inject signals via adapter strategy
        strategy = ScoringSignalStrategy(signals)
        bt = AdvancedBacktester(
            initial_capital=self.initial_capital,
            config=self.config,
        )
        bt.strategies['scoring'] = strategy
        stats = bt.run_backtest(df, pattern_analysis=pa, strategy_name='scoring')

        # Enrich with scoring metadata
        stats['symbol'] = symbol
        stats['scores_generated'] = len(scores)
        stats['buy_signals'] = len(signals)
        stats['action_counts'] = self._count_actions(scores)

        return stats

    @staticmethod
    def compute_forward_returns(df: pd.DataFrame, scores: List[Dict],
                                horizons: Tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
        """Measure actual forward returns for each historical score.

        Args:
            df: Full OHLCV DataFrame
            scores: Output from generate_historical_scores()
            horizons: Forward-looking periods in bars (default: 5d, 20d, 60d)

        Returns:
            DataFrame with columns: symbol, date, action, score, wave, fwd_5d, fwd_20d, fwd_60d
        """
        rows = []
        n = len(df)

        for s in scores:
            bar_idx = s['bar_idx']
            price = s['price']
            row = {
                'symbol': s.get('symbol', ''),
                'date': s['date'],
                'action': s['action'],
                'score': s.get('score', 0),
                'wave': s.get('wave', 0),
                'price': price,
            }
            for h in horizons:
                fwd_idx = bar_idx + h
                if fwd_idx < n:
                    fwd_price = float(df['close'].iloc[fwd_idx])
                    row[f'fwd_{h}d'] = (fwd_price / price - 1) * 100
                else:
                    row[f'fwd_{h}d'] = np.nan
            rows.append(row)

        if not rows:
            cols = ['symbol', 'date', 'action', 'score', 'wave', 'price']
            cols += [f'fwd_{h}d' for h in horizons]
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(rows)

    @staticmethod
    def analyze_score_buckets(forward_df: pd.DataFrame) -> Dict:
        """Analyze forward returns by action bucket.

        Args:
            forward_df: Output from compute_forward_returns()

        Returns:
            Dict with 'by_action' breakdown and 'correlation' stats
        """
        if forward_df.empty:
            return {'by_action': {}, 'correlation': {}, 'tests': {}}

        fwd_cols = [c for c in forward_df.columns if c.startswith('fwd_')]

        # Per-action breakdown
        by_action = {}
        for action, group in forward_df.groupby('action'):
            action_stats = {'count': len(group)}
            for col in fwd_cols:
                valid = group[col].dropna()
                if len(valid) > 0:
                    action_stats[f'{col}_mean'] = round(float(valid.mean()), 3)
                    action_stats[f'{col}_median'] = round(float(valid.median()), 3)
                    action_stats[f'{col}_win_pct'] = round(float((valid > 0).mean()) * 100, 1)
                else:
                    action_stats[f'{col}_mean'] = None
                    action_stats[f'{col}_median'] = None
                    action_stats[f'{col}_win_pct'] = None
            by_action[action] = action_stats

        # Score-return correlation (Spearman)
        from scipy import stats as sp_stats
        correlation = {}
        for col in fwd_cols:
            valid_mask = forward_df[col].notna() & forward_df['score'].notna()
            valid = forward_df[valid_mask]
            if len(valid) >= 5:
                rho, p_val = sp_stats.spearmanr(valid['score'], valid[col])
                correlation[col] = {
                    'spearman_rho': round(float(rho), 4),
                    'p_value': round(float(p_val), 4),
                }
            else:
                correlation[col] = {'spearman_rho': None, 'p_value': None}

        # Statistical tests
        tests = {}
        buy_actions = ('BUY', 'STRONG BUY', 'BUY DIP', 'BUY CORRECTION')
        for col in fwd_cols:
            buy_returns = forward_df[
                forward_df['action'].isin(buy_actions) & forward_df[col].notna()
            ][col]
            watch_returns = forward_df[
                (forward_df['action'] == 'WATCH') & forward_df[col].notna()
            ][col]

            col_tests = {}
            # T-test: BUY mean > 0
            if len(buy_returns) >= 3:
                t_stat, p_val = sp_stats.ttest_1samp(buy_returns, 0)
                col_tests['buy_gt_zero'] = {
                    't_stat': round(float(t_stat), 3),
                    'p_value': round(float(p_val / 2), 4),  # one-tailed
                    'significant': bool(p_val / 2 < 0.05),
                    'n': len(buy_returns),
                }
            # T-test: BUY mean > WATCH mean
            if len(buy_returns) >= 3 and len(watch_returns) >= 3:
                t_stat, p_val = sp_stats.ttest_ind(buy_returns, watch_returns)
                col_tests['buy_gt_watch'] = {
                    't_stat': round(float(t_stat), 3),
                    'p_value': round(float(p_val / 2), 4),  # one-tailed
                    'significant': bool(p_val / 2 < 0.05),
                    'n_buy': len(buy_returns),
                    'n_watch': len(watch_returns),
                }
            tests[col] = col_tests

        return {
            'by_action': by_action,
            'correlation': correlation,
            'tests': tests,
        }

    @staticmethod
    def _count_actions(scores: List[Dict]) -> Dict[str, int]:
        counts = {}
        for s in scores:
            a = s.get('action', 'UNKNOWN')
            counts[a] = counts.get(a, 0) + 1
        return counts


class ScoringWalkForwardValidator:
    """Walk-forward out-of-sample validation using the scoring pipeline.

    Uses expanding windows: training always starts from bar 0.
    Scores are generated on the test window only, converted to signals,
    and executed via AdvancedBacktester. Only trades entered during the
    test window count as OOS.
    """

    def __init__(self, train_bars: int = 252, test_bars: int = 63,
                 step_bars: int = 63, eval_interval: int = 5,
                 initial_capital: float = 100_000, config: Optional[Dict] = None):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.eval_interval = eval_interval
        self.initial_capital = initial_capital
        self.config = config or {}

    def validate(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run walk-forward validation on a single stock.

        Args:
            df: Full OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Aggregated OOS results
        """
        n = len(df)
        min_required = self.train_bars + self.test_bars
        if n < min_required:
            return {
                'symbol': symbol,
                'error': f'Need {min_required} bars, have {n}',
                'windows': 0,
                'total_trades': 0,
            }

        all_trades = []
        window_results = []

        train_end = self.train_bars
        while train_end + self.test_bars <= n:
            test_end = min(train_end + self.test_bars, n)

            result = self._run_single_window(df, symbol, train_end, test_end)
            if result:
                window_results.append(result)
                all_trades.extend(result.get('trades', []))

            train_end += self.step_bars
            if test_end >= n:
                break

        return self._aggregate_results(window_results, all_trades, symbol)

    def _run_single_window(self, df: pd.DataFrame, symbol: str,
                           train_end: int, test_end: int) -> Optional[Dict]:
        """Run scoring + backtest for a single walk-forward window."""
        sb = ScoringBacktester(
            initial_capital=self.initial_capital,
            eval_interval=self.eval_interval,
            config=self.config,
        )

        # Generate scores only in the test window (using data up to each eval point)
        test_scores = []
        for bar_idx in range(train_end, test_end, self.eval_interval):
            sub_df = df.iloc[:bar_idx + 1]
            try:
                result = analyze_stock(symbol, sub_df)
                if result is None:
                    continue
                action, reason = classify_action(result)
                result['action'] = action
                result['reason'] = reason
                result['bar_idx'] = bar_idx
                result['date'] = df.index[bar_idx]
                test_scores.append(result)
            except Exception:
                continue

        if not test_scores:
            return None

        signals = sb.scores_to_signals(test_scores)
        if not signals:
            return None

        # Build pattern analysis from latest test point
        pa = sb.build_pattern_analysis(df, test_end - 1)
        if not pa:
            pa = {
                'alignment_score': 0.5,
                'current_position': {'wave_number': 3, 'trend_direction': 'up'},
                'trend_context': {'trend': 'neutral'},
                'momentum': {},
                'confidence': 0.5,
            }

        # Run backtest on the full data up to test_end
        segment_df = df.iloc[:test_end]
        strategy = ScoringSignalStrategy(signals)
        bt = AdvancedBacktester(
            initial_capital=self.initial_capital,
            config=self.config,
        )
        bt.strategies['scoring'] = strategy
        stats = bt.run_backtest(segment_df, pattern_analysis=pa, strategy_name='scoring')

        # Filter to OOS trades only (entered during test window)
        test_start_date = df.index[train_end]
        oos_trades = [t for t in stats.get('trades', [])
                      if t['entry_date'] >= test_start_date]

        oos_profit = sum(t['profit'] for t in oos_trades)
        oos_wins = sum(1 for t in oos_trades if t['profit'] > 0)

        return {
            'train_end': str(df.index[train_end])[:10],
            'test_end': str(df.index[test_end - 1])[:10],
            'test_scores': len(test_scores),
            'test_signals': len(signals),
            'oos_trades': len(oos_trades),
            'oos_wins': oos_wins,
            'oos_profit': round(oos_profit, 2),
            'oos_win_rate': oos_wins / len(oos_trades) if oos_trades else 0,
            'trades': oos_trades,
        }

    @staticmethod
    def _aggregate_results(window_results: List[Dict], all_trades: List[Dict],
                           symbol: str) -> Dict:
        """Aggregate OOS results across all walk-forward windows."""
        n_windows = len(window_results)
        if n_windows == 0:
            return {
                'symbol': symbol,
                'windows': 0,
                'total_trades': 0,
                'status': 'no_valid_windows',
            }

        total_trades = len(all_trades)
        total_profit = sum(t['profit'] for t in all_trades)
        wins = sum(1 for t in all_trades if t['profit'] > 0)
        losses = total_trades - wins

        win_profits = [t['profit'] for t in all_trades if t['profit'] > 0]
        loss_profits = [t['profit'] for t in all_trades if t['profit'] <= 0]

        avg_win = float(np.mean(win_profits)) if win_profits else 0
        avg_loss = float(np.mean(loss_profits)) if loss_profits else 0
        profit_factor = (abs(sum(win_profits) / sum(loss_profits))
                         if loss_profits and sum(loss_profits) != 0
                         else float('inf'))

        profitable_windows = sum(1 for w in window_results if w['oos_profit'] > 0)

        return {
            'symbol': symbol,
            'windows': n_windows,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'oos_win_rate': wins / total_trades if total_trades > 0 else 0,
            'oos_profit': round(total_profit, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'profitable_windows': profitable_windows,
            'window_win_rate': profitable_windows / n_windows if n_windows > 0 else 0,
            'window_details': [
                {k: v for k, v in w.items() if k != 'trades'}
                for w in window_results
            ],
            'trades': all_trades,
        }
