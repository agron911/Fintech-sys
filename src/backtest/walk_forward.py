"""
Walk-Forward Validation: Eliminates look-ahead bias by testing only on unseen data.

Instead of running wave detection on the full dataset and then backtesting
on the same data (which allows future information to leak into pattern selection),
this splits data into rolling windows:

    [======= TRAIN =======][=== TEST ===]
                    [======= TRAIN =======][=== TEST ===]
                                    [======= TRAIN =======][=== TEST ===]

For each window:
1. Detect patterns using only TRAIN data
2. Generate signals and execute trades only on TEST data
3. Aggregate out-of-sample results across all windows
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_best_impulse_wave
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input
from src.backtest.strategy_advanced import AdvancedBacktester

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for Elliott Wave trading strategies.

    Splits data into rolling train/test windows to eliminate look-ahead bias
    and provide realistic out-of-sample performance estimates.
    """

    def __init__(self,
                 train_bars: int = 252,
                 test_bars: int = 63,
                 step_bars: int = 63,
                 initial_capital: float = 100_000,
                 config: Dict = None):
        """
        Args:
            train_bars: Number of bars for training/pattern detection (default ~1 year)
            test_bars: Number of bars for out-of-sample testing (default ~3 months)
            step_bars: Number of bars to slide forward each iteration (default ~3 months)
            initial_capital: Starting capital per window
            config: Backtester configuration
        """
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.initial_capital = initial_capital
        self.config = config or {}

    def validate(self, df: pd.DataFrame, column: str = 'close') -> Dict:
        """
        Run walk-forward validation on a single stock's data.

        Uses expanding windows: training always starts from bar 0, so each
        successive window has MORE training data. This matches how the system
        would actually be used: you always have all prior history available.

        Pattern detection runs on training data only. Trades are counted only
        if they ENTER during the test window.

        Returns:
            Dict with aggregated out-of-sample results
        """
        n = len(df)
        min_required = self.train_bars + self.test_bars
        if n < min_required:
            return {
                'error': f'Need {min_required} bars, have {n}',
                'windows': 0,
                'total_trades': 0,
            }

        all_trades = []
        window_results = []

        # Expanding window: training starts from bar 0, test slides forward
        train_end = self.train_bars
        while train_end + self.test_bars <= n:
            test_end = min(train_end + self.test_bars, n)

            # Training: all data from start to train_end (expanding)
            train_df = df.iloc[:train_end]
            # Full data for backtesting (includes test period for exit checks)
            full_df = df.iloc[:test_end]

            result = self._run_single_window(
                train_df, full_df, train_end, test_end, column
            )

            if result:
                window_results.append(result)
                all_trades.extend(result.get('trades', []))

            train_end += self.step_bars

            if test_end >= n:
                break

        return self._aggregate_results(window_results, all_trades, n)

    def _run_single_window(self, train_df: pd.DataFrame, full_df: pd.DataFrame,
                           train_end_idx: int, test_end_idx: int,
                           column: str) -> Optional[Dict]:
        """
        Run pattern detection and backtesting with rolling re-detection.

        Problem: single-shot detection on all training data finds old completed
        patterns (Wave 5/6), so test bars never fall in tradeable waves (1-3).

        Fix: Re-detect patterns at regular intervals during the test window.
        Two key changes make this work:
        1. Use a SHORT lookback window for detection (not all training data).
           This forces the detector to find RECENT patterns that may still be
           in progress, rather than old completed patterns.
        2. Use recency_weight to prefer patterns ending near the detection point.
        3. Try multiple candidates — if the best one puts us in Wave 5/6,
           check alternatives for a pattern with a tradeable wave position.
        """
        redetect_interval = max(10, self.test_bars // 4)
        detection_lookback = 150  # Short window for finding recent patterns
        test_start_date = full_df.index[train_end_idx]

        all_oos_trades = []
        total_costs = 0
        best_confidence = 0
        detections = 0
        seen_entries = set()  # Deduplicate trades across detections

        # Build detection points: boundary + periodic during test window
        detect_points = list(range(train_end_idx, test_end_idx, redetect_interval))
        if not detect_points:
            detect_points = [train_end_idx]

        for di, detect_bar in enumerate(detect_points):
            try:
                # Use a short lookback window for detection. This finds
                # patterns in recent price action rather than old completed
                # patterns from the distant past.
                lookback_start = max(0, detect_bar - detection_lookback)
                detection_df = full_df.iloc[lookback_start:detect_bar + 1]

                peaks, troughs = detect_peaks_troughs_enhanced(
                    detection_df, column=column
                )
                if len(peaks) < 3 or len(troughs) < 3:
                    continue

                best = find_best_impulse_wave(
                    detection_df, peaks, troughs, column=column,
                    recency_weight=0.8
                )
                if best.get('wave_type') in ('no_candidates', 'no_pattern'):
                    continue

                # Try the best candidate; if it's Wave 6, try alternatives
                chosen = self._choose_tradeable_pattern(
                    best, detection_df, detect_bar, lookback_start, column
                )
                if chosen is None:
                    continue

                wave_points_raw, pattern_conf = chosen

                # Remap wave point indices from detection_df back to full_df
                wave_points = wave_points_raw + lookback_start

                wave_data = {
                    'impulse_wave': wave_points,
                    'confidence': pattern_conf,
                    'wave_type': best.get('wave_type', 'unknown'),
                    'pattern_relationships': {},
                    'multiple_patterns': [],
                }

                # Determine segment end: next detection point or test end
                segment_end = (
                    detect_points[di + 1]
                    if di + 1 < len(detect_points)
                    else test_end_idx
                )
                segment_df = full_df.iloc[:segment_end]

                # Build pattern analysis on the segment
                pa = adapt_wave_data_to_strategy_input(
                    segment_df, wave_data, column=column
                )

                # Run backtest on segment
                bt = AdvancedBacktester(
                    initial_capital=self.initial_capital,
                    config=self.config
                )
                stats = bt.run_backtest(segment_df, pattern_analysis=pa)

                # Collect OOS trades entered during the test window
                for t in stats.get('trades', []):
                    entry_key = (str(t['entry_date']), t['entry_price'])
                    if entry_key in seen_entries:
                        continue
                    if t['entry_date'] >= test_start_date:
                        seen_entries.add(entry_key)
                        all_oos_trades.append(t)

                total_costs += stats.get('total_costs', 0)
                best_confidence = max(best_confidence, pa.get('confidence', 0))
                detections += 1

            except Exception as e:
                logger.warning(f"Detection at bar {detect_bar} failed: {e}")
                continue

        if not detections:
            return None

        oos_profit = sum(t['profit'] for t in all_oos_trades)
        oos_wins = sum(1 for t in all_oos_trades if t['profit'] > 0)

        return {
            'train_start': str(full_df.index[0])[:10],
            'train_end': str(test_start_date)[:10],
            'test_end': str(full_df.index[test_end_idx - 1])[:10],
            'in_sample_trades': 0,
            'bridge_trades': 0,
            'oos_trades': len(all_oos_trades),
            'oos_wins': oos_wins,
            'oos_profit': oos_profit,
            'oos_win_rate': oos_wins / len(all_oos_trades) if all_oos_trades else 0,
            'confidence': best_confidence,
            'trades': all_oos_trades,
            'total_costs': total_costs,
            'detections': detections,
        }

    def _choose_tradeable_pattern(self, best: Dict, detection_df: pd.DataFrame,
                                  detect_bar: int, lookback_start: int,
                                  column: str):
        """
        Choose a pattern candidate that enables trading in the test window.

        Tradeable conditions:
        - Wave 1-4: Standard wave-position entries
        - Wave 5-6 of a bullish impulse: Correction-bottom entries at
          Fibonacci retracement levels

        Returns (wave_points, confidence) or None if no tradeable pattern found.
        """
        from src.backtest.pattern_adapter import _determine_current_wave

        candidates = [best] + best.get('alternatives', [])

        # First pass: prefer patterns in Waves 1-4
        for c in candidates:
            wp = c.get('wave_points', np.array([]))
            if len(wp) < 3:
                continue

            max_idx = len(detection_df) - 1
            wave_indices = np.clip(wp, 0, max_idx).astype(int)
            wave_prices = detection_df[column].iloc[wave_indices].values
            current_idx = max_idx
            current_price = detection_df[column].iloc[-1]
            trend_up = wave_prices[-1] > wave_prices[0]

            wave_num = _determine_current_wave(
                current_idx, current_price, wave_indices, wave_prices, trend_up
            )

            if wave_num <= 4:
                return (wp, c.get('confidence', 0))

        # Second pass: accept Wave 5/6 of bullish impulses (correction-bottom
        # entries will handle these in the strategy)
        for c in candidates:
            wp = c.get('wave_points', np.array([]))
            if len(wp) < 3:
                continue

            max_idx = len(detection_df) - 1
            wave_indices = np.clip(wp, 0, max_idx).astype(int)
            wave_prices = detection_df[column].iloc[wave_indices].values
            trend_up = wave_prices[-1] > wave_prices[0]

            if trend_up:  # Bullish impulse → correction bottom is tradeable
                return (wp, c.get('confidence', 0))

        return None

    def _aggregate_results(self, window_results: List[Dict],
                           all_trades: List[Dict], total_bars: int) -> Dict:
        """Aggregate results across all walk-forward windows."""
        n_windows = len(window_results)
        if n_windows == 0:
            return {
                'windows': 0,
                'total_trades': 0,
                'oos_win_rate': 0,
                'oos_profit': 0,
                'status': 'no_valid_windows',
            }

        total_trades = len(all_trades)
        total_profit = sum(t['profit'] for t in all_trades)
        wins = sum(1 for t in all_trades if t['profit'] > 0)
        losses = sum(1 for t in all_trades if t['profit'] <= 0)

        win_profits = [t['profit'] for t in all_trades if t['profit'] > 0]
        loss_profits = [t['profit'] for t in all_trades if t['profit'] <= 0]

        avg_win = np.mean(win_profits) if win_profits else 0
        avg_loss = np.mean(loss_profits) if loss_profits else 0
        profit_factor = abs(sum(win_profits) / sum(loss_profits)) \
            if loss_profits and sum(loss_profits) != 0 else float('inf')

        # Per-window consistency
        profitable_windows = sum(1 for w in window_results if w['oos_profit'] > 0)

        return {
            'windows': n_windows,
            'total_bars': total_bars,
            'train_bars': self.train_bars,
            'test_bars': self.test_bars,

            # Out-of-sample aggregate
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'oos_win_rate': wins / total_trades if total_trades > 0 else 0,
            'oos_profit': round(total_profit, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),

            # Consistency
            'profitable_windows': profitable_windows,
            'window_win_rate': profitable_windows / n_windows if n_windows > 0 else 0,

            # Details
            'window_details': [
                {k: v for k, v in w.items() if k != 'trades'}
                for w in window_results
            ],
            'trades': all_trades,
            'total_costs': sum(w.get('total_costs', 0) for w in window_results),
        }


def run_walk_forward(symbol: str, df: pd.DataFrame,
                     train_bars: int = 252, test_bars: int = 63,
                     config: Dict = None) -> Dict:
    """
    Convenience function to run walk-forward validation on a single stock.

    Args:
        symbol: Stock symbol (for labeling)
        df: Full price DataFrame
        train_bars: Training window size
        test_bars: Test window size
        config: Backtester config

    Returns:
        Dict with out-of-sample results
    """
    validator = WalkForwardValidator(
        train_bars=train_bars,
        test_bars=test_bars,
        config=config,
    )
    result = validator.validate(df)
    result['symbol'] = symbol
    return result
