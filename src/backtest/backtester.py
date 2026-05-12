import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from src.utils.config import load_config
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_elliott_wave_pattern_enhanced
from src.backtest.strategy_advanced import AdvancedBacktester, MultiTimeframeAlignmentStrategy
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input
from src.backtest.monte_carlo import MonteCarloSimulator
import logging

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.results = []
        self.data_dir = Path(config['stk2_dir'])
        self.processed_dir = Path(config['processed_dir'])

    def run(self, symbols: List[str]):
        """Run walk-forward backtest for a list of symbols."""
        window_size = self.config.get('backtest_window_size', 504)
        step_size = self.config.get('backtest_step_size', 63)

        for symbol in symbols:
            df = self.load_from_file(symbol)
            if df is None or 'close' not in df.columns:
                logger.info(f"Skipping {symbol}: Data not found or missing 'close' column")
                continue

            all_signals = self._walk_forward_detect(df, window_size, step_size)

            if not all_signals:
                logger.info(f"Skipping {symbol}: No signals generated across {len(df)} bars")
                continue

            advanced = AdvancedBacktester(
                initial_capital=self.config.get('initial_capital', 100000),
                config=self.config
            )
            stats = advanced.simulate(df, all_signals)

            profit = stats.get('total_profit', 0) if isinstance(stats, dict) else 0

            self.results.append({
                'symbol': symbol,
                'profit': profit,
                'stats': stats,
            })

            logger.info(f"{symbol}: Profit={profit:.2f}, Trades={stats.get('total_trades', 0)}")

    def _walk_forward_detect(self, df: pd.DataFrame, window_size: int = 504,
                             step_size: int = 63) -> List[Dict]:
        """
        Walk-forward wave detection with rolling windows.

        For each window, detects Elliott Wave patterns and generates signals.
        Only keeps signals from the out-of-sample portion (last step_size bars)
        to avoid lookahead bias.
        """
        all_signals = []
        n_bars = len(df)

        if n_bars < 200:
            return []

        effective_window = min(window_size, n_bars)

        n_windows = 0
        n_no_pattern = 0
        n_no_multi = 0
        n_no_signals = 0
        n_oos_filtered = 0

        for start in range(0, max(1, n_bars - effective_window + 1), step_size):
            end = min(start + effective_window, n_bars)
            window_df = df.iloc[start:end].copy()

            if len(window_df) < 200:
                continue

            n_windows += 1

            wave_data = find_elliott_wave_pattern_enhanced(window_df, column='close')
            if not wave_data or wave_data.get('wave_type') == 'no_pattern':
                n_no_pattern += 1
                continue
            if len(wave_data.get('multiple_patterns', [])) == 0:
                n_no_multi += 1
                continue

            pattern_analysis = adapt_wave_data_to_strategy_input(
                window_df, wave_data, column='close'
            )

            strategy = MultiTimeframeAlignmentStrategy(self.config)
            signals = strategy.generate_signals(window_df, pattern_analysis)

            pre_oos_count = len(signals)
            if start > 0 and step_size < len(window_df):
                oos_start_date = window_df.index[-step_size]
                signals = [s for s in signals if s['date'] >= oos_start_date]

            if pre_oos_count > 0 and len(signals) == 0:
                n_oos_filtered += 1
            elif pre_oos_count == 0:
                n_no_signals += 1

            all_signals.extend(signals)

        signal_by_key = {}
        for s in all_signals:
            key = (s['date'], s['type'])
            if key not in signal_by_key or s.get('confidence', 0) > signal_by_key[key].get('confidence', 0):
                signal_by_key[key] = s

        deduped = sorted(signal_by_key.values(), key=lambda x: x['date'])
        logger.info(f"Walk-forward: {len(deduped)} signals from {n_bars} bars "
                    f"(window={effective_window}, step={step_size})")
        if n_windows > 0:
            logger.info(f"  Windows: {n_windows} total, {n_no_pattern} no-pattern, "
                        f"{n_no_multi} no-multi, {n_no_signals} no-signals, "
                        f"{n_oos_filtered} OOS-filtered")
        return deduped

    def summarize(self) -> pd.DataFrame:
        """
        Summarize backtest results and calculate metrics.
        Automatically persists results to data/backtest_results/.
        """
        if not self.results:
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.processed_dir / 'backtest_results.csv', index=False)

        profitable = results_df['profit'] > 0
        win_rate = profitable.mean()
        avg_profit = results_df['profit'].mean()
        best_trade = results_df.loc[results_df['profit'].idxmax()] if not results_df.empty else None
        worst_trade = results_df.loc[results_df['profit'].idxmin()] if not results_df.empty else None

        total_trades = 0
        total_wins = 0
        total_sharpe = 0
        total_max_dd = 0
        n_with_stats = 0
        for result in self.results:
            stats = result.get('stats', {})
            if isinstance(stats, dict) and stats.get('total_trades', 0) > 0:
                total_trades += stats['total_trades']
                total_wins += stats.get('winning_trades', 0)
                total_sharpe += stats.get('sharpe_ratio', 0)
                total_max_dd = max(total_max_dd, stats.get('max_drawdown_pct', 0))
                n_with_stats += 1

        trade_win_rate = total_wins / total_trades if total_trades > 0 else 0

        # Collect benchmark data
        total_bh_return = 0
        n_with_bh = 0
        all_wave_trades = {}
        for result in self.results:
            stats = result.get('stats', {})
            if isinstance(stats, dict):
                bh = stats.get('buy_hold_return_pct')
                if bh is not None:
                    total_bh_return += bh
                    n_with_bh += 1
                for wave_num, wd in stats.get('wave_breakdown', {}).items():
                    if wave_num not in all_wave_trades:
                        all_wave_trades[wave_num] = {'trades': 0, 'wins': 0, 'profit': 0}
                    all_wave_trades[wave_num]['trades'] += wd['trades']
                    all_wave_trades[wave_num]['wins'] += int(wd['win_rate'] * wd['trades'])
                    all_wave_trades[wave_num]['profit'] += wd['total_profit']

        avg_active = avg_profit / self.config.get('initial_capital', 100000) * 100 if avg_profit else 0
        avg_bh = total_bh_return / n_with_bh if n_with_bh else 0

        print(f"\n{'=' * 60}")
        print(f"  BACKTEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Symbols tested: {len(results_df)}")
        print(f"  Profitable:     {profitable.sum()}/{len(results_df)} ({win_rate:.0%})")
        print(f"  Average profit: ${avg_profit:,.2f}")
        print(f"{'=' * 60}")
        print(f"  ACTIVE vs BUY-AND-HOLD")
        print(f"{'=' * 60}")
        print(f"  Active return:  {avg_active:>8.1f}%  (avg per symbol)")
        print(f"  Buy & hold:     {avg_bh:>8.1f}%  (avg per symbol)")
        alpha_str = f"{avg_active - avg_bh:+.1f}%"
        verdict = "OUTPERFORMS" if avg_active > avg_bh else "UNDERPERFORMS"
        print(f"  Alpha:          {alpha_str:>8s}  ({verdict})")
        print(f"{'=' * 60}")
        print(f"  TRADE-LEVEL METRICS")
        print(f"{'=' * 60}")
        print(f"  Total trades:   {total_trades}")
        print(f"  Trade win rate: {trade_win_rate:.1%}")
        print(f"  Avg Sharpe:     {total_sharpe / n_with_stats:.2f}" if n_with_stats else "  Avg Sharpe:     N/A")
        print(f"  Worst drawdown: {total_max_dd:.1f}%")
        if best_trade is not None:
            print(f"  Best symbol:    {best_trade['symbol']} (${best_trade['profit']:,.2f})")
        if worst_trade is not None:
            print(f"  Worst symbol:   {worst_trade['symbol']} (${worst_trade['profit']:,.2f})")
        if all_wave_trades:
            print(f"{'=' * 60}")
            print(f"  PERFORMANCE BY WAVE TYPE")
            print(f"{'=' * 60}")
            for wn in sorted(all_wave_trades.keys()):
                wd = all_wave_trades[wn]
                wr = wd['wins'] / wd['trades'] * 100 if wd['trades'] > 0 else 0
                label = {1: 'Wave 1', 2: 'Wave 2', 3: 'Wave 3', 4: 'Wave 4',
                         5: 'Wave 5', 6: 'Correction'}.get(wn, f'Wave {wn}')
                print(f"  {label:<12s} {wd['trades']:>4d} trades  "
                      f"win {wr:>5.1f}%  P/L ${wd['profit']:>+10,.0f}")
        print(f"{'=' * 60}")

        self.save_results()

        return results_df

    def load_from_file(self, symbol: str) -> Optional[pd.DataFrame]:
        file_path = self.data_dir / f"{symbol}.txt"
        try:
            df = pd.read_csv(file_path, sep="\t", index_col='Date', parse_dates=True)
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            logger.info(f"Error loading {symbol} from {file_path}: {e}")
            return None

    def analyze_stock(self, symbol: str, min_price_change: float = 0.02) -> Dict[str, Any]:
        """Analyze a stock using enhanced Elliott Wave detection."""
        try:
            df = self.load_from_file(symbol)
            if df is None or len(df) < 50:
                return {"error": "insufficient_data"}

            # Use enhanced detection
            wave_data = find_elliott_wave_pattern_enhanced(
                df,
                column='close',
                min_points=6,
                max_points=12
            )

            return {
                "symbol": symbol,
                "wave_data": wave_data,
                "min_price_change": min_price_change
            }
        except Exception as e:
            return {"error": str(e)}

    def run_monte_carlo(self, n_simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo simulation on collected backtest trades.
        Call after run() has populated self.results.
        """
        all_trades = []
        for result in self.results:
            stats = result.get('stats', {})
            if isinstance(stats, dict):
                trades = stats.get('trades', [])
                all_trades.extend(trades)

        if not all_trades:
            logger.info("Monte Carlo: No trades available for simulation")
            return {'error': 'no_trades', 'n_trades': 0}

        initial_capital = self.config.get('initial_capital', 100000)
        mc = MonteCarloSimulator(n_simulations=n_simulations)
        mc_results = mc.run(all_trades, initial_capital=initial_capital)

        mc.print_report(mc_results)
        return mc_results

    def run_rs_rotation_benchmark(self, symbols: List[str], top_n: int = 10,
                                  rebalance_bars: int = 63, lookback_bars: int = 126) -> Dict:
        """
        Compute a quarterly-rebalance momentum rotation benchmark.

        Every rebalance_bars (default 63 = ~1 quarter), selects the top_n stocks
        by lookback_bars momentum (default 126 = ~6 months), equal-weights them,
        and tracks portfolio performance.
        """
        # Load all data
        all_data = {}
        for symbol in symbols:
            df = self.load_from_file(symbol)
            if df is not None and 'close' in df.columns and len(df) > lookback_bars:
                all_data[symbol] = df

        if len(all_data) < top_n:
            logger.info(f"RS Benchmark: only {len(all_data)} symbols with data, need {top_n}")
            return {'error': 'insufficient_symbols'}

        # Find common date range
        all_dates = None
        for df in all_data.values():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates &= set(df.index)

        if not all_dates or len(all_dates) < lookback_bars + rebalance_bars:
            return {'error': 'insufficient_overlap'}

        dates = sorted(all_dates)
        initial_capital = self.config.get('initial_capital', 100000)
        capital = initial_capital
        holdings = {}  # symbol -> shares
        peak = capital
        max_dd = 0
        equity_history = []

        for i in range(lookback_bars, len(dates), rebalance_bars):
            rebal_date = dates[i]

            # Compute momentum for each symbol
            momentum = {}
            for symbol, df in all_data.items():
                if rebal_date in df.index:
                    current_idx = df.index.get_loc(rebal_date)
                    if current_idx >= lookback_bars:
                        current_price = df['close'].iloc[current_idx]
                        past_price = df['close'].iloc[current_idx - lookback_bars]
                        if past_price > 0:
                            momentum[symbol] = (current_price / past_price - 1) * 100

            if len(momentum) < top_n:
                continue

            # Select top N by momentum
            ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [s for s, _ in ranked[:top_n]]

            # Sell current holdings
            for sym, shares in holdings.items():
                if rebal_date in all_data[sym].index:
                    price = all_data[sym].loc[rebal_date, 'close']
                    capital += shares * price
            holdings = {}

            # Buy equal-weight top N
            per_stock = capital / top_n
            for sym in top_symbols:
                if rebal_date in all_data[sym].index:
                    price = all_data[sym].loc[rebal_date, 'close']
                    if price > 0:
                        shares = int(per_stock / price)
                        if shares > 0:
                            holdings[sym] = shares
                            capital -= shares * price

            # Track equity
            total_equity = capital
            for sym, shares in holdings.items():
                if rebal_date in all_data[sym].index:
                    total_equity += shares * all_data[sym].loc[rebal_date, 'close']
            equity_history.append((rebal_date, total_equity))

            if total_equity > peak:
                peak = total_equity
            dd = (peak - total_equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Final equity
        if dates and holdings:
            last_date = dates[-1]
            final_equity = capital
            for sym, shares in holdings.items():
                if last_date in all_data[sym].index:
                    final_equity += shares * all_data[sym].loc[last_date, 'close']
        else:
            final_equity = capital + sum(
                shares * all_data[sym]['close'].iloc[-1]
                for sym, shares in holdings.items()
                if sym in all_data
            ) if holdings else capital

        total_return = (final_equity / initial_capital - 1) * 100
        years = (dates[-1] - dates[lookback_bars]).days / 365.25 if len(dates) > lookback_bars else 1
        annual_return = ((final_equity / initial_capital) ** (1 / max(years, 0.1)) - 1) * 100

        result = {
            'strategy': f'Top-{top_n} RS Rotation (quarterly)',
            'initial_capital': initial_capital,
            'final_capital': round(final_equity, 2),
            'total_return_pct': round(total_return, 1),
            'annual_return_pct': round(annual_return, 1),
            'max_drawdown_pct': round(max_dd * 100, 1),
            'years': round(years, 1),
            'rebalances': len(equity_history),
        }

        print(f"\n{'=' * 60}")
        print(f"  RS ROTATION BENCHMARK")
        print(f"{'=' * 60}")
        print(f"  Strategy:     Top-{top_n} by 6M momentum, quarterly rebalance")
        print(f"  Universe:     {len(all_data)} stocks, {result['years']} years")
        print(f"  Rebalances:   {result['rebalances']}")
        print(f"  Total return: {result['total_return_pct']:>+8.1f}%")
        print(f"  Annual CAGR:  {result['annual_return_pct']:>+8.1f}%")
        print(f"  Max drawdown: {result['max_drawdown_pct']:>8.1f}%")
        print(f"  Final equity: ${result['final_capital']:>12,.0f}")
        print(f"{'=' * 60}")

        return result

    def save_results(self) -> Optional[str]:
        """Persist backtest results to a timestamped JSON file in data/backtest_results/."""
        if not self.results:
            logger.info("No results to save")
            return None

        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / 'data' / 'backtest_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime('%Y-%m-%d')

        records = []
        for result in self.results:
            stats = result.get('stats', {})
            if not isinstance(stats, dict):
                stats = {}

            trades = stats.get('trades', [])
            date_start = None
            date_end = None
            if trades:
                entry_dates = [str(t.get('entry_date', '')) for t in trades]
                exit_dates = [str(t.get('exit_date', '')) for t in trades]
                all_dates = [d for d in entry_dates + exit_dates if d]
                if all_dates:
                    date_start = min(all_dates)
                    date_end = max(all_dates)

            records.append({
                'symbol': result.get('symbol', 'unknown'),
                'date_range_start': date_start,
                'date_range_end': date_end,
                'strategy': self.config.get('default_strategy', 'multiframe'),
                'total_return_pct': stats.get('total_return_pct', 0),
                'total_profit': stats.get('total_profit', 0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                'max_drawdown_pct': stats.get('max_drawdown_pct', 0),
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'initial_capital': stats.get('initial_capital', 0),
                'final_capital': stats.get('final_capital', 0),
                'buy_hold_return_pct': stats.get('buy_hold_return_pct', 0),
                'buy_hold_annual_pct': stats.get('buy_hold_annual_pct', 0),
                'alpha': stats.get('alpha', 0),
                'backtest_years': stats.get('backtest_years', 0),
                'wave_breakdown': stats.get('wave_breakdown', {}),
            })

        symbols = list({r['symbol'] for r in records})
        if len(symbols) == 1:
            label = symbols[0]
        else:
            label = f"{len(symbols)}_symbols"

        filename = f"backtest_{label}_{today}.json"
        filepath = output_dir / filename

        counter = 1
        while filepath.exists():
            counter += 1
            filename = f"backtest_{label}_{today}_{counter}.json"
            filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filepath}")
        return str(filepath)
