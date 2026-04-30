import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from src.utils.config import load_config
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_elliott_wave_pattern_enhanced
from src.backtest.strategy_advanced import AdvancedBacktester
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

    def run(self, symbols: List[str], min_price_changes: List[float] = None):
        """
        Run backtest for a list of symbols and parameter combinations.
        """
        if min_price_changes is None:
            min_price_changes = [0.01, 0.03, 0.05, 0.1]

        for symbol in symbols:
            df = self.load_from_file(symbol)
            if df is None or 'close' not in df.columns:
                logger.info(f"Skipping {symbol}: Data not found or missing 'close' column")
                continue

            for min_price_change in min_price_changes:
                # Use enhanced pattern detection
                wave_data = find_elliott_wave_pattern_enhanced(df, column='close')

                # Skip if no meaningful pattern found
                if not wave_data or wave_data.get('wave_type') == 'no_pattern' or len(wave_data.get('multiple_patterns', [])) == 0:
                    logger.info(f"Skipping {symbol} (min_price_change={min_price_change}): No valid Elliott Wave pattern found")
                    continue

                # Transform wave_data into the format strategies expect
                pattern_analysis = adapt_wave_data_to_strategy_input(df, wave_data, column='close')

                # Use the advanced backtester to evaluate the detected pattern
                advanced = AdvancedBacktester(initial_capital=self.config.get('initial_capital', 100000), config=self.config)
                stats = advanced.run_backtest(df, pattern_analysis, strategy_name=self.config.get('default_strategy', 'multiframe'))

                profit = stats.get('total_profit', 0) if isinstance(stats, dict) else 0

                self.results.append({
                    'symbol': symbol,
                    'profit': profit,
                    'stats': stats,
                    'min_price_change': min_price_change
                })

                logger.info(f"{symbol}: Profit={profit:.2f}, min_price_change={min_price_change}")

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

        print(f"\n{'=' * 50}")
        print(f"  BACKTEST SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Win rate:       {win_rate:.2%}")
        print(f"  Average profit: {avg_profit:.2f}")
        if best_trade is not None:
            print(f"  Best trade:     {best_trade['symbol']} (Profit: {best_trade['profit']:.2f}, min_price_change: {best_trade['min_price_change']})")
        if worst_trade is not None:
            print(f"  Worst trade:    {worst_trade['symbol']} (Profit: {worst_trade['profit']:.2f}, min_price_change: {worst_trade['min_price_change']})")
        print(f"{'=' * 50}")

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
                'min_price_change': result.get('min_price_change'),
                'total_return_pct': stats.get('total_return_pct', 0),
                'total_profit': stats.get('total_profit', 0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                'max_drawdown_pct': stats.get('max_drawdown_pct', 0),
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'initial_capital': stats.get('initial_capital', 0),
                'final_capital': stats.get('final_capital', 0),
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
