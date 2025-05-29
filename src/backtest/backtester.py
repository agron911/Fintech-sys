import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from investment_system.src.utils.config import load_config
# from investment_system.src.utils.data_utils import load_from_database  # Placeholder, implement as needed
from investment_system.src.analysis.elliott_wave import detect_peaks_troughs, refined_elliott_wave_suggestion
from investment_system.src.backtest.strategy import backtest_elliott_strategy

class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.results = []
        self.data_dir = Path(config['stk2_dir'])
        self.processed_dir = Path(config['processed_dir'])

    def run(self, symbols: List[str], min_price_changes: List[float] = [0.01, 0.03, 0.05, 0.1]):
        """
        Run backtest for a list of symbols and parameter combinations.
        """
        for symbol in symbols:
            df = self.load_from_file(symbol)
            if df is None or 'close' not in df.columns:
                print(f"Skipping {symbol}: Data not found or missing 'close' column")
                continue
            
            for min_price_change in min_price_changes:
                peaks, troughs = detect_peaks_troughs(df, column='close')
                wave_points = refined_elliott_wave_suggestion(df, peaks, troughs, min_price_change=min_price_change)
                if len(wave_points) < 6:
                    print(f"Skipping {symbol} (min_price_change={min_price_change}): Not enough wave points")
                    continue
                profit, trade = backtest_elliott_strategy(df, wave_points, column='close')
                if profit is not None:
                    self.results.append({
                        'symbol': symbol,
                        'profit': profit,
                        'trade': trade,
                        'min_price_change': min_price_change
                    })
                    print(f"{symbol}: Profit={profit:.2f}, min_price_change={min_price_change}")

    def summarize(self) -> pd.DataFrame:
        """
        Summarize backtest results and calculate metrics.
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
        
        print(f"\nWin rate: {win_rate:.2%}")
        print(f"Average profit: {avg_profit:.2f}")
        if best_trade is not None:
            print(f"Best trade: {best_trade['symbol']} (Profit: {best_trade['profit']:.2f}, min_price_change: {best_trade['min_price_change']})")
        if worst_trade is not None:
            print(f"Worst trade: {worst_trade['symbol']} (Profit: {worst_trade['profit']:.2f}, min_price_change: {worst_trade['min_price_change']})")
        
        return results_df

    def load_from_file(self, symbol: str) -> Optional[pd.DataFrame]:
        file_path = self.data_dir / f"{symbol}.txt"
        try:
            df = pd.read_csv(file_path, sep="\t", index_col='Date', parse_dates=True)
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            print(f"Error loading {symbol} from {file_path}: {e}")
            return None
