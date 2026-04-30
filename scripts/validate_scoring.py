#!/usr/bin/env python3
"""
Scoring Validation — Empirically test whether the signal scoring pipeline
produces actionable predictions.

Usage:
    python scripts/validate_scoring.py --forward-returns             # Score bucket analysis
    python scripts/validate_scoring.py --backtest                     # Full trade simulation
    python scripts/validate_scoring.py --monte-carlo                  # MC on backtest trades
    python scripts/validate_scoring.py --walk-forward                 # OOS validation
    python scripts/validate_scoring.py --all                          # Everything
    python scripts/validate_scoring.py --symbols AAPL MSFT --all      # Specific stocks
    python scripts/validate_scoring.py --max-stocks 50 --all          # Random sample
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import warnings
warnings.filterwarnings('ignore')
import logging

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import load_config
from src.backtest.scoring_backtest import (
    ScoringBacktester,
    ScoringWalkForwardValidator,
)
from src.backtest.monte_carlo import MonteCarloSimulator


logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_stock_data(data_dir: Path, symbol: str) -> pd.DataFrame | None:
    """Load stock data from tab-separated file."""
    file_path = data_dir / f"{symbol}.txt"
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path, sep="\t", index_col='Date', parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        # Drop duplicate Date column if present
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        if 'close' not in df.columns or len(df) < 100:
            return None
        return df
    except Exception:
        return None


def get_symbols(data_dir: Path, args) -> list[str]:
    """Determine which symbols to validate."""
    if args.symbols:
        return args.symbols

    # Load all available symbols from data directory
    symbols = []
    for f in sorted(data_dir.glob('*.txt')):
        sym = f.stem
        if sym.startswith('.') or '_' in sym:
            continue
        symbols.append(sym)

    if args.max_stocks and args.max_stocks < len(symbols):
        random.seed(42)
        symbols = random.sample(symbols, args.max_stocks)

    return symbols


def run_forward_returns(data_dir: Path, symbols: list[str], eval_interval: int):
    """Mode: Forward returns analysis by score bucket."""
    print(f"\n{'='*60}")
    print(f"  FORWARD RETURNS ANALYSIS")
    print(f"  {len(symbols)} stocks, eval every {eval_interval} bars")
    print(f"{'='*60}\n")

    sb = ScoringBacktester(eval_interval=eval_interval)
    all_fwd = []

    for i, sym in enumerate(symbols):
        df = load_stock_data(data_dir, sym)
        if df is None:
            continue
        scores = sb.generate_historical_scores(df, sym)
        if not scores:
            continue
        fwd = sb.compute_forward_returns(df, scores)
        if not fwd.empty:
            all_fwd.append(fwd)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(symbols)} stocks...")

    if not all_fwd:
        print("  No forward returns data generated.")
        return None

    combined = pd.concat(all_fwd, ignore_index=True)
    analysis = sb.analyze_score_buckets(combined)

    # Print results
    print(f"\n  Total score observations: {len(combined)}")
    print(f"\n  {'Action':<16} {'Count':>6}  {'5d_mean':>8}  {'20d_mean':>9}  {'60d_mean':>9}  {'5d_win%':>8}  {'20d_win%':>9}")
    print(f"  {'-'*16} {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}")

    for action in ['STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION', 'WATCH', 'HOLD', 'WAIT', 'AVOID', 'EXIT', 'SKIP']:
        stats = analysis['by_action'].get(action)
        if not stats:
            continue
        fwd5 = stats.get('fwd_5d_mean')
        fwd20 = stats.get('fwd_20d_mean')
        fwd60 = stats.get('fwd_60d_mean')
        win5 = stats.get('fwd_5d_win_pct')
        win20 = stats.get('fwd_20d_win_pct')
        print(f"  {action:<16} {stats['count']:>6}"
              f"  {fwd5:>7.2f}%" if fwd5 is not None else "",
              end="")
        if fwd5 is not None:
            print(f"  {fwd20:>8.2f}%" if fwd20 is not None else "       N/A", end="")
            print(f"  {fwd60:>8.2f}%" if fwd60 is not None else "       N/A", end="")
            print(f"  {win5:>7.1f}%" if win5 is not None else "      N/A", end="")
            print(f"  {win20:>8.1f}%" if win20 is not None else "       N/A", end="")
        print()

    # Correlation
    print(f"\n  Score-Return Correlation (Spearman):")
    for col, corr in analysis['correlation'].items():
        rho = corr.get('spearman_rho')
        p = corr.get('p_value')
        if rho is not None:
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"    {col}: rho={rho:.4f}, p={p:.4f}{sig}")

    # T-tests
    print(f"\n  Statistical Tests:")
    for col, col_tests in analysis['tests'].items():
        for test_name, t in col_tests.items():
            sig_str = "SIGNIFICANT" if t.get('significant') else "not significant"
            print(f"    {col} {test_name}: t={t['t_stat']:.3f}, p={t['p_value']:.4f} ({sig_str})")

    print()
    return combined, analysis


def run_backtest(data_dir: Path, symbols: list[str], eval_interval: int,
                 initial_capital: float):
    """Mode: Full trade simulation."""
    print(f"\n{'='*60}")
    print(f"  SCORING BACKTEST")
    print(f"  {len(symbols)} stocks, ${initial_capital:,.0f} capital")
    print(f"{'='*60}\n")

    all_trades = []
    total_stats = {
        'stocks_tested': 0,
        'stocks_with_trades': 0,
        'total_signals': 0,
    }

    sb = ScoringBacktester(
        initial_capital=initial_capital,
        eval_interval=eval_interval,
    )

    for i, sym in enumerate(symbols):
        df = load_stock_data(data_dir, sym)
        if df is None:
            continue

        stats = sb.run_backtest(df, sym)
        total_stats['stocks_tested'] += 1

        if stats.get('total_trades', 0) > 0:
            total_stats['stocks_with_trades'] += 1
            all_trades.extend(stats.get('trades', []))

        total_stats['total_signals'] += stats.get('buy_signals', 0)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(symbols)} stocks...")

    if not all_trades:
        print("  No trades generated.")
        return None

    # Aggregate stats
    wins = sum(1 for t in all_trades if t['profit'] > 0)
    losses = len(all_trades) - wins
    total_profit = sum(t['profit'] for t in all_trades)
    win_profits = [t['profit'] for t in all_trades if t['profit'] > 0]
    loss_profits = [t['profit'] for t in all_trades if t['profit'] <= 0]
    avg_win = float(np.mean(win_profits)) if win_profits else 0
    avg_loss = float(np.mean(loss_profits)) if loss_profits else 0
    pf = abs(sum(win_profits) / sum(loss_profits)) if loss_profits and sum(loss_profits) != 0 else float('inf')

    # Sharpe (trade-level)
    profits_arr = np.array([t['profit'] for t in all_trades])
    sharpe = float(profits_arr.mean() / profits_arr.std() * np.sqrt(len(profits_arr))) if profits_arr.std() > 0 else 0

    # Max drawdown
    equity = initial_capital + np.cumsum(profits_arr)
    equity = np.insert(equity, 0, initial_capital)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / running_max
    max_dd = float(dd.max())

    print(f"  Stocks tested:   {total_stats['stocks_tested']}")
    print(f"  Stocks w/trades: {total_stats['stocks_with_trades']}")
    print(f"  Total signals:   {total_stats['total_signals']}")
    print(f"  Total trades:    {len(all_trades)}")
    print(f"  Win rate:        {wins/len(all_trades)*100:.1f}% ({wins}W / {losses}L)")
    print(f"  Total profit:    ${total_profit:,.0f}")
    print(f"  Total return:    {total_profit/initial_capital*100:.1f}%")
    print(f"  Avg win:         ${avg_win:,.0f}")
    print(f"  Avg loss:        ${avg_loss:,.0f}")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd*100:.1f}%")
    print()

    return all_trades


def run_monte_carlo(trades: list[dict], initial_capital: float):
    """Mode: Monte Carlo simulation on backtest trades."""
    if not trades or len(trades) < 3:
        print("  Not enough trades for Monte Carlo (need >= 3).")
        return None

    mc = MonteCarloSimulator(n_simulations=10000)
    results = mc.run(trades, initial_capital=initial_capital)
    mc.print_report(results)
    return results


def run_walk_forward(data_dir: Path, symbols: list[str], eval_interval: int,
                     initial_capital: float):
    """Mode: Walk-forward OOS validation."""
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD VALIDATION (Scoring Pipeline)")
    print(f"  {len(symbols)} stocks, train=252, test=63")
    print(f"{'='*60}\n")

    all_trades = []
    total_windows = 0
    profitable_windows = 0
    stocks_validated = 0

    for i, sym in enumerate(symbols):
        df = load_stock_data(data_dir, sym)
        if df is None or len(df) < 315:  # 252 + 63
            continue

        wf = ScoringWalkForwardValidator(
            train_bars=252,
            test_bars=63,
            step_bars=63,
            eval_interval=eval_interval,
            initial_capital=initial_capital,
        )
        result = wf.validate(df, sym)

        if result.get('windows', 0) > 0:
            stocks_validated += 1
            total_windows += result['windows']
            profitable_windows += result.get('profitable_windows', 0)
            all_trades.extend(result.get('trades', []))

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(symbols)} stocks...")

    if not stocks_validated:
        print("  No stocks had enough data for walk-forward validation.")
        return None

    wins = sum(1 for t in all_trades if t['profit'] > 0)
    losses = len(all_trades) - wins
    total_profit = sum(t['profit'] for t in all_trades)

    win_profits = [t['profit'] for t in all_trades if t['profit'] > 0]
    loss_profits = [t['profit'] for t in all_trades if t['profit'] <= 0]
    pf = abs(sum(win_profits) / sum(loss_profits)) if loss_profits and sum(loss_profits) != 0 else float('inf')

    print(f"  Stocks validated:     {stocks_validated}")
    print(f"  Total windows:        {total_windows}")
    print(f"  Profitable windows:   {profitable_windows}/{total_windows} ({profitable_windows/total_windows*100:.0f}%)" if total_windows > 0 else "")
    print(f"  OOS trades:           {len(all_trades)}")
    print(f"  OOS win rate:         {wins/len(all_trades)*100:.1f}% ({wins}W / {losses}L)" if all_trades else "  OOS trades:           0")
    print(f"  OOS profit:           ${total_profit:,.0f}" if all_trades else "")
    print(f"  OOS profit factor:    {pf:.2f}" if all_trades else "")
    print()

    return all_trades


def main():
    parser = argparse.ArgumentParser(
        description='Validate the signal scoring pipeline empirically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--forward-returns', action='store_true',
                        help='Run forward returns analysis by score bucket')
    parser.add_argument('--backtest', action='store_true',
                        help='Run full trade simulation')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo simulation on backtest trades')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward OOS validation')
    parser.add_argument('--all', action='store_true',
                        help='Run all validation modes')
    parser.add_argument('--symbols', nargs='+',
                        help='Specific stock symbols to validate')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Maximum number of stocks to sample')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Score evaluation interval in bars (default: 5)')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Initial capital (default: 100000)')

    args = parser.parse_args()

    if not any([args.forward_returns, args.backtest, args.monte_carlo,
                args.walk_forward, args.all]):
        parser.print_help()
        sys.exit(1)

    config = load_config()
    data_dir = Path(config['stk2_dir'])

    symbols = get_symbols(data_dir, args)
    if not symbols:
        print("No symbols found. Check data directory.")
        sys.exit(1)

    print(f"  Data directory: {data_dir}")
    print(f"  Symbols: {len(symbols)}")

    trades = None

    if args.forward_returns or args.all:
        run_forward_returns(data_dir, symbols, args.eval_interval)

    if args.backtest or args.monte_carlo or args.all:
        trades = run_backtest(data_dir, symbols, args.eval_interval, args.capital)

    if args.monte_carlo or args.all:
        if trades:
            run_monte_carlo(trades, args.capital)
        else:
            print("  No trades available for Monte Carlo. Run --backtest first.")

    if args.walk_forward or args.all:
        run_walk_forward(data_dir, symbols, args.eval_interval, args.capital)


if __name__ == '__main__':
    main()
