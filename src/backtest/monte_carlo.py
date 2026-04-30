"""
Monte Carlo Simulation for Backtest Robustness

After a backtest produces a trade list, randomizes trade order to calculate
the distribution of max drawdown, final equity, and Sharpe ratio.
Reports confidence intervals to assess strategy robustness.

Usage:
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_simulations=10000)
    results = mc.run(trades, initial_capital=100000)
"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation that reshuffles trade sequence to estimate
    the distribution of outcomes from the same set of trades.
    """

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        """
        Args:
            n_simulations: Number of random permutations to run
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed

    def run(self, trades: List[Dict], initial_capital: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation by reshuffling trade order.

        Args:
            trades: List of trade dicts (must have 'profit' key)
            initial_capital: Starting capital

        Returns:
            Dict with distribution statistics and confidence intervals
        """
        if not trades or len(trades) < 3:
            return {
                'error': 'insufficient_trades',
                'n_trades': len(trades) if trades else 0,
            }

        profits = np.array([t['profit'] for t in trades])
        n_trades = len(profits)

        rng = np.random.default_rng(self.seed)

        final_equities = np.empty(self.n_simulations)
        max_drawdowns = np.empty(self.n_simulations)
        sharpe_ratios = np.empty(self.n_simulations)

        for i in range(self.n_simulations):
            # Shuffle trade order
            shuffled = rng.permutation(profits)

            # Compute equity curve
            equity = initial_capital + np.cumsum(shuffled)
            equity = np.insert(equity, 0, initial_capital)

            # Final equity
            final_equities[i] = equity[-1]

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdowns[i] = drawdowns.max()

            # Sharpe ratio (trade-level)
            if shuffled.std() > 0:
                sharpe_ratios[i] = shuffled.mean() / shuffled.std() * np.sqrt(n_trades)
            else:
                sharpe_ratios[i] = 0

        return {
            'n_simulations': self.n_simulations,
            'n_trades': n_trades,
            'initial_capital': initial_capital,

            # Final equity distribution
            'final_equity': {
                'median': float(np.median(final_equities)),
                'mean': float(np.mean(final_equities)),
                'p5': float(np.percentile(final_equities, 5)),
                'p25': float(np.percentile(final_equities, 25)),
                'p75': float(np.percentile(final_equities, 75)),
                'p95': float(np.percentile(final_equities, 95)),
            },

            # Max drawdown distribution
            'max_drawdown': {
                'median': float(np.median(max_drawdowns)),
                'mean': float(np.mean(max_drawdowns)),
                'p5': float(np.percentile(max_drawdowns, 5)),    # Best case
                'p50': float(np.percentile(max_drawdowns, 50)),
                'p95': float(np.percentile(max_drawdowns, 95)),   # Worst case
            },

            # Sharpe distribution
            'sharpe_ratio': {
                'median': float(np.median(sharpe_ratios)),
                'mean': float(np.mean(sharpe_ratios)),
                'p5': float(np.percentile(sharpe_ratios, 5)),
                'p95': float(np.percentile(sharpe_ratios, 95)),
            },

            # Risk assessment
            'probability_of_profit': float((final_equities > initial_capital).mean()),
            'probability_of_ruin': float((final_equities < initial_capital * 0.5).mean()),
            'worst_case_dd_95': float(np.percentile(max_drawdowns, 95)),
        }

    def print_report(self, results: Dict):
        """Print formatted Monte Carlo report."""
        if 'error' in results:
            print(f"  Monte Carlo: {results['error']}")
            return

        fe = results['final_equity']
        dd = results['max_drawdown']
        sr = results['sharpe_ratio']

        print(f"\n{'='*60}")
        print(f"  MONTE CARLO SIMULATION ({results['n_simulations']:,} runs, {results['n_trades']} trades)")
        print(f"{'='*60}")
        print(f"\n  Final Equity (starting ${results['initial_capital']:,.0f}):")
        print(f"    5th percentile:  ${fe['p5']:>12,.0f}")
        print(f"    25th percentile: ${fe['p25']:>12,.0f}")
        print(f"    Median:          ${fe['median']:>12,.0f}")
        print(f"    75th percentile: ${fe['p75']:>12,.0f}")
        print(f"    95th percentile: ${fe['p95']:>12,.0f}")
        print(f"\n  Max Drawdown:")
        print(f"    Best case (5th):  {dd['p5']*100:>6.1f}%")
        print(f"    Median:           {dd['median']*100:>6.1f}%")
        print(f"    Worst case (95th):{dd['p95']*100:>6.1f}%")
        print(f"\n  Sharpe Ratio:")
        print(f"    5th percentile:  {sr['p5']:>6.2f}")
        print(f"    Median:          {sr['median']:>6.2f}")
        print(f"    95th percentile: {sr['p95']:>6.2f}")
        print(f"\n  Probability of profit: {results['probability_of_profit']*100:.1f}%")
        print(f"  Probability of ruin:   {results['probability_of_ruin']*100:.1f}%")
        print(f"{'='*60}\n")
