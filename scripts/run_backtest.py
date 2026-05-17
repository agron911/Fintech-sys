import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
from src.utils.config import load_config
from src.utils.log_setup import setup_logging
from src.backtest.backtester import Backtester


def main():
    parser = argparse.ArgumentParser(description="Run backtest on stock symbols")
    parser.add_argument(
        '--monte-carlo', action='store_true',
        help='Run Monte Carlo simulation on trade results after backtest'
    )
    parser.add_argument(
        '--mc-simulations', type=int, default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )
    parser.add_argument(
        '--symbols', type=str, default=None,
        help='Comma-separated list of symbols to backtest (e.g. AAPL,MSFT,TSLA)'
    )
    parser.add_argument(
        '--window-size', type=int, default=504,
        help='Walk-forward window size in bars (default: 504 = ~2 years)'
    )
    parser.add_argument(
        '--step-size', type=int, default=63,
        help='Walk-forward step size in bars (default: 63 = ~1 quarter)'
    )
    parser.add_argument(
        '--rs-benchmark', action='store_true',
        help='Run quarterly RS rotation benchmark for comparison'
    )
    parser.add_argument(
        '--random-benchmark', action='store_true',
        help='Run random entry benchmark to test if wave detection adds alpha'
    )
    parser.add_argument(
        '--random-iterations', type=int, default=100,
        help='Number of random benchmark iterations (default: 100)'
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    config['backtest_window_size'] = args.window_size
    config['backtest_step_size'] = args.step_size

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        # Load stock lists
        international = pd.read_csv(config["international_file"])["code"]
        listed = pd.read_excel(config["list_file"])["code"]
        otc = pd.read_excel(config["otclist_file"])["code"]
        symbols = list(international) + list(listed) + list(otc) + ['TWII']

    backtester = Backtester(config)
    backtester.run(symbols)
    backtester.summarize()

    if args.monte_carlo:
        backtester.run_monte_carlo(n_simulations=args.mc_simulations)

    if args.rs_benchmark:
        backtester.run_rs_rotation_benchmark(symbols)

    if args.random_benchmark:
        backtester.run_random_entry_benchmark(
            symbols, n_iterations=args.random_iterations
        )


if __name__ == "__main__":
    main()
