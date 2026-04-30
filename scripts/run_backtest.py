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
    args = parser.parse_args()

    setup_logging()
    config = load_config()

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


if __name__ == "__main__":
    main()
