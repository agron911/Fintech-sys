#!/usr/bin/env python3
"""Fetch stock data from Yahoo Finance.

Usage:
    python scripts/run_fetch_data.py --all           # Fetch all configured stocks
    python scripts/run_fetch_data.py --tw            # Fetch Taiwan listed + OTC stocks
    python scripts/run_fetch_data.py --us            # Fetch US / international stocks
    python scripts/run_fetch_data.py --add BE TSLA   # Fetch specific US stock symbols
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.crawler.yahoo_finance import (
    delete_files, crawl_all_ch, crawl_otc_yf,
    fetch_stock_data, save_stock_data, _ensure_init,
)
import src.crawler.yahoo_finance as _yf_module

def fetch_individual(symbols):
    """Fetch data for specific US stock symbols."""
    _ensure_init()
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        df = fetch_stock_data(symbol, "", _yf_module.start, _yf_module.end)
        if df is not None and not df.empty:
            save_stock_data(df, symbol)
            print(f"  Saved {symbol}")
        else:
            print(f"  FAIL: no data for {symbol}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data from Yahoo Finance")
    parser.add_argument('--add', nargs='+', help='Fetch specific US stock symbols')
    parser.add_argument('--tw', action='store_true', help='Fetch Taiwan listed + OTC stocks')
    parser.add_argument('--us', action='store_true', help='Fetch US / international stocks only')
    parser.add_argument('--all', action='store_true', help='Fetch all configured stocks (TW + US)')
    args = parser.parse_args()

    # If no flags provided, show help and exit
    if not (args.add or args.tw or args.us or args.all):
        parser.print_help()
        print("\nError: please specify at least one flag (--all, --tw, --us, or --add SYMBOL ...)")
        sys.exit(1)

    if args.add:
        print(f"Fetching data for {len(args.add)} stocks...")
        fetch_individual(args.add)
    elif args.all:
        print("Incremental update mode...")
        delete_files()
        print("Crawling all stocks (listed + OTC + international)...")
        crawl_all_ch()
        crawl_otc_yf()
    elif args.tw:
        print("Incremental update mode...")
        delete_files()
        print("Crawling Taiwan listed + OTC stocks...")
        crawl_all_ch()
        crawl_otc_yf()
    elif args.us:
        print("Incremental update mode...")
        delete_files()
        print("Crawling US / international stocks...")
        crawl_all_ch()  # crawl_all_ch handles international stocks too

    print("Done.")
