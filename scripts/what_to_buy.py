#!/usr/bin/env python3
"""
WHAT TO BUY NOW — Single command to find actionable buy candidates.

Usage:
    python scripts/what_to_buy.py                    # Scan current stock list
    python scripts/what_to_buy.py --expand            # Add popular US stocks to scan
    python scripts/what_to_buy.py --add AAPL GOOG META  # Add specific stocks
    python scripts/what_to_buy.py --refresh           # Fetch fresh data first

The system will:
1. Load/fetch price data
2. Run Elliott Wave pattern detection
3. Check trend alignment (SMA50/200)
4. Check momentum (RSI, MACD, ADX, velocity)
5. Show ONLY stocks where all signals align for a BUY
6. Print exact entry price, stop loss, targets, and risk
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config
from src.analysis.core.signal_scoring import (
    analyze_stock as analyze,
    classify_action,
    apply_market_regime,
    apply_relative_strength,
    apply_rs_guard,
    apply_rs_score_adjustment,
    reclassify_borderline,
    grade_conviction,
    _classify_market,
    SCORING_CONFIG,
)
from src.analysis.market_structure import analyze_market_structure, format_market_structure
from src.crawler.yahoo_finance import fetch_stock_data, save_stock_data

# Extended universe of liquid US stocks across sectors (~180 stocks)
EXPANDED_UNIVERSE = [
    # Mega-cap Tech
    'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'NFLX', 'CRM', 'ADBE', 'INTC', 'QCOM',
    'AVGO', 'MU', 'MRVL', 'ANET', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'ZS', 'NET',
    'ORCL', 'CSCO', 'IBM', 'NOW', 'ADSK', 'FTNT', 'ESTC', 'DELL', 'HPE',
    # Semiconductors
    'ASML', 'LRCX', 'KLAC', 'MCHP', 'ON', 'SWKS', 'TXN', 'ADI', 'AMAT', 'TSM', 'SMCI',
    # AI / Quantum / Emerging Tech
    'PLTR', 'APP', 'IONQ', 'SOUN', 'TTD', 'RKLB', 'U', 'ROKU', 'SNAP',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN', 'BLK', 'BX',
    'SCHW', 'AXP', 'BK', 'ICE', 'WFC', 'HOOD', 'SOFI', 'AFRM', 'MARA',
    # Healthcare / Biotech
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'BMY', 'GILD', 'AMGN', 'REGN',
    'ISRG', 'DHR', 'MDT', 'VRTX', 'HIMS', 'NVO', 'TMO',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'MPC', 'EQT',
    # Consumer / Retail
    'COST', 'WMT', 'TGT', 'NKE', 'SBUX', 'MCD', 'DIS', 'HD', 'LOW', 'LULU',
    'CMG', 'BKNG', 'ABNB', 'DLTR', 'CHWY', 'CELH', 'TOST', 'DASH', 'DKNG',
    # Industrial / Defense
    'CAT', 'DE', 'GE', 'BA', 'LMT', 'RTX', 'GD', 'HON', 'UNP', 'UPS',
    # Staples / Dividend
    'PG', 'KO', 'PEP', 'CL', 'PM', 'MO', 'MDLZ', 'AFL',
    # Telecom / Media
    'T', 'VZ', 'CMCSA', 'SPOT', 'EA',
    # International ADRs
    'BIDU', 'PDD', 'SE', 'SHOP', 'MELI', 'NTES',
    # EV / Clean energy
    'TSLA', 'RIVN', 'LCID', 'FSLR', 'ENPH', 'ACHR',
    # Other
    'PINS', 'ZM', 'F', 'GM', 'CCL', 'LVS', 'TRIP', 'WBA', 'CVS',
    'DOW', 'MMM', 'LIN', 'APD', 'NEE', 'AEP', 'ADP', 'APH', 'VRT', 'TEM',
]


def load_stock(symbol, data_dir):
    filepath = Path(data_dir) / f"{symbol}.txt"
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath, sep='\t')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')
        df.columns = [c.lower() for c in df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        return df
    except Exception:
        return None


def fetch_stock(symbol, data_dir, start='2020-01-01', end='2026-12-31', suffix=''):
    """Fetch a single stock from Yahoo Finance using the crawler module."""
    try:
        df = fetch_stock_data(symbol, suffix, start, end)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return False
        save_stock_data(df, symbol, folder=Path(data_dir))
        return True
    except Exception:
        return False






def main():
    parser = argparse.ArgumentParser(
        description='What to buy now',
        epilog="""Examples:
  python scripts/what_to_buy.py --expand              # Scan ~75 US stocks
  python scripts/what_to_buy.py --tw                  # Scan Taiwan stocks only
  python scripts/what_to_buy.py --expand --tw          # Scan US + Taiwan stocks
  python scripts/what_to_buy.py --tw --tw-top 50      # Top 50 TW stocks by volume
  python scripts/what_to_buy.py --refresh --expand     # Fetch missing US data first
  python scripts/what_to_buy.py --refresh --tw         # Fetch missing TW data first
  python scripts/what_to_buy.py --add 2330 --tw        # Add specific TW stock
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--expand', action='store_true',
                        help='Expand universe to ~70 popular US stocks')
    parser.add_argument('--tw', action='store_true',
                        help='Include Taiwan stocks (TWSE listed + OTC)')
    parser.add_argument('--tw-top', type=int, default=100,
                        help='Max number of TW stocks to scan (default: 100, use 0 for all)')
    parser.add_argument('--add', nargs='+', help='Add specific stock symbols')
    parser.add_argument('--refresh', action='store_true',
                        help='Fetch/update data from Yahoo Finance')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed analysis for WATCH/EXIT candidates')
    parser.add_argument('--top', type=int, default=10,
                        help='Show top N candidates in each category')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Save analysis results to a JSON file')
    parser.add_argument('--sector', nargs='+', metavar='SECTOR',
                        help='Filter by sector (e.g., --sector Tech Semis Energy)')
    parser.add_argument('--sort', choices=['score', 'rs', 'rr', 'momentum'],
                        default='score', help='Sort results by (default: score)')
    parser.add_argument('--format', choices=['text', 'csv', 'json'],
                        default='text', dest='output_format',
                        help='Output format (default: text)')
    args = parser.parse_args()

    config = load_config()
    data_dir = config.get('stk2_dir', 'data/raw')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Build symbol list — US stocks
    symbols = []
    try:
        intl = pd.read_csv(config['international_file'])
        symbols.extend(list(intl['code']))
    except FileNotFoundError:
        logging.warning(
            "International stock list not found: %s — run with --expand or create the file.",
            config.get('international_file', '(not configured)'),
        )
    except Exception as e:
        logging.warning(
            "Failed to load international stock list '%s': %s",
            config.get('international_file', '(not configured)'), e,
        )

    if args.expand:
        for s in EXPANDED_UNIVERSE:
            if s not in symbols:
                symbols.append(s)

    # Build symbol list — Taiwan stocks
    tw_symbols = []  # Track TW symbols separately for suffix handling
    if args.tw:
        tw_listed = []
        tw_otc = []
        try:
            listed_df = pd.read_excel(config['list_file'])
            tw_listed = [str(c) for c in listed_df.iloc[:, 0]]
        except FileNotFoundError:
            logging.warning(
                "TWSE listed stock list not found: %s — Taiwan listed stocks will be skipped. "
                "Download the file or remove --tw flag.",
                config.get('list_file', '(not configured)'),
            )
        except Exception as e:
            logging.warning(
                "Failed to load TWSE listed stock list '%s': %s",
                config.get('list_file', '(not configured)'), e,
            )
        try:
            otc_df = pd.read_excel(config['otclist_file'])
            tw_otc = [str(c) for c in otc_df.iloc[:, 0]]
        except FileNotFoundError:
            logging.warning(
                "OTC stock list not found: %s — Taiwan OTC stocks will be skipped. "
                "Download the file or remove --tw flag.",
                config.get('otclist_file', '(not configured)'),
            )
        except Exception as e:
            logging.warning(
                "Failed to load OTC stock list '%s': %s",
                config.get('otclist_file', '(not configured)'), e,
            )

        tw_all = tw_listed + tw_otc
        tw_limit = args.tw_top if args.tw_top > 0 else len(tw_all)
        tw_symbols = tw_all[:tw_limit]

        for s in tw_symbols:
            if s not in symbols:
                symbols.append(s)

        print("Taiwan stocks: {} TWSE + {} OTC = {} total, scanning {}".format(
            len(tw_listed), len(tw_otc), len(tw_all), len(tw_symbols)))

    if args.add:
        for s in args.add:
            s = s.upper()
            if s not in symbols:
                symbols.append(s)

    if not symbols:
        symbols = ['TSLA', 'NVDA', 'AMD', 'TSM']

    # Build suffix map for TW stocks
    tw_suffix_map = {}  # symbol -> '.TW' or '.TWO'
    if args.tw:
        try:
            listed_df = pd.read_excel(config['list_file'])
            for c in listed_df.iloc[:, 0]:
                tw_suffix_map[str(c)] = '.TW'
        except Exception as e:
            logging.warning(
                "Failed to build TWSE suffix map from '%s': %s",
                config.get('list_file', '(not configured)'), e,
            )
        try:
            otc_df = pd.read_excel(config['otclist_file'])
            for c in otc_df.iloc[:, 0]:
                tw_suffix_map[str(c)] = '.TWO'
        except Exception as e:
            logging.warning(
                "Failed to build OTC suffix map from '%s': %s",
                config.get('otclist_file', '(not configured)'), e,
            )

    # Fetch data if requested
    if args.refresh:
        to_fetch = list(symbols)  # refresh all to get latest prices
        print("Refreshing data for {} stocks...".format(len(to_fetch)))
        ok_count = 0
        fail_count = 0
        for i, s in enumerate(to_fetch, 1):
            suffix = tw_suffix_map.get(s, '')
            sys.stdout.write("\r  Fetching... [{}/{}] {}{}   ".format(i, len(to_fetch), s, suffix))
            sys.stdout.flush()
            ok = fetch_stock(s, data_dir, suffix=suffix)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                sys.stdout.write("\r  [{}/{}] {}{} FAIL\n".format(i, len(to_fetch), s, suffix))
                sys.stdout.flush()
        sys.stdout.write("\n")
        print("  Refreshed: {} OK, {} failed".format(ok_count, fail_count))

    # Scan
    total = len(symbols)
    results = []
    no_data = 0

    print("\nScanning {} stocks...".format(total))

    for idx, symbol in enumerate(symbols, 1):
        sys.stdout.write("\r  Scanning stocks... [{}/{}] {}   ".format(idx, total, symbol))
        sys.stdout.flush()
        df = load_stock(symbol, data_dir)
        if df is None:
            no_data += 1
            continue
        r = analyze(symbol, df)
        if r:
            action, reason = classify_action(r)
            r['action'] = action
            r['reason'] = reason
            r['market'] = _classify_market(symbol)
            r.update(grade_conviction(r))
            results.append(r)
    sys.stdout.write("\n")

    # --- Post-processing pipeline ---
    apply_relative_strength(results)
    apply_rs_guard(results)
    apply_rs_score_adjustment(results)
    reclassify_borderline(results)
    market_regime, regime_msg, exit_pct = apply_market_regime(results)

    # --- Sector filter ---
    if args.sector:
        from src.analysis.market_structure import SECTOR_MAP as _SM
        allowed = set(s.upper() for s in args.sector)
        valid_sectors = set()
        for s in allowed:
            for name in set(_SM.values()):
                if s in name.upper() or name.upper() in s:
                    valid_sectors.add(name)
        results = [r for r in results if _SM.get(r['symbol'], '') in valid_sectors]

    # --- CSV / JSON output ---
    if args.output_format == 'csv':
        import csv
        writer = csv.writer(sys.stdout)
        writer.writerow(['Symbol','Action','Score','Price','Wave','Trend',
                         'R:R','RSI','ADX','Mom6M','RS_Rank','Grades','Reason'])
        for r in results:
            grades = f"[{r.get('trend_grade','?')}/{r.get('timing_grade','?')}/{r.get('risk_grade','?')}]"
            writer.writerow([
                r['symbol'], r.get('action',''), r.get('score',0),
                f"{r['price']:.2f}", r.get('wave',0), r.get('trend',''),
                f"{r.get('rr',0):.1f}", f"{r.get('rsi',0):.0f}",
                f"{r.get('adx',0):.0f}", f"{r.get('mom_6m',0):.1f}",
                r.get('rs_rank',''), grades, r.get('reason',''),
            ])
        return

    # --- Market structure dashboard ---
    ms = analyze_market_structure(results)
    print()
    print(format_market_structure(ms))

    # Separate into categories
    buys = [r for r in results if r['action'] in ('STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION')]
    watches = [r for r in results if r['action'] == 'WATCH']
    holds = [r for r in results if r['action'] in ('HOLD', 'WAIT')]
    exits = [r for r in results if r['action'] == 'EXIT']
    avoids = [r for r in results if r['action'] in ('AVOID', 'SKIP')]

    # Sort
    sort_key = {
        'score': lambda x: x.get('score', 0),
        'rs': lambda x: -(x.get('rs_rank') or 9999),
        'rr': lambda x: x.get('rr', 0),
        'momentum': lambda x: x.get('mom_6m', 0),
    }[args.sort]
    buys.sort(key=sort_key, reverse=True)
    watches.sort(key=sort_key, reverse=True)

    # Output
    print()
    print("=" * 90)
    print("  WHAT TO BUY NOW — {}".format(datetime.now().strftime('%Y-%m-%d %H:%M')))
    print("  {} stocks scanned, {} with data, {} analyzed".format(
        total, total - no_data, len(results)))
    regime_colors = {'FAVORABLE': '\033[32m', 'MIXED': '\033[33m', 'CAUTION': '\033[31m'}
    print("  Market regime: {}{}{}  ({})".format(
        regime_colors.get(market_regime, ''), market_regime, '\033[0m', regime_msg))
    print("=" * 90)

    if buys:
        print()
        print("  \033[1;32m>>> BUY CANDIDATES ({}) <<<\033[0m".format(len(buys)))
        print()
        for r in buys:
            risk_pct = abs(r['price'] - r['stop']) / r['price'] * 100
            rs_info = f"  RS#{r['rs_rank']}" if 'rs_rank' in r else ""

            print("  \033[32m{}\033[0m  {}{}".format(r['action'], r['symbol'], rs_info))
            print("    {}".format(r.get('conviction_summary', r['reason'])))
            print("    Trend: {}  Timing: {}  Risk: {}".format(
                r.get('trend_grade', '?'), r.get('timing_grade', '?'), r.get('risk_grade', '?')))
            cur = 'NT$' if r.get('market') == 'TW' else '$'
            print("    Price: {}{:.2f}  |  Wave: {}  |  Conf: {:.0f}%  |  6M: {:+.1f}%".format(
                cur, r['price'], r['wave'], r['conf'] * 100, r.get('mom_6m', 0)))
            print()
            print("    \033[1mTRADE PLAN:\033[0m")
            print("      Entry:   {}{:.2f}  (current price)".format(cur, r['price']))
            print("      Stop:    {}{:.2f}  ({:.1f}% risk)".format(cur, r['stop'], risk_pct))
            print("      Target1: {}{:.2f}  (1.618 Fib extension)".format(cur, r['target1']))
            print("      Target2: {}{:.2f}  (2.618 Fib extension)".format(cur, r['target2']))
            print("      R:R      {:.1f} : 1".format(r['rr']))
            print()
            print("    Signals:  RSI={:.0f}  MACD={}  ADX={:.0f}  Speed={}".format(
                r['rsi'], r['macd_cross'] or '-', r['adx'], r['speed']))
            if r.get('rsi_div'):
                print("    *** {} ***".format(r['rsi_div'].upper()))
            print()
            print("    " + "-" * 70)
            print()
    else:
        print()
        print("  \033[33m>>> NO BUY CANDIDATES RIGHT NOW <<<\033[0m")
        print()
        print("  This is NORMAL. The system protects you by only buying when")
        print("  ALL signals align: wave structure + trend + momentum + risk/reward.")
        print("  Patience is a feature, not a bug.")
        print()

    if watches:
        top_n = args.top if hasattr(args, 'top') else 10
        print("  \033[33mWATCH LIST ({}) — Close to a buy, monitor daily:\033[0m".format(len(watches)))
        for r in watches[:top_n]:
            grades = f"[{r.get('trend_grade','?')}/{r.get('timing_grade','?')}/{r.get('risk_grade','?')}]"
            rs = f" RS#{r['rs_rank']}" if 'rs_rank' in r else ""
            cur = 'NT$' if r.get('market') == 'TW' else '$'
            print("    {:<8s}  {}{:<8.2f}  {:<7s} Wave {}  6M={:+.1f}%{}  |  {}".format(
                r['symbol'], cur, r['price'], grades, r['wave'], r.get('mom_6m', 0), rs, r['reason']))
        if len(watches) > top_n:
            print("    ... and {} more".format(len(watches) - top_n))
        print()

    if exits:
        print("  \033[31mEXIT / TAKE PROFIT ({}):\033[0m".format(len(exits)))
        for r in exits:
            cur = 'NT$' if r.get('market') == 'TW' else '$'
            print("    {:<8s}  {}{:<8.2f}  Wave {}  |  {}".format(
                r['symbol'], cur, r['price'], r['wave'], r['reason']))
            if args.verbose if hasattr(args, 'verbose') else False:
                print("      RSI={:.0f}({})  MACD={}  ADX={:.0f}  Speed={}  6M={:+.1f}%".format(
                    r['rsi'], r['rsi_zone'], r['macd_cross'] or '-', r['adx'],
                    r['speed'], r['mom_6m']))
        print()

    if holds:
        print("  WAITING ({}) — Impulse complete, watching for new wave:".format(len(holds)))
        for r in holds[:5]:
            cur = 'NT$' if r.get('market') == 'TW' else '$'
            print("    {:<8s}  {}{:<8.2f}  Wave {}  trend={}/{}".format(
                r['symbol'], cur, r['price'], r['wave'], r['pdir'], r['trend']))
        if len(holds) > 5:
            print("    ... and {} more".format(len(holds) - 5))
        print()

    if avoids:
        print("  AVOID ({}) — Bearish or low quality:".format(len(avoids)))
        for r in avoids[:5]:
            cur = 'NT$' if r.get('market') == 'TW' else '$'
            print("    {:<8s}  {}{:<8.2f}  |  {}".format(r['symbol'], cur, r['price'], r['reason']))
        if len(avoids) > 5:
            print("    ... and {} more".format(len(avoids) - 5))
        print()

    # Tip
    if no_data > 0:
        print("  TIP: {} stocks had no data. Run with --refresh to fetch.".format(no_data))
    if not args.expand:
        print("  TIP: Run with --expand to scan 70+ US stocks across all sectors.")
    print()

    # JSON export if --output is provided
    if args.output:
        export = {
            'scan_date': datetime.now().isoformat(),
            'market_regime': {
                'regime': market_regime,
                'exit_pct': round(exit_pct, 1),
                'message': regime_msg,
            },
            'buy_candidates': [
                {
                    'symbol': r['symbol'],
                    'score': r.get('score', 0),
                    'price': round(r['price'], 2),
                    'action': r['action'],
                    'wave_position': r['wave'],
                    'entry': round(r['price'], 2),
                    'stop': round(r['stop'], 2),
                    'target1': round(r['target1'], 2),
                    'target2': round(r['target2'], 2),
                    'rr': round(r['rr'], 2),
                    'confidence': round(r['conf'] * 100, 1),
                    'reason': r['reason'],
                }
                for r in buys
            ],
            'watch_list': [
                {
                    'symbol': r['symbol'],
                    'score': r.get('score', 0),
                    'price': round(r['price'], 2),
                    'action': r['action'],
                    'wave_position': r['wave'],
                    'reason': r['reason'],
                }
                for r in watches
            ],
            'summary': {
                'total_scanned': total,
                'with_data': total - no_data,
                'analyzed': len(results),
                'buy_candidates': len(buys),
                'watch_list': len(watches),
                'holds': len(holds),
                'exits': len(exits),
                'avoids': len(avoids),
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print("  Results saved to {}".format(args.output))


if __name__ == '__main__':
    main()
