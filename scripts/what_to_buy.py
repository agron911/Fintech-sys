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
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_best_impulse_wave
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input

# Extended universe of liquid US stocks across sectors
EXPANDED_UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'NFLX', 'CRM', 'ADBE', 'INTC', 'QCOM',
    'AVGO', 'MU', 'MRVL', 'ANET', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'ZS', 'NET',
    # Semiconductors
    'ASML', 'LRCX', 'KLAC', 'MCHP', 'ON', 'SWKS', 'TXN',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'BMY', 'GILD',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'OXY',
    # Consumer
    'COST', 'WMT', 'TGT', 'NKE', 'SBUX', 'MCD', 'DIS',
    # Industrial
    'CAT', 'DE', 'GE', 'BA', 'LMT', 'RTX',
    # EV / Clean energy
    'RIVN', 'LCID', 'FSLR', 'ENPH',
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
    """Fetch a single stock from Yahoo Finance.

    Args:
        symbol: Base ticker (e.g. 'AAPL' or '2330')
        data_dir: Directory to save the file
        start/end: Date range
        suffix: Yahoo Finance suffix ('' for US, '.TW' for TWSE, '.TWO' for OTC)
    """
    try:
        import yfinance as yf
        ticker = f"{symbol}{suffix}"
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            return False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.drop(columns=['Adj Close'], errors='ignore')
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y/%m/%d')
        df['Date'] = df.index
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        filepath = Path(data_dir) / f"{symbol}.txt"
        df.to_csv(filepath, sep='\t', index=False)
        return True
    except Exception:
        return False


def analyze(symbol, df):
    """Analyze one stock. Returns result dict or None."""
    if len(df) < 60:
        return None

    # Use last 500 bars for speed
    if len(df) > 500:
        df = df.iloc[-500:]

    peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')
    if len(peaks) < 3 or len(troughs) < 3:
        return None

    best = find_best_impulse_wave(df, peaks, troughs, column='close')
    if best.get('wave_type') in ('no_candidates', 'no_pattern'):
        return None

    wave_data = {
        'impulse_wave': best.get('wave_points', np.array([])),
        'confidence': best.get('confidence', 0),
        'wave_type': best.get('wave_type', 'unknown'),
        'pattern_relationships': {},
        'multiple_patterns': [],
    }

    pa = adapt_wave_data_to_strategy_input(df, wave_data, column='close')
    tc = pa.get('trend_context', {})
    pos = pa.get('current_position', {})
    mom = pa.get('momentum', {})

    if not pos:
        return None

    price = float(df['close'].iloc[-1])
    wave = pos.get('wave_number', 0)
    trend = tc.get('trend', 'neutral')
    pdir = pos.get('trend_direction', 'unknown')
    conf = pa.get('confidence', 0)

    entry_ok = mom.get('entry_ok', False)
    exit_warn = mom.get('exit_warning', False)
    composite = mom.get('composite_score', 0)
    regime = mom.get('regime', '?')
    stop_mult = mom.get('stop_multiplier', 1.0)

    rsi = mom.get('rsi', {})
    macd = mom.get('macd', {})
    adx = mom.get('adx', {})
    vel = mom.get('velocity', {})

    # Compute key levels
    w2_low = pos.get('wave_2_low', price * 0.95)
    w1_range = pos.get('wave_1_range', price * 0.05)
    w2_end = pos.get('wave_2_end', price)

    stop = w2_low * 0.98 * stop_mult + price * (1 - stop_mult) if w2_low else price * 0.95
    target1 = w2_end + w1_range * 1.618 if w1_range > 0 else price * 1.10
    target2 = w2_end + w1_range * 2.618 if w1_range > 0 else price * 1.20
    risk = abs(price - stop)
    reward = abs(target1 - price)
    rr = reward / risk if risk > 0 else 0

    # 6M momentum
    p6m = float(df['close'].iloc[-min(126, len(df))])
    mom_6m = (price / p6m - 1) * 100

    return {
        'symbol': symbol,
        'price': price,
        'wave': wave,
        'trend': trend,
        'pdir': pdir,
        'conf': conf,
        'entry_ok': entry_ok,
        'exit_warn': exit_warn,
        'composite': composite,
        'regime': regime,
        'rsi': rsi.get('value', 50),
        'rsi_zone': rsi.get('zone', '?'),
        'rsi_div': rsi.get('divergence', None),
        'macd_cross': macd.get('crossover', None),
        'macd_mom': macd.get('momentum', '?'),
        'adx': adx.get('value', 0),
        'adx_trending': adx.get('trending', False),
        'vel_5d': vel.get('velocity_5d', 0),
        'vel_20d': vel.get('velocity_20d', 0),
        'speed': vel.get('speed_regime', '?'),
        'mom_6m': mom_6m,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'rr': rr,
    }


def classify_action(r):
    """Classify into BUY / WATCH / HOLD / AVOID."""
    wave = r['wave']
    trend = r['trend']
    pdir = r['pdir']
    entry_ok = r['entry_ok']
    exit_warn = r['exit_warn']
    conf = r['conf']

    # Hard filters
    if trend == 'bearish':
        return 'AVOID', 'Bearish trend'
    if exit_warn and wave >= 5:
        return 'EXIT', 'Wave {} + exit warning'.format(wave)
    if pdir == 'down' and trend == 'bullish':
        return 'AVOID', 'Counter-trend pattern'
    if conf < 0.20:
        return 'SKIP', 'Low confidence ({:.0f}%)'.format(conf * 100)

    # BUY conditions: ALL must be true
    if (wave in [1, 2, 3] and
            trend == 'bullish' and
            pdir == 'up' and
            entry_ok and
            not exit_warn and
            conf >= 0.25 and
            r['rr'] >= 2.0):
        if wave == 3:
            return 'STRONG BUY', 'Wave 3 + momentum confirmed'
        elif wave == 2:
            return 'BUY', 'Wave 2 completion, preparing for Wave 3'
        else:
            return 'BUY', 'Wave 1, early trend'

    if (wave == 4 and
            trend == 'bullish' and
            entry_ok and
            r['rr'] >= 1.5):
        return 'BUY DIP', 'Wave 4 pullback in uptrend'

    # WATCH: close to a buy but missing something
    if wave in [1, 2, 3, 4] and trend == 'bullish' and pdir == 'up':
        missing = []
        if not entry_ok:
            missing.append('momentum')
        if r['rr'] < 2.0:
            missing.append('risk/reward')
        if conf < 0.25:
            missing.append('confidence')
        return 'WATCH', 'Missing: {}'.format(', '.join(missing) if missing else 'timing')

    # WATCH: Bullish trend stocks completing correction — potential new wave forming
    if wave in [5, 6] and trend == 'bullish' and entry_ok and not exit_warn:
        return 'WATCH', 'Bullish trend, watching for new Wave 1'

    if wave in [5, 6]:
        return 'HOLD', 'Wave {} — wait for new impulse'.format(wave)

    return 'WAIT', 'No clear setup'


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
    args = parser.parse_args()

    config = load_config()
    data_dir = config.get('stk2_dir', 'data/raw')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Build symbol list — US stocks
    symbols = []
    try:
        intl = pd.read_csv(config['international_file'])
        symbols.extend(list(intl['code']))
    except Exception:
        pass

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
        except Exception:
            pass
        try:
            otc_df = pd.read_excel(config['otclist_file'])
            tw_otc = [str(c) for c in otc_df.iloc[:, 0]]
        except Exception:
            pass

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
        except Exception:
            pass
        try:
            otc_df = pd.read_excel(config['otclist_file'])
            for c in otc_df.iloc[:, 0]:
                tw_suffix_map[str(c)] = '.TWO'
        except Exception:
            pass

    # Fetch data if requested
    if args.refresh:
        missing = [s for s in symbols if not (Path(data_dir) / f"{s}.txt").exists()]
        if missing:
            print("Fetching data for {} stocks...".format(len(missing)))
            for i, s in enumerate(missing, 1):
                suffix = tw_suffix_map.get(s, '')
                ok = fetch_stock(s, data_dir, suffix=suffix)
                status = 'OK' if ok else 'FAIL'
                print("  [{}/{}] {}{} {}".format(i, len(missing), s, suffix, status))

    # Scan
    total = len(symbols)
    results = []
    no_data = 0

    print("\nScanning {} stocks...".format(total))

    for symbol in symbols:
        df = load_stock(symbol, data_dir)
        if df is None:
            no_data += 1
            continue
        r = analyze(symbol, df)
        if r:
            action, reason = classify_action(r)
            r['action'] = action
            r['reason'] = reason
            results.append(r)

    # Separate into categories
    buys = [r for r in results if r['action'] in ('STRONG BUY', 'BUY', 'BUY DIP')]
    watches = [r for r in results if r['action'] == 'WATCH']
    holds = [r for r in results if r['action'] in ('HOLD', 'WAIT')]
    exits = [r for r in results if r['action'] == 'EXIT']
    avoids = [r for r in results if r['action'] in ('AVOID', 'SKIP')]

    # Sort buys by risk/reward
    buys.sort(key=lambda x: x['rr'], reverse=True)
    watches.sort(key=lambda x: x['conf'], reverse=True)

    # Output
    print()
    print("=" * 90)
    print("  WHAT TO BUY NOW — {}".format(datetime.now().strftime('%Y-%m-%d %H:%M')))
    print("  {} stocks scanned, {} with data, {} analyzed".format(
        total, total - no_data, len(results)))
    print("=" * 90)

    if buys:
        print()
        print("  \033[1;32m>>> BUY CANDIDATES ({}) <<<\033[0m".format(len(buys)))
        print()
        for r in buys:
            risk_pct = abs(r['price'] - r['stop']) / r['price'] * 100

            print("  \033[32m{}\033[0m  {}".format(r['action'], r['symbol']))
            print("    {}".format(r['reason']))
            print("    Price: ${:.2f}  |  Wave: {}  |  Confidence: {:.0f}%".format(
                r['price'], r['wave'], r['conf'] * 100))
            print()
            print("    \033[1mTRADE PLAN:\033[0m")
            print("      Entry:   ${:.2f}  (current price)".format(r['price']))
            print("      Stop:    ${:.2f}  ({:.1f}% risk)".format(r['stop'], risk_pct))
            print("      Target1: ${:.2f}  (1.618 Fib extension)".format(r['target1']))
            print("      Target2: ${:.2f}  (2.618 Fib extension)".format(r['target2']))
            print("      R:R      {:.1f} : 1".format(r['rr']))
            print()
            print("    Position sizing (1% account risk):")
            print("      $10K account  ->  {} shares, ${:.0f} position".format(
                max(1, int(100 / (r['price'] - r['stop']))) if r['price'] != r['stop'] else 1,
                min(10000 * 0.4, 100 / max(0.01, r['price'] - r['stop']) * r['price'])
            ))
            print("      $100K account ->  {} shares, ${:.0f} position".format(
                max(1, int(1000 / max(0.01, r['price'] - r['stop']))),
                min(100000 * 0.4, 1000 / max(0.01, r['price'] - r['stop']) * r['price'])
            ))
            print()
            print("    Signals:  RSI={:.0f}  MACD={}  ADX={:.0f}  Speed={}  6M={:+.1f}%".format(
                r['rsi'], r['macd_cross'] or '-', r['adx'], r['speed'], r['mom_6m']))
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
            print("    {:<8s}  ${:<8.2f}  Wave {}  Conf {:.0f}%  |  {}".format(
                r['symbol'], r['price'], r['wave'], r['conf'] * 100, r['reason']))
            if args.verbose if hasattr(args, 'verbose') else False:
                print("      RSI={:.0f}({})  MACD={}  ADX={:.0f}  Speed={}  6M={:+.1f}%  R:R={:.1f}".format(
                    r['rsi'], r['rsi_zone'], r['macd_cross'] or '-', r['adx'],
                    r['speed'], r['mom_6m'], r['rr']))
                if r.get('rsi_div'):
                    print("      *** {} ***".format(r['rsi_div'].upper()))
        if len(watches) > top_n:
            print("    ... and {} more".format(len(watches) - top_n))
        print()

    if exits:
        print("  \033[31mEXIT / TAKE PROFIT ({}):\033[0m".format(len(exits)))
        for r in exits:
            print("    {:<8s}  ${:<8.2f}  Wave {}  |  {}".format(
                r['symbol'], r['price'], r['wave'], r['reason']))
            if args.verbose if hasattr(args, 'verbose') else False:
                print("      RSI={:.0f}({})  MACD={}  ADX={:.0f}  Speed={}  6M={:+.1f}%".format(
                    r['rsi'], r['rsi_zone'], r['macd_cross'] or '-', r['adx'],
                    r['speed'], r['mom_6m']))
        print()

    if holds:
        print("  WAITING ({}) — Impulse complete, watching for new wave:".format(len(holds)))
        for r in holds[:5]:
            print("    {:<8s}  ${:<8.2f}  Wave {}  trend={}/{}".format(
                r['symbol'], r['price'], r['wave'], r['pdir'], r['trend']))
        if len(holds) > 5:
            print("    ... and {} more".format(len(holds) - 5))
        print()

    if avoids:
        print("  AVOID ({}) — Bearish or low quality:".format(len(avoids)))
        for r in avoids[:5]:
            print("    {:<8s}  ${:<8.2f}  |  {}".format(r['symbol'], r['price'], r['reason']))
        if len(avoids) > 5:
            print("    ... and {} more".format(len(avoids) - 5))
        print()

    # Tip
    if no_data > 0:
        print("  TIP: {} stocks had no data. Run with --refresh to fetch.".format(no_data))
    if not args.expand:
        print("  TIP: Run with --expand to scan 70+ US stocks across all sectors.")
    print()


if __name__ == '__main__':
    main()
