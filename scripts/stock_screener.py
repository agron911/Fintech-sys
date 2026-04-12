"""
Stock Screener: Scan, rank, and generate actionable trade recommendations.

Usage:
    python scripts/stock_screener.py                  # Scan all stocks, show top 10
    python scripts/stock_screener.py --top 5          # Show top 5
    python scripts/stock_screener.py --fetch           # Fetch data first, then scan
    python scripts/stock_screener.py --symbols TSLA NVDA AMD  # Scan specific stocks
    python scripts/stock_screener.py --output results.json     # Save to JSON
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.config import load_config
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_best_impulse_wave, find_elliott_wave_pattern_enhanced
from src.backtest.pattern_adapter import adapt_wave_data_to_strategy_input, _compute_trend_context
from src.backtest.trade_setup_generator import TradeSetupGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_stock_data(symbol: str, data_dir: str) -> Optional[pd.DataFrame]:
    """Load stock data from tab-separated file."""
    file_path = Path(data_dir) / f"{symbol}.txt"
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path, sep='\t')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')
        df.columns = [c.lower() for c in df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        return df
    except Exception as e:
        logger.warning(f"Error loading {symbol}: {e}")
        return None


def get_all_symbols(config: dict) -> List[str]:
    """Get all stock symbols from config files."""
    symbols = []
    try:
        intl = pd.read_csv(config['international_file'])
        symbols.extend(list(intl['code']))
    except Exception:
        pass
    try:
        listed = pd.read_excel(config['list_file'])
        symbols.extend(list(listed['code']))
    except Exception:
        pass
    try:
        otc = pd.read_excel(config['otclist_file'])
        symbols.extend(list(otc['code']))
    except Exception:
        pass
    return symbols


def fetch_data_if_needed(symbols: List[str], data_dir: str):
    """Fetch missing stock data using Yahoo Finance."""
    missing = [s for s in symbols if not (Path(data_dir) / f"{s}.txt").exists()]
    if not missing:
        print("All data files exist.")
        return

    print(f"Fetching data for {len(missing)} stocks...")
    try:
        from src.crawler.yahoo_finance import fetch_stock_data
        config = load_config()
        start = config.get('start_date', '2020-01-01')
        end = config.get('end_date', '2026-12-31')

        for i, symbol in enumerate(missing, 1):
            print(f"  [{i}/{len(missing)}] Fetching {symbol}...")
            try:
                fetch_stock_data(symbol, '', start, end)
            except Exception as e:
                print(f"    Failed: {e}")
    except ImportError:
        print("Cannot import crawler. Run scripts/run_fetch_data.py first.")


# ──────────────────────────────────────────────────────────────────────
# Analysis & Scoring
# ──────────────────────────────────────────────────────────────────────

def analyze_stock(symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Run full Elliott Wave analysis on a single stock.
    Returns a scored result dict or None if no pattern found.
    """
    if len(df) < 60:
        return None

    try:
        # Detect pattern
        wave_data = find_elliott_wave_pattern_enhanced(df, column='close')

        if not wave_data or wave_data.get('wave_type') in ('no_pattern', 'no_candidates'):
            return None

        # Run through adapter (adds trend context)
        pa = adapt_wave_data_to_strategy_input(df, wave_data, column='close')
        confidence = pa.get('confidence', 0)
        if confidence < 0.15:
            return None

        trend_ctx = pa.get('trend_context', {})
        position = pa.get('current_position')
        if not position:
            return None

        # Price metrics
        current_price = float(df['close'].iloc[-1])
        price_6m_ago = float(df['close'].iloc[-min(126, len(df))]) if len(df) >= 20 else current_price
        momentum_6m = (current_price / price_6m_ago - 1) if price_6m_ago > 0 else 0

        # Volatility (annualized)
        returns = df['close'].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 20 else 0

        # Wave info
        wave_number = position.get('wave_number', 0)
        trend_direction = position.get('trend_direction', 'unknown')

        # Generate trade setup using TradeSetupGenerator
        setup_gen = TradeSetupGenerator()
        wave_3_setup = setup_gen.identify_wave_3_entry(df, wave_data, current_price)
        wave_5_exit = setup_gen.identify_wave_5_exit(df, wave_data, current_price)

        # Determine action and compute composite score
        action, action_detail = _determine_action(
            wave_number, trend_direction, trend_ctx.get('trend', 'neutral'),
            confidence, wave_3_setup, wave_5_exit, momentum_6m
        )

        # Composite ranking score (higher = better opportunity)
        score = _compute_composite_score(
            confidence, trend_ctx.get('trend', 'neutral'),
            trend_direction, wave_number, momentum_6m, volatility,
            wave_3_setup, wave_5_exit
        )

        # Estimate holding period
        hold_estimate = _estimate_holding_period(df, position)

        # Compute key levels
        levels = _compute_key_levels(position, wave_3_setup, current_price)

        return {
            'symbol': symbol,
            'score': round(score, 3),
            'action': action,
            'action_detail': action_detail,
            'confidence': round(confidence, 3),
            'current_price': round(current_price, 2),
            'wave_number': wave_number,
            'trend_direction': trend_direction,
            'broad_trend': trend_ctx.get('trend', 'neutral'),
            'momentum_6m': round(momentum_6m * 100, 1),
            'volatility': round(volatility * 100, 1),
            'hold_estimate_days': hold_estimate,
            'entry_price': levels.get('entry'),
            'stop_loss': levels.get('stop_loss'),
            'target_1': levels.get('target_1'),
            'target_2': levels.get('target_2'),
            'risk_reward': levels.get('risk_reward'),
            'wave_3_setup': wave_3_setup is not None,
            'wave_5_exit': wave_5_exit is not None,
        }

    except Exception as e:
        logger.warning(f"Analysis failed for {symbol}: {e}")
        return None


def _determine_action(wave_num, pattern_dir, broad_trend, confidence,
                      wave_3_setup, wave_5_exit, momentum):
    """Determine BUY / SELL / HOLD recommendation."""

    # SELL signals
    if wave_5_exit:
        strength = wave_5_exit.get('action', 'EXIT')
        return 'SELL', f"Wave 5 completion — {strength}"

    if wave_num == 6:
        return 'SELL', "Post-impulse correction phase — take profits"

    if wave_num == 5 and confidence > 0.3:
        return 'HOLD_TIGHT', "In Wave 5 — trail stops, prepare to exit"

    # BUY signals
    if wave_3_setup and broad_trend == 'bullish' and pattern_dir == 'up':
        strength = wave_3_setup.signal_strength.value
        return 'STRONG_BUY', f"Wave 3 entry setup ({strength}) — highest probability trade"

    if wave_num in [1, 2] and broad_trend == 'bullish' and pattern_dir == 'up':
        if confidence > 0.35:
            return 'BUY', f"Early impulse (Wave {wave_num}) — trend-aligned entry"
        return 'WATCH', f"Wave {wave_num} forming — wait for Wave 2 completion"

    if wave_num == 3 and broad_trend == 'bullish' and pattern_dir == 'up':
        return 'BUY', "In Wave 3 — strongest impulse wave, add on pullbacks"

    if wave_num == 4 and broad_trend == 'bullish' and pattern_dir == 'up':
        if confidence > 0.35:
            return 'BUY_DIP', "Wave 4 correction — buy the dip for Wave 5"
        return 'WATCH', "Wave 4 correction — wait for support confirmation"

    # Counter-trend or unclear
    if broad_trend == 'bearish' and pattern_dir == 'down':
        return 'AVOID', "Bearish trend with downtrend impulse — stay away"

    if broad_trend == 'bearish' and pattern_dir == 'up':
        return 'WATCH', "Counter-trend rally in bear market — risky"

    return 'HOLD', "No clear setup — monitor for wave development"


def _compute_composite_score(confidence, broad_trend, pattern_dir, wave_num,
                             momentum, volatility, wave_3_setup, wave_5_exit):
    """
    Composite score for ranking stocks. Higher = better opportunity.
    Range: roughly 0-100.
    """
    score = 0

    # Base: pattern confidence (0-50 points)
    score += confidence * 50

    # Trend alignment bonus (0-20 points)
    if broad_trend == 'bullish' and pattern_dir == 'up':
        score += 20
    elif broad_trend == 'bearish' and pattern_dir == 'down':
        score += 10  # Less tradeable (long-only system)
    elif broad_trend != 'neutral' and broad_trend != pattern_dir:
        score -= 10  # Counter-trend penalty

    # Wave position bonus (0-15 points)
    wave_scores = {1: 8, 2: 12, 3: 15, 4: 6, 5: 3, 6: -5}
    score += wave_scores.get(wave_num, 0)

    # Momentum alignment (0-10 points)
    if momentum > 0 and pattern_dir == 'up':
        score += min(momentum * 20, 10)
    elif momentum < 0 and pattern_dir == 'down':
        score += min(abs(momentum) * 10, 5)

    # Wave 3 setup bonus (strongest signal)
    if wave_3_setup:
        score += 15
        if wave_3_setup.risk_reward_ratio > 3:
            score += 5

    # Wave 5 exit penalty (not a buy opportunity)
    if wave_5_exit:
        score -= 20

    # Volatility: moderate is best, extreme is risky
    if 0.15 < volatility < 0.40:
        score += 5
    elif volatility > 0.60:
        score -= 5

    return max(score, 0)


def _estimate_holding_period(df, position):
    """Estimate expected holding period in trading days."""
    wave_indices = position.get('wave_indices', [])
    if len(wave_indices) < 3:
        return None

    # Average wave duration from detected structure
    wave_durations = []
    for i in range(len(wave_indices) - 1):
        wave_durations.append(wave_indices[i + 1] - wave_indices[i])

    avg_duration = int(np.mean(wave_durations))
    wave_num = position.get('wave_number', 3)

    # Remaining waves to hold through
    if wave_num <= 3:
        # Entered early — hold through waves 3, 4, 5
        remaining_waves = 5 - wave_num + 1
        return avg_duration * remaining_waves
    elif wave_num == 4:
        # Hold through wave 5 only
        return avg_duration
    elif wave_num == 5:
        # Near exit
        return max(avg_duration // 2, 5)
    else:
        return 0  # Should exit


def _compute_key_levels(position, wave_3_setup, current_price):
    """Compute entry, stop, and target price levels."""
    levels = {}

    if wave_3_setup:
        levels['entry'] = round(wave_3_setup.entry_price, 2)
        levels['stop_loss'] = round(wave_3_setup.stop_loss, 2)
        if wave_3_setup.targets:
            levels['target_1'] = round(wave_3_setup.targets[0], 2)
            if len(wave_3_setup.targets) >= 2:
                levels['target_2'] = round(wave_3_setup.targets[1], 2)
        risk = abs(wave_3_setup.entry_price - wave_3_setup.stop_loss)
        reward = abs(wave_3_setup.targets[0] - wave_3_setup.entry_price) if wave_3_setup.targets else 0
        levels['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
    else:
        # Fallback levels from wave structure
        w2_low = position.get('wave_2_low') if position else None
        w1_range = position.get('wave_1_range', 0) if position else 0

        if w2_low and w1_range > 0:
            levels['entry'] = round(current_price, 2)
            levels['stop_loss'] = round(w2_low * 0.98, 2)
            levels['target_1'] = round(current_price + w1_range * 1.618, 2)
            levels['target_2'] = round(current_price + w1_range * 2.618, 2)
            risk = abs(current_price - levels['stop_loss'])
            reward = abs(levels['target_1'] - current_price)
            levels['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
        else:
            levels['entry'] = round(current_price, 2)
            levels['stop_loss'] = round(current_price * 0.95, 2)
            levels['target_1'] = round(current_price * 1.10, 2)
            levels['target_2'] = round(current_price * 1.20, 2)
            levels['risk_reward'] = 2.0

    return levels


# ──────────────────────────────────────────────────────────────────────
# Output Formatting
# ──────────────────────────────────────────────────────────────────────

ACTION_COLORS = {
    'STRONG_BUY': '\033[1;32m',  # Bold green
    'BUY': '\033[32m',           # Green
    'BUY_DIP': '\033[36m',       # Cyan
    'WATCH': '\033[33m',         # Yellow
    'HOLD': '\033[37m',          # White
    'HOLD_TIGHT': '\033[33m',    # Yellow
    'SELL': '\033[31m',          # Red
    'AVOID': '\033[91m',         # Light red
}
RESET = '\033[0m'


def print_results(results: List[Dict], top_k: int):
    """Print formatted results table."""
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    top = results[:top_k]

    print(f"\n{'='*90}")
    print(f"  ELLIOTT WAVE STOCK SCREENER — Top {len(top)} of {len(results)} stocks analyzed")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}\n")

    for i, r in enumerate(top, 1):
        color = ACTION_COLORS.get(r['action'], '')
        action_str = f"{color}{r['action']}{RESET}"

        print(f"  #{i}  {r['symbol']:<8s}  Score: {r['score']:6.1f}  |  {action_str}")
        print(f"      {r['action_detail']}")
        print(f"      Price: ${r['current_price']:<10.2f}  Wave: {r['wave_number']}  "
              f"Trend: {r['broad_trend']:<8s}  Confidence: {r['confidence']:.0%}")
        print(f"      6M Momentum: {r['momentum_6m']:+.1f}%  Volatility: {r['volatility']:.1f}%")

        if r.get('entry_price'):
            print(f"      Entry: ${r['entry_price']:.2f}  "
                  f"Stop: ${r['stop_loss']:.2f}  "
                  f"T1: ${r['target_1']:.2f}  "
                  f"T2: ${r.get('target_2', 0):.2f}  "
                  f"R:R {r.get('risk_reward', 0):.1f}")

        if r.get('hold_estimate_days'):
            weeks = r['hold_estimate_days'] / 5
            print(f"      Est. Hold: ~{r['hold_estimate_days']} trading days (~{weeks:.0f} weeks)")

        print()

    # Summary
    buy_count = sum(1 for r in top if r['action'] in ('STRONG_BUY', 'BUY', 'BUY_DIP'))
    sell_count = sum(1 for r in top if r['action'] in ('SELL',))
    watch_count = sum(1 for r in top if r['action'] in ('WATCH', 'HOLD', 'HOLD_TIGHT'))

    print(f"{'─'*90}")
    print(f"  Summary: {buy_count} BUY  |  {sell_count} SELL  |  {watch_count} WATCH/HOLD")
    print(f"{'─'*90}\n")


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    results.sort(key=lambda x: x['score'], reverse=True)
    output = {
        'generated_at': datetime.now().isoformat(),
        'total_stocks_analyzed': len(results),
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Elliott Wave Stock Screener')
    parser.add_argument('--top', type=int, default=10, help='Show top K results (default: 10)')
    parser.add_argument('--fetch', action='store_true', help='Fetch missing data before scanning')
    parser.add_argument('--symbols', nargs='+', help='Scan specific symbols only')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    config = load_config()
    data_dir = config.get('stk2_dir', 'data/raw')

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_all_symbols(config)

    if not symbols:
        print("No stock symbols found. Check config files.")
        return

    # Fetch data if requested
    if args.fetch:
        fetch_data_if_needed(symbols, data_dir)

    # Scan
    print(f"Scanning {len(symbols)} stocks...")
    results = []
    skipped = 0

    for i, symbol in enumerate(symbols, 1):
        df = load_stock_data(symbol, data_dir)
        if df is None:
            skipped += 1
            continue

        if args.verbose:
            print(f"  [{i}/{len(symbols)}] Analyzing {symbol}...")

        result = analyze_stock(symbol, df)
        if result:
            results.append(result)

    if skipped:
        print(f"  ({skipped} stocks skipped — no data file)")

    if not results:
        print("\nNo patterns found. Make sure stock data exists in data/raw/")
        print("Run: python scripts/stock_screener.py --fetch")
        return

    # Output
    print_results(results, args.top)

    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()
