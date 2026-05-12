"""
Market Structure Analysis — aggregates individual stock analyses into a
market-level view: regime, breadth, wave distribution, cycle position,
sector rotation, and action bias.

Replaces the question "what should I buy?" with "where are we in the cycle?"
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Any

# Sector classification for known US tickers
_US_SECTOR_MAP = {
    # Technology
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech', 'GOOGL': 'Tech',
    'META': 'Tech', 'AMZN': 'Tech', 'NFLX': 'Tech', 'CRM': 'Tech',
    'ORCL': 'Tech', 'ADBE': 'Tech', 'NOW': 'Tech', 'INTU': 'Tech',
    'SHOP': 'Tech', 'SQ': 'Tech', 'PYPL': 'Tech', 'UBER': 'Tech',
    'ABNB': 'Tech', 'SNAP': 'Tech', 'PINS': 'Tech', 'SPOT': 'Tech',
    'ZM': 'Tech', 'DDOG': 'Tech', 'NET': 'Tech', 'CRWD': 'Tech',
    'ZS': 'Tech', 'PANW': 'Tech', 'PLTR': 'Tech', 'SNOW': 'Tech',
    'MDB': 'Tech', 'TEAM': 'Tech', 'TWLO': 'Tech', 'U': 'Tech',
    'CSCO': 'Tech', 'IBM': 'Tech', 'ADSK': 'Tech', 'FTNT': 'Tech',
    'ESTC': 'Tech', 'DELL': 'Tech', 'HPE': 'Tech', 'ANET': 'Tech',
    'APP': 'Tech', 'IONQ': 'Tech', 'SOUN': 'Tech', 'TTD': 'Tech',
    'RKLB': 'Tech', 'ROKU': 'Tech', 'EA': 'Tech', 'CMCSA': 'Tech',
    'BIDU': 'Tech', 'NTES': 'Tech', 'ADP': 'Tech', 'APH': 'Tech',
    'VRT': 'Tech', 'TEM': 'Tech',
    # Semiconductors
    'NVDA': 'Semis', 'AMD': 'Semis', 'AVGO': 'Semis', 'TSM': 'Semis',
    'QCOM': 'Semis', 'INTC': 'Semis', 'MU': 'Semis', 'AMAT': 'Semis',
    'LRCX': 'Semis', 'KLAC': 'Semis', 'MRVL': 'Semis', 'ON': 'Semis',
    'TXN': 'Semis', 'ADI': 'Semis', 'ASML': 'Semis', 'ARM': 'Semis',
    'SMCI': 'Semis', 'AAOI': 'Semis', 'MCHP': 'Semis', 'SWKS': 'Semis',
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'WFC': 'Finance', 'C': 'Finance', 'BRK-B': 'Finance', 'V': 'Finance',
    'MA': 'Finance', 'AXP': 'Finance', 'SCHW': 'Finance', 'BLK': 'Finance',
    'COIN': 'Finance', 'HOOD': 'Finance', 'SOFI': 'Finance',
    'BX': 'Finance', 'BK': 'Finance', 'ICE': 'Finance', 'AFRM': 'Finance',
    'MARA': 'Finance', 'AFL': 'Finance',
    # Healthcare
    'JNJ': 'Health', 'UNH': 'Health', 'LLY': 'Health', 'PFE': 'Health',
    'ABBV': 'Health', 'MRK': 'Health', 'TMO': 'Health', 'ABT': 'Health',
    'AMGN': 'Health', 'GILD': 'Health', 'ISRG': 'Health', 'MRNA': 'Health',
    'BIIB': 'Health', 'REGN': 'Health', 'VRTX': 'Health',
    'BMY': 'Health', 'DHR': 'Health', 'MDT': 'Health', 'HIMS': 'Health',
    'NVO': 'Health', 'CVS': 'Health', 'WBA': 'Health',
    # Consumer
    'TSLA': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer', 'MCD': 'Consumer',
    'DIS': 'Consumer', 'HD': 'Consumer', 'LOW': 'Consumer', 'TGT': 'Consumer',
    'COST': 'Consumer', 'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer',
    'PEP': 'Consumer', 'LULU': 'Consumer', 'RIVN': 'Consumer', 'LCID': 'Consumer',
    'CMG': 'Consumer', 'BKNG': 'Consumer', 'DLTR': 'Consumer', 'CHWY': 'Consumer',
    'CELH': 'Consumer', 'TOST': 'Consumer', 'DASH': 'Consumer', 'DKNG': 'Consumer',
    'CL': 'Consumer', 'PM': 'Consumer', 'MO': 'Consumer', 'MDLZ': 'Consumer',
    'F': 'Consumer', 'GM': 'Consumer', 'CCL': 'Consumer', 'LVS': 'Consumer',
    'TRIP': 'Consumer', 'PDD': 'Consumer', 'SE': 'Consumer', 'MELI': 'Consumer',
    'ACHR': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'OXY': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy',
    'PSX': 'Energy', 'HAL': 'Energy', 'EQT': 'Energy',
    'FSLR': 'Energy', 'ENPH': 'Energy', 'NEE': 'Energy', 'AEP': 'Energy',
    # Industrial
    'CAT': 'Industrial', 'DE': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial',
    'HON': 'Industrial', 'UPS': 'Industrial', 'RTX': 'Industrial', 'LMT': 'Industrial',
    'UNP': 'Industrial', 'MMM': 'Industrial', 'GD': 'Industrial',
    'DOW': 'Industrial', 'LIN': 'Industrial', 'APD': 'Industrial',
    # Telecom
    'T': 'Telecom', 'VZ': 'Telecom',
}

# Sector classification for TW (Taiwan) tickers
_TW_SECTOR_MAP = {
    # Semiconductors
    '2330': 'Semis', '3711': 'Semis', '2303': 'Semis', '3034': 'Semis',
    '2344': 'Semis', '3443': 'Semis', '6770': 'Semis', '2408': 'Semis',
    '3661': 'Semis', '2454': 'Semis', '5274': 'Semis', '3529': 'Semis',
    '2379': 'Semis', '6505': 'Semis', '3105': 'Semis', '2449': 'Semis',
    '6239': 'Semis', '8150': 'Semis', '5347': 'Semis', '6414': 'Semis',
    '3450': 'Semis', '8016': 'Semis', '6166': 'Semis', '2363': 'Semis',
    # Technology / Electronics
    '2317': 'Tech', '2382': 'Tech', '3231': 'Tech', '2356': 'Tech',
    '2353': 'Tech', '3013': 'Tech', '2324': 'Tech', '2395': 'Tech',
    '3044': 'Tech', '6669': 'Tech', '2301': 'Tech', '3037': 'Tech',
    '2345': 'Tech', '2357': 'Tech', '3017': 'Tech', '6285': 'Tech',
    '3035': 'Tech', '2365': 'Tech', '2377': 'Tech', '2376': 'Tech',
    # Finance
    '2881': 'Finance', '2882': 'Finance', '2883': 'Finance', '2884': 'Finance',
    '2885': 'Finance', '2886': 'Finance', '2887': 'Finance', '2888': 'Finance',
    '2889': 'Finance', '2890': 'Finance', '2891': 'Finance', '2892': 'Finance',
    '5880': 'Finance', '2880': 'Finance', '2834': 'Finance', '2838': 'Finance',
    '2867': 'Finance', '6005': 'Finance',
    # Telecom
    '2412': 'Telecom', '3045': 'Telecom', '4904': 'Telecom', '6561': 'Telecom',
    # Consumer / Food
    '1216': 'Consumer', '1301': 'Consumer', '2912': 'Consumer', '1795': 'Consumer',
    '9910': 'Consumer', '1227': 'Consumer', '1262': 'Consumer', '1215': 'Consumer',
    '1295': 'Consumer', '2207': 'Consumer', '9945': 'Consumer',
    # Healthcare / Biotech
    '4142': 'Health', '6446': 'Health', '4743': 'Health', '1760': 'Health',
    '4147': 'Health', '6472': 'Health', '4119': 'Health', '1733': 'Health',
    # Industrial / Materials
    '1303': 'Industrial', '1326': 'Industrial', '2002': 'Industrial',
    '1402': 'Industrial', '2101': 'Industrial', '1101': 'Industrial',
    '1102': 'Industrial', '1513': 'Industrial', '2603': 'Industrial',
    '2609': 'Industrial', '2615': 'Industrial', '2618': 'Industrial',
    # Energy / Utilities
    '6547': 'Energy', '3576': 'Energy', '6591': 'Energy',
}

SECTOR_MAP = {**_US_SECTOR_MAP, **_TW_SECTOR_MAP}


def analyze_market_structure(scan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate individual stock scan results into a market-level structure view.

    Args:
        scan_results: List of result dicts from analyze_stock() + classify_action()

    Returns:
        Dict with regime, breadth, wave_distribution, sector_rotation,
        cycle_position, conviction_level, and action_bias.
    """
    if not scan_results:
        return _empty_structure()

    n = len(scan_results)

    # --- Breadth ---
    above_sma200 = sum(1 for r in scan_results
                       if r.get('price_vs_sma200', 1.0) > 1.0)
    bullish_trend = sum(1 for r in scan_results if r.get('trend') == 'bullish')
    bearish_trend = sum(1 for r in scan_results if r.get('trend') == 'bearish')

    breadth_pct = above_sma200 / n * 100
    bullish_pct = bullish_trend / n * 100

    # --- Wave Distribution ---
    wave_counts = Counter(r.get('wave', 0) for r in scan_results)
    early_trend = sum(wave_counts.get(w, 0) for w in [1, 2])
    strongest = wave_counts.get(3, 0)
    maturing = sum(wave_counts.get(w, 0) for w in [4, 5])
    corrective = wave_counts.get(6, 0) + wave_counts.get(0, 0)

    wave_distribution = {
        'early_trend': {'count': early_trend, 'pct': early_trend / n * 100},
        'strongest': {'count': strongest, 'pct': strongest / n * 100},
        'maturing': {'count': maturing, 'pct': maturing / n * 100},
        'corrective': {'count': corrective, 'pct': corrective / n * 100},
    }

    # --- Cycle Position ---
    if strongest / n > 0.30:
        cycle = 'Mid-cycle expansion'
    elif early_trend / n > 0.25:
        cycle = 'Early recovery'
    elif maturing / n > 0.35:
        cycle = 'Late-cycle'
    elif corrective / n > 0.40:
        cycle = 'Correction / reset'
    else:
        cycle = 'Transition'

    # --- Regime ---
    avg_adx = _safe_mean([r.get('adx', 0) for r in scan_results])
    avg_rsi = _safe_mean([r.get('rsi', 50) for r in scan_results])
    action_counts = Counter(r.get('action', '') for r in scan_results)

    exit_avoid_pct = (action_counts.get('EXIT', 0) + action_counts.get('AVOID', 0)) / n * 100

    if breadth_pct >= 65 and avg_adx >= 22:
        regime = 'Trending Bullish'
        risk_level = 'Low' if exit_avoid_pct < 15 else 'Moderate'
    elif breadth_pct >= 50:
        regime = 'Bullish with caution'
        risk_level = 'Moderate'
    elif breadth_pct >= 35:
        regime = 'Mixed / Rotating'
        risk_level = 'Elevated'
    elif breadth_pct >= 20:
        regime = 'Weakening'
        risk_level = 'High'
    else:
        regime = 'Bearish'
        risk_level = 'High'

    # --- Conviction Level ---
    if breadth_pct >= 65 and strongest / n > 0.20 and exit_avoid_pct < 20:
        conviction = 'HIGH'
        action_bias = 'Stay invested, add on dips to SMA50'
    elif breadth_pct >= 50 and exit_avoid_pct < 30:
        conviction = 'MODERATE'
        action_bias = 'Hold positions, selective new entries in Wave 3 setups'
    elif breadth_pct >= 35:
        conviction = 'LOW'
        action_bias = 'Reduce exposure, tighten stops, no new positions'
    else:
        conviction = 'DEFENSIVE'
        action_bias = 'Raise cash, hedge, wait for breadth recovery above 50%'

    # --- Watch-for triggers ---
    watch_for = []
    if breadth_pct > 50 and breadth_pct < 65:
        watch_for.append(f'Breadth narrowing — currently {breadth_pct:.0f}%, watch for drop below 50%')
    if maturing / n > 0.30:
        watch_for.append(f'{maturing / n * 100:.0f}% of stocks in Wave 4-5 — late cycle risk')
    if avg_rsi > 65:
        watch_for.append(f'Market avg RSI {avg_rsi:.0f} — overbought, expect pullback')
    if avg_rsi < 35:
        watch_for.append(f'Market avg RSI {avg_rsi:.0f} — oversold, watch for reversal')
    if exit_avoid_pct > 25:
        watch_for.append(f'{exit_avoid_pct:.0f}% EXIT/AVOID signals — market stress')

    # --- Sector Rotation ---
    sector_rotation = _compute_sector_rotation(scan_results)

    return {
        'regime': regime,
        'risk_level': risk_level,
        'conviction': conviction,
        'action_bias': action_bias,
        'cycle_position': cycle,
        'breadth': {
            'above_sma200_pct': round(breadth_pct, 1),
            'bullish_pct': round(bullish_pct, 1),
            'bearish_pct': round(bearish_trend / n * 100, 1),
            'avg_adx': round(avg_adx, 1),
            'avg_rsi': round(avg_rsi, 1),
        },
        'wave_distribution': wave_distribution,
        'action_counts': dict(action_counts),
        'sector_rotation': sector_rotation,
        'watch_for': watch_for,
        'total_stocks': n,
    }


def format_market_structure(ms: Dict[str, Any]) -> str:
    """Format market structure as a printable dashboard string."""
    if not ms or ms.get('total_stocks', 0) == 0:
        return "  No market structure data available.\n"

    b = ms['breadth']
    wd = ms['wave_distribution']
    lines = []
    w = 64

    lines.append('=' * w)
    lines.append('  MARKET STRUCTURE DASHBOARD')
    lines.append('=' * w)
    lines.append('')
    lines.append(f'  REGIME:     {ms["regime"]}')
    lines.append(f'  RISK:       {ms["risk_level"]}')
    lines.append(f'  CYCLE:      {ms["cycle_position"]}')
    lines.append(f'  CONVICTION: {ms["conviction"]}')
    lines.append(f'  BIAS:       {ms["action_bias"]}')
    lines.append('')
    lines.append('-' * w)
    lines.append(f'  BREADTH ({ms["total_stocks"]} stocks)')
    lines.append('-' * w)
    lines.append(f'  Above SMA200:  {b["above_sma200_pct"]:>5.1f}%')
    lines.append(f'  Bullish trend: {b["bullish_pct"]:>5.1f}%')
    lines.append(f'  Avg ADX:       {b["avg_adx"]:>5.1f}    Avg RSI: {b["avg_rsi"]:.1f}')
    lines.append('')
    lines.append('-' * w)
    lines.append('  WAVE DISTRIBUTION')
    lines.append('-' * w)

    bar_width = 30
    for label, key in [('Wave 1-2 (early)',  'early_trend'),
                       ('Wave 3   (strong)', 'strongest'),
                       ('Wave 4-5 (mature)', 'maturing'),
                       ('Corrective       ', 'corrective')]:
        pct = wd[key]['pct']
        cnt = wd[key]['count']
        filled = int(pct / 100 * bar_width)
        bar = '#' * filled + '.' * (bar_width - filled)
        lines.append(f'  {label} [{bar}] {pct:>5.1f}% ({cnt})')

    # Sector rotation
    sr = ms.get('sector_rotation', [])
    if sr:
        lines.append('')
        lines.append('-' * w)
        lines.append('  SECTOR ROTATION')
        lines.append('-' * w)
        for s in sr:
            direction = s.get('direction', '')
            arrow = {'leading': '>>', 'neutral': '--', 'lagging': '<<'}.get(direction, '  ')
            lines.append(f'  {arrow} {s["sector"]:<12s}  '
                         f'avg_mom={s["avg_momentum"]:>+6.1f}%  '
                         f'bullish={s["bullish_pct"]:.0f}%  '
                         f'({s["count"]} stocks)')

    # Action summary
    ac = ms.get('action_counts', {})
    if ac:
        lines.append('')
        lines.append('-' * w)
        lines.append('  SIGNAL DISTRIBUTION')
        lines.append('-' * w)
        buy_total = sum(ac.get(a, 0) for a in ('STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION'))
        lines.append(f'  BUY signals:   {buy_total:>4d}    '
                     f'WATCH: {ac.get("WATCH", 0):>4d}    '
                     f'EXIT: {ac.get("EXIT", 0):>4d}    '
                     f'AVOID: {ac.get("AVOID", 0):>4d}')

    # Watch-for alerts
    wf = ms.get('watch_for', [])
    if wf:
        lines.append('')
        lines.append('-' * w)
        lines.append('  WATCH FOR')
        lines.append('-' * w)
        for alert in wf:
            lines.append(f'  ! {alert}')

    lines.append('=' * w)
    return '\n'.join(lines) + '\n'


def _compute_sector_rotation(results: List[Dict]) -> List[Dict]:
    """Group stocks by sector and compute relative strength."""
    sector_data = defaultdict(list)
    for r in results:
        sector = SECTOR_MAP.get(r.get('symbol', ''), 'Other')
        sector_data[sector].append(r)

    if not sector_data:
        return []

    rotation = []
    for sector, stocks in sector_data.items():
        avg_mom = _safe_mean([s.get('mom_6m', 0) for s in stocks])
        bullish_count = sum(1 for s in stocks if s.get('trend') == 'bullish')
        bullish_pct = bullish_count / len(stocks) * 100 if stocks else 0

        if avg_mom > 5 and bullish_pct > 60:
            direction = 'leading'
        elif avg_mom < -5 or bullish_pct < 30:
            direction = 'lagging'
        else:
            direction = 'neutral'

        rotation.append({
            'sector': sector,
            'avg_momentum': round(avg_mom, 1),
            'bullish_pct': round(bullish_pct, 1),
            'count': len(stocks),
            'direction': direction,
        })

    rotation.sort(key=lambda x: x['avg_momentum'], reverse=True)
    return rotation


def _safe_mean(values: list) -> float:
    nums = [v for v in values if v is not None and v == v]  # exclude NaN
    return sum(nums) / len(nums) if nums else 0


def _empty_structure() -> Dict[str, Any]:
    return {
        'regime': 'UNKNOWN',
        'risk_level': 'UNKNOWN',
        'conviction': 'UNKNOWN',
        'action_bias': 'Insufficient data',
        'cycle_position': 'Unknown',
        'breadth': {},
        'wave_distribution': {},
        'action_counts': {},
        'sector_rotation': [],
        'watch_for': [],
        'total_stocks': 0,
    }
