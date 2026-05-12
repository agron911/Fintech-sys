#!/usr/bin/env python3
"""
Validate stock data files for corruption, staleness, and quality issues.

Usage:
    python scripts/validate_data.py              # Validate all files
    python scripts/validate_data.py --fix        # Auto-fix merge conflicts
    python scripts/validate_data.py --summary    # Show summary only
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_date(date_str: str) -> datetime:
    """Parse a date string, supporting both YYYY/MM/DD and YYYY-MM-DD."""
    date_str = date_str.strip()
    for fmt in ('%Y/%m/%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f'unrecognized date format: {date_str}')


def validate_file(filepath: Path, fix: bool = False) -> dict:
    """Validate a single data file. Returns dict of issues found."""
    issues = []
    categories = set()
    symbol = filepath.stem

    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return {'symbol': symbol, 'issues': [f'cannot_read: {e}'],
                'categories': {'format_errors'}}

    lines = content.strip().split('\n')

    # Check minimum size
    if len(lines) < 10:
        issues.append(f'truncated: only {len(lines)} lines')
        categories.add('format_errors')

    # Check for git merge conflict markers
    conflict_lines = [i for i, l in enumerate(lines)
                      if l.startswith('<<<<<<< ') or l == '======='
                      or l.startswith('>>>>>>> ')]
    if conflict_lines:
        issues.append(f'merge_conflicts: {len(conflict_lines)} markers')
        categories.add('format_errors')
        if fix:
            clean = [l for l in lines
                     if not l.startswith('<<<<<<< ')
                     and l != '======='
                     and not l.startswith('>>>>>>> ')]
            filepath.write_text('\n'.join(clean) + '\n', encoding='utf-8')
            issues[-1] += ' (FIXED)'

    # Check header
    if lines and not lines[0].startswith('Date'):
        issues.append('missing_header')
        categories.add('format_errors')

    # Validate ALL data rows (not just first 20)
    bad_rows = 0
    all_dates = []
    closes = []
    volumes = []

    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 5:
            bad_rows += 1
            continue
        try:
            date = _parse_date(parts[0])
            o, h, l, c = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
            vol = int(float(parts[5])) if len(parts) > 5 else 0

            all_dates.append(date)
            closes.append(c)
            volumes.append(vol)

            if h < l:
                bad_rows += 1
            if o <= 0 or c <= 0:
                bad_rows += 1
        except (ValueError, IndexError):
            bad_rows += 1

    if bad_rows > 0:
        issues.append(f'bad_rows: {bad_rows}')
        categories.add('format_errors')

    # --- Duplicate date detection ---
    if all_dates:
        date_counts = Counter(all_dates)
        duplicates = [(d, c) for d, c in date_counts.items() if c > 1]
        if duplicates:
            issues.append(
                f'  {len(duplicates)} duplicate date(s): '
                f'{[(d.strftime("%Y-%m-%d"), c) for d, c in duplicates[:5]]}'
            )
            categories.add('duplicates')

    # --- Gap detection (>5 calendar days) ---
    if len(all_dates) >= 2:
        sorted_dates = sorted(all_dates)
        gaps = []
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i - 1]).days
            if gap > 5:
                gaps.append((sorted_dates[i - 1], sorted_dates[i], gap))
        if gaps:
            issues.append(
                f'  {len(gaps)} suspicious gap(s) (>5 days): showing first 3'
            )
            for start, end, days in gaps[:3]:
                issues.append(
                    f'    {start.strftime("%Y-%m-%d")} -> '
                    f'{end.strftime("%Y-%m-%d")} ({days} days)'
                )
            categories.add('gaps')

    # --- Price jump detection (>50% close-to-close) ---
    if len(closes) >= 2:
        jumps = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                change = abs(closes[i] - closes[i - 1]) / closes[i - 1]
                if change > 0.5:
                    jumps.append((all_dates[i].strftime('%Y-%m-%d'),
                                  closes[i - 1],
                                  closes[i], change * 100))
        if jumps:
            issues.append(
                f'  {len(jumps)} price jump(s) (>50%): showing first 3'
            )
            for date, prev, curr, pct in jumps[:3]:
                issues.append(
                    f'    {date}: ${prev:.2f} -> ${curr:.2f} ({pct:+.0f}%)'
                )
            categories.add('price_jumps')

    # --- Zero-volume flagging ---
    if volumes:
        zero_vol = sum(1 for v in volumes if v == 0)
        if zero_vol > 0:
            issues.append(f'  {zero_vol} zero-volume row(s)')
            categories.add('zero_volume')

    # Check staleness (last parsed date)
    if all_dates:
        last_date = max(all_dates)
        days_old = (datetime.now() - last_date).days
        if days_old > 30:
            issues.append(
                f'stale: {days_old} days old '
                f'(last: {last_date.strftime("%Y-%m-%d")})'
            )

    return {'symbol': symbol, 'issues': issues, 'categories': categories}


def main():
    parser = argparse.ArgumentParser(description='Validate stock data files')
    parser.add_argument('--fix', action='store_true',
                        help='Auto-fix merge conflicts')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary only')
    parser.add_argument('--dir', default='data/raw', help='Data directory')
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f'Error: {data_dir} does not exist')
        sys.exit(1)

    files = sorted(data_dir.glob('*.txt'))
    print(f'Validating {len(files)} files in {data_dir}...')

    results = []
    for f in files:
        r = validate_file(f, fix=args.fix)
        results.append(r)

    # --- Cross-file duplicate detection ---
    import hashlib
    hash_to_symbols = {}
    for f in files:
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()
            sym = f.stem
            hash_to_symbols.setdefault(h, []).append(sym)
        except Exception:
            pass

    dup_groups = {h: syms for h, syms in hash_to_symbols.items() if len(syms) > 1}
    dup_symbols = set()
    for syms in dup_groups.values():
        dup_symbols.update(syms)

    if dup_groups:
        for r in results:
            if r['symbol'] in dup_symbols:
                group = [s for h, syms in dup_groups.items()
                         for s in syms if r['symbol'] in syms]
                others = [s for s in group if s != r['symbol']][:3]
                r['issues'].append(
                    f'DUPLICATE FILE: identical data as {", ".join(others)}'
                )
                r['categories'].add('cross_duplicates')

    # Summarize
    clean = [r for r in results if not r['issues']]
    problems = [r for r in results if r['issues']]

    # --- Categorized summary ---
    category_counts = {
        'duplicates': 0,
        'cross_duplicates': 0,
        'gaps': 0,
        'price_jumps': 0,
        'zero_volume': 0,
        'format_errors': 0,
    }
    for r in results:
        for cat in r.get('categories', set()):
            if cat in category_counts:
                category_counts[cat] += 1

    print(f'\nData Validation Summary:')
    print(f'  Total files: {len(results)}')
    print(f'  Healthy: {len(clean)}')
    print(f'  Issues: {len(problems)}')
    if problems:
        print(f'    Duplicate dates: {category_counts["duplicates"]} files')
        print(f'    Duplicate files: {category_counts["cross_duplicates"]} files ({len(dup_groups)} groups)')
        print(f'    Gaps: {category_counts["gaps"]} files')
        print(f'    Price jumps: {category_counts["price_jumps"]} files')
        print(f'    Zero volume: {category_counts["zero_volume"]} files')
        print(f'    Format errors: {category_counts["format_errors"]} files')

    if problems and not args.summary:
        # Group by issue type
        issue_counts = {}
        for r in problems:
            for issue in r['issues']:
                key = issue.strip().split(':')[0]
                issue_counts[key] = issue_counts.get(key, 0) + 1

        print('\nIssue breakdown:')
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f'  {issue}: {count} files')

        print(f'\nFirst 20 problematic files:')
        for r in problems[:20]:
            print(f"  {r['symbol']}: {', '.join(r['issues'])}")
        if len(problems) > 20:
            print(f'  ... and {len(problems) - 20} more')

    if problems:
        merge_issues = sum(1 for r in problems
                          if any('merge_conflict' in i for i in r['issues']))
        if merge_issues and not args.fix:
            print(
                f'\nTIP: Run with --fix to auto-fix '
                f'{merge_issues} files with merge conflicts'
            )

    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
