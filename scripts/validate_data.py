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
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate_file(filepath: Path, fix: bool = False) -> dict:
    """Validate a single data file. Returns dict of issues found."""
    issues = []
    symbol = filepath.stem

    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return {'symbol': symbol, 'issues': [f'cannot_read: {e}']}

    lines = content.strip().split('\n')

    # Check minimum size
    if len(lines) < 10:
        issues.append(f'truncated: only {len(lines)} lines')

    # Check for git merge conflict markers
    conflict_lines = [i for i, l in enumerate(lines)
                      if l.startswith('<<<<<<< ') or l == '======='
                      or l.startswith('>>>>>>> ')]
    if conflict_lines:
        issues.append(f'merge_conflicts: {len(conflict_lines)} markers')
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

    # Spot-check a few data rows
    bad_rows = 0
    for line in lines[1:min(20, len(lines))]:
        parts = line.split('\t')
        if len(parts) < 5:
            bad_rows += 1
            continue
        try:
            o, h, l, c = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if h < l:
                bad_rows += 1
            if o <= 0 or c <= 0:
                bad_rows += 1
        except (ValueError, IndexError):
            bad_rows += 1
    if bad_rows > 0:
        issues.append(f'bad_rows: {bad_rows} in first 20')

    # Check staleness (last date in file)
    if len(lines) > 1:
        last_line = lines[-1]
        try:
            last_date_str = last_line.split('\t')[0].strip()
            last_date = datetime.strptime(last_date_str, '%Y/%m/%d')
            days_old = (datetime.now() - last_date).days
            if days_old > 30:
                issues.append(f'stale: {days_old} days old (last: {last_date_str})')
        except (ValueError, IndexError):
            pass

    return {'symbol': symbol, 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description='Validate stock data files')
    parser.add_argument('--fix', action='store_true', help='Auto-fix merge conflicts')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
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

    # Summarize
    clean = [r for r in results if not r['issues']]
    problems = [r for r in results if r['issues']]

    print(f'\nResults: {len(clean)} clean, {len(problems)} with issues')

    if problems and not args.summary:
        # Group by issue type
        issue_counts = {}
        for r in problems:
            for issue in r['issues']:
                key = issue.split(':')[0]
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
            print(f'\nTIP: Run with --fix to auto-fix {merge_issues} files with merge conflicts')

    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
