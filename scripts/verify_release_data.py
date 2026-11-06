#!/usr/bin/env python3
"""Check active stock files before publishing a data release."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def last_price_date(path):
    """Read a price file's final date, rejecting LFS pointers and empty files."""
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8").strip().split("\t")
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(header):
            raise ValueError("missing price columns (possibly a Git LFS pointer)")
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2048))
        line = stream.read().decode("utf-8").strip().splitlines()[-1]
    columns = line.split("\t")
    if len(columns) < 6:
        raise ValueError("incomplete price row")
    # A header-only file also fails here instead of counting as valid data.
    date = datetime.strptime(columns[0].strip().replace("/", "-"), "%Y-%m-%d").date()
    prices = [float(value) for value in columns[1:5]]
    if any(not math.isfinite(value) or value <= 0 for value in prices):
        raise ValueError("invalid price")
    return date


def verify_data(data_dir, *, today=None, max_age_days=7, min_files=10,
                min_fresh_ratio=0.9):
    today = today or datetime.now(timezone.utc).date()
    files = sorted(Path(data_dir).glob("*.txt"))
    fresh = 0
    stale = []
    invalid = []
    for path in files:
        try:
            date = last_price_date(path)
            age = (today - date).days
            if age < -1:
                raise ValueError("price date is in the future")
            if age <= max_age_days:
                fresh += 1
            else:
                stale.append({"file": path.name, "last_date": date.isoformat()})
        except (OSError, ValueError, IndexError) as error:
            invalid.append({"file": path.name, "error": str(error)})
    ratio = fresh / len(files) if files else 0
    return {
        "date": today.isoformat(),
        "files": len(files),
        "fresh": fresh,
        "fresh_ratio": round(ratio, 4),
        "stale": stale,
        "invalid": invalid,
        "passed": fresh >= min_files and ratio >= min_fresh_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--min-files", type=int, default=10)
    parser.add_argument("--min-fresh-ratio", type=float, default=0.9)
    args = parser.parse_args()
    result = verify_data(args.dir, max_age_days=args.max_age_days,
                         min_files=args.min_files, min_fresh_ratio=args.min_fresh_ratio)
    summary = {key: value for key, value in result.items() if key not in {"stale", "invalid"}}
    summary["stale"] = len(result["stale"])
    summary["invalid"] = len(result["invalid"])
    print(json.dumps(summary, indent=2))
    for category in ("stale", "invalid"):
        if result[category]:
            print(f"{category.capitalize()} examples: {json.dumps(result[category][:5])}")
    if not result["passed"]:
        raise SystemExit("Data release failed freshness validation")


if __name__ == "__main__":
    main()
