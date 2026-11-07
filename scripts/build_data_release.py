#!/usr/bin/env python3
"""Package full stock data plus verified incremental release assets."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.data_snapshots import build_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-deltas", type=int, default=8)
    args = parser.parse_args()
    if args.keep_deltas < 1:
        parser.error("--keep-deltas must be positive")
    report = build_release(args.source, args.output, previous=args.previous,
                           previous_manifest=args.previous_manifest,
                           keep_deltas=args.keep_deltas)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
