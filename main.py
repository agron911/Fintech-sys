#!/usr/bin/env python3
"""
Fintech-sys CLI dispatcher.

Usage:
    python main.py              # Show help
    python main.py scan         # Scan for buy candidates
    python main.py fetch        # Fetch stock data
    python main.py backtest     # Run backtest
    python main.py validate     # Validate data files
    python main.py gui          # Launch GUI
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

COMMANDS = {
    "scan":     ("what_to_buy.py",      "Scan stocks for buy candidates"),
    "fetch":    ("run_fetch_data.py",   "Fetch / update stock data from Yahoo Finance"),
    "backtest": ("run_backtest.py",     "Run backtest on strategies"),
    "validate": ("validate_data.py",    "Validate downloaded data files"),
    "gui":      ("run_gui.py",          "Launch the graphical interface"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fintech-sys -- Elliott Wave analysis & trading toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {cmd:<12s} {desc}" for cmd, (_, desc) in COMMANDS.items()
        ),
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    for cmd, (script, desc) in COMMANDS.items():
        sub.add_parser(cmd, help=desc, add_help=False)

    args, remaining = parser.parse_known_args()

    if args.command is None:
        print()
        print("  Welcome to Fintech-sys")
        print("  Elliott Wave analysis & trading toolkit")
        print()
        parser.print_help()
        sys.exit(0)

    script_name = COMMANDS[args.command][0]
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script_path)] + remaining,
        cwd=str(Path(__file__).resolve().parent),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
