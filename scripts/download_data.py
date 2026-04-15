#!/usr/bin/env python3
"""
Download the latest stock data from GitHub Releases.

Usage:
    python scripts/download_data.py              # Download latest data
    python scripts/download_data.py --info       # Show release info without downloading
    python scripts/download_data.py --force      # Re-download even if data exists

The CI workflow uploads fresh stock price data every Friday/Saturday.
This script downloads and extracts it to data/raw/.
"""
import sys
import os
import argparse
import json
import tarfile
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

REPO = "agron911/Fintech-sys"
TAG = "data-latest"
ASSET_NAME = "stock-data.tar.gz"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_release_info():
    """Fetch release metadata from GitHub API."""
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("No data release found. Run the CI workflow first, or fetch directly:")
            print("  python scripts/what_to_buy.py --expand --refresh")
            sys.exit(1)
        raise


def get_download_url(release_info):
    """Extract the tarball download URL from release info."""
    for asset in release_info.get("assets", []):
        if asset["name"] == ASSET_NAME:
            return asset["browser_download_url"], asset["size"]
    print(f"Asset '{ASSET_NAME}' not found in release.")
    sys.exit(1)


def download_and_extract(url, size):
    """Download tarball and extract to project root."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {size / 1024 / 1024:.1f} MB ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = 0
            while True:
                chunk = resp.read(1024 * 256)  # 256 KB chunks
                if not chunk:
                    break
                tmp.write(chunk)
                total += len(chunk)
                pct = total / size * 100 if size else 0
                print(f"\r  {total / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()

    print("Extracting ...")
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(path=PROJECT_ROOT)

    os.unlink(tmp_path)

    file_count = len(list(DATA_DIR.glob("*.txt")))
    print(f"Done. {file_count} stock files in {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download latest stock data from GitHub Releases")
    parser.add_argument("--info", action="store_true", help="Show release info without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()

    release = get_release_info()

    print(f"Release: {release['name']}")
    print(f"Updated: {release['published_at'][:10]}")
    print(f"Notes:   {release.get('body', '').split(chr(10))[0]}")

    if args.info:
        return

    # Check if data already exists
    existing = list(DATA_DIR.glob("*.txt")) if DATA_DIR.exists() else []
    if existing and not args.force:
        print(f"\ndata/raw/ already has {len(existing)} files.")
        print("Use --force to re-download, or --info to check release date.")
        answer = input("Download anyway? [y/N] ").strip().lower()
        if answer != "y":
            print("Skipped.")
            return

    url, size = get_download_url(release)
    download_and_extract(url, size)


if __name__ == "__main__":
    main()
