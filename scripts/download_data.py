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
import ssl
from pathlib import Path
from email.utils import parsedate_to_datetime

REPO = "agron911/Fintech-sys"
TAG = "data-latest"
ASSET_NAME = "stock-data.tar.gz"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


class CertificateVerificationError(RuntimeError):
    """The local Python environment cannot verify an HTTPS certificate."""


def create_ssl_context(ca_bundle=None):
    """Use OS certificates and public roots, preserving configured CA bundles."""
    if ca_bundle is not None:
        return ssl.create_default_context(cafile=str(Path(ca_bundle).expanduser()))

    try:
        import truststore
    except ImportError:
        context = ssl.create_default_context()
    else:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    cafile = os.environ.get("SSL_CERT_FILE")
    capath = os.environ.get("SSL_CERT_DIR")
    if cafile or capath:
        context.load_verify_locations(cafile=cafile, capath=capath)
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())
    return context


def open_url(request, *, timeout, ssl_context):
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
    except (urllib.error.URLError, ssl.SSLCertVerificationError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise CertificateVerificationError(
                f"Unable to verify the HTTPS certificate: {reason}.\n"
                "Install certificate support with the same Python used to run this script:\n"
                "  python -m pip install --upgrade certifi truststore\n"
                "If your company network uses its own certificate issuer, install its CA\n"
                "in the OS trust store or pass --ca-bundle /path/to/company-ca.pem."
            ) from error
        raise


def get_direct_release_info(ssl_context):
    """Read public asset headers when GitHub's metadata API is rate limited."""
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET_NAME}"
    req = urllib.request.Request(url, method="HEAD")
    with open_url(req, timeout=30, ssl_context=ssl_context) as response:
        modified = response.headers.get("Last-Modified")
        updated = parsedate_to_datetime(modified).isoformat() if modified else "unknown"
        size = int(response.headers.get("Content-Length", 0))
    return {
        "name": f"Stock Data ({updated[:10]})",
        "published_at": updated,
        "body": "Current data-latest release asset (metadata API rate limited).",
        "assets": [{"name": ASSET_NAME, "updated_at": updated,
                    "browser_download_url": url, "size": size}],
    }


def get_release_info(ssl_context=None):
    """Fetch release metadata from GitHub API."""
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        with open_url(req, timeout=10, ssl_context=ssl_context or create_ssl_context()) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            detail = e.read().decode("utf-8", errors="replace")
            if (e.code == 429 or e.headers.get("X-RateLimit-Remaining") == "0"
                    or "rate limit" in detail.lower()):
                print("GitHub API rate limit reached; checking the release asset directly.")
                return get_direct_release_info(ssl_context or create_ssl_context())
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


def download_and_extract(url, size, ssl_context=None):
    """Download tarball and extract to project root."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {size / 1024 / 1024:.1f} MB ...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
            req = urllib.request.Request(url)
            with open_url(req, timeout=300,
                          ssl_context=ssl_context or create_ssl_context()) as resp:
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
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    file_count = len(list(DATA_DIR.glob("*.txt")))
    print(f"Done. {file_count} stock files in {DATA_DIR}")


def update_data(args, ssl_context):
    release = get_release_info(ssl_context)

    print(f"Release: {release['name']}")
    asset = next((item for item in release.get("assets", [])
                  if item["name"] == ASSET_NAME), {})
    print(f"Updated: {asset.get('updated_at') or release['published_at']}")
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
    download_and_extract(url, size, ssl_context)


def main():
    parser = argparse.ArgumentParser(description="Download latest stock data from GitHub Releases")
    parser.add_argument("--info", action="store_true", help="Show release info without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--ca-bundle", type=Path, help="Trusted CA certificates in PEM format")
    args = parser.parse_args()
    try:
        ssl_context = create_ssl_context(args.ca_bundle)
        update_data(args, ssl_context)
    except CertificateVerificationError as error:
        parser.exit(1, f"{error}\n")
    except FileNotFoundError as error:
        parser.exit(1, f"Cannot open file: {error}\n")


if __name__ == "__main__":
    main()
