"""Verify HTTPS issuer handling and direct data/raw extraction."""

import io
import json
import os
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scripts import download_data


class DownloadTLSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("openssl") is None:
            raise unittest.SkipTest("openssl is required for the local HTTPS test")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        cls.ca = cls.base / "ca.pem"
        cls.key = cls.base / "server.key"
        cls.cert = cls.base / "server.pem"
        extensions = cls.base / "extensions.txt"
        extensions.write_text("subjectAltName=DNS:localhost\n", encoding="utf-8")
        commands = [
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(cls.base / "ca.key"), "-out", str(cls.ca), "-days", "1",
             "-subj", "/CN=Fintech test issuer"],
            ["openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(cls.key),
             "-out", str(cls.base / "server.csr"), "-subj", "/CN=localhost"],
            ["openssl", "x509", "-req", "-in", str(cls.base / "server.csr"),
             "-CA", str(cls.ca), "-CAkey", str(cls.base / "ca.key"), "-CAcreateserial",
             "-out", str(cls.cert), "-days", "1", "-extfile", str(extensions)],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True)
        contents = (
            b"Date\tOpen\tHigh\tLow\tClose\tVolume\tDate\n"
            b"2026/11/06\t100\t102\t99\t101\t1000\t2026/11/06\n"
        )
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            info = tarfile.TarInfo("data/raw/AAPL.txt")
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))
        cls.archive = archive.getvalue()
        cls.contents = contents

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(cls.archive)))
                self.send_header("Last-Modified", "Sat, 07 Nov 2026 04:00:00 GMT")
                self.end_headers()

            def do_GET(self):
                if self.path == "/asset":
                    body = cls.archive
                else:
                    body = json.dumps({
                        "name": "Test release",
                        "assets": [{"name": download_data.ASSET_NAME,
                                    "size": len(cls.archive),
                                    "browser_download_url": cls.url + "/asset"}],
                    }).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cls.cert, cls.key)
        cls.server.socket = context.wrap_socket(cls.server.socket, server_side=True)
        cls.url = f"https://localhost:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        env = patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1",
                                      "no_proxy": "localhost,127.0.0.1"})
        env.start()
        self.addCleanup(env.stop)

    def test_unknown_issuer_is_rejected_with_actionable_error(self):
        context = download_data.create_ssl_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        with self.assertRaisesRegex(download_data.CertificateVerificationError,
                                    "certifi truststore"):
            download_data.open_url(self.url, timeout=5, ssl_context=context)

    def test_explicit_trusted_issuer_allows_https(self):
        context = download_data.create_ssl_context(self.ca)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        with download_data.open_url(self.url, timeout=5, ssl_context=context) as response:
            self.assertEqual(json.load(response)["name"], "Test release")

    def test_release_and_asset_use_trusted_context_and_extract_directly_into_raw(self):
        context = download_data.create_ssl_context(self.ca)
        real_urlopen = urllib.request.urlopen
        seen = []

        def local_urlopen(request, *, timeout, context):
            seen.append(context)
            url = request.full_url
            if "api.github.com" in url:
                url = self.url
            return real_urlopen(url, timeout=timeout, context=context)

        with patch.object(download_data.urllib.request, "urlopen", side_effect=local_urlopen), \
                patch.object(download_data, "PROJECT_ROOT", self.root), \
                patch.object(download_data, "DATA_DIR", self.root / "data/raw"):
            release = download_data.get_release_info(context)
            url, size = download_data.get_download_url(release)
            download_data.download_and_extract(url, size, context)
        self.assertEqual(seen, [context, context])
        self.assertEqual((self.root / "data/raw/AAPL.txt").read_bytes(), self.contents)
        self.assertFalse((self.root / "data/raw/data").exists())
        self.assertFalse((self.root / "data/raw/archive_temp").exists())

    def test_certificate_failure_preserves_existing_data_and_cleans_download(self):
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        stock = raw / "AAPL.txt"
        stock.write_bytes(self.contents)
        real_temporary_file = tempfile.NamedTemporaryFile

        def download_file(**kwargs):
            return real_temporary_file(dir=self.root, **kwargs)

        with patch.object(download_data, "DATA_DIR", raw), \
                patch.object(download_data.tempfile, "NamedTemporaryFile", download_file):
            with self.assertRaises(download_data.CertificateVerificationError):
                download_data.download_and_extract(
                    self.url + "/asset", len(self.archive), download_data.create_ssl_context(),
                )
        self.assertEqual(stock.read_bytes(), self.contents)
        self.assertEqual(list(self.root.glob("*.tar.gz")), [])

    def test_missing_ca_bundle_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            download_data.create_ssl_context(self.root / "missing.pem")

    def test_api_rate_limit_falls_back_to_verified_public_asset(self):
        context = download_data.create_ssl_context(self.ca)
        real_open_url = download_data.open_url

        def limited_api(request, *, timeout, ssl_context):
            if "api.github.com" in request.full_url:
                raise urllib.error.HTTPError(
                    request.full_url, 403, "rate limited", {"X-RateLimit-Remaining": "0"},
                    io.BytesIO(b'{"message":"API rate limit exceeded"}'),
                )
            self.assertEqual(request.get_method(), "HEAD")
            self.assertIs(ssl_context, context)
            return real_open_url(urllib.request.Request(self.url, method="HEAD"),
                                 timeout=timeout, ssl_context=ssl_context)

        with patch.object(download_data, "open_url", side_effect=limited_api):
            release = download_data.get_release_info(context)
        url, size = download_data.get_download_url(release)
        self.assertEqual(size, len(self.archive))
        self.assertEqual(release["assets"][0]["updated_at"], "2026-11-07T04:00:00+00:00")
        self.assertIn("/releases/download/data-latest/stock-data.tar.gz", url)


if __name__ == "__main__":
    unittest.main()
