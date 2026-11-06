"""Release checks must reject stale snapshots and LFS placeholder files."""

from datetime import date
import tempfile
import unittest
from pathlib import Path

from scripts.verify_release_data import verify_data


class ReleaseDataTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name)
        self.today = date(2026, 11, 6)

    def price_file(self, name, last_date="2026/11/05"):
        path = self.raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Date\tOpen\tHigh\tLow\tClose\tVolume\tDate\n"
            f"{last_date}\t100\t102\t99\t101\t500\t{last_date}\n",
            encoding="utf-8",
        )

    def check(self):
        return verify_data(self.raw, today=self.today, min_files=1)

    def test_recent_prices_pass(self):
        self.price_file("AAPL.txt")
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["fresh"], 1)

    def test_stale_snapshot_fails_even_when_file_exists(self):
        self.price_file("AAPL.txt", "2026/09/11")
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["stale"][0]["last_date"], "2026-09-11")

    def test_lfs_pointer_and_header_only_files_cannot_pass(self):
        (self.raw / "AAPL.txt").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:abcdef\nsize 100\n", encoding="utf-8",
        )
        (self.raw / "MSFT.txt").write_text(
            "Date\tOpen\tHigh\tLow\tClose\tVolume\tDate\n", encoding="utf-8",
        )
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["invalid"]), 2)

    def test_archived_copies_cannot_inflate_freshness(self):
        self.price_file("AAPL.txt", "2026/09/11")
        self.price_file("archive_temp/AAPL.txt")
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["files"], 1)

    def test_broad_fetch_failure_fails_despite_some_fresh_files(self):
        self.price_file("AAPL.txt")
        self.price_file("MSFT.txt", "2026/09/11")
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["fresh_ratio"], 0.5)

    def test_missing_data_directory_fails(self):
        result = verify_data(self.raw / "missing", today=self.today, min_files=1)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
