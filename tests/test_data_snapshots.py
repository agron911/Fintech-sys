"""Check incremental downloads preserve exact stock histories and recover safely."""

from datetime import date, timedelta
import io
import json
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import data_snapshots as snapshots
from scripts import download_data


class DataSnapshotsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.previous = self.base / "previous"
        self.current = self.base / "current"
        self.client = self.base / "client"
        self.output = self.base / "release"
        self.previous.mkdir()
        rng = random.Random(42)
        for name in ["AAPL.txt", "MSFT.txt"]:
            rows = [b"Date\tOpen\tHigh\tLow\tClose\tVolume\tDate\r\n"]
            for index in range(1000):
                day = (date(2020, 1, 1) + timedelta(days=index)).strftime("%Y/%m/%d")
                price = rng.uniform(100, 200)
                rows.append(f"{day}\t{price}\t201\t99\t{price}\t1000\t{day}\r\n".encode())
            (self.previous / name).write_bytes(b"".join(rows))
        shutil.copytree(self.previous, self.current)
        shutil.copytree(self.previous, self.client)
        self.append_row(self.current / "AAPL.txt")
        self.report = snapshots.build_release(self.current, self.output, previous=self.previous)
        self.manifest = snapshots.read_json(self.output / snapshots.MANIFEST_NAME)
        self.args = SimpleNamespace(info=False, force=True, full=False)

    @staticmethod
    def append_row(path, day="2026/11/06"):
        with path.open("ab") as stream:
            stream.write(f"{day}\t150\t152\t149\t151\t1000\t{day}\r\n".encode())

    def assert_target(self, directory=None, manifest=None):
        manifest = manifest or self.manifest
        self.assertEqual(snapshots.inventory(directory or self.client, manifest["files"]),
                         manifest["files"])

    def update(self, manifest=None, artifacts=None):
        manifest = manifest or self.manifest
        artifacts = artifacts or {p.name: p for p in self.output.glob("*.tar.gz")}
        downloads = []

        def fetch(asset, destination, ssl_context):
            downloads.append(asset["name"])
            shutil.copyfile(artifacts[asset["name"]], destination)
            return asset["size"]

        with patch.object(download_data, "DATA_DIR", self.client), \
                patch.object(download_data, "download_asset", side_effect=fetch):
            download_data.update_from_manifest(self.args, manifest, object())
        return downloads

    def test_append_patch_is_small_and_byte_exact(self):
        delta = self.manifest["deltas"][0]
        self.assertLess(delta["size"], self.manifest["full"]["size"] / 10)
        count = snapshots.apply_delta(self.output / delta["name"], self.client,
                                      delta["base"], delta["target"])
        self.assertEqual(count, 1)
        self.assert_target()
        self.assertEqual((self.client / "AAPL.txt").read_bytes(),
                         (self.current / "AAPL.txt").read_bytes())

    def test_client_uses_delta_instead_of_full_archive(self):
        downloads = self.update()
        self.assertEqual(downloads, [self.manifest["deltas"][0]["name"]])
        self.assert_target()

    def test_existing_current_data_is_adopted_without_price_download(self):
        shutil.copyfile(self.current / "AAPL.txt", self.client / "AAPL.txt")
        self.assertEqual(self.update(), [])
        self.assertEqual(snapshots.load_state(self.client)["snapshot"], self.manifest["snapshot"])
        self.assertEqual(self.update(), [])

    def test_local_edit_uses_full_repair_and_preserves_custom_stock(self):
        stock = self.client / "AAPL.txt"
        stock.write_bytes(stock.read_bytes().replace(b"1000", b"1001", 1))
        (self.client / "CUSTOM.txt").write_bytes(b"local data")
        self.assertEqual(self.update(), [snapshots.FULL_NAME])
        self.assert_target()
        self.assertEqual((self.client / "CUSTOM.txt").read_bytes(), b"local data")

    def test_bad_delta_falls_back_to_full_snapshot(self):
        delta = self.manifest["deltas"][0]
        invalid = self.base / "invalid.tar.gz"
        invalid.write_bytes(b"not a tar archive")
        assets = {p.name: p for p in self.output.glob("*.tar.gz")}
        assets[delta["name"]] = invalid
        self.assertEqual(self.update(artifacts=assets), [delta["name"], snapshots.FULL_NAME])
        self.assert_target()

    def test_historical_revision_new_and_removed_stocks_are_exact(self):
        content = (self.current / "AAPL.txt").read_bytes().replace(b"1000", b"999", 1)
        (self.current / "AAPL.txt").write_bytes(content)
        (self.current / "MSFT.txt").unlink()
        shutil.copyfile(self.previous / "MSFT.txt", self.current / "TSLA.txt")
        (self.client / "CUSTOM.txt").write_bytes(b"keep")
        out = self.base / "revised"
        snapshots.build_release(self.current, out, previous=self.previous)
        manifest = snapshots.read_json(out / snapshots.MANIFEST_NAME)
        delta = manifest["deltas"][0]
        snapshots.apply_delta(out / delta["name"], self.client, delta["base"], delta["target"])
        self.assert_target(manifest=manifest)
        self.assertFalse((self.client / "MSFT.txt").exists())
        self.assertEqual((self.client / "CUSTOM.txt").read_bytes(), b"keep")

    def test_modified_base_is_rejected_before_any_other_file_is_changed(self):
        (self.client / "MSFT.txt").write_bytes(b"edited locally")
        before = snapshots.inventory(self.client)
        delta = self.manifest["deltas"][0]
        with self.assertRaisesRegex(snapshots.SnapshotError, "history"):
            snapshots.apply_delta(self.output / delta["name"], self.client,
                                  delta["base"], delta["target"])
        self.assertEqual(snapshots.inventory(self.client), before)

    def test_out_of_bounds_patch_is_rejected_before_installation(self):
        delta = self.manifest["deltas"][0]
        with tarfile.open(self.output / delta["name"]) as tar:
            index = json.load(tar.extractfile("delta.json"))
            payloads = {m.name: tar.extractfile(m).read() for m in tar.getmembers()
                        if m.name.startswith("payload/")}
        index["changes"]["AAPL.txt"]["operations"] = [["copy", 0, 10 ** 9]]
        bad = self.base / "bad-range.tar.gz"
        with snapshots.archive_writer(bad, 6) as tar:
            for name, content in payloads.items():
                snapshots.add_bytes(tar, name, content)
            snapshots.add_bytes(tar, "delta.json", snapshots.json_bytes(index))
        before = snapshots.inventory(self.client)
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.apply_delta(bad, self.client, delta["base"], delta["target"])
        self.assertEqual(snapshots.inventory(self.client), before)

    def test_full_archive_with_bad_checksum_preserves_current_data(self):
        bad = self.base / "bad-full.tar.gz"
        with snapshots.archive_writer(bad, 6) as tar:
            snapshots.add_bytes(tar, "data/raw/AAPL.txt", b"corrupted data")
        before = snapshots.inventory(self.client)
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.install_full(bad, self.client, self.manifest["files"])
        self.assertEqual(snapshots.inventory(self.client), before)

    def test_full_archive_rejects_paths_outside_raw(self):
        bad = self.base / "unsafe.tar.gz"
        with snapshots.archive_writer(bad, 6) as tar:
            snapshots.add_bytes(tar, "data/raw/../../outside.txt", b"outside")
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.install_full(bad, self.client, self.manifest["files"])
        self.assertFalse((self.base / "outside.txt").exists())

    def test_missed_updates_use_a_retained_chain(self):
        third = self.base / "third"
        shutil.copytree(self.current, third)
        self.append_row(third / "MSFT.txt", "2026/11/09")
        out = self.base / "third-release"
        snapshots.build_release(third, out, previous=self.current,
                                previous_manifest=self.output / snapshots.MANIFEST_NAME)
        manifest = snapshots.read_json(out / snapshots.MANIFEST_NAME)
        assets = {p.name: p for folder in [self.output, out] for p in folder.glob("*.tar.gz")}
        self.assertEqual(self.update(manifest, assets), [d["name"] for d in manifest["deltas"]])
        self.assert_target(manifest=manifest)

    def test_expired_chain_uses_full_snapshot(self):
        third = self.base / "third"
        shutil.copytree(self.current, third)
        self.append_row(third / "MSFT.txt", "2026/11/09")
        out = self.base / "expired-release"
        snapshots.build_release(third, out, previous=self.current,
                                previous_manifest=self.output / snapshots.MANIFEST_NAME,
                                keep_deltas=1)
        manifest = snapshots.read_json(out / snapshots.MANIFEST_NAME)
        assets = {p.name: p for folder in [self.output, out] for p in folder.glob("*.tar.gz")}
        self.assertEqual(self.update(manifest, assets), [snapshots.FULL_NAME])
        self.assert_target(manifest=manifest)

    def test_explicit_full_repair_bypasses_incremental_update(self):
        self.args.full = True
        self.assertEqual(self.update(), [snapshots.FULL_NAME])
        self.assert_target()

    def test_artifacts_are_deterministic_for_immutable_delta_names(self):
        out = self.base / "repeat-release"
        report = snapshots.build_release(self.current, out, previous=self.previous)
        self.assertEqual(report["full"]["sha256"], self.report["full"]["sha256"])
        self.assertEqual(report["delta"], self.report["delta"])


if __name__ == "__main__":
    unittest.main()
