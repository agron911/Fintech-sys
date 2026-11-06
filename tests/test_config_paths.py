"""Check that edge launches use the same data directory as the downloader."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils import config as config_module


class ConfigPathsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "repo"
        self.launch_dir = self.base / "shortcut"
        self.launch_dir.mkdir()
        (self.root / "config").mkdir(parents=True)
        self.settings = {
            "data_dir": "data",
            "stk2_dir": "data/raw",
            "adjustments_dir": "data/lists/adjustments",
            "processed_dir": "data/processed",
            "databases_dir": "data/raw/databases",
            "international_file": "data/lists/international.txt",
            "list_file": "data/lists/adjustments/list.xlsx",
            "otclist_file": "data/lists/adjustments/otclist.xlsx",
            "start_date": "2002-01-01",
        }
        self.config_path = self.root / "config/config.json"
        self.write_config()
        original_cwd = Path.cwd()
        os.chdir(self.launch_dir)
        self.addCleanup(os.chdir, original_cwd)
        module_file = self.root / "src/utils/config.py"
        patcher = patch.object(config_module, "__file__", str(module_file))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.settings), encoding="utf-8")

    def test_repo_paths_allow_reading_downloaded_prices_from_other_directory(self):
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        (raw / "AAPL.txt").write_text(
            "Date\tOpen\tHigh\tLow\tClose\tVolume\tDate\n"
            "2026/10/16\t100\t103\t99\t102\t1000\t2026/10/16\n",
            encoding="utf-8",
        )

        settings = config_module.load_config()

        with (Path(settings["stk2_dir"]) / "AAPL.txt").open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(rows[0]["Date"], "2026/10/16")
        self.assertEqual(rows[0]["Close"], "102")
        self.assertEqual(Path.cwd(), self.launch_dir)
        self.assertFalse((self.launch_dir / "data").exists())

    def test_stock_lists_and_directories_resolve_from_repo_without_rewriting_config(self):
        settings = config_module.load_config()

        for key, value in self.settings.items():
            if key != "start_date":
                self.assertEqual(settings[key], str(self.root / value))
        self.assertTrue((self.root / "data/raw/databases").is_dir())
        self.assertFalse(Path(settings["list_file"]).exists())
        self.assertEqual(json.loads(self.config_path.read_text()), self.settings)

    def test_custom_absolute_data_directory_is_preserved(self):
        external = self.base / "shared-prices"
        self.settings["stk2_dir"] = str(external)
        self.write_config()

        settings = config_module.load_config()

        self.assertEqual(settings["stk2_dir"], str(external))
        self.assertTrue(external.is_dir())


if __name__ == "__main__":
    unittest.main()
