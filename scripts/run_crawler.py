import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.crawler.yahoo_finance import delete_files, YahooFinanceCrawler
from src.utils.config import load_config
from src.utils.log_setup import setup_logging

def main():
    setup_logging()
    config = load_config()
    crawler = YahooFinanceCrawler(config)

    # Cleanup old files
    delete_files()

    # Load stock lists
    international = pd.read_csv(config["international_file"])["code"]
    listed = pd.read_excel(config["list_file"])["code"]
    otc = pd.read_excel(config["otclist_file"])["code"]

    # Crawl data
    crawler.crawl(international, suffix="")
    crawler.crawl(listed, suffix=".TW")
    crawler.crawl(otc, suffix=".TWO")

if __name__ == "__main__":
    main() 