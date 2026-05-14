from pathlib import Path
import pandas as pd
import yfinance as yf
import traceback
from src.utils.config import load_config
import datetime as dt
import os
import requests
import numpy as np
import random
import time
import logging

logger = logging.getLogger(__name__)

from .base_crawler import BaseCrawler
from retrying import retry


# yf.pdr_override()

# Update interval in days - files older than this will be re-fetched
UPDATE_INTERVAL_DAYS = 1

# Lazy initialization state
_initialized = False
config = None
data_dir = None
stk2_dir = None
adjustments_dir = None
start = None
end = None
international_file = None
list_file = None
otclist_file = None
save_file_path = None


def _ensure_init():
    global _initialized, config, data_dir, stk2_dir, adjustments_dir
    global start, end, international_file, list_file, otclist_file, save_file_path
    if _initialized:
        return
    config = load_config()
    data_dir = Path(config['data_dir'])
    stk2_dir = Path(config['stk2_dir'])
    adjustments_dir = Path(config['adjustments_dir'])
    start = pd.to_datetime(config['start_date'])
    end = pd.to_datetime(config['end_date'])
    international_file = Path(config['international_file'])
    list_file = Path(config['list_file'])
    otclist_file = Path(config['otclist_file'])
    save_file_path = stk2_dir
    logger.info(save_file_path)
    _initialized = True

def should_update_file(file_path, max_age_days=UPDATE_INTERVAL_DAYS):
    """Check if a file should be updated based on data recency (not file mtime)."""
    if not file_path.exists():
        return True

    # Check the actual last data date inside the file — this is the real freshness indicator.
    # File mtime is unreliable because scans touch files without adding new data.
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 50:
                return True
            f.seek(max(0, size - 200))
            last_lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
            last_line = last_lines[-1]
            last_date_str = last_line.split('\t')[0].strip()
            last_date = dt.datetime.strptime(last_date_str, "%Y/%m/%d")
            data_age = (dt.datetime.now() - last_date).days
            if data_age >= max_age_days:
                logger.info(f"Data in {file_path.name} is {data_age} days old, re-fetching")
                return True
    except Exception:
        # Can't parse last date — check file mtime as fallback
        file_age = (dt.datetime.now() - dt.datetime.fromtimestamp(file_path.stat().st_mtime)).days
        if file_age >= max_age_days:
            return True

    return False

def delete_files():
    """Legacy function - kept for compatibility. Use incremental updates instead."""
    logger.info("Incremental update mode: only updating files older than {} days".format(UPDATE_INTERVAL_DAYS))
    return "Incremental update mode enabled"





def save_stock_data(df, stock_code, folder=None, long_tail=False):
    """Save stock data to TSV file.

    Delegates to YahooFinanceCrawler for consistent formatting.
    The long_tail parameter appends '_long_tail' to the filename.
    """
    _ensure_init()
    if df is None or (hasattr(df, 'empty') and df.empty):
        logger.warning(f"Cannot save empty data for {stock_code}")
        return

    target_dir = folder if folder is not None else save_file_path
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    crawler = YahooFinanceCrawler(config)
    original_data_dir = crawler.data_dir
    crawler.data_dir = target_dir

    save_symbol = f"{stock_code}_long_tail" if long_tail else stock_code
    crawler.save_data(df, save_symbol)
    crawler.data_dir = original_data_dir


def fetch_stock_data(stock_code, suffix, start, end):
    """Fetch stock data from Yahoo Finance.

    Delegates to YahooFinanceCrawler for consistent date formatting and retry logic.
    """
    _ensure_init()
    start_str = str(start).split(' ')[0] if start else '2002-01-01'
    end_str = str(end).split(' ')[0] if end else '2026-12-31'
    crawler = YahooFinanceCrawler(config)
    df = crawler.fetch_data(stock_code, start_str, end_str, suffix=suffix)
    return df


def crawl_all_ch():
    _ensure_init()
    logger.info(f'international_file absolute path: {international_file.resolve()}')
    international_stock = pd.read_csv(international_file)
    international_suffix = ""
    for code in international_stock["code"]:
        file_path = save_file_path / f"{code}.txt"
        if not should_update_file(file_path):
            continue
        
        df = fetch_stock_data(code, international_suffix, start, end)
        if df is not None:
            save_stock_data(df, code)
            logger.info(f"Crawled: {code}")

    stock_list = pd.read_excel(list_file)
    stock_list["code"] = stock_list.iloc[:, 0]

    listed_code = '.TW'

    for code in stock_list.code:
        file_path = save_file_path / f"{code}.txt"
        if not should_update_file(file_path):
            continue
        
        df = fetch_stock_data(code, listed_code, start, end)
        if df is not None:
            save_stock_data(df, code)
            logger.info(f"Crawled: {code}")
        else:
            logger.info(f"Error fetching data for {code}")
            
    file_path = save_file_path / "TWII.txt"
    if should_update_file(file_path):
        twii_df = fetch_stock_data("^TWII", "", start, end)
        if twii_df is not None:
            save_stock_data(twii_df, "TWII")

def crawl_otc_yf():
    _ensure_init()
    stock_list = pd.read_excel(otclist_file)
    all_otc_stock = stock_list.iloc[:, 0]
    otc_code = ".TWO"
    
    for code in all_otc_stock:
        file_path = save_file_path / f"{code}.txt"
        if not should_update_file(file_path):
            continue
        
        df = fetch_stock_data(code, otc_code, start, end)
        if df is not None:
            save_stock_data(df, code)
            logger.info(f"Crawled: {code}")
        else:
            logger.info(f"Error fetching data for {code}")
     


# execute the delete and crawl crawl_all_ch which is not deprecated

# delete_files()
# crawl_all_ch()
# crawl_otc_yf()

class YahooFinanceCrawler(BaseCrawler):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(config['stk2_dir'])

    def fetch_data(self, symbol: str, start: str, end: str, suffix: str = "") -> pd.DataFrame:
        """Fetch data for a symbol with retry on network errors.

        Args:
            symbol: base ticker symbol (e.g. '1295')
            start: start date string
            end: end date string
            suffix: optional suffix to append when calling yfinance (e.g. '.TW')

        Returns:
            Cleaned DataFrame or None on failure
        """
        symbol_with_suffix = f"{symbol}{suffix}" if suffix else symbol
        try:
            df = self._fetch_with_retry(symbol_with_suffix, start, end)
            if df is None or df.empty:
                self.logger.warning(f"No data returned for {symbol_with_suffix}")
                return None
            return self.clean_data(df)
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol_with_suffix}: {e}")
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _fetch_with_retry(self, symbol_with_suffix: str, start: str, end: str) -> pd.DataFrame:
        """Inner fetch that allows retry decorator to work on network errors."""
        self.logger.info(f"Fetching data for {symbol_with_suffix} from {start} to {end}")
        return yf.download(symbol_with_suffix, start=start, end=end, progress=False, auto_adjust=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop Adj Close if present
        df = df.drop(columns=["Adj Close"], errors="ignore")

        # Format index as date string
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime("%Y/%m/%d")

        # Add Date column
        df['Date'] = df.index

        return df

    def save_data(self, df: pd.DataFrame, symbol: str):
        if df is None or df.empty:
            self.logger.warning(f"Cannot save empty data for {symbol}")
            return

        try:
            # Ensure data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Reformat to match expected output format
            df_save = df.copy()

            # Ensure Date is in index
            if 'Date' in df_save.columns:
                df_save = df_save.drop(columns=['Date'])

            # Add Date column from index
            df_save.insert(0, 'Date', df_save.index)

            # Reorder columns: Date, Open, High, Low, Close, Volume, Date
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df_save = df_save[[col for col in expected_cols if col in df_save.columns]]

            # Duplicate Date column at end
            df_save['Date_end'] = df_save['Date'].astype(str)

            # Build new column list based on actual columns present
            new_columns = list(df_save.columns[:-1])  # All columns except Date_end
            new_columns.append('Date')  # Rename Date_end to Date
            df_save.columns = new_columns

            save_path = self.data_dir / f"{symbol}.txt"
            df_save.to_csv(save_path, sep="\t", index=False)
            self.logger.info(f"Saved data for {symbol} to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")
            traceback.print_exc()

    def crawl(self, symbols: list, suffix: str = ""):
        """Crawl data for symbols with incremental updates.
        
        Only fetches data for symbols whose files are older than UPDATE_INTERVAL_DAYS.
        """
        skipped_count = 0
        crawled_count = 0
        
        for symbol in symbols:
            # Check if file needs updating
            file_path = self.data_dir / f"{symbol}.txt"
            if not should_update_file(file_path):
                skipped_count += 1
                continue
            
            # Fetch and save data
            df = self.fetch_data(symbol, self.config["start_date"], self.config["end_date"], suffix=suffix)
            if df is not None:
                self.save_data(df, symbol)
                self.logger.info(f"Crawled data for {symbol}")
                crawled_count += 1
            else:
                self.logger.warning(f"Failed to fetch data for {symbol}")
        
        self.logger.info(f"Crawl complete - Fetched: {crawled_count}, Skipped: {skipped_count}")