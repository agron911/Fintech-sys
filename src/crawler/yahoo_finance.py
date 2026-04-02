from pathlib import Path
import pandas as pd
import yfinance as yf
import traceback
from src.utils.config import load_config
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import requests
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import random
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging

logger = logging.getLogger(__name__)

from .base_crawler import BaseCrawler
from retrying import retry


# yf.pdr_override()


# Load config
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

def delete_files():
    path = save_file_path
    for file in path.glob("*.txt"):
        file.chmod(0o777)
        file.unlink()
    return "Files deleted"





def save_stock_data(df, stock_code, folder=save_file_path, long_tail=False):
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()

    # Ensure we have the required columns after flattening
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns after MultiIndex flattening: {missing_cols}")
        # If columns are still missing, this might be random data - skip column validation
        # Just save whatever we have
        df['Date'] = df.index
        if long_tail:
            df.to_csv(folder / f"{stock_code}_long_tail.txt", sep="\t", index=False)
        else:
            df.to_csv(folder / f"{stock_code}.txt", sep="\t", index=False)
        return

    # Insert 'Date' as the first column (from index)
    df['Date'] = df.index
    # Reorder columns and duplicate 'Date' at the end
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date_end'] = df['Date'].astype(str)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Date_end']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    if long_tail:
        df.to_csv(folder / f"{stock_code}_long_tail.txt", sep="\t", index=False)
    else:
        df.to_csv(folder / f"{stock_code}.txt", sep="\t", index=False)


def fetch_stock_data(stock_code, suffix, start, end):
    try:
        logger.info(f"{stock_code}{suffix}")
        df = yf.download(f"{stock_code}{suffix}", start=start, end=end)

        # Check if DataFrame is empty
        if df is None or df.empty:
            logger.info(f"No data returned for {stock_code}{suffix}")
            return df  # Return empty DataFrame (not None) to match expected behavior

        df = df.drop(columns=["Adj Close"], errors="ignore")

        # Only format date index if it's a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime("%Y/%m/%d , %r")
            df.index = df.index.str.split(",").str[0]

        df["Date"] = df.index
        return df
    except Exception as e:
        logger.info(f"Error fetching data for {stock_code}: {e}")
        traceback.print_exc()
        return None


def crawl_all_ch():
    logger.info(f'international_file absolute path: {international_file.resolve()}')
    international_stock = pd.read_csv(international_file)
    international_suffix = ""
    for code in international_stock["code"]:
        df = fetch_stock_data(code, international_suffix, start, end)
        if df is not None:
            save_stock_data(df, code)
            logger.info(f"Crawled: {code}")

    stock_list = pd.read_excel(list_file)
    stock_list["code"] = stock_list.iloc[:, 0]

    listed_code = '.TW'

    for code in stock_list.code:
        df = fetch_stock_data(code, listed_code, start, end)
        if df is not None:
            save_stock_data(df, code)
            logger.info(f"Crawled: {code}")

        else:
            logger.info(f"Error fetching data for {code}")
            
    df = fetch_stock_data("^TWII", "", start, end)

    if df is not None:
        save_stock_data(df, "TWII")

def crawl_otc_yf():
    stock_list = pd.read_excel(otclist_file)
    all_otc_stock = stock_list.iloc[:, 0]
    otc_code = ".TWO"
    
    for code in all_otc_stock:
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

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_data(self, symbol: str, start: str, end: str, suffix: str = "") -> pd.DataFrame:
        """Fetch data for a symbol. If suffix is provided it will be appended to the symbol

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
            self.logger.info(f"Fetching data for {symbol_with_suffix} from {start} to {end}")
            df = yf.download(symbol_with_suffix, start=start, end=end, progress=False)

            if df is None or df.empty:
                self.logger.warning(f"No data returned for {symbol_with_suffix}")
                return None

            return self.clean_data(df)
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol_with_suffix}: {e}")
            traceback.print_exc()
            return None

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
        for symbol in symbols:
            # Pass suffix through to fetch_data so logging and downloads show the full ticker
            df = self.fetch_data(symbol, self.config["start_date"], self.config["end_date"], suffix=suffix)
            if df is not None:
                self.save_data(df, symbol)
                self.logger.info(f"Crawled data for {symbol}")
            else:
                self.logger.warning(f"Failed to fetch data for {symbol}")