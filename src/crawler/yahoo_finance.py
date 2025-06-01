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
print(save_file_path)

def delete_files():
    path = save_file_path
    for file in path.glob("*.txt"):
        file.chmod(0o777)
        file.unlink()
    return "Files deleted"





def save_stock_data(df, stock_code, folder= save_file_path, long_tail=False):
    # Ensure columns are named correctly
    df = df.rename(columns={
        df.columns[0]: 'Price',
        df.columns[1]: 'Close',
        df.columns[2]: 'High',
        df.columns[3]: 'Low',
        df.columns[4]: 'Open',
        df.columns[5]: 'Volume',
        df.columns[6]: 'Date',
    })
    # Only keep the expected columns
    df = df[['Price', 'Close', 'High', 'Low', 'Open', 'Volume', 'Date']]
    if long_tail:
        df.to_csv(folder / f"{stock_code}_long_tail.txt", sep="\t", index=False)
    else:
        df.to_csv(folder / f"{stock_code}.txt", sep="\t", index=False)


def fetch_stock_data(stock_code, suffix, start, end):
    try:
        df = yf.download(f"{stock_code}{suffix}", start=start, end=end)
        df = df.drop(columns=["Adj Close"], errors="ignore")
        df.index = df.index.strftime("%Y/%m/%d , %r")
        df.index = df.index.str.split(",").str[0]
        df["Date"] = df.index
        return df
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        traceback.print_exc()
        return None


def crawl_all_ch():
    print('international_file absolute path:', international_file.resolve())
    international_stock = pd.read_csv(international_file)
    international_suffix = ""
    for code in international_stock["code"]:
        df = fetch_stock_data(code, international_suffix, start, end)
        if df is not None:
            save_stock_data(df, code)
            print(f"Crawled: {code}")

    stock_list = pd.read_excel(list_file)
    stock_list["code"] = stock_list.iloc[:, 0]

    listed_code = '.TW'

    for code in stock_list.code:
        df = fetch_stock_data(code, listed_code, start, end)
        if df is not None:
            save_stock_data(df, code)
            print(f"Crawled: {code}")

        else:
            print(f"Error fetching data for {code}")
            
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
            print(f"Crawled: {code}")

        else:
            print(f"Error fetching data for {code}")
     


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
    def fetch_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, start=start, end=end)
            return self.clean_data(df)
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            traceback.print_exc()
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.drop(columns=["Adj Close"], errors="ignore")
        df.index = df.index.strftime("%Y/%m/%d")
        df['Date'] = df.index
        return df

    def save_data(self, df: pd.DataFrame, symbol: str):
        if df is not None and not df.empty:
            save_path = self.data_dir / f"{symbol}.txt"
            df.to_csv(save_path, sep="\t", index=True)
            self.logger.info(f"Saved data for {symbol} to {save_path}")

    def crawl(self, symbols: list, suffix: str = ""):
        for symbol in symbols:
            df = self.fetch_data(f"{symbol}{suffix}", self.config["start_date"], self.config["end_date"])
            if df is not None:
                self.save_data(df, symbol)
                self.logger.info(f"Crawled data for {symbol}")
            else:
                self.logger.warning(f"Failed to fetch data for {symbol}")