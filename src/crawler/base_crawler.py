from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import logging

class BaseCrawler(ABC):
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config["data_dir"])
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def fetch_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        pass

    def save_data(self, df: pd.DataFrame, symbol: str, long_tail: bool = False):
        folder = self.data_dir
        filename = f"{symbol}_long_tail.txt" if long_tail else f"{symbol}.txt"
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
        df = df[['Price', 'Close', 'High', 'Low', 'Open', 'Volume', 'Date']]
        df.to_csv(folder / filename, sep="\t", index=False)
        self.logger.info(f"Saved data for {symbol} to {folder / filename}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame."""
        if df is None or df.empty:
            return None
        df = df.drop(columns=["Adj Close"], errors="ignore")
        df.index = df.index.strftime("%Y/%m/%d")
        df["Date"] = df.index
        return df
