import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional
from investment_system.src.utils.config import load_config

def init_database(db_path: Path) -> sqlite3.Connection:
    """
    Initialize SQLite database and create stocks table if it doesn't exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()
    return conn

def save_to_database(df: pd.DataFrame, symbol: str, db_path: Path):
    """
    Save stock data to SQLite database.
    """
    conn = init_database(db_path)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df['symbol'] = symbol
    df['date'] = df.index.strftime('%Y-%m-%d')
    df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].to_sql('stocks', conn, if_exists='replace', index=False)
    conn.close()

def load_from_database(symbol: str, db_path: Path) -> Optional[pd.DataFrame]:
    """
    Load stock data from SQLite database.
    """
    conn = init_database(db_path)
    query = f"SELECT * FROM stocks WHERE symbol = ? ORDER BY date"
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    if df.empty:
        return None
    df.set_index('date', inplace=True)
    return df 