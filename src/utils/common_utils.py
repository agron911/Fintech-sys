import gc
from pathlib import Path
import numpy as np
import pandas as pd

# -------- Data Loading & Preprocessing --------
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load raw tab-delimited data from file_path, parse dates, clean, validate OHLC, filter by recent years.
    """
    df = pd.read_csv(file_path, sep='\t')
    # Parse date and set index
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date')
    df = df.rename(columns=str.lower)

    # Convert numeric columns
    for col in ['price','open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop essential NaNs
    essential = ['open','high','low','close']
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    df = df.sort_index()

    # Validate OHLC
    mask = (df['high'] >= df['low']) & (df['high'] >= df['open']) & (df['high'] >= df['close'])
    mask &= (df['low'] <= df['open']) & (df['low'] <= df['close'])
    df = df.loc[mask]

    # Keep last 10 years
    if len(df) > 2500:
        cutoff = df.index.max() - pd.DateOffset(years=10)
        df = df[df.index >= cutoff]

    # Remove duplicates
    return df[~df.index.duplicated(keep='first')]

# -------- OHLC Resampling --------
def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample df to freq with OHLCV aggregation and validation.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure types
    for col in ['open','high','low','close','volume']:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
    # Aggregate
    ohlc = df.resample(freq).agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna(how='all')
    # Validate
    valid = (ohlc['high']>=ohlc['low']) & (ohlc['high']>=ohlc['open']) & (ohlc['high']>=ohlc['close'])
    valid &= (ohlc['low']<=ohlc['open']) & (ohlc['low']<=ohlc['close']) & (ohlc['volume']>=0)
    return ohlc.loc[valid]

# -------- Mapping & Annotations --------
def map_points_to_ohlc(original: pd.DataFrame, ohlc: pd.DataFrame, indices: np.ndarray, column: str, max_days: int=14) -> pd.Series:
    """
    Map indices from original df into the resampled OHLC df based on closest date within max_days.
    """
    series = pd.Series(index=ohlc.index, dtype=float)
    for idx in indices:
        if 0 <= idx < len(original):
            date, price = original.index[idx], original[column].iloc[idx]
            diffs = np.abs(ohlc.index - date)
            pos = diffs.argmin()
            if diffs[pos].days <= max_days:
                series.iloc[pos] = price
    return series

# -------- Styling & Utilities --------
def get_confidence_description(conf: float) -> str:
    if conf >= 0.8: return "Very High"
    if conf >= 0.6: return "High"
    if conf >= 0.4: return "Moderate"
    if conf >= 0.2: return "Low"
    return "Very Low"

_imps = {1:'lightgreen', 3:'green', 5:'orange'}
_corr = 'red'
_trans = 'purple'
def get_position_color(position: str) -> str:
    if 'Impulse' in position:
        for w,c in _imps.items():
            if f"{w}" in position: return c
        return 'lightgreen'
    if 'Corrective' in position: return _corr
    if 'Post-' in position or 'Transitional' in position: return _trans
    return 'gray'
