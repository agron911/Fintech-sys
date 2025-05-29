import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend initially
import numpy as np
import pandas as pd

def resample_ohlc(df, freq):
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]
        if df.empty:
            return pd.DataFrame()
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        try:
            df_ohlc = df[required_cols].resample(freq).agg(ohlc_dict)
            df_ohlc = df_ohlc.dropna(how='all')
        except Exception as e:
            print(f"Resampling error: {e}")
            return df[required_cols].copy()
        if 'volume' in df_ohlc.columns:
            df_ohlc['volume'] = df_ohlc['volume'].fillna(0)
        if df_ohlc.empty and not df.empty:
            df_ohlc = pd.DataFrame(
                [[df['open'].iloc[0], df['high'].iloc[0], df['low'].iloc[0], df['close'].iloc[0], df.get('volume', pd.Series([0])).iloc[0]]],
                index=[df.index[0]],
                columns=required_cols
            )
        return df_ohlc
    except Exception as e:
        print(f"Error in resample_ohlc: {e}")
        return pd.DataFrame()

def cleanup_matplotlib():
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        gc.collect()
    except Exception as e:
        print(f"Error cleaning up matplotlib: {e}")

def safe_plot_cleanup(fig):
    try:
        if fig:
            fig.clear()
            import matplotlib.pyplot as plt
            plt.close(fig)
        gc.collect()
    except Exception as e:
        print(f"Error cleaning up figure: {e}")

def limit_dataframe_size(df, max_rows=10000):
    if len(df) > max_rows:
        return df.tail(max_rows).copy()
    return df

def resample_ohlc_improved(df, freq):
    """
    Improved OHLC resampling with better error handling
    """
    try:
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Resample using proper aggregation
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove rows with NaN values
        resampled = resampled.dropna()
        
        # Validate OHLC relationships
        valid_mask = (
            (resampled['high'] >= resampled['low']) &
            (resampled['high'] >= resampled['open']) &
            (resampled['high'] >= resampled['close']) &
            (resampled['low'] <= resampled['open']) &
            (resampled['low'] <= resampled['close']) &
            (resampled['volume'] >= 0)
        )
        
        return resampled[valid_mask]
        
    except Exception as e:
        print(f"Error in resample_ohlc_improved: {e}")
        return pd.DataFrame() 