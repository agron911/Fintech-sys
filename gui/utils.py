import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend initially
import numpy as np
import pandas as pd

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

# Remove the definitions of resample_ohlc and resample_ohlc_improved.
# If any usage remains, import resample_ohlc from src.utils.common_utils. 