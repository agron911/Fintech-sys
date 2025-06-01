import pandas as pd
from src.analysis.core.corrective import detect_corrective_patterns
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
import numpy as np

def test_detect_pattern_on_real_data_windows():
    # The file uses tab as delimiter and has a header row, skip the second row (Ticker)
    df = pd.read_csv('tests/2015.txt', delimiter='\t', skiprows=[1])
    # Clean up column names and parse dates (convert to lowercase)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
    # Remove any rows with missing close values
    df = df[df['close'].notnull()]
    found = False
    window_size = 500
    step = 100
    for start in range(0, len(df) - window_size, step):
        window = df.iloc[start:start+window_size]
        peaks, troughs = detect_peaks_troughs_enhanced(window)
        print(f"Window {start}-{start+window_size}: peaks={peaks}, troughs={troughs}")
        result = detect_corrective_patterns(window, 0)
        print(f"Pattern in window {start}-{start+window_size}: {result['type']}")
        if result['type'] != 'unknown':
            print("Pattern found:", result)
            found = True
            break
    assert found, "No pattern detected in any window" 