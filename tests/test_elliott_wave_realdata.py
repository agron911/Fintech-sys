import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.core.corrective_patterns import detect_corrective_patterns
from src.analysis.core.peaks import detect_peaks_troughs_enhanced
import numpy as np

def test_detect_pattern_on_real_data_windows():
    # The file uses tab as delimiter and has a header row
    df = pd.read_csv('tests/2015.txt', delimiter='\t')
    # Clean up column names and parse dates (convert to lowercase)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
    # Remove any rows with missing or non-numeric close values
    df = df[pd.to_numeric(df['close'], errors='coerce').notnull()]
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
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

def test_detect_peaks_troughs_on_tsla():
    df = pd.read_csv('data/raw/TSLA.txt', delimiter='\t')
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
    # Remove any rows with missing or non-numeric close values
    df = df[pd.to_numeric(df['close'], errors='coerce').notnull()]
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    peaks, troughs = detect_peaks_troughs_enhanced(df)
    print(f"TSLA: Detected {len(peaks)} peaks, {len(troughs)} troughs.")
    print(f"Peak indices: {peaks}")
    print(f"Trough indices: {troughs}")
    assert len(peaks) > 0 or len(troughs) > 0, "No peaks or troughs detected in TSLA!" 