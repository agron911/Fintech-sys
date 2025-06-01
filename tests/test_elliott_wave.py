import numpy as np
import pandas as pd
import pytest
from src.analysis import elliott_wave
from src.analysis.core.corrective import detect_corrective_patterns

# --- Synthetic Data Setup ---
def make_simple_df():
    # 10 days, simple uptrend with volume
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'close': [10, 12, 11, 15, 13, 17, 15, 18, 16, 20],
        'volume': [100, 120, 110, 130, 115, 140, 120, 150, 125, 160]
    }
    return pd.DataFrame(data, index=dates)

def make_wave_points():
    # Indices for a 5-wave impulse (synthetic, not real market)
    return np.array([0, 1, 2, 3, 4])

def make_wave_points_long():
    # 6 points for more complete tests
    return np.array([0, 1, 2, 3, 4, 5])

# --- Tests for Validation Functions ---
def test_validate_wave_4_overlap_no_overlap():
    df = make_simple_df()
    wave_points = np.array([0, 1, 2, 3, 4])
    result = elliott_wave.validate_wave_4_overlap(df, wave_points)
    assert result['valid']
    assert result['reason'] == 'no_overlap_clean_impulse'

def test_validate_wave_4_overlap_with_overlap():
    df = make_simple_df()
    # Force overlap: wave_4_end inside wave_1 territory
    wave_points = np.array([0, 1, 2, 3, 1])
    result = elliott_wave.validate_wave_4_overlap(df, wave_points)
    assert 'has_overlap' in result
    assert result['has_overlap']
    assert not result['valid'] or result['is_diagonal']

def test_validate_diagonal_triangle_insufficient():
    df = make_simple_df()
    wave_points = np.array([0, 1, 2])
    result = elliott_wave.validate_diagonal_triangle(df, wave_points)
    assert not result['is_valid_diagonal']
    assert result['reason'] == 'insufficient_points'

def test_validate_wave_directions():
    # Uptrend impulse
    waves = {1: 2, 2: -1, 3: 4, 4: -2, 5: 3}
    valid, conf = elliott_wave.validate_wave_directions(waves)
    assert valid
    assert conf > 0
    # Downtrend impulse
    waves = {1: -2, 2: 1, 3: -4, 4: 2, 5: -3}
    valid, conf = elliott_wave.validate_wave_directions(waves)
    assert valid
    assert conf > 0

def test_validate_fibonacci_relationships():
    # Use ratios close to Fibonacci
    waves = {1: 10, 2: -6.18, 3: 16.18, 4: -3.82, 5: 6.18}
    dates = pd.date_range('2023-01-01', periods=6)
    score = elliott_wave.validate_fibonacci_relationships(waves, dates)
    assert 0 <= score <= 0.8

def test_validate_volume_patterns():
    df = make_simple_df()
    wave_points = np.array([0, 1, 2, 3, 4])
    score = elliott_wave.validate_volume_patterns(df, wave_points)
    assert 0 <= score <= 1

def test_validate_alternation_principle():
    df = make_simple_df()
    wave_points = np.array([0, 1, 2, 3, 4])
    score = elliott_wave.validate_alternation_principle(df, wave_points)
    assert 0 <= score <= 1

# --- Detection Functions ---
def test_detect_peaks_troughs():
    df = make_simple_df()
    peaks, troughs = elliott_wave.detect_peaks_troughs(df)
    assert isinstance(peaks, np.ndarray)
    assert isinstance(troughs, np.ndarray)

# This test checks the main detection pipeline (may return empty if data is too simple)
def test_detect_elliott_wave_complete_runs():
    df = make_simple_df()
    result = elliott_wave.detect_elliott_wave_complete(df)
    assert isinstance(result, dict)
    assert 'impulse_wave' in result
    assert 'confidence' in result

# --- Utility ---
def test_get_confidence_description():
    assert elliott_wave.get_confidence_description(0.85) == 'Very High'
    assert elliott_wave.get_confidence_description(0.65) == 'High'
    assert elliott_wave.get_confidence_description(0.45) == 'Moderate'
    assert elliott_wave.get_confidence_description(0.25) == 'Low'
    assert elliott_wave.get_confidence_description(0.05) == 'Very Low'

# --- Corrective Pattern Detection Tests ---
def make_zigzag_df():
    # A: 10->15 (+5), B: 15->12 (-3), C: 12->17 (+5)
    # B retraces 60% of A, C/A = 1.0
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'close': [10, 15, 12, 17, 14],
        'volume': [100, 110, 105, 120, 115]
    }
    return pd.DataFrame(data, index=dates)

def make_flat_df():
    # A: 10->15 (+5), B: 15->10 (-5), C: 10->15 (+5)
    # B retraces 100% of A
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'close': [10, 15, 10, 15, 10],
        'volume': [100, 110, 105, 120, 115]
    }
    return pd.DataFrame(data, index=dates)

def make_triangle_df():
    # Five points, alternating up/down, diminishing amplitude
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'close': [10, 12, 11, 11.8, 11.2],
        'volume': [100, 110, 105, 108, 106]
    }
    return pd.DataFrame(data, index=dates)

def test_detect_zigzag_pattern():
    df = make_zigzag_df()
    from src.analysis.core.peaks import detect_peaks_troughs_enhanced
    peaks, troughs = detect_peaks_troughs_enhanced(df)
    print('Zigzag peaks:', peaks, 'troughs:', troughs)
    result = detect_corrective_patterns(df, 0)
    print(result)
    assert result['type'] == 'ZIGZAG'
    assert result['confidence'] > 0.5
    assert len(result['points']) == 3

def test_detect_flat_pattern():
    df = make_flat_df()
    from src.analysis.core.peaks import detect_peaks_troughs_enhanced
    peaks, troughs = detect_peaks_troughs_enhanced(df)
    print('Flat peaks:', peaks, 'troughs:', troughs)
    result = detect_corrective_patterns(df, 0)
    print(result)
    assert 'FLAT' in result['type']
    assert result['confidence'] > 0.5
    assert len(result['points']) == 3

def test_detect_triangle_pattern():
    df = make_triangle_df()
    from src.analysis.core.peaks import detect_peaks_troughs_enhanced
    peaks, troughs = detect_peaks_troughs_enhanced(df)
    print('Triangle peaks:', peaks, 'troughs:', troughs)
    result = detect_corrective_patterns(df, 0)
    print(result)
    if result['type'] is not None:
        assert 'TRIANGLE' in result['type']
        assert len(result['points']) == 5 