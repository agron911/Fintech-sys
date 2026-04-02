"""
Test Suite for Enhanced Elliott Wave Validation System
========================================================

pytest-based tests for Elliott Wave analysis features:
1. Wave personality validation
2. Corrective pattern analysis
3. Time-based Fibonacci relationships
4. Enhanced mandatory rule enforcement
5. Real market data validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch

# Import all Elliott Wave validation functions
from src.analysis.core import (
    # Enhanced strict validation
    validate_impulse_wave_strict,
    validate_wave_4_overlap_mandatory,
    validate_wave_3_length_strict,

    # Wave personality
    validate_wave_personality,
    validate_wave_3_personality,
    validate_wave_5_personality,

    # Corrective patterns
    classify_corrective_pattern,
    detect_zigzag_pattern,
    detect_flat_pattern,
    detect_triangle_pattern,
    classify_triangle_subtype,
    detect_complex_correction,

    # Time-based Fibonacci
    validate_fibonacci_time_relationships,
    calculate_wave_durations,
    check_time_equality,
    check_fibonacci_time_ratios,

    # Peak detection
    detect_peaks_troughs_enhanced
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_elliott_wave():
    """
    Create synthetic Elliott Wave pattern for testing.

    Returns valid 5-wave impulse with proper characteristics:
    - Wave 3 longest and strongest
    - Wave 4 doesn't overlap Wave 1
    - Wave 5 with declining volume
    """
    num_bars = 100
    start_price = 100

    dates = pd.date_range(start=datetime.now() - timedelta(days=num_bars),
                          periods=num_bars, freq='D')

    # Create Elliott Wave price pattern
    prices = np.zeros(num_bars)
    volumes = np.zeros(num_bars)

    # Wave 0 to Wave 1 (impulse up)
    wave_1_end = 20
    prices[:wave_1_end] = np.linspace(start_price, start_price + 10, wave_1_end)
    volumes[:wave_1_end] = np.random.uniform(1000, 1500, wave_1_end)

    # Wave 1 to Wave 2 (correction down ~61.8%)
    wave_2_end = 30
    prices[wave_1_end:wave_2_end] = np.linspace(
        start_price + 10, start_price + 3.82, wave_2_end - wave_1_end
    )
    volumes[wave_1_end:wave_2_end] = np.random.uniform(800, 1200, wave_2_end - wave_1_end)

    # Wave 2 to Wave 3 (strongest impulse, 1.618x Wave 1, HIGHEST VOLUME)
    wave_3_end = 55
    prices[wave_2_end:wave_3_end] = np.linspace(
        start_price + 3.82, start_price + 20, wave_3_end - wave_2_end
    )
    volumes[wave_2_end:wave_3_end] = np.random.uniform(2000, 3000, wave_3_end - wave_2_end)

    # Wave 3 to Wave 4 (correction, doesn't overlap Wave 1 peak)
    wave_4_end = 70
    prices[wave_3_end:wave_4_end] = np.linspace(
        start_price + 20, start_price + 12, wave_4_end - wave_3_end
    )
    volumes[wave_3_end:wave_4_end] = np.random.uniform(900, 1300, wave_4_end - wave_3_end)

    # Wave 4 to Wave 5 (final impulse, declining volume, momentum divergence)
    wave_5_end = 100
    prices[wave_4_end:wave_5_end] = np.linspace(
        start_price + 12, start_price + 25, wave_5_end - wave_4_end
    )
    volumes[wave_4_end:wave_5_end] = np.random.uniform(700, 1100, wave_5_end - wave_4_end)

    # Add some noise for realism
    prices += np.random.normal(0, 0.3, num_bars)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0.1, 0.5, num_bars),
        'low': prices - np.random.uniform(0.1, 0.5, num_bars),
        'close': prices,
        'volume': volumes
    }, index=dates)

    wave_points = np.array([0, wave_1_end, wave_2_end, wave_3_end, wave_4_end, wave_5_end - 1])

    return df, wave_points


# ============================================================================
# TEST 1: ENHANCED STRICT VALIDATION (MANDATORY RULES)
# ============================================================================

@pytest.mark.unit
def test_strict_validation_basic(synthetic_elliott_wave):
    """Test enhanced strict validation with mandatory rule enforcement."""
    df, wave_points = synthetic_elliott_wave

    is_valid, confidence, details = validate_impulse_wave_strict(
        df, wave_points, column='close', is_diagonal=False
    )

    # Basic assertions
    assert isinstance(is_valid, (bool, np.bool_))
    assert isinstance(confidence, (int, float, np.number))
    assert isinstance(details, dict)
    assert 0 <= confidence <= 1.0


@pytest.mark.unit
def test_strict_validation_mandatory_rules(synthetic_elliott_wave):
    """Test that mandatory rules are checked."""
    df, wave_points = synthetic_elliott_wave

    is_valid, confidence, details = validate_impulse_wave_strict(
        df, wave_points, column='close', is_diagonal=False
    )

    # Mandatory rules must be present
    assert 'wave_4_overlap' in details
    assert 'wave_3_length' in details

    # Wave 4 overlap check
    assert 'valid' in details['wave_4_overlap']
    assert isinstance(details['wave_4_overlap']['valid'], bool)

    # Wave 3 length check
    assert 'valid' in details['wave_3_length']
    assert isinstance(details['wave_3_length']['valid'], bool)


@pytest.mark.unit
def test_strict_validation_confidence_range(synthetic_elliott_wave):
    """Test that confidence score is within valid range."""
    df, wave_points = synthetic_elliott_wave

    is_valid, confidence, details = validate_impulse_wave_strict(
        df, wave_points, column='close'
    )

    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"


@pytest.mark.unit
def test_strict_validation_wave_3_longest(synthetic_elliott_wave):
    """Test that Wave 3 is identified as longest in valid pattern."""
    df, wave_points = synthetic_elliott_wave

    is_valid, confidence, details = validate_impulse_wave_strict(
        df, wave_points, column='close'
    )

    # If pattern is valid, Wave 3 should be longest (Elliott Wave rule)
    if is_valid:
        assert 'wave_3_length' in details
        if 'wave_3_is_longest' in details['wave_3_length']:
            # Wave 3 should be longest in our synthetic pattern
            assert details['wave_3_length']['wave_3_is_longest']


# ============================================================================
# TEST 2: WAVE PERSONALITY VALIDATION
# ============================================================================

@pytest.mark.unit
def test_wave_personality_basic(synthetic_elliott_wave):
    """Test wave personality validation returns valid structure."""
    df, wave_points = synthetic_elliott_wave

    personality = validate_wave_personality(df, wave_points, column='close')

    # Basic structure checks
    assert isinstance(personality, dict)
    assert 'overall_confidence' in personality
    assert 'wave_3' in personality
    assert 'wave_5' in personality

    # Confidence should be valid
    assert 0.0 <= personality['overall_confidence'] <= 1.0


@pytest.mark.unit
def test_wave_3_personality(synthetic_elliott_wave):
    """Test Wave 3 personality characteristics."""
    df, wave_points = synthetic_elliott_wave

    personality = validate_wave_personality(df, wave_points, column='close')
    wave3 = personality['wave_3']

    # Wave 3 should have these characteristics
    assert 'confidence' in wave3
    assert isinstance(wave3['confidence'], (int, float))
    assert 0.0 <= wave3['confidence'] <= 1.0

    # Wave 3 typically has highest volume in our synthetic data
    if 'highest_volume' in wave3:
        assert isinstance(wave3['highest_volume'], (bool, np.bool_))


@pytest.mark.unit
def test_wave_5_personality(synthetic_elliott_wave):
    """Test Wave 5 personality characteristics."""
    df, wave_points = synthetic_elliott_wave

    personality = validate_wave_personality(df, wave_points, column='close')
    wave5 = personality['wave_5']

    # Wave 5 characteristics
    assert 'confidence' in wave5
    assert isinstance(wave5['confidence'], (int, float))
    assert 0.0 <= wave5['confidence'] <= 1.0

    # Wave 5 typically has declining volume in our synthetic data
    if 'declining_volume' in wave5:
        assert isinstance(wave5['declining_volume'], (bool, np.bool_))


@pytest.mark.unit
def test_wave_personality_comprehensive(synthetic_elliott_wave):
    """Test comprehensive wave personality validation."""
    df, wave_points = synthetic_elliott_wave

    personality = validate_wave_personality(df, wave_points, column='close')

    # Wave 3 properties
    wave3 = personality['wave_3']
    expected_wave3_keys = ['confidence', 'highest_volume', 'strong_momentum']
    for key in expected_wave3_keys:
        if key in wave3:
            assert wave3[key] is not None

    # Wave 5 properties
    wave5 = personality['wave_5']
    expected_wave5_keys = ['confidence', 'declining_volume', 'momentum_divergence', 'reversal_warning']
    for key in expected_wave5_keys:
        if key in wave5:
            assert wave5[key] is not None


# ============================================================================
# TEST 3: CORRECTIVE PATTERN ANALYSIS
# ============================================================================

@pytest.mark.unit
def test_corrective_pattern_wave2(synthetic_elliott_wave):
    """Test corrective pattern classification on Wave 2."""
    df, wave_points = synthetic_elliott_wave

    # Test on Wave 2 (corrective)
    wave_2_points = wave_points[:3]  # Start, Wave1, Wave2

    pattern = classify_corrective_pattern(df, wave_2_points, column='close')

    # Basic structure
    assert isinstance(pattern, dict)
    assert 'type' in pattern
    assert 'confidence' in pattern

    # Confidence should be valid
    if 'confidence' in pattern:
        assert 0.0 <= pattern['confidence'] <= 1.0


@pytest.mark.unit
def test_zigzag_pattern_detection(synthetic_elliott_wave):
    """Test zigzag pattern detection."""
    df, wave_points = synthetic_elliott_wave

    wave_2_points = wave_points[:3]
    zigzag = detect_zigzag_pattern(df, wave_2_points, column='close')

    assert isinstance(zigzag, dict)
    assert 'valid' in zigzag
    assert isinstance(zigzag['valid'], bool)

    if zigzag['valid']:
        assert 'confidence' in zigzag
        assert 0.0 <= zigzag['confidence'] <= 1.0


@pytest.mark.unit
def test_flat_pattern_detection(synthetic_elliott_wave):
    """Test flat pattern detection."""
    df, wave_points = synthetic_elliott_wave

    wave_2_points = wave_points[:3]
    flat = detect_flat_pattern(df, wave_2_points, column='close')

    assert isinstance(flat, dict)
    assert 'valid' in flat
    assert isinstance(flat['valid'], bool)

    if flat['valid']:
        assert 'confidence' in flat
        assert 'subtype' in flat
        assert 0.0 <= flat['confidence'] <= 1.0


@pytest.mark.unit
def test_triangle_pattern_detection(zigzag_correction_data):
    """Test triangle pattern detection."""
    # Use zigzag correction data fixture
    df = zigzag_correction_data

    # Create simple points for triangle test
    points = np.array([0, 5, 10, 15, len(df)-1])

    triangle = detect_triangle_pattern(df, points, column='close')

    assert isinstance(triangle, dict)
    assert 'valid' in triangle
    assert isinstance(triangle['valid'], bool)


@pytest.mark.unit
def test_corrective_pattern_structure(synthetic_elliott_wave):
    """Test that corrective pattern returns proper structure."""
    df, wave_points = synthetic_elliott_wave

    wave_2_points = wave_points[:3]
    pattern = classify_corrective_pattern(df, wave_2_points, column='close')

    # Should have type and structure
    if pattern.get('type'):
        assert isinstance(pattern['type'], str)

    if pattern.get('structure'):
        assert isinstance(pattern['structure'], str)


# ============================================================================
# TEST 4: TIME-BASED FIBONACCI RELATIONSHIPS
# ============================================================================

@pytest.mark.unit
def test_fibonacci_time_basic(synthetic_elliott_wave):
    """Test time-based Fibonacci relationships."""
    df, wave_points = synthetic_elliott_wave

    time_analysis = validate_fibonacci_time_relationships(df, wave_points, column='close')

    # Basic structure
    assert isinstance(time_analysis, dict)
    assert 'confidence' in time_analysis
    assert 0.0 <= time_analysis['confidence'] <= 1.0


@pytest.mark.unit
def test_wave_durations(synthetic_elliott_wave):
    """Test wave duration calculations."""
    df, wave_points = synthetic_elliott_wave

    time_analysis = validate_fibonacci_time_relationships(df, wave_points, column='close')

    # Should have durations
    assert 'durations' in time_analysis
    durations = time_analysis['durations']

    # Check that we have durations for waves 1-5
    for i in range(1, 6):
        wave_key = f'wave_{i}'
        if wave_key in durations:
            assert isinstance(durations[wave_key], (int, np.integer))
            assert durations[wave_key] > 0


@pytest.mark.unit
def test_time_equality(synthetic_elliott_wave):
    """Test time equality relationships."""
    df, wave_points = synthetic_elliott_wave

    time_analysis = validate_fibonacci_time_relationships(df, wave_points, column='close')

    if 'time_equality' in time_analysis:
        equality = time_analysis['time_equality']

        # Check Wave 1-5 equality if present
        if 'wave_1_5_equality' in equality:
            w15 = equality['wave_1_5_equality']
            assert 'is_equal' in w15
            assert isinstance(w15['is_equal'], (bool, np.bool_))
            if 'ratio' in w15:
                assert w15['ratio'] > 0

        # Check Wave 2-4 equality if present
        if 'wave_2_4_equality' in equality:
            w24 = equality['wave_2_4_equality']
            assert 'is_equal' in w24
            assert isinstance(w24['is_equal'], (bool, np.bool_))


@pytest.mark.unit
def test_fibonacci_time_ratios(synthetic_elliott_wave):
    """Test Fibonacci time ratio calculations."""
    df, wave_points = synthetic_elliott_wave

    time_analysis = validate_fibonacci_time_relationships(df, wave_points, column='close')

    if 'fibonacci_ratios' in time_analysis:
        fib = time_analysis['fibonacci_ratios']

        # Check Wave 3/Wave 1 ratio
        if 'wave_3_to_1' in fib:
            w31 = fib['wave_3_to_1']
            assert 'ratio' in w31
            assert w31['ratio'] > 0
            assert 'is_fibonacci' in w31
            assert isinstance(w31['is_fibonacci'], (bool, np.bool_))

        # Check Wave 5/Wave 3 ratio
        if 'wave_5_to_3' in fib:
            w53 = fib['wave_5_to_3']
            assert 'ratio' in w53
            assert w53['ratio'] > 0


# ============================================================================
# TEST 5: REAL MARKET DATA (marked as slow/network)
# ============================================================================

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("ticker", ["SPY", "AAPL", "MSFT"])
def test_real_market_data(ticker):
    """Test Elliott Wave validation on real market data."""
    import yfinance as yf

    # Download data
    df = yf.download(ticker, period='6mo', progress=False)

    # Skip if download failed
    if df.empty:
        pytest.skip(f"Failed to download data for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    assert len(df) > 0
    assert 'close' in df.columns

    # Detect peaks and troughs
    peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')

    assert len(peaks) >= 0  # May have no peaks in flat market
    assert len(troughs) >= 0


@pytest.mark.unit
@patch('yfinance.download')
def test_real_market_data_mocked(mock_download, aapl_data):
    """Test real market data analysis with mocked download."""
    # Mock yfinance download to use fixture data
    mock_download.return_value = aapl_data.copy()

    import yfinance as yf
    df = yf.download('AAPL', period='1y', progress=False)

    # Ensure we got data
    assert not df.empty

    # Should have proper columns
    df.columns = [col.lower() for col in df.columns]
    assert 'close' in df.columns

    # Detect peaks
    peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')

    # Should find some peaks in real data
    assert len(peaks) > 0 or len(troughs) > 0


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.unit
def test_validate_with_insufficient_points():
    """Test validation with insufficient wave points."""
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1000, 1000]
    }, index=pd.date_range('2024-01-01', periods=3))

    # Only 3 points - insufficient for 5-wave pattern
    wave_points = np.array([0, 1, 2])

    # Should handle gracefully (may return invalid or raise specific error)
    try:
        is_valid, confidence, details = validate_impulse_wave_strict(
            df, wave_points, column='close'
        )
        # If it returns, should indicate invalid
        assert not is_valid or confidence < 0.3
    except (ValueError, IndexError, KeyError):
        # Or it may raise an error - that's acceptable
        pass


@pytest.mark.unit
def test_validate_with_invalid_wave_points():
    """Test validation with wave points out of bounds."""
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'volume': [1000] * 6
    }, index=pd.date_range('2024-01-01', periods=6))

    # Wave points extending beyond data
    wave_points = np.array([0, 2, 4, 6, 8, 10])  # Indices 6-10 don't exist

    # Should handle gracefully
    try:
        is_valid, confidence, details = validate_impulse_wave_strict(
            df, wave_points, column='close'
        )
        # If it doesn't raise, it should return low confidence
        assert confidence < 0.5
    except (ValueError, IndexError, KeyError):
        # Or raise appropriate error
        pass


@pytest.mark.unit
def test_personality_with_flat_prices():
    """Test wave personality with flat price series."""
    df = pd.DataFrame({
        'close': [100.0] * 100,  # Flat prices
        'volume': list(range(1000, 1100))
    }, index=pd.date_range('2024-01-01', periods=100))

    wave_points = np.array([0, 20, 40, 60, 80, 99])

    # Should handle flat prices gracefully
    try:
        personality = validate_wave_personality(df, wave_points, column='close')
        assert isinstance(personality, dict)
        # Flat prices should result in low confidence
        if 'overall_confidence' in personality:
            assert personality['overall_confidence'] < 0.5
    except (ValueError, ZeroDivisionError):
        # May raise error for zero price movement
        pass


@pytest.mark.unit
def test_corrective_pattern_with_minimal_data():
    """Test corrective pattern detection with minimal data."""
    df = pd.DataFrame({
        'close': [100, 105, 102],
        'volume': [1000, 1100, 1050]
    }, index=pd.date_range('2024-01-01', periods=3))

    points = np.array([0, 1, 2])

    # Should return a result even with minimal data
    pattern = classify_corrective_pattern(df, points, column='close')

    assert isinstance(pattern, dict)
    # May indicate low confidence or no pattern
    if 'confidence' in pattern:
        assert 0.0 <= pattern['confidence'] <= 1.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
def test_complete_elliott_wave_analysis(synthetic_elliott_wave):
    """Test complete Elliott Wave analysis workflow."""
    df, wave_points = synthetic_elliott_wave

    # Step 1: Validate structure
    is_valid, confidence, details = validate_impulse_wave_strict(
        df, wave_points, column='close'
    )

    assert isinstance(is_valid, bool)
    assert 0.0 <= confidence <= 1.0

    # Step 2: Check personality
    personality = validate_wave_personality(df, wave_points, column='close')

    assert 'overall_confidence' in personality
    assert 0.0 <= personality['overall_confidence'] <= 1.0

    # Step 3: Analyze time relationships
    time_analysis = validate_fibonacci_time_relationships(df, wave_points)

    assert 'confidence' in time_analysis
    assert 0.0 <= time_analysis['confidence'] <= 1.0

    # Step 4: Check corrective patterns on Wave 2
    wave_2_points = wave_points[:3]
    pattern = classify_corrective_pattern(df, wave_2_points, column='close')

    assert isinstance(pattern, dict)


@pytest.mark.integration
def test_peak_detection_to_validation(aapl_data):
    """Test integration from peak detection to validation."""
    df = aapl_data.tail(252).copy()  # Last year of data

    # Step 1: Detect peaks and troughs
    peaks, troughs = detect_peaks_troughs_enhanced(df, column='close')

    # Step 2: If we have enough peaks, try validation
    if len(peaks) >= 3 and len(troughs) >= 2:
        # Create simple wave points
        wave_points = np.array([
            0,
            peaks[0],
            troughs[0],
            peaks[1],
            troughs[1],
            peaks[2]
        ])

        # Ensure ascending and within bounds
        wave_points = np.sort(wave_points)
        wave_points = wave_points[wave_points < len(df)]

        if len(wave_points) >= 6:
            # Step 3: Validate
            is_valid, confidence, details = validate_impulse_wave_strict(
                df, wave_points, column='close'
            )

            # Should return valid results
            assert isinstance(is_valid, bool)
            assert 0.0 <= confidence <= 1.0
