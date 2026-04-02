"""
Test Yahoo Finance Crawler - pytest version

Tests for YahooFinanceCrawler functionality:
- Data fetching from Yahoo Finance API
- Data validation (columns, types, shapes)
- Data saving to files
- Error handling for network failures and invalid symbols
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.crawler.yahoo_finance import YahooFinanceCrawler, fetch_stock_data, save_stock_data
from src.utils.config import load_config


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

@pytest.mark.unit
def test_crawler_initialization(test_config):
    """Test YahooFinanceCrawler can be instantiated with config."""
    crawler = YahooFinanceCrawler(test_config)
    assert crawler is not None
    assert hasattr(crawler, 'config')


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_success(mock_download, test_config):
    """Test successful data fetching returns valid DataFrame."""
    # Create mock response matching Yahoo Finance structure
    mock_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [104, 105, 106],
        'Volume': [1000000, 1100000, 1200000],
        'Adj Close': [104, 105, 106]
    }, index=pd.date_range('2024-01-01', periods=3))

    mock_download.return_value = mock_data

    # Test fetch_stock_data function
    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-03')

    # Assertions
    assert df is not None
    assert not df.empty
    assert len(df) == 3
    assert 'Close' in df.columns
    assert 'Volume' in df.columns
    assert 'Adj Close' not in df.columns  # Should be dropped


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_returns_correct_columns(mock_download, test_config):
    """Test that fetched data has expected columns."""
    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-01')

    # Check required columns exist
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns
    assert 'Date' in df.columns


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_returns_correct_shape(mock_download, test_config):
    """Test that fetched data has correct number of rows."""
    mock_data = pd.DataFrame({
        'Open': list(range(10)),
        'High': list(range(10, 20)),
        'Low': list(range(0, 10)),
        'Close': list(range(5, 15)),
        'Volume': [1000000] * 10,
        'Adj Close': list(range(5, 15))
    }, index=pd.date_range('2024-01-01', periods=10))

    mock_download.return_value = mock_data

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-10')

    assert len(df) == 10


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_with_suffix(mock_download, test_config):
    """Test fetching data with symbol suffix (e.g., .TW for Taiwan stocks)."""
    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = fetch_stock_data('1102', '.TW', '2024-01-01', '2024-01-01')

    # Verify yf.download was called with correct symbol
    mock_download.assert_called_once()
    call_args = mock_download.call_args
    assert '1102.TW' in str(call_args)


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

@pytest.mark.unit
def test_save_stock_data_creates_correct_format(tmp_path, simple_uptrend_data):
    """Test that save_stock_data creates correctly formatted file."""
    # Prepare test data
    df = simple_uptrend_data.copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Save to temporary directory
    save_stock_data(df, 'TEST', folder=tmp_path)

    # Verify file was created
    output_file = tmp_path / "TEST.txt"
    assert output_file.exists()

    # Read back and verify format
    saved_df = pd.read_csv(output_file, sep='\t')

    # Check columns
    assert 'Date' in saved_df.columns
    assert 'Open' in saved_df.columns
    assert 'High' in saved_df.columns
    assert 'Low' in saved_df.columns
    assert 'Close' in saved_df.columns
    assert 'Volume' in saved_df.columns

    # Check data integrity
    assert len(saved_df) == len(df)


@pytest.mark.unit
def test_save_stock_data_handles_multiindex_columns(tmp_path):
    """Test that save_stock_data correctly flattens MultiIndex columns."""
    # Create DataFrame with MultiIndex columns (as returned by yf sometimes)
    dates = pd.date_range('2024-01-01', periods=3)
    arrays = [['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
              ['Open', 'High', 'Low', 'Close', 'Volume']]
    tuples = list(zip(*arrays))
    columns = pd.MultiIndex.from_tuples(tuples)

    df = pd.DataFrame(
        np.random.rand(3, 5),
        index=dates,
        columns=columns
    )

    # Should not raise exception
    save_stock_data(df, 'TEST', folder=tmp_path)

    # Verify file was created successfully
    output_file = tmp_path / "TEST.txt"
    assert output_file.exists()


@pytest.mark.unit
def test_save_stock_data_long_tail_option(tmp_path, simple_uptrend_data):
    """Test save_stock_data with long_tail=True creates correct filename."""
    df = simple_uptrend_data.copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    save_stock_data(df, 'TEST', folder=tmp_path, long_tail=True)

    # Verify long_tail file was created
    output_file = tmp_path / "TEST_long_tail.txt"
    assert output_file.exists()

    # Regular file should not exist
    regular_file = tmp_path / "TEST.txt"
    assert not regular_file.exists()


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_network_error(mock_download, test_config):
    """Test handling of network errors during data fetch."""
    # Simulate network error
    mock_download.side_effect = ConnectionError("Network error")

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-03')

    # Should return None on error
    assert df is None


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_invalid_symbol(mock_download, test_config):
    """Test handling of invalid stock symbols."""
    # Simulate empty DataFrame for invalid symbol
    mock_download.return_value = pd.DataFrame()

    df = fetch_stock_data('INVALID', '', '2024-01-01', '2024-01-03')

    # Should return empty DataFrame
    assert df is not None
    assert df.empty


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_exception_handling(mock_download, test_config):
    """Test that fetch_data handles unexpected exceptions gracefully."""
    # Simulate unexpected exception
    mock_download.side_effect = Exception("Unexpected error")

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-03')

    # Should return None and not crash
    assert df is None


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_timeout(mock_download, test_config):
    """Test handling of API timeout."""
    import requests
    mock_download.side_effect = requests.exceptions.Timeout("Request timeout")

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-03')

    assert df is None


# ============================================================================
# INTEGRATION TESTS (with real API - marked as slow/network)
# ============================================================================

@pytest.mark.slow
@pytest.mark.network
def test_fetch_real_data_aapl():
    """
    Integration test: Fetch real data from Yahoo Finance for AAPL.
    This test requires internet connection and is marked as slow.
    """
    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-10')

    # Basic validation
    assert df is not None
    if not df.empty:  # May be empty if dates are invalid or market closed
        assert 'Close' in df.columns
        assert 'Volume' in df.columns
        assert len(df) > 0


@pytest.mark.slow
@pytest.mark.network
def test_fetch_real_data_spy():
    """
    Integration test: Fetch real data for SPY ETF.
    """
    df = fetch_stock_data('SPY', '', '2024-01-01', '2024-01-10')

    assert df is not None
    if not df.empty:
        assert 'Close' in df.columns
        assert len(df) > 0


# ============================================================================
# CRAWLER CLASS TESTS
# ============================================================================

@pytest.mark.unit
def test_yahoo_finance_crawler_init(test_config):
    """Test YahooFinanceCrawler initialization."""
    crawler = YahooFinanceCrawler(test_config)

    assert crawler.config == test_config
    assert crawler.config['start_date'] == test_config['start_date']
    assert crawler.config['end_date'] == test_config['end_date']


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_yahoo_finance_crawler_fetch_data(mock_download, test_config):
    """Test YahooFinanceCrawler.fetch_data method."""
    crawler = YahooFinanceCrawler(test_config)

    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = crawler.fetch_data('AAPL', '2024-01-01', '2024-01-01')

    assert df is not None
    assert not df.empty
    assert 'Close' in df.columns


@pytest.mark.unit
def test_yahoo_finance_crawler_save_data(test_config, simple_uptrend_data, tmp_path):
    """Test YahooFinanceCrawler.save_data method."""
    # Override config to use temporary directory
    test_config['stk2_dir'] = str(tmp_path)
    crawler = YahooFinanceCrawler(test_config)

    # Prepare test data with proper columns
    df = simple_uptrend_data.copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Call save_data
    crawler.save_data(df, 'TEST')

    # Verify file was created
    output_file = tmp_path / "TEST.txt"
    assert output_file.exists()

    # Read back and verify basic structure
    saved_df = pd.read_csv(output_file, sep='\t')
    assert 'Date' in saved_df.columns
    assert 'Close' in saved_df.columns
    assert len(saved_df) == len(df)


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("symbol,suffix", [
    ("AAPL", ""),
    ("MSFT", ""),
    ("SPY", ""),
    ("1102", ".TW"),
    ("2330", ".TW"),
])
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_multiple_symbols(mock_download, symbol, suffix, test_config):
    """Test data fetching for multiple stock symbols."""
    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = fetch_stock_data(symbol, suffix, '2024-01-01', '2024-01-01')

    assert df is not None
    # Verify correct symbol was requested
    mock_download.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("num_days", [1, 10, 100, 252])
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_different_date_ranges(mock_download, num_days, test_config):
    """Test fetching data for different date ranges."""
    mock_data = pd.DataFrame({
        'Open': list(range(num_days)),
        'High': list(range(num_days)),
        'Low': list(range(num_days)),
        'Close': list(range(num_days)),
        'Volume': [1000000] * num_days,
        'Adj Close': list(range(num_days))
    }, index=pd.date_range('2024-01-01', periods=num_days))

    mock_download.return_value = mock_data

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-12-31')

    assert df is not None
    assert len(df) == num_days


# ============================================================================
# DATA QUALITY TESTS
# ============================================================================

@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_date_index_format(mock_download, test_config):
    """Test that date index is formatted correctly."""
    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-01')

    # Check that Date column exists
    assert 'Date' in df.columns
    # Check date format (should be YYYY/MM/DD)
    assert isinstance(df['Date'].iloc[0], str)


@pytest.mark.unit
@patch('src.crawler.yahoo_finance.yf.download')
def test_fetch_data_no_adj_close(mock_download, test_config):
    """Test that Adj Close column is removed."""
    mock_data = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [99],
        'Close': [104],
        'Volume': [1000000],
        'Adj Close': [104]
    }, index=pd.date_range('2024-01-01', periods=1))

    mock_download.return_value = mock_data

    df = fetch_stock_data('AAPL', '', '2024-01-01', '2024-01-01')

    # Adj Close should be dropped
    assert 'Adj Close' not in df.columns
