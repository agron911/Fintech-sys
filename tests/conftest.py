"""
Shared pytest fixtures for Elliott Wave Investment System tests.

This module provides reusable test fixtures for:
- Loading real stock data from data/raw/ files
- Mocking external dependencies (Yahoo Finance API)
- Creating synthetic test data
- Test configuration
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """Load test configuration matching production config structure."""
    return {
        "data_dir": "data",
        "stk2_dir": "data/raw",
        "adjustments_dir": "data/lists/adjustments",
        "processed_dir": "data/processed",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "international_file": "data/lists/international.txt",
        "list_file": "data/lists/adjustments/list.xlsx",
        "otclist_file": "data/lists/adjustments/otclist.xlsx",
        "databases_dir": "data/raw/databases",
        "fibonacci_tolerance": 0.05
    }


@pytest.fixture
def project_root_path():
    """Return project root directory path."""
    return project_root


# ============================================================================
# REAL STOCK DATA FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def aapl_data():
    """Load AAPL stock data from data/raw/AAPL.txt."""
    data_path = project_root / "data" / "raw" / "AAPL.txt"
    if not data_path.exists():
        pytest.skip(f"AAPL data file not found: {data_path}")

    df = pd.read_csv(data_path, sep='\t', index_col=0, parse_dates=True)
    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    # Remove duplicate 'date' column if exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df


@pytest.fixture(scope="session")
def msft_data():
    """Load MSFT stock data from data/raw/MSFT.txt."""
    data_path = project_root / "data" / "raw" / "MSFT.txt"
    if not data_path.exists():
        pytest.skip(f"MSFT data file not found: {data_path}")

    df = pd.read_csv(data_path, sep='\t', index_col=0, parse_dates=True)
    df.columns = [col.lower() for col in df.columns]
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df


@pytest.fixture(scope="session")
def spy_data():
    """Load SPY ETF data from data/raw/SPY.txt."""
    data_path = project_root / "data" / "raw" / "SPY.txt"
    if not data_path.exists():
        pytest.skip(f"SPY data file not found: {data_path}")

    df = pd.read_csv(data_path, sep='\t', index_col=0, parse_dates=True)
    df.columns = [col.lower() for col in df.columns]
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df


@pytest.fixture(scope="session")
def taiwan_stock_1102_data():
    """Load Taiwan stock 1102 data from data/raw/1102.txt."""
    data_path = project_root / "data" / "raw" / "1102.txt"
    if not data_path.exists():
        pytest.skip(f"1102 data file not found: {data_path}")

    df = pd.read_csv(data_path, sep='\t', index_col=0, parse_dates=True)
    df.columns = [col.lower() for col in df.columns]
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df


@pytest.fixture
def sample_stock_data(aapl_data):
    """
    Return a subset of AAPL data for quick testing.
    Returns 252 trading days (~1 year) of data.
    """
    return aapl_data.tail(252).copy()


# ============================================================================
# SYNTHETIC TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def simple_uptrend_data():
    """
    Create simple uptrend OHLCV data for testing.
    10 bars with increasing prices and volume.
    """
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'high': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'close': [10, 12, 11, 15, 13, 17, 15, 18, 16, 20],
        'volume': [100, 120, 110, 130, 115, 140, 120, 150, 125, 160]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def simple_downtrend_data():
    """
    Create simple downtrend OHLCV data for testing.
    10 bars with decreasing prices.
    """
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'open': [20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
        'high': [21, 20, 19, 18, 17, 16, 15, 14, 13, 12],
        'low': [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
        'close': [20, 18, 19, 15, 17, 13, 15, 12, 14, 10],
        'volume': [160, 150, 140, 130, 120, 110, 100, 90, 80, 70]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def elliott_wave_impulse_data():
    """
    Create synthetic Elliott Wave impulse pattern (5-wave structure).
    - Wave 1: Up
    - Wave 2: Retracement (38.2% Fibonacci)
    - Wave 3: Strong up (1.618x Wave 1)
    - Wave 4: Retracement (23.6% Fibonacci)
    - Wave 5: Final up
    """
    dates = pd.date_range('2023-01-01', periods=100)

    # Create 5-wave impulse structure
    wave1_bars = 20  # Wave 1: 100 -> 120 (+20)
    wave2_bars = 15  # Wave 2: 120 -> 112 (-8, 40% retrace)
    wave3_bars = 30  # Wave 3: 112 -> 144 (+32, 1.6x Wave 1)
    wave4_bars = 20  # Wave 4: 144 -> 137 (-7, 22% retrace)
    wave5_bars = 15  # Wave 5: 137 -> 157 (+20)

    close_prices = []
    volumes = []

    # Wave 1: Upward impulse
    close_prices.extend(np.linspace(100, 120, wave1_bars))
    volumes.extend(np.linspace(1000, 1200, wave1_bars))

    # Wave 2: Corrective down
    close_prices.extend(np.linspace(120, 112, wave2_bars))
    volumes.extend(np.linspace(1200, 1000, wave2_bars))

    # Wave 3: Strongest impulse (highest volume)
    close_prices.extend(np.linspace(112, 144, wave3_bars))
    volumes.extend(np.linspace(1000, 1500, wave3_bars))  # Wave 3 has highest volume

    # Wave 4: Corrective down (lower volume)
    close_prices.extend(np.linspace(144, 137, wave4_bars))
    volumes.extend(np.linspace(1500, 900, wave4_bars))

    # Wave 5: Final impulse (declining volume)
    close_prices.extend(np.linspace(137, 157, wave5_bars))
    volumes.extend(np.linspace(900, 800, wave5_bars))  # Wave 5 lower volume than Wave 3

    # Generate OHLCV from close prices
    data = {
        'close': close_prices,
        'open': [c * 0.99 for c in close_prices],
        'high': [c * 1.01 for c in close_prices],
        'low': [c * 0.98 for c in close_prices],
        'volume': volumes
    }

    return pd.DataFrame(data, index=dates[:len(close_prices)])


@pytest.fixture
def zigzag_correction_data():
    """
    Create zigzag correction pattern (A-B-C structure).
    A: Down, B: Up (50-60% retracement), C: Down
    """
    dates = pd.date_range('2023-01-01', periods=20)
    data = {
        'close': [100, 95, 90, 85, 80,  # A down
                  80, 85, 88, 90, 92,    # B up (60% retrace)
                  92, 88, 85, 82, 78, 75,  # C down
                  75, 74, 73, 72],
        'volume': [1000] * 20
    }
    df = pd.DataFrame(data, index=dates)
    df['open'] = df['close'] * 0.99
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.98
    return df


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_yahoo_finance():
    """
    Mock YahooFinanceCrawler with realistic API response structure.
    Returns a mock that simulates successful data fetching.
    """
    mock_crawler = Mock()

    # Create realistic OHLCV data for mock responses
    def mock_fetch_data(symbol, start_date, end_date):
        """Simulate fetching data from Yahoo Finance."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter business days only
        dates = dates[dates.dayofweek < 5]

        # Generate synthetic but realistic price data
        base_price = 100
        num_bars = len(dates)

        # Random walk with upward drift
        returns = np.random.normal(0.0005, 0.02, num_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        data = {
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, num_bars)),
            'high': prices * (1 + np.random.uniform(0.00, 0.02, num_bars)),
            'low': prices * (1 + np.random.uniform(-0.02, 0.00, num_bars)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, num_bars)
        }

        df = pd.DataFrame(data, index=dates)
        df.columns = [col.capitalize() for col in df.columns]
        return df

    mock_crawler.fetch_data.side_effect = mock_fetch_data
    mock_crawler.save_data.return_value = None  # Mock successful save

    return mock_crawler


@pytest.fixture
def mock_logger():
    """Mock logger to suppress output during tests."""
    mock = MagicMock()
    return mock


# ============================================================================
# BACKTEST FIXTURES
# ============================================================================

@pytest.fixture
def backtest_config():
    """Standard backtesting configuration."""
    return {
        "initial_capital": 100000,
        "risk_per_trade": 0.02,  # 2% risk per trade
        "max_position_size": 0.15,  # 15% of portfolio
        "max_concurrent_positions": 5,
        "daily_drawdown_limit": 0.05,  # 5%
        "weekly_drawdown_limit": 0.08,  # 8%
        "monthly_drawdown_limit": 0.12,  # 12%
        "commission": 0.001,  # 0.1%
        "slippage": 0.0005,  # 0.05%
    }


@pytest.fixture
def mock_wave_pattern():
    """
    Mock Elliott Wave pattern detection result.
    Returns structure matching actual wave detection output.
    """
    return {
        'pattern_type': 'IMPULSE',
        'points': np.array([0, 20, 35, 65, 85, 100]),  # Wave turning points
        'confidence': 0.75,
        'wave_labels': ['start', '1', '2', '3', '4', '5'],
        'timeframe': 'daily',
        'trend': 'UP',
        'current_wave': 5,
        'wave_completion': 0.85,
        'fibonacci_levels': {
            'wave2_retrace': 0.382,
            'wave4_retrace': 0.236,
            'wave3_extension': 1.618
        },
        'validation_scores': {
            'direction': 0.9,
            'wave_2': 0.8,
            'wave_3': 0.9,
            'wave_4_overlap': 1.0,
            'fibonacci': 0.7,
            'volume': 0.6
        }
    }


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def wave_points_5wave():
    """Standard 5-wave impulse indices."""
    return np.array([0, 20, 35, 65, 85, 100])


@pytest.fixture
def wave_points_3wave():
    """Standard 3-wave corrective indices."""
    return np.array([0, 30, 60, 100])


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup after test
    np.random.seed(None)


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

# Common Fibonacci ratios for parametrized tests
FIBONACCI_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]

# Common stock symbols for parametrized tests
TEST_SYMBOLS = ["AAPL", "MSFT", "SPY"]
TAIWAN_SYMBOLS = ["1102", "1103", "1216"]


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (multiple components)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (backtests, benchmarks)"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )
    config.addinivalue_line(
        "markers", "real_data: Tests using real market data"
    )
