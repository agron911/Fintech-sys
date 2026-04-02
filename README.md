# Stock Price Tracing System

This project is a stock price tracing and analysis system that fetches, processes, and analyzes stock data from Yahoo Finance and other sources. It is designed to be cross-platform and easy to set up on any laptop.

## Features
- Download and cache stock price data
- Analyze long-tail stocks
- Modular, extensible crawler system
- GUI for user interaction (wxPython)
- Configurable data directories and date ranges

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Edit `config.json`** (optional):
   - Set your preferred data directories and date ranges.

## Usage

- Run the main GUI:
  ```
  python main.py
  ```
- **Run the modular crawler:**
  ```
  python scripts/run_crawler.py
  ```
  This will fetch stock data for all configured lists and save them to the appropriate directories.
- Data and results will be stored in the directories specified in `config.json`.

## Crawler System

The crawler is now modular and extensible:
- **BaseCrawler**: Abstract class in `src/crawler/base_crawler.py` defines the interface and common logic.
- **YahooFinanceCrawler**: Implementation in `src/crawler/yahoo_finance.py` for Yahoo Finance, with retry logic and robust error handling.
- **run_crawler.py**: Script in `scripts/` to orchestrate crawling for all stock lists.
- Logging is configured via `src/utils/app_logging.py`.
- The crawler uses the `retrying` package for automatic retries on network errors.

## Configuration

Settings are stored in `config.json`:
```json
{
  "data_dir": "databases",
  "stk2_dir": "stk2",
  "adjustments_dir": "adjustments",
  "start_date": "2000-01-01",
  "end_date": "2025-12-24",
  "international_file": "international.txt",
  "list_file": "adjustments/list.xlsx",
  "otclist_file": "adjustments/otclist.xlsx"
}
```

## Notes
- Make sure you have [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/) in your project directory for Selenium.
- For Excel file support, `openpyxl` is included in requirements.
- The crawler requires the `retrying` package (included in requirements).

## License
MIT 

## Project Structure

```
investment_system/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── config/                  # Configuration files
│   └── config.json         # Main configuration file
│
├── data/                    # Data storage
│   ├── lists/              # Stock lists and symbols
│   ├── processed/          # Processed stock data
│   └── raw/                # Raw downloaded data
│
├── scripts/                 # Executable scripts
│   ├── run_crawler.py      # Data crawler script
│   ├── run_backtest.py     # Backtesting script
│   └── run_gui.py          # GUI launcher script
│
├── src/                     # Source code
│   ├── analysis/           # Analysis modules
│   │   ├── elliott_wave.py # Elliott Wave analysis
│   │   ├── indicators.py   # Technical indicators
│   │   ├── core/           # Core analysis components
│   │   │   ├── alternation.py
│   │   │   ├── corrective.py
│   │   │   ├── fib_utils.py
│   │   │   ├── impulse.py
│   │   │   ├── models.py
│   │   │   ├── peaks.py
│   │   │   ├── position.py
│   │   │   ├── trendlines.py
│   │   │   ├── validation.py
│   │   │   └── volume.py
│   │   └── plotters/       # Visualization components
│   │       └── impulse.py
│   │
│   ├── backtest/           # Backtesting system
│   │   ├── backtester.py   # Backtesting engine
│   │   └── strategy.py     # Trading strategies
│   │
│   ├── crawler/            # Data collection
│   │   ├── base_crawler.py # Base crawler class
│   │   └── yahoo_finance.py # Yahoo Finance implementation
│   │
│   └── utils/              # Utility functions
│       ├── common_utils.py # Common utilities
│       ├── config.py       # Configuration handling
│       ├── data_utils.py   # Data processing utilities
│       └── logging.py      # Logging configuration
│
├── gui/                    # Graphical user interface
│   ├── main.py            # Main GUI application
│   ├── frame.py           # GUI frame components
│   ├── handlers.py        # Event handlers
│   ├── utils.py           # GUI utilities
│   └── constants.py       # GUI constants
│
└── tests/                 # Test suite
    └── test_candlestick_overlay.py
```

### Directory Descriptions

- **config/**: Contains all configuration files including the main `config.json` for system settings
- **data/**: Stores all data files including raw downloaded data and processed results
- **scripts/**: Contains executable scripts for running different components of the system
- **src/**: Core source code organized into functional modules:
  - **analysis/**: Technical analysis tools and indicators
  - **backtest/**: Backtesting engine and trading strategies
  - **crawler/**: Data collection modules for different sources
  - **utils/**: Shared utility functions and helpers
- **gui/**: Complete GUI implementation using wxPython
- **tests/**: Unit and integration tests for system components 