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
   python main.py gui
  ```
- **Run the modular crawler:**
  ```
  python scripts/run_crawler.py
  ```
  This will fetch stock data for all configured lists and save them to the appropriate directories.
- Data and results will be stored in the directories specified in `config.json`.

## Data Updates

Stock snapshots are published to the **`data-latest` GitHub Release** every
Friday and Saturday at 00:00 UTC, after changes to the fetch workflow or crawler,
and through a manual workflow run. The `data-updates` branch is historical.

### Edge PCs

Run these commands from the repository folder to update the app and its data:

```bash
git switch main
git pull --ff-only origin main
python main.py update-data --force
```

For subsequent data updates, run just:

```bash
python main.py update-data --force
```

The download extracts the current release into this repo's `data/raw/`.
`--force` skips the confirmation prompt. Git pull updates application code;
the data command downloads the release asset without using Git LFS.

### Update Strategy

CI restores the previous snapshot and refreshes stocks whose price data is old.
Before publishing, it checks for at least 10 fresh files and requires 90% of
active files to contain prices from the last 7 days. Historical `archive_temp/`
copies are excluded from the archive and stock count.

To check release metadata or local freshness:

```bash
python main.py update-data --info
python scripts/verify_release_data.py
```

The metadata's update time refers to the asset's replacement time. Individual
stocks can still have older prices if their fetch fails or trading is suspended.

### HTTPS Certificate Errors on an Edge PC

If Python reports `CERTIFICATE_VERIFY_FAILED` or `unable to get local issuer
certificate`, update its certificate support and retry:

```bash
python -m pip install --upgrade certifi truststore
python main.py update-data --force
```

The downloader uses the operating system's certificate store when `truststore`
is installed, and adds public CA certificates from `certifi`. It also honors
`SSL_CERT_FILE` and `SSL_CERT_DIR`.

For a company network with its own HTTPS certificate issuer, use its approved
CA bundle or install that CA in the operating system's certificate store:

```bash
python main.py update-data --force --ca-bundle /path/to/company-ca.pem
```

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
