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
- Logging is configured via `src/utils/logging.py`.
- The crawler uses the `retrying` package for automatic retries on network errors.

## Configuration

Settings are stored in `config.json`:
```json
{
  "data_dir": "databases",
  "stk2_dir": "stk2",
  "adjustments_dir": "adjustments",
  "start_date": "2000-01-01",
  "end_date": "2024-12-24",
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
├── README.md
├── requirements.txt
├── config/
│   └── config.json
├── data/
│   ├── lists/
│   │   ├── international.txt
│   │   └── adjustments/
│   │       ├── list.xlsx
│   │       └── otclist.xlsx
│   └── raw/
│       └── databases/
│           └── All_stock_data.db
├── scripts/
│   ├── run_crawler.py
│   ├── run_backtest.py
│   └── run_gui.py
├── src/
│   ├── analysis/
│   │   └── elliott_wave.py
│   ├── backtest/
│   │   ├── backtester.py
│   │   └── strategy.py
│   ├── crawler/
│   │   ├── base_crawler.py
│   │   ├── yahoo_finance.py
│   │   └── data_cleaner.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── data_utils.py
└── gui/
    └── main.py
```

- `config/`: Configuration files (e.g., `config.json`).
- `data/`: Data files, including stock lists and the SQLite database.
- `scripts/`: Entry-point scripts for running the GUI, crawler, and backtester.
- `src/`: Core source code (analysis, backtesting, crawling, utilities).
- `gui/`: GUI application code. 