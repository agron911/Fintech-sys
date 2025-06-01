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
├── .gitignore
├── config/
│   └── config.json
├── data/
│   ├── lists/
│   │   ├── international.txt
│   │   └── adjustments/
│   │       ├── list.xlsx
│   │       ├── otclist.xlsx
│   │       ├── list.xls
│   │       └── otclist.xls
│   ├── raw/
│   │   ├── 3045.txt
│   │   └── ...
├── scripts/
│   ├── run_crawler.py
│   ├── run_backtest.py
│   └── run_gui.py
├── src/
│   ├── analysis/
│   │   ├── core/
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
│   │   ├── plotters/
│   │   │   └── impulse.py
│   │   ├── elliott_wave.py
│   │   └── indicators.py
│   ├── backtest/
│   │   ├── backtester.py
│   │   └── strategy.py
│   ├── crawler/
│   │   ├── base_crawler.py
│   │   ├── yahoo_finance.py
│   └── utils/
│       ├── common_utils.py
│       ├── config.py
│       ├── data_utils.py
│       └── logging.py
├── gui/
│   ├── main.py
│   ├── frame.py
│   ├── handlers.py
│   ├── utils.py
│   └── constants.py
└── tests/
    └── test_candlestick_overlay.py
```

- `config/`: Configuration files (e.g., `config.json`).
- `data/`: Data files, including stock lists, adjustments, raw/processed data, and the SQLite database.
- `htmls/`: (Empty or for HTML outputs)
- `models/`: (Empty or for ML/statistical models)
- `scripts/`: Entry-point scripts for running the GUI, crawler, and backtester.
- `src/`: Core source code (analysis, backtesting, crawling, utilities).
  - `src/analysis/`: Contains modules for stock analysis, including `elliott_wave.py` for Elliott Wave analysis and `indicators.py` for technical indicators.
    - `src/analysis/core/`: Core analysis modules like position, alternation, volume, etc.
    - `src/analysis/plotters/`: Modules for plotting analysis results.
- `gui/`: GUI application code.
- `tests/`: Test scripts. 