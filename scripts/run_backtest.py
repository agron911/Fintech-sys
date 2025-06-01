import pandas as pd
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.backtest.backtester import Backtester

def main():
    setup_logging()
    config = load_config()
    
    # Load stock lists
    international = pd.read_csv(config["international_file"])["code"]
    listed = pd.read_excel(config["list_file"])["code"]
    otc = pd.read_excel(config["otclist_file"])["code"]
    symbols = list(international) + list(listed) + list(otc) + ['TWII']
    
    backtester = Backtester(config)
    backtester.run(symbols)
    backtester.summarize()

if __name__ == "__main__":
    main() 