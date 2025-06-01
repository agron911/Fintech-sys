import warnings
warnings.filterwarnings('ignore')

from datetime import date
import wx
import os
import pandas as pd
import threading
from pathlib import Path
from src.crawler.yahoo_finance import YahooFinanceCrawler
from src.backtest.backtester import Backtester
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.analysis.elliott_wave import detect_peaks_troughs, refined_elliott_wave_suggestion, plot_peaks_troughs, detect_elliott_wave_complete, plot_elliott_wave_analysis
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import matplotlib.patches as patches
from gui.frame import MyFrame

class App(wx.App):
    def OnInit(self):
        frame = MyFrame()
        frame.Show()
        return True

if __name__ == "__main__":
    # Ensure sys.path includes project root
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    app = App()
    app.MainLoop()

