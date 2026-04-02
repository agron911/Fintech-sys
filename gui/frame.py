import wx
import os
import pandas as pd
import threading
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path
from src.crawler.yahoo_finance import YahooFinanceCrawler
from src.backtest.backtester import Backtester
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.analysis.plotters.elliott import plot_elliott_wave_analysis, plot_elliott_wave_analysis_enhanced
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from gui.constants import WEBLIST, CHART_TYPES
from gui.handlers import (
    handle_storing_path, handle_crawl_data, handle_run_backtest, handle_show_elliott_wave, handle_analyze_current_position, handle_chart_type_change,
    EnableButtonsEvent, EVT_ENABLE_BUTTONS
)
from src.utils.common_utils import resample_ohlc, map_points_to_ohlc
import wx.lib.newevent
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

UpdateOutputEvent, EVT_UPDATE_OUTPUT = wx.lib.newevent.NewEvent()
UpdatePlotEvent, EVT_UPDATE_PLOT = wx.lib.newevent.NewEvent()

class MyFrame(wx.Frame):
    def __init__(self):
        screen_size = wx.DisplaySize()
        default_width = min(screen_size[0] * 0.9, 1000)
        default_height = min(screen_size[1] * 0.9, 700)
        super().__init__(None, title="Investment System Interface", size=(int(default_width), int(default_height)))
        self.root = os.getcwd()
        self.logger = setup_logging()
        self.config = load_config()
        self.crawler = YahooFinanceCrawler(self.config)
        self.backtester = Backtester(self.config)
        self.panel = wx.ScrolledWindow(self, -1)
        self.panel.SetScrollbars(20, 20, 50, 50)
        self.figure, self.ax = plt.subplots(figsize=(default_width / 100, default_height / 200))
        self.canvas = FigureCanvas(self.panel, -1, self.figure)
        self.multi_path_build()
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.output = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        text1 = wx.StaticText(self.panel, label="Target Web Crawling: ", size=(130, -1))
        self.combo1 = wx.ComboBox(self.panel, choices=WEBLIST, value="ALL", style=wx.CB_READONLY)
        hbox1.Add(text1, 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        hbox1.Add(self.combo1, 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        text2 = wx.StaticText(self.panel, label="Storing Path: ")
        self.input1 = wx.TextCtrl(self.panel, style=wx.TE_READONLY)
        self.input1.SetValue(self.config.get('stk2_dir', os.path.join(self.config['data_dir'], 'raw')))
        hbox1.Add(text2, 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        hbox1.Add(self.input1, 1, flag=wx.ALL | wx.EXPAND, border=5)
        button0 = wx.Button(self.panel, label="Change Path")
        self.Bind(wx.EVT_BUTTON, lambda event: handle_storing_path(self, event), button0)
        hbox1.Add(button0, 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        text3 = wx.StaticText(self.panel, label="Stock: ", size=(130, -1))
        self.combo_stock = wx.ComboBox(self.panel, choices=self.get_stock_list(), value="Select Stock", style=wx.CB_READONLY)
        hbox2.Add(text3, 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        hbox2.Add(self.combo_stock, 1, flag=wx.ALL | wx.EXPAND, border=5)
        self.combo_chart_type = wx.ComboBox(self.panel, choices=CHART_TYPES, value="Line", style=wx.CB_READONLY)
        hbox2.Add(wx.StaticText(self.panel, label="Chart Type: "), 0, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
        hbox2.Add(self.combo_chart_type, 0, flag=wx.ALL | wx.EXPAND, border=5)
        self.chart_type = "Line"
        self.scanned_timeframe = None  # Store which timeframe was used for scanning
        self.combo_chart_type.Bind(wx.EVT_COMBOBOX, lambda event: handle_chart_type_change(self, event))
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.crawl_button = wx.Button(self.panel, label="Crawl Data")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_crawl_data(event), self.crawl_button)
        self.backtest_button = wx.Button(self.panel, label="Run Backtest")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_run_backtest(event), self.backtest_button)
        self.plot_button = wx.Button(self.panel, label="Show Elliott Wave")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_show_elliott_wave(event), self.plot_button)
        self.current_pos_button = wx.Button(self.panel, label="Analyze Current Position")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_analyze_current_position(event), self.current_pos_button)
        self.scan_button = wx.Button(self.panel, label="Scan All Stocks")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_scan_all_stocks(event), self.scan_button)
        self.load_scan_button = wx.Button(self.panel, label="Load Last Scan")
        self.Bind(wx.EVT_BUTTON, lambda event: self.safe_handle_load_scan(event), self.load_scan_button)
        hbox3.Add(self.crawl_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.backtest_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.plot_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.current_pos_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.scan_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.load_scan_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        # Add stocks with patterns listbox on the left
        self.stocks_list = wx.ListBox(self.panel, style=wx.LB_SINGLE, size=(200, -1))
        self.stocks_list.Bind(wx.EVT_LISTBOX, self.on_stock_selected)
        hbox5.Add(self.stocks_list, 0, flag=wx.ALL | wx.EXPAND, border=5)
        hbox5.Add(self.canvas, 1, flag=wx.ALL | wx.EXPAND, border=5)
        self.vbox.Add(hbox1, 0, flag=wx.ALL | wx.EXPAND)
        self.vbox.Add(hbox2, 0, flag=wx.ALL | wx.EXPAND)
        self.vbox.Add(hbox3, 0, flag=wx.ALL | wx.EXPAND)
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4.Add(self.output, 1, flag=wx.ALL | wx.EXPAND, border=5)
        self.vbox.Add(hbox4, 0, flag=wx.ALL | wx.EXPAND)
        self.vbox.Add(hbox5, 2, flag=wx.ALL | wx.EXPAND)
        self.panel.SetSizer(self.vbox)
        self.panel.Layout()
        self.SetMinSize((600, 400))
        self.SetMaxSize((-1, -1))
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(EVT_UPDATE_OUTPUT, self.on_update_output)
        self.Bind(EVT_UPDATE_PLOT, self.on_update_plot)
        self.Bind(EVT_ENABLE_BUTTONS, self.on_enable_buttons)
        self.buttons_enabled = True

        # Initialize scan cache
        from src.utils.scan_cache import ScanCache
        self.scan_cache = ScanCache()

        # Auto-load last scan on startup
        wx.CallAfter(self.auto_load_last_scan)
    def on_resize(self, event):
        try:
            self.panel.Layout()
            new_size = self.panel.GetClientSize()
            if new_size.width > 0 and new_size.height > 0:
                self.canvas.SetSize(new_size)
                fig_width = max(6, new_size.width / 100)
                fig_height = max(4, new_size.height / 150)
                if hasattr(self, 'canvas') and hasattr(self.canvas, 'figure'):
                    self.canvas.figure.set_size_inches(fig_width, fig_height)
                    self.canvas.figure.tight_layout(pad=2.0, rect=[0.05, 0.15, 0.95, 0.95])
                    self.canvas.figure.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)
                    self.canvas.draw()
            event.Skip()
        except Exception as e:
            logger.error(f"Error in resize handler: {e}")
            event.Skip()
    def multi_path_build(self):
        for folder in ['data/raw', 'models', 'htmls', 'data/lists/adjustments']:
            os.makedirs(os.path.join(self.root, folder), exist_ok=True)
    def get_stock_list(self):
        symbols = []
        try:
            if Path(self.config['international_file']).exists():
                symbols += list(pd.read_csv(self.config['international_file'])['code'])
            else:
                self.output.AppendText(f"Warning: {self.config['international_file']} not found.\n")
        except Exception as e:
            self.output.AppendText(f"Error loading international.txt: {e}\n")
        try:
            if Path(self.config['list_file']).exists():
                symbols += list(pd.read_excel(self.config['list_file'])['code'].astype(str))
            else:
                self.output.AppendText(f"Warning: {self.config['list_file']} not found.\n")
        except Exception as e:
            self.output.AppendText(f"Error loading list.xlsx: {e}\n")
        try:
            if Path(self.config['otclist_file']).exists():
                symbols += list(pd.read_excel(self.config['otclist_file'])['code'].astype(str))
            else:
                self.output.AppendText(f"Warning: {self.config['otclist_file']} not found.\n")
        except Exception as e:
            self.output.AppendText(f"Error loading otclist.xlsx: {e}\n")
        symbols.append('TWII')
        return sorted(set(symbols))
    def on_update_output(self, event):
        try:
            if hasattr(event, 'message'):
                self.output.AppendText(event.message)
        except Exception as e:
            logger.error(f"Error updating output: {e}")
    def on_update_plot(self, event):
        try:
            if hasattr(event, 'figure'):
                self.canvas.figure = event.figure
                self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    def on_enable_buttons(self, event):
        """Handle EnableButtonsEvent from threaded handlers."""
        try:
            if hasattr(event, 'enable'):
                self.enable_buttons(event.enable)
        except Exception as e:
            logger.error(f"Error enabling buttons: {e}")
    def enable_buttons(self, enabled=True):
        try:
            buttons = [self.crawl_button, self.backtest_button, self.plot_button,
                      self.current_pos_button, self.scan_button, self.load_scan_button]
            for button in buttons:
                if button:
                    button.Enable(enabled)
            self.buttons_enabled = enabled
        except Exception as e:
            logger.error(f"Error toggling buttons: {e}")
    def safe_handle_crawl_data(self, event):
        try:
            if not self.buttons_enabled:
                return
            self.enable_buttons(False)
            handle_crawl_data(self, event)
        except Exception as e:
            self.output.AppendText(f"Error in crawl data handler: {e}\n")
            self.enable_buttons(True)
    def safe_handle_run_backtest(self, event):
        try:
            if not self.buttons_enabled:
                return
            self.enable_buttons(False)
            handle_run_backtest(self, event)
        except Exception as e:
            self.output.AppendText(f"Error in backtest handler: {e}\n")
            self.enable_buttons(True)
    def safe_handle_show_elliott_wave(self, event):
        try:
            handle_show_elliott_wave(self, event)
        except Exception as e:
            self.output.AppendText(f"Error in Elliott Wave handler: {e}\n")
    def safe_handle_analyze_current_position(self, event):
        try:
            handle_analyze_current_position(self, event)
        except Exception as e:
            self.output.AppendText(f"Error in current position handler: {e}\n")

    def safe_handle_scan_all_stocks(self, event):
        """Handle scan all stocks button click"""
        try:
            if not self.buttons_enabled:
                return
            self.enable_buttons(False)
            from gui.handlers import handle_scan_all_stocks
            handle_scan_all_stocks(self, event)
        except Exception as e:
            self.output.AppendText(f"Error in scan handler: {e}\n")
            self.enable_buttons(True)

    def safe_handle_load_scan(self, event):
        """Handle load last scan button click"""
        try:
            from gui.handlers import handle_load_scan
            handle_load_scan(self, event)
        except Exception as e:
            self.output.AppendText(f"Error loading scan: {e}\n")

    def auto_load_last_scan(self):
        """Automatically load last scan results on startup"""
        try:
            cache_info = self.scan_cache.get_cache_info()
            if cache_info and cache_info['exists']:
                self.output.AppendText(f"\n{'='*70}\n")
                self.output.AppendText(f"LAST SCAN AVAILABLE\n")
                self.output.AppendText(f"{'='*70}\n")
                self.output.AppendText(f"Timestamp: {cache_info['age_formatted']}\n")
                self.output.AppendText(f"Timeframe: {cache_info['timeframe']}\n")
                self.output.AppendText(f"Patterns found: {cache_info['patterns_found']}/{cache_info['total_scanned']}\n")
                self.output.AppendText(f"\nClick 'Load Last Scan' to restore results\n")
                self.output.AppendText(f"{'='*70}\n\n")
        except Exception as e:
            # Silent fail - not critical
            logger.debug(f"Auto-load check failed: {e}")

    def on_stock_selected(self, event):
        """Handle stock selection from the listbox"""
        try:
            selection = self.stocks_list.GetStringSelection()
            if selection:
                # Extract stock symbol from the display string (format: "SYMBOL (confidence%)")
                symbol = selection.split()[0]
                # Set the combo box to this stock
                self.combo_stock.SetValue(symbol)

                # If we have a scanned timeframe, use that specific timeframe
                # Otherwise use the current chart type
                if self.scanned_timeframe:
                    # Map timeframe back to chart type
                    timeframe_to_chart = {
                        'day': 'Candlestick (Day)',
                        'week': 'Candlestick (Week)',
                        'month': 'Candlestick (Month)'
                    }
                    chart_type_for_scan = timeframe_to_chart.get(self.scanned_timeframe, self.chart_type)

                    # Temporarily set chart type to match scan
                    original_chart_type = self.chart_type
                    self.chart_type = chart_type_for_scan
                    self.combo_chart_type.SetValue(chart_type_for_scan)

                    self.output.AppendText(f"Displaying {symbol} using scanned timeframe: {self.scanned_timeframe}\n")

                # Automatically show Elliott Wave for this stock
                from gui.handlers import handle_show_elliott_wave
                handle_show_elliott_wave(self, None)
        except Exception as e:
            self.output.AppendText(f"Error selecting stock: {e}\n")
    def _fix_date_labels(self, ax, data_df):
        """Fix date label formatting and layout issues."""
        try:
            start_date = data_df.index.min()
            end_date = data_df.index.max()
            date_range = (end_date - start_date).days
            if date_range <= 30:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            elif date_range <= 365:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            elif date_range <= 1825:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            for label in ax.xaxis.get_majorticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
            ax.margins(x=0.02)
        except Exception as e:
            logger.error(f"Error in _fix_date_labels: {e}")

    def _plot_candlestick_elliott_wave(self, df, wave_data, symbol):
        """Plot candlestick chart with Elliott Wave analysis"""
        try:
            # Determine resampling frequency
            freq_map = {
                "Candlestick (Day)": 'D',
                "Candlestick (Week)": 'W',
                "Candlestick (Month)": 'M'
            }
            freq = freq_map.get(self.chart_type, 'D')
            
            # Resample OHLC data
            df_ohlc = resample_ohlc(df, freq)
            
            # Clean resampled data
            df_ohlc = df_ohlc.dropna()
            if df_ohlc.empty:
                self.output.AppendText("Error: No data after resampling\n")
                return
                
            # Ensure proper datetime index
            if not isinstance(df_ohlc.index, pd.DatetimeIndex):
                df_ohlc.index = pd.to_datetime(df_ohlc.index)
            
            # Validate OHLC relationships
            df_ohlc = df_ohlc[
                (df_ohlc['high'] >= df_ohlc['low']) &
                (df_ohlc['high'] >= df_ohlc['open']) &
                (df_ohlc['high'] >= df_ohlc['close']) &
                (df_ohlc['low'] <= df_ohlc['open']) &
                (df_ohlc['low'] <= df_ohlc['close'])
            ]
            
            if df_ohlc.empty:
                self.output.AppendText("Error: No valid OHLC data after validation\n")
                return
            
            self.output.AppendText(f"Resampled data: {len(df_ohlc)} candles\n")
            
            # Create marker series for Elliott Wave points
            additional_plots = []
            
            # Map impulse wave points to resampled data
            impulse_wave = wave_data.get('impulse_wave', np.array([]))
            if len(impulse_wave) > 0:
                impulse_series = self._map_points_to_ohlc(df, df_ohlc, impulse_wave, 'close')
                if impulse_series.notna().sum() > 0:
                    additional_plots.append(
                        mpf.make_addplot(
                            impulse_series,
                            type='scatter',
                            markersize=100,
                            marker='^',
                            color='blue'
                        )
                    )
            
            # Map corrective wave points
            corrective_wave = wave_data.get('corrective_wave', np.array([]))
            if len(corrective_wave) > 0:
                corrective_series = self._map_points_to_ohlc(df, df_ohlc, corrective_wave, 'close')
                if corrective_series.notna().sum() > 0:
                    additional_plots.append(
                        mpf.make_addplot(
                            corrective_series,
                            type='scatter',
                            markersize=100,
                            marker='v',
                            color='magenta'
                        )
                    )
            
            # Map peaks and troughs
            peaks = wave_data.get('peaks', np.array([]))
            troughs = wave_data.get('troughs', np.array([]))
            
            if len(peaks) > 0:
                peaks_series = self._map_points_to_ohlc(df, df_ohlc, peaks, 'close')
                if peaks_series.notna().sum() > 0:
                    additional_plots.append(
                        mpf.make_addplot(
                            peaks_series,
                            type='scatter',
                            markersize=50,
                            marker='o',
                            color='red'
                        )
                    )
            
            if len(troughs) > 0:
                troughs_series = self._map_points_to_ohlc(df, df_ohlc, troughs, 'close')
                if troughs_series.notna().sum() > 0:
                    additional_plots.append(
                        mpf.make_addplot(
                            troughs_series,
                            type='scatter',
                            markersize=50,
                            marker='o',
                            color='green'
                        )
                    )
            
            # Configure mplfinance style
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick={'up': 'green', 'down': 'red'},
                volume='inherit'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                y_on_right=False,
                gridcolor='gray',
                facecolor='white',
                figcolor='white',
                gridaxis='both'
            )
            
            # Get current figure size
            fig_width, fig_height = self.figure.get_size_inches()
            
            # Create the plot
            confidence = wave_data.get('confidence', 0.0)
            title = f"{self.chart_type} for {symbol} - Elliott Wave (Confidence: {confidence:.2f})"
            
            try:
                fig, axes = mpf.plot(
                    df_ohlc,
                    type='candle',
                    style=style,
                    title=title,
                    volume=True,
                    figsize=tuple(self.figure.get_size_inches()),
                    panel_ratios=(4, 1),
                    addplot=additional_plots if additional_plots else None,
                    returnfig=True,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    tight_layout=True,
                    show_nontrading=False
                )
                
                # Get axes
                price_ax = axes[0]
                volume_ax = axes[1] if len(axes) > 1 else None
                
                # Add annotations for Elliott Wave points
                self._add_elliott_wave_annotations(price_ax, df, df_ohlc, wave_data)
                
                # Update canvas
                self.canvas.figure = fig
                self.ax = price_ax
                self.canvas.draw()
                
            except Exception as plot_error:
                self.output.AppendText(f"mplfinance plot error: {plot_error}\n")
                # Fallback to simple line plot
                self._plot_line_elliott_wave(df, wave_data, symbol)
                
        except Exception as e:
            self.output.AppendText(f"Candlestick plot error: {e}\n")
            import traceback
            self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")

    def _map_points_to_ohlc(self, original_df, ohlc_df, point_indices, column):
        """Map point indices from original data to resampled OHLC data"""
        mapped_series = pd.Series(index=ohlc_df.index, dtype=float)
        
        for idx in point_indices:
            if 0 <= idx < len(original_df):
                original_date = original_df.index[idx]
                original_price = original_df[column].iloc[idx]
                
                # Find closest date in OHLC data
                if len(ohlc_df) > 0:
                    time_diffs = np.abs(ohlc_df.index - original_date)
                    closest_idx = time_diffs.argmin()
                    closest_date = ohlc_df.index[closest_idx]
                    
                    # Only map if the time difference is reasonable
                    time_diff = abs((closest_date - original_date).days)
                    if time_diff <= 7:  # Within a week
                        mapped_series[closest_date] = original_price
        
        return mapped_series

    def _add_elliott_wave_annotations(self, ax, original_df, ohlc_df, wave_data):
        """Add comprehensive Elliott Wave annotations"""
        try:
            impulse_wave = wave_data.get('impulse_wave', np.array([]))
            corrective_wave = wave_data.get('corrective_wave', np.array([]))
            
            # Clear, bold wave labels
            impulse_labels = ['1', '2', '3', '4', '5']
            corrective_labels = ['A', 'B', 'C']
            
            # Plot impulse wave with connecting lines and clear labels
            if len(impulse_wave) > 1:
                impulse_dates = []
                impulse_prices = []
                
                # Get dates and prices for impulse waves
                for idx in impulse_wave:
                    if 0 <= idx < len(original_df):
                        original_date = original_df.index[idx]
                        original_price = original_df['close'].iloc[idx]
                        
                        # Find corresponding date in OHLC
                        if len(ohlc_df) > 0:
                            time_diffs = np.abs(ohlc_df.index - original_date)
                            closest_idx = time_diffs.argmin()
                            plot_date = ohlc_df.index[closest_idx]
                            
                            impulse_dates.append(plot_date)
                            impulse_prices.append(original_price)
                
                # Draw THICK connecting line for impulse wave
                if len(impulse_dates) > 1:
                    ax.plot(impulse_dates, impulse_prices, 'b-', linewidth=4, alpha=0.8, 
                        zorder=15, label='Impulse Wave 1-2-3-4-5')
                
                # Add LARGE, CLEAR wave number labels
                for i, idx in enumerate(impulse_wave):
                    if 0 <= idx < len(original_df) and i < len(impulse_labels):
                        original_date = original_df.index[idx]
                        original_price = original_df['close'].iloc[idx]
                        
                        if len(ohlc_df) > 0:
                            time_diffs = np.abs(ohlc_df.index - original_date)
                            closest_idx = time_diffs.argmin()
                            plot_date = ohlc_df.index[closest_idx]
                            
                            # Determine label position (peaks up, troughs down)
                            is_peak = i % 2 == 0  # Waves 1, 3, 5 are typically peaks
                            y_offset = 40 if is_peak else -40
                            va_pos = 'bottom' if is_peak else 'top'
                            
                            # LARGE, BOLD, COLORED labels with white background
                            ax.annotate(
                                impulse_labels[i],
                                xy=(plot_date, original_price),
                                xytext=(0, y_offset),
                                textcoords='offset points',
                                ha='center',
                                va=va_pos,
                                fontsize=16,  # LARGER font
                                fontweight='bold',
                                color='blue',
                                bbox=dict(boxstyle='circle,pad=0.5', fc='white', ec='blue', 
                                        linewidth=2, alpha=1.0),  # Solid white background
                                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                                zorder=20
                            )
            
            # Plot corrective wave with different style
            if len(corrective_wave) > 1:
                corrective_dates = []
                corrective_prices = []
                
                for idx in corrective_wave:
                    if 0 <= idx < len(original_df):
                        original_date = original_df.index[idx]
                        original_price = original_df['close'].iloc[idx]
                        
                        if len(ohlc_df) > 0:
                            time_diffs = np.abs(ohlc_df.index - original_date)
                            closest_idx = time_diffs.argmin()
                            plot_date = ohlc_df.index[closest_idx]
                            
                            corrective_dates.append(plot_date)
                            corrective_prices.append(original_price)
                
                # Draw THICK dashed line for corrective wave
                if len(corrective_dates) > 1:
                    ax.plot(corrective_dates, corrective_prices, 'm--', linewidth=4, 
                        alpha=0.8, zorder=15, label='Corrective Wave A-B-C')
                
                # Add corrective wave labels
                for i, idx in enumerate(corrective_wave):
                    if 0 <= idx < len(original_df) and i < len(corrective_labels):
                        original_date = original_df.index[idx]
                        original_price = original_df['close'].iloc[idx]
                        
                        if len(ohlc_df) > 0:
                            time_diffs = np.abs(ohlc_df.index - original_date)
                            closest_idx = time_diffs.argmin()
                            plot_date = ohlc_df.index[closest_idx]
                            
                            # Corrective waves alternate direction
                            is_peak = i % 2 == 0
                            y_offset = 40 if is_peak else -40
                            va_pos = 'bottom' if is_peak else 'top'
                            
                            ax.annotate(
                                corrective_labels[i],
                                xy=(plot_date, original_price),
                                xytext=(0, y_offset),
                                textcoords='offset points',
                                ha='center',
                                va=va_pos,
                                fontsize=16,
                                fontweight='bold',
                                color='magenta',
                                bbox=dict(boxstyle='circle,pad=0.5', fc='white', ec='magenta', 
                                        linewidth=2, alpha=1.0),
                                arrowprops=dict(arrowstyle='->', color='magenta', lw=2),
                                zorder=20
                            )
            
            # Add current wave indicator
            if len(impulse_wave) > 0:
                current_wave = len(impulse_wave)
                current_price = original_df['close'].iloc[-1]
                current_date = ohlc_df.index[-1] if len(ohlc_df) > 0 else original_df.index[-1]
                
                # Large "CURRENT" indicator
                ax.annotate(
                    f'CURRENT\nWave {current_wave}',
                    xy=(current_date, current_price),
                    xytext=(50, 50),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    fontsize=14,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.8', fc='yellow', ec='red', 
                            linewidth=3, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                    zorder=25
                )
            
            # Add legend
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, framealpha=0.6, borderaxespad=0)
            
            # Add title with current wave info
            confidence = wave_data.get('confidence', 0.0)
            current_wave_info = f"Wave {len(impulse_wave)}" if len(impulse_wave) > 0 else "Unknown"
            ax.set_title(f'Elliott Wave Analysis - Current: {current_wave_info} (Confidence: {confidence:.2f})', 
                        fontsize=16, fontweight='bold', pad=20)
            
        except Exception as e:
            logger.error(f"Error adding enhanced annotations: {e}")

    def _add_fibonacci_levels(self, ax, original_df, ohlc_df, impulse_wave):
        """Add Fibonacci retracement and extension levels"""
        try:
            # Get wave 1 for retracement levels
            wave1_start_price = original_df['close'].iloc[impulse_wave[0]]
            wave1_end_price = original_df['close'].iloc[impulse_wave[1]]
            wave1_range = wave1_end_price - wave1_start_price
            
            # Common Fibonacci levels
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            # Draw retracement levels
            for level in fib_levels:
                fib_price = wave1_end_price - (wave1_range * level)
                ax.axhline(y=fib_price, color='gold', linestyle='--', alpha=0.3, linewidth=1)
                ax.text(ax.get_xlim()[1], fib_price, f' {level:.1%}', 
                       fontsize=8, color='gold', va='center', ha='left',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            
        except Exception as e:
            logger.error(f"Error adding Fibonacci levels: {e}")

    def _add_wave_info_panel(self, ax, wave_data):
        """Add information panel showing wave analysis details"""
        validation = wave_data.get('validation_details', {})
        
        info_text = []
        if 'wave_2_retracement' in validation:
            info_text.append(f"Wave 2 Retracement: {validation['wave_2_retracement']:.1%}")
        if 'wave_lengths' in validation:
            info_text.append(f"Wave 3 Strength: {'✓' if validation['wave_lengths'][2] == max(validation['wave_lengths']) else '⚠'}")
        if 'fibonacci' in validation:
            info_text.append(f"Fibonacci Score: {validation['fibonacci']:.2f}")
        if 'alternation' in validation:
            info_text.append(f"Wave Alternation: {validation['alternation']:.2f}")
        
        if info_text:
            panel_text = '\n'.join(info_text)
            ax.text(0.02, 0.98, panel_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    def _plot_line_elliott_wave(self, df, wave_data, symbol):
        """Fallback line plot for Elliott Wave analysis"""
        self.ax.clear()
        
        # Plot price line
        self.ax.plot(df.index, df['close'], label='Price', alpha=0.7, linewidth=1)
        
        # Plot Elliott Wave analysis
        plot_elliott_wave_analysis(
            df,
            wave_data,
            column='close',
            title=f"Elliott Wave Analysis for {symbol}",
            ax=self.ax
        )
        
        # Fix date labels
        self._fix_date_labels(self.ax, df)
        self.canvas.draw()

    def _plot_line_elliott_wave_enhanced(self, df, wave_data, symbol):
        """Enhanced line plotting for Elliott Wave analysis."""
        self.ax.clear()
        if len(wave_data.get('impulse_wave', [])) < 2 or wave_data.get('wave_type', '') == 'no_pattern':
            # Defensive: No valid pattern found
            self.ax.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5, alpha=0.8)
            self.ax.set_title(f"{symbol} - No valid Elliott Wave pattern found", fontsize=14, color='red')
            self.ax.text(0.5, 0.5, "No valid Elliott Wave pattern found for this data.",
                         transform=self.ax.transAxes, ha='center', va='center', fontsize=12, color='red')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self._fix_date_labels(self.ax, df)
            self.canvas.draw()
            if hasattr(self, 'output'):
                self.output.AppendText("No valid Elliott Wave pattern found for this data.\n")
            return
        plot_elliott_wave_analysis_enhanced(
            df, wave_data, column='close',
            title=f"Elliott Wave Analysis for {symbol}", ax=self.ax
        )
        self._fix_date_labels(self.ax, df)
        self.canvas.draw()

    def _plot_candlestick_elliott_wave_enhanced(self, df, wave_data, symbol):
        """Enhanced candlestick plotting with comprehensive Elliott Wave analysis."""
        # Defensive: No valid pattern found
        if len(wave_data.get('impulse_wave', [])) < 2 or wave_data.get('wave_type', '') == 'no_pattern':
            self.ax.clear()
            self.ax.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5, alpha=0.8)
            self.ax.set_title(f"{symbol} - No valid Elliott Wave pattern found", fontsize=14, color='red')
            self.ax.text(0.5, 0.5, "No valid Elliott Wave pattern found for this data.",
                         transform=self.ax.transAxes, ha='center', va='center', fontsize=12, color='red')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self._fix_date_labels(self.ax, df)
            self.canvas.draw()
            if hasattr(self, 'output'):
                self.output.AppendText("No valid Elliott Wave pattern found for this data.\n")
            return
        
        # Use the existing enhanced plotting function
        plot_elliott_wave_analysis_enhanced(
            df, wave_data, column='close',
            title=f"Elliott Wave Analysis for {symbol}", ax=self.ax
        )
        self._fix_date_labels(self.ax, df)
        self.canvas.draw()

    def _plot_line_elliott_wave_multiple(self, df, wave_data, symbol):
        """CORRECTED: Plot line chart with multiple timeframe Elliott Wave patterns"""
        try:
            # Clear the canvas and create a new axes
            self.canvas.figure.clf()
            ax = self.canvas.figure.add_subplot(111)
            
            # Plot price line
            ax.plot(df.index, df['close'], color='black', linewidth=1, alpha=0.6, label='Price')
            
            # Plot multiple patterns with timeframe-specific styling
            multiple_patterns = wave_data.get('multiple_patterns', [])
            colors = {'recent': 'red', 'medium': 'blue', 'extended': 'green', 'long_term': 'orange'}
            line_styles = {0: '-', 1: '--', 2: '-.', 3: ':'}
            
            for i, pattern in enumerate(multiple_patterns[:4]):  # Limit to 4 patterns
                # Use original indices to map back to full dataframe
                wave_points = pattern.get('original_indices', pattern['points'])
                
                if len(wave_points) >= 2:
                    color = colors.get(pattern['time_frame'], 'purple')
                    style = line_styles.get(i, '-')
                    alpha = 1.0 if i == 0 else 0.7
                    linewidth = 3 if i == 0 else 2
                    
                    # Ensure indices are within bounds
                    valid_indices = [idx for idx in wave_points if 0 <= idx < len(df)]
                    
                    if len(valid_indices) >= 2:
                        dates = df.index[valid_indices]
                        prices = df['close'].iloc[valid_indices]
                        
                        ax.plot(dates, prices, color=color, linestyle=style,
                               linewidth=linewidth, alpha=alpha, marker='o', markersize=6,
                               label=f"{pattern['time_frame'].title()} ({pattern['confidence']:.1%})")
                        
                        # Add wave numbers for primary pattern
                        if i == 0:
                            for j, (date, price) in enumerate(zip(dates, prices)):
                                ax.annotate(f"{j+1}", (date, price),
                                          xytext=(5, 10), textcoords='offset points',
                                          fontsize=10, fontweight='bold', color=color)
            
            ax.set_title(f"{symbol} - Multi-Timeframe Elliott Wave Analysis", fontsize=14, fontweight='bold')
            ax.set_ylabel("Price ($)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.output.AppendText(f"Error in corrected multiple pattern line plot: {e}\n")
            import traceback
            self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")
            # Fallback to single pattern plot
            self._plot_line_elliott_wave_enhanced(df, wave_data, symbol)

    def _plot_candlestick_elliott_wave_multiple(self, df, wave_data, symbol):
        """Plot candlestick chart with multiple Elliott Wave patterns"""
        try:
            # Clear the canvas and create a new axes
            self.canvas.figure.clf()
            ax = self.canvas.figure.add_subplot(111)
            
            # Plot candlestick data
            from mplfinance import plot as mpf_plot
            import matplotlib.dates as mdates
            
            # Basic candlestick plot
            ohlc_data = df[['open', 'high', 'low', 'close']].copy()
            
            # Plot multiple patterns with different colors and styles
            multiple_patterns = wave_data.get('multiple_patterns', [])
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            line_styles = ['-', '--', '-.', ':', '-']
            
            for i, pattern in enumerate(multiple_patterns[:5]):  # Limit to 5 patterns
                wave_points = pattern['points']
                if len(wave_points) >= 2:
                    color = colors[i % len(colors)]
                    style = line_styles[i % len(line_styles)]
                    alpha = 1.0 if i == 0 else 0.7  # Primary pattern more prominent
                    linewidth = 3 if i == 0 else 2
                    
                    # Plot wave lines
                    dates = df.index[wave_points]
                    prices = df['close'].iloc[wave_points]
                    
                    ax.plot(dates, prices, color=color, linestyle=style, 
                           linewidth=linewidth, alpha=alpha,
                           label=f"Pattern {i+1} ({pattern['time_frame']}, {pattern['confidence']:.1%})")
                    
                    # Add wave labels for primary pattern
                    if i == 0:
                        for j, (date, price) in enumerate(zip(dates, prices)):
                            ax.annotate(f"W{j+1}", (date, price), 
                                      xytext=(5, 10), textcoords='offset points',
                                      fontsize=10, fontweight='bold', color=color)
            
            # Add candlestick data
            ax.plot(df.index, df['close'], color='black', alpha=0.3, linewidth=1)
            
            # Formatting
            ax.set_title(f"{symbol} - Multiple Elliott Wave Patterns", fontsize=14, fontweight='bold')
            ax.set_ylabel("Price ($)", fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.output.AppendText(f"Error in multiple pattern candlestick plot: {e}\n")
            import traceback
            self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")
            # Fallback to single pattern plot
            self._plot_candlestick_elliott_wave_enhanced(df, wave_data, symbol)

    def create_user_friendly_elliott_wave_display(self, df: pd.DataFrame, wave_data: Dict[str, Any], symbol: str):
        """
        Create a user-friendly multi-panel Elliott Wave display that's easy to understand
        """
        # Extract patterns from the correct location in the data structure
        # The structure is: patterns_by_timeframe -> timeframe -> multiple_patterns
        multiple_patterns = []

        # Try new structure first (patterns_by_timeframe)
        patterns_by_tf = wave_data.get('patterns_by_timeframe', {})
        if patterns_by_tf:
            # Collect all patterns from all timeframes
            for timeframe, tf_data in patterns_by_tf.items():
                tf_patterns = tf_data.get('multiple_patterns', [])
                multiple_patterns.extend(tf_patterns)

        # Fallback to old structure (direct multiple_patterns)
        if not multiple_patterns:
            multiple_patterns = wave_data.get('multiple_patterns', [])

        if not multiple_patterns:
            self._create_no_pattern_display(symbol)
            return
        
        # Create subplots based on number of patterns (max 3 panels)
        num_patterns = min(len(multiple_patterns), 3)
        fig, axes = plt.subplots(num_patterns, 1, figsize=(15, 5 * num_patterns))
        
        if num_patterns == 1:
            axes = [axes]
        
        # Clear the current canvas
        self.canvas.figure.clf()
        
        for i, pattern in enumerate(multiple_patterns[:3]):
            ax = axes[i] if num_patterns > 1 else axes[0]
            
            # Get the timeframe-specific data range
            timeframe_data = self._get_timeframe_specific_data(df, pattern, symbol)
            
            # Plot this specific timeframe
            self._plot_single_timeframe_pattern(ax, timeframe_data, pattern, i == 0)
        
        plt.tight_layout()
        self.canvas.figure = fig
        self.canvas.draw()

    def _get_timeframe_specific_data(self, df: pd.DataFrame, pattern: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Get data specific to this timeframe for better visualization"""

        # Get the timeframe that was used for analysis
        timeframe = pattern.get('time_frame', 'day')

        # Apply the SAME filter that was used during pattern detection
        # This is critical - pattern indices are relative to the FILTERED data!
        from src.analysis.core.impulse import filter_by_candlestick_type
        subset_df = filter_by_candlestick_type(df, timeframe)

        start_date = subset_df.index[0]
        end_date = subset_df.index[-1]

        # Create human-readable timeframe description
        timeframe_description = f"{timeframe.title()}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        # Pattern points are already indices into the filtered dataframe!
        # No mapping needed - just use them directly
        wave_points = pattern.get('points', [])

        # Validate that indices are within bounds
        valid_indices = []
        for idx in wave_points:
            if 0 <= idx < len(subset_df):
                valid_indices.append(idx)
            else:
                # Index out of bounds - this shouldn't happen but log it
                logger.warning(f"Pattern index {idx} out of bounds for {symbol} (len={len(subset_df)})")

        return {
            'df': subset_df,
            'wave_indices': np.array(valid_indices),
            'timeframe_desc': timeframe_description,
            'confidence': pattern['confidence'],
            'wave_type': pattern['wave_type'],
            'recency_score': pattern['recency_score'],
            'pattern_rank': pattern.get('composite_score', 0)
        }

    def _plot_single_timeframe_pattern(self, ax, timeframe_data: Dict[str, Any], pattern: Dict[str, Any], is_primary: bool):
        """Plot a single timeframe pattern with clear labeling"""
        
        df_subset = timeframe_data['df']
        wave_indices = timeframe_data['wave_indices']
        
        # Plot the price line for this timeframe
        ax.plot(df_subset.index, df_subset['close'], 'k-', linewidth=1, alpha=0.7, label='Price')
        
        # Plot the Elliott Wave pattern
        if len(wave_indices) >= 2:
            # Filter valid indices
            valid_indices = [idx for idx in wave_indices if 0 <= idx < len(df_subset)]
            
            if len(valid_indices) >= 2:
                wave_dates = df_subset.index[valid_indices]
                wave_prices = df_subset['close'].iloc[valid_indices]
                
                # Use different colors for different priority
                if is_primary:
                    color = 'red'
                    linewidth = 4
                    alpha = 1.0
                    marker_size = 120
                else:
                    color = 'blue'
                    linewidth = 3
                    alpha = 0.8
                    marker_size = 100
                
                # Plot wave connections
                ax.plot(wave_dates, wave_prices, color=color, linewidth=linewidth, 
                       alpha=alpha, marker='o', markersize=8, 
                       markerfacecolor='white', markeredgecolor=color, 
                       markeredgewidth=2, label='Elliott Wave Pattern')
                
                # Add wave numbers
                for i, (date, price) in enumerate(zip(wave_dates, wave_prices)):
                    # Determine if this is a peak or trough for label positioning
                    offset = 20 if i % 2 == 0 else -30
                    va = 'bottom' if i % 2 == 0 else 'top'
                    
                    ax.annotate(f'{i+1}', 
                               xy=(date, price),
                               xytext=(0, offset),
                               textcoords='offset points',
                               ha='center', va=va,
                               fontsize=14, fontweight='bold',
                               color=color,
                               bbox=dict(boxstyle='circle,pad=0.5', 
                                       facecolor='white', 
                                       edgecolor=color,
                                       linewidth=2,
                                       alpha=0.9),
                               zorder=10)
        
        # Title with clear information
        confidence = timeframe_data['confidence']
        recency = timeframe_data['recency_score']
        timeframe_desc = timeframe_data['timeframe_desc']
        
        # Priority indicator
        priority = "🎯 PRIMARY" if is_primary else f"📊 SECONDARY"
        
        title = f"{priority} - {timeframe_desc}\nConfidence: {confidence:.1%} | Recency: {recency:.1%}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        # Add trading relevance indicator
        if recency > 0.8:
            relevance_text = "🔥 VERY CURRENT - High Trading Relevance"
            relevance_color = 'green'
        elif recency > 0.6:
            relevance_text = "📈 RECENT - Good Trading Relevance"
            relevance_color = 'orange'
        elif recency > 0.4:
            relevance_text = "📊 MODERATE - Some Trading Relevance"
            relevance_color = 'blue'
        else:
            relevance_text = "📉 HISTORICAL - Limited Current Relevance"
            relevance_color = 'gray'
        
        # Add relevance text box
        ax.text(0.02, 0.98, relevance_text, transform=ax.transAxes,
               fontsize=10, fontweight='bold', color=relevance_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=relevance_color, alpha=0.9),
               verticalalignment='top')
        
        # Format axes
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Smart date formatting based on timeframe length
        date_range = (df_subset.index[-1] - df_subset.index[0]).days
        
        if date_range <= 90:  # 3 months or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif date_range <= 365:  # 1 year or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        else:  # More than 1 year
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add current price indicator
        current_price = df_subset['close'].iloc[-1]
        current_date = df_subset.index[-1]
        
        ax.scatter([current_date], [current_price], s=200, c='gold', 
                  marker='*', edgecolors='black', linewidth=2, 
                  zorder=15, label='Current Price')
        
        ax.legend(loc='upper left', fontsize=10)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_draggable(True)

    def _create_no_pattern_display(self, symbol: str):
        """Create a display when no patterns are found"""
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        
        ax.text(0.5, 0.5, f'No Elliott Wave Patterns Found for {symbol}\n\n'
                          f'Try:\n'
                          f'• Different chart type (Day/Week/Month)\n'
                          f'• Different time period\n'
                          f'• Check if data is sufficient',
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
        ax.set_title(f'{symbol} - Elliott Wave Analysis', fontsize=16, fontweight='bold')
        ax.axis('off')
        self.canvas.draw()

    