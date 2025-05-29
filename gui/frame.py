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
from investment_system.src.crawler.yahoo_finance import YahooFinanceCrawler
from investment_system.src.backtest.backtester import Backtester
from investment_system.src.utils.config import load_config
from investment_system.src.utils.logging import setup_logging
from investment_system.src.analysis.elliott_wave import detect_peaks_troughs, refined_elliott_wave_suggestion, plot_peaks_troughs, detect_corrective_waves, detect_elliott_wave_complete, plot_elliott_wave_analysis
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from investment_system.gui.constants import WEBLIST, CHART_TYPES
from investment_system.gui.handlers import (
    handle_storing_path, handle_crawl_data, handle_run_backtest, handle_show_elliott_wave, handle_analyze_current_position, handle_chart_type_change
)
from investment_system.gui.utils import resample_ohlc
import wx.lib.newevent

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
        self.input1.SetValue(os.path.join(self.config['data_dir'], 'raw'))
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
        hbox3.Add(self.crawl_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.backtest_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.plot_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox3.Add(self.current_pos_button, 1, flag=wx.ALL | wx.EXPAND, border=5)
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
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
        self.buttons_enabled = True
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
            print(f"Error in resize handler: {e}")
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
            print(f"Error updating output: {e}")
    def on_update_plot(self, event):
        try:
            if hasattr(event, 'figure'):
                self.canvas.figure = event.figure
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating plot: {e}")
    def enable_buttons(self, enabled=True):
        try:
            buttons = [self.crawl_button, self.backtest_button, self.plot_button, self.current_pos_button]
            for button in buttons:
                if button:
                    button.Enable(enabled)
            self.buttons_enabled = enabled
        except Exception as e:
            print(f"Error toggling buttons: {e}")
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
            print(f"Error in _fix_date_labels: {e}")

    def _plot_candlestick_elliott_wave(self, df, wave_data, symbol):
        """Plot candlestick chart with Elliott Wave analysis"""
        try:
            from .utils import resample_ohlc_improved
            # Determine resampling frequency
            freq_map = {
                "Candlestick (Day)": 'D',
                "Candlestick (Week)": 'W',
                "Candlestick (Month)": 'M'
            }
            freq = freq_map.get(self.chart_type, 'D')
            
            # Resample OHLC data
            df_ohlc = resample_ohlc_improved(df, freq)
            
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
                    figsize=(fig_width, fig_height),
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
        """Add text annotations for Elliott Wave points"""
        try:
            impulse_wave = wave_data.get('impulse_wave', np.array([]))
            corrective_wave = wave_data.get('corrective_wave', np.array([]))
            
            # Annotate impulse wave points
            for i, idx in enumerate(impulse_wave):
                if 0 <= idx < len(original_df):
                    original_date = original_df.index[idx]
                    original_price = original_df['close'].iloc[idx]
                    
                    # Find corresponding date in OHLC data
                    if len(ohlc_df) > 0:
                        time_diffs = np.abs(ohlc_df.index - original_date)
                        closest_idx = time_diffs.argmin()
                        plot_date = ohlc_df.index[closest_idx]
                        
                        ax.annotate(
                            f'W{i+1}',
                            xy=(plot_date, original_price),
                            xytext=(0, 15),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            fontweight='bold',
                            color='blue',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
                        )
            
            # Annotate corrective wave points
            labels = ['A', 'B', 'C']
            for i, idx in enumerate(corrective_wave):
                if 0 <= idx < len(original_df) and i < len(labels):
                    original_date = original_df.index[idx]
                    original_price = original_df['close'].iloc[idx]
                    
                    if len(ohlc_df) > 0:
                        time_diffs = np.abs(ohlc_df.index - original_date)
                        closest_idx = time_diffs.argmin()
                        plot_date = ohlc_df.index[closest_idx]
                        
                        ax.annotate(
                            labels[i],
                            xy=(plot_date, original_price),
                            xytext=(0, -15),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            fontsize=9,
                            fontweight='bold',
                            color='magenta',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
                        )
            
            # Add confidence indicator
            confidence = wave_data.get('confidence', 0.0)
            ax.text(
                0.02, 0.98,
                f"Confidence: {confidence:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
            
        except Exception as e:
            print(f"Error adding annotations: {e}")

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