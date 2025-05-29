import os
import wx
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path
from investment_system.src.analysis.elliott_wave import detect_elliott_wave_complete, plot_elliott_wave_analysis
from investment_system.src.analysis.elliott_wave import detect_current_wave_position
from .utils import resample_ohlc_improved
from .constants import CHART_TYPES
import wx.lib.newevent

# Create custom events for thread-safe communication
UpdateOutputEvent, EVT_UPDATE_OUTPUT = wx.lib.newevent.NewEvent()
UpdatePlotEvent, EVT_UPDATE_PLOT = wx.lib.newevent.NewEvent()

def handle_storing_path(self, event):
    with wx.DirDialog(self, "Select storage path", defaultPath=self.input1.GetValue(),
                      style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dialog:
        if dialog.ShowModal() == wx.ID_OK:
            self.input1.SetValue(dialog.GetPath())

def handle_crawl_data(self, event):
    def crawl_task():
        try:
            select_type = self.combo1.GetValue()
            symbols = self.get_stock_list()
            suffix = "" if select_type == "ALL" else ".TW" if select_type == "listed" else ".TWO"
            if select_type == "ALL":
                symbols = [s for s in symbols if s != 'TWII'] + ['TWII']
            elif select_type == "listed":
                symbols = list(pd.read_excel(self.config['list_file'])['code'].astype(str))
            elif select_type == "otc":
                symbols = list(pd.read_excel(self.config['otclist_file'])['code'])
            wx.PostEvent(self, UpdateOutputEvent(message=f"Starting crawl for {select_type}...\n"))
            self.crawler.crawl(symbols, suffix)
            wx.PostEvent(self, UpdateOutputEvent(message="Crawling completed.\n"))
        except Exception as e:
            wx.PostEvent(self, UpdateOutputEvent(message=f"Error during crawling: {e}\n"))
        finally:
            wx.CallAfter(lambda: self.enable_buttons(True))
    
    import threading
    threading.Thread(target=crawl_task, daemon=True).start()

def handle_run_backtest(self, event):
    def backtest_task():
        try:
            symbol = self.combo_stock.GetValue()
            if symbol == "Select Stock":
                wx.PostEvent(self, UpdateOutputEvent(message="Please select a stock.\n"))
                return
            wx.PostEvent(self, UpdateOutputEvent(message=f"Running backtest for {symbol}...\n"))
            self.backtester.run([symbol])
            results_df = self.backtester.summarize()
            if not results_df.empty:
                for _, row in results_df.iterrows():
                    message = f"{row['symbol']}: Profit={row['profit']:.2f}, min_price_change={row['min_price_change']}\n"
                    wx.PostEvent(self, UpdateOutputEvent(message=message))
            else:
                wx.PostEvent(self, UpdateOutputEvent(message=f"No results for {symbol}.\n"))
        except Exception as e:
            wx.PostEvent(self, UpdateOutputEvent(message=f"Error during backtest: {e}\n"))
        finally:
            wx.CallAfter(lambda: self.enable_buttons(True))
    
    import threading
    threading.Thread(target=backtest_task, daemon=True).start()

def handle_show_elliott_wave(self, event):
    symbol = self.combo_stock.GetValue()
    if symbol == "Select Stock":
        self.output.AppendText("Please select a stock.\n")
        return
    
    try:
        file_path = os.path.join(self.config['stk2_dir'], f"{symbol}.txt")
        if not os.path.exists(file_path):
            self.output.AppendText(f"No data file found for {symbol}.\n")
            return

        # Load and clean data
        df = pd.read_csv(file_path, sep='\t', skiprows=1, 
                        names=['Price', 'Close', 'High', 'Low', 'Open', 'Volume', 'Date'])
        
        # Clean and validate data
        df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        
        # Convert to numeric and handle errors
        for col in ['close', 'high', 'low', 'open', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['close', 'high', 'low', 'open'])
        df = df.sort_index()
        
        # Filter to last 10 years
        cutoff_date = df.index.max() - pd.DateOffset(years=10)
        df = df[df.index >= cutoff_date]
        
        # Ensure we have a proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
        
        self.output.AppendText(f"Loaded DataFrame with {len(df)} rows\n")
        
        if df.empty or len(df) < 10:
            self.output.AppendText(f"Insufficient data to plot for {symbol}.\n")
            return

        # Detect Elliott Wave patterns
        wave_data = detect_elliott_wave_complete(df, column='close')
        impulse_wave = wave_data['impulse_wave']
        corrective_wave = wave_data['corrective_wave']
        peaks = wave_data['peaks']
        troughs = wave_data['troughs']
        confidence = wave_data['confidence']
        
        self.output.AppendText(f"Elliott Wave Analysis Results:\n")
        self.output.AppendText(f"- Impulse wave points: {len(impulse_wave)}\n")
        self.output.AppendText(f"- Corrective wave points: {len(corrective_wave)}\n")
        self.output.AppendText(f"- Confidence: {confidence:.2f}\n")

        if self.chart_type.startswith("Candlestick"):
            self._plot_candlestick_elliott_wave(df, wave_data, symbol)
        else:
            self._plot_line_elliott_wave(df, wave_data, symbol)
            
        self.output.AppendText(f"Elliott Wave plot displayed for {symbol}.\n")
        
    except Exception as e:
        self.output.AppendText(f"Error plotting Elliott Wave for {symbol}: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")

def handle_analyze_current_position(self, event):
    symbol = self.combo_stock.GetValue()
    if symbol == "Select Stock":
        self.output.AppendText("Please select a stock.\n")
        return
    try:
        file_path = os.path.join(self.config['stk2_dir'], f"{symbol}.txt")
        if not os.path.exists(file_path):
            self.output.AppendText(f"No data file found for {symbol}.\n")
            return
        
        # Load and preprocess data
        df = pd.read_csv(file_path, sep='\t', skiprows=1, names=['Price', 'Close', 'High', 'Low', 'Open', 'Volume', 'Date'])
        df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        
        # Convert all columns to numeric
        for col in ['close', 'high', 'low', 'open', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only rows with valid close prices
        df = df[pd.to_numeric(df['close'], errors='coerce').notna()]
        df = df.sort_index()
        
        if df.empty or len(df) < 10:
            self.output.AppendText(f"Insufficient data for analysis: {len(df)} rows\n")
            return
        
        # Analyze current position
        position_data = detect_current_wave_position(df, column='close')
        
        # Display results
        self.output.AppendText(f"\n--- CURRENT ELLIOTT WAVE POSITION FOR {symbol} ---\n")
        self.output.AppendText(f"Position: {position_data['position']}\n")
        self.output.AppendText(f"Confidence: {position_data['confidence']:.2f}\n")
        self.output.AppendText(f"Forecast: {position_data['forecast']}\n")
        
        # Update the plot with additional visualization for current position
        _update_current_position_plot(self, df, position_data)
        
    except Exception as e:
        self.output.AppendText(f"Error analyzing current position: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")

def handle_chart_type_change(self, event):
    self.chart_type = self.combo_chart_type.GetValue()

def _update_current_position_plot(self, df, position_data):
    """
    Update plot to highlight the current wave position with improved error handling.
    """
    try:
        # Get recent data for the plot (last 2 years)
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        recent_df = df[df.index >= start_date].copy()
        recent_df = recent_df[recent_df.index.notna()]
        if not isinstance(recent_df.index, pd.DatetimeIndex):
            recent_df.index = pd.to_datetime(recent_df.index, errors='coerce')
        recent_df = recent_df[recent_df.index.notna()]
        
        # Get position information
        position = position_data['position']
        confidence = position_data['confidence']
        forecast = position_data['forecast']
        
        # Check if we have enough data
        if recent_df.empty or len(recent_df) < 5:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Insufficient data for current position analysis.\nPosition: {position}\nConfidence: {confidence:.2f}",
                         ha='center', va='center', fontsize=12, transform=self.ax.transAxes)
            self.ax.set_title(f"Current Elliott Wave Analysis - {position}")
            self.canvas.draw()
            return
        
        # Check if we want a candlestick chart
        if self.chart_type.startswith("Candlestick"):
            # Ensure all OHLCV columns are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in recent_df.columns:
                    recent_df[col] = pd.to_numeric(recent_df[col], errors='coerce')
            
            # Drop any rows with NaN values in essential columns
            essential_cols = ['open', 'high', 'low', 'close']
            recent_df = recent_df.dropna(subset=[col for col in essential_cols if col in recent_df.columns])
            
            # Determine frequency
            if self.chart_type == "Candlestick (Day)":
                freq = 'D'
            elif self.chart_type == "Candlestick (Week)":
                freq = 'W'
            elif self.chart_type == "Candlestick (Month)":
                freq = 'M'
            else:
                freq = 'D'
            
            # Resample data
            df_ohlc = resample_ohlc_improved(recent_df, freq)
            # --- Fix: Clean index after resampling ---
            df_ohlc = df_ohlc[df_ohlc.index.notna()]
            if not isinstance(df_ohlc.index, pd.DatetimeIndex):
                df_ohlc.index = pd.to_datetime(df_ohlc.index, errors='coerce')
            df_ohlc = df_ohlc[df_ohlc.index.notna()]
            # --- Drop rows where any OHLC is NaN ---
            df_ohlc = df_ohlc.dropna(subset=['open', 'high', 'low', 'close'])
            self.output.AppendText(f"[DEBUG] df_ohlc after dropna: {df_ohlc.shape}, index: {df_ohlc.index.min()} to {df_ohlc.index.max()}\n")
            if df_ohlc.empty:
                self.ax.clear()
                self.ax.text(0.5, 0.5, f"Not enough data to plot candlestick chart.\nPosition: {position}\nConfidence: {confidence:.2f}\nForecast: {forecast}",
                             ha='center', va='center', fontsize=10, transform=self.ax.transAxes)
                self.ax.set_title("Current Elliott Wave Analysis - Insufficient Data")
                self.canvas.draw()
                return
            
            # Prepare plot styles
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick={'up':'green', 'down':'red'},
                volume='inherit'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                y_on_right=False,
                gridcolor='gray',
                facecolor='white',
                edgecolor='black',
                figcolor='white',
                gridaxis='both',
                rc={'grid.alpha': 0.3},
                base_mpf_style='charles'
            )
            
            # Create title with position info
            title = f"Current Elliott Wave Analysis - {position} (Conf: {confidence:.2f})"
            
            # Determine the position label color
            position_color = 'green' if 'Impulse' in position else 'red' if 'Corrective' in position else 'orange'
            
            # Create additional plots safely
            additional_plots = []
            
            # Only add position marker if we have valid data
            if len(df_ohlc) > 0 and 'close' in df_ohlc.columns:
                try:
                    last_price = df_ohlc['close'].iloc[-1]
                    last_date = df_ohlc.index[-1]
                    
                    # Create a properly formatted Series for the position marker
                    position_marker_series = pd.Series(index=df_ohlc.index, dtype=float)
                    position_marker_series[last_date] = last_price
                    
                    # Only add if we have valid data
                    if not pd.isna(last_price):
                        position_marker = mpf.make_addplot(
                            position_marker_series,
                            type='scatter',
                            markersize=120,
                            marker='*',
                            color=position_color,
                            panel=0
                        )
                        additional_plots.append(position_marker)
                        
                except Exception as e:
                    self.output.AppendText(f"Warning: Could not create position marker: {e}\n")
            
            # Calculate appropriate figure size based on current canvas
            fig_width, fig_height = self.figure.get_size_inches()
            
            try:
                # Create a new figure
                fig, axes = mpf.plot(
                    df_ohlc,
                    type='candle',
                    style=style,
                    title=title,
                    volume=True,
                    figsize=(fig_width, fig_height),
                    panel_ratios=(4, 1),  # Price to volume ratio
                    addplot=additional_plots if additional_plots else None,
                    returnfig=True,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    tight_layout=False,
                    show_nontrading=False
                )
                
                # Get the price axis
                ax1 = axes[0]
                volume_ax = axes[1] if len(axes) > 1 else None
                
                # Add forecast annotation as a text box at the bottom
                forecast_box = {
                    'boxstyle': 'round,pad=0.5',
                    'facecolor': 'white',
                    'alpha': 0.9,
                    'edgecolor': position_color
                }
                
                # Truncate long forecast text
                forecast_text = forecast if len(forecast) < 100 else forecast[:97] + "..."
                
                # Add forecast text at bottom of chart with fixed position
                ax1.text(
                    0.5, 0.02,
                    f"Forecast: {forecast_text}",
                    transform=ax1.transAxes,
                    fontsize=9,
                    ha='center',
                    bbox=forecast_box
                )
                
                # Add confidence indicator
                ax1.text(
                    0.02, 0.96,
                    f"Confidence: {confidence:.2f}",
                    transform=ax1.transAxes,
                    fontsize=10,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                
                # Highlight the current position region
                try:
                    highlight_start = max(0, len(df_ohlc) - int(len(df_ohlc) * 0.1))
                    if highlight_start < len(df_ohlc) and len(df_ohlc) > 5:
                        highlight_date_start = df_ohlc.index[highlight_start]
                        highlight_date_end = df_ohlc.index[-1]
                        x_start = mdates.date2num(highlight_date_start)
                        x_end = mdates.date2num(highlight_date_end)
                        y_min = df_ohlc['low'].min() * 0.98
                        y_max = df_ohlc['high'].max() * 1.02
                        rect = patches.Rectangle(
                            (x_start, y_min),
                            x_end - x_start,
                            y_max - y_min,
                            linewidth=1,
                            edgecolor=position_color,
                            facecolor=position_color,
                            alpha=0.1
                        )
                        ax1.add_patch(rect)
                except Exception as e:
                    self.output.AppendText(f"Warning: Could not add highlight rectangle: {e}\n")
                
                # Update the figure in the canvas
                self.canvas.figure = fig
                
                # Store the current axes for later reference
                self.ax = ax1
                
            except Exception as e:
                self.output.AppendText(f"Error creating candlestick plot: {e}\n")
                # Fallback to simple plot
                self._create_fallback_plot(recent_df, position_data)
                return
        
        else:
            # Original line chart implementation
            self.ax.clear()
            
            if recent_df.empty or len(recent_df) < 2:
                self.ax.text(0.5, 0.5, f"Not enough data to plot line chart.\nPosition: {position}\nConfidence: {confidence:.2f}",
                             ha='center', va='center', fontsize=12, color='red', transform=self.ax.transAxes)
                self.ax.set_title("Current Elliott Wave Analysis - Insufficient Data")
                self.canvas.draw()
                return
            
            self.ax.plot(recent_df.index, recent_df['close'], label='Price', color='blue')
            
            # Add visual indicator for current position
            try:
                if len(recent_df) >= 15:
                    if 'Impulse' in position:
                        self.ax.axvspan(recent_df.index[-15], recent_df.index[-1],
                                        alpha=0.2, color='green', label=f'Current: {position}')
                    elif 'Corrective' in position:
                        self.ax.axvspan(recent_df.index[-15], recent_df.index[-1],
                                        alpha=0.2, color='red', label=f'Current: {position}')
                    else:
                        self.ax.axvspan(recent_df.index[-15], recent_df.index[-1],
                                        alpha=0.2, color='yellow', label=f'Current: {position}')
            except Exception as e:
                self.output.AppendText(f"Warning: Could not add position highlight: {e}\n")
            
            # Add forecast annotation
            try:
                forecast_text = forecast if len(forecast) < 50 else forecast[:47] + "..."
                self.ax.annotate(
                    forecast_text,
                    xy=(recent_df.index[-1], recent_df['close'].iloc[-1]),
                    xytext=(recent_df.index[-1], recent_df['close'].min()),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    horizontalalignment='right', verticalalignment='bottom',
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )
            except Exception as e:
                self.output.AppendText(f"Warning: Could not add forecast annotation: {e}\n")
            
            # Add confidence indicator
            self.ax.text(
                0.02, 0.02,
                f"Confidence: {confidence:.2f}",
                transform=self.ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
            
            self.ax.set_title(f"Current Elliott Wave Analysis - {position}")
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.legend()
            self.ax.grid(True)
        
        # Refresh the canvas
        self.canvas.draw()
        
    except Exception as e:
        self.output.AppendText(f"Error in _update_current_position_plot: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")
        # Create a simple fallback plot
        self._create_fallback_plot(df, position_data)

def _create_fallback_plot(self, df, position_data):
    """
    Create a simple fallback plot when candlestick plotting fails.
    """
    try:
        self.ax.clear()
        
        # Get recent data
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(years=1)
        recent_df = df[df.index >= start_date].copy()
        
        if recent_df.empty or len(recent_df) < 2:
            self.ax.text(0.5, 0.5, "Insufficient data for analysis", 
                         ha='center', va='center', fontsize=12, transform=self.ax.transAxes)
        else:
            # Simple line plot
            self.ax.plot(recent_df.index, recent_df['close'], label='Price', color='blue')
            
            position = position_data['position']
            confidence = position_data['confidence']
            forecast = position_data['forecast']
            
            # Add position information as text
            info_text = f"Position: {position}\nConfidence: {confidence:.2f}\nForecast: {forecast[:50]}..."
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            self.ax.set_title("Current Elliott Wave Analysis (Fallback View)")
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.grid(True)
        
        self.canvas.draw()
        
    except Exception as e:
        self.output.AppendText(f"Error in fallback plot: {e}\n")

def _fix_date_labels(self, ax, df):
    # Implementation of _fix_date_labels method
    pass 