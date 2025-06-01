import os
import wx
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import threading
from pathlib import Path
from typing import Dict, Any, List

from src.analysis.elliott_wave import (
    detect_elliott_wave_complete, 
    plot_elliott_wave_analysis_enhanced, 
    detect_current_wave_position_enhanced,
    validate_impulse_wave_rules,
    WaveType
)
from src.utils.common_utils import (
    load_and_preprocess_data,
    resample_ohlc,
    map_points_to_ohlc,
    get_confidence_description,
    get_position_color
)
from .constants import CHART_TYPES
import wx.lib.newevent
from src.utils.async_utils import run_in_thread
from src.analysis.core import detect_peaks_troughs_enhanced
from src.analysis.plotters import plot_impulse

# Create custom events for thread-safe communication (consolidated)
UpdateOutputEvent, EVT_UPDATE_OUTPUT = wx.lib.newevent.NewEvent()
UpdatePlotEvent, EVT_UPDATE_PLOT = wx.lib.newevent.NewEvent()


def handle_storing_path(self, event):
    """Handle storage path selection dialog."""
    with wx.DirDialog(self, "Select storage path", defaultPath=self.input1.GetValue(),
                      style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dialog:
        if dialog.ShowModal() == wx.ID_OK:
            self.input1.SetValue(dialog.GetPath())


@run_in_thread
def handle_crawl_data(self, event):
    """Start crawling data based on selected type and symbol list."""
    try:
        select_type = self.combo1.GetValue()
        
        # Determine symbol list and suffix based on selection
        if select_type == "ALL":
            symbols = [s for s in self.get_stock_list() if s != 'TWII'] + ['TWII']
            suffix = ""
        elif select_type == "listed":
            symbols = list(pd.read_excel(self.config['list_file'])['code'].astype(str))
            suffix = ".TW"
        elif select_type == "otc":
            symbols = list(pd.read_excel(self.config['otclist_file'])['code'].astype(str))
            suffix = ".TWO"
        else:
            wx.PostEvent(self, UpdateOutputEvent(message=f"Unknown crawl type: {select_type}\n"))
            return
            
        wx.PostEvent(self, UpdateOutputEvent(message=f"Starting crawl for {select_type} ({len(symbols)} symbols)...\n"))
        self.crawler.crawl(symbols, suffix)
        wx.PostEvent(self, UpdateOutputEvent(message="Crawling completed successfully.\n"))
        
    except Exception as e:
        wx.PostEvent(self, UpdateOutputEvent(message=f"Crawling failed: {e}\n"))


@run_in_thread
def handle_run_backtest(self, event):
    """Run backtest for the selected stock and display summary."""
    try:
        symbol = self.combo_stock.GetValue()
        if symbol == "Select Stock":
            wx.PostEvent(self, UpdateOutputEvent(message="Please select a stock first.\n"))
            return

        wx.PostEvent(self, UpdateOutputEvent(message=f"Running backtest for {symbol}...\n"))
        self.backtester.run([symbol])
        results_df = self.backtester.summarize()
        
        if not results_df.empty:
            for _, row in results_df.iterrows():
                message = f"{row['symbol']}: Profit={row['profit']:.2f}, min_price_change={row['min_price_change']}\n"
                wx.PostEvent(self, UpdateOutputEvent(message=message))
            wx.PostEvent(self, UpdateOutputEvent(message="Backtest completed successfully.\n"))
        else:
            wx.PostEvent(self, UpdateOutputEvent(message=f"No backtest results found for {symbol}.\n"))
            
    except Exception as e:
        wx.PostEvent(self, UpdateOutputEvent(message=f"Backtest failed: {e}\n"))


def load_ready_data(symbol, config, output):
    file_path = os.path.join(config['stk2_dir'], f"{symbol}.txt")
    if not os.path.exists(file_path):
        output.AppendText(f"No data file found for {symbol}.\n")
        return None
    df = load_and_preprocess_data(file_path)
    if df is None or len(df) < 50:
        output.AppendText(f"Insufficient data for Elliott Wave analysis: {len(df) if df is not None else 0} rows\n")
        return None
    return df


def handle_show_elliott_wave(self, event):
    """Enhanced Elliott Wave analysis with comprehensive validation."""
    symbol = self.combo_stock.GetValue()
    if symbol == "Select Stock":
        self.output.AppendText("Please select a stock.\n")
        return
    
    try:
        df = load_ready_data(symbol, self.config, self.output)
        if df is None:
            return

        self.output.AppendText("Running comprehensive Elliott Wave analysis...\n")
        wave_data = detect_elliott_wave_complete(df, column='close')
        
        # Display comprehensive results
        display_wave_analysis_results(self.output, symbol, wave_data)
        
        # Update plot based on chart type
        if self.chart_type.startswith("Candlestick"):
            self._plot_candlestick_elliott_wave_enhanced(df, wave_data, symbol)
        else:
            self._plot_line_elliott_wave_enhanced(df, wave_data, symbol)
            
        self.output.AppendText(f"Enhanced Elliott Wave analysis completed for {symbol}.\n")
        
    except Exception as e:
        self.output.AppendText(f"Error in Elliott Wave analysis for {symbol}: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")


def handle_analyze_current_position(self, event):
    """Enhanced current position analysis with multiple timeframe validation."""
    symbol = self.combo_stock.GetValue()
    if symbol == "Select Stock":
        self.output.AppendText("Please select a stock.\n")
        return
    
    try:
        df = load_ready_data(symbol, self.config, self.output)
        if df is None:
            return
        
        self.output.AppendText("Analyzing current Elliott Wave position...\n")
        
        # Enhanced position analysis
        position_data = detect_current_wave_position_enhanced(df, column='close')
        
        # Display comprehensive position analysis
        display_position_analysis_results(self.output, symbol, position_data)
        
        # Update the plot with enhanced visualization
        update_current_position_plot_enhanced(self, df, position_data, symbol)
        
    except Exception as e:
        self.output.AppendText(f"Error analyzing current position: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")


def handle_chart_type_change(self, event):
    """Handle chart type selection change."""
    self.chart_type = self.combo_chart_type.GetValue()


# ============================================================================
# DISPLAY FUNCTIONS - Consolidated and Enhanced
# ============================================================================

def display_wave_analysis_results(output_widget, symbol: str, wave_data: Dict[str, Any]):
    """Display comprehensive Elliott Wave analysis results."""
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    corrective_wave = wave_data.get('corrective_wave', np.array([]))
    confidence = wave_data.get('confidence', 0.0)
    wave_type = wave_data.get('wave_type', 'unknown')
    validation_details = wave_data.get('validation_details', {})
    
    output_widget.AppendText(f"\n{'='*50}\n")
    output_widget.AppendText(f"ELLIOTT WAVE ANALYSIS RESULTS FOR {symbol}\n")
    output_widget.AppendText(f"{'='*50}\n")
    
    # Basic pattern information
    output_widget.AppendText(f"Pattern Type: {wave_type.replace('_', ' ').title()}\n")
    output_widget.AppendText(f"Overall Confidence: {confidence:.2f} ({get_confidence_description(confidence)})\n")
    output_widget.AppendText(f"Impulse Wave Points: {len(impulse_wave)}\n")
    output_widget.AppendText(f"Corrective Wave Points: {len(corrective_wave)}\n")
    
    # Validation details
    if validation_details and not validation_details.get('error'):
        output_widget.AppendText(f"\nVALIDATION DETAILS:\n")
        output_widget.AppendText(f"{'-'*20}\n")
        
        # Wave 2 retracement
        if 'wave_2_retracement' in validation_details:
            retr = validation_details['wave_2_retracement']
            status = "✓ Valid" if 0.236 <= retr <= 0.786 else "⚠ Marginal" if retr <= 1.0 else "✗ Invalid"
            output_widget.AppendText(f"Wave 2 Retracement: {retr:.1%} {status}\n")
        
        # Wave lengths
        if 'wave_lengths' in validation_details:
            lengths = validation_details['wave_lengths']
            wave_3_strongest = len(lengths) >= 2 and lengths[1] == max(lengths)
            status = "✓ Wave 3 Strongest" if wave_3_strongest else "⚠ Wave 3 Not Strongest"
            output_widget.AppendText(f"Wave 3 Strength: {status}\n")
        
        # Fibonacci relationships
        if 'fibonacci' in validation_details:
            fib_score = validation_details['fibonacci']
            status = "✓ Strong" if fib_score > 0.6 else "~ Moderate" if fib_score > 0.3 else "⚠ Weak"
            output_widget.AppendText(f"Fibonacci Relationships: {fib_score:.2f} {status}\n")
        
        # Volume patterns
        if 'volume' in validation_details:
            vol_score = validation_details['volume']
            status = "✓ Confirming" if vol_score > 0.6 else "~ Neutral" if vol_score > 0.4 else "⚠ Diverging"
            output_widget.AppendText(f"Volume Patterns: {vol_score:.2f} {status}\n")
        
        # Alternation
        if 'alternation' in validation_details:
            alt_score = validation_details['alternation']
            status = "✓ Clear" if alt_score > 0.6 else "~ Present" if alt_score > 0.3 else "⚠ Weak"
            output_widget.AppendText(f"Wave Alternation: {alt_score:.2f} {status}\n")
        
        # Diagonal information
        if validation_details.get('is_diagonal', False):
            output_widget.AppendText(f"\n⚠️  DIAGONAL TRIANGLE DETECTED\n")
            output_widget.AppendText(f"   - Wave 4 overlap with Wave 1 is allowed\n")
            output_widget.AppendText(f"   - Pattern shows converging trend lines\n")
            output_widget.AppendText(f"   - Each wave subdivides into 3-wave structures\n")
    
    # Error information
    elif validation_details.get('error'):
        output_widget.AppendText(f"\nVALIDATION ISSUES:\n")
        output_widget.AppendText(f"{'-'*20}\n")
        error = validation_details['error']
        error_messages = {
            "wave_2_100_percent_retracement": "✗ Wave 2 retraces more than 100% of Wave 1",
            "wave_3_shortest": "✗ Wave 3 is the shortest wave (violates Elliott Wave rule)",
            "wave_4_overlap_violation": "✗ Wave 4 overlaps Wave 1 in non-diagonal pattern",
            "direction_violation": "✗ Wave directions don't follow Elliott Wave rules"
        }
        output_widget.AppendText(f"{error_messages.get(error, f'✗ {error}')}\n")
    
    output_widget.AppendText(f"\n{'='*50}\n")


def display_position_analysis_results(output_widget, symbol: str, position_data: Dict[str, Any]):
    """Display comprehensive current position analysis results."""
    position = position_data.get('position', 'Unknown')
    confidence = position_data.get('confidence', 0.0)
    forecast = position_data.get('forecast', 'No forecast available')
    wave_type = position_data.get('wave_type', 'unknown')
    details = position_data.get('details', {})
    timeframe_details = position_data.get('timeframe_details', {})
    
    output_widget.AppendText(f"\n{'='*60}\n")
    output_widget.AppendText(f"CURRENT ELLIOTT WAVE POSITION ANALYSIS FOR {symbol}\n")
    output_widget.AppendText(f"{'='*60}\n")
    
    # Main position information
    output_widget.AppendText(f"Current Position: {position}\n")
    output_widget.AppendText(f"Confidence Level: {confidence:.2f} ({get_confidence_description(confidence)})\n")
    output_widget.AppendText(f"Wave Type: {wave_type.replace('_', ' ').title()}\n")
    
    # Timeframe analysis details
    if timeframe_details:
        primary_tf = timeframe_details.get('primary_timeframe', 'unknown')
        analysis_count = timeframe_details.get('analysis_count', 0)
        output_widget.AppendText(f"Primary Timeframe: {primary_tf}\n")
        output_widget.AppendText(f"Analyses Completed: {analysis_count}\n")
        
        if 'all_positions' in timeframe_details:
            output_widget.AppendText(f"\nMulti-Timeframe Analysis:\n")
            for pos in timeframe_details['all_positions']:
                output_widget.AppendText(f"  • {pos}\n")
    
    # Detailed information
    if details:
        output_widget.AppendText(f"\nDetailed Analysis:\n")
        output_widget.AppendText(f"{'-'*30}\n")
        
        for key, value in details.items():
            if key == 'error':
                output_widget.AppendText(f"Error: {value}\n")
            elif key == 'wave_count':
                output_widget.AppendText(f"Waves Identified: {value}\n")
            elif key == 'days_since_impulse':
                output_widget.AppendText(f"Days Since Last Impulse: {value}\n")
            elif key == 'days_since_correction':
                output_widget.AppendText(f"Days Since Correction End: {value}\n")
            elif key == 'price_change_pct':
                direction = "↑" if value > 0 else "↓" if value < 0 else "→"
                output_widget.AppendText(f"Price Change: {direction} {abs(value):.1f}%\n")
            elif key == 'corrective_phase':
                output_widget.AppendText(f"Corrective Phase: Wave {value}\n")
            elif key == 'corrective_complete':
                status = "Complete" if value else "In Progress"
                output_widget.AppendText(f"Corrective Status: {status}\n")
    
    # Forecast
    output_widget.AppendText(f"\nFORECAST & IMPLICATIONS:\n")
    output_widget.AppendText(f"{'-'*30}\n")
    output_widget.AppendText(f"{forecast}\n")
    
    # Trading implications based on position
    trading_implications = get_trading_implications(position, confidence)
    if trading_implications:
        output_widget.AppendText(f"\nTRADING IMPLICATIONS:\n")
        output_widget.AppendText(f"{'-'*30}\n")
        output_widget.AppendText(f"{trading_implications}\n")
    
    output_widget.AppendText(f"\n{'='*60}\n")


def get_trading_implications(position: str, confidence: float) -> str:
    """Get trading implications based on the current Elliott Wave position."""
    if confidence < 0.3:
        return "⚠️ Low confidence - avoid making trading decisions based on this analysis."
    
    implications = {
        # Impulse wave positions
        "Impulse Wave 1": "📈 Early trend development. Consider initial positions with tight stops.",
        "Impulse 1": "📈 Early trend development. Consider initial positions with tight stops.",
        "Impulse Wave 2": "⏸️ Corrective pullback in new trend. Look for buying opportunities near support.",
        "Impulse 2": "⏸️ Corrective pullback in new trend. Look for buying opportunities near support.",
        "Impulse Wave 3": "🚀 Strongest wave typically. Strong momentum expected - ride the trend.",
        "Impulse 3": "🚀 Strongest wave typically. Strong momentum expected - ride the trend.",
        "Impulse Wave 4": "⏸️ Corrective phase. Expect sideways/pullback movement. Prepare for final wave 5.",
        "Impulse 4": "⏸️ Corrective phase. Expect sideways/pullback movement. Prepare for final wave 5.",
        "Impulse Wave 5": "⚠️ Final impulse wave. Be cautious - trend reversal approaching. Consider profit-taking.",
        "Impulse 5": "⚠️ Final impulse wave. Be cautious - trend reversal approaching. Consider profit-taking.",
        
        # Corrective wave positions
        "Corrective Wave A": "📉 Beginning of correction. Consider reducing positions or hedging.",
        "Corrective A": "📉 Beginning of correction. Consider reducing positions or hedging.",
        "Corrective Wave B": "⏫ Counter-trend bounce. Opportunity to exit longs or enter shorts.",
        "Corrective B": "⏫ Counter-trend bounce. Opportunity to exit longs or enter shorts.",
        "Corrective Wave C": "📉 Final corrective wave. Completion signals potential trend reversal.",
        "Corrective C": "📉 Final corrective wave. Completion signals potential trend reversal.",
        
        # Transitional positions
        "Post-Corrective": "🔄 New impulse wave likely starting. Look for entry opportunities in new trend direction.",
        "Post-Impulse": "🔄 Major correction expected. Consider profit-taking and defensive positioning.",
        "Transitional": "⏳ Pattern unclear. Wait for clearer signals before major position changes."
    }
    
    # Look for exact or partial matches
    for key, implication in implications.items():
        if key in position:
            return implication
    
    return "🤔 Position unclear - wait for more definitive Elliott Wave structure to develop."


# ============================================================================
# PLOTTING FUNCTIONS - Enhanced and Consolidated
# ============================================================================

def update_current_position_plot_enhanced(self, df: pd.DataFrame, position_data: Dict[str, Any], symbol: str):
    """Enhanced current position plot with comprehensive visualization."""
    try:
        position = position_data.get('position', 'Unknown')
        confidence = position_data.get('confidence', 0.0)
        
        # Get recent data for visualization (last 2 years)
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(years=2)
        recent_df = df[df.index >= start_date].copy()
        
        if recent_df.empty or len(recent_df) < 10:
            create_error_plot(self, "Insufficient data for position visualization", position, confidence)
            return
        
        # Detect Elliott Wave pattern in recent data for context
        context_wave_data = detect_elliott_wave_complete(recent_df, column='close')
        
        if self.chart_type.startswith("Candlestick"):
            create_candlestick_position_plot(self, recent_df, position_data, context_wave_data, symbol)
        else:
            create_line_position_plot(self, recent_df, position_data, context_wave_data, symbol)
        
        self.canvas.draw()
        
    except Exception as e:
        self.output.AppendText(f"Error creating position plot: {e}\n")
        create_error_plot(self, f"Plot error: {str(e)}", position_data.get('position', 'Error'), 0.0)


def create_candlestick_position_plot(self, df: pd.DataFrame, position_data: Dict[str, Any], 
                                   wave_data: Dict[str, Any], symbol: str):
    """Create enhanced candlestick plot with position analysis."""
    try:
        # Determine resampling frequency
        freq_map = {
            "Candlestick (Day)": 'D',
            "Candlestick (Week)": 'W', 
            "Candlestick (Month)": 'M'
        }
        freq = freq_map.get(self.chart_type, 'D')
        
        # Resample data
        df_ohlc = resample_ohlc(df, freq)
        df_ohlc = df_ohlc.dropna()
        
        if df_ohlc.empty:
            self.output.AppendText("No data after resampling for candlestick chart\n")
            create_line_position_plot(self, df, position_data, wave_data, symbol)
            return
        
        # Prepare additional plots for Elliott Wave context
        additional_plots = []
        
        # Add impulse wave markers if available
        impulse_wave = wave_data.get('impulse_wave', np.array([]))
        if len(impulse_wave) > 0:
            impulse_series = map_points_to_ohlc(df, df_ohlc, impulse_wave, 'close')
            if impulse_series.notna().sum() > 0:
                additional_plots.append(
                    mpf.make_addplot(impulse_series, type='scatter', markersize=80, 
                                   marker='^', color='blue', alpha=0.8)
                )
        
        # Add corrective wave markers if available
        corrective_wave = wave_data.get('corrective_wave', np.array([]))
        if len(corrective_wave) > 0:
            corrective_series = map_points_to_ohlc(df, df_ohlc, corrective_wave, 'close')
            if corrective_series.notna().sum() > 0:
                additional_plots.append(
                    mpf.make_addplot(corrective_series, type='scatter', markersize=80,
                                   marker='v', color='magenta', alpha=0.8)
                )
        
        # Highlight current position area
        if len(df_ohlc) > 10:
            highlight_length = min(10, len(df_ohlc) // 4)
            highlight_series = pd.Series(index=df_ohlc.index, dtype=float)
            highlight_series.iloc[-highlight_length:] = df_ohlc['close'].iloc[-highlight_length:]
            
            position_color = get_position_color(position_data.get('position', ''))
            additional_plots.append(
                mpf.make_addplot(highlight_series, type='line', color=position_color,
                               width=4, alpha=0.6, secondary_y=False)
            )
        
        # Configure plot style
        mc = mpf.make_marketcolors(up='green', down='red', edge='inherit',
                                 wick={'up': 'green', 'down': 'red'})
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False,
                                 gridcolor='gray', facecolor='white', figcolor='white')
        
        # Create title
        confidence = position_data.get('confidence', 0.0)
        position = position_data.get('position', 'Unknown')
        title = f"{symbol} - Current Position: {position} (Confidence: {confidence:.2f})"
        
        # Create plot
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
        
        # Add annotations
        ax = axes[0]
        add_position_annotations(ax, df, df_ohlc, position_data, wave_data)
        
        # Update canvas
        self.canvas.figure = fig
        self.ax = ax
        
    except Exception as e:
        self.output.AppendText(f"Candlestick position plot error: {e}\n")
        create_line_position_plot(self, df, position_data, wave_data, symbol)


def create_line_position_plot(self, df: pd.DataFrame, position_data: Dict[str, Any],
                            wave_data: Dict[str, Any], symbol: str):
    """Create enhanced line plot with position analysis."""
    try:
        self.ax.clear()
        
        # Plot price line
        self.ax.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5, alpha=0.8)
        
        # Plot Elliott Wave context if available
        impulse_wave = wave_data.get('impulse_wave', np.array([]))
        if len(impulse_wave) > 0:
            self.ax.plot(df.index[impulse_wave], df['close'].iloc[impulse_wave],
                        'bo-', label='Impulse Wave', markersize=8, linewidth=2, alpha=0.9)
            
            # Label waves
            for i, idx in enumerate(impulse_wave):
                if 0 <= idx < len(df):
                    self.ax.annotate(f'W{i+1}', (df.index[idx], df['close'].iloc[idx]),
                                   xytext=(0, 15), textcoords='offset points',
                                   ha='center', fontweight='bold', color='blue',
                                   bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.8))
        
        corrective_wave = wave_data.get('corrective_wave', np.array([]))
        if len(corrective_wave) > 0:
            self.ax.plot(df.index[corrective_wave], df['close'].iloc[corrective_wave],
                        'mo--', label='Corrective Wave', markersize=8, linewidth=2, alpha=0.9)
            
            # Label corrective waves
            labels = ['A', 'B', 'C', 'D', 'E']
            for i, idx in enumerate(corrective_wave):
                if 0 <= idx < len(df) and i < len(labels):
                    self.ax.annotate(labels[i], (df.index[idx], df['close'].iloc[idx]),
                                   xytext=(0, -20), textcoords='offset points',
                                   ha='center', fontweight='bold', color='magenta',
                                   bbox=dict(boxstyle='round,pad=0.3', fc='lightpink', alpha=0.8))
        
        # Highlight current position area
        position = position_data.get('position', '')
        confidence = position_data.get('confidence', 0.0)
        
        if len(df) > 20:
            highlight_start = max(0, len(df) - 20)
            highlight_color = get_position_color(position)
            self.ax.axvspan(df.index[highlight_start], df.index[-1],
                          alpha=0.2, color=highlight_color, 
                          label=f'Current: {position}')
        
        # Add current position marker
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1]
        marker_color = get_position_color(position)
        self.ax.scatter([current_date], [current_price], s=200, c=marker_color,
                       marker='*', edgecolors='black', linewidth=2, zorder=10,
                       label='Current Position')
        
        # Add forecast annotation
        forecast = position_data.get('forecast', '')
        if forecast:
            forecast_text = forecast[:80] + "..." if len(forecast) > 80 else forecast
            self.ax.text(0.02, 0.98, f"Forecast: {forecast_text}",
                        transform=self.ax.transAxes, fontsize=9,
                        verticalalignment='top', wrap=True,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        # Add confidence indicator
        confidence_color = 'green' if confidence > 0.6 else 'orange' if confidence > 0.3 else 'red'
        self.ax.text(0.98, 0.02, f"Confidence: {confidence:.2f}",
                    transform=self.ax.transAxes, fontsize=10,
                    horizontalalignment='right', color=confidence_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Styling
        title = f"{symbol} - Elliott Wave Position Analysis"
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Price', fontsize=12)
        self.ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))
        self.ax.grid(True, alpha=0.3)
        
        # Fix date labels
        self._fix_date_labels(self.ax, df)
        
    except Exception as e:
        self.output.AppendText(f"Line position plot error: {e}\n")
        create_error_plot(self, f"Line plot error: {str(e)}", position_data.get('position', 'Error'), 0.0)


def create_error_plot(self, error_message: str, position: str, confidence: float):
    """Create a simple error plot when other plotting methods fail."""
    try:
        self.ax.clear()
        self.ax.text(0.5, 0.5, f"Plot Error\n{error_message}\n\nPosition: {position}\nConfidence: {confidence:.2f}",
                    ha='center', va='center', fontsize=12, transform=self.ax.transAxes,
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        self.ax.set_title("Elliott Wave Analysis - Plot Error")
        self.canvas.draw()
    except:
        pass  # If even this fails, just skip


def add_position_annotations(ax, original_df: pd.DataFrame, ohlc_df: pd.DataFrame,
                           position_data: Dict[str, Any], wave_data: Dict[str, Any]):
    """Add comprehensive annotations for position analysis."""
    try:
        position = position_data.get('position', 'Unknown')
        confidence = position_data.get('confidence', 0.0)
        forecast = position_data.get('forecast', '')
        
        # Add position information box
        info_text = f"Position: {position}\nConfidence: {confidence:.2f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        # Add forecast at bottom
        if forecast:
            forecast_short = forecast[:60] + "..." if len(forecast) > 60 else forecast
            ax.text(0.5, 0.02, f"Forecast: {forecast_short}",
                   transform=ax.transAxes, fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
        
        # Highlight current position area
        if len(ohlc_df) > 5:
            highlight_start = max(0, len(ohlc_df) - 5)
            start_date = ohlc_df.index[highlight_start]
            end_date = ohlc_df.index[-1]
            
            y_min = ohlc_df['low'].min() * 0.995
            y_max = ohlc_df['high'].max() * 1.005
            
            rect = patches.Rectangle(
                (mdates.date2num(start_date), y_min),
                mdates.date2num(end_date) - mdates.date2num(start_date),
                y_max - y_min,
                linewidth=2, edgecolor=get_position_color(position),
                facecolor=get_position_color(position), alpha=0.1
            )
            ax.add_patch(rect)
        
    except Exception as e:
        print(f"Error adding position annotations: {e}")


# ============================================================================
# ENHANCED METHODS FOR FRAME CLASS INTEGRATION
# ============================================================================

def _plot_candlestick_elliott_wave_enhanced(self, df, wave_data, symbol):
    """Enhanced candlestick plotting with comprehensive Elliott Wave analysis."""
    create_candlestick_position_plot(
        self, df, {'position': 'Analysis Mode', 'confidence': wave_data.get('confidence', 0.0)}, wave_data, symbol
    )


def _plot_line_elliott_wave_enhanced(self, df: pd.DataFrame, wave_data: Dict[str, Any], symbol: str):
    """Enhanced line plotting for Elliott Wave analysis."""
    self.ax.clear()
    plot_elliott_wave_analysis_enhanced(df, wave_data, column='close', 
                                      title=f"Elliott Wave Analysis for {symbol}", ax=self.ax)
    self._fix_date_labels(self.ax, df)
    self.canvas.draw()


# ============================================================================
# UTILITY FUNCTIONS - Enhanced Error Handling and Logging
# ============================================================================

def safe_execute_with_fallback(primary_func, fallback_func, *args, **kwargs):
    """Execute primary function with fallback on error."""
    try:
        return primary_func(*args, **kwargs)
    except Exception as e:
        print(f"Primary function failed: {e}, trying fallback...")
        try:
            return fallback_func(*args, **kwargs)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise fallback_error


def validate_symbol_selection(symbol: str) -> bool:
    """Validate that a proper symbol is selected."""
    return symbol and symbol != "Select Stock" and symbol.strip() != ""


def validate_data_file(file_path: str) -> bool:
    """Validate that data file exists and is accessible."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def get_safe_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get safe information about a dataframe."""
    if df is None:
        return {"valid": False, "length": 0, "columns": []}
    
    try:
        return {
            "valid": True,
            "length": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df.index.min() if not df.empty else None,
                "end": df.index.max() if not df.empty else None
            },
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        return {"valid": False, "error": str(e), "length": 0}


# ============================================================================
# COMPATIBILITY FUNCTIONS - Maintain backward compatibility
# ============================================================================

# These functions ensure existing code continues to work
def handle_storing_path_legacy(self, event):
    """Legacy wrapper for storing path handler."""
    return handle_storing_path(self, event)


def handle_crawl_data_legacy(self, event):
    """Legacy wrapper for crawl data handler."""
    return handle_crawl_data(self, event)


def handle_run_backtest_legacy(self, event):
    """Legacy wrapper for backtest handler."""
    return handle_run_backtest(self, event)


def handle_show_elliott_wave_legacy(self, event):
    """Legacy wrapper for Elliott Wave handler."""
    return handle_show_elliott_wave(self, event)


def handle_analyze_current_position_legacy(self, event):
    """Legacy wrapper for current position handler."""
    return handle_analyze_current_position(self, event)


# ============================================================================
# MAIN EXPORTS - Functions that should be available when importing this module
# ============================================================================

__all__ = [
    # Event definitions
    'UpdateOutputEvent', 'EVT_UPDATE_OUTPUT', 'UpdatePlotEvent', 'EVT_UPDATE_PLOT',
    
    # Main handlers
    'handle_storing_path', 'handle_crawl_data', 'handle_run_backtest',
    'handle_show_elliott_wave', 'handle_analyze_current_position', 'handle_chart_type_change',
    
    # Display functions
    'display_wave_analysis_results', 'display_position_analysis_results',
    
    # Plotting functions
    'create_candlestick_position_plot', 'create_line_position_plot', 
    'update_current_position_plot_enhanced',
    
    # Enhanced methods for frame integration
    '_plot_candlestick_elliott_wave_enhanced', '_plot_line_elliott_wave_enhanced',
    
    # Utility functions
    'run_in_thread', 'get_trading_implications', 'validate_symbol_selection',
    'validate_data_file', 'safe_execute_with_fallback'
]