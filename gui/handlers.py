import os
import wx
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import threading
from pathlib import Path
from typing import Dict, Any, List
from functools import wraps
import logging

from src.utils.common_utils import (
    load_and_preprocess_data,
    resample_ohlc,
    map_points_to_ohlc,
    get_position_color
)
from src.analysis.core.utils import calculate_base_confidence, validate_data_quality

logger = logging.getLogger(__name__)


def run_in_thread(func):
    """Decorator to run function in a separate thread to avoid blocking GUI."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
    return wrapper


from .constants import CHART_TYPES
import wx.lib.newevent
from src.analysis.core import detect_peaks_troughs_enhanced
from src.analysis.core.impulse import find_elliott_wave_pattern_enhanced, find_elliott_wave_patterns_advanced
from src.analysis.core.position import detect_current_wave_position_enhanced
from src.analysis.plotters.elliott import (
    plot_elliott_wave_analysis, 
    plot_elliott_wave_analysis_enhanced,
    plot_multiple_elliott_patterns_advanced,
    plot_pattern_comparison_chart
)

# Create custom events for thread-safe communication (consolidated)
UpdateOutputEvent, EVT_UPDATE_OUTPUT = wx.lib.newevent.NewEvent()
UpdatePlotEvent, EVT_UPDATE_PLOT = wx.lib.newevent.NewEvent()
EnableButtonsEvent, EVT_ENABLE_BUTTONS = wx.lib.newevent.NewEvent()


def handle_storing_path(self, event):
    """Handle storage path selection dialog."""
    with wx.DirDialog(self, "Select storage path", defaultPath=self.input1.GetValue(),
                      style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dialog:
        if dialog.ShowModal() == wx.ID_OK:
            self.input1.SetValue(dialog.GetPath())


@run_in_thread
def handle_crawl_data(self, event):
    """Start crawling data based on selected type and symbol list."""
    import traceback
    try:
        select_type = self.combo1.GetValue()

        # Determine symbol list based on selection
        if select_type == "ALL":
            symbols = [s for s in self.get_stock_list() if s != 'TWII'] + ['TWII']
        elif select_type == "listed":
            symbols = list(pd.read_excel(self.config['list_file'])['code'].astype(str))
        elif select_type == "otc":
            symbols = list(pd.read_excel(self.config['otclist_file'])['code'].astype(str))
        else:
            wx.PostEvent(self, UpdateOutputEvent(message=f"Unknown crawl type: {select_type}\n"))
            return

        wx.PostEvent(self, UpdateOutputEvent(message=f"Starting crawl for {select_type} ({len(symbols)} symbols)...\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"Date range: {self.config['start_date']} to {self.config['end_date']}\n"))

        # Prepare sets of listed/otc codes to determine suffix per symbol
        try:
            listed_set = set(pd.read_excel(self.config['list_file'])['code'].astype(str))
        except Exception:
            listed_set = set()
        try:
            otc_set = set(pd.read_excel(self.config['otclist_file'])['code'].astype(str))
        except Exception:
            otc_set = set()

        # Crawl with progress feedback
        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                # Determine suffix for this symbol (TWII has no suffix)
                if symbol == 'TWII':
                    per_suffix = ""
                elif symbol in listed_set:
                    per_suffix = ".TW"
                elif symbol in otc_set:
                    per_suffix = ".TWO"
                else:
                    # Fallback heuristic: numeric tickers -> listed
                    per_suffix = ".TW" if str(symbol).strip().isdigit() else ""

                # Call fetch_data with the symbol and per-symbol suffix
                df = self.crawler.fetch_data(symbol,
                                            self.config["start_date"],
                                            self.config["end_date"],
                                            suffix=per_suffix)
                if df is not None and not df.empty:
                    self.crawler.save_data(df, symbol)
                    success_count += 1
                    wx.PostEvent(self, UpdateOutputEvent(
                        message=f"[{i}/{len(symbols)}] [OK] {symbol}{per_suffix}: {len(df)} bars\n"))
                else:
                    fail_count += 1
                    wx.PostEvent(self, UpdateOutputEvent(
                        message=f"[{i}/{len(symbols)}] [FAIL] {symbol}{per_suffix}: No data\n"))

            except Exception as e:
                fail_count += 1
                wx.PostEvent(self, UpdateOutputEvent(
                    message=f"[{i}/{len(symbols)}] [FAIL] {symbol}: {str(e)}\n"))

        wx.PostEvent(self, UpdateOutputEvent(
            message=f"\nCrawling completed: {success_count} succeeded, {fail_count} failed\n"))

        # Re-enable buttons after completion
        wx.PostEvent(self, EnableButtonsEvent(enable=True))

    except Exception as e:
        error_msg = f"Crawling failed: {e}\n{traceback.format_exc()}\n"
        wx.PostEvent(self, UpdateOutputEvent(message=error_msg))
        # Re-enable buttons after error
        wx.PostEvent(self, EnableButtonsEvent(enable=True))


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

        # Re-enable buttons after completion
        wx.PostEvent(self, EnableButtonsEvent(enable=True))

    except Exception as e:
        wx.PostEvent(self, UpdateOutputEvent(message=f"Backtest failed: {e}\n"))
        # Re-enable buttons after error
        wx.PostEvent(self, EnableButtonsEvent(enable=True))


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


@run_in_thread
def handle_show_elliott_wave(self, event):
    """Enhanced Elliott Wave analysis handler with advanced multi-pattern detection"""
    symbol = self.combo_stock.GetValue()
    if symbol == "Select Stock":
        self.output.AppendText("Please select a stock.\n")
        return
    
    try:
        df = load_ready_data(symbol, self.config, self.output)
        if df is None:
            return

        # Determine candlestick_type from chart_type
        chart_type_map = {
            'Candlestick (Day)': 'day',
            'Candlestick (Week)': 'week',
            'Candlestick (Month)': 'month',
            'Line': 'day'
        }
        candlestick_type = chart_type_map.get(self.chart_type, 'day')

        # PERFORMANCE OPTIMIZATION: Only analyze selected timeframe instead of all 3
        # This provides 3-5x speed improvement by avoiding redundant analysis
        # To analyze all timeframes, add a checkbox in GUI or use a separate "Multi-Timeframe Analysis" button
        patterns_data = find_elliott_wave_patterns_advanced(
            df,
            column='close',
            candlestick_types=[candlestick_type],  # Only current timeframe (was: ['day', 'week', 'month'])
            max_patterns_per_timeframe=2,  # Reduced from 3 for faster execution
            pattern_relationships=False  # Skip cross-timeframe analysis for speed (was: True)
        )

        # Display advanced analysis results (thread-safe)
        wx.PostEvent(self, UpdateOutputEvent(message=f"\n{'='*70}\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"ADVANCED MULTI-PATTERN ELLIOTT WAVE ANALYSIS FOR {symbol}\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"{'='*70}\n"))

        # Display results
        display_advanced_multi_pattern_results_threadsafe(self, symbol, patterns_data)

        # Create advanced visualization (must run in main thread)
        wx.CallAfter(self.create_user_friendly_elliott_wave_display, df, patterns_data, symbol)

        # Show pattern summary
        composite_patterns = patterns_data.get('composite_patterns', [])
        wx.PostEvent(self, UpdateOutputEvent(message=f"Advanced analysis completed for {symbol}. Found {len(composite_patterns)} composite pattern(s).\n"))

        # Re-enable buttons after completion
        wx.PostEvent(self, EnableButtonsEvent(enable=True))

    except Exception as e:
        import traceback
        wx.PostEvent(self, UpdateOutputEvent(message=f"Error in advanced Elliott Wave analysis for {symbol}: {e}\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"Traceback: {traceback.format_exc()}\n"))
        # Re-enable buttons after error
        wx.PostEvent(self, EnableButtonsEvent(enable=True))


@run_in_thread
def handle_analyze_current_position(self, event):
    """Handle current position analysis with enhanced detection"""
    try:
        symbol = self.combo_stock.GetValue()
        if not validate_symbol_selection(symbol):
            self.output.AppendText("Please select a valid stock symbol.\n")
            return

        # Load and prepare data
        df = load_ready_data(symbol, self.config, self.output)
        if df is None or len(df) < 50:
            self.output.AppendText("Insufficient data for analysis.\n")
            return

        # Use advanced multi-pattern detection
        patterns_data = find_elliott_wave_patterns_advanced(df, column='close')

        # Get current position with enhanced detection
        position_data = detect_current_wave_position_enhanced(df, patterns_data, column='close')

        # Display results
        display_advanced_position_analysis_results(self.output, symbol, position_data, patterns_data)

        # Update plot
        update_advanced_position_plot(self, df, position_data, patterns_data, symbol)

    except Exception as e:
        self.output.AppendText(f"Error in advanced position analysis: {str(e)}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")


@run_in_thread
def handle_scan_all_stocks(self, event):
    """Scan all stocks for Elliott Wave patterns and populate the listbox"""
    import traceback

    try:
        # Get list of all available stocks
        all_stocks = self.get_stock_list()

        if not all_stocks:
            wx.PostEvent(self, UpdateOutputEvent(message="No stocks found to scan.\n"))
            wx.PostEvent(self, EnableButtonsEvent(enable=True))
            return

        wx.PostEvent(self, UpdateOutputEvent(message=f"\n{'='*70}\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"SCANNING {len(all_stocks)} STOCKS FOR ELLIOTT WAVE PATTERNS\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"{'='*70}\n"))
        wx.PostEvent(self, UpdateOutputEvent(message=f"Chart Type: {self.chart_type}\n"))

        # Determine candlestick_type from chart_type
        chart_type_map = {
            'Candlestick (Day)': 'day',
            'Candlestick (Week)': 'week',
            'Candlestick (Month)': 'month',
            'Line': 'day'
        }
        candlestick_type = chart_type_map.get(self.chart_type, 'day')

        # Store the scanned timeframe so clicked stocks use the same timeframe
        self.scanned_timeframe = candlestick_type

        # Store results
        stocks_with_patterns = []

        for i, symbol in enumerate(all_stocks, 1):
            try:
                # Load data
                file_path = os.path.join(self.config['stk2_dir'], f"{symbol}.txt")
                if not os.path.exists(file_path):
                    continue

                df = load_and_preprocess_data(file_path)
                if df is None or len(df) < 50:
                    continue

                # Run pattern detection
                patterns_data = find_elliott_wave_patterns_advanced(
                    df,
                    column='close',
                    candlestick_types=[candlestick_type],
                    max_patterns_per_timeframe=1,  # Fast scan - only best pattern
                    pattern_relationships=False  # Skip for speed
                )

                # Check if patterns found
                composite_patterns = patterns_data.get('composite_patterns', [])
                best_pattern = patterns_data.get('best_pattern', {})

                if composite_patterns and 'error' not in best_pattern:
                    confidence = best_pattern.get('composite_confidence', 0.0)
                    pattern_group = best_pattern.get('pattern_group', 'unknown')

                    # Only include patterns with reasonable confidence
                    if confidence >= 0.05:  # Minimum threshold
                        stocks_with_patterns.append({
                            'symbol': symbol,
                            'confidence': confidence,
                            'pattern_group': pattern_group,
                            'patterns_count': len(composite_patterns)
                        })

                        status_icon = '✓' if pattern_group == 'primary' else '~'
                        wx.PostEvent(self, UpdateOutputEvent(
                            message=f"[{i}/{len(all_stocks)}] {status_icon} {symbol}: {confidence:.1%} ({pattern_group})\n"
                        ))
                else:
                    wx.PostEvent(self, UpdateOutputEvent(
                        message=f"[{i}/{len(all_stocks)}] ✗ {symbol}: No pattern\n"
                    ))

            except Exception as e:
                wx.PostEvent(self, UpdateOutputEvent(
                    message=f"[{i}/{len(all_stocks)}] ✗ {symbol}: Error - {str(e)}\n"
                ))
                continue

        # Sort by confidence (highest first)
        stocks_with_patterns.sort(key=lambda x: x['confidence'], reverse=True)

        # Save to cache
        try:
            if hasattr(self, 'scan_cache'):
                success = self.scan_cache.save_scan_results(
                    stocks_with_patterns,
                    candlestick_type,
                    self.chart_type,
                    len(all_stocks)
                )
                if success:
                    wx.PostEvent(self, UpdateOutputEvent(message="✓ Scan results saved to cache\n"))
        except Exception as e:
            wx.PostEvent(self, UpdateOutputEvent(message=f"Warning: Could not save cache: {e}\n"))

        # Update listbox (must be done in main thread)
        wx.CallAfter(_update_stocks_listbox, self, stocks_with_patterns)

        # Summary
        wx.PostEvent(self, UpdateOutputEvent(message=f"\n{'='*70}\n"))
        wx.PostEvent(self, UpdateOutputEvent(
            message=f"SCAN COMPLETE: Found {len(stocks_with_patterns)} stocks with Elliott Wave patterns\n"
        ))
        wx.PostEvent(self, UpdateOutputEvent(message=f"{'='*70}\n"))

        if stocks_with_patterns:
            wx.PostEvent(self, UpdateOutputEvent(message="\nTop patterns by confidence:\n"))
            for stock in stocks_with_patterns[:10]:
                icon = '🟢' if stock['pattern_group'] == 'primary' else '🟡'
                wx.PostEvent(self, UpdateOutputEvent(
                    message=f"  {icon} {stock['symbol']}: {stock['confidence']:.1%}\n"
                ))

        # Re-enable buttons
        wx.PostEvent(self, EnableButtonsEvent(enable=True))

    except Exception as e:
        error_msg = f"Scan failed: {e}\n{traceback.format_exc()}\n"
        wx.PostEvent(self, UpdateOutputEvent(message=error_msg))
        wx.PostEvent(self, EnableButtonsEvent(enable=True))


def _update_stocks_listbox(self, stocks_with_patterns):
    """Update the stocks listbox with found patterns (main thread)"""
    try:
        self.stocks_list.Clear()
        for stock in stocks_with_patterns:
            icon = '🟢' if stock['pattern_group'] == 'primary' else '🟡'
            display_text = f"{stock['symbol']} ({stock['confidence']:.0%}) {icon}"
            self.stocks_list.Append(display_text)
    except Exception as e:
        logger.error(f"Error updating listbox: {e}")


def handle_load_scan(self, event):
    """Load previously cached scan results"""
    try:
        if not hasattr(self, 'scan_cache'):
            self.output.AppendText("Error: Scan cache not initialized\n")
            return

        # Load cache
        cache_data = self.scan_cache.load_scan_results()

        if not cache_data:
            self.output.AppendText("No cached scan results found.\n")
            self.output.AppendText("Run 'Scan All Stocks' first to create cache.\n")
            return

        # Extract data
        stocks_with_patterns = cache_data['stocks']
        timeframe = cache_data['timeframe']
        chart_type = cache_data['chart_type']
        timestamp = cache_data['timestamp']
        total_scanned = cache_data.get('total_scanned', 0)

        # Restore scanned timeframe
        self.scanned_timeframe = timeframe

        # Update listbox
        wx.CallAfter(_update_stocks_listbox, self, stocks_with_patterns)

        # Display info
        self.output.AppendText(f"\n{'='*70}\n")
        self.output.AppendText(f"LOADED CACHED SCAN RESULTS\n")
        self.output.AppendText(f"{'='*70}\n")
        self.output.AppendText(f"Cached: {cache_data.get('timestamp', 'Unknown')}\n")
        self.output.AppendText(f"Chart Type: {chart_type}\n")
        self.output.AppendText(f"Timeframe: {timeframe}\n")
        self.output.AppendText(f"Total scanned: {total_scanned}\n")
        self.output.AppendText(f"Patterns found: {len(stocks_with_patterns)}\n")
        self.output.AppendText(f"{'='*70}\n")

        if stocks_with_patterns:
            self.output.AppendText("\nTop patterns by confidence:\n")
            for stock in stocks_with_patterns[:10]:
                icon = '🟢' if stock['pattern_group'] == 'primary' else '🟡'
                self.output.AppendText(f"  {icon} {stock['symbol']}: {stock['confidence']:.1%}\n")

        self.output.AppendText(f"\n✓ Loaded {len(stocks_with_patterns)} stocks from cache\n")
        self.output.AppendText(f"Click any stock to view its pattern\n")

    except Exception as e:
        import traceback
        self.output.AppendText(f"Error loading cache: {e}\n")
        self.output.AppendText(f"{traceback.format_exc()}\n")


def handle_chart_type_change(self, event):
    """Handle chart type selection change."""
    self.chart_type = self.combo_chart_type.GetValue()


# ============================================================================
# ADVANCED DISPLAY FUNCTIONS
# ============================================================================

def display_advanced_multi_pattern_results_threadsafe(frame_self, symbol: str, patterns_data: Dict[str, Any]):
    """Thread-safe version using wx.PostEvent for display"""

    composite_patterns = patterns_data.get('composite_patterns', [])
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    best_pattern = patterns_data.get('best_pattern', {})

    if not composite_patterns:
        wx.PostEvent(frame_self, UpdateOutputEvent(message="❌ No valid Elliott Wave patterns found across timeframes\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message="💡 Try different chart type or check data quality\n"))
        return

    # Best Pattern Summary
    if 'error' not in best_pattern:
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"🎯 BEST COMPOSITE PATTERN\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"{'-'*40}\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Composite Confidence: {best_pattern.get('composite_confidence', 0):.1%}\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Support Count: {best_pattern.get('support_count', 0)}\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Conflict Count: {best_pattern.get('conflict_count', 0)}\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Pattern Group: {best_pattern.get('pattern_group', 'unknown')}\n"))

        base_pattern = best_pattern.get('base_pattern', {})
        if base_pattern:
            pattern = base_pattern.get('pattern', {})
            wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Base Timeframe: {base_pattern.get('timeframe', 'unknown')}\n"))
            wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Base Confidence: {pattern.get('confidence', 0):.1%}\n"))
            wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Pattern Type: {pattern.get('wave_type', 'unknown')}\n"))

    # Pattern Hierarchy
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"\n📊 PATTERN HIERARCHY\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"{'-'*25}\n"))

    primary = pattern_hierarchy.get('primary')
    if primary:
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"🟢 PRIMARY: {primary['timeframe']} ({primary['pattern']['confidence']:.1%})   Confirmations: {primary.get('confirmations', 0)}\n"))

    supporting = pattern_hierarchy.get('supporting', [])
    if supporting:
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"🟡 SUPPORTING: {len(supporting)} patterns\n"))
        for i, sup in enumerate(supporting[:3]):
            wx.PostEvent(frame_self, UpdateOutputEvent(message=f"   {i+1}. {sup['timeframe']} ({sup['pattern']['confidence']:.1%})\n"))

    # Pattern Relationships
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"\n🔗 PATTERN RELATIONSHIPS\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"{'-'*25}\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Overall Alignment Score: {pattern_relationships.get('alignment_score', 0):.1%}\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Confirmations: {len(pattern_relationships.get('confirmations', []))}\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Conflicts: {len(pattern_relationships.get('conflicts', []))}\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"Nested Patterns: {len(pattern_relationships.get('nested_patterns', []))}\n"))

    # Composite Patterns
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"\n🎯 COMPOSITE PATTERNS ({len(composite_patterns)})\n"))
    wx.PostEvent(frame_self, UpdateOutputEvent(message=f"{'-'*35}\n"))

    for i, comp in enumerate(composite_patterns[:3]):
        group_icon = '🟢' if comp.get('pattern_group') == 'primary' else '🟡'
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"{group_icon} Composite #{i+1} ({comp.get('pattern_group', 'unknown').upper()})   Confidence: {comp.get('composite_confidence', 0):.1%}\n"))
        wx.PostEvent(frame_self, UpdateOutputEvent(message=f"   Support: {comp.get('support_count', 0)}, Conflicts: {comp.get('conflict_count', 0)}\n"))
        base = comp.get('base_pattern', {})
        if base:
            wx.PostEvent(frame_self, UpdateOutputEvent(message=f"   Base: {base.get('timeframe', 'unknown')} ({base.get('pattern', {}).get('confidence', 0):.1%})\n"))

def display_advanced_multi_pattern_results(output_widget, symbol: str, patterns_data: Dict[str, Any]):
    """Display comprehensive advanced multi-pattern analysis results (legacy version for non-threaded calls)"""

    composite_patterns = patterns_data.get('composite_patterns', [])
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    best_pattern = patterns_data.get('best_pattern', {})
    
    output_widget.AppendText(f"\n{'='*70}\n")
    output_widget.AppendText(f"ADVANCED MULTI-PATTERN ELLIOTT WAVE ANALYSIS FOR {symbol}\n")
    output_widget.AppendText(f"{'='*70}\n")
    
    if not composite_patterns:
        output_widget.AppendText("❌ No valid Elliott Wave patterns found across timeframes\n")
        output_widget.AppendText("💡 Try different chart type or check data quality\n")
        return
    
    # Best Pattern Summary
    if 'error' not in best_pattern:
        output_widget.AppendText(f"🎯 BEST COMPOSITE PATTERN\n")
        output_widget.AppendText(f"{'-'*40}\n")
        output_widget.AppendText(f"Composite Confidence: {best_pattern.get('composite_confidence', 0):.1%}\n")
        output_widget.AppendText(f"Support Count: {best_pattern.get('support_count', 0)}\n")
        output_widget.AppendText(f"Conflict Count: {best_pattern.get('conflict_count', 0)}\n")
        output_widget.AppendText(f"Pattern Group: {best_pattern.get('pattern_group', 'unknown')}\n")
        
        base_pattern = best_pattern.get('base_pattern', {})
        if base_pattern:
            pattern = base_pattern.get('pattern', {})
            output_widget.AppendText(f"Base Timeframe: {base_pattern.get('timeframe', 'unknown')}\n")
            output_widget.AppendText(f"Base Confidence: {pattern.get('confidence', 0):.1%}\n")
            output_widget.AppendText(f"Pattern Type: {pattern.get('wave_type', 'unknown')}\n")
    
    # Pattern Hierarchy
    output_widget.AppendText(f"\n📊 PATTERN HIERARCHY\n")
    output_widget.AppendText(f"{'-'*25}\n")
    
    primary = pattern_hierarchy.get('primary')
    if primary:
        output_widget.AppendText(f"🟢 PRIMARY: {primary['timeframe']} ({primary['pattern']['confidence']:.1%})\n")
        output_widget.AppendText(f"   Confirmations: {primary.get('confirmations', 0)}\n")
    
    supporting = pattern_hierarchy.get('supporting', [])
    if supporting:
        output_widget.AppendText(f"🟡 SUPPORTING: {len(supporting)} patterns\n")
        for i, sup in enumerate(supporting[:3]):
            output_widget.AppendText(f"   {i+1}. {sup['timeframe']} ({sup['pattern']['confidence']:.1%})\n")
    
    conflicting = pattern_hierarchy.get('conflicting', [])
    if conflicting:
        output_widget.AppendText(f"🔴 CONFLICTING: {len(conflicting)} patterns\n")
        for i, conf in enumerate(conflicting[:3]):
            output_widget.AppendText(f"   {i+1}. {conf['timeframe']} ({conf['pattern']['confidence']:.1%})\n")
    
    independent = pattern_hierarchy.get('independent', [])
    if independent:
        output_widget.AppendText(f"⚪ INDEPENDENT: {len(independent)} patterns\n")
        for i, ind in enumerate(independent[:3]):
            output_widget.AppendText(f"   {i+1}. {ind['timeframe']} ({ind['pattern']['confidence']:.1%})\n")
    
    # Pattern Relationships
    output_widget.AppendText(f"\n🔗 PATTERN RELATIONSHIPS\n")
    output_widget.AppendText(f"{'-'*25}\n")
    
    alignment_score = pattern_relationships.get('alignment_score', 0.0)
    output_widget.AppendText(f"Overall Alignment Score: {alignment_score:.1%}\n")
    
    confirmations = pattern_relationships.get('confirmations', [])
    conflicts = pattern_relationships.get('conflicts', [])
    nested = pattern_relationships.get('nested_patterns', [])
    
    output_widget.AppendText(f"Confirmations: {len(confirmations)}\n")
    output_widget.AppendText(f"Conflicts: {len(conflicts)}\n")
    output_widget.AppendText(f"Nested Patterns: {len(nested)}\n")
    
    # Trading Signals
    trading_signals = pattern_relationships.get('trading_signals', [])
    if trading_signals:
        output_widget.AppendText(f"\n📈 TRADING SIGNALS\n")
        output_widget.AppendText(f"{'-'*20}\n")
        for i, signal in enumerate(trading_signals):
            signal_type = signal['type'].replace('_', ' ').title()
            confidence = signal['confidence']
            reason = signal['reason']
            output_widget.AppendText(f"{i+1}. {signal_type} ({confidence:.1%})\n")
            output_widget.AppendText(f"   {reason}\n")
    
    # Risk Assessments
    risk_assessments = pattern_relationships.get('risk_assessments', [])
    if risk_assessments:
        output_widget.AppendText(f"\n⚠️ RISK ASSESSMENTS\n")
        output_widget.AppendText(f"{'-'*20}\n")
        for i, risk in enumerate(risk_assessments):
            level = risk['level'].title()
            risk_type = risk['type'].replace('_', ' ').title()
            description = risk['description']
            mitigation = risk.get('mitigation', '')
            output_widget.AppendText(f"{i+1}. {level} Risk - {risk_type}\n")
            output_widget.AppendText(f"   {description}\n")
            if mitigation:
                output_widget.AppendText(f"   Mitigation: {mitigation}\n")
    
    # Composite Patterns Summary
    output_widget.AppendText(f"\n🎯 COMPOSITE PATTERNS ({len(composite_patterns)})\n")
    output_widget.AppendText(f"{'-'*35}\n")
    
    for i, composite in enumerate(composite_patterns[:5]):
        confidence = composite.get('composite_confidence', 0)
        support_count = composite.get('support_count', 0)
        conflict_count = composite.get('conflict_count', 0)
        pattern_group = composite.get('pattern_group', 'unknown')
        
        status_icon = "🟢" if pattern_group == 'primary' else "🟡" if pattern_group == 'alternative' else "⚪"
        
        output_widget.AppendText(f"{status_icon} Composite #{i+1} ({pattern_group.upper()})\n")
        output_widget.AppendText(f"   Confidence: {confidence:.1%}\n")
        output_widget.AppendText(f"   Support: {support_count}, Conflicts: {conflict_count}\n")
        
        base_pattern = composite.get('base_pattern', {})
        if base_pattern:
            pattern = base_pattern.get('pattern', {})
            output_widget.AppendText(f"   Base: {base_pattern.get('timeframe', 'unknown')} ({pattern.get('confidence', 0):.1%})\n")


def display_advanced_position_analysis_results(output_widget, symbol: str, 
                                             position_data: Dict[str, Any], 
                                             patterns_data: Dict[str, Any]):
    """Display advanced position analysis results with multi-pattern context"""
    
    output_widget.AppendText(f"\n{'='*60}\n")
    output_widget.AppendText(f"ADVANCED POSITION ANALYSIS FOR {symbol}\n")
    output_widget.AppendText(f"{'='*60}\n")
    
    # Basic position info
    position = position_data.get('position', 'unknown')
    confidence = position_data.get('confidence', 0.0)
    
    output_widget.AppendText(f"Current Position: {position.replace('_', ' ').title()}\n")
    output_widget.AppendText(f"Position Confidence: {confidence:.1%}\n")
    
    # Multi-pattern context
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    
    if pattern_hierarchy.get('primary'):
        primary = pattern_hierarchy['primary']
        output_widget.AppendText(f"Primary Pattern: {primary['timeframe']} ({primary['pattern']['confidence']:.1%})\n")
    
    alignment_score = pattern_relationships.get('alignment_score', 0.0)
    output_widget.AppendText(f"Pattern Alignment: {alignment_score:.1%}\n")
    
    # Trading implications with multi-pattern context
    implications = get_advanced_trading_implications(position, confidence, patterns_data)
    output_widget.AppendText(f"\n📊 TRADING IMPLICATIONS\n")
    output_widget.AppendText(f"{'-'*25}\n")
    output_widget.AppendText(implications)


def get_advanced_trading_implications(position: str, confidence: float, 
                                    patterns_data: Dict[str, Any]) -> str:
    """Get advanced trading implications considering multi-pattern context"""
    
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    trading_signals = pattern_relationships.get('trading_signals', [])
    risk_assessments = pattern_relationships.get('risk_assessments', [])
    
    implications = []
    
    # Base position implications
    if position == 'wave_1':
        implications.append("🟢 Early trend formation - Consider long position")
        implications.append("📈 Set stop loss below recent low")
    elif position == 'wave_2':
        implications.append("🟡 Retracement phase - Wait for completion")
        implications.append("📊 Look for Fibonacci retracement levels")
    elif position == 'wave_3':
        implications.append("🟢 Strong momentum phase - Strong buy signal")
        implications.append("📈 This is typically the strongest wave")
    elif position == 'wave_4':
        implications.append("🟡 Consolidation phase - Partial profit taking")
        implications.append("📊 Prepare for final wave")
    elif position == 'wave_5':
        implications.append("🟠 Final wave - Consider exit strategies")
        implications.append("⚠️ Watch for divergence signals")
    else:
        implications.append("⚪ Position unclear - Use additional confirmation")
    
    # Multi-pattern context
    if pattern_hierarchy.get('primary'):
        primary = pattern_hierarchy['primary']
        implications.append(f"🎯 Primary pattern: {primary['timeframe']} timeframe")
    
    supporting_count = len(pattern_hierarchy.get('supporting', []))
    if supporting_count > 0:
        implications.append(f"✅ {supporting_count} supporting patterns confirm trend")
    
    conflicting_count = len(pattern_hierarchy.get('conflicting', []))
    if conflicting_count > 0:
        implications.append(f"⚠️ {conflicting_count} conflicting patterns - exercise caution")
    
    # Trading signals
    if trading_signals:
        for signal in trading_signals[:2]:
            signal_type = signal['type'].replace('_', ' ').title()
            implications.append(f"📈 Signal: {signal_type} ({signal['confidence']:.1%})")
    
    # Risk warnings
    high_risks = [r for r in risk_assessments if r['level'] == 'high']
    if high_risks:
        implications.append(f"🚨 {len(high_risks)} high-risk factors detected")
    
    return "\n".join(implications)


def plot_pattern_statistics(ax, patterns_data: Dict[str, Any]):
    """Plot pattern statistics and metrics"""
    
    composite_patterns = patterns_data.get('composite_patterns', [])
    pattern_relationships = patterns_data.get('pattern_relationships', {})
    
    if not composite_patterns:
        ax.text(0.5, 0.5, 'No patterns found', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        return
    
    # Prepare data for bar chart
    labels = []
    confidences = []
    support_counts = []
    conflict_counts = []
    
    for i, composite in enumerate(composite_patterns[:5]):
        labels.append(f"Pattern {i+1}")
        confidences.append(composite.get('composite_confidence', 0))
        support_counts.append(composite.get('support_count', 0))
        conflict_counts.append(composite.get('conflict_count', 0))
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Create grouped bar chart
    ax.bar(x - width, confidences, width, label='Confidence', color='blue', alpha=0.7)
    ax.bar(x, support_counts, width, label='Support', color='green', alpha=0.7)
    ax.bar(x + width, conflict_counts, width, label='Conflicts', color='red', alpha=0.7)
    
    ax.set_xlabel('Patterns')
    ax.set_ylabel('Count/Score')
    ax.set_title('Pattern Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)


def update_advanced_position_plot(self, df: pd.DataFrame, 
                                position_data: Dict[str, Any], 
                                patterns_data: Dict[str, Any], 
                                symbol: str):
    """Update plot with advanced position analysis"""
    try:
        # Clear the canvas
        self.canvas.figure.clf()
        
        # Create subplot layout
        fig = self.canvas.figure
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # Main chart with position
        ax_main = fig.add_subplot(gs[0])
        
        # Plot price and patterns
        plot_multiple_elliott_patterns_advanced(
            df, patterns_data, column='close',
            title=f"{symbol} - Current Position Analysis",
            ax=ax_main, show_relationships=True, show_hierarchy=False
        )
        
        # Add position annotation
        position = position_data.get('position', 'unknown')
        confidence = position_data.get('confidence', 0.0)
        
        # Find current price position
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1]
        
        ax_main.annotate(f"Position: {position.replace('_', ' ').title()}\nConfidence: {confidence:.1%}",
                        xy=(current_date, current_price),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='black'))
        
        # Position timeline subplot
        ax_timeline = fig.add_subplot(gs[1])
        plot_position_timeline(ax_timeline, patterns_data, position_data)
        
        fig.tight_layout()
        self.canvas.draw()
        
    except Exception as e:
        self.output.AppendText(f"Error in advanced position plot: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")


def plot_position_timeline(ax, patterns_data: Dict[str, Any], position_data: Dict[str, Any]):
    """Plot position timeline showing wave progression"""
    
    pattern_hierarchy = patterns_data.get('pattern_hierarchy', {})
    primary = pattern_hierarchy.get('primary')
    
    if not primary:
        ax.text(0.5, 0.5, 'No primary pattern found', ha='center', va='center', 
               transform=ax.transAxes, fontsize=10)
        return
    
    pattern = primary['pattern']
    points = pattern.get('points', [])
    
    if len(points) < 5:
        ax.text(0.5, 0.5, 'Insufficient wave points', ha='center', va='center', 
               transform=ax.transAxes, fontsize=10)
        return
    
    # Create timeline
    wave_labels = ['Start', 'Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
    wave_positions = np.arange(len(points))
    
    # Color code based on current position
    position = position_data.get('position', 'unknown')
    colors = []
    
    for i, label in enumerate(wave_labels[:len(points)]):
        if position in label.lower() or (position == 'wave_1' and i == 1) or \
           (position == 'wave_2' and i == 2) or (position == 'wave_3' and i == 3) or \
           (position == 'wave_4' and i == 4) or (position == 'wave_5' and i == 5):
            colors.append('red')  # Current position
        else:
            colors.append('lightblue')  # Other positions
    
    # Plot timeline
    bars = ax.bar(wave_positions, [1] * len(points), color=colors, alpha=0.7)
    
    # Add labels
    ax.set_xticks(wave_positions)
    ax.set_xticklabels(wave_labels[:len(points)], rotation=45)
    ax.set_ylabel('Wave Progress')
    ax.set_title('Elliott Wave Timeline')
    ax.grid(True, alpha=0.3)
    
    # Highlight current position
    if position != 'unknown':
        ax.set_title(f'Current Position: {position.replace("_", " ").title()}', 
                    fontweight='bold', color='red')


# ============================================================================
# ENHANCED METHODS FOR FRAME CLASS INTEGRATION
# ============================================================================

def _plot_candlestick_elliott_wave_enhanced(self, df, wave_data, symbol):
    """Enhanced candlestick plotting with comprehensive Elliott Wave analysis."""
    # Use the existing enhanced plotting function
    self.ax.clear()
    plot_elliott_wave_analysis_enhanced(
        df, wave_data, column='close',
        title=f"Elliott Wave Analysis for {symbol}", ax=self.ax
    )
    self._fix_date_labels(self.ax, df)
    self.canvas.draw()


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

def validate_symbol_selection(symbol: str) -> bool:
    """Validate that a proper symbol is selected."""
    return symbol and symbol != "Select Stock" and symbol.strip() != ""



# ============================================================================
# MAIN EXPORTS - Functions that should be available when importing this module
# ============================================================================

__all__ = [
    # Event definitions
    'UpdateOutputEvent', 'EVT_UPDATE_OUTPUT', 'UpdatePlotEvent', 'EVT_UPDATE_PLOT',

    # Main handlers
    'handle_storing_path', 'handle_crawl_data', 'handle_run_backtest',
    'handle_show_elliott_wave', 'handle_analyze_current_position', 'handle_chart_type_change',
    'handle_scan_all_stocks', 'handle_load_scan', '_update_stocks_listbox',

    # Display functions
    'display_wave_analysis_results', 'display_position_analysis_results',
    'display_enhanced_wave_analysis_results_user_friendly',

    # Plotting functions
    'update_current_position_plot_enhanced',

    # Enhanced methods for frame integration
    '_plot_candlestick_elliott_wave_enhanced', '_plot_line_elliott_wave_enhanced',

    # Utility functions
    'run_in_thread', 'get_trading_implications', 'validate_symbol_selection'
]

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

def _plot_line_elliott_wave_multiple(self, df, wave_data, symbol):
    """Plot line chart with multiple Elliott Wave patterns"""
    try:
        # Clear the canvas and create a new axes
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        
        # Plot price line
        ax.plot(df.index, df['close'], color='black', linewidth=1, alpha=0.6, label='Price')
        
        # Plot multiple patterns
        multiple_patterns = wave_data.get('multiple_patterns', [])
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':', '-']
        
        for i, pattern in enumerate(multiple_patterns[:5]):
            wave_points = pattern['points']
            if len(wave_points) >= 2:
                color = colors[i % len(colors)]
                style = line_styles[i % len(line_styles)]
                alpha = 1.0 if i == 0 else 0.7
                linewidth = 3 if i == 0 else 2
                
                dates = df.index[wave_points]
                prices = df['close'].iloc[wave_points]
                
                ax.plot(dates, prices, color=color, linestyle=style,
                       linewidth=linewidth, alpha=alpha, marker='o', markersize=6,
                       label=f"Pattern {i+1} ({pattern['confidence']:.1%})")
                
                # Add wave numbers for primary pattern
                if i == 0:
                    for j, (date, price) in enumerate(zip(dates, prices)):
                        ax.annotate(f"{j+1}", (date, price),
                                  xytext=(5, 10), textcoords='offset points',
                                  fontsize=10, fontweight='bold', color=color)
        
        ax.set_title(f"{symbol} - Multiple Elliott Wave Patterns", fontsize=14, fontweight='bold')
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Update canvas
        self.canvas.draw()
        
    except Exception as e:
        self.output.AppendText(f"Error in multiple pattern line plot: {e}\n")
        import traceback
        self.output.AppendText(f"Traceback: {traceback.format_exc()}\n")
        # Fallback to single pattern plot
        self._plot_line_elliott_wave_enhanced(df, wave_data, symbol)

def display_enhanced_wave_analysis_results(output_widget, symbol: str, wave_data: Dict[str, Any]):
    """CORRECTED: Enhanced display function showing multiple patterns with trading relevance"""
    
    multiple_patterns = wave_data.get('multiple_patterns', [])
    primary_confidence = wave_data.get('confidence', 0.0)
    primary_wave_type = wave_data.get('wave_type', 'unknown')
    
    output_widget.AppendText(f"\n{'='*60}\n")
    output_widget.AppendText(f"CORRECTED ELLIOTT WAVE ANALYSIS FOR {symbol}\n")
    output_widget.AppendText(f"{'='*60}\n")
    
    if not multiple_patterns:
        output_widget.AppendText("❌ No valid Elliott Wave patterns found in any timeframe\n")
        output_widget.AppendText("💡 Try different chart type or check data quality\n")
        return
    
    # Primary Pattern Summary
    output_widget.AppendText(f"🎯 PRIMARY PATTERN (Highest Relevance)\n")
    output_widget.AppendText(f"{'-'*40}\n")
    output_widget.AppendText(f"Type: {primary_wave_type.replace('_', ' ').title()}\n")
    output_widget.AppendText(f"Confidence: {primary_confidence:.1%}\n")
    
    primary = multiple_patterns[0]
    output_widget.AppendText(f"Timeframe: {primary['time_frame'].title()}\n")
    output_widget.AppendText(f"Recency Score: {primary['recency_score']:.1%}\n")
    output_widget.AppendText(f"Data Range: {primary['data_range_used']}\n")
    output_widget.AppendText(f"Period: {primary['start_date'].strftime('%Y-%m-%d')} to {primary['end_date'].strftime('%Y-%m-%d')}\n")
    output_widget.AppendText(f"Price Range: ${primary['price_range'][0]:.2f} - ${primary['price_range'][1]:.2f}\n")
    
    # All Patterns Overview
    output_widget.AppendText(f"\n📊 ALL TIMEFRAME PATTERNS ({len(multiple_patterns)} found)\n")
    output_widget.AppendText(f"{'-'*50}\n")
    
    for i, pattern in enumerate(multiple_patterns, 1):
        status_icon = "🟢" if i == 1 else "🟡" if pattern['confidence'] > 0.4 else "🔴"
        relevance = "PRIMARY" if i == 1 else "SECONDARY" if i <= 3 else "ALTERNATIVE"
        
        output_widget.AppendText(f"{status_icon} Pattern #{i} ({relevance})\n")
        output_widget.AppendText(f"   Timeframe: {pattern['time_frame'].title()}\n")
        output_widget.AppendText(f"   Data Range: {pattern['data_range_used']}\n")
        output_widget.AppendText(f"   Confidence: {pattern['confidence']:.1%}\n")
        output_widget.AppendText(f"   Recency: {pattern['recency_score']:.1%}\n")
        output_widget.AppendText(f"   Type: {pattern['wave_type'].replace('_', ' ').title()}\n")
        output_widget.AppendText(f"   Period: {pattern['start_date'].strftime('%Y-%m-%d')} to {pattern['end_date'].strftime('%Y-%m-%d')}\n")
        
        # Relevance indicator
        if pattern['recency_score'] > 0.8:
            output_widget.AppendText(f"   🔥 VERY CURRENT - High trading relevance\n")
        elif pattern['recency_score'] > 0.6:
            output_widget.AppendText(f"   📈 RECENT - Good trading relevance\n")
        elif pattern['recency_score'] > 0.4:
            output_widget.AppendText(f"   📊 MODERATE - Some trading relevance\n")
        else:
            output_widget.AppendText(f"   📉 HISTORICAL - Limited current relevance\n")
        
        output_widget.AppendText(f"\n")
    
    # Timeframe Analysis
    timeframes = set(p['time_frame'] for p in multiple_patterns)
    output_widget.AppendText(f"🔍 TIMEFRAME ANALYSIS\n")
    output_widget.AppendText(f"{'-'*25}\n")
    output_widget.AppendText(f"Timeframes analyzed: {', '.join(timeframes)}\n")
    
    recent_patterns = [p for p in multiple_patterns if p['time_frame'] in ['recent', 'medium']]
    if recent_patterns:
        output_widget.AppendText(f"✅ {len(recent_patterns)} current/recent patterns found\n")
        avg_confidence = np.mean([p['confidence'] for p in recent_patterns])
        
        if avg_confidence > 0.6:
            output_widget.AppendText(f"🎯 HIGH reliability - Strong pattern confirmation\n")
        elif avg_confidence > 0.4:
            output_widget.AppendText(f"⚖️ MODERATE reliability - Use additional confirmation\n")
        else:
            output_widget.AppendText(f"⚠️ LOW reliability - Exercise caution\n")
    else:
        output_widget.AppendText(f"⚠️ No current patterns - Focus on longer-term analysis\n")
    
    output_widget.AppendText(f"\n⚡ TRADING RECOMMENDATIONS\n")
    output_widget.AppendText(f"{'-'*30}\n")
    
    if primary_confidence > 0.5 and primary['recency_score'] > 0.6:
        output_widget.AppendText(f"1. 🎯 Focus on {primary['time_frame'].upper()} timeframe pattern\n")
        output_widget.AppendText(f"2. 📊 Use {len(multiple_patterns)-1} other patterns for confirmation\n")
        output_widget.AppendText(f"3. ⏰ Monitor for pattern completion/invalidation\n")
    else:
        output_widget.AppendText(f"1. ⏳ Wait for higher confidence patterns to develop\n")
        output_widget.AppendText(f"2. 📈 Monitor multiple timeframes for alignment\n")
        output_widget.AppendText(f"3. 🔍 Look for additional technical confirmation\n")
    
    output_widget.AppendText(f"\n{'='*60}\n")