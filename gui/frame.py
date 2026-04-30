import wx
import wx.adv
import wx.html
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
from src.utils.log_setup import setup_logging
from src.analysis.plotters.elliott import plot_elliott_wave_analysis, plot_elliott_wave_analysis_enhanced
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from gui.constants import WEBLIST, CHART_TYPES
from gui.handlers import (
    handle_storing_path, handle_crawl_data, handle_run_backtest, handle_show_elliott_wave, handle_analyze_current_position, handle_chart_type_change,
    EnableButtonsEvent, EVT_ENABLE_BUTTONS,
    UpdateOutputEvent, EVT_UPDATE_OUTPUT,
    UpdatePlotEvent, EVT_UPDATE_PLOT,
)
from src.utils.common_utils import resample_ohlc, map_points_to_ohlc
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CompatListCtrl(wx.ListCtrl):
    """wx.ListCtrl subclass providing wx.ListBox-compatible Clear/Append methods
    so that handlers.py (which cannot be modified) keeps working."""

    def Clear(self):
        """ListBox-compatible: delete all items."""
        self.DeleteAllItems()

    def Append(self, text):
        """ListBox-compatible: append a single display string.
        The text is placed in column 0; additional columns are left empty
        so callers can fill them via SetItem if desired."""
        idx = self.GetItemCount()
        self.InsertItem(idx, text)

    def GetStringSelection(self):
        """ListBox-compatible: return text from column 0 of the focused item."""
        sel = self.GetFirstSelected()
        if sel == -1:
            return ""
        return self.GetItemText(sel, 0)


# Menu item IDs
ID_SCAN = wx.NewIdRef()
ID_WAVE = wx.NewIdRef()
ID_BACKTEST = wx.NewIdRef()
ID_POSITION = wx.NewIdRef()
ID_LOAD_SCAN = wx.NewIdRef()
ID_EXPORT = wx.NewIdRef()
ID_FETCH_ALL = wx.NewIdRef()
ID_FETCH_US = wx.NewIdRef()
ID_FETCH_TW = wx.NewIdRef()
ID_VALIDATE = wx.NewIdRef()
ID_ABOUT = wx.NewIdRef()
ID_SHORTCUTS = wx.NewIdRef()


class MyFrame(wx.Frame):
    def __init__(self):
        screen_size = wx.DisplaySize()
        default_width = max(int(screen_size[0] * 0.85), 900)
        default_height = max(int(screen_size[1] * 0.85), 600)
        super().__init__(
            None,
            title="Fintech-sys \u2014 Elliott Wave Trading System",
            size=(default_width, default_height),
        )
        self.SetMinSize((900, 600))

        self.root = os.getcwd()
        self.logger = setup_logging()
        self.config = load_config()
        self.crawler = YahooFinanceCrawler(self.config)
        self.backtester = Backtester(self.config)
        self.multi_path_build()

        # Persistent state
        self.chart_type = "Line"
        self.scanned_timeframe = None
        self.buttons_enabled = True
        self.all_scan_items = []  # unfiltered scan data for search filtering

        # --- Menu bar ---
        self._create_menu_bar()

        # --- Status bar ---
        self._create_status_bar()

        # --- Main panel (holds toolbar + splitter) ---
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Toolbar row ---
        toolbar_panel = wx.Panel(self.panel)
        tb_sizer = wx.BoxSizer(wx.HORIZONTAL)

        lbl_target = wx.StaticText(toolbar_panel, label=" Target:")
        lbl_target.SetFont(lbl_target.GetFont().Bold())
        lbl_target.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        tb_sizer.Add(lbl_target, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        self.combo1 = wx.ComboBox(
            toolbar_panel, choices=WEBLIST, value="ALL", style=wx.CB_READONLY,
        )
        tb_sizer.Add(self.combo1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)

        tb_sizer.AddSpacer(10)
        lbl_stock = wx.StaticText(toolbar_panel, label=" Stock:")
        lbl_stock.SetFont(lbl_stock.GetFont().Bold())
        lbl_stock.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        tb_sizer.Add(lbl_stock, 0, wx.ALIGN_CENTER_VERTICAL)
        self.combo_stock = wx.ComboBox(
            toolbar_panel, choices=self.get_stock_list(),
            value="Select Stock", style=wx.CB_READONLY, size=(200, -1),
        )
        tb_sizer.Add(self.combo_stock, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)

        tb_sizer.AddStretchSpacer()
        lbl_chart = wx.StaticText(toolbar_panel, label=" Chart:")
        lbl_chart.SetFont(lbl_chart.GetFont().Bold())
        lbl_chart.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT))
        tb_sizer.Add(lbl_chart, 0, wx.ALIGN_CENTER_VERTICAL)
        self.combo_chart_type = wx.ComboBox(
            toolbar_panel, choices=CHART_TYPES, value="Line",
            style=wx.CB_READONLY, size=(180, -1),
        )
        self.combo_chart_type.Bind(
            wx.EVT_COMBOBOX, lambda event: handle_chart_type_change(self, event),
        )
        tb_sizer.Add(self.combo_chart_type, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)

        toolbar_panel.SetSizer(tb_sizer)
        main_sizer.Add(toolbar_panel, 0, wx.EXPAND | wx.BOTTOM, 2)

        # --- Action buttons row (use ScrolledWindow to guarantee visibility) ---
        btn_panel = wx.Panel(self.panel)
        btn_panel.SetBackgroundColour(wx.Colour(245, 245, 245))  # Light gray for contrast
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_style = wx.BU_EXACTFIT
        btn_names = [
            ("Crawl Data",        self.safe_handle_crawl_data),
            ("Scan All Stocks",   self.safe_handle_scan_all_stocks),
            ("Show Elliott Wave", self.safe_handle_show_elliott_wave),
            ("Analyze Position",  self.safe_handle_analyze_current_position),
            ("Run Backtest",      self.safe_handle_run_backtest),
            ("Load Last Scan",    self.safe_handle_load_scan),
        ]
        self._action_buttons = []
        for label, handler in btn_names:
            btn = wx.Button(btn_panel, label=label, style=btn_style, size=(-1, 32))
            btn.Bind(wx.EVT_BUTTON, handler)
            btn_sizer.Add(btn, 1, wx.EXPAND | wx.ALL, 3)
            self._action_buttons.append(btn)

        # Assign named references for enable_buttons compat
        (self.crawl_button, self.scan_button, self.plot_button,
         self.current_pos_button, self.backtest_button, self.load_scan_button) = self._action_buttons

        btn_panel.SetSizer(btn_sizer)
        main_sizer.Add(btn_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)

        # --- Splitter ---
        self.splitter = wx.SplitterWindow(
            self.panel, style=wx.SP_LIVE_UPDATE | wx.SP_3D,
        )
        self.splitter.SetMinimumPaneSize(250)

        # ---- Left panel ----
        left_panel = wx.Panel(self.splitter)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Filter row ---
        filter_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.search_ctrl = wx.SearchCtrl(left_panel, size=(-1, -1))
        self.search_ctrl.SetDescriptiveText("Filter stocks...")
        self.search_ctrl.ShowCancelButton(True)
        self.search_ctrl.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.on_search)
        self.search_ctrl.Bind(wx.EVT_TEXT, self.on_search)
        filter_sizer.Add(self.search_ctrl, 1, wx.EXPAND | wx.RIGHT, 4)

        self.action_filter = wx.Choice(left_panel,
                                       choices=["All", "BUY", "WATCH", "EXIT", "HOLD"])
        self.action_filter.SetSelection(0)
        self.action_filter.Bind(wx.EVT_CHOICE, self._on_action_filter)
        filter_sizer.Add(self.action_filter, 0, wx.ALIGN_CENTER_VERTICAL)

        left_sizer.Add(filter_sizer, 0, wx.EXPAND | wx.ALL, 4)

        # --- Stock list ---
        self.stocks_list = CompatListCtrl(
            left_panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL,
        )
        self.stocks_list.InsertColumn(0, "Symbol", width=70)
        self.stocks_list.InsertColumn(1, "Score", width=45)
        self.stocks_list.InsertColumn(2, "Action", width=60)
        self.stocks_list.InsertColumn(3, "Conf%", width=50)
        self.stocks_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_stock_selected)
        left_sizer.Add(self.stocks_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

        # --- Trade plan panel ---
        self.trade_panel = wx.Panel(left_panel)
        self.trade_panel.SetBackgroundColour(wx.Colour(250, 250, 245))
        tp_sizer = wx.BoxSizer(wx.VERTICAL)

        self.trade_header = wx.StaticText(self.trade_panel, label="Select a stock to see trade plan")
        header_font = self.trade_header.GetFont()
        header_font.SetPointSize(header_font.GetPointSize() + 1)
        header_font = header_font.Bold()
        self.trade_header.SetFont(header_font)
        tp_sizer.Add(self.trade_header, 0, wx.EXPAND | wx.ALL, 6)

        self.trade_details = wx.StaticText(self.trade_panel, label="")
        self.trade_details.SetForegroundColour(wx.Colour(60, 60, 60))
        tp_sizer.Add(self.trade_details, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.trade_panel.SetSizer(tp_sizer)
        left_sizer.Add(self.trade_panel, 0, wx.EXPAND | wx.ALL, 4)

        left_panel.SetSizer(left_sizer)

        # ---- Right panel (notebook) ----
        right_panel = wx.Panel(self.splitter)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        self.notebook = wx.Notebook(right_panel)

        # Tab 0: Dashboard (HTML)
        self.dashboard_panel = wx.Panel(self.notebook)
        dash_sizer = wx.BoxSizer(wx.VERTICAL)
        self.dashboard_html = wx.html.HtmlWindow(
            self.dashboard_panel,
            style=wx.html.HW_SCROLLBAR_AUTO,
        )
        self.dashboard_html.SetPage(
            "<html><body style='font-family:sans-serif;font-size:13px;padding:12px;'>"
            "<h2>Welcome to Fintech-sys</h2>"
            "<p>Press <b>Ctrl+S</b> or use <b>Analysis &gt; Scan All Stocks</b> to begin.</p>"
            "<p><b>Quick Start:</b></p>"
            "<ol>"
            "<li>Scan stocks to find candidates</li>"
            "<li>Click a stock to view Elliott Wave chart</li>"
            "<li>Analyze position for trading signals</li>"
            "</ol>"
            "</body></html>"
        )
        dash_sizer.Add(self.dashboard_html, 1, wx.EXPAND)
        self.dashboard_panel.SetSizer(dash_sizer)
        self.notebook.AddPage(self.dashboard_panel, "Dashboard")

        # Tab 1: Chart
        self.chart_panel = wx.Panel(self.notebook)
        chart_sizer = wx.BoxSizer(wx.VERTICAL)
        fig_w = max(6, default_width / 100)
        fig_h = max(4, default_height / 200)
        self.figure, self.ax = plt.subplots(figsize=(fig_w, fig_h))
        self.canvas = FigureCanvas(self.chart_panel, -1, self.figure)
        chart_sizer.Add(self.canvas, 1, wx.EXPAND)
        self.chart_panel.SetSizer(chart_sizer)
        self.notebook.AddPage(self.chart_panel, "Chart")

        # Tab 2: Log
        self.log_panel = wx.Panel(self.notebook)
        log_sizer = wx.BoxSizer(wx.VERTICAL)
        self.output = wx.TextCtrl(
            self.log_panel, style=wx.TE_MULTILINE | wx.TE_READONLY,
        )
        log_sizer.Add(self.output, 1, wx.EXPAND)
        self.log_panel.SetSizer(log_sizer)
        self.notebook.AddPage(self.log_panel, "Log")

        right_sizer.Add(self.notebook, 1, wx.EXPAND)
        right_panel.SetSizer(right_sizer)

        # Split
        self.splitter.SplitVertically(left_panel, right_panel, 280)
        main_sizer.Add(self.splitter, 1, wx.EXPAND)

        self.panel.SetSizer(main_sizer)

        # --- Accelerator table ---
        accel_tbl = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('S'), ID_SCAN),
            (wx.ACCEL_CTRL, ord('E'), ID_WAVE),
            (wx.ACCEL_CTRL, ord('B'), ID_BACKTEST),
            (wx.ACCEL_CTRL, ord('P'), ID_POSITION),
            (wx.ACCEL_CTRL, ord('L'), ID_LOAD_SCAN),
            (wx.ACCEL_CTRL, ord('Q'), wx.ID_EXIT),
        ])
        self.SetAcceleratorTable(accel_tbl)

        # --- Event bindings ---
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(EVT_UPDATE_OUTPUT, self.on_update_output)
        self.Bind(EVT_UPDATE_PLOT, self.on_update_plot)
        self.Bind(EVT_ENABLE_BUTTONS, self.on_enable_buttons)

        # Initialize scan cache
        from src.utils.scan_cache import ScanCache
        self.scan_cache = ScanCache()

        # Auto-load last scan on startup
        wx.CallAfter(self.auto_load_last_scan)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = wx.MenuBar()

        # -- File --
        file_menu = wx.Menu()
        file_menu.Append(ID_EXPORT, "Export Results\tCtrl+Shift+E")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "Quit\tCtrl+Q")
        menu_bar.Append(file_menu, "&File")

        # -- Analysis --
        analysis_menu = wx.Menu()
        analysis_menu.Append(ID_SCAN, "Scan All Stocks\tCtrl+S")
        analysis_menu.Append(ID_WAVE, "Show Elliott Wave\tCtrl+E")
        analysis_menu.Append(ID_POSITION, "Analyze Position\tCtrl+P")
        analysis_menu.Append(ID_LOAD_SCAN, "Load Last Scan\tCtrl+L")
        menu_bar.Append(analysis_menu, "&Analysis")

        # -- Data --
        data_menu = wx.Menu()
        data_menu.Append(ID_FETCH_ALL, "Fetch All Data")
        data_menu.Append(ID_FETCH_US, "Fetch US Only")
        data_menu.Append(ID_FETCH_TW, "Fetch Taiwan Only")
        data_menu.AppendSeparator()
        data_menu.Append(ID_VALIDATE, "Validate Data")
        menu_bar.Append(data_menu, "&Data")

        # -- Help --
        help_menu = wx.Menu()
        help_menu.Append(ID_ABOUT, "About")
        help_menu.Append(ID_SHORTCUTS, "Keyboard Shortcuts")
        menu_bar.Append(help_menu, "&Help")

        self.SetMenuBar(menu_bar)

        # Bind menu events
        self.Bind(wx.EVT_MENU, self.on_menu_export, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_scan_all_stocks(e), id=ID_SCAN)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_show_elliott_wave(e), id=ID_WAVE)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_run_backtest(e), id=ID_BACKTEST)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_analyze_current_position(e), id=ID_POSITION)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_load_scan(e), id=ID_LOAD_SCAN)
        self.Bind(wx.EVT_MENU, lambda e: self.safe_handle_crawl_data(e), id=ID_FETCH_ALL)
        self.Bind(wx.EVT_MENU, lambda e: self._fetch_subset(e, "listed"), id=ID_FETCH_US)
        self.Bind(wx.EVT_MENU, lambda e: self._fetch_subset(e, "otc"), id=ID_FETCH_TW)
        self.Bind(wx.EVT_MENU, self._on_validate_data, id=ID_VALIDATE)
        self.Bind(wx.EVT_MENU, self.on_menu_about, id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, self.on_menu_shortcuts, id=ID_SHORTCUTS)

        # Store IDs for enable/disable
        self._menu_action_ids = [ID_SCAN, ID_WAVE, ID_BACKTEST, ID_POSITION, ID_LOAD_SCAN]

    def _fetch_subset(self, event, subset):
        """Set combo1 to *subset* then trigger crawl."""
        self.combo1.SetValue(subset)
        self.safe_handle_crawl_data(event)

    def _on_validate_data(self, event):
        """Run data validation script."""
        import subprocess, sys
        script = Path(__file__).resolve().parent.parent / 'scripts' / 'validate_data.py'
        self.update_status("Validating data...", 0)
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=120,
            )
            output = result.stdout + result.stderr
            self.output.AppendText(f"\n{'='*50}\nData Validation Results\n{'='*50}\n")
            self.output.AppendText(output + "\n")
            self.notebook.SetSelection(2)  # Switch to Log tab
        except Exception as e:
            wx.MessageBox(f"Validation failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
        self.update_status("Ready", 0)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _create_status_bar(self):
        """Create status bar with progress gauge."""
        self.status_bar = self.CreateStatusBar(3)
        self.status_bar.SetStatusWidths([-60, -25, -15])
        self.status_bar.SetStatusText("Ready", 0)

        self.progress_gauge = wx.Gauge(self.status_bar, range=100, size=(120, 16))
        # Position gauge inside field 1 (on_resize also calls _reposition_gauge)
        self._reposition_gauge()

    def _reposition_gauge(self):
        """Place the progress gauge inside status bar field 1."""
        try:
            rect = self.status_bar.GetFieldRect(1)
            self.progress_gauge.SetPosition((rect.x + 2, rect.y + 2))
            self.progress_gauge.SetSize((rect.width - 4, rect.height - 4))
        except Exception:
            pass

    def _on_frame_size_for_gauge(self, event):
        """Keep gauge positioned when the frame resizes."""
        self._reposition_gauge()
        event.Skip()

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------
    def _get_canvas_figsize(self):
        """Return (width_inches, height_inches) that fits the chart panel."""
        dpi = self.canvas.figure.get_dpi() or 100
        cs = self.chart_panel.GetClientSize()
        w = max(4, cs.width / dpi)
        h = max(3, cs.height / dpi)
        return w, h

    def _fit_figure_to_canvas(self, fig=None):
        """Resize *fig* (default: self.canvas.figure) to fill the chart panel."""
        if fig is None:
            fig = self.canvas.figure
        w, h = self._get_canvas_figsize()
        fig.set_size_inches(w, h)
        try:
            fig.tight_layout(pad=1.5)
        except Exception:
            pass
        if fig is not self.canvas.figure:
            self.canvas.figure = fig
        self.canvas.SetSize(self.chart_panel.GetClientSize())
        self.canvas.draw_idle()

    def on_resize(self, event):
        try:
            self.panel.Layout()
            self._reposition_gauge()
            if hasattr(self, 'canvas') and self.canvas and hasattr(self.canvas, 'figure'):
                self._fit_figure_to_canvas()
            event.Skip()
        except Exception as e:
            logger.error(f"Error in resize handler: {e}")
            event.Skip()

    # ------------------------------------------------------------------
    # Helpers carried over from original
    # ------------------------------------------------------------------
    def multi_path_build(self):
        for folder in ['data/raw', 'models', 'htmls', 'data/lists/adjustments']:
            os.makedirs(os.path.join(self.root, folder), exist_ok=True)

    def get_stock_list(self):
        symbols = []
        try:
            if Path(self.config['international_file']).exists():
                symbols += list(pd.read_csv(self.config['international_file'])['code'])
        except Exception as e:
            logger.debug(f"Error loading international.txt: {e}")
        try:
            if Path(self.config['list_file']).exists():
                symbols += list(pd.read_excel(self.config['list_file'])['code'].astype(str))
        except Exception as e:
            logger.debug(f"Error loading list.xlsx: {e}")
        try:
            if Path(self.config['otclist_file']).exists():
                symbols += list(pd.read_excel(self.config['otclist_file'])['code'].astype(str))
        except Exception as e:
            logger.debug(f"Error loading otclist.xlsx: {e}")
        symbols.append('TWII')
        return sorted(set(symbols))

    # ------------------------------------------------------------------
    # Event handlers (output / plot / buttons)
    # ------------------------------------------------------------------
    def on_update_output(self, event):
        try:
            if hasattr(event, 'message'):
                self.output.AppendText(event.message)
        except Exception as e:
            logger.error(f"Error updating output: {e}")

    def on_update_plot(self, event):
        try:
            if hasattr(event, 'figure'):
                self._fit_figure_to_canvas(event.figure)
                self.notebook.SetSelection(1)  # Switch to Chart tab
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
            buttons = [
                self.crawl_button, self.backtest_button, self.plot_button,
                self.current_pos_button, self.scan_button, self.load_scan_button,
            ]
            for button in buttons:
                if button:
                    button.Enable(enabled)
            # Also toggle menu items
            menu_bar = self.GetMenuBar()
            if menu_bar:
                for mid in self._menu_action_ids:
                    mi = menu_bar.FindItemById(mid)
                    if mi:
                        mi.Enable(enabled)
            self.buttons_enabled = enabled
        except Exception as e:
            logger.error(f"Error toggling buttons: {e}")

    # ------------------------------------------------------------------
    # Safe handler wrappers (unchanged logic)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stock list search / filter
    # ------------------------------------------------------------------
    def _apply_filters(self):
        """Apply both text search and action filter to the stock list."""
        query = self.search_ctrl.GetValue().strip().upper()
        action_filter = self.action_filter.GetStringSelection()

        filtered = list(self.all_scan_items)

        # Text filter
        if query:
            filtered = [
                item for item in filtered
                if query in self._item_symbol(item).upper()
            ]

        # Action filter
        if action_filter and action_filter != "All":
            buy_actions = ('STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION')
            hold_actions = ('HOLD', 'WAIT', 'SKIP', 'AVOID')

            def matches_filter(item):
                a = self._item_action(item)
                if action_filter == "BUY":
                    return a in buy_actions
                elif action_filter == "WATCH":
                    return a == "WATCH"
                elif action_filter == "EXIT":
                    return a == "EXIT"
                elif action_filter == "HOLD":
                    return a in hold_actions
                return True
            filtered = [item for item in filtered if matches_filter(item)]

        self._populate_stocks_list(filtered)

    def on_search(self, event):
        self._apply_filters()

    def _on_action_filter(self, event):
        self._apply_filters()

    @staticmethod
    def _item_symbol(item):
        """Extract symbol from an item (dict or tuple)."""
        if isinstance(item, dict):
            return item.get('symbol', '')
        if isinstance(item, (list, tuple)) and len(item) > 0:
            return str(item[0])
        return str(item)

    @staticmethod
    def _item_action(item):
        """Extract action from an item (dict or tuple)."""
        if isinstance(item, dict):
            return item.get('action', '')
        if isinstance(item, (list, tuple)) and len(item) > 2:
            return str(item[2])
        return ''

    def _populate_stocks_list(self, items):
        """Display items in ListCtrl WITHOUT overwriting all_scan_items.

        This is used for filtered views. The master list (all_scan_items) is
        only set by _update_stocks_listbox during a full scan or cache load.
        """
        action_colors = {
            'STRONG BUY': wx.Colour(0, 120, 0),
            'BUY': wx.Colour(34, 139, 34),
            'BUY DIP': wx.Colour(60, 160, 60),
            'BUY CORRECTION': wx.Colour(80, 140, 80),
            'WATCH': wx.Colour(180, 140, 0),
            'EXIT': wx.Colour(200, 0, 0),
            'AVOID': wx.Colour(150, 150, 150),
            'HOLD': wx.Colour(100, 100, 160),
            'WAIT': wx.Colour(120, 120, 120),
            'SKIP': wx.Colour(180, 180, 180),
        }
        short_action_map = {
            'STRONG BUY': 'S.BUY',
            'BUY CORRECTION': 'BUY.C',
            'BUY DIP': 'BUY.D',
        }

        self.stocks_list.DeleteAllItems()
        for i, stock in enumerate(items):
            if isinstance(stock, dict):
                symbol = str(stock.get('symbol', ''))
                score = stock.get('score', '')
                action = stock.get('action', '')
                conf = stock.get('conf', stock.get('confidence', 0))
            elif isinstance(stock, (list, tuple)):
                symbol = str(stock[0]) if len(stock) > 0 else ''
                score = stock[1] if len(stock) > 1 else ''
                action = stock[2] if len(stock) > 2 else ''
                conf = stock[3] if len(stock) > 3 else ''
            else:
                continue

            score_str = str(score) if score else ''
            conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) and conf else str(conf) if conf else ''
            short_action = short_action_map.get(action, action)

            idx = self.stocks_list.InsertItem(i, symbol)
            self.stocks_list.SetItem(idx, 1, score_str)
            self.stocks_list.SetItem(idx, 2, str(short_action))
            self.stocks_list.SetItem(idx, 3, conf_str)

            color = action_colors.get(action)
            if color:
                self.stocks_list.SetItemTextColour(idx, color)

            if action in ('STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION'):
                font = self.stocks_list.GetFont().Bold()
                item = self.stocks_list.GetItem(idx)
                item.SetFont(font)
                self.stocks_list.SetItem(item)

        self.update_status(f"{self.stocks_list.GetItemCount()} stocks", 2)

    # ------------------------------------------------------------------
    # Stock selection
    # ------------------------------------------------------------------
    def on_stock_selected(self, event):
        """Handle stock selection: update trade plan panel, show chart."""
        try:
            idx = event.GetIndex()
            symbol = self.stocks_list.GetItemText(idx, 0)
            if not symbol:
                return

            # --- Update trade plan panel ---
            self._update_trade_plan(symbol)

            # Set the combo box to this stock
            self.combo_stock.SetValue(symbol)

            # If we have a scanned timeframe, use that specific timeframe
            if self.scanned_timeframe:
                timeframe_to_chart = {
                    'day': 'Candlestick (Day)',
                    'week': 'Candlestick (Week)',
                    'month': 'Candlestick (Month)',
                }
                chart_type_for_scan = timeframe_to_chart.get(
                    self.scanned_timeframe, self.chart_type,
                )
                self.chart_type = chart_type_for_scan
                self.combo_chart_type.SetValue(chart_type_for_scan)

            # Switch to Chart tab
            self.notebook.SetSelection(1)

            # Show Elliott Wave for this stock
            from gui.handlers import handle_show_elliott_wave
            handle_show_elliott_wave(self, None)
        except Exception as e:
            self.output.AppendText(f"Error selecting stock: {e}\n")

    def _update_trade_plan(self, symbol):
        """Populate the trade plan panel from scan results."""
        results = getattr(self, '_scan_results', {})
        r = results.get(symbol)

        if not r:
            self.trade_header.SetLabel(f"{symbol}")
            self.trade_details.SetLabel("No signal data. Run Scan first.")
            self.trade_panel.Layout()
            return

        action = r.get('action', '?')
        score = r.get('score', 0)
        price = r.get('price', 0)

        # Color the header based on action
        action_colors = {
            'STRONG BUY': wx.Colour(0, 120, 0),
            'BUY': wx.Colour(34, 139, 34),
            'BUY DIP': wx.Colour(60, 160, 60),
            'BUY CORRECTION': wx.Colour(80, 140, 80),
            'WATCH': wx.Colour(180, 140, 0),
            'EXIT': wx.Colour(200, 0, 0),
        }
        color = action_colors.get(action, wx.Colour(60, 60, 60))

        self.trade_header.SetLabel(f"{symbol}  {action}  (Score: {score})")
        self.trade_header.SetForegroundColour(color)

        # Build detail text
        lines = []
        if price:
            lines.append(f"Price: ${price:.2f}   Wave: {r.get('wave', '?')}   Trend: {r.get('trend', '?')}")

        stop = r.get('stop', 0)
        t1 = r.get('target1', 0)
        t2 = r.get('target2', 0)
        rr = r.get('rr', 0)

        if stop and t1:
            risk_pct = abs(price - stop) / price * 100 if price else 0
            lines.append(f"Entry: ${price:.2f}   Stop: ${stop:.2f} (-{risk_pct:.1f}%)")
            lines.append(f"Target1: ${t1:.2f}   Target2: ${t2:.2f}")
            lines.append(f"R:R  {rr:.1f} : 1")

        # Tier 3: Trailing stop + position sizing
        trailing = r.get('trailing_stop', 0)
        size_mult = r.get('size_mult', 0)
        size_note = r.get('size_note', '')
        if trailing and size_mult:
            lines.append(f"Trail: ${trailing:.2f} (2.5 ATR)   Position: {size_mult:.1f}x ({size_note})")

        rsi = r.get('rsi', 0)
        macd_cross = r.get('macd_cross', '-')
        macd_dir = r.get('macd_hist_dir', '')
        adx_val = r.get('adx', 0)
        adx_reg = r.get('adx_regime', '')
        speed = r.get('speed', '?')
        macd_str = f"{macd_cross or '-'}"
        if macd_dir:
            macd_str += f" ({macd_dir})"
        adx_str = f"{adx_val:.0f}"
        if adx_reg:
            adx_str += f" ({adx_reg})"
        lines.append(f"RSI: {rsi:.0f}  MACD: {macd_str}  ADX: {adx_str}  Speed: {speed}")

        # Volume info
        vol_ratio = r.get('vol_ratio', 0)
        vol_score = r.get('volume_score', 0)
        vel_acc = r.get('vel_accel', '')
        vol_parts = []
        if vol_ratio:
            vol_label = 'confirmed' if vol_ratio >= 1.0 else 'weak'
            vol_parts.append(f"Volume: {vol_ratio:.1f}x avg ({vol_label})")
        if vel_acc and vel_acc != 'flat':
            vel_label = vel_acc.replace('_', ' ')
            vol_parts.append(f"Momentum: {vel_label}")
        if vol_parts:
            lines.append("  ".join(vol_parts))

        reason = r.get('reason', '')
        if reason:
            lines.append(f"{reason}")

        self.trade_details.SetLabel("\n".join(lines))
        self.trade_panel.Layout()

    # ------------------------------------------------------------------
    # Auto-load last scan
    # ------------------------------------------------------------------
    def auto_load_last_scan(self):
        """Show last scan info on startup with stale warning."""
        try:
            cache_info = self.scan_cache.get_cache_info()
            if cache_info and cache_info['exists']:
                age = cache_info.get('age_formatted', 'unknown')
                patterns = cache_info.get('patterns_found', 0)
                total = cache_info.get('total_scanned', 0)

                # Check staleness
                age_hours = cache_info.get('age_hours', 0)
                stale_warning = ""
                if age_hours > 24:
                    stale_warning = f"\n  WARNING: Scan is {age_hours:.0f}h old — consider re-scanning!\n"

                self.output.AppendText(f"\nLast scan: {age} | {patterns}/{total} patterns found\n")
                if stale_warning:
                    self.output.AppendText(stale_warning)
                self.output.AppendText("Press Ctrl+L to load last scan, or Ctrl+S to run new scan.\n\n")

                self.update_status(f"Last scan: {age}", 0)
        except Exception as e:
            logger.debug(f"Auto-load check failed: {e}")

    # ------------------------------------------------------------------
    # Status / progress helpers
    # ------------------------------------------------------------------
    def update_status(self, text, field=0):
        """Update status bar text."""
        wx.CallAfter(self.status_bar.SetStatusText, text, field)

    def update_progress(self, current, total):
        """Update progress gauge and text."""
        pct = int(current / total * 100) if total > 0 else 0
        wx.CallAfter(self.progress_gauge.SetValue, pct)
        wx.CallAfter(self.status_bar.SetStatusText, f"{current}/{total}", 1)

    def reset_progress(self):
        """Reset progress gauge."""
        wx.CallAfter(self.progress_gauge.SetValue, 0)
        wx.CallAfter(self.status_bar.SetStatusText, "", 1)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def update_dashboard(self, scan_results=None):
        """Update dashboard with trader-focused signal summary using HTML tables."""
        try:
            if scan_results is None:
                return

            if isinstance(scan_results, list):
                scan_results = {'results': scan_results, 'total_scanned': len(scan_results)}

            results = scan_results.get('results', scan_results.get('stocks', []))
            buys = scan_results.get('buys', [r for r in results if r.get('action', '') in ('STRONG BUY', 'BUY', 'BUY DIP', 'BUY CORRECTION')])
            watches = scan_results.get('watches', [r for r in results if r.get('action') == 'WATCH'])
            exits = scan_results.get('exits', [r for r in results if r.get('action') == 'EXIT'])
            holds = [r for r in results if r.get('action') in ('HOLD', 'WAIT')]
            avoids = [r for r in results if r.get('action') in ('AVOID', 'SKIP')]
            regime = scan_results.get('market_regime', '')
            regime_msg = scan_results.get('regime_msg', '')
            total_scanned = scan_results.get('total_scanned', len(results))
            scan_time = scan_results.get('scan_time', '')

            H = []  # HTML accumulator
            H.append("<html><body>")

            # ── HEADER ──
            H.append("<h2>Signal Dashboard</h2>")
            if scan_time:
                H.append(f"<font color='#888' size='2'>{scan_time}</font>")

            # ── MARKET REGIME ──
            if regime:
                regime_html = {
                    'FAVORABLE': ("<table bgcolor='#d5f5e3' width='100%'><tr><td>"
                                  "<b>FAVORABLE</b> -- Markets healthy, normal entries OK"
                                  "</td></tr></table>"),
                    'MIXED': ("<table bgcolor='#fef9e7' width='100%'><tr><td>"
                              "<b>!! MIXED !!</b> -- Selective entries only -- "
                              "raise conviction bar (score &gt; 75)"
                              "</td></tr></table>"),
                    'CAUTION': ("<table bgcolor='#fdedec' width='100%'><tr><td>"
                                "<font color='#c0392b'><b>XX CAUTION XX</b></font> -- "
                                "Market overheated -- protect positions, tighten stops, "
                                "wait for regime to improve"
                                "</td></tr></table>"),
                    'UNKNOWN': ("<table bgcolor='#fef9e7' width='100%'><tr><td>"
                                "<b>?? UNKNOWN ??</b> -- "
                                "Loaded from cache -- re-scan for live regime"
                                "</td></tr></table>"),
                }
                H.append(regime_html.get(regime, f"<p>{regime}</p>"))

            # ── SIGNAL DISTRIBUTION ──
            n_other = len(holds) + len(avoids)
            H.append("<table cellpadding='4'><tr>")
            H.append(f"<td><b>Scanned:</b> {total_scanned}</td>")
            H.append(f"<td><b>Analyzed:</b> {len(results)}</td>")
            H.append(f"<td bgcolor='#27ae60'><font color='#fff'><b>BUY {len(buys)}</b></font></td>")
            H.append(f"<td bgcolor='#f39c12'><font color='#fff'><b>WATCH {len(watches)}</b></font></td>")
            H.append(f"<td bgcolor='#e74c3c'><font color='#fff'><b>EXIT {len(exits)}</b></font></td>")
            H.append(f"<td bgcolor='#95a5a6'><font color='#fff'><b>OTHER {n_other}</b></font></td>")
            H.append("</tr></table>")

            # ── SIGNAL QUALITY SUMMARY ──
            if buys:
                buy_scores = [r.get('score', 0) for r in buys]
                avg_score = sum(buy_scores) / len(buy_scores)
                max_score = max(buy_scores)
                strong = sum(1 for s in buy_scores if s >= 75)
                vol_confirmed = sum(1 for r in buys if r.get('vol_ratio', 0) >= 1.0)
                trend_strong = sum(1 for r in buys if r.get('adx_regime') == 'strong_trend')

                H.append("<table cellpadding='3'><tr>")
                H.append(f"<td bgcolor='#ecf0f1'>Avg Score <b>{avg_score:.0f}</b></td>")
                H.append(f"<td bgcolor='#ecf0f1'>Best <b>{max_score}</b></td>")
                H.append(f"<td bgcolor='#ecf0f1'>Strong (75+) <b>{strong}</b></td>")
                H.append(f"<td bgcolor='#ecf0f1'>Vol Confirmed <b>{vol_confirmed}/{len(buys)}</b></td>")
                H.append(f"<td bgcolor='#ecf0f1'>Trend Strong <b>{trend_strong}/{len(buys)}</b></td>")
                H.append("</tr></table>")

            # ── BUY CANDIDATES (ALL) ──
            if buys:
                H.append(f"<h3>BUY Candidates ({len(buys)})</h3>")
                H.append("<table border='1' cellpadding='4' cellspacing='0' width='100%'>")
                H.append("<tr bgcolor='#2c3e50'>"
                         "<th><font color='#fff'>Symbol</font></th>"
                         "<th><font color='#fff'>Action</font></th>"
                         "<th align='right'><font color='#fff'>Score</font></th>"
                         "<th align='right'><font color='#fff'>Price</font></th>"
                         "<th align='right'><font color='#fff'>Stop</font></th>"
                         "<th align='right'><font color='#fff'>Target 1</font></th>"
                         "<th align='right'><font color='#fff'>Target 2</font></th>"
                         "<th align='right'><font color='#fff'>Trail</font></th>"
                         "<th align='right'><font color='#fff'>R:R</font></th>"
                         "<th align='right'><font color='#fff'>Risk%</font></th>"
                         "<th><font color='#fff'>Wave</font></th>"
                         "<th align='right'><font color='#fff'>Vol</font></th>"
                         "<th><font color='#fff'>Trend</font></th>"
                         "<th align='right'><font color='#fff'>RSI</font></th>"
                         "<th align='right'><font color='#fff'>Size</font></th>"
                         "<th><font color='#fff'>Note</font></th>"
                         "</tr>")
                for idx, r in enumerate(buys):
                    sym = r.get('symbol', '')
                    action = r.get('action', '')
                    score = r.get('score', 0)
                    price = r.get('price', 0)
                    stop = r.get('stop', 0)
                    t1 = r.get('target1', 0)
                    t2 = r.get('target2', 0)
                    trail = r.get('trailing_stop', 0)
                    rr = r.get('rr', 0)
                    risk_pct = abs(price - stop) / price * 100 if price else 0
                    wave = r.get('wave', '?')
                    vr = r.get('vol_ratio', 1.0)
                    adx_r = r.get('adx_regime', '')
                    trend_str = adx_r.replace('_', ' ') if adx_r else r.get('trend', '')
                    rsi = r.get('rsi', 0)
                    size = r.get('size_mult', 0)
                    s_note = r.get('size_note', '')

                    # Color coding
                    score_color = '#1a7a42' if score >= 75 else '#b7950b' if score >= 55 else '#c0392b'
                    vol_color = '#1a7a42' if vr >= 1.0 else '#c0392b' if vr < 0.5 else '#666'
                    rr_color = '#1a7a42' if rr >= 2.0 else '#b7950b' if rr >= 1.0 else '#c0392b'
                    row_bg = '#f8f9fa' if idx % 2 == 1 else '#ffffff'

                    H.append(
                        f"<tr bgcolor='{row_bg}'>"
                        f"<td><b>{sym}</b></td>"
                        f"<td>{action}</td>"
                        f"<td align='right'><font color='{score_color}'><b>{score}</b></font></td>"
                        f"<td align='right'>${price:.2f}</td>"
                        f"<td align='right'>${stop:.2f}</td>"
                        f"<td align='right'>${t1:.2f}</td>"
                        f"<td align='right'>${t2:.2f}</td>"
                        f"<td align='right'>${trail:.2f}</td>"
                        f"<td align='right'><font color='{rr_color}'>{rr:.1f}x</font></td>"
                        f"<td align='right'>{risk_pct:.1f}%</td>"
                        f"<td>W{wave}</td>"
                        f"<td align='right'><font color='{vol_color}'>{vr:.1f}x</font></td>"
                        f"<td>{trend_str}</td>"
                        f"<td align='right'>{rsi:.0f}</td>"
                        f"<td align='right'>{size:.1f}x</td>"
                        f"<td><font color='#666' size='1'>{s_note}</font></td>"
                        f"</tr>"
                    )
                H.append("</table>")

            # ── WATCH LIST (ALL) ──
            if watches:
                H.append(f"<h3>Watch List ({len(watches)})</h3>")
                H.append("<table border='1' cellpadding='4' cellspacing='0' width='100%'>")
                H.append("<tr bgcolor='#2c3e50'>"
                         "<th><font color='#fff'>Symbol</font></th>"
                         "<th align='right'><font color='#fff'>Score</font></th>"
                         "<th align='right'><font color='#fff'>Price</font></th>"
                         "<th align='right'><font color='#fff'>Stop</font></th>"
                         "<th align='right'><font color='#fff'>Target 1</font></th>"
                         "<th align='right'><font color='#fff'>R:R</font></th>"
                         "<th align='right'><font color='#fff'>RSI</font></th>"
                         "<th><font color='#fff'>Wave</font></th>"
                         "<th align='right'><font color='#fff'>Vol</font></th>"
                         "<th><font color='#fff'>Missing</font></th>"
                         "</tr>")
                for idx, r in enumerate(watches):
                    missing = []
                    if not r.get('entry_ok', False):
                        missing.append('momentum')
                    if r.get('rr', 0) < 2.0:
                        missing.append('R:R')
                    if r.get('conf', 0) < 0.25:
                        missing.append('confidence')
                    if r.get('exit_warn', False):
                        missing.append('no exit warn')
                    if r.get('vol_ratio', 1.0) < 0.5:
                        missing.append('volume')
                    if r.get('adx_regime') == 'no_trend':
                        missing.append('trend strength')
                    if r.get('w5_reversal_warn'):
                        missing.append('W5 exhaustion')
                    needs = ', '.join(missing) if missing else 'regime change'

                    sym = r.get('symbol', '')
                    score = r.get('score', 0)
                    price = r.get('price', 0)
                    stop = r.get('stop', 0)
                    t1 = r.get('target1', 0)
                    rr = r.get('rr', 0)
                    rsi = r.get('rsi', 0)
                    wave = r.get('wave', '?')
                    vr = r.get('vol_ratio', 1.0)
                    row_bg = '#f8f9fa' if idx % 2 == 1 else '#ffffff'

                    H.append(
                        f"<tr bgcolor='{row_bg}'>"
                        f"<td><b>{sym}</b></td>"
                        f"<td align='right'>{score}</td>"
                        f"<td align='right'>${price:.2f}</td>"
                        f"<td align='right'>${stop:.2f}</td>"
                        f"<td align='right'>${t1:.2f}</td>"
                        f"<td align='right'>{rr:.1f}x</td>"
                        f"<td align='right'>{rsi:.0f}</td>"
                        f"<td>W{wave}</td>"
                        f"<td align='right'>{vr:.1f}x</td>"
                        f"<td><font color='#c0392b'>{needs}</font></td>"
                        f"</tr>"
                    )
                H.append("</table>")

            # ── EXIT ALERTS (ALL) ──
            if exits:
                H.append(f"<h3>Exit Alerts ({len(exits)})</h3>")
                H.append("<table border='1' cellpadding='4' cellspacing='0' width='100%'>")
                H.append("<tr bgcolor='#2c3e50'>"
                         "<th><font color='#fff'>Symbol</font></th>"
                         "<th align='right'><font color='#fff'>Price</font></th>"
                         "<th align='right'><font color='#fff'>RSI</font></th>"
                         "<th align='right'><font color='#fff'>6M %</font></th>"
                         "<th><font color='#fff'>Wave</font></th>"
                         "<th><font color='#fff'>Reason</font></th>"
                         "</tr>")
                for idx, r in enumerate(exits):
                    sym = r.get('symbol', '')
                    price = r.get('price', 0)
                    rsi = r.get('rsi', 0)
                    mom6 = r.get('mom_6m', 0)
                    wave = r.get('wave', '?')
                    mom_color = '#1a7a42' if mom6 > 0 else '#c0392b'
                    row_bg = '#f8f9fa' if idx % 2 == 1 else '#ffffff'
                    H.append(
                        f"<tr bgcolor='{row_bg}'>"
                        f"<td><b>{sym}</b></td>"
                        f"<td align='right'>${price:.2f}</td>"
                        f"<td align='right'>{rsi:.0f}</td>"
                        f"<td align='right'><font color='{mom_color}'>{mom6:+.1f}%</font></td>"
                        f"<td>W{wave}</td>"
                        f"<td><font color='#e74c3c'>Exit warning</font></td>"
                        f"</tr>"
                    )
                H.append("</table>")

            # ── NO SIGNALS ──
            if not buys and not watches:
                H.append("<h3>No Buy Signals Right Now</h3>")
                H.append("<p>This is normal and protective. The system only signals when:</p>")
                H.append("<ul>")
                H.append("<li>Wave structure confirms trend (impulse pattern)</li>")
                H.append("<li>Trend is bullish (SMA50 &gt; SMA200)</li>")
                H.append("<li>Momentum confirms entry (RSI + MACD + ADX)</li>")
                H.append("<li>Risk/reward ratio is favorable (&gt; 1.5:1)</li>")
                H.append("</ul>")
                if regime == 'CAUTION':
                    H.append("<p><font color='#c0392b'><b>Current regime is CAUTION</b> -- "
                             "most stocks show exit signals. Patience protects capital.</font></p>")
                H.append("<p><b>Next steps:</b> Monitor WATCH list daily | "
                         "Tighten stops | Re-scan when regime improves (Ctrl+S)</p>")

            # ── TOP MOMENTUM ──
            movers = sorted(
                [r for r in results if r.get('mom_6m', 0) > 10 and r.get('trend') == 'bullish'],
                key=lambda x: x.get('mom_6m', 0),
                reverse=True,
            )[:15]
            if movers:
                H.append("<h3>Strongest Momentum</h3>")
                H.append("<table border='1' cellpadding='4' cellspacing='0' width='100%'>")
                H.append("<tr bgcolor='#2c3e50'>"
                         "<th><font color='#fff'>Symbol</font></th>"
                         "<th align='right'><font color='#fff'>Price</font></th>"
                         "<th align='right'><font color='#fff'>6M %</font></th>"
                         "<th><font color='#fff'>Wave</font></th>"
                         "<th><font color='#fff'>Action</font></th>"
                         "</tr>")
                for idx, r in enumerate(movers):
                    sym = r.get('symbol', '')
                    price = r.get('price', 0)
                    mom6 = r.get('mom_6m', 0)
                    wave = r.get('wave', '?')
                    action = r.get('action', '')
                    row_bg = '#f8f9fa' if idx % 2 == 1 else '#ffffff'
                    H.append(
                        f"<tr bgcolor='{row_bg}'>"
                        f"<td><b>{sym}</b></td>"
                        f"<td align='right'>${price:.2f}</td>"
                        f"<td align='right'><font color='#1a7a42'>{mom6:+.1f}%</font></td>"
                        f"<td>W{wave}</td>"
                        f"<td>{action}</td>"
                        f"</tr>"
                    )
                H.append("</table>")

            # ── SCAN AGE WARNING ──
            last_scan = getattr(self, '_last_scan_time', None)
            if last_scan:
                from datetime import datetime
                age_hours = (datetime.now() - last_scan).total_seconds() / 3600
                if age_hours > 24:
                    H.append(f"<p><font color='#c0392b'><b>Scan is {age_hours:.0f}h old</b> "
                             f"-- prices may have changed. Re-scan with Ctrl+S</font></p>")

            H.append("<p><font color='#888' size='2'>Click any stock in the list "
                     "to see chart + trade plan</font></p>")
            H.append("</body></html>")

            self.dashboard_html.SetPage("".join(H))
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")

    # ------------------------------------------------------------------
    # Menu action handlers
    # ------------------------------------------------------------------
    def on_menu_export(self, event):
        """Export results to JSON file."""
        import json
        with wx.FileDialog(
            self, "Export Results", wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()
            try:
                with open(path, 'w') as f:
                    json.dump(self.all_scan_items, f, indent=2, default=str)
                self.update_status(f"Exported to {path}", 0)
            except Exception as e:
                wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_menu_about(self, event):
        """Show about dialog."""
        info = wx.adv.AboutDialogInfo()
        info.SetName("Fintech-sys")
        info.SetVersion("2.0")
        info.SetDescription("Elliott Wave Trading System\nAdvanced stock analysis with multi-timeframe pattern detection.")
        wx.adv.AboutBox(info)

    def on_menu_shortcuts(self, event):
        """Show keyboard shortcuts dialog."""
        shortcuts = (
            "Keyboard Shortcuts\n"
            "----------------------------\n"
            "Ctrl+S   Scan All Stocks\n"
            "Ctrl+E   Show Elliott Wave\n"
            "Ctrl+B   Run Backtest\n"
            "Ctrl+P   Analyze Position\n"
            "Ctrl+L   Load Last Scan\n"
            "Ctrl+Q   Quit\n"
            "Ctrl+Shift+E   Export Results\n"
        )
        wx.MessageBox(shortcuts, "Keyboard Shortcuts", wx.OK | wx.ICON_INFORMATION)
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
            
            # Use canvas-fitted figure size
            fig_w, fig_h = self._get_canvas_figsize()

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
                    figsize=(fig_w, fig_h),
                    panel_ratios=(4, 1),
                    addplot=additional_plots if additional_plots else None,
                    returnfig=True,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    tight_layout=True,
                    show_nontrading=False
                )

                price_ax = axes[0]

                # Add annotations for Elliott Wave points
                self._add_elliott_wave_annotations(price_ax, df, df_ohlc, wave_data)

                # Fit to canvas
                self.ax = price_ax
                self._fit_figure_to_canvas(fig)
                
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
        self._fit_figure_to_canvas()

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
            self._fit_figure_to_canvas()
            if hasattr(self, 'output'):
                self.output.AppendText("No valid Elliott Wave pattern found for this data.\n")
            return
        plot_elliott_wave_analysis_enhanced(
            df, wave_data, column='close',
            title=f"Elliott Wave Analysis for {symbol}", ax=self.ax
        )
        self._fix_date_labels(self.ax, df)
        self._fit_figure_to_canvas()

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
            self._fit_figure_to_canvas()
            if hasattr(self, 'output'):
                self.output.AppendText("No valid Elliott Wave pattern found for this data.\n")
            return

        # Use the existing enhanced plotting function
        plot_elliott_wave_analysis_enhanced(
            df, wave_data, column='close',
            title=f"Elliott Wave Analysis for {symbol}", ax=self.ax
        )
        self._fix_date_labels(self.ax, df)
        self._fit_figure_to_canvas()

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

            # Update canvas
            self._fit_figure_to_canvas()

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

            # Update canvas
            self._fit_figure_to_canvas()

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

        # Use canvas size instead of hardcoded figsize
        fig_w, fig_h = self._get_canvas_figsize()
        fig, axes = plt.subplots(num_patterns, 1, figsize=(fig_w, fig_h))

        if num_patterns == 1:
            axes = [axes]

        for i, pattern in enumerate(multiple_patterns[:3]):
            ax = axes[i] if num_patterns > 1 else axes[0]
            timeframe_data = self._get_timeframe_specific_data(df, pattern, symbol)
            self._plot_single_timeframe_pattern(ax, timeframe_data, pattern, i == 0)

        self._fit_figure_to_canvas(fig)

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
        self._fit_figure_to_canvas()

    