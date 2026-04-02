"""
Centralized Configuration Manager for Investment System

This module provides a unified interface for accessing all configuration values
that were previously scattered throughout the codebase. It consolidates:
- GUI settings and constants
- Data processing parameters
- Elliott Wave analysis parameters
- Fibonacci levels and tolerances
- Visualization settings
- Performance and network settings
- Error handling configuration
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)



@dataclass
class GUIConfig:
    """GUI-specific configuration"""
    window_width: int = 1000
    window_height: int = 700
    min_width: int = 600
    min_height: int = 400
    screen_scale_factor: float = 0.9
    output_height: int = 150
    scrollbar_size: int = 20
    border_padding: int = 5
    label_width: int = 130
    chart_types: List[str] = field(default_factory=lambda: ["Line", "Candlestick (Day)", "Candlestick (Week)", "Candlestick (Month)"])
    default_chart_type: str = "Line"
    web_crawling_options: List[str] = field(default_factory=lambda: ["ALL", "listed", "otc"])
    default_web_crawling: str = "ALL"
    web_crawling_suffixes: Dict[str, str] = field(default_factory=lambda: {"listed": ".TW", "otc": ".TWO"})
    buttons_enabled_by_default: bool = True
    disable_buttons_during_operation: bool = True


@dataclass
class DataConfig:
    """Data processing configuration"""
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    adjustments_dir: str = "data/lists/adjustments"
    databases_dir: str = "data/raw/databases"
    models_dir: str = "models"
    htmls_dir: str = "htmls"
    international_file: str = "data/lists/international.txt"
    list_file: str = "data/lists/adjustments/list.xlsx"
    otclist_file: str = "data/lists/adjustments/otclist.xlsx"
    start_date: str = "2002-01-01"
    end_date: str = "2026-12-20"
    default_years_back: int = 10
    min_data_points: int = 50
    min_ohlc_points: int = 20
    max_days_for_mapping: int = 14
    resample_frequencies: Dict[str, str] = field(default_factory=lambda: {"day": "D", "week": "W", "month": "M"})
    candlestick_filtering: Dict[str, Union[int, str]] = field(default_factory=lambda: {"day_years": 3, "week_years": 10, "month_years": "all"})


@dataclass
class ElliottWaveConfig:
    """Elliott Wave analysis configuration"""
    min_points: int = 6
    max_points: int = 12
    min_confidence_threshold: float = 0.2
    max_patterns_per_timeframe: int = 3
    max_patterns_total: int = 5
    pattern_relationships_enabled: bool = True
    timeframes: List[str] = field(default_factory=lambda: ["day", "week", "month"])
    default_timeframe: str = "day"
    timeframe_priority_weights: Dict[str, float] = field(default_factory=lambda: {"day": 0.5, "week": 0.3, "month": 0.2})
    window_size: int = 5
    min_prominence_pct: float = 0.5
    volume_threshold_ratio: float = 0.7
    fibonacci_tolerance: float = 0.05
    wave_equality_tolerance: float = 0.1
    overlap_tolerance: float = 0.1
    confidence_base: float = 0.5
    confidence_wave2_ideal: float = 0.2
    confidence_wave2_ok: float = 0.1
    confidence_wave3_strong: float = 0.2
    confidence_diagonal: float = 0.1
    confidence_clean_impulse: float = 0.2
    confidence_direction: float = 0.3
    confidence_fibonacci: float = 0.3
    confidence_volume: float = 0.2
    confidence_alternation: float = 0.15
    confidence_missing_data_penalty: float = 0.5
    confidence_over_confidence_penalty: float = 0.5


@dataclass
class FibonacciConfig:
    """Fibonacci levels and relationships configuration"""
    retracements: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])
    retracement_tolerance: float = 0.05
    extensions: List[float] = field(default_factory=lambda: [1.618, 2.618, 4.236])
    extension_tolerance: float = 0.1
    projections: List[float] = field(default_factory=lambda: [0.618, 1.0, 1.618])
    projection_tolerance: float = 0.05
    time_ratios: List[float] = field(default_factory=lambda: [0.618, 1.0, 1.618, 2.618])
    time_ratio_tolerance: float = 0.1
    wave2_ideal_range: List[float] = field(default_factory=lambda: [0.382, 0.618])
    wave2_acceptable_range: List[float] = field(default_factory=lambda: [0.236, 0.786])
    wave3_extension_range: List[float] = field(default_factory=lambda: [1.618, 2.618])
    wave4_retracement_range: List[float] = field(default_factory=lambda: [0.236, 0.5])


@dataclass
class VisualizationConfig:
    """Visualization and chart styling configuration"""
    figure_size: List[int] = field(default_factory=lambda: [12, 8])
    dpi: int = 100
    face_color: str = "white"
    edge_color: str = "black"
    grid_alpha: float = 0.3
    grid_style: str = "-"
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#2E8B57",
        "warning": "#FFA500",
        "danger": "#DC143C",
        "info": "#4682B4",
        "light": "#F8F9FA",
        "dark": "#343A40"
    })
    pattern_colors: Dict[str, str] = field(default_factory=lambda: {
        "impulse": "#2E86AB",
        "corrective": "#E63946",
        "primary": "#2E86AB",
        "supporting": "#A23B72",
        "conflicting": "#F18F01",
        "independent": "#C73E1D",
        "alternative": "#8B4513",
        "wave_line": "#F77F00"
    })
    confidence_colors: Dict[str, str] = field(default_factory=lambda: {
        "very_high": "#2E8B57",
        "high": "#32CD32",
        "moderate": "#FFA500",
        "low": "#FF6347",
        "very_low": "#DC143C"
    })
    fonts: Dict[str, Union[int, str]] = field(default_factory=lambda: {
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "annotation_size": 9,
        "title_weight": "bold",
        "label_weight": "normal"
    })
    annotations: Dict[str, Union[int, float, List[int]]] = field(default_factory=lambda: {
        "marker_size": 8,
        "line_width": 2,
        "alpha": 0.8,
        "bbox_padding": 0.2,
        "text_offset": [5, 10]
    })
    fibonacci_viz: Dict[str, Any] = field(default_factory=lambda: {
        "retracement_colors": ["#FFD700", "#FFA500", "#FF6347", "#FF4500", "#FF0000"],
        "extension_colors": ["#9370DB", "#8A2BE2", "#4B0082"],
        "line_style": ":",
        "alpha": 0.6,
        "line_width": 1.5
    })


@dataclass
class PerformanceConfig:
    """Performance and caching configuration"""
    caching_enabled: bool = True
    max_cache_size: int = 1000
    cache_ttl_hours: int = 24
    pattern_cache_enabled: bool = True
    data_cache_enabled: bool = True
    max_workers: int = 4
    thread_timeout_seconds: int = 300
    enable_async_processing: bool = True
    max_dataframe_size_mb: int = 100
    garbage_collection_threshold: float = 0.8
    enable_memory_monitoring: bool = True


@dataclass
class NetworkConfig:
    """Network and crawling configuration"""
    yahoo_finance_base_url: str = "https://finance.yahoo.com/quote/"
    retry_attempts: int = 3
    retry_delay_ms: int = 2000
    request_timeout_seconds: int = 30
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    requests_per_second: int = 2
    delay_between_requests_ms: int = 500
    respect_robots_txt: bool = True


@dataclass
class BacktestingConfig:
    """Backtesting configuration"""
    default_buy_on: int = 0
    default_sell_on: int = 5
    min_price_changes: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.1])
    default_min_price_change: float = 0.02
    calculate_sharpe_ratio: bool = True
    calculate_max_drawdown: bool = True
    calculate_win_rate: bool = True
    risk_free_rate: float = 0.02


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_levels: Dict[str, str] = field(default_factory=lambda: {
        "root": "INFO",
        "analysis": "DEBUG",
        "gui": "INFO",
        "network": "WARNING",
        "performance": "INFO"
    })
    main_log_file: str = "logs/investment_system.log"
    error_log_file: str = "logs/errors.log"
    performance_log_file: str = "logs/performance.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    include_thread_id: bool = True
    include_function_name: bool = True


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration"""
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 1
    exponential_backoff: bool = True
    log_all_errors: bool = True
    show_user_friendly_messages: bool = True
    include_stack_traces: bool = False
    use_simple_plot_on_error: bool = True
    use_cached_data_on_error: bool = True
    show_error_in_gui: bool = True


@dataclass
class DevelopmentConfig:
    """Development and debugging configuration"""
    debug_enabled: bool = False
    verbose_logging: bool = False
    show_debug_info: bool = False
    log_performance_metrics: bool = True
    use_mock_data: bool = False
    mock_data_path: str = "tests/mock_data/"
    run_integration_tests: bool = False
    profiling_enabled: bool = False
    profile_analysis_functions: bool = False
    profile_gui_operations: bool = False
    profiling_output_file: str = "profiles/performance_profile.prof"


class ConfigManager:
    """
    Centralized configuration manager that loads and provides access to all
    configuration values from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration YAML file
        """
        self.config_path = config_path or "config/app_config.yaml"
        self._config_data: Dict[str, Any] = {}
        self._load_config()
        self._create_config_objects()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f)
            else:
                logger.info(f"Warning: Configuration file {self.config_path} not found. Using defaults.")
                self._config_data = {}
        except Exception as e:
            logger.info(f"Error loading configuration: {e}. Using defaults.")
            self._config_data = {}
    
    def _create_config_objects(self):
        """Create typed configuration objects from loaded data"""
        gui_data = self._config_data.get('gui', {})
        data_data = self._config_data.get('data', {})
        elliott_data = self._config_data.get('elliott_wave', {})
        fibonacci_data = self._config_data.get('fibonacci', {})
        viz_data = self._config_data.get('visualization', {})
        perf_data = self._config_data.get('performance', {})
        network_data = self._config_data.get('network', {})
        backtest_data = self._config_data.get('backtesting', {})
        logging_data = self._config_data.get('logging', {})
        error_data = self._config_data.get('error_handling', {})
        dev_data = self._config_data.get('development', {})
        
        # Create configuration objects
        self.gui = GUIConfig(
            window_width=gui_data.get('window', {}).get('default_width', 1000),
            window_height=gui_data.get('window', {}).get('default_height', 700),
            min_width=gui_data.get('window', {}).get('min_width', 600),
            min_height=gui_data.get('window', {}).get('min_height', 400),
            screen_scale_factor=gui_data.get('window', {}).get('screen_scale_factor', 0.9),
            output_height=gui_data.get('components', {}).get('output_height', 150),
            scrollbar_size=gui_data.get('components', {}).get('scrollbar_size', 20),
            border_padding=gui_data.get('components', {}).get('border_padding', 5),
            label_width=gui_data.get('components', {}).get('label_width', 130),
            chart_types=gui_data.get('chart_types', ["Line", "Candlestick (Day)", "Candlestick (Week)", "Candlestick (Month)"]),
            default_chart_type=gui_data.get('chart_types', {}).get('default', "Line"),
            web_crawling_options=gui_data.get('web_crawling', {}).get('options', ["ALL", "listed", "otc"]),
            default_web_crawling=gui_data.get('web_crawling', {}).get('default', "ALL"),
            web_crawling_suffixes=gui_data.get('web_crawling', {}).get('suffixes', {"listed": ".TW", "otc": ".TWO"}),
            buttons_enabled_by_default=gui_data.get('buttons', {}).get('enabled_by_default', True),
            disable_buttons_during_operation=gui_data.get('buttons', {}).get('disable_during_operation', True)
        )
        
        self.data = DataConfig(
            data_dir=data_data.get('directories', {}).get('data_dir', "data"),
            raw_dir=data_data.get('directories', {}).get('raw_dir', "data/raw"),
            processed_dir=data_data.get('directories', {}).get('processed_dir', "data/processed"),
            adjustments_dir=data_data.get('directories', {}).get('adjustments_dir', "data/lists/adjustments"),
            databases_dir=data_data.get('directories', {}).get('databases_dir', "data/raw/databases"),
            models_dir=data_data.get('directories', {}).get('models_dir', "models"),
            htmls_dir=data_data.get('directories', {}).get('htmls_dir', "htmls"),
            international_file=data_data.get('files', {}).get('international_file', "data/lists/international.txt"),
            list_file=data_data.get('files', {}).get('list_file', "data/lists/adjustments/list.xlsx"),
            otclist_file=data_data.get('files', {}).get('otclist_file', "data/lists/adjustments/otclist.xlsx"),
            start_date=data_data.get('date_ranges', {}).get('start_date', "2002-01-01"),
            end_date=data_data.get('date_ranges', {}).get('end_date', "2026-12-20"),
            default_years_back=data_data.get('date_ranges', {}).get('default_years_back', 10),
            min_data_points=data_data.get('date_ranges', {}).get('min_data_points', 50),
            min_ohlc_points=data_data.get('processing', {}).get('min_ohlc_points', 20),
            max_days_for_mapping=data_data.get('processing', {}).get('max_days_for_mapping', 14),
            resample_frequencies=data_data.get('processing', {}).get('resample_frequencies', {"day": "D", "week": "W", "month": "M"}),
            candlestick_filtering=data_data.get('processing', {}).get('candlestick_filtering', {"day_years": 3, "week_years": 10, "month_years": "all"})
        )
        
        self.elliott_wave = ElliottWaveConfig(
            min_points=elliott_data.get('detection', {}).get('min_points', 6),
            max_points=elliott_data.get('detection', {}).get('max_points', 12),
            min_confidence_threshold=elliott_data.get('detection', {}).get('min_confidence_threshold', 0.2),
            max_patterns_per_timeframe=elliott_data.get('detection', {}).get('max_patterns_per_timeframe', 3),
            max_patterns_total=elliott_data.get('detection', {}).get('max_patterns_total', 5),
            pattern_relationships_enabled=elliott_data.get('detection', {}).get('pattern_relationships_enabled', True),
            timeframes=elliott_data.get('timeframes', ["day", "week", "month"]),
            default_timeframe=elliott_data.get('timeframes', {}).get('default', "day"),
            timeframe_priority_weights=elliott_data.get('timeframes', {}).get('priority_weights', {"day": 0.5, "week": 0.3, "month": 0.2}),
            window_size=elliott_data.get('validation', {}).get('window_size', 5),
            min_prominence_pct=elliott_data.get('validation', {}).get('min_prominence_pct', 0.5),
            volume_threshold_ratio=elliott_data.get('validation', {}).get('volume_threshold_ratio', 0.7),
            fibonacci_tolerance=elliott_data.get('validation', {}).get('fibonacci_tolerance', 0.05),
            wave_equality_tolerance=elliott_data.get('validation', {}).get('wave_equality_tolerance', 0.1),
            overlap_tolerance=elliott_data.get('validation', {}).get('overlap_tolerance', 0.1),
            confidence_base=elliott_data.get('confidence', {}).get('base', 0.5),
            confidence_wave2_ideal=elliott_data.get('confidence', {}).get('wave2_ideal', 0.2),
            confidence_wave2_ok=elliott_data.get('confidence', {}).get('wave2_ok', 0.1),
            confidence_wave3_strong=elliott_data.get('confidence', {}).get('wave3_strong', 0.2),
            confidence_diagonal=elliott_data.get('confidence', {}).get('diagonal', 0.1),
            confidence_clean_impulse=elliott_data.get('confidence', {}).get('clean_impulse', 0.2),
            confidence_direction=elliott_data.get('confidence', {}).get('direction', 0.3),
            confidence_fibonacci=elliott_data.get('confidence', {}).get('fibonacci', 0.3),
            confidence_volume=elliott_data.get('confidence', {}).get('volume', 0.2),
            confidence_alternation=elliott_data.get('confidence', {}).get('alternation', 0.15),
            confidence_missing_data_penalty=elliott_data.get('confidence', {}).get('missing_data_penalty', 0.5),
            confidence_over_confidence_penalty=elliott_data.get('confidence', {}).get('over_confidence_penalty', 0.5)
        )
        
        self.fibonacci = FibonacciConfig(
            retracements=fibonacci_data.get('retracements', [0.236, 0.382, 0.5, 0.618, 0.786]),
            retracement_tolerance=fibonacci_data.get('retracements', {}).get('tolerance', 0.05),
            extensions=fibonacci_data.get('extensions', [1.618, 2.618, 4.236]),
            extension_tolerance=fibonacci_data.get('extensions', {}).get('tolerance', 0.1),
            projections=fibonacci_data.get('projections', [0.618, 1.0, 1.618]),
            projection_tolerance=fibonacci_data.get('projections', {}).get('tolerance', 0.05),
            time_ratios=fibonacci_data.get('time_ratios', [0.618, 1.0, 1.618, 2.618]),
            time_ratio_tolerance=fibonacci_data.get('time_ratios', {}).get('tolerance', 0.1),
            wave2_ideal_range=fibonacci_data.get('wave_relationships', {}).get('wave2_ideal_range', [0.382, 0.618]),
            wave2_acceptable_range=fibonacci_data.get('wave_relationships', {}).get('wave2_acceptable_range', [0.236, 0.786]),
            wave3_extension_range=fibonacci_data.get('wave_relationships', {}).get('wave3_extension_range', [1.618, 2.618]),
            wave4_retracement_range=fibonacci_data.get('wave_relationships', {}).get('wave4_retracement_range', [0.236, 0.5])
        )
        
        self.visualization = VisualizationConfig(
            figure_size=viz_data.get('chart_style', {}).get('figure_size', [12, 8]),
            dpi=viz_data.get('chart_style', {}).get('dpi', 100),
            face_color=viz_data.get('chart_style', {}).get('face_color', "white"),
            edge_color=viz_data.get('chart_style', {}).get('edge_color', "black"),
            grid_alpha=viz_data.get('chart_style', {}).get('grid_alpha', 0.3),
            grid_style=viz_data.get('chart_style', {}).get('grid_style', "-"),
            colors=viz_data.get('colors', {}),
            pattern_colors=viz_data.get('pattern_colors', {}),
            confidence_colors=viz_data.get('confidence_colors', {}),
            fonts=viz_data.get('fonts', {}),
            annotations=viz_data.get('annotations', {}),
            fibonacci_viz=viz_data.get('fibonacci_viz', {})
        )
        
        self.performance = PerformanceConfig(
            caching_enabled=perf_data.get('caching', {}).get('enabled', True),
            max_cache_size=perf_data.get('caching', {}).get('max_cache_size', 1000),
            cache_ttl_hours=perf_data.get('caching', {}).get('cache_ttl_hours', 24),
            pattern_cache_enabled=perf_data.get('caching', {}).get('pattern_cache_enabled', True),
            data_cache_enabled=perf_data.get('caching', {}).get('data_cache_enabled', True),
            max_workers=perf_data.get('threading', {}).get('max_workers', 4),
            thread_timeout_seconds=perf_data.get('threading', {}).get('thread_timeout_seconds', 300),
            enable_async_processing=perf_data.get('threading', {}).get('enable_async_processing', True),
            max_dataframe_size_mb=perf_data.get('memory', {}).get('max_dataframe_size_mb', 100),
            garbage_collection_threshold=perf_data.get('memory', {}).get('garbage_collection_threshold', 0.8),
            enable_memory_monitoring=perf_data.get('memory', {}).get('enable_memory_monitoring', True)
        )
        
        self.network = NetworkConfig(
            yahoo_finance_base_url=network_data.get('yahoo_finance', {}).get('base_url', "https://finance.yahoo.com/quote/"),
            retry_attempts=network_data.get('yahoo_finance', {}).get('retry_attempts', 3),
            retry_delay_ms=network_data.get('yahoo_finance', {}).get('retry_delay_ms', 2000),
            request_timeout_seconds=network_data.get('yahoo_finance', {}).get('request_timeout_seconds', 30),
            user_agent=network_data.get('yahoo_finance', {}).get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
            requests_per_second=network_data.get('rate_limiting', {}).get('requests_per_second', 2),
            delay_between_requests_ms=network_data.get('rate_limiting', {}).get('delay_between_requests_ms', 500),
            respect_robots_txt=network_data.get('rate_limiting', {}).get('respect_robots_txt', True)
        )
        
        self.backtesting = BacktestingConfig(
            default_buy_on=backtest_data.get('strategy', {}).get('default_buy_on', 0),
            default_sell_on=backtest_data.get('strategy', {}).get('default_sell_on', 5),
            min_price_changes=backtest_data.get('strategy', {}).get('min_price_changes', [0.01, 0.03, 0.05, 0.1]),
            default_min_price_change=backtest_data.get('strategy', {}).get('default_min_price_change', 0.02),
            calculate_sharpe_ratio=backtest_data.get('metrics', {}).get('calculate_sharpe_ratio', True),
            calculate_max_drawdown=backtest_data.get('metrics', {}).get('calculate_max_drawdown', True),
            calculate_win_rate=backtest_data.get('metrics', {}).get('calculate_win_rate', True),
            risk_free_rate=backtest_data.get('metrics', {}).get('risk_free_rate', 0.02)
        )
        
        self.logging = LoggingConfig(
            log_levels=logging_data.get('levels', {}),
            main_log_file=logging_data.get('files', {}).get('main_log', "logs/investment_system.log"),
            error_log_file=logging_data.get('files', {}).get('error_log', "logs/errors.log"),
            performance_log_file=logging_data.get('files', {}).get('performance_log', "logs/performance.log"),
            max_file_size_mb=logging_data.get('files', {}).get('max_file_size_mb', 10),
            backup_count=logging_data.get('files', {}).get('backup_count', 5),
            timestamp_format=logging_data.get('format', {}).get('timestamp_format', "%Y-%m-%d %H:%M:%S"),
            include_thread_id=logging_data.get('format', {}).get('include_thread_id', True),
            include_function_name=logging_data.get('format', {}).get('include_function_name', True)
        )
        
        self.error_handling = ErrorHandlingConfig(
            max_retry_attempts=error_data.get('recovery', {}).get('max_retry_attempts', 3),
            retry_delay_seconds=error_data.get('recovery', {}).get('retry_delay_seconds', 1),
            exponential_backoff=error_data.get('recovery', {}).get('exponential_backoff', True),
            log_all_errors=error_data.get('reporting', {}).get('log_all_errors', True),
            show_user_friendly_messages=error_data.get('reporting', {}).get('show_user_friendly_messages', True),
            include_stack_traces=error_data.get('reporting', {}).get('include_stack_traces', False),
            use_simple_plot_on_error=error_data.get('fallbacks', {}).get('use_simple_plot_on_error', True),
            use_cached_data_on_error=error_data.get('fallbacks', {}).get('use_cached_data_on_error', True),
            show_error_in_gui=error_data.get('fallbacks', {}).get('show_error_in_gui', True)
        )
        
        self.development = DevelopmentConfig(
            debug_enabled=dev_data.get('debug', {}).get('enabled', False),
            verbose_logging=dev_data.get('debug', {}).get('verbose_logging', False),
            show_debug_info=dev_data.get('debug', {}).get('show_debug_info', False),
            log_performance_metrics=dev_data.get('debug', {}).get('log_performance_metrics', True),
            use_mock_data=dev_data.get('testing', {}).get('use_mock_data', False),
            mock_data_path=dev_data.get('testing', {}).get('mock_data_path', "tests/mock_data/"),
            run_integration_tests=dev_data.get('testing', {}).get('run_integration_tests', False),
            profiling_enabled=dev_data.get('profiling', {}).get('enabled', False),
            profile_analysis_functions=dev_data.get('profiling', {}).get('profile_analysis_functions', False),
            profile_gui_operations=dev_data.get('profiling', {}).get('profile_gui_operations', False),
            profiling_output_file=dev_data.get('profiling', {}).get('output_file', "profiles/performance_profile.prof")
        )
    
    def get_fibonacci_retracements(self) -> List[float]:
        """Get Fibonacci retracement levels"""
        return self.fibonacci.retracements
    
    def get_fibonacci_extensions(self) -> List[float]:
        """Get Fibonacci extension levels"""
        return self.fibonacci.extensions
    
    def get_chart_types(self) -> List[str]:
        """Get available chart types"""
        return self.gui.chart_types
    
    def get_web_crawling_options(self) -> List[str]:
        """Get web crawling options"""
        return self.gui.web_crawling_options
    
    def get_timeframes(self) -> List[str]:
        """Get available timeframes for analysis"""
        return self.elliott_wave.timeframes
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color for confidence level"""
        if confidence >= 0.8:
            return self.visualization.confidence_colors.get('very_high', '#2E8B57')
        elif confidence >= 0.6:
            return self.visualization.confidence_colors.get('high', '#32CD32')
        elif confidence >= 0.4:
            return self.visualization.confidence_colors.get('moderate', '#FFA500')
        elif confidence >= 0.2:
            return self.visualization.confidence_colors.get('low', '#FF6347')
        else:
            return self.visualization.confidence_colors.get('very_low', '#DC143C')
    
    def get_pattern_color(self, pattern_type: str) -> str:
        """Get color for pattern type"""
        return self.visualization.pattern_colors.get(pattern_type, self.visualization.colors['primary'])
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        self._create_config_objects()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'gui': self.gui.__dict__,
            'data': self.data.__dict__,
            'elliott_wave': self.elliott_wave.__dict__,
            'fibonacci': self.fibonacci.__dict__,
            'visualization': self.visualization.__dict__,
            'performance': self.performance.__dict__,
            'network': self.network.__dict__,
            'backtesting': self.backtesting.__dict__,
            'logging': self.logging.__dict__,
            'error_handling': self.error_handling.__dict__,
            'development': self.development.__dict__
        }


# Global configuration instance
@lru_cache(maxsize=1)
def get_config() -> ConfigManager:
    """
    Get the global configuration instance (singleton pattern).
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager()


# Convenience functions for backward compatibility
def get_fibonacci_retracements() -> List[float]:
    """Get Fibonacci retracement levels"""
    return get_config().get_fibonacci_retracements()


def get_fibonacci_extensions() -> List[float]:
    """Get Fibonacci extension levels"""
    return get_config().get_fibonacci_extensions()


def get_chart_types() -> List[str]:
    """Get available chart types"""
    return get_config().get_chart_types()


def get_web_crawling_options() -> List[str]:
    """Get web crawling options"""
    return get_config().get_web_crawling_options()


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence level"""
    return get_config().get_confidence_color(confidence)


def get_pattern_color(pattern_type: str) -> str:
    """Get color for pattern type"""
    return get_config().get_pattern_color(pattern_type) 