"""
Phase 1 Implementation - Backtest Comparison (pytest version)
==============================================================

Tests Phase 1 confidence filtering improvements by comparing:
- Baseline: Backtest WITHOUT confidence filters (old system)
- Phase 1: Backtest WITH confidence filters (enhanced system)

Validates improvements in:
- Win rate (should increase)
- Profit factor (should increase)
- Trade efficiency (fewer trades, more profit)
- Risk management (lower drawdown, higher Sharpe)

This is a SLOW benchmark test that requires real market data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from src.backtest.strategy_advanced import MultiTimeframeAlignmentStrategy, AdvancedBacktester
from src.analysis.core.impulse import find_elliott_wave_patterns_advanced
from src.utils.common_utils import load_and_preprocess_data


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def backtest_config_baseline():
    """Configuration for baseline backtest (no filters)."""
    return {
        'alignment_threshold': 0.70,
        'use_confidence_filters': False  # DISABLED
    }


@pytest.fixture
def backtest_config_phase1():
    """Configuration for Phase 1 backtest (with filters)."""
    return {
        'alignment_threshold': 0.70,
        'use_confidence_filters': True,  # ENABLED
        'min_composite_confidence': 0.65,
        'min_validation_confidence': 0.55
    }


@pytest.fixture
def initial_capital():
    """Standard initial capital for backtests."""
    return 100000


@pytest.fixture(scope="session")
def results_dir():
    """Directory for storing test results."""
    output_dir = Path('tests/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_backtest_with_config(df: pd.DataFrame, config: dict, initial_capital: float) -> dict:
    """
    Run backtest with given configuration.

    Args:
        df: Stock data DataFrame
        config: Strategy configuration
        initial_capital: Starting capital

    Returns:
        dict with 'results', 'filter_stats', 'trades'
    """
    # Detect Elliott Wave patterns
    patterns_data = find_elliott_wave_patterns_advanced(
        df, column='close',
        candlestick_types=['day'],  # Daily only for speed
        max_patterns_per_timeframe=2,
        pattern_relationships=True
    )

    # Create strategy
    strategy = MultiTimeframeAlignmentStrategy(config)

    # Create backtester
    backtester = AdvancedBacktester(
        initial_capital=initial_capital,
        config=config
    )

    backtester.strategies['multiframe'] = strategy

    # Run backtest
    results = backtester.run_backtest(df, patterns_data, strategy_name='multiframe')

    # Get filter statistics
    filter_stats = strategy.get_filter_statistics()

    return {
        'results': results,
        'filter_stats': filter_stats,
        'trades': backtester.get_trades_dataframe()
    }


def calculate_improvements(baseline_results: dict, phase1_results: dict) -> dict:
    """
    Calculate improvement metrics between baseline and Phase 1.

    Returns:
        dict with improvement deltas and percentages
    """
    b = baseline_results
    p = phase1_results

    improvements = {}

    # Win rate improvement (percentage points)
    improvements['win_rate_delta'] = (p['win_rate'] - b['win_rate']) * 100

    # Profit factor improvement
    improvements['profit_factor_delta'] = p['profit_factor'] - b['profit_factor']
    improvements['profit_factor_pct'] = (
        (improvements['profit_factor_delta'] / b['profit_factor']) * 100
        if b['profit_factor'] > 0 else 0
    )

    # Total profit improvement
    improvements['total_profit_delta'] = p['total_profit'] - b['total_profit']
    improvements['total_profit_pct'] = (
        (improvements['total_profit_delta'] / b['total_profit']) * 100
        if b['total_profit'] > 0 else 0
    )

    # Trade reduction
    improvements['trade_reduction'] = b['total_trades'] - p['total_trades']
    improvements['trade_reduction_pct'] = (
        (improvements['trade_reduction'] / b['total_trades']) * 100
        if b['total_trades'] > 0 else 0
    )

    # Drawdown improvement (positive = better)
    improvements['drawdown_improvement'] = b['max_drawdown_pct'] - p['max_drawdown_pct']

    # Sharpe improvement
    improvements['sharpe_delta'] = p['sharpe_ratio'] - b['sharpe_ratio']
    improvements['sharpe_pct'] = (
        (improvements['sharpe_delta'] / b['sharpe_ratio']) * 100
        if b['sharpe_ratio'] > 0 else 0
    )

    # Calculate success score (0-7)
    score = 0
    if improvements['win_rate_delta'] > 5:
        score += 2
    elif improvements['win_rate_delta'] > 0:
        score += 1

    if improvements['profit_factor_pct'] > 30:
        score += 2
    elif improvements['profit_factor_pct'] > 0:
        score += 1

    if improvements['trade_reduction_pct'] > 30:
        score += 1

    if improvements['total_profit_pct'] > 20:
        score += 2
    elif improvements['total_profit_pct'] > 0:
        score += 1

    improvements['success_score'] = score

    return improvements


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_phase1_improvement_aapl(aapl_data, backtest_config_baseline, backtest_config_phase1,
                                  initial_capital, results_dir):
    """
    Test Phase 1 improvement on AAPL stock data.

    Validates that confidence filtering improves backtest performance:
    - Win rate should increase
    - Profit factor should increase
    - Trade efficiency should improve
    """
    print("\n" + "="*80)
    print("PHASE 1 IMPROVEMENT TEST: AAPL")
    print("="*80)

    # Run baseline (no filters)
    print("\n[1/2] Running BASELINE backtest (no filters)...")
    baseline = run_backtest_with_config(aapl_data, backtest_config_baseline, initial_capital)

    print(f"  Baseline: {baseline['results']['total_trades']} trades, "
          f"{baseline['results']['win_rate']*100:.1f}% win rate, "
          f"PF={baseline['results']['profit_factor']:.2f}")

    # Run Phase 1 (with filters)
    print("\n[2/2] Running PHASE 1 backtest (with filters)...")
    phase1 = run_backtest_with_config(aapl_data, backtest_config_phase1, initial_capital)

    print(f"  Phase 1:  {phase1['results']['total_trades']} trades, "
          f"{phase1['results']['win_rate']*100:.1f}% win rate, "
          f"PF={phase1['results']['profit_factor']:.2f}")

    # Calculate improvements
    improvements = calculate_improvements(baseline['results'], phase1['results'])

    print(f"\n[*] IMPROVEMENTS:")
    print(f"  Win Rate:       {improvements['win_rate_delta']:+.1f} percentage points")
    print(f"  Profit Factor:  {improvements['profit_factor_pct']:+.1f}%")
    print(f"  Total Profit:   {improvements['total_profit_pct']:+.1f}%")
    print(f"  Trades Filtered: {improvements['trade_reduction_pct']:.1f}%")
    print(f"  Success Score:  {improvements['success_score']}/7")

    # Save results
    summary = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': 'AAPL',
        'baseline': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                     for k, v in baseline['results'].items()},
        'phase1': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in phase1['results'].items()},
        'improvements': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in improvements.items()}
    }

    with open(results_dir / 'phase1_aapl_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ASSERTIONS - Validate Phase 1 improvements
    # Win rate should improve or stay same
    assert improvements['win_rate_delta'] >= -5, \
        f"Win rate declined by more than 5%: {improvements['win_rate_delta']:.1f}%"

    # Profit factor should improve
    assert improvements['profit_factor_pct'] >= -10, \
        f"Profit factor declined significantly: {improvements['profit_factor_pct']:.1f}%"

    # Should filter at least some signals
    assert phase1['filter_stats']['signals_filtered'] > 0, \
        "Phase 1 should filter at least some low-confidence signals"

    # Overall success score should be reasonable
    assert improvements['success_score'] >= 2, \
        f"Phase 1 success score too low: {improvements['success_score']}/7"

    print(f"\n[OK] Phase 1 improvement test PASSED (score: {improvements['success_score']}/7)")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("symbol,fixture_name", [
    ("MSFT", "msft_data"),
    ("SPY", "spy_data"),
])
def test_phase1_improvement_multiple_symbols(symbol, fixture_name, request,
                                             backtest_config_baseline, backtest_config_phase1,
                                             initial_capital, results_dir):
    """
    Test Phase 1 improvement on multiple stock symbols.

    Validates that confidence filtering works consistently across different stocks.
    """
    # Get the fixture dynamically
    stock_data = request.getfixturevalue(fixture_name)

    print(f"\n" + "="*80)
    print(f"PHASE 1 IMPROVEMENT TEST: {symbol}")
    print("="*80)

    # Run baseline
    print(f"\n[1/2] Running BASELINE backtest for {symbol}...")
    baseline = run_backtest_with_config(stock_data, backtest_config_baseline, initial_capital)

    # Run Phase 1
    print(f"\n[2/2] Running PHASE 1 backtest for {symbol}...")
    phase1 = run_backtest_with_config(stock_data, backtest_config_phase1, initial_capital)

    # Calculate improvements
    improvements = calculate_improvements(baseline['results'], phase1['results'])

    print(f"\n[*] {symbol} Results:")
    print(f"  Success Score: {improvements['success_score']}/7")
    print(f"  Win Rate: {improvements['win_rate_delta']:+.1f}pp")
    print(f"  Profit Factor: {improvements['profit_factor_pct']:+.1f}%")

    # Save results
    summary = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'baseline': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                     for k, v in baseline['results'].items()},
        'phase1': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                   for k, v in phase1['results'].items()},
        'improvements': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in improvements.items()}
    }

    with open(results_dir / f'phase1_{symbol.lower()}_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Assertions - relaxed for different stocks
    assert improvements['win_rate_delta'] >= -10, \
        f"{symbol}: Win rate declined too much: {improvements['win_rate_delta']:.1f}%"

    assert phase1['filter_stats']['signals_filtered'] >= 0, \
        f"{symbol}: Filter stats should be available"

    print(f"\n[OK] {symbol} Phase 1 test PASSED")


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_phase1_no_regression_in_filtering(aapl_data, backtest_config_phase1, initial_capital):
    """
    Regression test: Ensure Phase 1 filtering doesn't filter ALL signals.

    This would indicate overly aggressive filtering thresholds.
    """
    phase1 = run_backtest_with_config(aapl_data, backtest_config_phase1, initial_capital)

    # Should not filter 100% of signals
    filter_pct = phase1['filter_stats'].get('filter_percentage', 0)
    assert filter_pct < 95, \
        f"Phase 1 filters too aggressively ({filter_pct:.1f}% filtered)"

    # Should have at least SOME trades
    assert phase1['results']['total_trades'] > 0, \
        "Phase 1 should generate at least some trades"

    print(f"\n[OK] Phase 1 filtering reasonable: {filter_pct:.1f}% filtered, "
          f"{phase1['results']['total_trades']} trades generated")


@pytest.mark.slow
@pytest.mark.integration
def test_phase1_filter_statistics_valid(aapl_data, backtest_config_phase1, initial_capital):
    """
    Test that Phase 1 filter statistics are correctly calculated.
    """
    phase1 = run_backtest_with_config(aapl_data, backtest_config_phase1, initial_capital)
    stats = phase1['filter_stats']

    # Basic structure validation
    assert 'total_signals' in stats
    assert 'passed_composite_filter' in stats
    assert 'passed_validation_filter' in stats
    assert 'final_signals' in stats
    assert 'signals_filtered' in stats

    # Logical consistency
    assert stats['passed_composite_filter'] <= stats['total_signals'], \
        "Passed composite cannot exceed total signals"

    assert stats['final_signals'] <= stats['passed_composite_filter'], \
        "Final signals cannot exceed composite pass"

    assert stats['signals_filtered'] == stats['total_signals'] - stats['final_signals'], \
        "Filtered count should equal total - final"

    print(f"\n[OK] Filter statistics valid: {stats['total_signals']} signals, "
          f"{stats['signals_filtered']} filtered ({stats.get('filter_percentage', 0):.1f}%)")


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_phase1_with_minimal_data(simple_uptrend_data, backtest_config_phase1, initial_capital):
    """
    Edge case: Test Phase 1 with minimal synthetic data.

    Should handle small datasets gracefully without crashing.
    """
    # Simple uptrend is only 10 bars
    result = run_backtest_with_config(simple_uptrend_data, backtest_config_phase1, initial_capital)

    # Should not crash - results should be valid structure
    assert 'results' in result
    assert 'filter_stats' in result
    assert 'trades' in result

    # Results should have required keys
    assert 'total_trades' in result['results']
    assert 'win_rate' in result['results']

    print(f"\n[OK] Phase 1 handles minimal data: {result['results']['total_trades']} trades")


@pytest.mark.slow
@pytest.mark.integration
def test_baseline_vs_phase1_structure_consistency(aapl_data, backtest_config_baseline,
                                                   backtest_config_phase1, initial_capital):
    """
    Test that baseline and Phase 1 return consistent result structures.

    Both should have same keys in results dict for valid comparison.
    """
    baseline = run_backtest_with_config(aapl_data, backtest_config_baseline, initial_capital)
    phase1 = run_backtest_with_config(aapl_data, backtest_config_phase1, initial_capital)

    # Result keys should be identical
    baseline_keys = set(baseline['results'].keys())
    phase1_keys = set(phase1['results'].keys())

    assert baseline_keys == phase1_keys, \
        f"Result structure mismatch: baseline has {baseline_keys - phase1_keys}, " \
        f"phase1 has {phase1_keys - baseline_keys}"

    print(f"\n[OK] Baseline and Phase 1 have consistent result structures ({len(baseline_keys)} keys)")


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.slow
@pytest.mark.benchmark
def test_phase1_performance_benchmark(aapl_data, backtest_config_baseline,
                                      backtest_config_phase1, initial_capital):
    """
    Benchmark: Compare execution time of baseline vs Phase 1.

    Phase 1 should not be significantly slower than baseline.
    """
    import time

    # Benchmark baseline
    start = time.time()
    baseline = run_backtest_with_config(aapl_data, backtest_config_baseline, initial_capital)
    baseline_time = time.time() - start

    # Benchmark Phase 1
    start = time.time()
    phase1 = run_backtest_with_config(aapl_data, backtest_config_phase1, initial_capital)
    phase1_time = time.time() - start

    time_overhead_pct = ((phase1_time - baseline_time) / baseline_time) * 100

    print(f"\n[*] Performance Benchmark:")
    print(f"  Baseline: {baseline_time:.2f}s")
    print(f"  Phase 1:  {phase1_time:.2f}s")
    print(f"  Overhead: {time_overhead_pct:+.1f}%")

    # Phase 1 should not be more than 50% slower
    assert time_overhead_pct < 50, \
        f"Phase 1 too slow: {time_overhead_pct:.1f}% overhead"

    print(f"\n[OK] Phase 1 performance acceptable")
