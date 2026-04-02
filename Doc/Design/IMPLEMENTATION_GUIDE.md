# Elliott Wave Investment System - Implementation Guide

## Overview

This guide explains how to integrate the newly implemented investment-grade enhancements into your Elliott Wave analysis system. These components transform academic pattern recognition into a practical trading system with:

✅ **Multiple wave count management** - Handle uncertainty with alternative interpretations
✅ **Fibonacci cluster analysis** - Identify high-probability convergence zones
✅ **Automated trade setup generation** - Concrete entry/exit rules
✅ **Wave-aware position sizing** - Adjust sizing based on wave characteristics
✅ **Confidence calibration** - Validate scores against actual outcomes

---

## New Components

### 1. Multiple Wave Count Manager
**File:** `src/analysis/core/multiple_counts.py`

**Purpose:** Manages competing Elliott Wave interpretations, as real-world analysis always has multiple valid counts.

**Key Features:**
- Track 3-5 alternative wave scenarios simultaneously
- Dynamic confidence updates as new price data arrives
- Automatic invalidation when price violates wave rules
- Consensus target calculation from all scenarios
- Scenario ranking by confidence

**Example Usage:**
```python
from src.analysis.core.multiple_counts import MultipleCountManager, WaveCountScenario

# Initialize manager
manager = MultipleCountManager(max_scenarios=5)

# Add primary scenario
primary = WaveCountScenario(
    wave_points=np.array([0, 20, 15, 45, 40, 60]),
    wave_type=WaveType.IMPULSE,
    degree=WaveDegree.MINOR,
    confidence=0.75,
    validation_details={'base_confidence': 0.75},
    invalidation_level=100.0,
    targets=[150.0, 160.0, 170.0],
    current_wave=3,
    pattern_name="Primary Impulse"
)
manager.add_scenario(primary)

# Update with new data
update = manager.update_with_new_data(
    latest_price=138.0,
    latest_time=datetime.now(),
    df=price_dataframe
)

# Get consensus targets
consensus = manager.get_consensus_targets()
print(f"Consensus target: ${consensus['consensus_targets'][0]['price']:.2f}")
print(f"Scenario agreement: {consensus['scenario_agreement']:.1%}")

# Get human-readable summary
print(manager.get_scenario_summary())
```

---

### 2. Fibonacci Cluster Analyzer
**File:** `src/analysis/core/fibonacci_clusters.py`

**Purpose:** Identifies price zones where multiple Fibonacci projections converge - statistically significant turning points.

**Key Features:**
- Retracements, extensions, and projections from multiple waves
- Cluster detection (3+ methods converging)
- Time-price synchronicity detection
- Support/resistance identification
- Trading recommendations based on clusters

**Example Usage:**
```python
from src.analysis.core.fibonacci_clusters import FibonacciClusterAnalyzer

analyzer = FibonacciClusterAnalyzer(tolerance=0.02)  # 2% clustering tolerance

# Find clusters from wave analysis
clusters = analyzer.find_clusters(wave_data, current_price=138.0)

print(f"Found {len(clusters)} significant clusters")
for cluster in clusters[:3]:
    print(f"  {cluster.central_price:.2f}: {cluster.significance} methods converging")

# Get comprehensive analysis
analysis = analyzer.analyze_wave_for_clusters(df, wave_points, current_wave=3)
print(f"Next targets: {analysis['next_targets']}")
print(f"Recommendation:\n{analysis['recommendation']}")
```

---

### 3. Trade Setup Generator
**File:** `src/backtest/trade_setup_generator.py`

**Purpose:** Converts wave analysis into specific, actionable trade setups with entry/exit rules.

**Key Features:**
- Wave 3 entry setups (highest probability)
- Wave 5 exit signals
- Correction entry at Fibonacci levels
- Risk/reward calculations
- Entry condition validation
- Signal strength classification

**Example Usage:**
```python
from src.backtest.trade_setup_generator import TradeSetupGenerator

generator = TradeSetupGenerator({
    'stop_buffer_pct': 0.02,
    'min_risk_reward': 2.0
})

# Scan for all available setups
setups = generator.scan_for_setups(df, wave_data, fib_clusters)

for setup in setups:
    print(f"\n{setup.setup_type.value.upper()}")
    print(f"Signal: {setup.signal_strength.value}")
    print(f"Confidence: {setup.confidence:.1%}")
    print(f"Entry: ${setup.entry_price:.2f}")
    print(f"Stop: ${setup.stop_loss:.2f}")
    print(f"Targets: {[f'${t:.2f}' for t in setup.targets]}")
    print(f"R:R: {setup.risk_reward_ratio:.2f}")
    print(f"Conditions:")
    for cond in setup.entry_conditions:
        print(f"  ✓ {cond}")
```

---

### 4. Wave-Aware Position Sizer
**File:** `src/utils/wave_aware_sizing.py`

**Purpose:** Adjusts position sizing based on wave characteristics and probability.

**Key Features:**
- Wave-specific multipliers (Wave 3 max, Wave 4 min)
- Timeframe alignment bonus
- Fibonacci confluence adjustments
- Progressive entry planning for Wave 3
- Aggressive sizing for high-conviction setups

**Example Usage:**
```python
from src.utils.wave_aware_sizing import (
    WaveAwarePositionSizer,
    WavePositionMetrics
)

# Initialize sizer
sizer = WaveAwarePositionSizer(
    portfolio_value=100000,
    max_risk_per_trade=0.02
)

# Define wave metrics
metrics = WavePositionMetrics(
    wave_number=3,
    wave_type='impulse',
    confidence=0.80,
    is_diagonal=False,
    complex_correction=False,
    timeframe_alignment=0.85,
    fibonacci_confluence=4
)

# Calculate position size
shares = sizer.calculate_wave_adjusted_size(
    entry_price=150.0,
    stop_loss=145.0,
    wave_metrics=metrics
)

print(f"Position size: {shares} shares")
print(f"Position value: ${shares * 150:.2f}")

# Get progressive entry plan for Wave 3
stages = sizer.calculate_progressive_entry_sizes(150.0, 145.0, metrics)
for stage in stages:
    print(f"Stage {stage['stage']}: {stage['shares']} shares at {stage['condition']}")
```

---

### 5. Confidence Calibrator
**File:** `src/analysis/core/confidence_calibrator.py`

**Purpose:** Validates confidence scores against actual outcomes to ensure reliability.

**Key Features:**
- Prediction tracking with outcomes
- Calibration analysis (expected vs. actual)
- Brier score and log loss metrics
- Bucket-specific calibration
- Adjustment recommendations
- Historical persistence

**Example Usage:**
```python
from src.analysis.core.confidence_calibrator import ConfidenceCalibrator
from pathlib import Path

# Initialize with persistence
calibrator = ConfidenceCalibrator(
    storage_path=Path('data/calibration_history.json')
)

# Record prediction
record = calibrator.record_prediction(
    symbol='AAPL',
    wave_analysis={
        'current_wave': 3,
        'confidence': 0.75,
        'wave_type': 'impulse',
        'pattern_name': 'Primary Impulse'
    },
    current_price=150.0
)

# Later, update with outcome
outcome = {
    'actual_wave': 3,
    'target_reached': True,
    'pattern_valid': True,
    'invalidation_triggered': False
}
calibrator.update_outcome(0, outcome, current_price=165.0)

# Analyze calibration
calibration = calibrator.calibrate()
print(f"Expected accuracy: {calibration.expected_accuracy:.1%}")
print(f"Actual accuracy: {calibration.actual_accuracy:.1%}")
print(f"Well calibrated: {calibration.is_well_calibrated}")

# Generate full report
print(calibrator.generate_report())

# Save history
calibrator.save_history()
```

---

## Integration Workflow

### Step 1: Enhanced Wave Analysis with Multiple Counts

```python
from src.analysis.elliott_wave import find_elliott_wave_pattern_enhanced
from src.analysis.core.multiple_counts import MultipleCountManager
from src.analysis.core.fibonacci_clusters import FibonacciClusterAnalyzer

# Run standard Elliott Wave analysis
primary_wave = find_elliott_wave_pattern_enhanced(df)

# Initialize managers
count_manager = MultipleCountManager()
fib_analyzer = FibonacciClusterAnalyzer()

# Add primary count
primary_scenario = create_scenario_from_wave_data(primary_wave)
count_manager.add_scenario(primary_scenario)

# Find alternative counts (you would implement alternative pattern detection)
# For now, use variations with slightly different wave points
alternatives = generate_alternative_counts(df, primary_wave)
for alt in alternatives:
    count_manager.add_scenario(alt)

# Find Fibonacci clusters
current_price = df['close'].iloc[-1]
clusters = fib_analyzer.find_clusters(primary_wave, current_price)

print(f"Active scenarios: {len(count_manager.scenarios)}")
print(f"Fibonacci clusters: {len(clusters)}")
```

### Step 2: Generate Trade Setups

```python
from src.backtest.trade_setup_generator import TradeSetupGenerator

generator = TradeSetupGenerator({
    'stop_buffer_pct': 0.02,
    'min_risk_reward': 2.0
})

# Scan for setups using primary wave and clusters
setups = generator.scan_for_setups(df, primary_wave, clusters)

if setups:
    best_setup = setups[0]  # Highest confidence
    print(f"Best setup: {best_setup.setup_type.value}")
    print(f"Entry: ${best_setup.entry_price:.2f}")
    print(f"R:R: {best_setup.risk_reward_ratio:.2f}")
```

### Step 3: Calculate Position Size

```python
from src.utils.wave_aware_sizing import WaveAwarePositionSizer, WavePositionMetrics

sizer = WaveAwarePositionSizer(
    portfolio_value=100000,
    max_risk_per_trade=0.02
)

# Build metrics from wave data
metrics = WavePositionMetrics(
    wave_number=primary_wave['current_wave'],
    wave_type=primary_wave.get('wave_type', 'impulse'),
    confidence=primary_wave['confidence'],
    is_diagonal=primary_wave.get('is_diagonal', False),
    complex_correction=False,
    timeframe_alignment=0.80,  # Would calculate from multi-timeframe analysis
    fibonacci_confluence=len([c for c in clusters if c.significance >= 3])
)

shares = sizer.calculate_wave_adjusted_size(
    best_setup.entry_price,
    best_setup.stop_loss,
    metrics
)

print(f"Position size: {shares} shares")
```

### Step 4: Record Prediction for Calibration

```python
from src.analysis.core.confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator(Path('data/calibration_history.json'))

record = calibrator.record_prediction(
    symbol='AAPL',
    wave_analysis=primary_wave,
    current_price=df['close'].iloc[-1]
)

# Store record index for later outcome update
prediction_tracker[('AAPL', datetime.now())] = len(calibrator.predictions) - 1
```

### Step 5: Execute Trade (Paper or Live)

```python
from src.utils.risk_management import PortfolioRiskManager, Position

portfolio = PortfolioRiskManager(
    initial_capital=100000,
    config={
        'max_portfolio_exposure': 0.40,
        'max_daily_drawdown': 0.05,
        'max_total_drawdown': 0.20,
        'max_concurrent_positions': 5
    }
)

# Check if can open position
can_open, reason = portfolio.can_open_position(shares * best_setup.entry_price)

if can_open:
    position = Position(
        symbol='AAPL',
        entry_date=datetime.now(),
        entry_price=best_setup.entry_price,
        shares=shares,
        stop_loss=best_setup.stop_loss,
        profit_target=best_setup.targets[0],
        wave_number=primary_wave['current_wave'],
        confidence=primary_wave['confidence'],
        strategy='elliott_wave_enhanced'
    )

    portfolio.add_position(position)
    print(f"Position opened: {shares} shares @ ${best_setup.entry_price:.2f}")
else:
    print(f"Cannot open position: {reason}")
```

### Step 6: Monitor and Update

```python
# Daily update loop
while position_is_open:
    # Update wave counts with new data
    update_result = count_manager.update_with_new_data(
        latest_price=current_price,
        latest_time=datetime.now(),
        df=df
    )

    if update_result['invalidated_count'] > 0:
        print(f"WARNING: {update_result['invalidated_count']} scenarios invalidated")

        # Re-evaluate position
        if not count_manager.scenarios:
            print("All scenarios invalidated - consider exit")

    # Check if targets reached or stop hit
    if current_price >= best_setup.targets[0]:
        print("Target 1 reached - take partial profits")

    if current_price <= best_setup.stop_loss:
        print("Stop loss hit - exit position")
        break
```

---

## Configuration Recommendations

### For Conservative Trading

```python
config = {
    # Position sizing
    'max_risk_per_trade': 0.01,  # 1% per trade
    'wave_multipliers': {
        3: 0.8,  # Even Wave 3 reduced
        5: 0.3,
        4: 0.2
    },

    # Trade generation
    'min_risk_reward': 3.0,  # Higher R:R required
    'alignment_threshold': 0.80,  # Require strong alignment

    # Multiple counts
    'min_scenarios': 2,  # Need at least 2 valid counts
    'min_agreement': 0.70  # Scenarios must mostly agree
}
```

### For Aggressive Trading (Higher Risk)

```python
config = {
    # Position sizing
    'max_risk_per_trade': 0.02,  # 2% per trade
    'wave_multipliers': {
        3: 1.2,  # Overweight Wave 3
        5: 0.6,
        4: 0.3
    },

    # Trade generation
    'min_risk_reward': 2.0,
    'alignment_threshold': 0.65,

    # Multiple counts
    'min_scenarios': 1,
    'min_agreement': 0.60
}
```

---

## Testing & Validation Workflow

### 1. Backtest with Enhanced System

```python
from src.backtest.backtester import Backtester

# Create enhanced backtester that uses new components
backtester = EnhancedElliottWaveBacktester(
    initial_capital=100000,
    use_multiple_counts=True,
    use_fibonacci_clusters=True,
    use_wave_sizing=True
)

results = backtester.run('AAPL', start_date='2020-01-01', end_date='2024-01-01')

print(f"Total return: {results['total_return']:.1%}")
print(f"Win rate: {results['win_rate']:.1%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Max drawdown: {results['max_drawdown']:.1%}")
```

### 2. Calibrate on Historical Data

```python
# Run calibration on completed backtests
calibrator = ConfidenceCalibrator()

for trade in backtest_results['trades']:
    # Record prediction
    record = calibrator.record_prediction(
        symbol=trade['symbol'],
        wave_analysis=trade['wave_analysis'],
        current_price=trade['entry_price']
    )

    # Update outcome
    outcome = {
        'pattern_valid': trade['profit'] > 0,
        'target_reached': trade['exit_reason'] == 'target',
        'actual_wave': trade['wave_number']
    }
    calibrator.update_outcome(
        len(calibrator.predictions) - 1,
        outcome,
        trade['exit_price']
    )

# Analyze
calibration = calibrator.calibrate()
print(calibrator.generate_report())

# Apply suggested adjustments
suggestions = calibrator.suggest_adjustments(calibration)
if suggestions['overall_bias'] == 'overconfident':
    print("Adjusting confidence scores downward...")
    # Apply adjustments to validation weights
```

---

## Next Steps for Production

1. **Multi-Timeframe Analysis**
   - Implement fractal wave analyzer
   - Check alignment across daily, weekly, hourly
   - Boost confidence when timeframes agree

2. **Real-Time Data Integration**
   - Connect to live data feed
   - Implement continuous monitoring
   - Alert system for trade setups

3. **Paper Trading**
   - Test with paper trading account
   - Track all predictions and outcomes
   - Build calibration database

4. **Performance Dashboard**
   - Visualize active scenarios
   - Show Fibonacci clusters on charts
   - Display position sizing recommendations
   - Track calibration metrics over time

5. **Risk Management Enhancements**
   - Portfolio-level wave exposure tracking
   - Correlation analysis between positions
   - Dynamic stop adjustment system

---

## Key Principles for Success

1. **Embrace Uncertainty**
   - Always maintain 2-4 alternative counts
   - Never assume single interpretation is correct
   - Update beliefs as new data arrives

2. **Trust the Process**
   - Follow position sizing rules strictly
   - Don't override system based on emotion
   - Use calibration to improve over time

3. **Respect Wave Structure**
   - Wave 3 is highest probability - size accordingly
   - Wave 4 is dangerous - minimize exposure
   - Wave 5 is exhaustion - take profits

4. **Validate Everything**
   - Track all predictions and outcomes
   - Regularly review calibration
   - Adjust when evidence suggests bias

5. **Manage Risk First**
   - Position sizing is more important than entry
   - Always use stops
   - Never risk more than configured limits

---

## Support & Resources

- **Elliott Wave Principle** (Frost & Prechter) - The definitive guide
- **Practical Elliott Wave Trading Strategies** (Rule) - Practical applications
- Your existing documentation: `STRATEGY_QUICKSTART.md` and `PERFORMANCE_OPTIMIZATION_GUIDE.md`

## Questions?

Review the code examples in each module's `if __name__ == "__main__"` section for working demonstrations of all features.
