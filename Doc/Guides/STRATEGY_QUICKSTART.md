# Investment Strategy Quick Start Guide

## Overview

This guide helps you get started with the investment strategies created for your Elliott Wave analysis system.

## What Has Been Created

### 1. **INVESTMENT_STRATEGY.md** - Complete Strategy Documentation
   - 4 detailed trading strategies optimized for your Elliott Wave system
   - Risk management protocols
   - Position sizing guidelines
   - Backtesting requirements
   - Implementation roadmap

### 2. **src/backtest/strategy_advanced.py** - Advanced Strategy Implementation
   - `MultiTimeframeAlignmentStrategy` - Primary strategy (⭐⭐⭐)
   - `FibonacciMeanReversionStrategy` - Supplementary strategy (⭐⭐)
   - `PatternBreakoutStrategy` - Aggressive strategy (⭐⭐⭐)
   - `AdvancedBacktester` - Full backtesting engine with risk management
   - `PositionManager` - Position sizing logic
   - `RiskManager` - Portfolio risk controls

### 3. **config/strategy_config.json** - Strategy Configuration
   - All strategy parameters
   - Risk management settings
   - Portfolio allocation rules
   - Elliott Wave preferences
   - Backtesting configuration

### 4. **src/utils/risk_management.py** - Risk Management Utilities
   - `PositionSizer` - 4 position sizing methods (risk-based, fixed %, volatility, Kelly)
   - `StopLossCalculator` - 6 stop loss calculation methods
   - `PortfolioRiskManager` - Overall portfolio risk tracking
   - Helper functions for Sharpe ratio, Sortino ratio, max drawdown

---

## Quick Start - Run Your First Backtest

### Step 1: Test the Risk Management Module

```bash
# Run the risk management utilities example
python src/utils/risk_management.py
```

This will show you:
- Position sizing calculations
- Stop loss calculations
- Portfolio risk checks

### Step 2: Run a Simple Backtest

Create a test script `test_strategy.py`:

```python
import pandas as pd
from src.backtest.strategy_advanced import AdvancedBacktester
from src.crawler.yahoo_finance import YahooFinanceCrawler
import json

# Load configuration
with open('config/strategy_config.json', 'r') as f:
    config = json.load(f)

# Initialize crawler and fetch data
crawler = YahooFinanceCrawler('data/raw')
df = crawler.fetch_data('AAPL', '2020-01-01', '2025-11-15')

# For testing, create dummy pattern analysis
# (In production, this comes from your Elliott Wave analyzer)
pattern_analysis = {
    'alignment_score': 0.75,
    'current_position': {
        'wave_number': 3,
        'wave_1_high': df['High'].iloc[50],
        'wave_2_low': df['Low'].iloc[75],
        'impulse_high': df['High'].iloc[50],
        'impulse_low': df['Low'].iloc[0],
        'impulse_avg_volume': df['Volume'].iloc[0:50].mean()
    }
}

# Initialize backtester
backtester = AdvancedBacktester(
    initial_capital=100000,
    config=config['strategies']['multiframe_alignment']['parameters']
)

# Run backtest
results = backtester.run_backtest(
    df=df,
    pattern_analysis=pattern_analysis,
    strategy_name='multiframe'
)

# Print results
backtester.print_summary()

# Get detailed trades
trades_df = backtester.get_trades_dataframe()
if not trades_df.empty:
    print("\nDetailed Trades:")
    print(trades_df[['entry_date', 'exit_date', 'profit', 'profit_pct', 'exit_reason']])
```

Run it:
```bash
python test_strategy.py
```

### Step 3: Integrate with Your Elliott Wave Analyzer

Modify your existing `gui/handlers.py` to use the new strategies:

```python
# Add to imports
from src.backtest.strategy_advanced import AdvancedBacktester
import json

# Modify handle_run_backtest function
@run_in_thread
def handle_run_backtest(self, event):
    """Run advanced backtest with full risk management."""
    try:
        symbol = self.combo_stock.GetValue()
        if symbol == "Select Stock":
            wx.PostEvent(self, UpdateOutputEvent(message="Please select a stock first.\n"))
            return

        # Load strategy configuration
        with open('config/strategy_config.json', 'r') as f:
            strategy_config = json.load(f)

        wx.PostEvent(self, UpdateOutputEvent(
            message=f"Running advanced backtest for {symbol}...\n"))

        # Get price data
        df = load_stock_data(symbol)  # Your existing data loading function

        # Run Elliott Wave analysis
        from src.analysis.elliott_wave import analyze_patterns_multiframe
        pattern_analysis = analyze_patterns_multiframe(
            df_daily=df,
            df_weekly=resample_to_weekly(df),
            df_monthly=resample_to_monthly(df),
            config=self.config
        )

        # Run backtest with advanced strategy
        backtester = AdvancedBacktester(
            initial_capital=strategy_config['risk_management']['portfolio']['initial_capital'],
            config=strategy_config['strategies']['multiframe_alignment']
        )

        results = backtester.run_backtest(
            df=df,
            pattern_analysis=pattern_analysis,
            strategy_name='multiframe'
        )

        # Display results in GUI
        wx.PostEvent(self, UpdateOutputEvent(
            message=f"\n{'='*60}\nBACKTEST RESULTS - {symbol}\n{'='*60}\n"))

        for key, value in results.items():
            if isinstance(value, float):
                wx.PostEvent(self, UpdateOutputEvent(
                    message=f"{key}: {value:.2f}\n"))
            else:
                wx.PostEvent(self, UpdateOutputEvent(
                    message=f"{key}: {value}\n"))

    except Exception as e:
        error_msg = f"Backtest failed: {e}\n{traceback.format_exc()}\n"
        wx.PostEvent(self, UpdateOutputEvent(message=error_msg))
```

---

## Strategy Selection Guide

### Use **Multi-Timeframe Alignment** when:
- You want the most conservative, high-probability trades
- You have daily, weekly, and monthly data available
- You prefer swing trading (2-8 week holds)
- **Expected**: 60-65% win rate, 5-8% avg profit

### Use **Fibonacci Mean Reversion** when:
- Market is in a correction (Wave 2 or Wave 4)
- Price has pulled back to key Fibonacci levels
- You want shorter holding periods (1-4 weeks)
- **Expected**: 65-75% win rate, 3-6% avg profit

### Use **Pattern Breakout** when:
- Wave 3 is starting (breakout above Wave 1 high)
- Volume confirms the breakout (1.5x+ average)
- You want larger profit potential (8-15%)
- You can accept lower win rate (50-60%)
- **Expected**: 50-60% win rate, 8-15% avg profit

---

## Risk Management Checklist

Before opening ANY trade, check:

- [ ] **Position Size**: Calculated using risk-based formula (2% risk max)
- [ ] **Stop Loss**: Set at wave structure level or 2% below entry
- [ ] **Portfolio Exposure**: Total exposure < 40% of portfolio
- [ ] **Concurrent Positions**: < 5 open positions
- [ ] **Daily Drawdown**: Haven't hit 5% daily loss limit
- [ ] **Alignment Score**: ≥ 70% for primary strategy
- [ ] **Fibonacci Validation**: Price within tolerance of Fib level
- [ ] **Volume Confirmation**: Volume meets minimum threshold

---

## Configuration Tuning

### Make Trading More Conservative:
```json
{
  "alignment_threshold": 0.80,          // Was 0.70 - require higher confidence
  "fibonacci_tolerance": 0.10,           // Was 0.15 - stricter pattern matching
  "max_concurrent_positions": 3,         // Was 5 - fewer simultaneous trades
  "stop_loss_pct": 0.015,               // Was 0.02 - tighter stops
  "max_risk_per_trade": 0.01            // Was 0.02 - risk only 1% per trade
}
```

### Make Trading More Aggressive:
```json
{
  "alignment_threshold": 0.60,          // Was 0.70 - accept more signals
  "fibonacci_tolerance": 0.20,           // Was 0.15 - looser pattern matching
  "max_concurrent_positions": 8,         // Was 5 - more simultaneous trades
  "stop_loss_pct": 0.03,                // Was 0.02 - wider stops
  "max_portfolio_exposure": 0.60        // Was 0.40 - more capital deployed
}
```

---

## Performance Monitoring

### Daily Checklist:
1. Review all open positions
2. Check if any stop losses hit
3. Update trailing stops for profitable positions
4. Check daily drawdown vs. limit
5. Look for new high-probability signals

### Weekly Review:
1. Calculate win rate for the week
2. Review closed trades - what worked, what didn't?
3. Check if any parameter adjustments needed
4. Review alignment with market conditions
5. Update trade journal

### Monthly Analysis:
1. Run full backtest on recent data
2. Compare live results vs. backtest expectations
3. Calculate Sharpe ratio, Sortino ratio, max drawdown
4. Adjust strategy parameters if needed
5. Review portfolio allocation
6. Re-optimize if performance degraded

---

## Common Issues & Solutions

### Issue: Too Few Signals Generated
**Solution**:
- Lower `alignment_threshold` from 0.70 to 0.65
- Increase `fibonacci_tolerance` from 0.15 to 0.20
- Enable more strategy types (enable triangle_breakout, etc.)
- Review data quality - ensure sufficient history

### Issue: Too Many Losing Trades
**Solution**:
- Increase `alignment_threshold` to 0.75+
- Decrease `fibonacci_tolerance` to 0.10 (stricter patterns)
- Increase `volume_multiplier` to 1.5+ (stronger confirmation)
- Review if stop losses are too tight
- Check if holding periods are too short

### Issue: Large Drawdowns
**Solution**:
- Decrease `max_risk_per_trade` to 0.01 (1%)
- Reduce `max_concurrent_positions` to 3
- Implement correlation checks (reduce positions if correlated)
- Widen stop losses using ATR-based method
- Take partial profits earlier

### Issue: Profits Not Meeting Targets
**Solution**:
- Review if exiting too early (adjust profit_targets)
- Use trailing stops instead of fixed targets
- Ensure Wave 3/5 completion detection is accurate
- Check if volume divergence exits are premature
- Consider scaling out (50% at Target 1, 50% at Target 2)

---

## Next Steps

### Immediate (This Week):
1. ✅ Run test backtest on AAPL or SPY
2. ✅ Review strategy documentation thoroughly
3. ✅ Test risk management calculations
4. ✅ Configure strategy_config.json for your preferences

### Short Term (This Month):
1. Integrate strategies with existing Elliott Wave GUI
2. Run backtests on 10+ Taiwan stocks
3. Compare performance across strategies
4. Paper trade Strategy 1 (Multi-Timeframe) in real-time
5. Build trade journal template

### Medium Term (Next 3 Months):
1. Optimize parameters using walk-forward testing
2. Add additional indicators (RSI, MACD) for confirmation
3. Implement automated alerts for high-probability signals
4. Build dashboard for portfolio monitoring
5. Start live trading with minimum position sizes

### Long Term (6+ Months):
1. Integrate broker API for automated trading
2. Add machine learning pattern classification
3. Implement regime detection (bull/bear/sideways)
4. Build multi-asset portfolio optimization
5. Scale up to full capital allocation

---

## Resources & Support

### Documentation:
- [INVESTMENT_STRATEGY.md](./INVESTMENT_STRATEGY.md) - Full strategy guide
- [config/strategy_config.json](./config/strategy_config.json) - All parameters
- [src/backtest/strategy_advanced.py](./src/backtest/strategy_advanced.py) - Implementation code
- [src/utils/risk_management.py](./src/utils/risk_management.py) - Risk utilities

### Key Files to Review:
- Elliott Wave Analysis: [src/analysis/elliott_wave.py](./src/analysis/elliott_wave.py)
- Pattern Detection: [src/analysis/core/impulse.py](./src/analysis/core/impulse.py)
- Validation: [src/analysis/core/validation.py](./src/analysis/core/validation.py)
- GUI Handlers: [gui/handlers.py](./gui/handlers.py)

### Testing Commands:
```bash
# Test risk management utilities
python src/utils/risk_management.py

# Run your existing backtester (for comparison)
python scripts/run_backtest.py

# Run GUI with crawler
python gui/main.py
```

---

## Strategy Performance Targets

### Year 1 (Learning & Optimization):
- **Target Return**: 15-25% annually
- **Win Rate**: 55-60%
- **Max Drawdown**: < 15%
- **Focus**: Learn patterns, optimize parameters, build discipline

### Year 2 (Scaling):
- **Target Return**: 25-35% annually
- **Win Rate**: 60-65%
- **Max Drawdown**: < 12%
- **Focus**: Increase position sizes, refine entries/exits

### Year 3+ (Mature Strategy):
- **Target Return**: 30-50% annually
- **Win Rate**: 65-70%
- **Max Drawdown**: < 10%
- **Focus**: Automated execution, portfolio management, multiple strategies

---

## Important Reminders

⚠️ **Risk Warnings:**
- Start with paper trading (simulated trades)
- Begin with minimum position sizes (1-2% risk)
- Never risk more than you can afford to lose
- Markets are unpredictable - even best strategies have losing streaks
- Always use stop losses - no exceptions
- Review and adjust strategies quarterly

✅ **Success Factors:**
- Discipline in following the strategy rules
- Consistent position sizing (don't over-leverage)
- Keeping detailed trade journal
- Regular performance review and optimization
- Emotional control during drawdowns
- Continuous learning and improvement

---

## Questions?

If you need help:
1. Review the full strategy documentation in INVESTMENT_STRATEGY.md
2. Check configuration examples in strategy_config.json
3. Read the code comments in strategy_advanced.py
4. Test with small position sizes first
5. Keep a detailed journal of what works and what doesn't

**Remember**: The best strategy is the one you can execute consistently with discipline!

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Compatible With**: Elliott Wave Investment System v2.0+
