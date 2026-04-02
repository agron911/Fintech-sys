# Investment Strategy for Elliott Wave Analysis System

## Executive Summary

This strategy is specifically designed to leverage your system's **multi-timeframe Elliott Wave detection**, **Fibonacci validation**, and **volume pattern analysis** capabilities. It focuses on **swing trading** with 1-8 week holding periods targeting **3-8% average returns per trade** with a **60-65% win rate**.

---

## Strategy 1: Multi-Timeframe Wave Alignment (PRIMARY STRATEGY) ⭐⭐⭐

### Overview
Trade only when daily, weekly, and monthly Elliott Wave patterns align in the same direction with 70%+ confidence score.

### Entry Rules
1. **Pattern Alignment Check** (REQUIRED)
   - Daily timeframe: Wave 1 or Wave 3 starting
   - Weekly timeframe: In Wave 3 or Wave 5 (uptrend)
   - Monthly timeframe: Not in Wave 4 or Wave 2 (no major correction)
   - Alignment score ≥ 70% (from `pattern_relationships.py`)

2. **Fibonacci Confirmation** (REQUIRED)
   - Entry price within 2% of Fibonacci support level
   - Valid levels: 38.2%, 50%, or 61.8% retracement of previous wave
   - Use fibonacci_tolerance = 0.15 (normal mode)

3. **Volume Validation** (REQUIRED)
   - If entering Wave 3: Current volume > Wave 1 volume by 20%+
   - If entering Wave 1: Volume increasing from previous correction
   - No volume divergence signals present

4. **Entry Timing**
   - **Option A** (Conservative): Buy when Wave 2 completes at 61.8% retracement
   - **Option B** (Aggressive): Buy when Wave 1 breaks above previous Wave 5 high
   - **Option C** (Wave 3 Momentum): Buy on confirmation of Wave 3 start with volume surge

### Exit Rules
1. **Profit Targets** (Take profit at FIRST hit)
   - **Target 1**: Wave 3 completion (161.8% extension of Wave 1) → Exit 50% position
   - **Target 2**: Wave 5 completion (100% extension of Wave 1) → Exit remaining 50%
   - **Target 3**: Fibonacci time target reached (expected completion date)

2. **Stop Loss** (Hard stops, NO exceptions)
   - **Initial Stop**: 2% below Wave 2 low (if entering at Wave 3 start)
   - **Trailing Stop**: Move to break-even when profit reaches 3%
   - **Wave Invalidation Stop**: Exit if Wave 4 overlaps Wave 1 price territory
   - **Volume Divergence Stop**: Exit if Wave 5 volume < 70% of Wave 3 volume

3. **Time-Based Exit**
   - Exit if position held > 12 weeks without reaching profit target
   - Re-evaluate if wave structure changes (correction appearing)

### Position Sizing
- **Base Position**: 5-10% of portfolio per trade
- **High Confidence** (alignment > 80%): Up to 15% of portfolio
- **Medium Confidence** (alignment 70-80%): 5-8% of portfolio
- **Never exceed 40% total portfolio in active positions**

### Risk Management
- **Maximum Loss Per Trade**: 2% of portfolio
  - Calculate position size: Portfolio × 0.02 / (Entry Price - Stop Loss)
- **Maximum Daily Drawdown**: 5% of portfolio
  - Stop trading for the day if this limit hit
- **Maximum Concurrent Positions**: 5 trades
  - Diversify across different sectors/markets

### Expected Performance
- **Win Rate**: 60-65%
- **Average Win**: +5-8%
- **Average Loss**: -2%
- **Risk/Reward Ratio**: 2.5:1 to 4:1
- **Holding Period**: 2-8 weeks
- **Annual Target**: 25-40% return (assumes ~30 trades/year)

### Implementation Code
```python
# In src/backtest/strategy.py - NEW FUNCTION

def strategy_multiframe_alignment(df_daily, df_weekly, df_monthly, config):
    """
    Multi-timeframe alignment strategy.

    Parameters:
    - df_daily: Daily OHLCV dataframe
    - df_weekly: Weekly OHLCV dataframe
    - df_monthly: Monthly OHLCV dataframe
    - config: Strategy configuration dict

    Returns:
    - trades: List of trade dictionaries
    """
    from src.analysis.elliott_wave import analyze_patterns_multiframe
    from src.analysis.core.fib_utils import calculate_fibonacci_levels
    from src.analysis.core.volume import validate_volume_patterns

    # Analyze patterns across timeframes
    patterns = analyze_patterns_multiframe(
        df_daily, df_weekly, df_monthly,
        fibonacci_tolerance=0.15
    )

    trades = []
    position = None  # Track current position

    for i in range(len(df_daily)):
        date = df_daily.index[i]
        price = df_daily['Close'].iloc[i]

        # Check alignment score
        alignment = patterns.get_alignment_score(date)

        if position is None and alignment >= 0.70:
            # ENTRY LOGIC

            # Get current wave position
            wave_pos = patterns.daily.get_current_position(date)

            # Entry Condition 1: Starting Wave 1 or Wave 3
            if wave_pos['wave_number'] in [1, 3]:
                # Fibonacci check
                fib_levels = calculate_fibonacci_levels(
                    wave_pos['previous_wave']
                )

                # Check if price near support
                near_support = any(
                    abs(price - level) / price < 0.02
                    for level in fib_levels['retracements']
                )

                if near_support:
                    # Volume check
                    vol_valid = validate_volume_patterns(
                        df_daily.iloc[max(0, i-10):i+1]
                    )

                    if vol_valid:
                        # Calculate stop loss
                        stop_loss = wave_pos['wave_2_low'] * 0.98

                        # Calculate position size (2% risk)
                        risk_per_share = price - stop_loss
                        position_size = (config['portfolio_value'] * 0.02) / risk_per_share

                        # Enter position
                        position = {
                            'entry_date': date,
                            'entry_price': price,
                            'stop_loss': stop_loss,
                            'size': position_size,
                            'wave_entered': wave_pos['wave_number'],
                            'alignment': alignment
                        }

        elif position is not None:
            # EXIT LOGIC

            # Stop loss hit
            if price <= position['stop_loss']:
                trades.append({
                    'symbol': config['symbol'],
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': date,
                    'exit_price': price,
                    'profit_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': 'stop_loss',
                    'alignment': position['alignment']
                })
                position = None
                continue

            # Profit target hit (Wave 3 or Wave 5 completion)
            wave_pos = patterns.daily.get_current_position(date)

            if wave_pos['wave_number'] in [3, 5] and wave_pos['completion'] > 0.8:
                # Near wave completion - take profit
                trades.append({
                    'symbol': config['symbol'],
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': date,
                    'exit_price': price,
                    'profit_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': f'wave_{wave_pos["wave_number"]}_complete',
                    'alignment': position['alignment']
                })
                position = None
                continue

            # Volume divergence exit
            if wave_pos['wave_number'] == 5:
                current_vol = df_daily['Volume'].iloc[i]
                wave3_vol = wave_pos.get('wave_3_volume', current_vol)

                if current_vol < wave3_vol * 0.7:
                    trades.append({
                        'symbol': config['symbol'],
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': date,
                        'exit_price': price,
                        'profit_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                        'exit_reason': 'volume_divergence',
                        'alignment': position['alignment']
                    })
                    position = None
                    continue

            # Time-based exit (12 weeks = ~60 trading days)
            days_held = (date - position['entry_date']).days
            if days_held > 84:  # 12 weeks
                trades.append({
                    'symbol': config['symbol'],
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': date,
                    'exit_price': price,
                    'profit_pct': (price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': 'time_exit',
                    'alignment': position['alignment']
                })
                position = None

    return trades
```

---

## Strategy 2: Fibonacci Retracement Mean Reversion (SUPPLEMENTARY) ⭐⭐

### Overview
Trade Wave 2 and Wave 4 corrections when price reaches validated Fibonacci retracement levels.

### Entry Rules
1. **Wave Identification** (REQUIRED)
   - Wave 1 must be completed and validated
   - Currently in Wave 2 or Wave 4 correction
   - Impulse wave length ≥ 5% price movement

2. **Fibonacci Entry Levels** (REQUIRED)
   - **Aggressive**: Enter at 38.2% retracement
   - **Conservative**: Enter at 61.8% retracement
   - **Ultra-Conservative**: Enter at 78.6% retracement (rare, high probability)
   - Price must touch within fibonacci_tolerance (0.15 = 15%)

3. **Volume Confirmation** (REQUIRED)
   - Volume declining during correction (selling pressure exhausting)
   - Volume at entry < 70% of impulse wave average volume

4. **Pattern Confirmation** (OPTIONAL but recommended)
   - Candlestick reversal patterns at Fibonacci level (hammer, bullish engulfing)
   - RSI oversold (< 30) if available
   - MACD bullish divergence if available

### Exit Rules
1. **Profit Target**
   - Wave 3 start: Exit when price breaks above Wave 1 high
   - Expected profit: 3-5% for 38.2% entries, 5-8% for 61.8% entries

2. **Stop Loss**
   - Set stop 1% below Fibonacci level entered
   - For 61.8% entry: Stop at 70% or 78.6% retracement
   - Never risk more than 1.5% per trade

3. **Time Exit**
   - Exit if correction lasts > 4 weeks without reversal
   - Re-evaluate wave structure

### Position Sizing
- **Standard**: 8% of portfolio
- **High Confidence** (61.8% or 78.6% level): 12% of portfolio
- Maximum 3 concurrent mean reversion trades

### Expected Performance
- **Win Rate**: 65-75% (high probability at key Fibonacci levels)
- **Average Win**: +3-6%
- **Average Loss**: -1.5%
- **Risk/Reward Ratio**: 2:1 to 4:1
- **Holding Period**: 1-4 weeks

---

## Strategy 3: Pattern Breakout Trading (AGGRESSIVE) ⭐⭐⭐

### Overview
Trade breakouts when Wave 3 starts after Wave 2 completes, using high alignment scores and volume confirmation.

### Entry Rules
1. **Pattern Setup** (REQUIRED)
   - Wave 1 and Wave 2 completed and validated
   - Wave 3 starting (breakout above Wave 1 high)
   - Alignment score ≥ 75% across timeframes

2. **Breakout Confirmation** (REQUIRED)
   - Price closes above Wave 1 high by ≥ 1%
   - Volume on breakout day ≥ 150% of 20-day average
   - No major resistance within 3% above entry

3. **Timing**
   - Enter on breakout close OR next day open
   - Avoid entering if extended > 2% above breakout level

### Exit Rules
1. **Profit Targets** (Scale out)
   - **Target 1** (50% position): 161.8% extension of Wave 1 (typical Wave 3 target)
   - **Target 2** (30% position): 200% extension of Wave 1 (extended Wave 3)
   - **Target 3** (20% position): Trail with 15% trailing stop

2. **Stop Loss**
   - Initial: 2% below breakout level (Wave 1 high)
   - Trailing: Move to break-even when +4% profit
   - Trail stops at 10% below highest price reached

3. **Early Exit Signals**
   - Volume declining > 3 consecutive days in Wave 3
   - Price breaks below 20-day moving average
   - Alignment score drops below 60%

### Position Sizing
- **Standard**: 10-12% of portfolio
- **Very High Confidence** (alignment > 85%, volume surge): 15% of portfolio
- Maximum 2 concurrent breakout trades

### Expected Performance
- **Win Rate**: 50-60% (lower frequency, higher quality)
- **Average Win**: +8-15%
- **Average Loss**: -2%
- **Risk/Reward Ratio**: 4:1 to 7:1
- **Holding Period**: 2-6 weeks

---

## Strategy 4: Triangle Compression Breakout (TACTICAL) ⭐⭐

### Overview
Trade triangle patterns (Wave 4 corrections) when breakout occurs with clear direction.

### Entry Rules
1. **Triangle Identification** (REQUIRED)
   - System detects triangle pattern (5 subtypes supported)
   - Minimum 3 waves completed (A-B-C)
   - Price range compressing (each wave smaller than previous)

2. **Breakout Signal** (REQUIRED)
   - Price breaks triangle boundary by ≥ 1.5%
   - Breakout direction = trend direction (upward triangle in uptrend)
   - Volume on breakout ≥ 130% of triangle average volume

3. **Entry Timing**
   - Enter on breakout close
   - Confirm with next candle (no immediate reversal)

### Exit Rules
1. **Profit Target**
   - Height of triangle base projected from breakout point
   - Example: Triangle from $100-$110 → Target = $110 + $10 = $120

2. **Stop Loss**
   - 1% below triangle breakout level
   - Or opposite triangle boundary (whichever is closer)

3. **Time Exit**
   - Exit if target not reached within 3 weeks
   - Triangles typically resolve quickly

### Position Sizing
- **Standard**: 6-8% of portfolio
- Maximum 2 concurrent triangle trades

### Expected Performance
- **Win Rate**: 60-70% (triangles are reliable)
- **Average Win**: +4-8%
- **Average Loss**: -1.5%
- **Risk/Reward Ratio**: 3:1 to 5:1
- **Holding Period**: 1-3 weeks

---

## Portfolio Construction & Risk Management

### Overall Portfolio Rules

1. **Maximum Exposure**
   - Total portfolio in active trades: 40% maximum
   - Keep 60% in cash for new opportunities and risk management

2. **Diversification**
   - No more than 2 trades in same sector
   - Mix Taiwan stocks (60%), US stocks (30%), ETFs (10%)
   - Avoid correlated positions (all tech stocks moving together)

3. **Risk Limits**
   - **Per Trade**: 2% maximum loss
   - **Daily**: 5% maximum portfolio drawdown
   - **Weekly**: 8% maximum portfolio drawdown
   - **Monthly**: 12% maximum portfolio drawdown

4. **Position Sizing Formula**
   ```
   Position Size = (Portfolio Value × Risk%) / (Entry Price - Stop Loss)

   Example:
   - Portfolio: $100,000
   - Risk per trade: 2% = $2,000
   - Entry: $50
   - Stop Loss: $48
   - Position Size = $2,000 / ($50 - $48) = $2,000 / $2 = 1,000 shares
   - Total Investment = 1,000 × $50 = $50,000 (50% of portfolio)
   ```

5. **Correlation Management**
   - Check correlation between active positions
   - If correlation > 0.7, reduce position sizes by 30%
   - Avoid adding correlated positions

### Money Management Rules

1. **After Winning Streak** (3+ consecutive wins)
   - Increase position sizes by 20% (but respect maximum limits)
   - Take profits more aggressively
   - Review strategy performance

2. **After Losing Streak** (3+ consecutive losses)
   - Decrease position sizes by 30%
   - Take break (1-2 days) to reassess
   - Review what went wrong (pattern misidentification? Poor risk management?)
   - Return to paper trading if needed

3. **Drawdown Protocol**
   - **-5% from peak**: Reduce all position sizes by 20%
   - **-10% from peak**: Reduce to minimum position sizes, no new trades
   - **-15% from peak**: Stop trading for 1 week, full strategy review
   - **-20% from peak**: Stop trading for 1 month, fundamental strategy reassessment

### Trade Management

1. **Trade Entry Checklist** (Must check ALL before entry)
   - [ ] Pattern validated by system (confidence ≥ 70%)
   - [ ] Fibonacci levels aligned
   - [ ] Volume confirms pattern
   - [ ] Stop loss level calculated
   - [ ] Position size calculated (respects 2% risk rule)
   - [ ] Total portfolio exposure < 40%
   - [ ] No correlation with existing positions
   - [ ] Entry price recorded for tracking

2. **Trade Exit Checklist**
   - [ ] Stop loss order placed immediately after entry
   - [ ] Profit targets calculated and recorded
   - [ ] Trailing stop strategy defined
   - [ ] Review position daily for exit signals
   - [ ] Exit without hesitation when stop/target hit

3. **Trade Journaling** (CRITICAL for improvement)
   - Record every trade in journal
   - Include:
     - Entry/exit dates and prices
     - Pattern type and alignment score
     - Profit/loss and R-multiple (profit ÷ initial risk)
     - What went right/wrong
     - Emotional state during trade
   - Review journal weekly
   - Calculate statistics monthly

---

## Backtesting Protocol

### Historical Testing Requirements

1. **Minimum Data Requirements**
   - Test on ≥ 3 years of historical data
   - Include different market conditions:
     - Bull market (2020-2021)
     - Bear market (2022)
     - Sideways market (2023)
     - High volatility periods

2. **Walk-Forward Testing**
   - In-sample period: 2 years (optimize parameters)
   - Out-of-sample period: 1 year (test performance)
   - Roll forward every 6 months

3. **Key Metrics to Track**
   ```
   - Total Return
   - Win Rate
   - Average Win / Average Loss
   - Profit Factor (Gross Profit ÷ Gross Loss)
   - Maximum Drawdown
   - Sharpe Ratio (if risk-free rate known)
   - Sortino Ratio (downside deviation)
   - Recovery Time (time to recover from drawdown)
   - Consecutive Losses (risk of ruin)
   ```

4. **Minimum Performance Standards**
   - Win Rate: ≥ 55%
   - Profit Factor: ≥ 1.5
   - Maximum Drawdown: ≤ 20%
   - Recovery Time: ≤ 3 months

### Parameter Optimization

Test these parameters systematically:

1. **Fibonacci Tolerance**: [0.05, 0.10, 0.15, 0.20]
2. **Alignment Score Threshold**: [0.60, 0.65, 0.70, 0.75, 0.80]
3. **Volume Confirmation**: [1.2x, 1.3x, 1.5x of average]
4. **Stop Loss Distance**: [1%, 1.5%, 2%, 2.5%]
5. **Profit Target Multiples**: [1.5R, 2R, 3R, 4R]

Use **grid search** or **genetic algorithm** for optimization.

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [x] Analyze existing system capabilities
- [ ] Create strategy configuration file
- [ ] Implement position sizing module
- [ ] Add stop loss/take profit logic to backtester

### Phase 2: Strategy Implementation (Week 3-4)
- [ ] Implement Strategy 1 (Multi-Timeframe Alignment)
- [ ] Implement Strategy 2 (Fibonacci Mean Reversion)
- [ ] Create trade entry/exit modules
- [ ] Add risk management checks

### Phase 3: Backtesting & Optimization (Week 5-6)
- [ ] Run historical backtests for all strategies
- [ ] Optimize parameters using walk-forward
- [ ] Calculate performance metrics
- [ ] Generate backtest reports

### Phase 4: Paper Trading (Week 7-10)
- [ ] Set up paper trading environment
- [ ] Trade strategies in real-time without real money
- [ ] Track all trades in journal
- [ ] Compare paper results to backtest expectations

### Phase 5: Live Trading (Week 11+)
- [ ] Start with minimum position sizes
- [ ] Trade Strategy 1 only (most conservative)
- [ ] Gradually add other strategies as confidence builds
- [ ] Monitor performance weekly
- [ ] Adjust parameters based on live results

---

## Risk Disclosures

### Strategy Limitations

1. **Elliott Wave Subjectivity**
   - Multiple valid wave counts possible
   - System confidence scores help but don't eliminate uncertainty
   - Always use confluence with other indicators

2. **Market Regime Changes**
   - Strategies optimized for trending markets
   - Performance degrades in choppy/sideways markets
   - Monitor market regime and adjust

3. **Black Swan Events**
   - Stop losses may not execute in extreme volatility
   - Gap downs can cause losses > 2% risk per trade
   - Maintain adequate cash reserves

4. **Overfitting Risk**
   - Parameters optimized on historical data may not work forward
   - Use walk-forward testing to mitigate
   - Re-optimize quarterly

### Recommended Safeguards

1. **Start Small**
   - Begin with 10% of intended capital
   - Scale up only after 3 months of profitable trading
   - Never risk capital you can't afford to lose

2. **Continuous Monitoring**
   - Review all open positions daily
   - Check for news/events affecting holdings
   - Update stop losses as needed

3. **Strategy Evolution**
   - Review strategy performance quarterly
   - Adapt to changing market conditions
   - Incorporate lessons learned from trade journal

4. **Professional Advice**
   - Consider consulting a financial advisor
   - Understand tax implications of trading
   - Ensure strategies align with financial goals

---

## Appendix: Quick Reference

### Strategy Comparison Matrix

| Strategy | Win Rate | Avg Win | Avg Loss | Risk/Reward | Holding Period | Complexity |
|----------|----------|---------|----------|-------------|----------------|------------|
| Multi-Timeframe Alignment | 60-65% | 5-8% | -2% | 2.5-4:1 | 2-8 weeks | High |
| Fibonacci Mean Reversion | 65-75% | 3-6% | -1.5% | 2-4:1 | 1-4 weeks | Medium |
| Pattern Breakout | 50-60% | 8-15% | -2% | 4-7:1 | 2-6 weeks | Medium |
| Triangle Breakout | 60-70% | 4-8% | -1.5% | 3-5:1 | 1-3 weeks | Low |

### Configuration File Example

```json
{
  "strategy": {
    "name": "multi_timeframe_alignment",
    "alignment_threshold": 0.70,
    "fibonacci_tolerance": 0.15,
    "volume_multiplier": 1.2,
    "validation_mode": "normal"
  },
  "risk_management": {
    "max_risk_per_trade": 0.02,
    "max_portfolio_exposure": 0.40,
    "max_daily_drawdown": 0.05,
    "max_concurrent_positions": 5,
    "position_sizing_method": "risk_based"
  },
  "position_management": {
    "stop_loss_pct": 0.02,
    "trailing_stop_pct": 0.10,
    "profit_target_r_multiple": 3.0,
    "time_exit_days": 84
  },
  "portfolio": {
    "initial_capital": 100000,
    "min_position_size": 0.05,
    "max_position_size": 0.15,
    "cash_reserve_pct": 0.60
  }
}
```

### Key Formulas

**Position Size Calculation**:
```
Position Size (shares) = (Portfolio Value × Risk %) / (Entry Price - Stop Loss)
```

**Risk-Reward Ratio**:
```
R = (Profit Target - Entry) / (Entry - Stop Loss)
```

**Profit Factor**:
```
Profit Factor = Gross Profit / Gross Loss
```

**Sharpe Ratio** (annualized):
```
Sharpe = (Average Return - Risk Free Rate) / Standard Deviation of Returns
```

**Maximum Drawdown**:
```
Max DD = (Trough Value - Peak Value) / Peak Value × 100%
```

---

## Contact & Support

For questions about this strategy:
1. Review system documentation in `/docs/`
2. Check backtest results in `/data/processed/`
3. Consult Elliott Wave theory resources
4. Join trading communities for peer discussion

**Disclaimer**: This strategy is for educational purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always conduct your own research and consult with licensed financial advisors before making investment decisions.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**System Compatibility**: Elliott Wave Investment System v2.0+
