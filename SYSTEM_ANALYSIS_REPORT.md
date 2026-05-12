# Fintech-sys: Comprehensive System Analysis Report

**Date:** 2026-05-06
**Analyzed by:** 5 Independent Research Agents (Architecture, Calculations, Backtesting, Data Pipeline, GUI)

---

## Executive Summary

Fintech-sys is an **Elliott Wave Theory-based equity trading system** covering US and Taiwan (TWSE/OTC) markets (~2,300 stocks). It ingests OHLCV data from Yahoo Finance, applies a multi-layered analysis pipeline rooted in Elliott Wave pattern detection combined with modern momentum indicators, and produces actionable buy/sell/hold recommendations with full trade plans (entry, stop, targets, position sizing). The system includes walk-forward backtesting, Monte Carlo simulation, and a wxPython desktop GUI with HTML dashboards and interactive charting.

---

## 1. System Architecture

### 1.1 Directory Structure

```
Fintech-sys/
|-- main.py                          CLI dispatcher (argparse -> subprocess)
|-- config/config.json               Central JSON configuration
|-- scripts/                         Entry point scripts (6 scripts)
|-- src/
|   |-- utils/                       Shared utilities (config, data, risk mgmt)
|   |-- crawler/                     Data acquisition (Yahoo Finance)
|   |-- analysis/
|   |   |-- core/                    Elliott Wave engine (17+ modules)
|   |   |-- plotters/                Matplotlib visualization
|   |   |-- elliott_wave.py          High-level EW API
|   |   `-- market_structure.py      Market regime/breadth/sector analysis
|   `-- backtest/                    Backtesting engine (4 modules)
|-- gui/                             wxPython desktop application
|-- tests/                           Test suite (24+ test cases)
|-- data/                            Raw/processed data, stock lists
`-- cache/                           Scan result cache
```

### 1.2 Entry Points

| Command | Script | Purpose |
|---------|--------|---------|
| `python main.py scan` | `scripts/what_to_buy.py` | Scan ~180 stocks for buy candidates |
| `python main.py fetch` | `scripts/run_fetch_data.py` | Fetch data from Yahoo Finance |
| `python main.py backtest` | `scripts/run_backtest.py` | Walk-forward backtest + Monte Carlo |
| `python main.py validate` | `scripts/validate_data.py` | Data quality validation + auto-fix |
| `python main.py gui` | `scripts/run_gui.py` | Launch desktop GUI |

### 1.3 Module Dependency Flow

```
                    main.py (dispatcher)
                         |
            scripts/ (entry points)
           /    |       |       \
         gui/  what_  run_    validate_
               to_buy backtest  data
                |       |
                v       v
    +---------------------------+
    |  src/analysis/core/       |  <-- CORE ENGINE
    |  signal_scoring.py        |  <-- INTEGRATION POINT
    +---------------------------+
          |           |
    src/backtest/   src/analysis/
    (strategy,       (elliott_wave,
     backtester,      market_structure)
     monte_carlo)          |
          |           src/analysis/core/
          +------+---------+
                 |
           src/utils/
           (config, data_utils,
            common_utils, risk_mgmt)
                 |
           src/crawler/
           (yahoo_finance)
```

### 1.4 Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Data Analysis | pandas, numpy, scipy | Core data manipulation, signal processing |
| Financial Data | yfinance | Yahoo Finance API |
| Visualization | matplotlib, mplfinance | Charting, candlestick plots |
| GUI | wxPython | Desktop application framework |
| Resilience | retrying | Network retry logic |
| File Handling | openpyxl | Excel file reading |
| Environment | python-dotenv | Environment variables |

---

## 2. Data Pipeline

### 2.1 Data Ingestion

**Source:** Yahoo Finance via `yfinance` library

**Stock Universes:**
- **US/International:** ~180 tickers from `data/lists/international.txt` (mega-cap tech, semis, AI/quantum, finance, healthcare, energy, consumer, industrial, telecom, EV, international ADRs)
- **Taiwan TWSE Listed:** From `data/lists/adjustments/list.xlsx` (suffix `.TW`)
- **Taiwan OTC:** From `data/lists/adjustments/otclist.xlsx` (suffix `.TWO`)
- **Total scanned:** ~2,296 stocks (based on cache)

**Date Range:** 2002-01-01 through 2026-12-20 (configurable)

**Incremental Updates:** Checks file modification time and last data date; skips re-fetching if file is fresh (<1 day old) AND data is current. Achieves 75-85% time savings over full downloads.

### 2.2 Raw Data Storage

**Format:** Tab-separated values (TSV), one file per stock in `data/raw/{SYMBOL}.txt`
```
Date    Open    High    Low    Close    Volume    Date
2002/01/02    0.330...    0.349...    ...
```
- Auto-adjusted prices (split-adjusted)
- Date column duplicated at end (artifact of save logic)

### 2.3 Data Validation (7 Categories)

| Check | Description | Auto-Fix? |
|-------|-------------|-----------|
| Format errors | Missing header, truncated files, bad rows, High < Low, negative prices | No |
| Merge conflicts | Git conflict markers in data files | Yes |
| Duplicate dates | Same date appearing multiple times | No |
| Cross-file duplicates | MD5 hash comparison across symbols | No |
| Gaps | Calendar gaps >5 days between trading dates | No |
| Price jumps | >50% close-to-close moves (splits, errors) | No |
| Zero volume | Trading days with zero volume | No |
| Staleness | Last data point >30 days old | No |

---

## 3. Analysis Engine: Complete Calculation Breakdown

### 3.1 Pipeline Overview

```
Raw OHLCV Data
    |
    v
(1) Peak/Trough Detection (3 methods fused)
    |
    v
(2) Elliott Wave Pattern Matching (impulse candidates + validation)
    |
    v
(3) Pattern Adaptation (wave data -> strategy format)
    |
    v
(4) Momentum Indicators (RSI + MACD + ADX + ATR + Velocity)
    |
    v
(5) Signal Scoring (30+ factors, 0-100+ scale)
    |
    v
(6) Action Classification (STRONG BUY -> AVOID)
    |
    v
(7) Conviction Grading (Trend/Timing/Risk: A-F)
    |
    v
(8) Relative Strength Ranking (cross-universe)
    |
    v
(9) Market Regime Overlay (FAVORABLE/MIXED/CAUTION)
```

### 3.2 Peak/Trough Detection

**File:** `src/analysis/core/peaks.py`

Three independent detection methods are combined (multi-scale fusion):

| Method | Algorithm | Key Parameters |
|--------|-----------|---------------|
| ZigZag | Trend-reversal detection | 5% threshold |
| scipy find_peaks | Adaptive prominence + distance | Prominence = price * (base + vol * 2), clamped [0.5%, 5%] |
| argrelextrema | Relative extrema | order=5 (5-bar neighborhood) |

**Post-Processing:**
1. Minimum move filter (day: 1.5%, week: 2.5%, month: 4%)
2. Clustering nearby extrema by time/price proximity
3. Minimum 2-period separation
4. Boundary detection (first/last data points)

### 3.3 Elliott Wave Pattern Detection

**File:** `src/analysis/core/impulse.py`, `intelligent_subwaves.py`, `flexible_sequence_builder.py`

**Candidate Generation:**
- Iterates over up to 8 most recent start points (peaks/troughs)
- Tests 4 range multipliers (0.4, 0.6, 0.8, 1.0) per start point
- Builds wave sequences via `FlexibleSequenceBuilder`
- Accepts candidates with 5+ points and confidence > 0.10

**Sequence Scoring (4-component weighted):**

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Wave Progression | 0.40 | Correct alternating direction |
| Wave Proportions | 0.30 | Wave 3 not shortest, Wave 2 retracement 30-80% |
| Trend Consistency | 0.20 | Efficiency ratio = |net_move| / total_movement |
| Time Relationships | 0.10 | Duration consistency across waves |

- **Bonus:** +0.10 if Wave 3 > 1.5x Wave 1 (extended Wave 3)
- **Reality adjustment:** All scores * 0.85 to prevent overconfidence

**Quality Validation (up to +0.70):**
- Wave 3/Wave 1 ratio in [0.618, 2.618]: +0.30
- Wave 2 depth in [0.236, 0.786]: +0.20
- Wave efficiency > 0.30: +0.20

### 3.4 Elliott Wave Validation Rules

**File:** `src/analysis/core/validation.py`

5 rule categories with weighted confidence:

| Rule | Weight | Description |
|------|--------|-------------|
| Wave Directions | 0.30 | Impulse waves (1,3,5) move with trend; corrective (2,4) against |
| Wave 2 Retracement | 0.20 | Hard fail if >100% of Wave 1; ideal 38.2-61.8% |
| Wave 3 Length | 0.25 | Hard fail if shortest impulse wave; bonus if longest |
| Wave 4 Overlap | 0.15 | No entry into Wave 1 territory (strict mode: instant reject) |
| Fibonacci Relationships | 0.25 | Multi-level Fibonacci ratio checks (see below) |

**Fibonacci Validation (up to 0.80 confidence):**

| Check | Contribution |
|-------|-------------|
| Wave 2 retracement matches Fibonacci level (5% tolerance) | +0.20 |
| Wave 3 extension vs Wave 1 | +0.25 |
| Wave 4 retracement of Wave 3 | +0.15 |
| Wave 5 vs Wave 1 projection | +0.20 |
| Wave 5 vs Wave 1+3 combined | +0.15 |
| Wave 5 vs Wave 3 ratio | +0.20 |
| Rate-of-Change projection | +0.40 |
| Geometric-Mean projection | +0.40 |
| Time relationships (Wave 2 vs Wave 4 durations) | +0.10 |

**Standard Fibonacci Ratios Used:**
- Retracements: [0.236, 0.382, 0.5, 0.618, 0.786]
- Extensions: [1.618, 2.618, 4.236]
- Projections: [0.618, 1.0, 1.618]
- Time Ratios: [0.618, 1.0, 1.618, 2.618]

**Configuration Profiles:**

| Parameter | Default | Strict | Relaxed |
|-----------|---------|--------|---------|
| Fibonacci tolerance | 0.15 | 0.05 | 0.20 |
| Wave 2 max retracement | 0.90 | 0.786 | 0.95 |
| Wave 3 min ratio | 1.0 | 1.1 | 0.9 |
| Overlap allowed | No | No | Yes |
| Acceptance threshold | 0.30 | 0.40 | 0.25 |
| Reality adjustment | 0.85 | 0.90 | 0.80 |

### 3.5 Corrective Pattern Analysis

**File:** `src/analysis/core/corrective_patterns.py`

| Pattern | Structure | Key Validation |
|---------|-----------|---------------|
| Zigzag | 5-3-5 | Wave B retracement 38.2-78.6%, Wave C/A near 1.0/1.272/1.618 |
| Flat (Regular) | 3-3-5 | Wave B 90-100% of A, Wave C near A endpoint |
| Flat (Expanded) | 3-3-5 | Wave B 100-138% of A |
| Flat (Running) | 3-3-5 | Wave B > 138% of A, C fails to reach A endpoint |
| Triangle | 3-3-3-3-3 | Converging wave sizes, overlapping sub-waves |
| Double Zigzag | W-X-Y | Two zigzags connected by X wave (30-90% of W) |
| Complex | W-X-Y-X-Z | Multiple corrective patterns |

### 3.6 Subwave Analysis (Fractal Validation)

**File:** `src/analysis/core/intelligent_subwaves.py`

| Wave | CRITICAL | IMPORTANT | OPTIONAL |
|------|----------|-----------|---------|
| Wave 3 | Duration >= 15 days | Duration >= 10 days | - |
| Wave 1 | - | Duration >= 30 days OR range >= 15% | Duration >= 15 days |
| Wave 5 | - | Extension >= 1.5x | Duration >= 20 days |
| Wave 2,4 | - | Duration >= 25 days | Duration >= 15 days |

Confidence adjustments: +0.03 per found subwave, -0.05 per missing critical, -0.025 per missing important. Capped at +/-0.20.

### 3.7 Momentum Indicators

**File:** `src/analysis/core/momentum_indicators.py`

#### RSI (Relative Strength Index)
- **Formula:** `RSI = 100 - 100/(1 + RS)` where `RS = EMA(gains, 14) / EMA(losses, 14)`
- **Zones:** Oversold < 30, Overbought > 70, Neutral otherwise
- **Strength:** `(RSI - 50) / 20`, clamped [-1.0, +1.0]
- **Divergence:** Compares last 20 bars vs prior 20 bars (bullish: lower price low + higher RSI low)

#### MACD (Moving Average Convergence Divergence)
- **Formula:** MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9), Histogram = MACD - Signal
- **Crossover detection:** Scans last 3 bars for MACD crossing above/below signal
- **Momentum acceleration:** `accel = (h1-h2) - (h2-h3)`, threshold: `0.001 * price`
- **Strength:** `histogram / (price * 0.02)`, clamped [-1.0, +1.0]

#### ADX (Average Directional Index)
- **Formula:** Standard Wilder's ADX using +DI, -DI, smoothed DX (EWM span=14)
- **Regime thresholds:** Strong trend > 25, Weak trend 15-25, No trend < 15
- **Trending gate:** ADX > 20
- **Strength:** `min(ADX / 40, 1.0)`

#### ATR Regime (Volatility Classification)
- **Formula:** Standard True Range with EWM smoothing

| ATR % of Price | Regime | Stop Multiplier |
|----------------|--------|----------------|
| < 1.0% | Calm | 0.7x (tighter) |
| 1.0-3.0% | Normal | 1.0x |
| 3.0-5.0% | Volatile | 1.5x (wider) |
| > 5.0% | Explosive | 2.0x (much wider) |

#### Velocity (Rate of Change)
- **Formula:** `ROC(n) = (price / price_n_ago - 1) * 100`
- **Multi-timeframe:** 5-day and 20-day ROC
- **Acceleration:** `ROC_5[now] - ROC_5[5_bars_ago]`

| ROC 20d | Speed Regime |
|---------|-------------|
| < -15% | crash |
| -15% to -5% | fast_down |
| -5% to -1% | slow_down |
| -1% to +1% | flat |
| +1% to +5% | slow_up |
| +5% to +15% | fast_up |
| > +15% | melt_up |

#### Composite Momentum Score

| Indicator | Weight | Score Calculation |
|-----------|--------|-----------------|
| RSI | 0.25 | strength + divergence bonus (+/-0.3) |
| MACD | 0.30 | strength + crossover bonus (+/-0.4), scaled by acceleration |
| ADX | 0.15 | strength * 0.5 (always positive) |
| Velocity | 0.30 | vel_20d/10 if timeframes agree, vel_5d/20 if conflicting |

**Decision Gates:**
- **`entry_ok`**: composite > 0.1 AND not overbought AND no bearish MACD cross AND not crashing AND ADX trending
- **`exit_warning`**: 2+ bearish conditions met (negative composite, bearish divergence, overbought RSI, bearish cross, crash speed)

**Confidence Adjustment Factors:**

| Condition | Factor |
|-----------|--------|
| composite > 0.3 AND trending | 1.2x |
| composite > 0.1 | 1.0x |
| composite > -0.1 | 0.8x |
| crash/fast_down speed | 0.3x |
| Other bearish | 0.6x |

### 3.8 Volume Analysis

**File:** `src/analysis/core/volume.py`

| Rule | Contribution |
|------|-------------|
| Wave 3 has highest volume | +0.40 |
| Impulse avg volume > corrective avg volume | +0.30 |
| Wave 5 volume < Wave 1 volume (declining) | +0.30 |

### 3.9 Wave Personality Validation

**File:** `src/analysis/core/wave_personality.py`

- **Wave 3:** Highest volume among impulse waves (+0.20), momentum > 1.2x Wave 1 (+0.15)
- **Wave 5:** Declining volume vs Wave 3 (+0.15), momentum divergence (+0.10), reversal warning
- Overall: Wave 3 confidence * 0.60 + Wave 5 confidence * 0.40, capped at 0.50

### 3.10 Market Regime Detection

**File:** `src/analysis/core/regime.py`

| ADX | ATR Percentile | Regime | Strategy Hint |
|-----|---------------|--------|--------------|
| > 25 | < 80th | trending | breakout |
| < 15 | any | ranging | mean_reversion |
| any | > 80th | volatile | reduce_size |
| other | other | normal | default |

**Regime-Based Confidence Adjustments:**
- Trending + aligned: confidence * 1.10
- Trending + misaligned: confidence * 0.95
- Ranging: confidence * 0.90
- Volatile: confidence * 0.92

### 3.11 Trend Alignment

**File:** `src/analysis/core/impulse.py`

Uses SMA50/SMA200 crossover:
- **Bullish:** SMA50 > SMA200 AND price > SMA200
- **Bearish:** SMA50 < SMA200 AND price < SMA200
- **Neutral:** otherwise

Pattern confidence adjustments:
- Trend-aligned: confidence * 1.3
- Counter-trend (bearish pattern in bullish market): confidence * 0.5
- Counter-trend (bullish pattern in bearish market): confidence * 0.6

---

## 4. Signal Scoring & Classification

### 4.1 Score Calculation (0-100+ scale)

**File:** `src/analysis/core/signal_scoring.py`

#### Base Scoring (max ~100 points)

| Factor | Points | Condition |
|--------|--------|-----------|
| Wave position | 5-30 | Wave 3=30, Wave 2=25, Wave 4=18, Wave 1=15, Wave 5=5 |
| Bullish trend | +15 | SMA50 > SMA200 |
| Pattern direction up | +10 | |
| Momentum confirmed | +15 | entry_ok = true |
| No exit warning | +5 | |
| Confidence | 0-10 | Scaled by pattern confidence |
| Risk/Reward | 4-15 | R:R >= 3.0=15, >= 2.0=12, >= 1.5=8, >= 1.0=4 |
| RSI oversold (<35) | +5 | |
| RSI overbought (>75) | -5 | |
| MACD bullish crossover | +5 | |
| RSI bullish divergence | +5 | |
| Correction setup | +20 | Fib retrace + Wave C + bullish + RSI < 40 + MACD bullish |

#### Tier 1: Momentum Quality

| Factor | Points |
|--------|--------|
| Momentum quality (composite * 10) | up to +10 |
| MACD expanding up | +5 |
| MACD contracting/expanding down | -3 |
| Velocity accelerating up | +5 |
| Velocity decelerating | -3 |
| Strong ADX trend | +5 |
| No ADX trend | -10 |
| Calm ATR | +3 |
| Explosive ATR | -5 |
| Price > 130% SMA200 (overextended) | -5 |
| Regime: reduce_size | -5 |
| Weak trend + high volatility | -8 |

#### Tier 2: Volume & Personality

| Factor | Points |
|--------|--------|
| Volume confirmation (vol_score * 10) | up to +10 |
| Volume disconfirm (< 0.2) | -5 |
| Wave 3 highest volume | +5 |
| Wave 5 reversal warning | -10 |
| Personality confidence (0-1 * 5) | up to +5 |
| Volume surge (> 1.5x) | +3 |
| Low volume (< 0.5x) | -3 |
| RS top quartile | +10 |
| RS bottom quartile | -5 |

### 4.2 Action Classification Thresholds

| Action | Score | Additional Conditions |
|--------|-------|----------------------|
| STRONG BUY | >= 75 | Wave 3 |
| BUY | >= 65 | - |
| BUY DIP | >= 55 | Wave 4 + bullish trend |
| BUY CORRECTION | >= 55 | Fib retracement + correction candidate |
| WATCH | >= 45 | - |
| EXIT | any | Wave >= 5 + exit warning + below SMA50 |
| AVOID | any | Bearish trend OR counter-trend pattern |
| HOLD | any | Wave 5-6, wait for new impulse |
| WAIT | < 45 | No clear setup |

### 4.3 Conviction Grading (A/B/C/D/F)

Three independent letter grades:

**Trend Grade (max 8 pts):** bullish +3, pattern up +2, ADX >= 25 +2, above SMA200 +1

**Timing Grade (max 10 pts):** Wave 3 +3, Wave 2 +2, entry_ok +2, RSI 30-60 +1, RSI < 30 +2, MACD bullish +1, accelerating +1

**Risk Grade (max 7 pts):** R:R >= 3.0 +4, >= 2.0 +3, confidence >= 0.5 +2, calm ATR +1

Grade mapping: ratio >= 0.8 = A, >= 0.6 = B, >= 0.4 = C, >= 0.2 = D, else F

### 4.4 Post-Processing Pipeline

Applied sequentially to all scan results:
1. **Relative Strength:** Ranks all stocks by 6-month momentum; assigns RS rank and percentile
2. **RS Guard:** Suppresses EXIT for top-25% RS stocks (converts to HOLD)
3. **RS Score Adjustment:** Top quartile +10 points, bottom quartile -5 points
4. **Reclassify Borderline:** Promotes WATCH to BUY if RS-boosted >= 65; demotes BUY to WATCH if RS-penalized < 45
5. **Market Regime Overlay:**
   - CAUTION (>= 30% EXIT/AVOID): raises BUY threshold to 75
   - MIXED (>= 15%): blocks BUY DIP/CORRECTION unless score >= 65
   - FAVORABLE (< 15%): no restrictions

---

## 5. Market Structure Analysis

**File:** `src/analysis/market_structure.py`

### 5.1 Cycle Position

| Condition | Cycle |
|-----------|-------|
| > 30% stocks in Wave 3 | Mid-cycle expansion |
| > 25% in Wave 1-2 | Early recovery |
| > 35% in Wave 4-5 | Late-cycle |
| > 40% corrective | Correction / reset |
| Other | Transition |

### 5.2 Market Regime (Breadth-Based)

| Breadth (% > SMA200) | Regime |
|-----------------------|--------|
| >= 65% + ADX >= 22 | Trending Bullish |
| >= 50% | Bullish with caution |
| >= 35% | Mixed / Rotating |
| >= 20% | Weakening |
| < 20% | Bearish |

### 5.3 Sector Rotation

7 sectors (Tech, Semis, Finance, Health, Consumer, Energy, Industrial). Each scored by:
- Average 6-month momentum
- Bullish percentage
- Direction: leading (avg_mom > 5% AND bullish > 60%), lagging (avg_mom < -5% OR bullish < 30%), neutral

---

## 6. Backtesting System

### 6.1 Walk-Forward Validation

**File:** `src/backtest/backtester.py`

- **Rolling window:** 504 bars (~2 years), step: 63 bars (~1 quarter)
- For each window:
  1. Detect Elliott Wave patterns
  2. Adapt to strategy format
  3. Generate signals via `MultiTimeframeAlignmentStrategy`
  4. **Keep only out-of-sample signals** (last `step_size` bars) to prevent data snooping
- Signals deduplicated by (date, type), keeping highest confidence

### 6.2 Trading Strategies

#### Strategy 1: Multi-Timeframe Alignment (Primary)
- **Entry:** Wave 1/3 + Fibonacci support + volume >= 1.2x avg + RSI < 75 + MACD confirmation
- **Two confidence gates:**
  - Gate 1: Composite confidence >= 0.35
  - Gate 2: Validation confidence >= 0.30
- **Exit:** Wave 5 exhaustion (2+ signals), RSI crash (>15pt drop in 5 bars)

#### Strategy 2: Fibonacci Mean Reversion
- **Entry:** Wave 2/4 corrections at Fibonacci levels [38.2%, 61.8%, 78.6%] with declining volume

#### Strategy 3: Pattern Breakout
- **Entry:** Wave 3 breakout above Wave 1 high + volume >= 1.5x avg

#### Ensemble Mode
- BUY requires 2+ strategies to agree on same date
- SELL passes through from any strategy
- +0.10 consensus bonus

### 6.3 Position Sizing (Multi-Layer)

| Layer | Parameter | Value |
|-------|-----------|-------|
| Risk per trade | Max risk | 2% of portfolio |
| Position size | Shares | max_risk / risk_per_share |
| Max per position | Cap | 15% of portfolio |
| Max positions | Cap | 5 concurrent |
| Max daily drawdown | Circuit breaker | 5% |
| Portfolio drawdown halt | No new entries | 15% |
| Max position value | Cap | 40% of capital |
| Entry cooldown | Minimum gap | 5 bars |

**Wave-Aware Sizing Multipliers:**

| Wave | Size Multiplier |
|------|----------------|
| Wave 1 | 0.6x |
| Wave 2 | 0.7x |
| Wave 3 | 1.0x (full) |
| Wave 4 | 0.3x |
| Wave 5 | 0.5x |

### 6.4 Exit Logic (Checked Every Bar)

1. **Stop loss:** Always active, same-day stops prevented
2. **ATR-adaptive trailing stop** (after 1R profit):
   - Calm: 1.5x ATR, Normal: 2.5x ATR, Volatile: 3.5x ATR
   - Wave 5: 0.7x tighter, After 40 bars: 0.8x tighter
   - Stop only moves up, never down
   - After 1.5R profit: stop moves to breakeven + costs
3. **Partial profit-taking:** 50% at first Fibonacci target, stop to breakeven
4. **Full exit at highest target**
5. **Wave-aware fallback:** Wave 3: 20%, Wave 5: 10%, Default: 12%

### 6.5 Transaction Cost Model

| Market | Commission/Side | Sell Tax | Slippage |
|--------|----------------|----------|----------|
| Taiwan | 0.1425% | 0.3% | 0.05% |
| US | 0% | 0% | 0.03% |

### 6.6 Performance Metrics

| Metric | Description |
|--------|-------------|
| total_return_pct | Total P&L as % of initial capital |
| win_rate | Winning trades / total trades |
| profit_factor | Sum of wins / abs(sum of losses) |
| avg_r_multiple | Average R-multiple across all trades |
| max_drawdown_pct | Peak-to-trough drawdown |
| sharpe_ratio | Annualized, based on actual trade frequency |
| sortino_ratio | Downside-only volatility |
| alpha | Strategy return - buy-and-hold return |
| wave_breakdown | Per-wave: trades, win rate, total profit, avg R |

### 6.7 Monte Carlo Simulation

- 10,000 reshuffled simulations of actual trades
- Reports 5th/25th/50th/75th/95th percentile confidence intervals
- Metrics: final equity, max drawdown, Sharpe ratio
- Probability of profit and probability of ruin (< 50% of initial capital)

### 6.8 RS Rotation Benchmark

- Quarterly rebalance (every 63 bars)
- Selects top-N stocks by 6-month momentum
- Equal-weighted portfolio
- Reports CAGR, max drawdown, total return

---

## 7. GUI Desktop Application

### 7.1 Framework

wxPython desktop application with embedded matplotlib charting and HTML dashboard rendering.

### 7.2 Window Layout

```
+---------------------------------------------------------------+
| Menu Bar: File | Analysis | Data | Help                       |
+---------------------------------------------------------------+
| [Target v] [Stock v] [Chart Type v]                           |
+---------------------------------------------------------------+
| [Crawl] [Scan All] [Elliott Wave] [Analyze] [Backtest] [Load] |
+---------------------------------------------------------------+
| Left Panel (280px)     |  Right Panel                         |
| +--------------------+ | +----------------------------------+ |
| | Search + Filters   | | | Tab: Dashboard | Chart | Log    | |
| +--------------------+ | |                                  | |
| | Stock List         | | | Dashboard: HTML market overview  | |
| | Sym|Score|Act|Conf | | | Chart: matplotlib + mplfinance  | |
| |    |     |   |RS#  | | | Log: text console                | |
| +--------------------+ | |                                  | |
| | Trade Plan Panel   | | |                                  | |
| | Entry/Stop/Target  | | |                                  | |
| | R:R/Grades/RSI     | | |                                  | |
| +--------------------+ | +----------------------------------+ |
+---------------------------------------------------------------+
| Status Bar: [Status] [Progress] [Count]                       |
+---------------------------------------------------------------+
```

### 7.3 Dashboard (HTML)

- Core Index Summary (stock count, breadth, % bullish)
- Allocation Guidance (color-coded by breadth level)
- Market Regime indicator (FAVORABLE/MIXED/CAUTION)
- Signal Distribution bar
- Top Picks by Relative Strength (top 5)
- BUY Candidates table (18 columns)
- Watch List with "missing conditions"
- Exit Alerts with reasons
- Strongest Momentum (top 15)

### 7.4 Charting

- Line chart + Elliott Wave overlay
- Candlestick (Day/Week/Month) + Elliott Wave overlay
- Multi-pattern composite view (up to 3 timeframes)
- Wave labels (1-2-3-4-5, A-B-C), Fibonacci levels, current position markers
- Volume sub-panels
- Drag-able legends, adaptive date formatting

### 7.5 Filtering

- Text search (symbol name)
- Action type (BUY/WATCH/EXIT/HOLD)
- Grade quality (Grade A, A+B, C+)
- Sector (Tech, Semis, Finance, Health, Consumer, Energy, Industrial)
- Market (All/US/TW)

### 7.6 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+S | Scan All Stocks |
| Ctrl+E | Show Elliott Wave |
| Ctrl+B | Run Backtest |
| Ctrl+P | Analyze Position |
| Ctrl+L | Load Last Scan |
| Ctrl+Shift+E | Export Results |
| Ctrl+Q | Quit |

---

## 8. Key Tunable Parameters (~60+ Total)

All primary constants centralized in `SCORING_CONFIG` at `src/analysis/core/signal_scoring.py`, with additional parameters in:
- `src/analysis/core/validation.py` (ValidationConfig)
- `src/analysis/core/peaks.py` (per-timeframe tuning)
- `src/analysis/core/momentum_indicators.py` (indicator weights)
- `src/backtest/strategy_advanced.py` (strategy parameters)
- `config/config.json` (data paths, date range, fibonacci_tolerance)

---

## 9. Counter-Trend Suppression (4 Independent Layers)

The system has multiple redundant layers preventing counter-trend trades:

1. **Pattern detection** (`impulse.py`): Counter-trend patterns penalized * 0.5
2. **Pattern adapter** (`pattern_adapter.py`): Counter-trend confidence * 0.4
3. **Signal scoring** (`signal_scoring.py`): Instant AVOID classification
4. **Market regime overlay**: Raises thresholds during cautious markets

---

## 10. Strengths

1. **Multi-signal convergence:** Requires ALL signals to align (wave structure, trend, momentum, R:R, volume) before issuing BUY -- "patience is a feature"
2. **Walk-forward validation:** Prevents lookahead bias with out-of-sample-only signals
3. **Multi-layered risk management:** Position-level (2% risk, 15% cap), portfolio-level (40% max, 15% DD halt, 5 concurrent), market-level (regime detection)
4. **Market-aware costs:** Taiwan commission + tax vs US commission-free
5. **ATR-adaptive exits:** Stop distances adjust to volatility regime
6. **Wave-aware sizing:** More capital to high-probability setups (Wave 3)
7. **Relative strength overlay:** Cross-universe ranking adjusts individual signals
8. **Comprehensive validation:** 7-category data quality checks with auto-fix
9. **Deep Elliott Wave implementation:** 17+ modules covering impulse, corrective, Fibonacci, personality, subwaves, alternation, multi-degree
10. **Thread-safe GUI:** Background threading with custom events for responsive UI

---

## 11. Areas for Improvement

1. **Zero-trade generation on major stocks:** AAPL and MSFT backtest results show 0 trades for both baseline and Phase 1 -- the signal pipeline may be too restrictive
2. **Dual walk-forward implementations:** `backtester.py` (rolling windows) and `walk_forward.py` (expanding windows) could cause confusion
3. **Dual data APIs:** Functional `fetch_stock_data()` and class-based `YahooFinanceCrawler` coexist with slightly different date formatting
4. **Hardcoded sector mappings:** ~100 US tickers hardcoded in `market_structure.py`; needs updates as markets evolve
5. **Low-trade Sharpe instability:** Sharpe ratio estimation from actual trade frequency can be unstable with few trades
6. **No ML/statistical learning:** scikit-learn is in requirements but not used; parameter optimization is manual
7. **Single data source:** Yahoo Finance is the only data provider; no fallback

---

## 12. System Flow Diagram

```
                     Yahoo Finance (yfinance)
                            |
                    [Data Fetching Layer]
                    src/crawler/yahoo_finance.py
                            |
                    data/raw/{SYMBOL}.txt (TSV)
                            |
                    [Validation Layer]
                    scripts/validate_data.py
                            |
                +-----------+-----------+
                |                       |
        [Scan Pipeline]         [Backtest Pipeline]
        what_to_buy.py          run_backtest.py
                |                       |
        load_stock()            walk_forward_detect()
                |                       |
        detect_peaks_troughs    find_elliott_wave_pattern
                |                       |
        find_best_impulse_wave  adapt_wave_data
                |                       |
        adapt_wave_data         MultiTimeframeAlignment
                |               Strategy.generate_signals()
        classify_action()               |
                |               AdvancedBacktester.simulate()
        grade_conviction()              |
                |               _calculate_statistics()
        apply_relative_strength()       |
                |               Monte Carlo (optional)
        apply_market_regime()           |
                |               save_results()
        analyze_market_structure()
                |
        +-------+-------+
        |               |
    CLI Output      GUI Display
    (text/JSON)     (Dashboard + Charts)
```

---

*Report generated by multi-agent analysis of 40+ source files across 5 concurrent research threads.*
