"""
Advanced Elliott Wave Trading Strategies with Risk Management
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages position sizing and risk for individual trades."""

    def __init__(self, portfolio_value: float, risk_per_trade: float = 0.02):
        """
        Initialize position manager.

        Args:
            portfolio_value: Total portfolio value
            risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 = 2%)
        """
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade
        self.max_risk_amount = portfolio_value * risk_per_trade

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk management.

        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price

        Returns:
            Number of shares to buy
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            logger.warning("Risk per share is zero, using default position size")
            return int(self.portfolio_value * 0.1 / entry_price)

        shares = int(self.max_risk_amount / risk_per_share)

        # Ensure we don't exceed reasonable portfolio allocation
        max_shares = int(self.portfolio_value * 0.15 / entry_price)  # Max 15% per position
        shares = min(shares, max_shares)

        return max(shares, 1)  # At least 1 share

    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and recalculate risk amounts."""
        self.portfolio_value = new_value
        self.max_risk_amount = new_value * self.risk_per_trade


class RiskManager:
    """Manages overall portfolio risk and drawdown limits."""

    def __init__(self, initial_capital: float, max_daily_dd: float = 0.05,
                 max_concurrent_positions: int = 5):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting portfolio value
            max_daily_dd: Maximum daily drawdown (e.g., 0.05 = 5%)
            max_concurrent_positions: Maximum number of simultaneous positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_daily_dd = max_daily_dd
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_start_capital = initial_capital
        self.open_positions = []

    def can_open_position(self) -> bool:
        """Check if we can open a new position based on risk limits."""
        # Check position limit
        if len(self.open_positions) >= self.max_concurrent_positions:
            logger.info(f"Cannot open position: at max concurrent limit ({self.max_concurrent_positions})")
            return False

        # Check daily drawdown
        daily_dd = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_dd >= self.max_daily_dd:
            logger.warning(f"Cannot open position: daily drawdown limit reached ({daily_dd:.2%})")
            return False

        return True

    def update_capital(self, new_capital: float):
        """Update current capital and track peak."""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def reset_daily(self):
        """Reset daily tracking (call at start of each trading day)."""
        self.daily_start_capital = self.current_capital


class MultiTimeframeAlignmentStrategy:
    """
    Strategy 1: Multi-Timeframe Wave Alignment
    Trades when daily, weekly, and monthly patterns align with high confidence.

    PHASE 1 ENHANCEMENT: Now includes confidence-based filtering
    - Filters patterns by composite confidence (multi-timeframe alignment)
    - Validates pattern structure before trading
    - Tracks filtering statistics
    """

    def __init__(self, config: Dict):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration dict with parameters:
                - alignment_threshold: Minimum alignment score (default 0.70)
                - fibonacci_tolerance: Fibonacci level tolerance (default 0.15)
                - volume_multiplier: Volume confirmation multiplier (default 1.2)
                - stop_loss_pct: Stop loss percentage (default 0.02)
                - profit_targets: List of R-multiples for profit taking
                - use_confidence_filters: Enable Phase 1 filtering (default True)
                - min_composite_confidence: Minimum composite confidence (default 0.35)
                - min_validation_confidence: Minimum validation confidence (default 0.30)
        """
        self.config = config
        self.alignment_threshold = config.get('alignment_threshold', 0.40)
        self.fib_tolerance = config.get('fibonacci_tolerance', 0.05)  # Tightened from 0.15
        self.volume_mult = config.get('volume_multiplier', 1.2)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.profit_targets = config.get('profit_targets', [2.0, 3.0, 4.0])

        # PHASE 1: Confidence filtering configuration
        self.use_confidence_filters = config.get('use_confidence_filters', True)
        self.min_composite_confidence = config.get('min_composite_confidence', 0.35)
        self.min_validation_confidence = config.get('min_validation_confidence', 0.30)

        # Statistics tracking for Phase 1
        self.filter_stats = {
            'total_signals': 0,
            'passed_composite_filter': 0,
            'passed_validation_filter': 0,
            'final_signals': 0
        }

    def generate_signals(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[Dict]:
        """
        Generate entry and exit signals based on multi-timeframe alignment.

        PHASE 1 ENHANCEMENT: Now includes confidence filtering
        - GATE 1: Composite confidence (multi-timeframe pattern quality)
        - GATE 2: Validation confidence (Elliott Wave rule compliance)

        Args:
            df: Price dataframe with OHLCV
            pattern_analysis: Elliott Wave pattern analysis results

        Returns:
            List of signal dictionaries (filtered by confidence gates)
        """
        # PHASE 1 GATE 1: Check composite confidence BEFORE generating any signals
        if self.use_confidence_filters:
            composite_passed = self._check_composite_confidence_filter(pattern_analysis)
            if not composite_passed:
                logger.info("Pattern filtered: Composite confidence below threshold")
                return []  # Don't generate any signals for low-quality patterns

        signals = []

        # TREND FILTER: Skip BUY signals in bearish trends or counter-trend patterns
        trend_ctx = pattern_analysis.get('trend_context', {})
        broad_trend = trend_ctx.get('trend', 'neutral')
        pattern_dir = pattern_analysis.get('current_position', {}).get('trend_direction', 'up')

        # Only generate BUY signals for trend-aligned setups
        buy_allowed = not (broad_trend == 'bearish' or
                           (broad_trend == 'bullish' and pattern_dir == 'down'))

        # Need at least 20 bars of history for volume average
        start_bar = 20

        for i in range(start_bar, len(df)):
            date = df.index[i]
            price = df['close'].iloc[i]

            # Get volume safely
            vol_col = 'volume' if 'volume' in df.columns else 'Volume'
            if vol_col not in df.columns:
                continue
            volume = df[vol_col].iloc[i]

            # Get alignment score
            alignment = self._get_alignment_score(date, pattern_analysis)

            if alignment >= self.alignment_threshold:
                # Get wave position for this specific bar
                wave_pos = self._get_wave_position_at_bar(i, df, pattern_analysis)

                wave_num = wave_pos.get('wave_number', 0) if wave_pos else 0

                # --- Momentum gate: skip entries when momentum is hostile ---
                momentum = pattern_analysis.get('momentum', {})
                momentum_entry_ok = momentum.get('entry_ok', True)
                stop_mult = momentum.get('stop_multiplier', 1.0)

                # --- BUY signals: Wave 1 and Wave 3 entries (trend-aligned only) ---
                if wave_num in [1, 3] and buy_allowed:
                    # Check Fibonacci support
                    fib_levels = self._calculate_fib_levels(wave_pos)

                    if self._near_fib_support(price, fib_levels):
                        # Check volume
                        avg_volume = df[vol_col].iloc[max(0, i-20):i].mean()
                        if avg_volume <= 0:
                            continue

                        if volume >= avg_volume * self.volume_mult:
                            # Momentum gate: skip if momentum says no
                            if not momentum_entry_ok:
                                logger.debug(f"Signal filtered at {date}: momentum not confirming")
                                continue

                            # Generate BUY signal with velocity-adjusted stop
                            base_stop = self._calculate_stop_loss(price, wave_pos)
                            # Widen stop in volatile regimes, tighten in calm
                            stop_loss = price - (price - base_stop) * stop_mult

                            signal = {
                                'date': date,
                                'type': 'BUY',
                                'price': price,
                                'stop_loss': stop_loss,
                                'wave_number': wave_pos['wave_number'],
                                'alignment': alignment,
                                'confidence': self._calculate_confidence(alignment, wave_pos, volume, avg_volume)
                            }

                            # PHASE 1 GATE 2: Validate signal before adding
                            if self.use_confidence_filters:
                                if self._check_validation_confidence_filter(df, pattern_analysis, signal):
                                    self.filter_stats['final_signals'] += 1
                                    signals.append(signal)
                                else:
                                    logger.debug(f"Signal filtered at {date}: Failed validation confidence")
                            else:
                                signals.append(signal)

                # --- MOMENTUM EXIT: Any wave, if momentum crashes ---
                elif momentum.get('exit_warning', False) and wave_num >= 3:
                    vel = momentum.get('velocity', {})
                    speed = vel.get('speed_regime', 'flat')
                    if speed in ('crash', 'fast_down'):
                        signals.append({
                            'date': date,
                            'type': 'SELL',
                            'price': price,
                            'stop_loss': price * 1.03,
                            'wave_number': wave_num,
                            'alignment': alignment,
                            'confidence': 0.8,
                            'sell_reason': 'momentum_crash',
                        })

                # --- Wave 5/6: SELL exits or correction-bottom BUY ---
                elif wave_num in [5, 6]:
                    avg_volume = df[vol_col].iloc[max(0, i-20):i].mean()
                    if avg_volume <= 0:
                        continue

                    # --- Correction-bottom BUY: after bullish impulse completes,
                    # enter when price retraces to Fibonacci support levels ---
                    trend_dir = wave_pos.get('trend_direction', 'unknown')
                    if wave_num == 6 and trend_dir == 'up' and buy_allowed:
                        w_prices = wave_pos.get('wave_prices', [])
                        if len(w_prices) >= 2:
                            impulse_start = w_prices[0]
                            impulse_end = w_prices[-1]
                            impulse_range = impulse_end - impulse_start
                            if impulse_range > 0:
                                for fib_ratio in [0.382, 0.5, 0.618]:
                                    fib_level = impulse_end - impulse_range * fib_ratio
                                    if abs(price - fib_level) / price <= self.fib_tolerance:
                                        # Volume should be declining (correction exhaustion)
                                        if volume < avg_volume * 0.9:
                                            if not momentum_entry_ok:
                                                continue
                                            stop_loss = impulse_end - impulse_range * 0.786
                                            signal = {
                                                'date': date,
                                                'type': 'BUY',
                                                'price': price,
                                                'stop_loss': stop_loss,
                                                'wave_number': wave_num,
                                                'alignment': alignment,
                                                'confidence': 0.5 + fib_ratio * 0.3,
                                                'entry_type': 'correction_bottom',
                                                'fib_level': fib_ratio,
                                            }
                                            if self.use_confidence_filters:
                                                if self._check_validation_confidence_filter(df, pattern_analysis, signal):
                                                    self.filter_stats['final_signals'] += 1
                                                    signals.append(signal)
                                            else:
                                                signals.append(signal)
                                            break  # Only one fib level per bar

                    # --- SELL signals: Wave 5 exhaustion ---
                    sell_confidence = 0.0
                    sell_reason = ''

                    if wave_num == 6 and trend_dir != 'up':
                        # Post-impulse correction of bearish impulse — sell
                        sell_confidence = 0.7
                        sell_reason = 'post_impulse_correction'
                    elif wave_num == 5:
                        # Check for Wave 5 exhaustion signals
                        exhaustion_signals = 0

                        # Volume divergence: Wave 5 volume < Wave 3 volume
                        if volume < avg_volume * 0.7:
                            exhaustion_signals += 1

                        # Price extension check
                        w1_range = wave_pos.get('wave_1_range', 0)
                        w4_end = wave_pos.get('wave_4_end', price)
                        if w1_range > 0:
                            w5_progress = abs(price - w4_end) / w1_range
                            if w5_progress >= 0.618:
                                exhaustion_signals += 1
                            if w5_progress >= 1.0:
                                exhaustion_signals += 1

                        if exhaustion_signals >= 2:
                            sell_confidence = min(0.4 + exhaustion_signals * 0.1, 0.8)
                            sell_reason = 'wave5_exhaustion'

                    if sell_confidence > 0.3:
                        signals.append({
                            'date': date,
                            'type': 'SELL',
                            'price': price,
                            'stop_loss': price * 1.03,  # Stop above for shorts
                            'wave_number': wave_num,
                            'alignment': alignment,
                            'confidence': sell_confidence,
                            'sell_reason': sell_reason,
                        })

        return signals

    def _get_alignment_score(self, date, pattern_analysis: Dict) -> float:
        """Calculate alignment score across timeframes."""
        return pattern_analysis.get('alignment_score', 0.0)

    def _get_wave_position(self, date, pattern_analysis: Dict) -> Optional[Dict]:
        """Get current wave position from pattern analysis."""
        return pattern_analysis.get('current_position', None)

    def _get_wave_position_at_bar(self, bar_idx: int, df: pd.DataFrame,
                                  pattern_analysis: Dict) -> Optional[Dict]:
        """Get wave position for a specific bar using wave point indices."""
        position = pattern_analysis.get('current_position')
        if position is None:
            return None

        wave_indices = position.get('wave_indices', [])
        if not wave_indices:
            return position

        # Determine which wave this bar is in
        wave_num = 0
        for i in range(len(wave_indices) - 1):
            if wave_indices[i] <= bar_idx <= wave_indices[i + 1]:
                wave_num = i + 1
                break
        else:
            if bar_idx > wave_indices[-1]:
                if len(wave_indices) >= 6:
                    wave_num = 6  # correction
                else:
                    wave_num = len(wave_indices)

        # Return position with bar-specific wave number
        bar_position = dict(position)
        bar_position['wave_number'] = wave_num
        return bar_position

    def _calculate_fib_levels(self, wave_pos: Dict) -> List[float]:
        """Calculate Fibonacci retracement levels."""
        if 'previous_wave_high' in wave_pos and 'previous_wave_low' in wave_pos:
            high = wave_pos['previous_wave_high']
            low = wave_pos['previous_wave_low']
            diff = high - low

            return [
                high - diff * 0.236,  # 23.6%
                high - diff * 0.382,  # 38.2%
                high - diff * 0.500,  # 50%
                high - diff * 0.618,  # 61.8%
                high - diff * 0.786   # 78.6%
            ]
        return []

    def _near_fib_support(self, price: float, fib_levels: List[float]) -> bool:
        """Check if price is near a Fibonacci support level."""
        for level in fib_levels:
            if abs(price - level) / price <= self.fib_tolerance:
                return True
        return False

    def _calculate_stop_loss(self, entry_price: float, wave_pos: Dict) -> float:
        """Calculate stop loss based on wave structure."""
        # Use Wave 2 low if available, otherwise percentage-based
        if 'wave_2_low' in wave_pos:
            return wave_pos['wave_2_low'] * 0.98
        else:
            return entry_price * (1 - self.stop_loss_pct)

    def _calculate_confidence(self, alignment: float, wave_pos: Dict,
                             volume: float, avg_volume: float) -> float:
        """Calculate overall trade confidence score."""
        # Weighted scoring
        alignment_score = alignment * 0.5
        wave_score = 0.3 if wave_pos.get('wave_number') == 3 else 0.2
        volume_score = min((volume / avg_volume - 1.0) * 0.2, 0.3)

        return min(alignment_score + wave_score + volume_score, 1.0)

    # ========================================================================
    # PHASE 1: Confidence Filtering Methods
    # ========================================================================

    def _check_composite_confidence_filter(self, pattern_analysis: Dict) -> bool:
        """
        PHASE 1 GATE 1: Filter by composite confidence from multi-pattern analysis

        Composite confidence comes from impulse.py's multi-timeframe detection:
        - Analyzes day, week, month patterns
        - Checks pattern relationships (confirmations/conflicts)
        - Calculates composite score

        Returns:
            True if pattern passes composite confidence threshold
        """
        self.filter_stats['total_signals'] += 1

        # Extract best pattern from advanced analysis
        best_pattern = pattern_analysis.get('best_pattern', {})

        if 'error' in best_pattern:
            logger.debug("No valid best_pattern found in analysis")
            return False

        # Get composite confidence
        composite_confidence = best_pattern.get('composite_confidence', 0.0)

        logger.debug(f"Composite confidence: {composite_confidence:.2%} "
                    f"(threshold: {self.min_composite_confidence:.2%})")

        # Check threshold
        if composite_confidence >= self.min_composite_confidence:
            self.filter_stats['passed_composite_filter'] += 1
            return True

        return False

    def _check_validation_confidence_filter(self, df: pd.DataFrame,
                                           pattern_analysis: Dict,
                                           signal: Dict) -> bool:
        """
        PHASE 1 GATE 2: Validate Elliott Wave pattern structure

        Uses validation.py to check:
        - Wave directions (alternating up/down)
        - Wave 2 retracement (not > 100% of Wave 1)
        - Wave 3 length (not shortest wave)
        - Wave 4 overlap (doesn't overlap Wave 1)
        - Fibonacci relationships

        Returns:
            True if pattern passes validation confidence threshold
        """
        try:
            # Import validation module
            from src.analysis.core.validation import validate_elliott_wave_pattern

            # Get wave points from best pattern
            best_pattern = pattern_analysis.get('best_pattern', {})
            base_pattern = best_pattern.get('base_pattern', {})
            pattern_data = base_pattern.get('pattern', {})
            wave_points = pattern_data.get('points', [])

            if len(wave_points) < 5:
                logger.debug("Insufficient wave points for validation")
                return False

            # Run validation
            validation_result = validate_elliott_wave_pattern(
                df,
                wave_points,
                column='close',
                pattern_type='impulse'
            )

            validation_confidence = validation_result.get('confidence', 0.0)

            logger.debug(f"Validation confidence: {validation_confidence:.2%} "
                        f"(threshold: {self.min_validation_confidence:.2%})")

            # Check threshold
            if validation_confidence >= self.min_validation_confidence:
                self.filter_stats['passed_validation_filter'] += 1

                # Add validation info to signal
                signal['validation_confidence'] = validation_confidence
                signal['validation_details'] = validation_result.get('validation_details', {})

                return True

            return False

        except Exception as e:
            logger.warning(f"Validation filter error: {e}")
            # If validation fails, don't reject signal (fail-safe)
            return True

    def get_filter_statistics(self) -> Dict:
        """
        Get Phase 1 filtering statistics

        Returns:
            Dictionary with filtering metrics
        """
        stats = self.filter_stats.copy()

        if stats['total_signals'] > 0:
            stats['composite_pass_rate'] = stats['passed_composite_filter'] / stats['total_signals']
            stats['validation_pass_rate'] = (stats['passed_validation_filter'] /
                                            stats['passed_composite_filter']
                                            if stats['passed_composite_filter'] > 0 else 0)
            stats['overall_pass_rate'] = stats['final_signals'] / stats['total_signals']
            stats['signals_filtered'] = stats['total_signals'] - stats['final_signals']
            stats['filter_percentage'] = (stats['signals_filtered'] / stats['total_signals']) * 100
        else:
            stats['composite_pass_rate'] = 0
            stats['validation_pass_rate'] = 0
            stats['overall_pass_rate'] = 0
            stats['signals_filtered'] = 0
            stats['filter_percentage'] = 0

        return stats


class FibonacciMeanReversionStrategy:
    """
    Strategy 2: Fibonacci Retracement Mean Reversion
    Trades corrections at validated Fibonacci retracement levels.
    """

    def __init__(self, config: Dict):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.fib_levels = config.get('fib_entry_levels', [0.382, 0.618, 0.786])
        self.fib_tolerance = config.get('fibonacci_tolerance', 0.05)  # Tightened from 0.15
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)
        self.volume_decline_threshold = config.get('volume_decline', 0.7)

    def generate_signals(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[Dict]:
        """
        Generate mean reversion signals at Fibonacci levels.

        Args:
            df: Price dataframe
            pattern_analysis: Elliott Wave analysis (adapted format)

        Returns:
            List of signal dictionaries
        """
        signals = []
        wave_pos = pattern_analysis.get('current_position', {})
        if not wave_pos:
            return signals

        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        if vol_col not in df.columns:
            return signals

        wave_indices = wave_pos.get('wave_indices', [])

        for i in range(20, len(df)):
            date = df.index[i]
            price = df['close'].iloc[i]
            volume = df[vol_col].iloc[i]

            # Determine wave number at this bar
            bar_wave = 0
            for wi in range(len(wave_indices) - 1):
                if wave_indices[wi] <= i <= wave_indices[wi + 1]:
                    bar_wave = wi + 1
                    break

            # Check if in correction (Wave 2 or Wave 4)
            if bar_wave in [2, 4]:
                impulse_high = wave_pos.get('impulse_high', 0)
                impulse_low = wave_pos.get('impulse_low', 0)

                if impulse_high > 0 and impulse_low > 0:
                    diff = impulse_high - impulse_low

                    for fib_ratio in self.fib_levels:
                        fib_level = impulse_high - diff * fib_ratio

                        if abs(price - fib_level) / price <= self.fib_tolerance:
                            impulse_avg_vol = wave_pos.get('impulse_avg_volume', volume * 2)
                            avg_volume = df[vol_col].iloc[i-20:i].mean()
                            if avg_volume <= 0:
                                continue

                            if volume < impulse_avg_vol * self.volume_decline_threshold:
                                stop_loss = fib_level * (1 - self.stop_loss_pct)

                                signals.append({
                                    'date': date,
                                    'type': 'BUY',
                                    'price': price,
                                    'stop_loss': stop_loss,
                                    'fib_level': fib_ratio,
                                    'wave_number': bar_wave,
                                    'confidence': self._calculate_confidence(fib_ratio, volume, avg_volume)
                                })
                                break

        return signals

    def _calculate_confidence(self, fib_ratio: float, volume: float, avg_volume: float) -> float:
        """Calculate confidence score for mean reversion trade."""
        # Higher Fibonacci levels = higher confidence
        fib_score = fib_ratio * 0.5  # 0.786 level = 0.39 score

        # Lower volume = higher confidence (exhaustion)
        vol_ratio = volume / avg_volume
        volume_score = max(0, (1.5 - vol_ratio) * 0.3)

        return min(fib_score + volume_score, 1.0)


class PatternBreakoutStrategy:
    """
    Strategy 3: Pattern Breakout Trading
    Trades Wave 3 starts with volume confirmation.
    """

    def __init__(self, config: Dict):
        """Initialize breakout strategy."""
        self.config = config
        self.alignment_threshold = config.get('alignment_threshold', 0.75)
        self.breakout_pct = config.get('breakout_percentage', 0.01)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)

    def generate_signals(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[Dict]:
        """Generate breakout signals."""
        signals = []
        wave_pos = pattern_analysis.get('current_position', {})
        if not wave_pos:
            return signals

        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        if vol_col not in df.columns:
            return signals

        wave_indices = wave_pos.get('wave_indices', [])
        wave_1_high = wave_pos.get('wave_1_high', 0)

        for i in range(20, len(df)):
            date = df.index[i]
            price = df['close'].iloc[i]
            volume = df[vol_col].iloc[i]

            # Determine wave number at this bar
            bar_wave = 0
            for wi in range(len(wave_indices) - 1):
                if wave_indices[wi] <= i <= wave_indices[wi + 1]:
                    bar_wave = wi + 1
                    break
            if i > wave_indices[-1] if wave_indices else True:
                if len(wave_indices) >= 3:
                    bar_wave = 3  # Could be starting wave 3

            # Check for Wave 3 starting (breakout above Wave 1 high)
            if bar_wave == 3 and wave_1_high > 0:
                if price > wave_1_high * (1 + self.breakout_pct):
                    avg_volume = df[vol_col].iloc[i-20:i].mean()
                    if avg_volume <= 0:
                        continue

                    if volume >= avg_volume * self.volume_multiplier:
                        alignment = pattern_analysis.get('alignment_score', 0)

                        if alignment >= self.alignment_threshold:
                            stop_loss = wave_1_high * 0.98

                            signals.append({
                                'date': date,
                                'type': 'BUY',
                                'price': price,
                                'stop_loss': stop_loss,
                                'wave_number': 3,
                                'alignment': alignment,
                                'breakout_level': wave_1_high,
                                'confidence': self._calculate_confidence(alignment, volume, avg_volume)
                            })

        return signals

    def _calculate_confidence(self, alignment: float, volume: float, avg_volume: float) -> float:
        """Calculate breakout confidence."""
        alignment_score = alignment * 0.6
        volume_score = min((volume / avg_volume - 1.0) * 0.2, 0.4)
        return min(alignment_score + volume_score, 1.0)


class TransactionCostModel:
    """Models realistic trading friction: commissions, taxes, slippage."""

    def __init__(self, commission_rate: float = 0.001425,
                 sell_tax_rate: float = 0.003,
                 slippage_rate: float = 0.0005):
        """
        Args:
            commission_rate: Commission per side (default 0.1425% for Taiwan TWSE)
            sell_tax_rate: Securities transaction tax on sells (default 0.3% for Taiwan)
            slippage_rate: Estimated slippage per trade (default 0.05%)
        """
        self.commission_rate = commission_rate
        self.sell_tax_rate = sell_tax_rate
        self.slippage_rate = slippage_rate

    def buy_cost(self, price: float, shares: int) -> float:
        notional = price * shares
        return notional * (self.commission_rate + self.slippage_rate)

    def sell_cost(self, price: float, shares: int) -> float:
        notional = price * shares
        return notional * (self.commission_rate + self.sell_tax_rate + self.slippage_rate)

    def round_trip_cost(self, entry_price: float, exit_price: float, shares: int) -> float:
        return self.buy_cost(entry_price, shares) + self.sell_cost(exit_price, shares)


class AdvancedBacktester:
    """
    Advanced backtester with full risk management and multiple strategies.
    """

    def __init__(self, initial_capital: float = 100000, config: Dict = None):
        """
        Initialize advanced backtester.

        Args:
            initial_capital: Starting portfolio value
            config: Strategy and risk configuration
        """
        self.initial_capital = initial_capital
        self.config = config or {}

        # Initialize risk management
        self.risk_manager = RiskManager(
            initial_capital,
            max_daily_dd=self.config.get('max_daily_drawdown', 0.05),
            max_concurrent_positions=self.config.get('max_concurrent_positions', 5)
        )

        self.position_manager = PositionManager(
            initial_capital,
            risk_per_trade=self.config.get('risk_per_trade', 0.02)
        )

        # Transaction cost model (default: US market, low friction)
        self.cost_model = TransactionCostModel(
            commission_rate=self.config.get('commission_rate', 0.001),
            sell_tax_rate=self.config.get('sell_tax_rate', 0.0),
            slippage_rate=self.config.get('slippage_rate', 0.0005),
        )
        self.total_costs = 0.0

        # Initialize strategies
        self.strategies = {
            'multiframe': MultiTimeframeAlignmentStrategy(self.config),
            'fibonacci_mr': FibonacciMeanReversionStrategy(self.config),
            'breakout': PatternBreakoutStrategy(self.config)
        }

        # Track results
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}

    def run_backtest(self, df: pd.DataFrame, pattern_analysis: Dict,
                     strategy_name: str = 'multiframe') -> Dict:
        """
        Run backtest with specified strategy.

        Args:
            df: Price dataframe with OHLCV
            pattern_analysis: Elliott Wave pattern analysis results
            strategy_name: Which strategy to use

        Returns:
            Dictionary of backtest results
        """
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Generate all signals
        signals = strategy.generate_signals(df, pattern_analysis)
        logger.info(f"Generated {len(signals)} signals for {strategy_name}")

        # Simulate trading
        current_capital = self.initial_capital
        self.risk_manager.current_capital = current_capital
        self.risk_manager.reset_daily()

        # Build signal lookup by date for O(1) access
        signal_map = {}
        for signal in signals:
            signal_map.setdefault(signal['date'], []).append(signal)

        last_entry_date = None

        # Iterate over EVERY bar — exits must be checked daily, not just on signal dates
        for bar_idx in range(len(df)):
            date = df.index[bar_idx]

            # Check exits FIRST (before processing new entries)
            if self.open_positions:
                current_capital = self._check_exits(df, date, current_capital)

            # Process signals for this date
            for signal in signal_map.get(date, []):
                if signal['type'] == 'BUY':
                    # Cooldown: skip if we opened a position within the last 5 bars
                    if last_entry_date is not None:
                        try:
                            last_idx = df.index.get_loc(last_entry_date)
                            if bar_idx - last_idx < 5:
                                continue
                        except (KeyError, TypeError):
                            pass

                    if self.risk_manager.can_open_position():
                        # Calculate position size
                        shares = self.position_manager.calculate_position_size(
                            signal['price'],
                            signal['stop_loss']
                        )

                        # Open position
                        position_value = shares * signal['price']

                        if position_value <= current_capital * 0.4:  # Max 40% per position
                            # Compute wave-based targets from pattern analysis
                            targets = self._compute_wave_targets(signal, pattern_analysis)

                            position = {
                                'entry_date': signal['date'],
                                'entry_price': signal['price'],
                                'shares': shares,
                                'stop_loss': signal['stop_loss'],
                                'wave_number': signal.get('wave_number'),
                                'confidence': signal.get('confidence', 0.5),
                                'targets': targets,
                            }

                            self.open_positions[signal['date']] = position
                            self.risk_manager.open_positions.append(position)
                            buy_cost = self.cost_model.buy_cost(signal['price'], shares)
                            current_capital -= position_value + buy_cost
                            self.total_costs += buy_cost
                            last_entry_date = signal['date']

                            logger.info(f"Opened position: {shares} shares @ ${signal['price']:.2f} (cost: ${buy_cost:.2f})")

                elif signal['type'] == 'SELL':
                    # SELL signal: close open positions (profit-taking / wave completion)
                    if self.open_positions:
                        sell_reason = signal.get('sell_reason', 'sell_signal')
                        for entry_date in list(self.open_positions.keys()):
                            current_capital = self._close_position(
                                entry_date, signal['date'], signal['price'],
                                sell_reason, current_capital
                            )

        # Close any remaining positions at end
        if len(df) > 0:
            current_capital = self._close_all_positions(df.index[-1], df['close'].iloc[-1], current_capital)

        # Calculate statistics
        return self._calculate_statistics()

    def _check_exits(self, df: pd.DataFrame, current_date, current_capital: float) -> float:
        """Check for exit conditions on open positions using wave-aware logic."""
        positions_to_close = []

        for entry_date, position in self.open_positions.items():
            if current_date not in df.index:
                continue

            # Skip exit checks on entry day — don't stop out same-day
            if current_date == entry_date:
                continue

            current_price = df.loc[current_date, 'close']

            # 1. Check stop loss (always active)
            if current_price <= position['stop_loss']:
                positions_to_close.append((entry_date, current_price, 'stop_loss'))
                continue

            # 2. Update trailing stop if position is in profit
            entry_price = position['entry_price']
            risk_per_share = abs(entry_price - position['stop_loss'])
            profit_per_share = current_price - entry_price

            # Track highest price since entry for trailing stop
            if 'highest_price' not in position:
                position['highest_price'] = entry_price
            position['highest_price'] = max(position['highest_price'], current_price)

            # Activate trailing stop after reaching 1R profit
            if risk_per_share > 0 and profit_per_share >= risk_per_share:
                # Trail at 2x the risk distance below highest price
                trail_distance = risk_per_share * 2.0
                trailing_stop = position['highest_price'] - trail_distance

                # Move stop up (never down)
                if trailing_stop > position['stop_loss']:
                    position['stop_loss'] = trailing_stop

                # Also move to breakeven after 1.5R profit
                if profit_per_share >= risk_per_share * 1.5:
                    position['stop_loss'] = max(position['stop_loss'], entry_price * 1.005)

            # 3. Wave-based profit targets
            wave_num = position.get('wave_number', 0)
            targets = position.get('targets', [])

            if targets:
                # Check if highest exceeded target is reached (sort descending)
                for target in sorted(targets, reverse=True):
                    if current_price >= target:
                        positions_to_close.append((entry_date, current_price, f'target_{target:.2f}'))
                        break
            else:
                # Fallback: wave-aware profit targets
                profit_pct = profit_per_share / entry_price if entry_price > 0 else 0

                if wave_num == 3:
                    # Wave 3 trades: let them run, exit at 20%+ or on trailing stop
                    if profit_pct >= 0.20:
                        positions_to_close.append((entry_date, current_price, 'wave3_target'))
                elif wave_num == 5:
                    # Wave 5: tighter target, exhaustion risk
                    if profit_pct >= 0.10:
                        positions_to_close.append((entry_date, current_price, 'wave5_target'))
                else:
                    # Default: moderate target
                    if profit_pct >= 0.12:
                        positions_to_close.append((entry_date, current_price, 'profit_target'))

        # Close positions, propagating updated capital
        for entry_date, exit_price, reason in positions_to_close:
            current_capital = self._close_position(entry_date, current_date, exit_price, reason, current_capital)
        return current_capital

    def _close_position(self, entry_date, exit_date, exit_price: float,
                       reason: str, current_capital: float) -> float:
        """Close a position and record the trade. Returns updated capital."""
        position = self.open_positions[entry_date]

        # Gross profit before costs
        sell_cost = self.cost_model.sell_cost(exit_price, position['shares'])
        self.total_costs += sell_cost
        round_trip = self.cost_model.buy_cost(position['entry_price'], position['shares']) + sell_cost

        profit = (exit_price - position['entry_price']) * position['shares'] - round_trip
        profit_pct = profit / (position['entry_price'] * position['shares']) * 100

        # Guard against division by zero in R-multiple calculation
        risk_per_share = abs(position['entry_price'] - position['stop_loss'])
        if risk_per_share > 0 and position['shares'] > 0:
            r_multiple = profit / (risk_per_share * position['shares'])
        else:
            r_multiple = 0.0

        trade = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': reason,
            'wave_number': position.get('wave_number'),
            'confidence': position.get('confidence'),
            'r_multiple': r_multiple
        }

        self.trades.append(trade)

        # Update capital and propagate to risk/position managers
        current_capital += exit_price * position['shares'] - sell_cost
        self.risk_manager.update_capital(current_capital)
        self.position_manager.update_portfolio_value(current_capital)

        # Remove from open positions
        del self.open_positions[entry_date]
        self.risk_manager.open_positions = [p for p in self.risk_manager.open_positions
                                            if p != position]

        logger.info(f"Closed position: {reason} - P/L: ${profit:.2f} ({profit_pct:.2f}%)")
        return current_capital

    def _close_all_positions(self, exit_date, exit_price: float, current_capital: float) -> float:
        """Close all remaining positions. Returns updated capital."""
        for entry_date in list(self.open_positions.keys()):
            current_capital = self._close_position(entry_date, exit_date, exit_price, 'end_of_backtest', current_capital)
        return current_capital

    def _compute_wave_targets(self, signal: Dict, pattern_analysis: Dict) -> List[float]:
        """Compute Fibonacci-based profit targets from wave structure."""
        targets = []
        wave_pos = pattern_analysis.get('current_position', {})
        if not wave_pos:
            return targets

        wave_num = signal.get('wave_number', 0)
        entry_price = signal['price']
        wave_1_range = wave_pos.get('wave_1_range', 0)

        if wave_1_range <= 0:
            return targets

        if wave_num == 3:
            # Wave 3 targets: Fibonacci extensions of Wave 1
            wave_2_end = wave_pos.get('wave_2_end', entry_price)
            targets = [
                wave_2_end + wave_1_range * 1.618,
                wave_2_end + wave_1_range * 2.618,
            ]
        elif wave_num == 1:
            # Wave 1: modest target
            targets = [entry_price + wave_1_range * 0.618]
        elif wave_num == 5:
            # Wave 5: conservative targets
            wave_4_end = wave_pos.get('wave_4_end', entry_price)
            targets = [
                wave_4_end + wave_1_range * 0.618,
                wave_4_end + wave_1_range * 1.0,
            ]

        # Filter out targets below entry
        targets = [t for t in targets if t > entry_price * 1.01]
        return targets

    def _calculate_statistics(self) -> Dict:
        """Calculate comprehensive backtest statistics."""
        if not self.trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_return_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'profit_factor': 0,
                'avg_r_multiple': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0
            }

        trades_df = pd.DataFrame(self.trades)

        winning_trades = trades_df[trades_df['profit'] > 0]
        losing_trades = trades_df[trades_df['profit'] <= 0]

        total_profit = trades_df['profit'].sum()
        final_capital = self.initial_capital + total_profit

        stats = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'total_profit': total_profit,
            'total_return_pct': (total_profit / self.initial_capital) * 100,
            'avg_win': winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['profit'].mean() if len(losing_trades) > 0 else 0,
            'avg_win_pct': winning_trades['profit_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['profit_pct'].mean() if len(losing_trades) > 0 else 0,
            'best_trade': trades_df['profit'].max(),
            'worst_trade': trades_df['profit'].min(),
            'avg_r_multiple': trades_df['r_multiple'].mean(),
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum())
                           if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else 0,
            'max_consecutive_wins': self._max_consecutive(trades_df, 'profit', lambda x: x > 0),
            'max_consecutive_losses': self._max_consecutive(trades_df, 'profit', lambda x: x <= 0),
        }

        # Calculate max drawdown
        equity = self.initial_capital
        equity_curve = [equity]

        for trade in self.trades:
            equity += trade['profit']
            equity_curve.append(equity)

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        stats['max_drawdown_pct'] = max_dd * 100
        stats['max_drawdown'] = max_dd
        stats['sharpe_ratio'] = self._calculate_sharpe(trades_df)
        stats['total_costs'] = self.total_costs
        stats['trades'] = self.trades

        return stats

    def _max_consecutive(self, df: pd.DataFrame, column: str, condition) -> int:
        """Calculate maximum consecutive occurrences of a condition."""
        max_streak = 0
        current_streak = 0

        for value in df[column]:
            if condition(value):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_sharpe(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized) using actual trade frequency."""
        if len(trades_df) < 2:
            return 0

        returns = trades_df['profit_pct'] / 100
        avg_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0

        # Estimate trades per year from actual data
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            try:
                entry_dates = pd.to_datetime(trades_df['entry_date'])
                total_days = (entry_dates.max() - entry_dates.min()).days
                if total_days > 0:
                    trades_per_year = len(trades_df) / (total_days / 252)
                else:
                    trades_per_year = 30
            except Exception:
                trades_per_year = 30
        else:
            trades_per_year = 30

        trades_per_year = max(trades_per_year, 1)
        sharpe = (avg_return - risk_free_rate / trades_per_year) / std_return * np.sqrt(trades_per_year)
        return sharpe

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Return trades as DataFrame for analysis."""
        return pd.DataFrame(self.trades)

    def print_summary(self):
        """Print formatted summary of backtest results."""
        stats = self._calculate_statistics()

        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial Capital:        ${self.initial_capital:,.2f}")
        logger.info(f"Final Capital:          ${stats['final_capital']:,.2f}")
        logger.info(f"Total Return:           {stats['total_return_pct']:.2f}%")
        logger.info(f"Total Profit:           ${stats['total_profit']:,.2f}")
        logger.info("-"*60)
        logger.info(f"Total Trades:           {stats['total_trades']}")
        logger.info(f"Winning Trades:         {stats['winning_trades']}")
        logger.info(f"Losing Trades:          {stats['losing_trades']}")
        logger.info(f"Win Rate:               {stats['win_rate']*100:.2f}%")
        logger.info("-"*60)
        logger.info(f"Average Win:            ${stats['avg_win']:.2f} ({stats['avg_win_pct']:.2f}%)")
        logger.info(f"Average Loss:           ${stats['avg_loss']:.2f} ({stats['avg_loss_pct']:.2f}%)")
        logger.info(f"Profit Factor:          {stats['profit_factor']:.2f}")
        logger.info(f"Average R-Multiple:     {stats['avg_r_multiple']:.2f}")
        logger.info("-"*60)
        logger.info(f"Best Trade:             ${stats['best_trade']:.2f}")
        logger.info(f"Worst Trade:            ${stats['worst_trade']:.2f}")
        logger.info(f"Max Drawdown:           {stats['max_drawdown_pct']:.2f}%")
        logger.info(f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}")
        logger.info("-"*60)
        logger.info(f"Max Consecutive Wins:   {stats['max_consecutive_wins']}")
        logger.info(f"Max Consecutive Losses: {stats['max_consecutive_losses']}")
        logger.info("="*60 + "\n")
