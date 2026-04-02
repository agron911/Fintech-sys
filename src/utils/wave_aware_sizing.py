"""
Wave-Aware Position Sizing
Extends standard risk management with Elliott Wave-specific logic.

Different waves have different probabilities of success:
- Wave 3: Highest probability → Maximum size
- Wave 5: Medium probability → Reduced size
- Wave 4: Lowest probability → Minimum size
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from src.utils.risk_management import PositionSizer, Position

logger = logging.getLogger(__name__)


@dataclass
class WavePositionMetrics:
    """Metrics for wave-based position sizing decisions."""
    wave_number: int
    wave_type: str
    confidence: float
    is_diagonal: bool
    complex_correction: bool
    timeframe_alignment: float  # 0-1
    fibonacci_confluence: int  # Number of Fib levels agreeing


class WaveAwarePositionSizer(PositionSizer):
    """
    Position sizer that adjusts for Elliott Wave characteristics.

    Key Principles:
    1. Wave 3 entries get maximum size (highest probability)
    2. Wave 5 entries get reduced size (exhaustion risk)
    3. Wave 4 trades use minimum size (choppy, unclear)
    4. Diagonal patterns use reduced size (more ambiguous)
    5. Multi-timeframe alignment increases size
    """

    def __init__(self, portfolio_value: float, max_risk_per_trade: float = 0.02,
                 wave_config: Dict = None):
        """
        Initialize wave-aware position sizer.

        Args:
            portfolio_value: Total portfolio value
            max_risk_per_trade: Maximum risk per trade (default 2%)
            wave_config: Wave-specific configuration
        """
        super().__init__(portfolio_value, max_risk_per_trade)

        # Wave-specific risk multipliers
        self.wave_config = wave_config or {}
        self.wave_multipliers = self.wave_config.get('wave_multipliers', {
            1: 0.6,   # Wave 1: Medium size (establishing trend)
            2: 0.7,   # Wave 2: Good size (high probability entry)
            3: 1.0,   # Wave 3: Maximum size (highest probability)
            4: 0.3,   # Wave 4: Small size (choppy, unclear)
            5: 0.5,   # Wave 5: Medium size (exhaustion risk)
        })

        # Pattern adjustments
        self.diagonal_multiplier = self.wave_config.get('diagonal_multiplier', 0.7)
        self.correction_multiplier = self.wave_config.get('correction_multiplier', 0.7)

        # Timeframe alignment bonus
        self.alignment_bonus_max = self.wave_config.get('alignment_bonus', 0.2)

        # Fibonacci confluence bonus
        self.fib_confluence_bonus = self.wave_config.get('fib_confluence_bonus', 0.05)

    def calculate_wave_adjusted_size(self, entry_price: float, stop_loss: float,
                                    wave_metrics: WavePositionMetrics) -> int:
        """
        Calculate position size with wave-specific adjustments.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            wave_metrics: Wave characteristics

        Returns:
            Number of shares to buy
        """
        # Start with base confidence
        adjusted_confidence = wave_metrics.confidence

        # Apply wave number multiplier
        wave_adjustment = self.wave_multipliers.get(wave_metrics.wave_number, 0.5)
        adjusted_confidence *= wave_adjustment

        logger.debug(f"Wave {wave_metrics.wave_number} multiplier: {wave_adjustment:.2f}")

        # Reduce for diagonal patterns
        if wave_metrics.is_diagonal:
            adjusted_confidence *= self.diagonal_multiplier
            logger.debug(f"Diagonal pattern: reduced by {self.diagonal_multiplier:.2f}")

        # Reduce for complex corrections
        if wave_metrics.complex_correction:
            adjusted_confidence *= self.correction_multiplier
            logger.debug(f"Complex correction: reduced by {self.correction_multiplier:.2f}")

        # Bonus for timeframe alignment
        if wave_metrics.timeframe_alignment > 0.7:
            alignment_bonus = self.alignment_bonus_max * wave_metrics.timeframe_alignment
            adjusted_confidence = min(adjusted_confidence + alignment_bonus, 1.0)
            logger.debug(f"Timeframe alignment bonus: +{alignment_bonus:.2f}")

        # Bonus for Fibonacci confluence
        if wave_metrics.fibonacci_confluence >= 3:
            fib_bonus = self.fib_confluence_bonus * (wave_metrics.fibonacci_confluence - 2)
            adjusted_confidence = min(adjusted_confidence + fib_bonus, 1.0)
            logger.debug(f"Fibonacci confluence bonus: +{fib_bonus:.2f}")

        # Ensure confidence stays in valid range
        adjusted_confidence = max(0.1, min(adjusted_confidence, 1.0))

        logger.info(f"Final adjusted confidence: {adjusted_confidence:.2f} "
                   f"(base: {wave_metrics.confidence:.2f})")

        # Calculate size using base risk-based method with adjusted confidence
        return self.risk_based_size(entry_price, stop_loss, adjusted_confidence)

    def calculate_wave_3_aggressive_size(self, entry_price: float, stop_loss: float,
                                        wave_metrics: WavePositionMetrics) -> int:
        """
        Calculate aggressive position size for high-conviction Wave 3 setups.

        Wave 3 is the highest probability wave, so we can safely use
        larger position sizes when confidence is high.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            wave_metrics: Must be Wave 3 with high confidence

        Returns:
            Number of shares (potentially larger than normal)
        """
        if wave_metrics.wave_number != 3:
            logger.warning("Aggressive sizing only for Wave 3")
            return self.calculate_wave_adjusted_size(entry_price, stop_loss, wave_metrics)

        # Check confidence threshold
        if wave_metrics.confidence < 0.70:
            logger.info("Confidence too low for aggressive Wave 3 sizing")
            return self.calculate_wave_adjusted_size(entry_price, stop_loss, wave_metrics)

        # Allow up to 2x normal risk for high-confidence Wave 3
        # But still cap total position size
        aggressive_confidence = min(wave_metrics.confidence * 1.5, 1.0)

        # Apply additional checks
        if wave_metrics.timeframe_alignment > 0.8 and wave_metrics.fibonacci_confluence >= 3:
            # Perfect setup - use maximum size
            aggressive_confidence = 1.0
            logger.info("Perfect Wave 3 setup - using maximum size")
        elif wave_metrics.is_diagonal:
            # Reduce even for Wave 3 if diagonal
            aggressive_confidence *= 0.8

        shares = self.risk_based_size(entry_price, stop_loss, aggressive_confidence)

        # Double-check we're not over-allocating
        position_value = shares * entry_price
        max_position_value = self.portfolio_value * 0.20  # Max 20% per position

        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            logger.warning(f"Capped position size to 20% of portfolio")

        return shares

    def calculate_progressive_entry_sizes(self, entry_price: float, stop_loss: float,
                                         wave_metrics: WavePositionMetrics,
                                         stages: int = 3) -> List[Dict[str, Any]]:
        """
        Calculate progressive entry sizes for scaling into Wave 3.

        Strategy:
        - Stage 1 (30%): At Wave 2 completion
        - Stage 2 (30%): When Wave 3 exceeds Wave 1 length
        - Stage 3 (40%): When Wave 3 breaks strongly above Wave 1

        Args:
            entry_price: Initial entry price
            stop_loss: Stop loss price
            wave_metrics: Wave 3 metrics
            stages: Number of entry stages (default 3)

        Returns:
            List of entry stage dictionaries
        """
        if wave_metrics.wave_number != 3:
            logger.warning("Progressive entries designed for Wave 3")
            return []

        # Calculate total target size
        total_shares = self.calculate_wave_adjusted_size(
            entry_price, stop_loss, wave_metrics
        )

        if stages == 3:
            # 3-stage entry
            allocations = [0.30, 0.30, 0.40]
            conditions = [
                "Wave 2 completion confirmed",
                "Wave 3 exceeds Wave 1 length (1.0x)",
                "Wave 3 strong momentum (1.618x Wave 1)"
            ]
            price_targets = [
                entry_price,
                entry_price * 1.05,  # ~5% higher
                entry_price * 1.12   # ~12% higher
            ]

        elif stages == 2:
            # 2-stage entry (simpler)
            allocations = [0.50, 0.50]
            conditions = [
                "Wave 2 completion confirmed",
                "Wave 3 momentum confirmed"
            ]
            price_targets = [
                entry_price,
                entry_price * 1.08
            ]

        else:
            # Single entry
            return [{
                'stage': 1,
                'shares': total_shares,
                'condition': "Wave 2 completion",
                'price_target': entry_price
            }]

        # Build entry stages
        entries = []
        for i, (allocation, condition, price) in enumerate(zip(allocations, conditions, price_targets), 1):
            entries.append({
                'stage': i,
                'shares': int(total_shares * allocation),
                'allocation_pct': allocation,
                'condition': condition,
                'price_target': price,
                'cumulative_shares': sum(int(total_shares * allocations[j]) for j in range(i))
            })

        return entries


class WaveBasedPositionManager:
    """
    Manages positions through wave progression.

    Handles:
    - Progressive entries in Wave 3
    - Position reduction during Wave 4
    - Profit taking in Wave 5
    - Stop loss adjustments based on wave structure
    """

    def __init__(self, sizer: WaveAwarePositionSizer):
        """
        Initialize manager.

        Args:
            sizer: Wave-aware position sizer
        """
        self.sizer = sizer
        self.positions: Dict[str, Position] = {}
        self.wave_states: Dict[str, WavePositionMetrics] = {}

    def scale_into_wave_3(self, symbol: str, current_price: float,
                         stop_loss: float, wave_metrics: WavePositionMetrics,
                         wave_progress: float) -> Optional[Dict[str, Any]]:
        """
        Determine if should add to Wave 3 position based on progress.

        Args:
            symbol: Trading symbol
            current_price: Current price
            stop_loss: Current stop loss
            wave_metrics: Wave metrics
            wave_progress: Wave 3 progress (0=start, 1=complete)

        Returns:
            Action dictionary or None
        """
        if symbol not in self.positions:
            logger.warning(f"No existing position for {symbol}")
            return None

        position = self.positions[symbol]
        stages = self.sizer.calculate_progressive_entry_sizes(
            position.entry_price, position.stop_loss, wave_metrics
        )

        # Determine which stage we're in
        for stage in stages:
            stage_num = stage['stage']

            # Check if this stage should trigger
            if stage_num == 2 and wave_progress >= 0.3:
                return {
                    'action': 'add',
                    'shares': stage['shares'],
                    'reason': stage['condition'],
                    'price': current_price
                }
            elif stage_num == 3 and wave_progress >= 0.6:
                return {
                    'action': 'add',
                    'shares': stage['shares'],
                    'reason': stage['condition'],
                    'price': current_price
                }

        return None

    def manage_wave_4_drawdown(self, symbol: str, current_price: float,
                              wave_metrics: WavePositionMetrics,
                              wave_4_retracement: float) -> Dict[str, Any]:
        """
        Manage position during Wave 4 correction.

        Strategy:
        - Tighten stops to protect Wave 3 profits
        - Consider partial profit-taking if deep retracement
        - Prepare for potential Wave 5 reentry

        Args:
            symbol: Trading symbol
            current_price: Current price
            wave_metrics: Wave 4 metrics
            wave_4_retracement: Retracement depth (0-1)

        Returns:
            Action dictionary
        """
        if symbol not in self.positions:
            return {'action': 'none', 'reason': 'No position'}

        position = self.positions[symbol]
        action = {'action': 'none', 'adjustments': []}

        # Deep Wave 4 (>38.2% retracement)
        if wave_4_retracement > 0.382:
            # Take 50% profits
            action['action'] = 'reduce'
            action['reduce_pct'] = 0.50
            action['reason'] = 'Deep Wave 4 correction - protect profits'

            # Tighten stop to breakeven
            action['adjust_stop'] = position.entry_price
            action['adjustments'].append('Stop moved to breakeven')

        # Normal Wave 4 (23.6-38.2%)
        elif wave_4_retracement > 0.236:
            # Trail stop below Wave 3 high
            if hasattr(position, 'wave_3_high'):
                action['adjust_stop'] = position.wave_3_high * 0.98
                action['adjustments'].append('Stop trailed to Wave 3 high')
            action['reason'] = 'Normal Wave 4 - trail stop'

        # Shallow Wave 4 (<23.6%)
        else:
            # Keep original stop
            action['reason'] = 'Shallow Wave 4 - maintain position'

        return action

    def determine_wave_5_exit_strategy(self, symbol: str, current_price: float,
                                      wave_5_progress: float,
                                      wave_metrics: WavePositionMetrics) -> Dict[str, Any]:
        """
        Determine exit strategy for Wave 5.

        Strategy:
        - Scale out at Fibonacci targets
        - Full exit on completion signals
        - Trail stops aggressively

        Args:
            symbol: Trading symbol
            current_price: Current price
            wave_5_progress: Wave 5 completion (0-1)
            wave_metrics: Wave 5 metrics

        Returns:
            Exit strategy dictionary
        """
        if symbol not in self.positions:
            return {'action': 'none'}

        strategy = {
            'progressive_exits': [],
            'stop_strategy': 'trailing',
            'full_exit_signals': []
        }

        # Define exit stages
        if wave_5_progress >= 0.3:  # 30% complete
            strategy['progressive_exits'].append({
                'exit_pct': 0.33,
                'reason': 'Wave 5 initial target (0.618x Wave 1)',
                'triggered': False
            })

        if wave_5_progress >= 0.6:  # 60% complete
            strategy['progressive_exits'].append({
                'exit_pct': 0.33,
                'reason': 'Wave 5 main target (1.0x Wave 1)',
                'triggered': False
            })

        if wave_5_progress >= 0.9:  # 90% complete
            strategy['progressive_exits'].append({
                'exit_pct': 0.34,  # Remaining
                'reason': 'Wave 5 completion approaching',
                'triggered': False
            })

        # Full exit signals
        if wave_metrics.confidence < 0.4:
            strategy['full_exit_signals'].append('Confidence deteriorated')

        return strategy


if __name__ == "__main__":
    # Example usage
    logger.info("Wave-Aware Position Sizing - Example\n")

    # Initialize sizer
    sizer = WaveAwarePositionSizer(
        portfolio_value=100000,
        max_risk_per_trade=0.02
    )

    # Test Wave 3 sizing
    logger.info("=" * 60)
    logger.info("WAVE 3 POSITION SIZING")
    logger.info("=" * 60)

    wave_3_metrics = WavePositionMetrics(
        wave_number=3,
        wave_type='impulse',
        confidence=0.80,
        is_diagonal=False,
        complex_correction=False,
        timeframe_alignment=0.85,
        fibonacci_confluence=4
    )

    entry = 150.0
    stop = 145.0

    # Normal Wave 3 size
    shares = sizer.calculate_wave_adjusted_size(entry, stop, wave_3_metrics)
    position_value = shares * entry
    risk_pct = (abs(entry - stop) * shares) / sizer.portfolio_value * 100

    logger.info(f"Entry: ${entry:.2f}")
    logger.info(f"Stop: ${stop:.2f}")
    logger.info(f"Wave: {wave_3_metrics.wave_number}")
    logger.info(f"Confidence: {wave_3_metrics.confidence:.1%}")
    logger.info(f"Timeframe Alignment: {wave_3_metrics.timeframe_alignment:.1%}")
    logger.info(f"Fib Confluence: {wave_3_metrics.fibonacci_confluence} levels")
    logger.info(f"\nPosition Size: {shares} shares")
    logger.info(f"Position Value: ${position_value:,.2f} ({position_value/sizer.portfolio_value:.1%} of portfolio)")
    logger.info(f"Risk Amount: ${abs(entry - stop) * shares:,.2f} ({risk_pct:.2f}% of portfolio)")

    # Aggressive Wave 3 size
    logger.info(f"\n--- AGGRESSIVE SIZING (high conviction) ---")
    aggressive_shares = sizer.calculate_wave_3_aggressive_size(entry, stop, wave_3_metrics)
    aggressive_value = aggressive_shares * entry
    aggressive_risk_pct = (abs(entry - stop) * aggressive_shares) / sizer.portfolio_value * 100

    logger.info(f"Aggressive Size: {aggressive_shares} shares")
    logger.info(f"Position Value: ${aggressive_value:,.2f} ({aggressive_value/sizer.portfolio_value:.1%} of portfolio)")
    logger.info(f"Risk Amount: ${abs(entry - stop) * aggressive_shares:,.2f} ({aggressive_risk_pct:.2f}% of portfolio)")

    # Progressive entries
    logger.info(f"\n--- PROGRESSIVE ENTRY PLAN ---")
    stages = sizer.calculate_progressive_entry_sizes(entry, stop, wave_3_metrics)

    for stage in stages:
        logger.info(f"\nStage {stage['stage']}: {stage['allocation_pct']:.0%} of position")
        logger.info(f"  Shares: {stage['shares']}")
        logger.info(f"  Condition: {stage['condition']}")
        logger.info(f"  Target Price: ${stage['price_target']:.2f}")
        logger.info(f"  Cumulative: {stage['cumulative_shares']} shares")

    # Test Wave 5 sizing (for comparison)
    logger.info(f"\n{'=' * 60}")
    logger.info("WAVE 5 POSITION SIZING (for comparison)")
    logger.info("=" * 60)

    wave_5_metrics = WavePositionMetrics(
        wave_number=5,
        wave_type='impulse',
        confidence=0.70,
        is_diagonal=False,
        complex_correction=False,
        timeframe_alignment=0.75,
        fibonacci_confluence=3
    )

    wave_5_shares = sizer.calculate_wave_adjusted_size(entry, stop, wave_5_metrics)
    wave_5_value = wave_5_shares * entry

    logger.info(f"Wave 5 Size: {wave_5_shares} shares")
    logger.info(f"Position Value: ${wave_5_value:,.2f} ({wave_5_value/sizer.portfolio_value:.1%} of portfolio)")
    logger.info(f"\nNote: Wave 5 size is {wave_5_shares/shares:.1%} of Wave 3 size")
    logger.info("      (Wave 5 has higher exhaustion risk)")
