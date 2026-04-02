"""
Trade Setup Generator
Generates specific, actionable trade setups from Elliott Wave analysis.

Transforms pattern recognition into concrete entry/exit rules with:
- Specific entry prices and conditions
- Wave-structure stop losses
- Multiple profit targets
- Risk/reward calculations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SetupType(Enum):
    """Types of trade setups."""
    WAVE_3_ENTRY = "wave_3_entry"
    WAVE_5_ENTRY = "wave_5_entry"
    WAVE_5_EXIT = "wave_5_exit"
    CORRECTION_ENTRY = "correction_entry"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class TradeSetup:
    """
    Represents a specific trading opportunity.

    This is the bridge between Elliott Wave analysis and actual trading:
    it provides concrete, actionable information for trade execution.
    """
    setup_type: SetupType
    entry_price: float
    entry_conditions: List[str]  # Specific triggers
    stop_loss: float
    targets: List[float]  # Multiple profit targets
    confidence: float
    risk_reward_ratio: float
    wave_context: Dict[str, Any]
    setup_date: datetime
    signal_strength: SignalStrength

    # Optional fields
    expiration_date: Optional[datetime] = None
    position_size_multiplier: float = 1.0  # Adjust based on setup quality
    notes: str = ""

    def __post_init__(self):
        """Validate setup data."""
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        if not self.targets:
            raise ValueError("Must have at least one target")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    def get_risk_amount(self, shares: int) -> float:
        """Calculate total risk amount."""
        return abs(self.entry_price - self.stop_loss) * shares

    def get_potential_reward(self, shares: int, target_level: int = 0) -> float:
        """Calculate potential reward for specific target."""
        if 0 <= target_level < len(self.targets):
            return abs(self.targets[target_level] - self.entry_price) * shares
        return 0

    def is_valid(self, current_date: datetime, current_price: float) -> bool:
        """Check if setup is still valid."""
        # Check expiration
        if self.expiration_date and current_date > self.expiration_date:
            return False

        # Check if stop loss hit before entry
        if current_price <= self.stop_loss:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'setup_type': self.setup_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'targets': self.targets,
            'confidence': self.confidence,
            'risk_reward': self.risk_reward_ratio,
            'signal_strength': self.signal_strength.value,
            'setup_date': self.setup_date.isoformat(),
            'entry_conditions': self.entry_conditions,
            'notes': self.notes
        }


class TradeSetupGenerator:
    """
    Generate specific trade setups from Elliott Wave analysis.

    Key Setups:
    1. Wave 3 Entry - Highest probability (after Wave 2 completion)
    2. Wave 5 Entry - Medium probability (after Wave 4 completion)
    3. Wave 5 Exit - Reversal/profit taking
    4. Correction Entry - Mean reversion at Fibonacci levels
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize generator.

        Args:
            config: Configuration dict with risk parameters
        """
        self.config = config or {}
        self.default_stop_buffer = self.config.get('stop_buffer_pct', 0.02)
        self.min_risk_reward = self.config.get('min_risk_reward', 2.0)

    def identify_wave_3_entry(self, df: pd.DataFrame, wave_data: Dict[str, Any],
                             current_price: float) -> Optional[TradeSetup]:
        """
        Wave 3 entry setup (HIGHEST PROBABILITY).

        Entry Criteria:
        - Wave 2 has completed
        - Retraced 50-78.6% of Wave 1
        - Volume declining in Wave 2 (exhaustion)
        - Momentum divergence present (bullish)

        Stop: Below Wave 1 low
        Targets: 1.618x, 2.618x, 4.236x Wave 1 length

        Args:
            df: Price dataframe
            wave_data: Elliott Wave analysis
            current_price: Current price

        Returns:
            TradeSetup if conditions met, None otherwise
        """
        current_wave = wave_data.get('current_wave', 0)

        # Must be at Wave 2 completion or early Wave 3
        if current_wave not in [2, 3]:
            return None

        impulse_wave = wave_data.get('impulse_wave', np.array([]))
        if len(impulse_wave) < 3:
            return None

        # Get wave points
        wave_1_start_idx = impulse_wave[0]
        wave_1_end_idx = impulse_wave[1]
        wave_2_end_idx = impulse_wave[2]

        # Validate indices
        if wave_2_end_idx >= len(df):
            return None

        # Get prices
        wave_1_start = df['close'].iloc[wave_1_start_idx]
        wave_1_end = df['close'].iloc[wave_1_end_idx]
        wave_2_end = df['close'].iloc[wave_2_end_idx]

        # Calculate Wave 1 characteristics
        wave_1_range = abs(wave_1_end - wave_1_start)
        wave_1_low = min(wave_1_start, wave_1_end)

        # Check Wave 2 completion signals
        completion_signals = []
        signal_count = 0

        # 1. Fibonacci retracement (50-78.6%)
        retracement_pct = abs(wave_1_end - wave_2_end) / wave_1_range if wave_1_range > 0 else 0

        if 0.50 <= retracement_pct <= 0.786:
            completion_signals.append(f"Wave 2 retraced {retracement_pct:.1%} (ideal range)")
            signal_count += 2
        elif 0.382 <= retracement_pct < 0.50:
            completion_signals.append(f"Wave 2 retraced {retracement_pct:.1%} (acceptable)")
            signal_count += 1
        else:
            # Retracement outside ideal range
            return None

        # 2. Volume pattern (if available)
        if 'volume' in df.columns:
            wave_1_avg_vol = df['volume'].iloc[wave_1_start_idx:wave_1_end_idx].mean()
            wave_2_avg_vol = df['volume'].iloc[wave_1_end_idx:wave_2_end_idx].mean()

            if wave_2_avg_vol < wave_1_avg_vol * 0.8:  # 20% lower volume
                completion_signals.append("Volume declining in Wave 2")
                signal_count += 2

        # 3. Check for momentum divergence (simplified)
        if wave_2_end_idx >= 14:  # Need history for RSI
            try:
                # Simple momentum check: price makes new low but momentum improving
                recent_lows = df['close'].iloc[wave_2_end_idx-10:wave_2_end_idx].min()
                if wave_2_end > recent_lows * 1.01:  # Bouncing from low
                    completion_signals.append("Price showing reversal signs")
                    signal_count += 1
            except Exception:
                pass

        # Require at least 2 signals
        if signal_count < 2:
            logger.debug(f"Wave 3 setup: insufficient signals ({signal_count})")
            return None

        # Calculate entry, stops, and targets
        entry = current_price

        # Stop below Wave 1 low with buffer
        stop = wave_1_low * (1 - self.default_stop_buffer)

        # Targets based on Fibonacci extensions
        targets = [
            wave_2_end + (wave_1_range * 1.618),  # Minimum Wave 3
            wave_2_end + (wave_1_range * 2.618),  # Extended Wave 3
            wave_2_end + (wave_1_range * 4.236),  # Strong Wave 3
        ]

        # Calculate risk/reward
        risk = abs(entry - stop)
        reward = abs(targets[0] - entry)
        risk_reward = reward / risk if risk > 0 else 0

        # Reject if risk/reward too poor
        if risk_reward < self.min_risk_reward:
            logger.debug(f"Wave 3 setup: poor R:R {risk_reward:.2f}")
            return None

        # Determine signal strength
        if signal_count >= 4:
            strength = SignalStrength.STRONG
            position_multiplier = 1.0
        elif signal_count >= 3:
            strength = SignalStrength.MODERATE
            position_multiplier = 0.75
        else:
            strength = SignalStrength.WEAK
            position_multiplier = 0.5

        # Calculate confidence
        base_confidence = wave_data.get('confidence', 0.5)
        signal_boost = min(signal_count * 0.05, 0.2)
        confidence = min(base_confidence + signal_boost, 0.95)

        return TradeSetup(
            setup_type=SetupType.WAVE_3_ENTRY,
            entry_price=entry,
            entry_conditions=completion_signals,
            stop_loss=stop,
            targets=targets,
            confidence=confidence,
            risk_reward_ratio=risk_reward,
            wave_context=wave_data,
            setup_date=datetime.now(),
            signal_strength=strength,
            position_size_multiplier=position_multiplier,
            notes=f"Wave 3 entry after {retracement_pct:.1%} Wave 2 retracement"
        )

    def identify_wave_5_exit(self, df: pd.DataFrame, wave_data: Dict[str, Any],
                            current_price: float) -> Optional[Dict[str, Any]]:
        """
        Wave 5 completion/exit setup.

        Exit Signals:
        - Wave 5 at or beyond Fibonacci target
        - Declining volume (exhaustion)
        - Momentum divergence (bearish)
        - Trendline break
        - Wave 5 failure to extend

        Args:
            df: Price dataframe
            wave_data: Wave analysis
            current_price: Current price

        Returns:
            Exit signal dictionary if conditions met
        """
        current_wave = wave_data.get('current_wave', 0)

        if current_wave != 5:
            return None

        impulse_wave = wave_data.get('impulse_wave', np.array([]))
        if len(impulse_wave) < 5:
            return None

        completion_signals = []
        signal_score = 0

        # 1. Check Fibonacci target achievement
        wave_4_end_idx = impulse_wave[4]
        if wave_4_end_idx >= len(df):
            return None

        wave_4_end = df['close'].iloc[wave_4_end_idx]

        # Expected Wave 5 targets
        wave_1_length = abs(df['close'].iloc[impulse_wave[1]] -
                          df['close'].iloc[impulse_wave[0]])

        expected_targets = [
            wave_4_end + wave_1_length * 0.618,  # Wave 5 = 0.618 x Wave 1
            wave_4_end + wave_1_length * 1.000,  # Wave 5 = Wave 1
        ]

        # Check if we've reached targets
        for i, target in enumerate(expected_targets):
            if current_price >= target * 0.98:  # Within 2%
                completion_signals.append(f"Reached Fibonacci target {i+1} (${target:.2f})")
                signal_score += 2
                break

        # 2. Volume divergence
        if 'volume' in df.columns and len(impulse_wave) >= 5:
            wave_3_start_idx = impulse_wave[2]
            wave_3_end_idx = impulse_wave[3]
            wave_5_start_idx = impulse_wave[4]
            current_idx = len(df) - 1

            wave_3_avg_vol = df['volume'].iloc[wave_3_start_idx:wave_3_end_idx].mean()
            wave_5_current_vol = df['volume'].iloc[wave_5_start_idx:current_idx].mean()

            if wave_5_current_vol < wave_3_avg_vol * 0.7:  # 30% lower
                completion_signals.append("Declining volume in Wave 5")
                signal_score += 2

        # 3. Wave 5 extension check
        wave_5_current_length = abs(current_price - wave_4_end)
        if wave_5_current_length < wave_1_length * 0.382:
            completion_signals.append("Wave 5 failing to extend (weak)")
            signal_score += 1

        # 4. Time-based exhaustion
        wave_5_duration = len(df) - wave_5_start_idx
        avg_wave_duration = np.mean([
            impulse_wave[1] - impulse_wave[0],  # Wave 1
            impulse_wave[3] - impulse_wave[2],  # Wave 3
        ])

        if wave_5_duration > avg_wave_duration * 1.5:
            completion_signals.append("Wave 5 extended in time")
            signal_score += 1

        # Require at least 3 signals for exit
        if signal_score < 3:
            return None

        # Determine action strength
        if signal_score >= 5:
            action = "STRONG EXIT"
            exit_percentage = 100
        elif signal_score >= 4:
            action = "EXIT 75%"
            exit_percentage = 75
        else:
            action = "REDUCE 50%"
            exit_percentage = 50

        return {
            'setup_type': 'wave_5_exit',
            'action': action,
            'exit_percentage': exit_percentage,
            'confidence': min(signal_score / 6, 1.0),
            'signals': completion_signals,
            'suggested_exit_price': current_price,
            'alternative_action': 'Trail stop to protect profits',
            'invalidation': f"New high above ${current_price * 1.05:.2f}",
            'wave_context': wave_data
        }

    def identify_correction_entry(self, df: pd.DataFrame, wave_data: Dict[str, Any],
                                  fib_clusters: List[Any]) -> Optional[TradeSetup]:
        """
        Entry during corrections (Wave 2 or Wave 4) at Fibonacci support.

        Entry Criteria:
        - Price at major Fibonacci cluster
        - Multiple Fibonacci methods converge
        - Volume declining
        - Reversal candlestick pattern (optional)

        Args:
            df: Price dataframe
            wave_data: Wave analysis
            fib_clusters: List of Fibonacci clusters from analyzer

        Returns:
            TradeSetup if conditions met
        """
        current_wave = wave_data.get('current_wave', 0)

        if current_wave not in [2, 4]:
            return None

        if not fib_clusters:
            return None

        current_price = df['close'].iloc[-1]

        # Find nearest support cluster
        support_clusters = [c for c in fib_clusters if c.central_price < current_price]

        if not support_clusters:
            return None

        nearest_support = support_clusters[-1]  # Closest below current price

        # Check if price is near the cluster
        distance_pct = abs(current_price - nearest_support.central_price) / current_price

        if distance_pct > 0.03:  # More than 3% away
            return None

        # Check cluster significance
        if nearest_support.significance < 3:
            return None

        # Entry conditions
        entry_conditions = [
            f"Price near {nearest_support.significance}-method Fibonacci cluster",
            f"Cluster at ${nearest_support.central_price:.2f}",
            f"Wave {current_wave} correction pattern"
        ]

        # Volume check
        signal_score = nearest_support.significance

        if 'volume' in df.columns:
            recent_vol = df['volume'].iloc[-5:].mean()
            avg_vol = df['volume'].iloc[-20:-5].mean()

            if recent_vol < avg_vol * 0.8:
                entry_conditions.append("Volume declining (exhaustion)")
                signal_score += 1

        # Entry, stop, and targets
        entry = nearest_support.central_price

        # Stop below cluster with buffer
        cluster_low = nearest_support.price_range[0]
        stop = cluster_low * (1 - self.default_stop_buffer)

        # Targets based on wave structure
        if 'impulse_wave' in wave_data:
            wave_points = wave_data['impulse_wave']

            if current_wave == 2 and len(wave_points) >= 2:
                wave_1_end = df['close'].iloc[wave_points[1]]
                targets = [
                    wave_1_end * 1.05,  # Exceed Wave 1 by 5%
                    wave_1_end * 1.15,  # Wave 3 target
                    wave_1_end * 1.30,  # Extended Wave 3
                ]
            elif current_wave == 4 and len(wave_points) >= 4:
                wave_3_end = df['close'].iloc[wave_points[3]]
                targets = [
                    wave_3_end * 1.05,  # Exceed Wave 3 by 5%
                    wave_3_end * 1.10,  # Wave 5 target
                ]
            else:
                targets = [entry * 1.10, entry * 1.20]
        else:
            targets = [entry * 1.10, entry * 1.20]

        # Risk/reward
        risk = abs(entry - stop)
        reward = abs(targets[0] - entry)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self.min_risk_reward:
            return None

        # Confidence
        cluster_confidence = nearest_support.confidence
        base_confidence = wave_data.get('confidence', 0.5)
        confidence = (cluster_confidence + base_confidence) / 2

        # Signal strength
        if signal_score >= 5:
            strength = SignalStrength.STRONG
        elif signal_score >= 4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return TradeSetup(
            setup_type=SetupType.CORRECTION_ENTRY,
            entry_price=entry,
            entry_conditions=entry_conditions,
            stop_loss=stop,
            targets=targets,
            confidence=confidence,
            risk_reward_ratio=risk_reward,
            wave_context=wave_data,
            setup_date=datetime.now(),
            signal_strength=strength,
            notes=f"Wave {current_wave} correction entry at Fibonacci cluster"
        )

    def scan_for_setups(self, df: pd.DataFrame, wave_data: Dict[str, Any],
                       fib_clusters: List[Any] = None) -> List[TradeSetup]:
        """
        Scan for all available trade setups.

        Args:
            df: Price dataframe
            wave_data: Elliott Wave analysis
            fib_clusters: Optional Fibonacci cluster analysis

        Returns:
            List of valid trade setups
        """
        setups = []
        current_price = df['close'].iloc[-1]

        # Check Wave 3 entry
        wave_3_setup = self.identify_wave_3_entry(df, wave_data, current_price)
        if wave_3_setup:
            setups.append(wave_3_setup)
            logger.info(f"Found Wave 3 setup: R:R={wave_3_setup.risk_reward_ratio:.2f}")

        # Check Wave 5 exit
        wave_5_exit = self.identify_wave_5_exit(df, wave_data, current_price)
        if wave_5_exit:
            # Convert to pseudo-setup for consistency
            logger.info(f"Found Wave 5 exit signal: {wave_5_exit['action']}")

        # Check correction entry
        if fib_clusters:
            correction_setup = self.identify_correction_entry(df, wave_data, fib_clusters)
            if correction_setup:
                setups.append(correction_setup)
                logger.info(f"Found correction entry setup")

        return setups


if __name__ == "__main__":
    # Example usage
    logger.info("Trade Setup Generator - Example\n")

    generator = TradeSetupGenerator({
        'stop_buffer_pct': 0.02,
        'min_risk_reward': 2.0
    })

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = np.linspace(100, 130, 100) + np.random.randn(100) * 2
    volumes = np.random.rand(100) * 1000000

    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Sample wave data
    wave_data = {
        'impulse_wave': np.array([0, 20, 15, 50, 45, 80]),
        'current_wave': 3,
        'confidence': 0.75
    }

    current_price = df['close'].iloc[-1]

    # Generate Wave 3 setup
    setup = generator.identify_wave_3_entry(df, wave_data, current_price)

    if setup:
        logger.info("=" * 60)
        logger.info("TRADE SETUP DETECTED")
        logger.info("=" * 60)
        logger.info(f"Type: {setup.setup_type.value}")
        logger.info(f"Signal Strength: {setup.signal_strength.value}")
        logger.info(f"Confidence: {setup.confidence:.1%}")
        logger.info(f"\nEntry: ${setup.entry_price:.2f}")
        logger.info(f"Stop Loss: ${setup.stop_loss:.2f}")
        logger.info(f"Risk: ${abs(setup.entry_price - setup.stop_loss):.2f}")
        logger.info(f"\nTargets:")
        for i, target in enumerate(setup.targets, 1):
            reward = abs(target - setup.entry_price)
            r_multiple = reward / abs(setup.entry_price - setup.stop_loss)
            logger.info(f"  {i}. ${target:.2f} ({r_multiple:.1f}R)")
        logger.info(f"\nOverall R:R: {setup.risk_reward_ratio:.2f}")
        logger.info(f"\nEntry Conditions:")
        for condition in setup.entry_conditions:
            logger.info(f"  ✓ {condition}")
        logger.info(f"\nNotes: {setup.notes}")
    else:
        logger.info("No valid setup found")
