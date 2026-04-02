"""
Risk Management Utilities for Investment System
Standalone modules for position sizing, stop loss calculation, and portfolio risk management.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# Avoid accidental shadowing of the standard library `logging` module
# when this file is executed as a script (`python src/utils/risk_management.py`).
# In that case, the script directory (`src/utils`) is added to `sys.path[0]`,
# which can make a local `logging.py` (in `src/utils`) visible as the top-level
# `logging` module. Temporarily remove the script directory from `sys.path`
# while importing the stdlib `logging` module.
_script_dir = os.path.dirname(__file__)
removed_from_path = False
if _script_dir in sys.path:
    try:
        sys.path.remove(_script_dir)
        removed_from_path = True
    except ValueError:
        removed_from_path = False

import logging

# Restore path so other imports behave as before
if removed_from_path:
    sys.path.insert(0, _script_dir)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    profit_target: Optional[float] = None
    wave_number: Optional[int] = None
    confidence: float = 0.5
    strategy: str = "unknown"

    @property
    def current_value(self) -> float:
        """Calculate current position value."""
        return self.shares * self.entry_price

    @property
    def risk_amount(self) -> float:
        """Calculate risk amount (distance to stop loss)."""
        return abs(self.entry_price - self.stop_loss) * self.shares

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if profit target exists."""
        if self.profit_target:
            reward = abs(self.profit_target - self.entry_price) * self.shares
            risk = self.risk_amount
            return reward / risk if risk > 0 else None
        return None


class PositionSizer:
    """
    Calculate position sizes based on various risk management methods.
    """

    def __init__(self, portfolio_value: float, max_risk_per_trade: float = 0.02):
        """
        Initialize position sizer.

        Args:
            portfolio_value: Total portfolio value
            max_risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 = 2%)
        """
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade

    def risk_based_size(self, entry_price: float, stop_loss: float,
                       confidence: float = 1.0) -> int:
        """
        Calculate position size based on fixed risk per trade.

        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            confidence: Trade confidence (0-1), adjusts position size

        Returns:
            Number of shares to buy
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            logger.warning("Invalid risk per share (<=0), using minimum position")
            return 1

        # Calculate base position size for full risk
        max_risk_amount = self.portfolio_value * self.max_risk_per_trade
        base_shares = max_risk_amount / risk_per_share

        # Adjust by confidence (0.5 confidence = 50% of full size)
        adjusted_shares = base_shares * confidence

        return max(int(adjusted_shares), 1)

    def fixed_percentage_size(self, entry_price: float, allocation_pct: float = 0.10) -> int:
        """
        Calculate position size as fixed percentage of portfolio.

        Args:
            entry_price: Entry price
            allocation_pct: Portfolio percentage to allocate (e.g., 0.10 = 10%)

        Returns:
            Number of shares to buy
        """
        position_value = self.portfolio_value * allocation_pct
        shares = position_value / entry_price
        return max(int(shares), 1)

    def volatility_based_size(self, entry_price: float, stop_loss: float,
                             atr: float, atr_multiplier: float = 2.0) -> int:
        """
        Calculate position size based on volatility (ATR).

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            atr: Average True Range value
            atr_multiplier: Multiplier for ATR-based stop (default 2.0)

        Returns:
            Number of shares to buy
        """
        # Use wider of: fixed stop or ATR-based stop
        fixed_risk = abs(entry_price - stop_loss)
        atr_risk = atr * atr_multiplier

        effective_risk = max(fixed_risk, atr_risk)

        max_risk_amount = self.portfolio_value * self.max_risk_per_trade
        shares = max_risk_amount / effective_risk

        return max(int(shares), 1)

    def kelly_criterion_size(self, win_rate: float, avg_win: float,
                            avg_loss: float, entry_price: float,
                            kelly_fraction: float = 0.25) -> int:
        """
        Calculate position size using Kelly Criterion.

        WARNING: Use fractional Kelly (0.25 or lower) to avoid over-leveraging.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade percentage
            avg_loss: Average losing trade percentage (positive number)
            entry_price: Entry price
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Number of shares to buy
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            logger.warning("Invalid Kelly parameters, using default sizing")
            return self.fixed_percentage_size(entry_price, 0.05)

        # Kelly formula: f = (p * b - q) / b
        # where p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss
        b = avg_win / avg_loss
        q = 1 - win_rate

        kelly_pct = (win_rate * b - q) / b

        # Apply fractional Kelly for safety
        kelly_pct = max(0, min(kelly_pct * kelly_fraction, 0.20))  # Cap at 20%

        position_value = self.portfolio_value * kelly_pct
        shares = position_value / entry_price

        return max(int(shares), 1)

    def calculate_optimal_size(self, entry_price: float, stop_loss: float,
                              confidence: float, method: str = "risk_based",
                              **kwargs) -> int:
        """
        Calculate optimal position size using specified method.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Trade confidence (0-1)
            method: Sizing method ('risk_based', 'fixed_pct', 'volatility', 'kelly')
            **kwargs: Additional parameters for specific methods

        Returns:
            Number of shares to buy
        """
        if method == "risk_based":
            return self.risk_based_size(entry_price, stop_loss, confidence)

        elif method == "fixed_pct":
            allocation = kwargs.get('allocation_pct', 0.10)
            return self.fixed_percentage_size(entry_price, allocation)

        elif method == "volatility":
            atr = kwargs.get('atr', abs(entry_price - stop_loss))
            multiplier = kwargs.get('atr_multiplier', 2.0)
            return self.volatility_based_size(entry_price, stop_loss, atr, multiplier)

        elif method == "kelly":
            win_rate = kwargs.get('win_rate', 0.6)
            avg_win = kwargs.get('avg_win', 0.05)
            avg_loss = kwargs.get('avg_loss', 0.02)
            fraction = kwargs.get('kelly_fraction', 0.25)
            return self.kelly_criterion_size(win_rate, avg_win, avg_loss,
                                            entry_price, fraction)

        else:
            logger.warning(f"Unknown sizing method: {method}, using risk_based")
            return self.risk_based_size(entry_price, stop_loss, confidence)


class StopLossCalculator:
    """
    Calculate stop loss levels using various methods.
    """

    @staticmethod
    def percentage_stop(entry_price: float, stop_pct: float,
                       direction: str = "long") -> float:
        """
        Calculate stop loss as percentage from entry.

        Args:
            entry_price: Entry price
            stop_pct: Stop loss percentage (e.g., 0.02 = 2%)
            direction: 'long' or 'short'

        Returns:
            Stop loss price
        """
        if direction == "long":
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)

    @staticmethod
    def atr_stop(entry_price: float, atr: float, multiplier: float = 2.0,
                direction: str = "long") -> float:
        """
        Calculate stop loss based on Average True Range.

        Args:
            entry_price: Entry price
            atr: Average True Range value
            multiplier: ATR multiplier (typical 2-3)
            direction: 'long' or 'short'

        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier

        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    @staticmethod
    def wave_structure_stop(entry_price: float, wave_low: float,
                           buffer_pct: float = 0.02) -> float:
        """
        Calculate stop loss based on Elliott Wave structure.

        Args:
            entry_price: Entry price
            wave_low: Key wave level (e.g., Wave 2 low for Wave 3 entry)
            buffer_pct: Buffer below wave level (e.g., 0.02 = 2%)

        Returns:
            Stop loss price
        """
        return wave_low * (1 - buffer_pct)

    @staticmethod
    def fibonacci_stop(impulse_high: float, impulse_low: float,
                      fib_level: float = 0.786) -> float:
        """
        Calculate stop loss at Fibonacci retracement level.

        Args:
            impulse_high: High of impulse wave
            impulse_low: Low of impulse wave
            fib_level: Fibonacci level for stop (e.g., 0.786 = 78.6%)

        Returns:
            Stop loss price
        """
        range_size = impulse_high - impulse_low
        return impulse_high - (range_size * fib_level)

    @staticmethod
    def trailing_stop(current_price: float, highest_price: float,
                     trailing_pct: float = 0.10) -> float:
        """
        Calculate trailing stop loss.

        Args:
            current_price: Current price
            highest_price: Highest price since entry
            trailing_pct: Trailing percentage (e.g., 0.10 = 10%)

        Returns:
            Trailing stop price
        """
        return highest_price * (1 - trailing_pct)

    @staticmethod
    def volatility_adjusted_stop(entry_price: float, base_stop_pct: float,
                                volatility_ratio: float) -> float:
        """
        Adjust stop loss based on current volatility.

        Args:
            entry_price: Entry price
            base_stop_pct: Base stop percentage in normal volatility
            volatility_ratio: Current volatility / average volatility

        Returns:
            Adjusted stop loss price
        """
        # Wider stops in high volatility, tighter in low volatility
        adjusted_stop_pct = base_stop_pct * volatility_ratio
        adjusted_stop_pct = max(0.01, min(adjusted_stop_pct, 0.05))  # Clamp 1-5%

        return entry_price * (1 - adjusted_stop_pct)


class PortfolioRiskManager:
    """
    Manage overall portfolio risk and exposure.
    """

    def __init__(self, initial_capital: float, config: Dict):
        """
        Initialize portfolio risk manager.

        Args:
            initial_capital: Starting capital
            config: Risk management configuration
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.config = config

        # Extract config values
        self.max_portfolio_exposure = config.get('max_portfolio_exposure', 0.40)
        self.max_daily_dd = config.get('max_daily_drawdown', 0.05)
        self.max_total_dd = config.get('max_total_drawdown', 0.20)
        self.max_concurrent_positions = config.get('max_concurrent_positions', 5)
        self.max_correlation = config.get('max_correlation', 0.70)

        # Tracking
        self.positions: List[Position] = []
        self.daily_start_capital = initial_capital
        self.trade_history: List[Dict] = []

    def can_open_position(self, position_value: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened within risk limits.

        Args:
            position_value: Value of proposed position

        Returns:
            Tuple of (can_open, reason)
        """
        # Check position count
        if len(self.positions) >= self.max_concurrent_positions:
            return False, f"Max concurrent positions reached ({self.max_concurrent_positions})"

        # Check portfolio exposure
        current_exposure = sum(p.current_value for p in self.positions)
        total_exposure = current_exposure + position_value
        exposure_pct = total_exposure / self.current_capital

        if exposure_pct > self.max_portfolio_exposure:
            return False, f"Would exceed max exposure ({exposure_pct:.1%} > {self.max_portfolio_exposure:.1%})"

        # Check daily drawdown
        daily_dd = (self.daily_start_capital - self.current_capital) / self.daily_start_capital

        if daily_dd >= self.max_daily_dd:
            return False, f"Daily drawdown limit reached ({daily_dd:.1%})"

        # Check total drawdown
        total_dd = self.get_drawdown()

        if total_dd >= self.max_total_dd:
            return False, f"Total drawdown limit reached ({total_dd:.1%})"

        return True, "OK"

    def add_position(self, position: Position):
        """Add a position to the portfolio."""
        self.positions.append(position)
        logger.info(f"Added position: {position.symbol} - {position.shares} shares @ ${position.entry_price:.2f}")

    def remove_position(self, symbol: str, exit_price: float, exit_date: datetime) -> Optional[Dict]:
        """
        Remove a position and record the trade.

        Args:
            symbol: Symbol to remove
            exit_price: Exit price
            exit_date: Exit date

        Returns:
            Trade result dictionary
        """
        position = next((p for p in self.positions if p.symbol == symbol), None)

        if not position:
            logger.warning(f"Position not found: {symbol}")
            return None

        # Calculate P&L
        profit = (exit_price - position.entry_price) * position.shares
        profit_pct = (exit_price - position.entry_price) / position.entry_price * 100
        r_multiple = profit / position.risk_amount if position.risk_amount > 0 else 0

        trade = {
            'symbol': symbol,
            'entry_date': position.entry_date,
            'exit_date': exit_date,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'shares': position.shares,
            'profit': profit,
            'profit_pct': profit_pct,
            'r_multiple': r_multiple,
            'strategy': position.strategy,
            'wave_number': position.wave_number,
            'confidence': position.confidence
        }

        self.trade_history.append(trade)
        self.positions.remove(position)

        # Update capital
        self.current_capital += profit

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        logger.info(f"Closed position: {symbol} - P/L: ${profit:.2f} ({profit_pct:.2f}%)")

        return trade

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def get_exposure(self) -> float:
        """Get current portfolio exposure percentage."""
        if self.current_capital <= 0:
            return 0
        total_exposure = sum(p.current_value for p in self.positions)
        return total_exposure / self.current_capital

    def reset_daily(self):
        """Reset daily tracking (call at start of each day)."""
        self.daily_start_capital = self.current_capital
        logger.debug(f"Daily reset - Starting capital: ${self.current_capital:,.2f}")

    def get_statistics(self) -> Dict:
        """Get portfolio statistics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_return_pct': 0,
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'current_drawdown': self.get_drawdown(),
                'current_exposure': 0.0,
                'open_positions': len(self.positions),
                'avg_win': 0,
                'avg_loss': 0,
                'avg_r_multiple': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        df = pd.DataFrame(self.trade_history)
        winning_trades = df[df['profit'] > 0]
        losing_trades = df[df['profit'] <= 0]

        return {
            'total_trades': len(df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(df) if len(df) > 0 else 0,
            'total_profit': df['profit'].sum(),
            'total_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': self.get_drawdown(),
            'current_exposure': self.get_exposure(),
            'open_positions': len(self.positions),
            'avg_win': winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['profit'].mean() if len(losing_trades) > 0 else 0,
            'avg_r_multiple': df['r_multiple'].mean(),
            'best_trade': df['profit'].max(),
            'worst_trade': df['profit'].min()
        }

    def check_correlation(self, new_symbol: str, price_data: Dict[str, pd.Series]) -> float:
        """
        Check correlation between new symbol and existing positions.

        Args:
            new_symbol: Symbol to check
            price_data: Dict of symbol -> price series

        Returns:
            Maximum correlation found
        """
        if new_symbol not in price_data:
            logger.warning(f"No price data for {new_symbol}")
            return 0

        new_returns = price_data[new_symbol].pct_change().dropna()
        max_corr = 0

        for position in self.positions:
            if position.symbol in price_data:
                pos_returns = price_data[position.symbol].pct_change().dropna()

                # Align series
                common_index = new_returns.index.intersection(pos_returns.index)
                if len(common_index) > 20:  # Need minimum data
                    corr = new_returns[common_index].corr(pos_returns[common_index])
                    max_corr = max(max_corr, abs(corr))

        return max_corr

    def should_reduce_size_correlation(self, new_symbol: str,
                                      price_data: Dict[str, pd.Series]) -> Tuple[bool, float]:
        """
        Check if position size should be reduced due to correlation.

        Args:
            new_symbol: Symbol to check
            price_data: Price data for correlation calculation

        Returns:
            Tuple of (should_reduce, reduction_factor)
        """
        max_corr = self.check_correlation(new_symbol, price_data)

        if max_corr > self.max_correlation:
            reduction = self.config.get('correlation_reduction_factor', 0.30)
            return True, reduction

        return False, 1.0


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses only downside deviation).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0

    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Series of equity values over time

    Returns:
        Tuple of (max_drawdown_pct, start_date, end_date)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    max_dd = drawdown.min()
    end_date = drawdown.idxmin()

    # Find start of drawdown (last peak before max drawdown)
    start_date = equity_curve[:end_date][equity_curve[:end_date] == running_max[end_date]].index[-1]

    return abs(max_dd), start_date, end_date


if __name__ == "__main__":
    # Example usage
    logger.info("Risk Management Utilities - Example Usage\n")

    # Position sizing example
    sizer = PositionSizer(portfolio_value=100000, max_risk_per_trade=0.02)

    entry = 50.0
    stop = 48.0
    confidence = 0.8

    shares = sizer.risk_based_size(entry, stop, confidence)
    logger.info(f"Risk-based position size: {shares} shares")
    logger.info(f"Position value: ${shares * entry:,.2f}")
    logger.info(f"Risk amount: ${abs(entry - stop) * shares:,.2f}\n")

    # Stop loss calculation example
    calc = StopLossCalculator()

    pct_stop = calc.percentage_stop(entry, 0.02)
    logger.info(f"2% stop loss: ${pct_stop:.2f}")

    atr_stop = calc.atr_stop(entry, atr=1.5, multiplier=2.0)
    logger.info(f"ATR stop (2x): ${atr_stop:.2f}")

    wave_stop = calc.wave_structure_stop(entry, wave_low=45.0, buffer_pct=0.02)
    logger.info(f"Wave structure stop: ${wave_stop:.2f}\n")

    # Portfolio risk management example
    config = {
        'max_portfolio_exposure': 0.40,
        'max_daily_drawdown': 0.05,
        'max_total_drawdown': 0.20,
        'max_concurrent_positions': 5
    }

    portfolio = PortfolioRiskManager(initial_capital=100000, config=config)

    position = Position(
        symbol="AAPL",
        entry_date=datetime.now(),
        entry_price=150.0,
        shares=100,
        stop_loss=147.0,
        strategy="multiframe_alignment"
    )

    can_open, reason = portfolio.can_open_position(position.current_value)
    logger.info(f"Can open position: {can_open} - {reason}")

    if can_open:
        portfolio.add_position(position)
        stats = portfolio.get_statistics()
        logger.info(f"Portfolio exposure: {stats['current_exposure']:.1%}")
        logger.info(f"Open positions: {stats['open_positions']}")
