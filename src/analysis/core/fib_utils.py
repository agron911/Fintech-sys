from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

@dataclass
class FibonacciRatios:
    """Standard Fibonacci ratios used in Elliott Wave analysis"""
    RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSIONS = [1.618, 2.618, 4.236]
    PROJECTIONS = [0.618, 1.0, 1.618]
    TIME_RATIOS = [0.618, 1.0, 1.618, 2.618]
    WAVE5_FROM_3 = [0.618, 1.0, 1.382, 1.618, 2.0, 2.618]  # New ratios for Wave 5 vs Wave 3 projections
    RATE_OF_CHANGE = [0.618, 1.0, 1.618, 2.0, 2.618]  # Ratios for Rate-of-Change projections
    GEOMETRIC_MEAN = [0.382, 0.5, 0.618, 0.786]  # Ratios for Geometric-Mean projections

def calculate_rate_of_change_projection(p1: float, p2: float, p3: float, ratio: float) -> float:
    """
    Calculate price target using Rate-of-Change projection method.
    
    Args:
        p1: Price at start of Wave 1
        p2: Price at end of Wave 2
        p3: Price at end of Wave 3
        ratio: Fibonacci ratio to use for projection
        
    Returns:
        Projected price target
    """
    if p2 == 0:  # Prevent division by zero
        return 0.0
        
    # Calculate percentage change from Wave 2 to Wave 3
    pe = (p3 - p2) / p2
    
    # Project forward using the ratio
    p_target = p1 * (1 + pe) ** ratio
    
    return p_target

def validate_rate_of_change_projection(waves: dict, tolerance: float = 0.0) -> Tuple[float, List[float]]:
    """
    Validate Wave 5 using Rate-of-Change projection method.
    
    Args:
        waves: Dictionary of wave prices
        tolerance: Optional tolerance for ratio matching
        
    Returns:
        Tuple of (confidence_score, list_of_projected_targets)
    """
    confidence = 0.0
    fib = FibonacciRatios()
    projected_targets = []
    
    try:
        if all(k in waves for k in [1, 2, 3, 5]):
            p1 = waves[1]
            p2 = waves[2]
            p3 = waves[3]
            p5_actual = waves[5]
            
            # Calculate projections for each ratio
            for ratio in fib.RATE_OF_CHANGE:
                p5_projected = calculate_rate_of_change_projection(p1, p2, p3, ratio)
                projected_targets.append(p5_projected)
                
                # Check if actual price is close to projection
                if abs(p5_projected - p5_actual) / p5_actual < 0.05 + tolerance:
                    confidence += 0.2
                    
            # Cap confidence contribution
            confidence = min(confidence, 0.4)
            
    except Exception:
        pass
        
    return confidence, projected_targets

def calculate_geometric_mean_projection(p_low: float, p_high: float, ratio: float) -> float:
    """
    Calculate price target using Geometric-Mean projection method.
    
    Args:
        p_low: Lower pivot price (e.g., Wave 2 trough)
        p_high: Higher pivot price (e.g., Wave 3 peak)
        ratio: Weight ratio (w) for the geometric mean
        
    Returns:
        Projected price target using geometric mean
    """
    if p_low <= 0 or p_high <= 0:  # Prevent invalid inputs for geometric mean
        return 0.0
        
    # Calculate geometric mean projection
    p_target = (p_low ** (1 - ratio)) * (p_high ** ratio)
    
    return p_target

def validate_geometric_mean_projection(waves: dict, tolerance: float = 0.0) -> Tuple[float, List[float]]:
    """
    Validate Wave 5 using Geometric-Mean projection method.
    
    Args:
        waves: Dictionary of wave prices
        tolerance: Optional tolerance for ratio matching
        
    Returns:
        Tuple of (confidence_score, list_of_projected_targets)
    """
    confidence = 0.0
    fib = FibonacciRatios()
    projected_targets = []
    
    try:
        if all(k in waves for k in [2, 3, 5]):
            p_low = waves[2]  # Wave 2 trough
            p_high = waves[3]  # Wave 3 peak
            p5_actual = waves[5]
            
            # Calculate projections for each ratio
            for ratio in fib.GEOMETRIC_MEAN:
                p5_projected = calculate_geometric_mean_projection(p_low, p_high, ratio)
                projected_targets.append(p5_projected)
                
                # Check if actual price is close to projection
                if abs(p5_projected - p5_actual) / p5_actual < 0.05 + tolerance:
                    confidence += 0.25
                    
            # Cap confidence contribution
            confidence = min(confidence, 0.4)
            
    except Exception:
        pass
        
    return confidence, projected_targets

def validate_fibonacci_relationships(waves: dict, dates, tolerance: float = 0.0) -> float:
    """
    Core Fibonacci checks (waves 2–5 + time) with an optional extra tolerance.
    """
    confidence = 0.0
    fib = FibonacciRatios()
    try:
        # Wave 2 retracement of Wave 1
        if waves[1] != 0:
            wave_2_ratio = abs(waves[2] / waves[1])
            if any(abs(wave_2_ratio - ratio) < 0.05 + tolerance for ratio in fib.RETRACEMENTS):
                confidence += 0.2
        # Wave 3 extension relative to Wave 1
        if waves[1] != 0:
            wave_3_ratio = abs(waves[3] / waves[1])
            if any(abs(wave_3_ratio - ratio) < 0.1 + tolerance for ratio in fib.EXTENSIONS):
                confidence += 0.25
            elif wave_3_ratio >= 1.618 - tolerance:  # At least 1.618 extension
                confidence += 0.15
        # Wave 4 retracement of Wave 3
        if waves[3] != 0:
            wave_4_ratio = abs(waves[4] / waves[3])
            if any(abs(wave_4_ratio - ratio) < 0.05 + tolerance for ratio in fib.RETRACEMENTS[:3]):
                confidence += 0.15

        # Wave 5 relationships - combined analysis
        wave5_confidence = 0.0
        
        # Wave 5 vs Wave 1
        if waves[5] != 0 and waves[1] != 0:
            wave_5_ratio = abs(waves[5] / waves[1])
            if any(abs(wave_5_ratio - ratio) < 0.1 + tolerance for ratio in fib.PROJECTIONS):
                wave5_confidence += 0.2

        # Wave 5 vs Wave 1+3 combination
        if waves[5] != 0:
            waves_1_3_total = abs(waves[1]) + abs(waves[3])
            if waves_1_3_total != 0:
                wave_5_total_ratio = abs(waves[5]) / waves_1_3_total
                if any(abs(wave_5_total_ratio - ratio) < 0.1 + tolerance for ratio in fib.PROJECTIONS):
                    wave5_confidence += 0.15

        # New: Wave 5 vs Wave 3 projection
        if waves[5] != 0 and waves[3] != 0:
            wave_5_3_ratio = abs(waves[5] / waves[3])
            if any(abs(wave_5_3_ratio - ratio) < 0.1 + tolerance for ratio in fib.WAVE5_FROM_3):
                wave5_confidence += 0.2
                
        # Add Rate-of-Change projection validation
        roc_confidence, _ = validate_rate_of_change_projection(waves, tolerance)
        wave5_confidence += roc_confidence
        
        # Add Geometric-Mean projection validation
        geom_confidence, _ = validate_geometric_mean_projection(waves, tolerance)
        wave5_confidence += geom_confidence

        # Cap Wave 5 contribution to 0.3 of total confidence
        confidence += min(wave5_confidence, 0.3)

        # Time relationships (if dates available)
        if len(dates) >= 5:
            try:
                durations = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                if len(durations) >= 4:
                    # Wave 2 vs Wave 4 time relationship
                    if durations[1] != 0:
                        time_ratio = durations[3] / durations[1]
                        if any(abs(time_ratio - ratio) < 0.2 + tolerance for ratio in fib.TIME_RATIOS):
                            confidence += 0.1
            except:
                pass  # Time analysis failed, skip
    except Exception:
        pass  # Fibonacci analysis failed, return current confidence
    return min(confidence, 0.8)  # Cap at 0.8 