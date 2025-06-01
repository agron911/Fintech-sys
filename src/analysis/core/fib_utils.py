from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class FibonacciRatios:
    """Standard Fibonacci ratios used in Elliott Wave analysis"""
    RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSIONS = [1.618, 2.618, 4.236]
    PROJECTIONS = [0.618, 1.0, 1.618]
    TIME_RATIOS = [0.618, 1.0, 1.618, 2.618]

def validate_fibonacci_relationships(waves: Dict[int, float], dates: pd.DatetimeIndex) -> float:
    """
    Comprehensive Fibonacci relationship validation with proper ratios.
    """
    confidence = 0.0
    fib = FibonacciRatios()
    try:
        # Wave 2 retracement of Wave 1
        if waves[1] != 0:
            wave_2_ratio = abs(waves[2] / waves[1])
            if any(abs(wave_2_ratio - ratio) < 0.05 for ratio in fib.RETRACEMENTS):
                confidence += 0.2
        # Wave 3 extension relative to Wave 1
        if waves[1] != 0:
            wave_3_ratio = abs(waves[3] / waves[1])
            if any(abs(wave_3_ratio - ratio) < 0.1 for ratio in fib.EXTENSIONS):
                confidence += 0.25
            elif wave_3_ratio >= 1.618:
                confidence += 0.15
        # Wave 4 retracement of Wave 3
        if waves[3] != 0:
            wave_4_ratio = abs(waves[4] / waves[3])
            if any(abs(wave_4_ratio - ratio) < 0.05 for ratio in fib.RETRACEMENTS[:3]):
                confidence += 0.15
        # Wave 5 relationship to Wave 1
        if waves[5] != 0 and waves[1] != 0:
            wave_5_ratio = abs(waves[5] / waves[1])
            if any(abs(wave_5_ratio - ratio) < 0.1 for ratio in fib.PROJECTIONS):
                confidence += 0.2
        # Wave 5 relationship to Wave 1+3 combination
        if waves[5] != 0:
            waves_1_3_total = abs(waves[1]) + abs(waves[3])
            if waves_1_3_total != 0:
                wave_5_total_ratio = abs(waves[5]) / waves_1_3_total
                if any(abs(wave_5_total_ratio - ratio) < 0.1 for ratio in fib.PROJECTIONS):
                    confidence += 0.15
        # Time relationships (if dates available)
        if len(dates) >= 5:
            try:
                durations = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                if len(durations) >= 4:
                    if durations[1] != 0:
                        time_ratio = durations[3] / durations[1]
                        if any(abs(time_ratio - ratio) < 0.2 for ratio in fib.TIME_RATIOS):
                            confidence += 0.1
            except:
                pass
    except Exception:
        pass
    return min(confidence, 0.8) 