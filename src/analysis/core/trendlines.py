import numpy as np
from typing import Dict, Any

def analyze_diagonal_trend_lines(prices: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """
    Analyze trend lines in a potential diagonal triangle.
    """
    try:
        odd_waves = [(indices[i], prices[i]) for i in [0, 2, 4] if i < len(prices)]
        even_waves = [(indices[i], prices[i]) for i in [1, 3] if i < len(prices)]
        if len(odd_waves) < 2 or len(even_waves) < 2:
            return {"valid_trend_lines": False, "reason": "insufficient_points_for_trends"}
        odd_slope = (odd_waves[-1][1] - odd_waves[0][1]) / max(1, odd_waves[-1][0] - odd_waves[0][0])
        even_slope = (even_waves[-1][1] - even_waves[0][1]) / max(1, even_waves[-1][0] - even_waves[0][0])
        slope_difference = abs(odd_slope - even_slope)
        same_direction = (odd_slope > 0 and even_slope > 0) or (odd_slope < 0 and even_slope < 0)
        valid_convergence = slope_difference > 0.001 and same_direction
        return {
            "valid_trend_lines": valid_convergence,
            "odd_slope": odd_slope,
            "even_slope": even_slope,
            "slope_difference": slope_difference,
            "same_direction": same_direction,
            "trend_type": "converging" if abs(odd_slope) > abs(even_slope) else "diverging"
        }
    except Exception as e:
        return {"valid_trend_lines": False, "reason": f"trend_analysis_error_{str(e)}"} 