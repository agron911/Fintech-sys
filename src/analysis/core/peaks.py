import numpy as np
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, List, Dict, Any
import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def get_candlestick_config(candlestick_type: str) -> Dict[str, Any]:
    """Get configuration specific to candlestick type"""
    configs = {
        'day': {
            'min_wave_periods': 5,
            'min_move_pct': 0.015,  # Reduced from 0.02 to 0.015 (1.5%) for better detection
            'cluster_window': 5,
            'subwave_min_duration': 5,
            'lookback_years': 3
        },
        'week': {
            'min_wave_periods': 3,
            'min_move_pct': 0.025,  # Reduced from 0.03 to 0.025 (2.5%) for better detection
            'cluster_window': 3,
            'subwave_min_duration': 3,  # 3 weeks instead of 5 days
            'lookback_years': 10
        },
        'month': {
            'min_wave_periods': 2,
            'min_move_pct': 0.04,  # Reduced from 0.05 to 0.04 (4%) for better detection
            'cluster_window': 2,
            'subwave_min_duration': 2,
            'lookback_years': 20
        }
    }
    return configs.get(candlestick_type, configs['day'])

def compute_atr_adaptive(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute adaptive ATR with fallback for missing OHLC data.
    """
    try:
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high = df['high']
            low = df['low']
            close = df['close']
            
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        else:
            # Fallback using only close prices
            close = df['close']
            true_range = np.abs(close.pct_change()) * close
            
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr.fillna(atr.mean())
    except Exception:
        # Ultimate fallback
        return df['close'].rolling(window=window).std()

def compute_local_volatility(price_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute local volatility using rolling standard deviation of returns.
    """
    returns = price_series.pct_change()
    local_vol = returns.rolling(window=window, min_periods=5).std()
    return local_vol.fillna(local_vol.mean())

def zigzag_peaks_troughs(price_series: pd.Series, threshold_pct: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale ZigZag approach to identify significant turning points.

    Parameters:
    - threshold_pct: Minimum percentage move to qualify as a turning point (default 3% for better pattern detection)
    """
    prices = price_series.values
    dates = price_series.index
    
    # Start with the first price point
    peaks = []
    troughs = []
    extrema = []  # (index, price, type)
    
    if len(prices) < 3:
        return np.array([]), np.array([])
    
    # Initialize with first point
    current_trend = None
    last_extreme_idx = 0
    last_extreme_price = prices[0]
    
    for i in range(1, len(prices)):
        current_price = prices[i]
        
        # Calculate percentage change from last extreme
        pct_change = (current_price - last_extreme_price) / last_extreme_price
        
        if current_trend is None:
            # Establish initial trend
            if abs(pct_change) >= threshold_pct:
                if pct_change > 0:
                    current_trend = 'up'
                    troughs.append(last_extreme_idx)
                else:
                    current_trend = 'down'
                    peaks.append(last_extreme_idx)
                last_extreme_idx = i
                last_extreme_price = current_price
        
        elif current_trend == 'up':
            if current_price > last_extreme_price:
                # New high, update current extreme
                last_extreme_idx = i
                last_extreme_price = current_price
            elif pct_change <= -threshold_pct:
                # Trend reversal down
                peaks.append(last_extreme_idx)
                current_trend = 'down'
                last_extreme_idx = i
                last_extreme_price = current_price
        
        elif current_trend == 'down':
            if current_price < last_extreme_price:
                # New low, update current extreme
                last_extreme_idx = i
                last_extreme_price = current_price
            elif pct_change >= threshold_pct:
                # Trend reversal up
                troughs.append(last_extreme_idx)
                current_trend = 'up'
                last_extreme_idx = i
                last_extreme_price = current_price
    
    # Add the final extreme
    if current_trend == 'up' and last_extreme_idx not in peaks:
        peaks.append(last_extreme_idx)
    elif current_trend == 'down' and last_extreme_idx not in troughs:
        troughs.append(last_extreme_idx)
    
    return np.array(sorted(peaks)), np.array(sorted(troughs))

def cluster_nearby_extrema(indices: np.ndarray, prices: pd.Series, 
                          time_window: int = 5, price_threshold_pct: float = 0.02,
                          is_peak: bool = True) -> np.ndarray:
    """
    Advanced clustering of nearby extrema with both time and price considerations.
    
    Parameters:
    - time_window: Maximum days between points to consider for clustering
    - price_threshold_pct: Maximum price difference (as %) to cluster points
    - is_peak: Whether clustering peaks (True) or troughs (False)
    """
    if len(indices) == 0:
        return np.array([])
    
    clustered = []
    indices = np.sort(indices)
    i = 0
    
    while i < len(indices):
        cluster_group = [indices[i]]
        base_price = prices.iloc[indices[i]]
        
        # Look ahead for nearby points
        j = i + 1
        while j < len(indices):
            candidate_idx = indices[j]
            candidate_price = prices.iloc[candidate_idx]
            
            # Check time proximity
            time_diff = candidate_idx - indices[i]
            if time_diff > time_window:
                break
            
            # Check price proximity
            price_diff_pct = abs(candidate_price - base_price) / abs(base_price)
            if price_diff_pct <= price_threshold_pct:
                cluster_group.append(candidate_idx)
                j += 1
            else:
                break
        
        # Select the most extreme point in the cluster
        if is_peak:
            best_idx = max(cluster_group, key=lambda idx: prices.iloc[idx])
        else:
            best_idx = min(cluster_group, key=lambda idx: prices.iloc[idx])
        
        clustered.append(best_idx)
        i = j if j > i + 1 else i + 1
    
    return np.array(clustered)

def adaptive_prominence_calculation(df: pd.DataFrame, atr: pd.Series, 
                                  base_prominence_pct: float = 0.01) -> float:
    """
    Calculate adaptive prominence based on recent volatility and price level.
    """
    try:
        # Get recent price level and volatility
        recent_price = df['close'].iloc[-20:].mean()
        recent_atr = atr.iloc[-20:].mean()
        
        # Calculate volatility-adjusted prominence
        volatility_factor = recent_atr / recent_price
        
        # Scale prominence based on volatility
        # Higher volatility = higher prominence needed to filter noise
        adaptive_prominence = recent_price * (base_prominence_pct + volatility_factor * 2)
        
        # Ensure reasonable bounds
        min_prominence = recent_price * 0.005  # 0.5%
        max_prominence = recent_price * 0.05   # 5%
        
        return np.clip(adaptive_prominence, min_prominence, max_prominence)
    
    except Exception:
        return df['close'].iloc[-1] * base_prominence_pct

def multi_scale_detection(df: pd.DataFrame, column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale peak and trough detection combining multiple approaches.

    PERFORMANCE OPTIMIZED: Reduced from 9 passes to 3 passes (3x faster)
    """
    price_series = df[column].ffill().bfill()

    all_peaks = []
    all_troughs = []

    # PERFORMANCE OPTIMIZATION: Use only ONE ZigZag threshold instead of 3
    # Method 1: ZigZag with optimal single threshold
    threshold = 0.05  # 5% - best balance between sensitivity and noise
    peaks_zz, troughs_zz = zigzag_peaks_troughs(price_series, threshold)
    all_peaks.extend(peaks_zz)
    all_troughs.extend(troughs_zz)

    # Method 2: Traditional scipy.signal with adaptive parameters
    atr = compute_atr_adaptive(df)
    adaptive_prominence = adaptive_prominence_calculation(df, atr)
    local_vol = compute_local_volatility(price_series)

    # Adaptive distance based on recent volatility
    avg_vol = local_vol.iloc[-50:].mean() if len(local_vol) > 50 else local_vol.mean()
    adaptive_distance = max(3, min(20, int(10 * avg_vol * 100)))  # Scale with volatility

    # PERFORMANCE OPTIMIZATION: Use only ONE smoothing window instead of 3
    # Apply optimal smoothing window (3 gives best noise reduction without losing peaks)
    smooth_window = 3
    smoothed = price_series.rolling(window=smooth_window, center=True).mean()
    smoothed = smoothed.ffill().bfill()

    try:
        peaks_scipy, _ = find_peaks(smoothed.values,
                                  distance=adaptive_distance,
                                  prominence=adaptive_prominence)
        troughs_scipy, _ = find_peaks(-smoothed.values,
                                    distance=adaptive_distance,
                                    prominence=adaptive_prominence)

        all_peaks.extend(peaks_scipy)
        all_troughs.extend(troughs_scipy)
    except Exception:
        pass

    # PERFORMANCE OPTIMIZATION: Use only ONE order instead of 3
    # Method 3: Relative extrema with optimal order
    order = 5  # Sweet spot for Elliott Wave patterns
    try:
        peaks_rel = argrelextrema(price_series.values, np.greater, order=order)[0]
        troughs_rel = argrelextrema(price_series.values, np.less, order=order)[0]

        all_peaks.extend(peaks_rel)
        all_troughs.extend(troughs_rel)
    except Exception:
        pass

    # Remove duplicates and sort
    unique_peaks = np.unique(all_peaks)
    unique_troughs = np.unique(all_troughs)
    
    return unique_peaks, unique_troughs

def detect_peaks_troughs_enhanced(df: pd.DataFrame, column: str = 'close',
                                   candlestick_type: str = 'day') -> Tuple[np.ndarray, np.ndarray]:
    config = get_candlestick_config(candlestick_type)
    return detect_peaks_troughs_enhanced_internal(
        df, column,
        min_move_pct=config['min_move_pct'],
        cluster_time_window=config['cluster_window'],
        cluster_price_threshold=config['min_move_pct'] * 0.67
    )

def detect_peaks_troughs_enhanced_internal(df: pd.DataFrame, column: str = 'close',
                                           min_move_pct: float = 0.03,
                                           cluster_time_window: int = 10,
                                           cluster_price_threshold: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced peak and trough detection specifically designed for Elliott Wave analysis.
    
    Key improvements:
    1. Multi-scale ZigZag approach
    2. Adaptive prominence based on volatility
    3. Intelligent clustering of nearby extrema
    4. Multiple detection methods combined
    5. Better handling of TSLA's volatility patterns
    
    Parameters:
    - min_move_pct: Minimum percentage move to qualify as significant (default 2%)
    - cluster_time_window: Days to look for clustering extrema
    - cluster_price_threshold: Price similarity threshold for clustering (1.5%)
    """
    
    if len(df) < 10:
        return np.array([]), np.array([])
    
    logger.info(f"[DEBUG] Starting enhanced detection on {len(df)} data points")
    
    # Step 1: Multi-scale detection
    raw_peaks, raw_troughs = multi_scale_detection(df, column)
    
    logger.info(f"[DEBUG] Raw detection found {len(raw_peaks)} peaks, {len(raw_troughs)} troughs")
    
    # Step 2: Filter by minimum move requirement
    price_series = df[column]
    
    def filter_by_move_size(extrema_indices: np.ndarray, is_peak: bool) -> np.ndarray:
        if len(extrema_indices) == 0:
            return extrema_indices
        
        filtered = []
        for idx in extrema_indices:
            if idx == 0 or idx >= len(price_series) - 1:
                continue
            
            current_price = price_series.iloc[idx]
            
            # Look for significant move before this extreme
            significant_move_found = False
            lookback = min(20, idx)  # Look back up to 20 periods
            
            for i in range(1, lookback + 1):
                prev_price = price_series.iloc[idx - i]
                move_pct = abs(current_price - prev_price) / abs(prev_price)
                
                if move_pct >= min_move_pct:
                    # Check if move direction matches extreme type
                    if is_peak and current_price > prev_price:
                        significant_move_found = True
                        break
                    elif not is_peak and current_price < prev_price:
                        significant_move_found = True
                        break
            
            if significant_move_found:
                filtered.append(idx)
        
        return np.array(filtered)
    
    # Apply move size filter
    filtered_peaks = filter_by_move_size(raw_peaks, is_peak=True)
    filtered_troughs = filter_by_move_size(raw_troughs, is_peak=False)
    
    logger.info(f"[DEBUG] After move filter: {len(filtered_peaks)} peaks, {len(filtered_troughs)} troughs")
    
    # Step 3: Cluster nearby extrema
    clustered_peaks = cluster_nearby_extrema(
        filtered_peaks, price_series, 
        time_window=cluster_time_window,
        price_threshold_pct=cluster_price_threshold,
        is_peak=True
    )
    
    clustered_troughs = cluster_nearby_extrema(
        filtered_troughs, price_series,
        time_window=cluster_time_window, 
        price_threshold_pct=cluster_price_threshold,
        is_peak=False
    )
    
    logger.info(f"[DEBUG] After clustering: {len(clustered_peaks)} peaks, {len(clustered_troughs)} troughs")
    
    # Step 4: Final validation - ensure alternating structure potential
    def ensure_minimum_separation(peaks: np.ndarray, troughs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure there's reasonable separation between peaks and troughs."""
        all_extrema = []
        for p in peaks:
            all_extrema.append((p, 'peak', price_series.iloc[p]))
        for t in troughs:
            all_extrema.append((t, 'trough', price_series.iloc[t]))
        
        # Sort by time
        all_extrema.sort(key=lambda x: x[0])
        
        # Remove extrema that are too close in time (less than 2 periods apart)
        filtered_extrema = []
        for i, (idx, ext_type, price) in enumerate(all_extrema):
            if i == 0:
                filtered_extrema.append((idx, ext_type, price))
                continue
            
            prev_idx = filtered_extrema[-1][0]
            if idx - prev_idx >= 2:  # At least 2 periods apart
                filtered_extrema.append((idx, ext_type, price))
        
        # Separate back into peaks and troughs
        final_peaks = [x[0] for x in filtered_extrema if x[1] == 'peak']
        final_troughs = [x[0] for x in filtered_extrema if x[1] == 'trough']
        
        return np.array(final_peaks), np.array(final_troughs)
    
    final_peaks, final_troughs = ensure_minimum_separation(clustered_peaks, clustered_troughs)
    
    logger.info(f"[DEBUG] Final result: {len(final_peaks)} peaks, {len(final_troughs)} troughs")
    
    # Step 5: Add boundary points if needed (start/end of series)
    boundary_peaks = list(final_peaks)
    boundary_troughs = list(final_troughs)
    
    # Add first point if it's a significant extreme
    if len(df) > 5:
        first_price = price_series.iloc[0]
        early_prices = price_series.iloc[1:6]
        
        if first_price > early_prices.max():  # First point is a peak
            if 0 not in boundary_peaks:
                boundary_peaks.insert(0, 0)
        elif first_price < early_prices.min():  # First point is a trough
            if 0 not in boundary_troughs:
                boundary_troughs.insert(0, 0)
    
    # Add last point if it's a significant extreme
    if len(df) > 5:
        last_idx = len(df) - 1
        last_price = price_series.iloc[last_idx]
        late_prices = price_series.iloc[-6:-1]
        
        if len(late_prices) > 0:
            if last_price > late_prices.max():  # Last point is a peak
                if last_idx not in boundary_peaks:
                    boundary_peaks.append(last_idx)
            elif last_price < late_prices.min():  # Last point is a trough
                if last_idx not in boundary_troughs:
                    boundary_troughs.append(last_idx)
    
    return np.array(sorted(boundary_peaks)), np.array(sorted(boundary_troughs))

# Utility function for testing and validation
def validate_extrema_quality(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, 
                           column: str = 'close') -> Dict[str, Any]:
    """
    Validate the quality of detected extrema for Elliott Wave analysis.
    """
    price_series = df[column]
    
    validation_results = {
        'total_peaks': len(peaks),
        'total_troughs': len(troughs),
        'total_extrema': len(peaks) + len(troughs),
        'avg_peak_prominence': 0.0,
        'avg_trough_prominence': 0.0,
        'min_separation_days': float('inf'),
        'alternation_potential': False
    }
    
    # Calculate average prominence
    if len(peaks) > 0:
        peak_prominences = []
        for peak_idx in peaks:
            if peak_idx > 0 and peak_idx < len(price_series) - 1:
                peak_price = price_series.iloc[peak_idx]
                left_min = price_series.iloc[max(0, peak_idx-5):peak_idx].min()
                right_min = price_series.iloc[peak_idx+1:min(len(price_series), peak_idx+6)].min()
                prominence = peak_price - max(left_min, right_min)
                peak_prominences.append(prominence / peak_price)
        
        if peak_prominences:
            validation_results['avg_peak_prominence'] = np.mean(peak_prominences)
    
    # Similar for troughs
    if len(troughs) > 0:
        trough_prominences = []
        for trough_idx in troughs:
            if trough_idx > 0 and trough_idx < len(price_series) - 1:
                trough_price = price_series.iloc[trough_idx]
                left_max = price_series.iloc[max(0, trough_idx-5):trough_idx].max()
                right_max = price_series.iloc[trough_idx+1:min(len(price_series), trough_idx+6)].max()
                prominence = min(left_max, right_max) - trough_price
                trough_prominences.append(prominence / trough_price)
        
        if trough_prominences:
            validation_results['avg_trough_prominence'] = np.mean(trough_prominences)
    
    # Check minimum separation
    all_extrema_indices = np.concatenate([peaks, troughs])
    if len(all_extrema_indices) > 1:
        all_extrema_indices = np.sort(all_extrema_indices)
        separations = np.diff(all_extrema_indices)
        validation_results['min_separation_days'] = int(np.min(separations))
    
    # Check alternation potential (can we form alternating sequences?)
    all_extrema = []
    for p in peaks:
        all_extrema.append((p, 'peak'))
    for t in troughs:
        all_extrema.append((t, 'trough'))
    
    all_extrema.sort()
    
    # Look for potential alternating sequences of length 5+
    max_alternating_length = 0
    current_length = 1
    
    for i in range(1, len(all_extrema)):
        if all_extrema[i][1] != all_extrema[i-1][1]:  # Different type
            current_length += 1
        else:
            max_alternating_length = max(max_alternating_length, current_length)
            current_length = 1
    
    max_alternating_length = max(max_alternating_length, current_length)
    validation_results['max_alternating_length'] = max_alternating_length
    validation_results['alternation_potential'] = max_alternating_length >= 5
    
    return validation_results

# Test function test_enhanced_detection() moved to tests/test_peaks.py for proper separation of concerns