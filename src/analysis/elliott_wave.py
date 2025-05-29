import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, List, Dict, Any
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


class WaveDegree:
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"

class ElliottWave:
    def __init__(self, points, degree, wave_type):
        self.points = points
        self.degree = degree
        self.wave_type = wave_type  # 'impulse' or 'corrective'
        self.sub_waves = []


def detect_peaks_troughs_enhanced(df: pd.DataFrame, column: str = 'close', window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced detection of peaks and troughs with adaptive parameters and filtering for quality.
    """
    # Smooth the price series
    smoothed = df[column].rolling(window=window, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    
    # Calculate dynamic parameters based on volatility
    volatility = df[column].pct_change().std() * 100
    distance = max(5, int(10 * (1 + volatility/100)))  # More volatile = more distance between peaks
    prominence = max(0.01, volatility * df[column].mean() * 0.0005)  # Prominence scaled to price and volatility
    
    # Find peaks and troughs using dynamic parameters
    peaks, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)
    
    # Filter for quality peaks and troughs
    avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
    
    # Only keep peaks/troughs with above average volume if volume data is available
    if 'volume' in df.columns:
        peaks = np.array([p for p in peaks if df['volume'].iloc[p] > avg_volume * 0.8])
        troughs = np.array([t for t in troughs if df['volume'].iloc[t] > avg_volume * 0.8])
    
    # Verify each peak is actually higher than nearby points and each trough is lower
    final_peaks = []
    for p in peaks:
        left_idx = max(0, p - window)
        right_idx = min(len(df) - 1, p + window)
        if df[column].iloc[p] >= df[column].iloc[left_idx:right_idx+1].max() * 0.99:
            final_peaks.append(p)
    
    final_troughs = []
    for t in troughs:
        left_idx = max(0, t - window)
        right_idx = min(len(df) - 1, t + window)
        if df[column].iloc[t] <= df[column].iloc[left_idx:right_idx+1].min() * 1.01:
            final_troughs.append(t)
    
    return np.array(final_peaks), np.array(final_troughs)


def find_elliott_wave_pattern(
    df: pd.DataFrame,
    column: str = 'close',
    min_points: int = 5,
    max_points: int = 8
) -> Tuple[np.ndarray, float]:
    """
    Find the best Elliott Wave pattern in the price series.
    """
    # Detect peaks and troughs
    peaks, troughs = detect_peaks_troughs_enhanced(df, column)
    
    if len(peaks) == 0 or len(troughs) == 0:
        return np.array([]), 0.0
    
    # Elliott Wave patterns start with either a peak or trough
    # We'll test both and see which gives a better result
    candidates = []
    
    # Try starting with a trough (for bullish patterns)
    if len(troughs) > 0:
        bullish_candidates = generate_wave_candidates(df, peaks, troughs, start_with='trough', 
                                                      column=column, min_points=min_points, max_points=max_points)
        candidates.extend(bullish_candidates)
    
    # Try starting with a peak (for bearish patterns)
    if len(peaks) > 0:
        bearish_candidates = generate_wave_candidates(df, peaks, troughs, start_with='peak', 
                                                      column=column, min_points=min_points, max_points=max_points)
        candidates.extend(bearish_candidates)
    
    # No candidates found
    if not candidates:
        return np.array([]), 0.0
    
    # Sort by confidence score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best candidate
    return candidates[0][0], candidates[0][1]


def generate_wave_candidates(
    df: pd.DataFrame,
    peaks: np.ndarray,
    troughs: np.ndarray,
    start_with: str = 'trough',
    column: str = 'close',
    min_points: int = 5,
    max_points: int = 8
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate Elliott Wave pattern candidates based on peaks and troughs.
    """
    candidates = []
    
    # Filter points by date range - only look at the last N years
    start_date = df.index.max() - pd.DateOffset(years=5)
    date_filter = df.index >= start_date
    
    # Get the valid starting points based on the pattern type
    if start_with == 'trough':
        # For bullish pattern, starting points are troughs
        start_points = [t for t in troughs if date_filter[t]]
        expected_sequence = ['trough', 'peak', 'trough', 'peak', 'trough']
    else:
        # For bearish pattern, starting points are peaks
        start_points = [p for p in peaks if date_filter[p]]
        expected_sequence = ['peak', 'trough', 'peak', 'trough', 'peak']
    
    if not start_points:
        return candidates
    
    # Create a combined array of labeled points
    labeled_peaks = [(int(p), 'peak') for p in peaks]
    labeled_troughs = [(int(t), 'trough') for t in troughs]
    labeled_points = sorted(labeled_peaks + labeled_troughs, key=lambda x: x[0])
    
    for start in start_points:
        # Find the index in labeled_points where this start occurs
        for idx, (pt_idx, pt_type) in enumerate(labeled_points):
            if pt_idx == start and pt_type == expected_sequence[0]:
                seq = [(pt_idx, pt_type)]
                seq_idx = idx
                pattern_idx = 1
                # Build the sequence
                for next_idx in range(seq_idx + 1, len(labeled_points)):
                    next_pt_idx, next_pt_type = labeled_points[next_idx]
                    if next_pt_type == expected_sequence[pattern_idx % len(expected_sequence)] and next_pt_idx > seq[-1][0]:
                        seq.append((next_pt_idx, next_pt_type))
                        pattern_idx += 1
                        if len(seq) >= max_points:
                            break
                # Only consider sequences of sufficient length
                if min_points <= len(seq) <= max_points:
                    indices = np.array([i for i, _ in seq])
                    conf = validate_fibonacci(df, indices, column, min_length=min_points)
                    candidates.append((indices, conf))
    return candidates


def detect_peaks_troughs(df: pd.DataFrame, column: str = 'close', distance: int = None, prominence: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local peaks and troughs in a price series with smoothing and volume confirmation.
    """
    # Smooth the price series using a 5-day moving average
    smoothed = df[column].rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')

    # Calculate dynamic distance and prominence based on volatility (using standard deviation)
    volatility = df[column].pct_change().std() * 100  # Daily percentage change standard deviation
    distance = distance or max(3, int(5 * (1 + volatility)))  # Increase distance for volatile stocks
    prominence = prominence or max(0.5, volatility * df[column].mean() * 0.01)  # Scale prominence with volatility

    # Detect peaks and troughs on smoothed data
    peaks, peak_props = find_peaks(smoothed, distance=distance, prominence=prominence)
    troughs, trough_props = find_peaks(-smoothed, distance=distance, prominence=prominence)

    # Volume confirmation: keep peaks/troughs with above-average volume
    avg_volume = df['volume'].mean()
    peaks = np.array([p for p in peaks if df['volume'].iloc[p] > avg_volume])
    troughs = np.array([t for t in troughs if df['volume'].iloc[t] > avg_volume])

    return peaks, troughs


def validate_fibonacci(df: pd.DataFrame, wave_points: np.ndarray, column: str = 'close', min_length: int = 5) -> float:
    """
    Validate wave points using Fibonacci retracement levels and Elliott Wave rules.
    Returns a confidence score between 0.0 and 1.0.
    """
    if len(wave_points) < min_length:
        return 0.5 if len(wave_points) >= 3 else 0.0  # Partial confidence for 3+ wave patterns
    
    prices = df[column].iloc[wave_points]
    dates = df.index[wave_points]
    
    # Check for strict chronological order
    if not all(dates[i] < dates[i+1] for i in range(len(dates)-1)):
        return 0.0  # Reject non-chronological sequences
    
    # Initialize confidence score
    confidence = 0.5  # Start with base confidence
    
    # Wave 1: Start to first peak/trough
    wave_1 = prices[1] - prices[0]
    wave_1_duration = (dates[1] - dates[0]).days if len(dates) > 1 else 1
    
    # Wave 2: Should retrace between 23.6-78.6% of Wave 1 (standard Fibonacci levels)
    wave_2 = prices[2] - prices[1]
    wave_2_duration = (dates[2] - dates[1]).days if len(dates) > 2 else 1
    wave_2_retracement = abs(wave_2 / wave_1) if wave_1 != 0 else 0
    
    # Validate Wave 2 retracement
    if wave_2_retracement > 0.786:  # Never exceed 78.6%
        return 0.0
    elif 0.382 <= wave_2_retracement <= 0.618:
        confidence += 0.2  # Ideal range
    elif 0.236 <= wave_2_retracement <= 0.786:
        confidence += 0.1  # Acceptable range
    
    # Wave 3: Should be the strongest and extend beyond Wave 1
    if len(wave_points) >= 4:
        wave_3 = prices[3] - prices[2]
        wave_3_duration = (dates[3] - dates[2]).days if len(dates) > 3 else 1
        
        # Direction check: Wave 3 should continue in the same direction as Wave 1
        if (wave_1 > 0 and wave_3 < 0) or (wave_1 < 0 and wave_3 > 0):
            return 0.0  # Wave 3 must continue the primary trend
        
        # Wave 3 should typically be longer than Wave 1 (relaxed: at least 70% of Wave 1)
        if abs(wave_3) >= 0.7 * abs(wave_1):
            confidence += 0.1
        else:
            confidence -= 0.1
    
    # Wave 4: Should retrace Wave 3 but not overlap with Wave 1 territory
    if len(wave_points) >= 5:
        wave_4 = prices[4] - prices[3]
        wave_4_duration = (dates[4] - dates[3]).days if len(dates) > 4 else 1
        wave_4_retracement = abs(wave_4 / wave_3) if wave_3 != 0 else 0
        
        # Direction check: Wave 4 should be in opposite direction of Wave 3
        if (wave_3 > 0 and wave_4 > 0) or (wave_3 < 0 and wave_4 < 0):
            return 0.0  # Wave 4 must be in opposite direction
        
        # Validate Wave 4 retracement (typically 23.6% to 50.0%)
        if 0.236 <= wave_4_retracement <= 0.5:
            confidence += 0.1
        
        # Non-overlap rule: Wave 4 should not enter Wave 1 territory
        if wave_1 > 0:  # Uptrend
            wave_1_high = max(prices[0], prices[1])
            wave_1_low = min(prices[0], prices[1])
            if wave_1_low <= prices[4] <= wave_1_high:
                return 0.0  # Wave 4 overlaps Wave 1 territory
        else:  # Downtrend
            wave_1_high = max(prices[0], prices[1])
            wave_1_low = min(prices[0], prices[1])
            if wave_1_low <= prices[4] <= wave_1_high:
                return 0.0  # Wave 4 overlaps Wave 1 territory
    
    # Wave 5: Final impulse in the direction of Waves 1 and 3
    if len(wave_points) >= 6:
        wave_5 = prices[5] - prices[4]
        wave_5_duration = (dates[5] - dates[4]).days if len(dates) > 5 else 1
        
        # Direction check: Wave 5 should continue in the same direction as Waves 1 and 3
        if (wave_1 > 0 and wave_5 < 0) or (wave_1 < 0 and wave_5 > 0):
            return 0.0  # Wave 5 must continue the primary trend
        
        # Typical Wave 5 is either similar to Wave 1 or extends to 1.618 * Wave 1
        wave_5_proportion = abs(wave_5 / wave_1) if wave_1 != 0 else 0
        if 0.5 <= wave_5_proportion <= 2.0:
            confidence += 0.1
    
    # Check alternation principle: Wave 2 and Wave 4 should differ in character
    # (Simple check: one should be sharp, one should be flat/sideways)
    if len(wave_points) >= 5:
        wave_2_slope = abs(wave_2) / wave_2_duration if wave_2_duration > 0 else 0
        wave_4_slope = abs(wave_4) / wave_4_duration if wave_4_duration > 0 else 0
        
        # If one wave is at least twice as steep as the other, they have different character
        if wave_2_slope > 2 * wave_4_slope or wave_4_slope > 2 * wave_2_slope:
            confidence += 0.1
    
    # Cap confidence at 1.0
    return min(1.0, confidence)


def refined_elliott_wave_suggestion(
    df: pd.DataFrame,
    peaks: np.ndarray,
    troughs: np.ndarray,
    min_wave_length: int = 5,
    max_wave_length: int = 8,
    min_price_change: float = None,
    column: str = 'close'
) -> Tuple[np.ndarray, float]:
    """
    Find the longest alternating sequence of peaks and troughs with significant price changes,
    ensuring chronological order and proper Elliott Wave structure.
    Returns a tuple of (wave_points, confidence_score).
    """
    volatility = df[column].pct_change().std() * 100
    min_price_change = min_price_change or (volatility * df[column].mean() * 0.005)

    # Combine and label peaks and troughs
    points = np.concatenate([[(int(i), 'peak') for i in peaks], [(int(i), 'trough') for i in troughs]])
    points = sorted(points, key=lambda x: x[0])  # Ensure points are sorted by index (chronological order)
    
    if not points:
        return np.array([]), 0.0
    
    # Filter points by significant price change
    filtered_points = [points[0]]
    for pt in points[1:]:
        prev_idx, _ = filtered_points[-1]
        curr_idx = pt[0]
        prev_price = df[column].iloc[int(prev_idx)]
        curr_price = df[column].iloc[int(curr_idx)]
        if abs(curr_price - prev_price) >= min_price_change:
            filtered_points.append(pt)
    
    # Find valid Elliott Wave sequences
    best_seq = []
    best_confidence = 0.0
    
    # Try different starting points
    for start in range(len(filtered_points) - min_wave_length + 1):
        # Initialize with alternating peak/trough pattern
        if start + 1 < len(filtered_points):
            # Decide if we start with peak or trough
            if filtered_points[start][1] == 'peak':
                expected_pattern = ['peak', 'trough', 'peak', 'trough', 'peak']
            else:
                expected_pattern = ['trough', 'peak', 'trough', 'peak', 'trough']
            
            # Build potential sequence
            seq = [filtered_points[start]]
            pattern_idx = 1  # Start with second element of pattern
            
            # Search for the next elements in chronological order
            for j in range(start + 1, len(filtered_points)):
                current_type = filtered_points[j][1]
                if current_type == expected_pattern[pattern_idx % len(expected_pattern)]:
                    prev_idx = seq[-1][0]
                    curr_idx = filtered_points[j][0]
                    
                    # Verify it's later in time
                    if curr_idx > prev_idx:
                        # Verify significant price change
                        prev_price = df[column].iloc[int(prev_idx)]
                        curr_price = df[column].iloc[int(curr_idx)]
                        if abs(curr_price - prev_price) >= min_price_change:
                            seq.append(filtered_points[j])
                            pattern_idx += 1
                            
                            # Stop if we've reached the maximum wave length
                            if len(seq) >= max_wave_length:
                                break
            
            # Check if we have a valid Elliott Wave sequence
            if len(seq) >= min_wave_length:
                candidate = np.array([int(i) for i, _ in seq])
                is_valid = validate_fibonacci(df, candidate, column)
                
                # Calculate confidence based on how well this matches Elliott Wave patterns
                # Base confidence on Fibonacci validation
                confidence = 0.8 if is_valid else 0.0
                
                # Add extra confidence for every wave point after the minimum
                confidence += min(0.2, (len(seq) - min_wave_length) * 0.05)
                
                # Check if this is the best sequence so far
                if confidence > best_confidence:
                    best_seq = seq
                    best_confidence = confidence
    
    # Convert the sequence to array of indices
    result = np.array([int(i) for i, _ in best_seq])
    
    # Extra check to ensure chronological order
    if len(result) > 0:
        # Verify the indices are strictly increasing
        is_chronological = np.all(np.diff(result) > 0)
        if not is_chronological:
            # If not chronological, sort them
            result = np.sort(result)
            # Reduce confidence due to sorting required
            best_confidence *= 0.8
    
    return result, best_confidence


def plot_peaks_troughs(
    df: pd.DataFrame,
    peaks: np.ndarray,
    troughs: np.ndarray,
    wave_points: np.ndarray = None,
    column: str = 'close',
    title: str = 'Price with Peaks, Troughs, and Elliott Waves'
):
    """
    Plot price with detected peaks, troughs, and Elliott Wave points.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label='Price', alpha=0.7)
    plt.plot(df.index[peaks], df[column].iloc[peaks], 'ro', label='Peaks')
    plt.plot(df.index[troughs], df[column].iloc[troughs], 'go', label='Troughs')
    if wave_points is not None and len(wave_points) > 0:
        plt.plot(df.index[wave_points], df[column].iloc[wave_points], 'bo', label='Wave Points')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('elliott_wave_plot.png')


def suggest_elliott_waves(peaks, troughs):
    """
    Suggest candidate Elliott wave points by combining and sorting peaks and troughs.
    Returns the first 8 points as a candidate wave (for demonstration).
    """
    import numpy as np
    points = np.sort(np.concatenate([peaks, troughs]))
    if len(points) >= 8:
        return points[:8]
    return points


def advanced_elliott_wave_suggestion(peaks, troughs):
    """
    Attempt to find the longest alternating sequence of peaks and troughs,
    which is a good candidate for Elliott wave analysis.
    Returns the indices of the detected wave points.
    """
    # Combine and label peaks/troughs
    points = np.concatenate([[(i, 'peak') for i in peaks], [(i, 'trough') for i in troughs]])
    # Sort by index
    points = sorted(points, key=lambda x: x[0])
    # Find the longest alternating sequence
    best_seq = []
    for start in range(len(points)):
        seq = [points[start]]
        for j in range(start + 1, len(points)):
            if points[j][1] != seq[-1][1]:  # alternate
                seq.append(points[j])
                if len(seq) >= max_wave_length:
                    break
        if min_wave_length <= len(seq) <= max_wave_length and len(seq) > len(best_seq):
            best_seq = seq
    # Return just the indices
    return np.array([i for i, _ in best_seq])


def detect_corrective_waves(df: pd.DataFrame, wave_points: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> np.ndarray:
    """
    Detect corrective waves (A-B-C) after a 5-wave sequence.
    Returns the indices of the detected corrective wave points (A, B, C) if found, else an empty array.
    """
    if len(wave_points) < 5:
        return np.array([])
    
    # Identify the last point from the impulse wave
    last_wave_idx = int(wave_points[-1])
    
    # Combine peaks and troughs after the last main wave point
    all_points = np.concatenate([[(int(i), 'peak') for i in peaks], [(int(i), 'trough') for i in troughs]])
    all_points = sorted([pt for pt in all_points if int(pt[0]) > last_wave_idx], key=lambda x: x[0])
    
    # Look for three alternating points (A, B, C)
    corrective = []
    if len(all_points) > 0:
        corrective.append(all_points[0])
        
        for pt in all_points[1:]:
            if pt[1] != corrective[-1][1]:  # Must alternate peak/trough
                corrective.append(pt)
            if len(corrective) == 3:
                break
    
    if len(corrective) == 3:
        # Verify chronological order
        indices = [int(i) for i, _ in corrective]
        if indices[0] < indices[1] < indices[2]:
            return np.array(indices)
    
    return np.array([])


def detect_elliott_wave_complete(df: pd.DataFrame, column: str = 'close') -> dict:
    """
    Detect the complete Elliott Wave structure (impulse and corrective) and return all relevant data.
    Returns a dictionary with impulse_wave, corrective_wave, peaks, troughs, and confidence.
    """
    # Find the best impulse wave pattern
    impulse_wave, confidence = find_elliott_wave_pattern(df, column=column)
    # Use enhanced peak/trough detection for plotting
    peaks, troughs = detect_peaks_troughs_enhanced(df, column=column)
    # Detect corrective wave (A-B-C) after impulse
    corrective_wave = np.array([])
    if len(impulse_wave) >= 5:
        # Use the same logic as detect_corrective_waves, but with enhanced peaks/troughs
        last_wave_idx = int(impulse_wave[-1])
        all_points = np.concatenate([
            [(int(i), 'peak') for i in peaks],
            [(int(i), 'trough') for i in troughs]
        ])
        all_points = sorted([pt for pt in all_points if int(pt[0]) > last_wave_idx], key=lambda x: x[0])
        corrective = []
        if len(all_points) > 0:
            corrective.append(all_points[0])
            for pt in all_points[1:]:
                if pt[1] != corrective[-1][1]:
                    corrective.append(pt)
                if len(corrective) == 3:
                    break
        if len(corrective) == 3:
            indices = [int(i) for i, _ in corrective]
            if indices[0] < indices[1] < indices[2]:
                corrective_wave = np.array(indices)
    return {
        'impulse_wave': impulse_wave,
        'corrective_wave': corrective_wave,
        'peaks': peaks,
        'troughs': troughs,
        'confidence': confidence
    }


def plot_elliott_wave_analysis(
    df: pd.DataFrame,
    wave_data: dict,
    column: str = 'close',
    title: str = 'Elliott Wave Analysis',
    ax=None
):
    """
    Plot price, peaks, troughs, impulse and corrective Elliott Waves, and annotate all points.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = ax.figure
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    corrective_wave = wave_data.get('corrective_wave', np.array([]))
    peaks = wave_data.get('peaks', np.array([]))
    troughs = wave_data.get('troughs', np.array([]))
    confidence = wave_data.get('confidence', 0.0)
    
    ax.clear()
    ax.plot(df.index, df[column], label='Price', alpha=0.7)
    if len(peaks) > 0:
        ax.plot(df.index[peaks], df[column].iloc[peaks], 'ro', label='Peaks')
    if len(troughs) > 0:
        ax.plot(df.index[troughs], df[column].iloc[troughs], 'go', label='Troughs')
    if len(impulse_wave) > 0:
        ax.plot(df.index[impulse_wave], df[column].iloc[impulse_wave], 'bo', label='Impulse Wave')
        ax.plot(df.index[impulse_wave], df[column].iloc[impulse_wave], 'b--', alpha=0.5)
        # Annotate impulse wave points
        for idx, wp in enumerate(impulse_wave):
            if 0 <= wp < len(df.index):
                date = df.index[wp]
                price = df[column].iloc[wp]
                ax.annotate(f'W{idx+1}', (date, price), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
        # Plot Fibonacci retracement levels if we have at least 5 wave points
        if len(impulse_wave) >= 5:
            wave_1_start = df[column].iloc[impulse_wave[0]]
            wave_1_end = df[column].iloc[impulse_wave[1]]
            fib_50 = wave_1_start + (wave_1_end - wave_1_start) * 0.5
            fib_618 = wave_1_start + (wave_1_end - wave_1_start) * 0.618
            ax.axhline(fib_50, color='orange', linestyle='--', alpha=0.5, label='Fib 50% (W1-W2)')
            ax.axhline(fib_618, color='purple', linestyle='--', alpha=0.5, label='Fib 61.8% (W1-W2)')
            wave_3_start = df[column].iloc[impulse_wave[2]]
            wave_3_end = df[column].iloc[impulse_wave[3]]
            fib_382 = wave_3_end + (wave_3_start - wave_3_end) * 0.382
            fib_50 = wave_3_end + (wave_3_start - wave_3_end) * 0.5
            ax.axhline(fib_382, color='orange', linestyle='-.', alpha=0.5, label='Fib 38.2% (W3-W4)')
            ax.axhline(fib_50, color='purple', linestyle='-.', alpha=0.5, label='Fib 50% (W3-W4)')
    if len(corrective_wave) > 0:
        ax.plot(df.index[corrective_wave], df[column].iloc[corrective_wave], 'mo', label='Corrective Wave')
        ax.plot(df.index[corrective_wave], df[column].iloc[corrective_wave], 'm--', alpha=0.5)
        for idx, cp in enumerate(corrective_wave):
            if 0 <= cp < len(df.index):
                date = df.index[cp]
                price = df[column].iloc[cp]
                label = ['A', 'B', 'C'][idx]
                ax.annotate(label, (date, price), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='magenta')
    ax.set_title(f"{title} (Confidence: {confidence:.2f})")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(df.index.min(), df.index.max())
    # Fix date labels
    try:
        start_date = df.index.min()
        end_date = df.index.max()
        date_range = (end_date - start_date).days
        if date_range <= 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif date_range <= 365:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for label in ax.xaxis.get_majorticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune='both'))
    except Exception as e:
        print(f"Error fixing date labels in plot_elliott_wave_analysis: {e}")
    fig.tight_layout()
    if ax is None:
        plt.show() 


def detect_current_wave_position(df, column='close'):
    """
    Analyze the current position in the Elliott Wave pattern.
    Returns the current wave position, confidence, and potential next movements.
    """
    # Get the complete dataset for historical analysis
    hist_df = df.iloc[:-30].copy()  # Use all but the most recent data for pattern detection
    recent_df = df.iloc[-100:].copy()  # Use recent data to assess current position

    # Detect the last complete Elliott wave pattern in historical data
    wave_data = detect_elliott_wave_complete(hist_df, column)
    impulse_wave = wave_data['impulse_wave']
    corrective_wave = wave_data['corrective_wave']

    # If no complete pattern was found, try with more recent data
    if len(impulse_wave) < 5:
        wave_data = detect_elliott_wave_complete(recent_df, column)
        impulse_wave = wave_data['impulse_wave']
        corrective_wave = wave_data['corrective_wave']

    # No pattern found, can't determine current position
    if len(impulse_wave) < 5:
        return {
            'position': 'Unknown',
            'confidence': 0.0,
            'forecast': 'Insufficient data for Elliott Wave analysis'
        }

    # Find the end of the last identified pattern
    last_idx = corrective_wave[-1] if len(corrective_wave) > 0 else impulse_wave[-1]
    last_pattern_end = df.index[last_idx]

    # Analyze data after the last pattern
    current_segment = df.loc[last_pattern_end:].copy()

    # If the last complete pattern was an impulse wave (5 waves)
    if len(impulse_wave) >= 5 and (len(corrective_wave) == 0 or corrective_wave[-1] < impulse_wave[-1]):
        # Likely in a corrective wave
        return analyze_corrective_position(current_segment, df, impulse_wave[-1])

    # If the last complete pattern was a corrective wave (following an impulse)
    elif len(corrective_wave) > 0:
        # Likely starting a new impulse wave
        return analyze_impulse_position(current_segment, df, corrective_wave[-1])

    # Fallback if we can't clearly identify
    return {
        'position': 'Transitional',
        'confidence': 0.3,
        'forecast': 'Possibly between patterns, monitor for clearer signals'
    }


def analyze_corrective_position(current_segment, full_df, last_impulse_end_idx):
    """
    Analyze current position if we're in a corrective wave.
    """
    if len(current_segment) < 3:
        return {
            'position': 'Potential Corrective A',
            'confidence': 0.4,
            'forecast': 'Too early to determine corrective pattern, monitor for wave A completion'
        }

    # Detect peaks and troughs in the current segment
    peaks, troughs = detect_peaks_troughs_enhanced(current_segment)

    if len(peaks) == 0 or len(troughs) == 0:
        return {
            'position': 'Corrective A in progress',
            'confidence': 0.5,
            'forecast': 'Initial corrective move, expect a reversal for wave B'
        }

    # Simple corrective pattern analysis (A-B-C)
    price_at_last_impulse = full_df['close'].iloc[last_impulse_end_idx]
    current_price = current_segment['close'].iloc[-1]
    price_movement = current_price - price_at_last_impulse

    # Identify A-B-C pattern from peaks and troughs
    if len(peaks) >= 1 and len(troughs) >= 1:
        if price_movement < 0:  # Downward correction
            if len(troughs) == 1 and len(peaks) >= 2:
                return {
                    'position': 'Corrective B',
                    'confidence': 0.7,
                    'forecast': 'Wave B upward retracement in progress, expect downward wave C to follow'
                }
            elif len(troughs) >= 2:
                return {
                    'position': 'Corrective C',
                    'confidence': 0.7,
                    'forecast': 'Completing corrective pattern, prepare for new impulse wave'
                }
        else:  # Upward correction
            if len(peaks) == 1 and len(troughs) >= 2:
                return {
                    'position': 'Corrective B',
                    'confidence': 0.7,
                    'forecast': 'Wave B downward retracement in progress, expect upward wave C to follow'
                }
            elif len(peaks) >= 2:
                return {
                    'position': 'Corrective C',
                    'confidence': 0.7,
                    'forecast': 'Completing corrective pattern, prepare for new impulse wave'
                }

    return {
        'position': 'Early Corrective',
        'confidence': 0.5,
        'forecast': 'Corrective pattern developing, monitor for clearer A-B-C structure'
    }


def analyze_impulse_position(current_segment, full_df, last_corrective_end_idx):
    """
    Analyze current position if we're in an impulse wave.
    """
    if len(current_segment) < 5:
        return {
            'position': 'Potential Impulse 1',
            'confidence': 0.4,
            'forecast': 'Too early to determine impulse pattern, monitor for wave 1 completion'
        }

    # Detect peaks and troughs in the current segment
    peaks, troughs = detect_peaks_troughs_enhanced(current_segment)

    if len(peaks) == 0 or len(troughs) == 0:
        return {
            'position': 'Impulse 1 in progress',
            'confidence': 0.5,
            'forecast': 'Initial impulse move, expect a pullback for wave 2'
        }

    # Count alternating peaks and troughs to estimate wave position
    points = sorted([(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs], key=lambda x: x[0])
    alternating_count = 1
    for i in range(1, len(points)):
        if points[i][1] != points[i-1][1]:  # Different type than previous
            alternating_count += 1

    # Determine wave position based on count of alternating points
    if alternating_count == 1:
        position = 'Impulse 1'
        forecast = 'Expect corrective wave 2 pullback soon'
    elif alternating_count == 2:
        position = 'Impulse 2'
        forecast = 'Corrective pullback in progress, expect powerful wave 3 after completion'
    elif alternating_count == 3:
        position = 'Impulse 3'
        forecast = 'Strongest wave in progress, watch for extended move'
    elif alternating_count == 4:
        position = 'Impulse 4'
        forecast = 'Corrective phase after wave 3, expect final wave 5 push'
    elif alternating_count >= 5:
        position = 'Impulse 5'
        forecast = 'Final impulse wave, prepare for trend reversal and corrective pattern'
    else:
        position = 'Early Impulse'
        forecast = 'Impulse pattern developing, monitor for clearer 5-wave structure'

    return {
        'position': position,
        'confidence': min(0.4 + (alternating_count * 0.1), 0.8),
        'forecast': forecast
    } 