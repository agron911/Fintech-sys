import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, List, Dict, Any, Optional
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from dataclasses import dataclass
from enum import Enum
from .core import detect_peaks_troughs_enhanced, validate_impulse_wave_rules
from .core.validation import validate_wave_4_overlap, validate_diagonal_triangle, analyze_diagonal_trend_lines, analyze_diagonal_volume_pattern, analyze_diagonal_wave_alternation
from .plotters import plot_impulse
from .core.impulse import find_elliott_wave_pattern_enhanced
from .core.corrective import detect_corrective_patterns
from .core.models import WaveType, WaveDegree, CorrectiveType, FibonacciRatios, WaveProperties, ElliottWave
from .core.position import detect_current_wave_position_enhanced
import matplotlib.cm as cm


class WaveDegree(Enum):
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"


class WaveType(Enum):
    IMPULSE = "impulse"
    DIAGONAL_LEADING = "diagonal_leading"
    DIAGONAL_ENDING = "diagonal_ending"
    ZIGZAG = "zigzag"
    FLAT_REGULAR = "flat_regular"
    FLAT_IRREGULAR = "flat_irregular"
    FLAT_RUNNING = "flat_running"
    TRIANGLE_CONTRACTING = "triangle_contracting"
    TRIANGLE_EXPANDING = "triangle_expanding"
    DOUBLE_CORRECTION = "double_correction"
    TRIPLE_CORRECTION = "triple_correction"


class CorrectiveType(Enum):
    SHARP = "sharp"
    FLAT = "flat"
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class WaveProperties:
    """Properties of an Elliott Wave"""
    start_idx: int
    end_idx: int
    price_start: float
    price_end: float
    duration: int
    volume_avg: float
    internal_structure: List[int]
    fibonacci_relationships: Dict[str, float]


class ElliottWave:
    def __init__(self, wave_points: np.ndarray, wave_type: WaveType, degree: WaveDegree, 
                 confidence: float, properties: WaveProperties):
        self.wave_points = wave_points
        self.wave_type = wave_type
        self.degree = degree
        self.confidence = confidence
        self.properties = properties
        self.sub_waves = []


def validate_wave_4_overlap(df: pd.DataFrame, wave_points: np.ndarray, 
                          column: str = 'close') -> Dict[str, Any]:
    """
    Properly validate Wave 4 overlap rule according to Elliott Wave theory.
    Wave 4 cannot overlap with Wave 1 territory in impulse waves.
    Overlap is ONLY allowed in diagonal triangles.
    """
    if len(wave_points) < 5:
        return {"valid": True, "reason": "insufficient_waves"}
    
    prices = df[column].iloc[wave_points].values
    
    # Define Wave 1 territory (from start to end of Wave 1)
    wave_1_start = prices[0]
    wave_1_end = prices[1]
    wave_1_high = max(wave_1_start, wave_1_end)
    wave_1_low = min(wave_1_start, wave_1_end)
    
    # Wave 4 end point
    wave_4_end = prices[4]
    
    # Check for overlap
    has_overlap = wave_1_low <= wave_4_end <= wave_1_high
    
    result = {
        "valid": True,
        "has_overlap": has_overlap,
        "wave_1_territory": (wave_1_low, wave_1_high),
        "wave_4_end": wave_4_end,
        "is_diagonal": False
    }
    
    if has_overlap:
        # Overlap detected - check if this could be a valid diagonal triangle
        diagonal_validation = validate_diagonal_triangle(df, wave_points, column)
        result.update(diagonal_validation)
        
        if diagonal_validation['is_valid_diagonal']:
            result["valid"] = True
            result["is_diagonal"] = True
            result["reason"] = "valid_diagonal_triangle"
        else:
            result["valid"] = False
            result["reason"] = "invalid_overlap_not_diagonal"
    else:
        result["reason"] = "no_overlap_clean_impulse"
    
    return result


def validate_diagonal_triangle(df: pd.DataFrame, wave_points: np.ndarray, 
                             column: str = 'close') -> Dict[str, Any]:
    """
    Comprehensive validation of diagonal triangle patterns.
    Diagonals have specific characteristics:
    1. Converging or diverging trend lines
    2. 3-3-3-3-3 internal structure (each wave subdivides into 3)
    3. Volume diminishment through the pattern
    4. Wave 4 can overlap Wave 1
    5. All waves move in the same direction as the overall trend
    """
    if len(wave_points) < 5:
        return {"is_valid_diagonal": False, "reason": "insufficient_points"}
    
    prices = df[column].iloc[wave_points].values
    indices = wave_points
    
    validation_result = {
        "is_valid_diagonal": False,
        "trend_lines_converge": False,
        "volume_diminishes": False,
        "wave_alternation": False,
        "confidence": 0.0
    }
    
    try:
        # 1. Trend line convergence/divergence check
        trend_analysis = analyze_diagonal_trend_lines(prices, indices)
        validation_result.update(trend_analysis)
        
        if not trend_analysis['valid_trend_lines']:
            validation_result["reason"] = "invalid_trend_lines"
            return validation_result
        
        # 2. Volume analysis (if available)
        if 'volume' in df.columns:
            volume_analysis = analyze_diagonal_volume_pattern(df, wave_points)
            validation_result.update(volume_analysis)
        else:
            validation_result["volume_diminishes"] = True  # Assume valid if no volume data
        
        # 3. Wave alternation and character analysis
        alternation_analysis = analyze_diagonal_wave_alternation(df, wave_points, column)
        validation_result.update(alternation_analysis)
        
        # 4. Overall validation
        validation_score = 0
        if trend_analysis['valid_trend_lines']: validation_score += 0.4
        if validation_result['volume_diminishes']: validation_score += 0.3
        if validation_result['wave_alternation']: validation_score += 0.3
        
        validation_result["confidence"] = validation_score
        validation_result["is_valid_diagonal"] = validation_score >= 0.6
        
        if validation_result["is_valid_diagonal"]:
            validation_result["reason"] = "valid_diagonal_pattern"
        else:
            validation_result["reason"] = f"insufficient_diagonal_characteristics_{validation_score:.2f}"
    
    except Exception as e:
        validation_result["reason"] = f"analysis_error_{str(e)}"
    
    return validation_result


def validate_wave_directions(waves: Dict[int, float]) -> Tuple[bool, float]:
    """
    Validate that wave directions follow Elliott Wave rules.
    Waves 1, 3, 5 should be in same direction (impulse direction)
    Waves 2, 4 should be opposite to impulse direction
    """
    try:
        # Determine overall trend direction from Wave 1
        trend_up = waves[1] > 0
        
        confidence = 0.0
        
        # Wave 3 should be in same direction as Wave 1
        if (waves[3] > 0) == trend_up:
            confidence += 0.3
        else:
            return False, 0.0
        
        # Wave 2 should be opposite to trend
        if (waves[2] > 0) != trend_up:
            confidence += 0.2
        else:
            return False, 0.0
            
        # Wave 4 should be opposite to trend
        if (waves[4] > 0) != trend_up:
            confidence += 0.2
        else:
            return False, 0.0
        
        # Wave 5 should be in same direction as trend (if present)
        if waves[5] != 0:
            if (waves[5] > 0) == trend_up:
                confidence += 0.3
            else:
                return False, 0.0
        
        return True, confidence
    
    except Exception:
        return False, 0.0


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
            elif wave_3_ratio >= 1.618:  # At least 1.618 extension
                confidence += 0.15
        
        # Wave 4 retracement of Wave 3
        if waves[3] != 0:
            wave_4_ratio = abs(waves[4] / waves[3])
            if any(abs(wave_4_ratio - ratio) < 0.05 for ratio in fib.RETRACEMENTS[:3]):  # 23.6%, 38.2%, 50%
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
                    # Wave 2 vs Wave 4 time relationship
                    if durations[1] != 0:
                        time_ratio = durations[3] / durations[1]
                        if any(abs(time_ratio - ratio) < 0.2 for ratio in fib.TIME_RATIOS):
                            confidence += 0.1
            except:
                pass  # Time analysis failed, skip
    
    except Exception:
        pass  # Fibonacci analysis failed, return current confidence
    
    return min(confidence, 0.8)  # Cap at 0.8


def validate_volume_patterns(df: pd.DataFrame, wave_points: np.ndarray) -> float:
    """
    Validate Elliott Wave volume patterns.
    - Wave 3 should have highest volume in the sequence
    - Wave 5 often shows volume divergence (declining volume with price advance)
    - Corrective waves (2, 4) typically have lower volume
    """
    if 'volume' not in df.columns or len(df['volume'].dropna()) == 0:
        return 0.5  # Neutral score if no volume data
    
    try:
        volumes = df['volume'].iloc[wave_points].values
        if len(volumes) < 5:
            return 0.5
        
        confidence = 0.0
        
        # Wave 3 should have highest or second highest volume
        wave_3_volume = volumes[3]
        max_volume = max(volumes)
        if wave_3_volume == max_volume:
            confidence += 0.4
        elif wave_3_volume >= max_volume * 0.85:  # Within 85% of max
            confidence += 0.3
        
        # Corrective waves should have lower volume than impulse waves
        impulse_volumes = [volumes[1], volumes[3]]  # Waves 1, 3
        corrective_volumes = [volumes[2], volumes[4]]  # Waves 2, 4
        
        if len(volumes) > 5:
            impulse_volumes.append(volumes[5])  # Wave 5
        
        avg_impulse_volume = np.mean(impulse_volumes)
        avg_corrective_volume = np.mean(corrective_volumes)
        
        if avg_impulse_volume > avg_corrective_volume:
            confidence += 0.3
        
        # Wave 5 volume divergence check (if Wave 5 exists)
        if len(volumes) > 5:
            wave_1_volume = volumes[1]
            wave_5_volume = volumes[5]
            
            # Wave 5 volume should be less than Wave 1 (divergence)
            if wave_5_volume < wave_1_volume:
                confidence += 0.3
        
        return min(confidence, 1.0)
    
    except Exception:
        return 0.5


def validate_alternation_principle(df: pd.DataFrame, wave_points: np.ndarray, 
                                 column: str = 'close') -> float:
    """
    Validate the alternation principle between waves 2 and 4.
    If Wave 2 is sharp and short, Wave 4 should be flat and long (and vice versa).
    """
    if len(wave_points) < 5:
        return 0.5
    
    try:
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]
        
        # Calculate characteristics of waves 2 and 4
        wave_2_magnitude = abs(prices[2] - prices[1])
        wave_4_magnitude = abs(prices[4] - prices[3])
        
        wave_2_duration = (dates[2] - dates[1]).days
        wave_4_duration = (dates[4] - dates[3]).days
        
        # Calculate retracement percentages
        wave_1_magnitude = abs(prices[1] - prices[0])
        wave_3_magnitude = abs(prices[3] - prices[2])
        
        wave_2_retracement = wave_2_magnitude / wave_1_magnitude if wave_1_magnitude != 0 else 0
        wave_4_retracement = wave_4_magnitude / wave_3_magnitude if wave_3_magnitude != 0 else 0
        
        confidence = 0.0
        
        # Alternation in depth: one deep, one shallow
        deep_shallow_alternation = abs(wave_2_retracement - wave_4_retracement) > 0.2
        if deep_shallow_alternation:
            confidence += 0.3
        
        # Alternation in duration: one short, one long
        if wave_2_duration > 0 and wave_4_duration > 0:
            duration_ratio = max(wave_2_duration, wave_4_duration) / min(wave_2_duration, wave_4_duration)
            if duration_ratio > 1.5:  # One is at least 50% longer
                confidence += 0.3
        
        # Alternation in character: sharp vs flat
        # Sharp = steep decline/advance, Flat = gradual movement
        wave_2_steepness = wave_2_magnitude / max(wave_2_duration, 1)
        wave_4_steepness = wave_4_magnitude / max(wave_4_duration, 1)
        
        if wave_2_steepness > 0 and wave_4_steepness > 0:
            steepness_ratio = max(wave_2_steepness, wave_4_steepness) / min(wave_2_steepness, wave_4_steepness)
            if steepness_ratio > 2.0:  # One is significantly steeper
                confidence += 0.4
        
        return min(confidence, 1.0)
    
    except Exception:
        return 0.5


def plot_subwaves_recursive(ax, df, subwaves, depth=1, column='close', max_depth=3):
    if not subwaves or depth > max_depth:
        return
    colors = cm.get_cmap('tab10')
    for i, sub in enumerate(subwaves):
        if sub and 'sequence' in sub:
            points = [pt[0] for pt in sub['sequence']]
            if len(points) > 1:
                ax.plot(df.index[points], df[column].iloc[points],
                        marker='o', linestyle='-', linewidth=max(0.5, 2-depth*0.5),
                        color=colors((depth*2+i)%10), alpha=0.7,
                        label=f'Subwave d{depth}.{i+1}')
            # Recurse into deeper subwaves
            plot_subwaves_recursive(ax, df, sub.get('subwaves', []), depth+1, column, max_depth)


def plot_elliott_wave_analysis_enhanced(df: pd.DataFrame, wave_data: Dict[str, Any],
                                      column: str = 'close', title: str = 'Elliott Wave Analysis',
                                      ax=None, show_validation_details: bool = True):
    """
    Enhanced plotting with comprehensive validation details and improved visualization.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure
    
    # Extract wave data
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    corrective_wave = wave_data.get('corrective_wave', np.array([]))
    peaks = wave_data.get('peaks', np.array([]))
    troughs = wave_data.get('troughs', np.array([]))
    confidence = wave_data.get('confidence', 0.0)
    wave_type = wave_data.get('wave_type', 'unknown')
    validation_details = wave_data.get('validation_details', {})
    
    ax.clear()
    
    # Plot price line with enhanced styling
    ax.plot(df.index, df[column], label='Price', alpha=0.8, color='black', linewidth=1.5)
    
    # Plot peaks and troughs with different markers
    if len(peaks) > 0:
        ax.scatter(df.index[peaks], df[column].iloc[peaks], 
                  c='red', marker='^', s=50, alpha=0.7, label='Peaks', zorder=5)
    
    if len(troughs) > 0:
        ax.scatter(df.index[troughs], df[column].iloc[troughs], 
                  c='green', marker='v', s=50, alpha=0.7, label='Troughs', zorder=5)
    
    # Plot impulse wave with enhanced visualization
    if len(impulse_wave) > 0:
        # Main impulse wave line
        ax.plot(df.index[impulse_wave], df[column].iloc[impulse_wave], 
                'bo-', label=f'Impulse Wave ({wave_type})', markersize=10, 
                linewidth=3, alpha=0.9, zorder=10)
        
        # Add wave labels with improved positioning
        for idx, wp in enumerate(impulse_wave):
            if 0 <= wp < len(df.index):
                date = df.index[wp]
                price = df[column].iloc[wp]
                
                # Determine label position based on wave direction
                if idx > 0:
                    prev_price = df[column].iloc[impulse_wave[idx-1]]
                    offset_y = 20 if price > prev_price else -30
                else:
                    offset_y = 20
                
                ax.annotate(f'W{idx+1}', (date, price), 
                           textcoords="offset points", xytext=(0, offset_y), 
                           ha='center', fontsize=12, fontweight='bold', color='blue',
                           bbox=dict(boxstyle="round,pad=0.4", fc="lightblue", alpha=0.9, edgecolor='blue'),
                           zorder=15)
        
        # Add Fibonacci levels if complete 5-wave pattern
        if len(impulse_wave) >= 5:
            add_fibonacci_levels_enhanced(ax, df, impulse_wave, column)
        
        # Plot recursive subwaves if present
        subwaves = wave_data.get('subwaves', None)
        if subwaves:
            plot_subwaves_recursive(ax, df, subwaves, depth=1, column=column, max_depth=3)
    
    # Plot corrective wave with distinct styling
    if len(corrective_wave) > 0:
        ax.plot(df.index[corrective_wave], df[column].iloc[corrective_wave], 
                'mo-', label='Corrective Wave (A-B-C)', markersize=10, 
                linewidth=3, alpha=0.9, linestyle='--', zorder=10)
        
        # Annotate corrective wave points
        corrective_labels = ['A', 'B', 'C', 'D', 'E']
        for idx, cp in enumerate(corrective_wave):
            if 0 <= cp < len(df.index) and idx < len(corrective_labels):
                date = df.index[cp]
                price = df[column].iloc[cp]
                
                ax.annotate(corrective_labels[idx], (date, price), 
                           textcoords="offset points", xytext=(0, -35), 
                           ha='center', fontsize=12, fontweight='bold', color='magenta',
                           bbox=dict(boxstyle="round,pad=0.4", fc="lightpink", alpha=0.9, edgecolor='magenta'),
                           zorder=15)
    
    # Add comprehensive information panel
    info_panel = create_info_panel(wave_data, validation_details, show_validation_details)
    ax.text(0.02, 0.98, info_panel, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.95, 
                     edgecolor='gray', linewidth=1), zorder=20)
    
    # Enhanced title with key information
    confidence_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
    title_text = f"{title}\n{wave_type.replace('_', ' ').title()} Pattern (Confidence: {confidence:.2f})"
    ax.set_title(title_text, fontsize=14, fontweight='bold', color=confidence_color, pad=20)
    
    # Styling improvements
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.80), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Set reasonable axis limits with padding
    if len(df) > 0:
        ax.set_xlim(df.index.min(), df.index.max())
        price_range = df[column].max() - df[column].min()
        ax.set_ylim(df[column].min() - price_range * 0.05, 
                   df[column].max() + price_range * 0.1)
    
    # Improved date formatting
    fix_date_labels_enhanced(ax, df)
    
    fig.tight_layout()
    if ax is None:
        plt.show()


def add_fibonacci_levels_enhanced(ax, df: pd.DataFrame, impulse_wave: np.ndarray, column: str):
    """
    Add comprehensive Fibonacci levels with proper labeling and color coding.
    """
    try:
        # Ensure impulse_wave indices are valid
        valid_indices = [idx for idx in impulse_wave if 0 <= idx < len(df)]
        if len(valid_indices) < 5:
            return
        
        prices = df[column].iloc[valid_indices]
        dates = df.index[valid_indices]
        
        if len(prices) < 5:
            return
        
        # Wave 1 Fibonacci retracement levels for Wave 2
        wave_1_start = prices.iloc[0]
        wave_1_end = prices.iloc[1]
        wave_1_range = wave_1_end - wave_1_start
        
        fib_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        colors = ['#FFD700', '#FFA500', '#FF6347', '#FF4500', '#FF0000']
        
        for i, fib in enumerate(fib_retracements):
            level = wave_1_end - (wave_1_range * fib)
            ax.axhline(level, color=colors[i], linestyle=':', alpha=0.6, linewidth=1.5)
            
            # Use valid date index for text positioning
            if len(dates) > 2:
                text_date = dates.iloc[2]
                ax.text(text_date, level, f'{fib:.1%}', 
                       fontsize=8, color=colors[i], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Wave 3 extension levels
        wave_1_length = abs(wave_1_range)
        wave_3_start = prices.iloc[2]
        
        extensions = [1.618, 2.618, 4.236]
        ext_colors = ['#9370DB', '#8A2BE2', '#4B0082']
        
        for i, ext in enumerate(extensions):
            if wave_1_range > 0:  # Uptrend
                level = wave_3_start + wave_1_length * ext
            else:  # Downtrend
                level = wave_3_start - wave_1_length * ext
            
            # Check if level is within reasonable range
            price_min = df[column].min()
            price_max = df[column].max()
            price_range = price_max - price_min
            
            if (price_min - price_range * 0.5) <= level <= (price_max + price_range * 0.5):
                ax.axhline(level, color=ext_colors[i], linestyle='-.', alpha=0.5, linewidth=1)
                
                # Use valid date index for text positioning
                if len(dates) > 3:
                    text_date = dates.iloc[3]
                    ax.text(text_date, level, f'{ext:.1f}x', 
                           fontsize=8, color=ext_colors[i], fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
    except Exception as e:
        print(f"Error adding Fibonacci levels: {e}")


def create_info_panel(wave_data: Dict[str, Any], validation_details: Dict[str, Any], 
                     show_details: bool = True) -> str:
    """
    Create a comprehensive information panel for the Elliott Wave analysis.
    """
    confidence = wave_data.get('confidence', 0.0)
    wave_type = wave_data.get('wave_type', 'unknown')
    
    panel_lines = [
        f"ELLIOTT WAVE ANALYSIS",
        f"{'='*25}",
        f"Pattern Type: {wave_type.replace('_', ' ').title()}",
        f"Confidence:  {confidence:.2f} ({get_confidence_description(confidence)})",
    ]
    
    if show_details and validation_details:
        panel_lines.append(f"{'='*25}")
        panel_lines.append("VALIDATION DETAILS:")
        
        # Add key validation metrics
        if 'wave_2_retracement' in validation_details:
            retr = validation_details['wave_2_retracement']
            panel_lines.append(f"Wave 2 Retracement: {retr:.1%}")
        
        if 'wave_lengths' in validation_details:
            lengths = validation_details['wave_lengths']
            panel_lines.append(f"Wave Lengths: {[f'{l:.1f}' for l in lengths]}")
        
        if 'fibonacci' in validation_details:
            fib_score = validation_details['fibonacci']
            panel_lines.append(f"Fibonacci Score: {fib_score:.2f}")
        
        if 'volume' in validation_details:
            vol_score = validation_details['volume']
            panel_lines.append(f"Volume Pattern: {vol_score:.2f}")
        
        if 'alternation' in validation_details:
            alt_score = validation_details['alternation']
            panel_lines.append(f"Alternation: {alt_score:.2f}")
        
        # Add diagonal information if applicable
        if validation_details.get('is_diagonal', False):
            panel_lines.append("⚠️  DIAGONAL PATTERN")
            panel_lines.append("   (Overlap Allowed)")
    
    return '\n'.join(panel_lines)


def get_confidence_description(confidence: float) -> str:
    """
    Get a textual description of the confidence level.
    """
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.6:
        return "High"
    elif confidence >= 0.4:
        return "Moderate"
    elif confidence >= 0.2:
        return "Low"
    else:
        return "Very Low"


def fix_date_labels_enhanced(ax, df: pd.DataFrame):
    """
    Enhanced date label formatting with better spacing and rotation.
    """
    try:
        if len(df) == 0:
            return
            
        start_date = df.index.min()
        end_date = df.index.max()
        date_range = (end_date - start_date).days
        
        # Dynamic date formatting based on range
        if date_range <= 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        elif date_range <= 90:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif date_range <= 365:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        elif date_range <= 1825:  # 5 years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        # Rotate and align labels
        for label in ax.xaxis.get_majorticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        
        # Limit number of ticks to prevent overcrowding
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
        
        # Add some margin
        ax.margins(x=0.02)
        
    except Exception as e:
        print(f"Error fixing date labels: {e}")


# Main detection function that replaces the original
def detect_elliott_wave_complete(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Main function for comprehensive Elliott Wave detection with all fixes applied.
    This replaces the original function with proper rule validation.
    """
    return find_elliott_wave_pattern_enhanced(df, column)


# Legacy compatibility functions
def validate_fibonacci(df: pd.DataFrame, wave_points: np.ndarray, column: str = 'close', 
                      min_length: int = 5) -> float:
    """Legacy function - redirects to enhanced validation."""
    if len(wave_points) < 5:
        return 0.0
    
    try:
        is_valid, confidence, _ = validate_impulse_wave_rules(df, wave_points, column)
        return confidence if is_valid else 0.0
    except:
        return 0.0


def refined_elliott_wave_suggestion(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                                  min_wave_length: int = 5, max_wave_length: int = 8,
                                  min_price_change: float = None, column: str = 'close') -> Tuple[np.ndarray, float]:
    """Legacy function - uses enhanced detection."""
    try:
        result = find_elliott_wave_pattern_enhanced(df, column, min_wave_length, max_wave_length)
        return result['impulse_wave'], result['confidence']
    except:
        return np.array([]), 0.0


def detect_peaks_troughs(df: pd.DataFrame, column: str = 'close', distance: int = None, 
                        prominence: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function - redirects to enhanced detection."""
    return detect_peaks_troughs_enhanced(df, column)


def plot_peaks_troughs(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                      wave_points: np.ndarray = None, column: str = 'close',
                      title: str = 'Price with Peaks, Troughs, and Elliott Waves'):
    """Legacy plotting function."""
    wave_data = {
        'impulse_wave': wave_points if wave_points is not None else np.array([]),
        'corrective_wave': np.array([]),
        'peaks': peaks,
        'troughs': troughs,
        'confidence': 0.5,
        'wave_type': 'legacy',
        'validation_details': {}
    }
    plot_elliott_wave_analysis_enhanced(df, wave_data, column, title, show_validation_details=False)


def detect_current_wave_position(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """Enhanced current wave position detection."""
    return detect_current_wave_position_enhanced(df, column)


# Alias for the main plotting function
def plot_elliott_wave_analysis(df: pd.DataFrame, wave_data: Dict[str, Any], column: str = 'close',
                              title: str = 'Elliott Wave Analysis', ax=None):
    """Main plotting function with all enhancements."""
    plot_elliott_wave_analysis_enhanced(df, wave_data, column, title, ax, show_validation_details=True)