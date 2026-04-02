import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, Dict, Any
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from enum import Enum
from src.analysis.core import detect_peaks_troughs_enhanced, validate_impulse_wave_rules
from src.analysis.core.validation import validate_wave_4_overlap, validate_diagonal_triangle, analyze_diagonal_trend_lines, analyze_diagonal_volume_pattern, analyze_diagonal_wave_alternation
from src.analysis.plotters.elliott import plot_elliott_wave_analysis, plot_elliott_wave_analysis_enhanced, plot_subwaves_recursive
from src.analysis.core.impulse import find_elliott_wave_pattern_enhanced
from src.analysis.core.corrective_patterns import detect_corrective_patterns
from src.analysis.core.position import detect_current_wave_position_enhanced
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from src.utils.common_utils import get_confidence_description


# WaveProperties and ElliottWave classes are imported from src.analysis.core.models at line 16
# Removed duplicate definitions to maintain single source of truth


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


def detect_current_wave_position(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """Enhanced current wave position detection."""
    return detect_current_wave_position_enhanced(df, column)