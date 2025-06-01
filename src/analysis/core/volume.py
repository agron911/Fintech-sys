import numpy as np
import pandas as pd
from typing import Dict, Any

def analyze_diagonal_volume_pattern(df: pd.DataFrame, wave_points: np.ndarray) -> Dict[str, Any]:
    """
    Analyze volume pattern in diagonal triangles. Volume should generally diminish through the pattern.
    """
    try:
        volumes = df['volume'].iloc[wave_points].values
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        volume_diminishes = volume_trend < 0
        mid_point = len(volumes) // 2
        first_half_avg = np.mean(volumes[:mid_point]) if mid_point > 0 else volumes[0]
        second_half_avg = np.mean(volumes[mid_point:]) if mid_point < len(volumes) else volumes[-1]
        volume_declining = first_half_avg > second_half_avg
        return {
            "volume_diminishes": volume_diminishes or volume_declining,
            "volume_trend_slope": volume_trend,
            "first_half_volume": first_half_avg,
            "second_half_volume": second_half_avg
        }
    except Exception as e:
        return {"volume_diminishes": False, "reason": f"volume_analysis_error_{str(e)}"}

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
        wave_3_volume = volumes[3]
        max_volume = max(volumes)
        if wave_3_volume == max_volume:
            confidence += 0.4
        elif wave_3_volume >= max_volume * 0.85:
            confidence += 0.3
        impulse_volumes = [volumes[1], volumes[3]]
        corrective_volumes = [volumes[2], volumes[4]]
        if len(volumes) > 5:
            impulse_volumes.append(volumes[5])
        avg_impulse_volume = np.mean(impulse_volumes)
        avg_corrective_volume = np.mean(corrective_volumes)
        if avg_impulse_volume > avg_corrective_volume:
            confidence += 0.3
        if len(volumes) > 5:
            wave_1_volume = volumes[1]
            wave_5_volume = volumes[5]
            if wave_5_volume < wave_1_volume:
                confidence += 0.3
        return min(confidence, 1.0)
    except Exception:
        return 0.5 