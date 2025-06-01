import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from ..core import detect_peaks_troughs_enhanced, validate_impulse_wave_rules
from .validation import validate_wave_4_overlap, validate_diagonal_triangle, analyze_diagonal_trend_lines, analyze_diagonal_volume_pattern, analyze_diagonal_wave_alternation
from enum import Enum
from ..core.models import WaveType

# (Add WaveType enum or import if needed)


def find_elliott_wave_pattern_enhanced(df: pd.DataFrame, column: str = 'close', 
                                     min_points: int = 5, max_points: int = 8) -> Dict[str, Any]:
    """
    Enhanced Elliott Wave pattern detection with comprehensive validation.
    """
    peaks, troughs = detect_peaks_troughs_enhanced(df, column)
    if len(peaks) == 0 or len(troughs) == 0:
        return {
            "impulse_wave": np.array([]),
            "corrective_wave": np.array([]),
            "peaks": peaks,
            "troughs": troughs,
            "confidence": 0.0,
            "wave_type": "no_pattern",
            "validation_details": {"error": "no_peaks_troughs"}
        }
    best_impulse = find_best_impulse_wave(df, peaks, troughs, column, min_points, max_points)
    corrective_pattern = np.array([])
    if len(best_impulse["wave_points"]) >= 5:
        corrective_start = best_impulse["wave_points"][-1]
        # Note: detect_corrective_patterns should be imported from corrective.py
        from .corrective import detect_corrective_patterns
        corrective_data = detect_corrective_patterns(df, corrective_start, column)
        if corrective_data["confidence"] > 0.5:
            corrective_pattern = corrective_data["points"]
    return {
        "impulse_wave": best_impulse["wave_points"],
        "corrective_wave": corrective_pattern,
        "peaks": peaks,
        "troughs": troughs,
        "confidence": best_impulse["confidence"],
        "wave_type": best_impulse["wave_type"],
        "validation_details": best_impulse["validation_details"]
    }

def find_best_impulse_wave(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray, 
                          column: str = 'close', min_points: int = 5, 
                          max_points: int = 8) -> Dict[str, Any]:
    """
    Find the best impulse wave pattern using comprehensive validation.
    """
    candidates = []
    for start_type in ['trough', 'peak']:
        candidates.extend(
            generate_impulse_candidates(df, peaks, troughs, start_type, column, min_points, max_points)
        )
    if not candidates:
        return {
            "wave_points": np.array([]),
            "confidence": 0.0,
            "wave_type": "no_candidates",
            "validation_details": {"error": "no_valid_candidates"}
        }
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[0]

def build_wave_sequence(df: pd.DataFrame, labeled_points: List[Tuple[int, str]], start_idx: int, end_idx: int, wave_type: str, depth: int = 0, max_depth: int = 3) -> dict:
    """
    Recursively build a wave sequence and validate internal structure.
    Returns a dict with 'sequence' and 'subwaves' if valid, else None.
    """
    if depth > max_depth:
        return None
    # Find subwaves: for impulse expect 5, for corrective expect 3
    expected_count = 5 if wave_type == 'impulse' else 3
    # Find candidate subwave points within [start_idx, end_idx]
    sub_seq = []
    sub_labeled = [pt for pt in labeled_points if start_idx <= pt[0] <= end_idx]
    if len(sub_labeled) < expected_count:
        return None
    # Try to find alternating sequence
    seq = [sub_labeled[0]]
    last_type = sub_labeled[0][1]
    for pt in sub_labeled[1:]:
        if pt[1] != last_type:
            seq.append(pt)
            last_type = pt[1]
        if len(seq) == expected_count:
            break
    if len(seq) != expected_count:
        return None
    # Recursively validate internal structure
    subwaves = []
    for i in range(len(seq) - 1):
        sub_start = seq[i][0]
        sub_end = seq[i+1][0]
        # For impulse: 1,3,5 are impulse; 2,4 are corrective
        # For corrective: A is impulse, B/C are corrective
        if wave_type == 'impulse':
            sub_type = 'impulse' if i % 2 == 0 else 'corrective'
        else:
            sub_type = 'impulse' if i == 0 else 'corrective'
        # Only recurse if segment is long enough
        if sub_end - sub_start > 2 and depth + 1 <= max_depth:
            sub_result = build_wave_sequence(df, labeled_points, sub_start, sub_end, sub_type, depth + 1, max_depth)
            subwaves.append(sub_result)
        else:
            subwaves.append(None)
    return {'sequence': seq, 'subwaves': subwaves}

def generate_impulse_candidates(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                              start_type: str, column: str = 'close', 
                              min_points: int = 5, max_points: int = 8, max_depth: int = 3) -> List[Dict[str, Any]]:
    """
    Generate impulse wave candidates with recursive validation.
    """
    candidates = []
    if start_type == 'trough':
        start_points = troughs
        expected_sequence = ['trough', 'peak', 'trough', 'peak', 'trough', 'peak', 'trough', 'peak']
    else:
        start_points = peaks
        expected_sequence = ['peak', 'trough', 'peak', 'trough', 'peak', 'trough', 'peak', 'trough']
    labeled_points = (
        [(int(p), 'peak') for p in peaks] + 
        [(int(t), 'trough') for t in troughs]
    )
    labeled_points.sort(key=lambda x: x[0])
    cutoff_date = df.index.max() - pd.DateOffset(years=5)
    recent_indices = df.index >= cutoff_date
    valid_start_points = [
        sp for sp in start_points 
        if sp < len(df) and sp in df[recent_indices].index.get_indexer(df.index)
    ]
    for start_idx in valid_start_points:
        # Find the furthest possible end_idx for this candidate
        for end_idx in range(start_idx + min_points - 1, min(start_idx + max_points, len(df))):
            result = build_wave_sequence(df, labeled_points, start_idx, end_idx, 'impulse', 0, max_depth)
            if result is not None:
                seq = result['sequence']
                if min_points <= len(seq) <= max_points:
                    wave_points = np.array([point[0] for point in seq])
                    is_valid, confidence, validation_details = validate_impulse_wave_rules(df, wave_points, column)
                    if is_valid:
                        wave_type = WaveType.IMPULSE
                        if validation_details.get('is_diagonal', False):
                            wave_type = WaveType.DIAGONAL_ENDING
                        candidates.append({
                            "wave_points": wave_points,
                            "confidence": confidence,
                            "wave_type": wave_type.value,
                            "validation_details": validation_details,
                            "subwaves": result['subwaves']
                        })
    return candidates 