import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
from src.analysis.core import detect_peaks_troughs_enhanced, validate_impulse_wave_rules

logger = logging.getLogger(__name__)
from src.analysis.core.validation import (
    validate_elliott_wave_pattern,
    validate_wave_4_overlap,
    validate_diagonal_triangle,
    analyze_diagonal_trend_lines,
    analyze_diagonal_volume_pattern,
    analyze_diagonal_wave_alternation,
    ValidationConfig
)
from enum import Enum
from src.analysis.core.models import WaveType
import itertools
from src.analysis.core.flexible_sequence_builder import (
    FlexibleSequenceBuilder,
    build_wave_sequence_enhanced,
    build_subwaves_flexible
)
from src.analysis.core.intelligent_subwaves import generate_impulse_candidates_with_intelligent_subwaves, SubwaveAnalysisConfig
from src.analysis.core.wave_cache import WaveAnalysisCache
from .pattern_enhancement import enhance_wave_detection_with_complex_patterns
from src.analysis.core.utils import (
    calculate_time_overlap, get_pattern_direction, calculate_confidence,
    validate_pattern_relationships, merge_related_patterns, check_wave_phase_alignment
)

# Utility function to filter DataFrame by candlestick type
def filter_by_candlestick_type(df: pd.DataFrame, candlestick_type: str, date_col: str = None) -> pd.DataFrame:
    """
    Filter DataFrame by candlestick type with appropriate data ranges.
    """
    if date_col:
        date_series = pd.to_datetime(df[date_col])
    else:
        date_series = pd.to_datetime(df.index)
    df = df.copy().sort_index()
    # Increased data ranges for better pattern detection
    if candlestick_type == 'day':
        cutoff = date_series.max() - pd.DateOffset(years=5)  # Increased from 3 to 5 years for more patterns
    elif candlestick_type == 'week':
        cutoff = date_series.max() - pd.DateOffset(years=15)  # Increased from 10 to 15 years for more patterns
    elif candlestick_type == 'month':
        return df  # Use all available data
    else:
        raise ValueError(f"Unknown candlestick type: {candlestick_type}")
    return df[date_series >= cutoff]

def find_elliott_wave_patterns_advanced(df: pd.DataFrame, 
                                      column: str = 'close',
                                      candlestick_types: List[str] = None,
                                      max_patterns_per_timeframe: int = 3,
                                      pattern_relationships: bool = True) -> Dict[str, Any]:
    """
    Advanced multi-pattern detection with cross-timeframe validation
    
    Args:
        df: DataFrame with price data
        column: Column name for price data
        candlestick_types: List of timeframes to analyze ['day', 'week', 'month']
        max_patterns_per_timeframe: Maximum patterns to find per timeframe
        pattern_relationships: Whether to analyze relationships between patterns
    
    Returns:
        Dict containing patterns across all timeframes with relationships
    """
    if candlestick_types is None:
        candlestick_types = ['day', 'week', 'month']
    
    all_patterns = {}
    cross_timeframe_analysis = {}
    
    # Step 1: Detect patterns in each timeframe
    for timeframe in candlestick_types:
        logger.debug(f"Analyzing {timeframe} timeframe")
        
        timeframe_patterns = find_elliott_wave_patterns_multi_timeframe(
            df, column, timeframe, max_patterns_per_timeframe
        )
        
        all_patterns[timeframe] = timeframe_patterns
    
    # Step 2: Analyze pattern relationships across timeframes
    if pattern_relationships:
        cross_timeframe_analysis = analyze_pattern_relationships(all_patterns)
    
    # Step 3: Create unified pattern hierarchy
    pattern_hierarchy = create_pattern_hierarchy(all_patterns, cross_timeframe_analysis)
    
    # Step 4: Calculate composite confidence scores
    composite_patterns = calculate_composite_confidence(pattern_hierarchy)
    
    return {
        'patterns_by_timeframe': all_patterns,
        'pattern_relationships': cross_timeframe_analysis,
        'pattern_hierarchy': pattern_hierarchy,
        'composite_patterns': composite_patterns,
        'best_pattern': select_best_composite_pattern(composite_patterns)
    }


def analyze_pattern_relationships(patterns_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Analyze relationships between patterns across different timeframes
    """
    relationships = {
        'confirmations': [],
        'conflicts': [],
        'nested_patterns': [],
        'pattern_alignment': {}
    }
    
    timeframes = list(patterns_dict.keys())
    
    for i, tf1 in enumerate(timeframes):
        for j, tf2 in enumerate(timeframes[i+1:], i+1):
            patterns1 = patterns_dict[tf1].get('multiple_patterns', [])
            patterns2 = patterns_dict[tf2].get('multiple_patterns', [])
            
            for p1 in patterns1:
                for p2 in patterns2:
                    relationship = analyze_pattern_pair(p1, p2, tf1, tf2)
                    
                    if relationship['type'] == 'confirmation':
                        relationships['confirmations'].append(relationship)
                    elif relationship['type'] == 'conflict':
                        relationships['conflicts'].append(relationship)
                    elif relationship['type'] == 'nested':
                        relationships['nested_patterns'].append(relationship)
    
    # Calculate overall alignment score
    relationships['alignment_score'] = calculate_alignment_score(relationships)
    
    return relationships


def analyze_pattern_pair(pattern1: Dict, pattern2: Dict, 
                        timeframe1: str, timeframe2: str) -> Dict[str, Any]:
    """
    Analyze relationship between two patterns from different timeframes
    """
    # Check time overlap
    overlap_ratio = calculate_time_overlap(
        pattern1['start_date'], pattern1['end_date'],
        pattern2['start_date'], pattern2['end_date']
    )
    
    # Check directional agreement
    direction1 = get_pattern_direction(pattern1)
    direction2 = get_pattern_direction(pattern2)
    directional_agreement = direction1 == direction2
    
    # Check wave phase alignment
    phase_alignment = check_wave_phase_alignment(pattern1, pattern2)
    
    # Determine relationship type
    if overlap_ratio > 0.7 and directional_agreement:
        relationship_type = 'confirmation'
        strength = min(pattern1['confidence'], pattern2['confidence']) * overlap_ratio
    elif overlap_ratio > 0.5 and not directional_agreement:
        relationship_type = 'conflict'
        strength = -abs(pattern1['confidence'] - pattern2['confidence'])
    elif is_pattern_nested(pattern1, pattern2, timeframe1, timeframe2):
        relationship_type = 'nested'
        strength = (pattern1['confidence'] + pattern2['confidence']) / 2
    else:
        relationship_type = 'independent'
        strength = 0
    
    return {
        'type': relationship_type,
        'pattern1': {'timeframe': timeframe1, 'confidence': pattern1['confidence']},
        'pattern2': {'timeframe': timeframe2, 'confidence': pattern2['confidence']},
        'overlap_ratio': overlap_ratio,
        'directional_agreement': directional_agreement,
        'direction': direction1,  # Actual direction ('bullish'/'bearish'/'neutral')
        'phase_alignment': phase_alignment,
        'strength': strength
    }


def create_pattern_hierarchy(patterns_dict: Dict, relationships: Dict) -> Dict[str, Any]:
    """
    Create a hierarchical structure of patterns based on timeframe and relationships
    """
    hierarchy = {
        'primary': None,
        'supporting': [],
        'conflicting': [],
        'independent': []
    }
    
    # Find patterns with highest confirmation count
    confirmation_counts = {}
    
    for confirmation in relationships.get('confirmations', []):
        p1_key = f"{confirmation['pattern1']['timeframe']}_0"
        p2_key = f"{confirmation['pattern2']['timeframe']}_0"
        
        confirmation_counts[p1_key] = confirmation_counts.get(p1_key, 0) + 1
        confirmation_counts[p2_key] = confirmation_counts.get(p2_key, 0) + 1
    
    # Select primary pattern (highest confirmations and confidence)
    best_score = 0
    for timeframe, data in patterns_dict.items():
        if data.get('multiple_patterns'):
            pattern = data['multiple_patterns'][0]
            key = f"{timeframe}_0"
            confirmations = confirmation_counts.get(key, 0)
            score = pattern['confidence'] * (1 + confirmations * 0.2)
            
            if score > best_score:
                best_score = score
                hierarchy['primary'] = {
                    'pattern': pattern,
                    'timeframe': timeframe,
                    'confirmations': confirmations
                }
    
    # Categorize other patterns
    if hierarchy['primary']:
        primary_tf = hierarchy['primary']['timeframe']
        primary_pattern = hierarchy['primary']['pattern']
        
        for timeframe, data in patterns_dict.items():
            for i, pattern in enumerate(data.get('multiple_patterns', [])):
                if timeframe == primary_tf and i == 0:
                    continue  # Skip primary pattern
                
                # Check relationship with primary
                rel = find_relationship_with_primary(
                    pattern, primary_pattern, timeframe, primary_tf, relationships
                )
                
                if rel == 'supporting':
                    hierarchy['supporting'].append({
                        'pattern': pattern,
                        'timeframe': timeframe
                    })
                elif rel == 'conflicting':
                    hierarchy['conflicting'].append({
                        'pattern': pattern,
                        'timeframe': timeframe
                    })
                else:
                    hierarchy['independent'].append({
                        'pattern': pattern,
                        'timeframe': timeframe
                    })
    
    return hierarchy


def calculate_composite_confidence(hierarchy: Dict) -> List[Dict[str, Any]]:
    """
    Calculate composite confidence scores considering pattern relationships
    """
    composites = []
    
    if not hierarchy.get('primary'):
        return composites
    
    # Primary pattern composite
    primary = hierarchy['primary']
    primary_confidence = primary['pattern']['confidence']
    
    # Boost confidence based on supporting patterns
    support_boost = sum(
        p['pattern']['confidence'] * 0.1 
        for p in hierarchy.get('supporting', [])
    )
    
    # Reduce confidence based on conflicts
    conflict_penalty = sum(
        p['pattern']['confidence'] * 0.05 
        for p in hierarchy.get('conflicting', [])
    )
    
    composite_confidence = min(
        primary_confidence + support_boost - conflict_penalty,
        0.95  # Cap at 95%
    )
    
    composites.append({
        'base_pattern': primary,
        'composite_confidence': composite_confidence,
        'support_count': len(hierarchy.get('supporting', [])),
        'conflict_count': len(hierarchy.get('conflicting', [])),
        'pattern_group': 'primary'
    })
    
    # Create alternative composites from independent patterns
    for independent in hierarchy.get('independent', []):
        composites.append({
            'base_pattern': independent,
            'composite_confidence': independent['pattern']['confidence'] * 0.8,
            'support_count': 0,
            'conflict_count': 0,
            'pattern_group': 'alternative'
        })
    
    return sorted(composites, key=lambda x: x['composite_confidence'], reverse=True)


# Helper functions
def is_pattern_nested(pattern1: Dict, pattern2: Dict, tf1: str, tf2: str) -> bool:
    """Check if one pattern is nested within another"""
    # Larger timeframe should contain smaller
    tf_order = {'day': 1, 'week': 7, 'month': 30}
    
    if tf_order.get(tf1, 1) > tf_order.get(tf2, 1):
        larger, smaller = pattern1, pattern2
    else:
        larger, smaller = pattern2, pattern1
    
    # Check if smaller pattern is contained within larger
    return (larger['start_date'] <= smaller['start_date'] and 
            larger['end_date'] >= smaller['end_date'])


def calculate_alignment_score(relationships: Dict) -> float:
    """Calculate overall pattern alignment score"""
    confirmations = len(relationships.get('confirmations', []))
    conflicts = len(relationships.get('conflicts', []))
    
    if confirmations + conflicts == 0:
        return 0.5
    
    return confirmations / (confirmations + conflicts)


def select_best_composite_pattern(composites: List[Dict]) -> Dict[str, Any]:
    """Select the best composite pattern for trading"""
    if not composites:
        return {'error': 'No composite patterns found'}
    
    # Weight factors: confidence, recency, support
    best_pattern = composites[0]
    best_score = 0
    
    for composite in composites:
        base_pattern = composite['base_pattern']['pattern']
        
        score = (
            composite['composite_confidence'] * 0.5 +
            base_pattern.get('recency_score', 0) * 0.3 +
            (composite['support_count'] / 10) * 0.2
        )
        
        if score > best_score:
            best_score = score
            best_pattern = composite
    
    return best_pattern


def find_relationship_with_primary(pattern: Dict, primary_pattern: Dict, 
                                 pattern_tf: str, primary_tf: str, 
                                 relationships: Dict) -> str:
    """Find relationship between a pattern and the primary pattern"""
    # Check if there's a direct relationship in the relationships dict
    for confirmation in relationships.get('confirmations', []):
        if (confirmation['pattern1']['timeframe'] == pattern_tf and 
            confirmation['pattern2']['timeframe'] == primary_tf):
            return 'supporting'
    
    for conflict in relationships.get('conflicts', []):
        if (conflict['pattern1']['timeframe'] == pattern_tf and 
            conflict['pattern2']['timeframe'] == primary_tf):
            return 'conflicting'
    
    return 'independent'


def find_elliott_wave_patterns_multi_timeframe(df: pd.DataFrame, 
                                             column: str = 'close',
                                             candlestick_type: str = 'day',
                                             max_patterns: int = 5) -> Dict[str, Any]:
    """
    Find multiple Elliott Wave patterns across different timeframes.
    
    Args:
        df: DataFrame with price data
        column: Column name for price data
        candlestick_type: Type of candlestick ('day', 'week', 'month')
        max_patterns: Maximum number of patterns to return
        
    Returns:
        Dict containing multiple patterns and their details
    """
    # Filter data based on candlestick type
    filtered_df = filter_by_candlestick_type(df, candlestick_type)
    
    # Detect peaks and troughs
    peaks, troughs = detect_peaks_troughs_enhanced(filtered_df, column)
    
    # Find best impulse wave
    best_wave = find_best_impulse_wave(filtered_df, peaks, troughs, column)
    
    # Initialize result with primary pattern
    result = {
        "impulse_wave": best_wave.get("wave_points", np.array([])),
        "confidence": best_wave.get("confidence", 0.0),
        "wave_type": best_wave.get("wave_type", "no_pattern"),
        "validation_details": best_wave.get("validation_details", {}),
        "multiple_patterns": []
    }
    
    # Add primary pattern to multiple_patterns
    if len(best_wave.get("wave_points", [])) >= 5:
        primary_pattern = {
            "points": best_wave["wave_points"],
            "confidence": best_wave["confidence"],
            "wave_type": best_wave["wave_type"],
            "time_frame": candlestick_type,
            "recency_score": 1.0,  # Primary pattern is most recent
            "start_date": filtered_df.index[best_wave["wave_points"][0]],
            "end_date": filtered_df.index[best_wave["wave_points"][-1]],
            "price_range": (
                filtered_df[column].iloc[best_wave["wave_points"]].min(),
                filtered_df[column].iloc[best_wave["wave_points"]].max()
            )
        }
        result["multiple_patterns"].append(primary_pattern)
    
    # Add alternative patterns
    if len(best_wave.get("alternatives", [])) > 0:
        for alt in best_wave["alternatives"][:max_patterns-1]:
            if len(alt.get("wave_points", [])) >= 5:
                alt_pattern = {
                    "points": alt["wave_points"],
                    "confidence": alt["confidence"],
                    "wave_type": alt["wave_type"],
                    "time_frame": candlestick_type,
                    "recency_score": 0.8,  # Alternative patterns are slightly less recent
                    "start_date": filtered_df.index[alt["wave_points"][0]],
                    "end_date": filtered_df.index[alt["wave_points"][-1]],
                    "price_range": (
                        filtered_df[column].iloc[alt["wave_points"]].min(),
                        filtered_df[column].iloc[alt["wave_points"]].max()
                    )
                }
                result["multiple_patterns"].append(alt_pattern)
    
    # Sort patterns by confidence
    result["multiple_patterns"].sort(key=lambda x: x["confidence"], reverse=True)
    
    # Limit to max_patterns
    result["multiple_patterns"] = result["multiple_patterns"][:max_patterns]
    
    return result

def find_elliott_wave_pattern_enhanced(df: pd.DataFrame, 
                                     column: str = 'close', 
                                     min_points: int = 6,
                                     max_points: int = 12, 
                                     candlestick_type: str = 'day') -> Dict[str, Any]:
    """
    Enhanced Elliott Wave pattern detection using advanced multi-pattern analysis
    """
    # Parameter validation
    required_cols = ['close', 'open', 'high', 'low']
    if df is None or len(df) < 20 or any(col not in df.columns for col in required_cols):
        return {
            "impulse_wave": np.array([]),
            "corrective_wave": np.array([]),
            "peaks": np.array([]),
            "troughs": np.array([]),
            "confidence": 0.0,
            "wave_type": "no_pattern",
            "validation_details": {"error": "invalid_or_insufficient_data"},
            "multiple_patterns": []
        }
    
    logger.debug(f"Advanced multi-pattern detection starting with {len(df)} data points")
    
    # Use advanced multi-pattern detection
    result = find_elliott_wave_patterns_advanced(
        df, column, candlestick_types=['day', 'week', 'month'], 
        max_patterns_per_timeframe=3, pattern_relationships=True
    )
    
    # Extract best pattern for backward compatibility
    best_pattern = result.get('best_pattern', {})
    if 'error' not in best_pattern:
        primary_pattern = best_pattern.get('base_pattern', {}).get('pattern', {})
        return {
            "impulse_wave": primary_pattern.get("points", np.array([])),
            "corrective_wave": np.array([]),
            "peaks": np.array([]),
            "troughs": np.array([]),
            "confidence": best_pattern.get("composite_confidence", 0.0),
            "wave_type": primary_pattern.get("wave_type", "no_pattern"),
            "validation_details": {},
            "multiple_patterns": result.get('composite_patterns', []),
            "pattern_hierarchy": result.get('pattern_hierarchy', {}),
            "pattern_relationships": result.get('pattern_relationships', {})
        }
    else:
        return {
            "impulse_wave": np.array([]),
            "corrective_wave": np.array([]),
            "peaks": np.array([]),
            "troughs": np.array([]),
            "confidence": 0.0,
            "wave_type": "no_pattern",
            "validation_details": {"error": "no_patterns_found"},
            "multiple_patterns": []
        }

def find_best_impulse_wave(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                          column: str = 'close', min_points: int = 6,
                          max_points: int = 12, candlestick_type: str = 'day',
                          recency_weight: float = 0.0) -> Dict[str, Any]:
    """
    Find the best impulse wave pattern using the intelligent subwave system.
    Prefers patterns that align with the broader SMA trend.

    Args:
        recency_weight: 0.0 = no recency preference (default).
            Values > 0 boost patterns whose last wave point is near the end
            of the data. Used by walk-forward validation to prefer in-progress
            patterns over old completed ones.
    """
    from .intelligent_subwaves import SubwaveAnalysisConfig
    candidates = []
    config = SubwaveAnalysisConfig(candlestick_type)
    for start_type in ['trough', 'peak']:
        candidates.extend(
            generate_impulse_candidates_with_intelligent_subwaves(
                df, peaks, troughs, start_type, column, min_points, max_points, config=config
            )
        )
    if not candidates:
        logger.debug("No candidates found by intelligent subwave system")
        return {
            "wave_points": np.array([]),
            "confidence": 0.0,
            "wave_type": "no_candidates",
            "validation_details": {"error": "no_valid_candidates"},
            "subwaves": [],
            "alternatives": [],
            "subwave_analysis": {}
        }

    # Compute broad trend to prefer trend-aligned patterns
    broad_trend = _get_broad_trend(df, column)
    data_len = len(df)

    for c in candidates:
        wp = c['wave_points']
        if len(wp) >= 2:
            prices = df[column].iloc[wp].values
            pattern_up = prices[-1] > prices[0]
            c['_pattern_direction'] = 'up' if pattern_up else 'down'

            # Trend alignment bonus/penalty
            if broad_trend == 'bullish' and pattern_up:
                c['_adjusted_confidence'] = c['confidence'] * 1.3
            elif broad_trend == 'bearish' and not pattern_up:
                c['_adjusted_confidence'] = c['confidence'] * 1.3
            elif broad_trend == 'bullish' and not pattern_up:
                c['_adjusted_confidence'] = c['confidence'] * 0.5
            elif broad_trend == 'bearish' and pattern_up:
                c['_adjusted_confidence'] = c['confidence'] * 0.6
            else:
                c['_adjusted_confidence'] = c['confidence']

            # Recency bias: prefer patterns ending near the data boundary.
            # A pattern whose last point is at the data end gets full credit;
            # one ending at 50% of the data gets a penalty proportional to
            # recency_weight. This helps walk-forward find in-progress patterns.
            if recency_weight > 0:
                last_wp = int(wp[-1])
                recency_ratio = last_wp / max(data_len - 1, 1)
                recency_factor = 1.0 - recency_weight * (1.0 - recency_ratio)
                c['_adjusted_confidence'] *= recency_factor
        else:
            c['_adjusted_confidence'] = c['confidence']
            c['_pattern_direction'] = 'unknown'

    candidates.sort(key=lambda x: x["_adjusted_confidence"], reverse=True)
    best = candidates[0]
    best['trend_alignment'] = broad_trend
    best['alternatives'] = candidates[1:6]  # Keep top alternatives
    logger.debug(f"Best candidate: direction={best.get('_pattern_direction')}, "
                 f"raw_conf={best['confidence']:.3f}, "
                 f"adj_conf={best['_adjusted_confidence']:.3f}, "
                 f"trend={broad_trend}")
    return best


def _get_broad_trend(df: pd.DataFrame, column: str = 'close') -> str:
    """Determine broad market trend using SMA50/SMA200."""
    prices = df[column]
    if len(prices) < 50:
        return 'neutral'

    sma50 = prices.rolling(50).mean().iloc[-1]
    current_price = float(prices.iloc[-1])

    if len(prices) >= 200:
        sma200 = prices.rolling(200).mean().iloc[-1]
        if sma50 > sma200 and current_price > sma200:
            return 'bullish'
        elif sma50 < sma200 and current_price < sma200:
            return 'bearish'
        return 'neutral'
    else:
        if current_price > sma50 * 1.05:
            return 'bullish'
        elif current_price < sma50 * 0.95:
            return 'bearish'
        return 'neutral'

def generate_impulse_candidates(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                              start_type: str, column: str = 'close', 
                              min_points: int = 6, max_points: int = 12, max_depth: int = 1) -> List[Dict[str, Any]]:
    """
    [INTELLIGENT SUBWAVE SYSTEM] Generate impulse wave candidates using the new advisory subwave analysis.
    """
    return generate_impulse_candidates_with_intelligent_subwaves(
        df, peaks, troughs, start_type, column, min_points, max_points
    )

# --- UNUSED ENHANCED FUNCTIONS REMOVED ---
# The following functions were never called and have been removed to reduce code complexity:
# - generate_impulse_candidates_enhanced (replaced by intelligent_subwaves system)
# - validate_impulse_wave_rules_enhanced (replaced by intelligent_subwaves system)
# - find_best_impulse_wave_enhanced (replaced by find_best_impulse_wave)
# - filter_by_candlestick_type_enhanced (duplicate of filter_by_candlestick_type)

# Test function test_enhanced_impulse_detection() moved to tests/test_elliott_wave.py 