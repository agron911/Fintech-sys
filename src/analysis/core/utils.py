"""
Shared utilities for Elliott Wave analysis to eliminate duplication.
This module consolidates common functions used across multiple analysis modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import math


def calculate_time_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """
    Calculate the overlap between two time periods.
    
    Args:
        start1, end1: Start and end indices of first period
        start2, end2: Start and end indices of second period
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if end1 <= start2 or end2 <= start1:
        return 0.0
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = overlap_end - overlap_start
    
    period1_length = end1 - start1
    period2_length = end2 - start2
    
    if period1_length == 0 or period2_length == 0:
        return 0.0
    
    # Return the smaller overlap ratio
    return min(overlap_length / period1_length, overlap_length / period2_length)


def get_pattern_direction(pattern: Dict[str, Any]) -> str:
    """
    Determine the overall direction of an Elliott Wave pattern.
    
    Args:
        pattern: Pattern dictionary containing wave points
    
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    points = pattern.get('points', [])
    if len(points) < 2:
        return 'neutral'
    
    # For impulse waves, compare start and end
    if pattern.get('wave_type') == 'impulse':
        start_price = pattern.get('start_price', 0)
        end_price = pattern.get('end_price', 0)
        if end_price > start_price:
            return 'bullish'
        elif end_price < start_price:
            return 'bearish'
        else:
            return 'neutral'
    
    # For corrective waves, analyze the overall structure
    if len(points) >= 3:
        # Check if the pattern ends higher or lower than it starts
        start_idx = points[0]
        end_idx = points[-1]
        
        # This would need price data to be more accurate
        # For now, use a simple heuristic based on wave structure
        return 'neutral'
    
    return 'neutral'


def calculate_base_confidence(pattern: Dict[str, Any], df: pd.DataFrame, column: str = 'close') -> float:
    """
    Calculate base confidence score for a pattern.
    
    Args:
        pattern: Pattern dictionary
        df: Price data DataFrame
        column: Column name for price data
    
    Returns:
        Base confidence score (0.0 to 1.0)
    """
    confidence = 0.0
    points = pattern.get('points', [])
    
    # Basic pattern completeness
    if len(points) >= 5:
        confidence += 0.3
    elif len(points) >= 3:
        confidence += 0.2
    else:
        confidence += 0.1
    
    # Pattern type validation
    wave_type = pattern.get('wave_type', 'unknown')
    if wave_type in ['impulse', 'corrective']:
        confidence += 0.2
    
    # Data quality check
    if len(df) >= 50:
        confidence += 0.1
    
    # Price movement validation
    if len(points) >= 2:
        start_idx = points[0]
        end_idx = points[-1]
        if 0 <= start_idx < len(df) and 0 <= end_idx < len(df):
            price_change = abs(df[column].iloc[end_idx] - df[column].iloc[start_idx])
            avg_price = df[column].mean()
            if avg_price > 0:
                movement_ratio = price_change / avg_price
                if 0.01 <= movement_ratio <= 0.5:  # Reasonable price movement
                    confidence += 0.2
                elif movement_ratio > 0.5:  # Very large movement
                    confidence += 0.1
    
    return min(confidence, 1.0)


def validate_pattern_structure(pattern: Dict[str, Any], df: pd.DataFrame, column: str = 'close') -> Tuple[bool, float, Dict[str, Any]]:
    """
    Basic pattern structure validation.
    
    Args:
        pattern: Pattern dictionary
        df: Price data DataFrame
        column: Column name for price data
    
    Returns:
        (is_valid, confidence, details)
    """
    points = pattern.get('points', [])
    if len(points) < 3:
        return False, 0.0, {'error': 'Insufficient wave points'}
    
    # Check for valid indices
    if not all(0 <= idx < len(df) for idx in points):
        return False, 0.0, {'error': 'Invalid wave point indices'}
    
    # Basic structure validation
    confidence = calculate_base_confidence(pattern, df, column)
    is_valid = confidence >= 0.3
    
    details = {
        'base_confidence': confidence,
        'wave_count': len(points),
        'wave_type': pattern.get('wave_type', 'unknown')
    }
    
    return is_valid, confidence, details


def calculate_pattern_metrics(pattern: Dict[str, Any], df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a pattern.
    
    Args:
        pattern: Pattern dictionary
        df: Price data DataFrame
        column: Column name for price data
    
    Returns:
        Dictionary of pattern metrics
    """
    points = pattern.get('points', [])
    if len(points) < 2:
        return {}
    
    metrics = {
        'start_index': points[0],
        'end_index': points[-1],
        'duration': points[-1] - points[0],
        'wave_count': len(points)
    }
    
    # Price metrics
    if 0 <= points[0] < len(df) and 0 <= points[-1] < len(df):
        start_price = df[column].iloc[points[0]]
        end_price = df[column].iloc[points[-1]]
        metrics.update({
            'start_price': start_price,
            'end_price': end_price,
            'price_change': end_price - start_price,
            'price_change_pct': ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
        })
    
    # Wave structure metrics
    if len(points) >= 3:
        wave_lengths = []
        for i in range(1, len(points)):
            if 0 <= points[i-1] < len(df) and 0 <= points[i] < len(df):
                length = abs(df[column].iloc[points[i]] - df[column].iloc[points[i-1]])
                wave_lengths.append(length)
        
        if wave_lengths:
            metrics.update({
                'avg_wave_length': np.mean(wave_lengths),
                'max_wave_length': max(wave_lengths),
                'min_wave_length': min(wave_lengths),
                'wave_length_std': np.std(wave_lengths)
            })
    
    return metrics


def merge_pattern_relationships(relationships1: Dict[str, Any], relationships2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two pattern relationship dictionaries.
    
    Args:
        relationships1: First relationship dictionary
        relationships2: Second relationship dictionary
    
    Returns:
        Merged relationship dictionary
    """
    merged = {}
    
    # Merge confirmations
    confirmations = relationships1.get('confirmations', []) + relationships2.get('confirmations', [])
    merged['confirmations'] = confirmations
    
    # Merge conflicts
    conflicts = relationships1.get('conflicts', []) + relationships2.get('conflicts', [])
    merged['conflicts'] = conflicts
    
    # Merge other relationship types
    for key in ['supporting', 'independent', 'nested']:
        items1 = relationships1.get(key, [])
        items2 = relationships2.get(key, [])
        merged[key] = items1 + items2
    
    # Merge composite metrics
    composite1 = relationships1.get('composite_metrics', {})
    composite2 = relationships2.get('composite_metrics', {})
    
    merged_composite = {}
    for key in set(composite1.keys()) | set(composite2.keys()):
        if key in composite1 and key in composite2:
            # Average the values
            merged_composite[key] = (composite1[key] + composite2[key]) / 2
        elif key in composite1:
            merged_composite[key] = composite1[key]
        else:
            merged_composite[key] = composite2[key]
    
    merged['composite_metrics'] = merged_composite
    
    return merged


def calculate_relationship_strength(pattern1: Dict[str, Any], pattern2: Dict[str, Any], 
                                  relationship_type: str) -> float:
    """
    Calculate the strength of a relationship between two patterns.
    
    Args:
        pattern1: First pattern dictionary
        pattern2: Second pattern dictionary
        relationship_type: Type of relationship ('confirmation', 'conflict', 'supporting', etc.)
    
    Returns:
        Relationship strength (0.0 to 1.0)
    """
    # Base strength based on relationship type
    base_strengths = {
        'confirmation': 0.8,
        'supporting': 0.6,
        'independent': 0.3,
        'conflict': 0.4,
        'nested': 0.7
    }
    
    base_strength = base_strengths.get(relationship_type, 0.5)
    
    # Adjust based on pattern confidences
    conf1 = pattern1.get('confidence', 0.5)
    conf2 = pattern2.get('confidence', 0.5)
    avg_confidence = (conf1 + conf2) / 2
    
    # Adjust based on time overlap
    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])
    
    if len(points1) >= 2 and len(points2) >= 2:
        overlap = calculate_time_overlap(points1[0], points1[-1], points2[0], points2[-1])
        # Higher overlap for confirmations, lower for conflicts
        if relationship_type == 'confirmation':
            overlap_factor = overlap
        elif relationship_type == 'conflict':
            overlap_factor = 1 - overlap
        else:
            overlap_factor = 0.5
    else:
        overlap_factor = 0.5
    
    # Calculate final strength
    strength = base_strength * avg_confidence * overlap_factor
    return min(strength, 1.0)


def normalize_confidence_scores(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize confidence scores across a list of patterns.
    
    Args:
        patterns: List of pattern dictionaries
    
    Returns:
        List of patterns with normalized confidence scores
    """
    if not patterns:
        return patterns
    
    # Extract current confidence scores
    confidences = [p.get('confidence', 0.0) for p in patterns]
    min_conf = min(confidences)
    max_conf = max(confidences)
    
    # Normalize to 0.1 to 1.0 range
    if max_conf > min_conf:
        normalized_patterns = []
        for pattern in patterns:
            norm_pattern = pattern.copy()
            current_conf = pattern.get('confidence', 0.0)
            normalized_conf = 0.1 + 0.9 * (current_conf - min_conf) / (max_conf - min_conf)
            norm_pattern['confidence'] = normalized_conf
            normalized_patterns.append(norm_pattern)
        return normalized_patterns
    else:
        # All confidences are the same, set to 0.5
        normalized_patterns = []
        for pattern in patterns:
            norm_pattern = pattern.copy()
            norm_pattern['confidence'] = 0.5
            normalized_patterns.append(norm_pattern)
        return normalized_patterns


def validate_data_quality(df: pd.DataFrame, column: str = 'close') -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the quality of input data for Elliott Wave analysis.
    
    Args:
        df: Price data DataFrame
        column: Column name for price data
    
    Returns:
        (is_valid, quality_metrics)
    """
    if df is None or len(df) == 0:
        return False, {'error': 'Empty or None DataFrame'}
    
    if column not in df.columns:
        return False, {'error': f'Column {column} not found in DataFrame'}
    
    # Check for sufficient data points
    min_required = 50
    if len(df) < min_required:
        return False, {'error': f'Insufficient data points: {len(df)} < {min_required}'}
    
    # Check for missing values
    missing_count = df[column].isnull().sum()
    missing_ratio = missing_count / len(df)
    
    if missing_ratio > 0.1:  # More than 10% missing
        return False, {'error': f'Too many missing values: {missing_ratio:.1%}'}
    
    # Check for price validity
    if (df[column] <= 0).any():
        return False, {'error': 'Invalid price values (non-positive)'}
    
    # Calculate quality metrics
    quality_metrics = {
        'data_points': len(df),
        'missing_ratio': missing_ratio,
        'price_range': df[column].max() - df[column].min(),
        'price_volatility': df[column].std(),
        'date_range': df.index[-1] - df.index[0] if len(df) > 1 else pd.Timedelta(0)
    }
    
    return True, quality_metrics


def merge_related_patterns(patterns: List[Dict[str, Any]], relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Merge related patterns based on their relationships.
    
    Args:
        patterns: List of pattern dictionaries
        relationships: Dictionary containing pattern relationships
        
    Returns:
        List of merged patterns
    """
    merged_patterns = []
    processed = set()
    
    for i, pattern1 in enumerate(patterns):
        if i in processed:
            continue
            
        merged_pattern = pattern1.copy()
        related_indices = [i]
        
        # Find related patterns
        for j, pattern2 in enumerate(patterns[i+1:], i+1):
            if j in processed:
                continue
                
            # Check if patterns are related
            if are_patterns_related(pattern1, pattern2, relationships):
                related_indices.append(j)
                processed.add(j)
                
                # Merge pattern data
                merged_pattern = merge_pattern_data(merged_pattern, pattern2)
        
        merged_patterns.append(merged_pattern)
        processed.add(i)
    
    return merged_patterns


def check_wave_phase_alignment(pattern1: Dict, pattern2: Dict) -> float:
    """
    Check if patterns are in similar wave phases.

    Args:
        pattern1: First pattern dictionary
        pattern2: Second pattern dictionary

    Returns:
        Alignment score between 0.0 and 1.0
    """
    # Simplified phase check - would be more sophisticated in practice
    wave_count1 = len(pattern1.get('points', []))
    wave_count2 = len(pattern2.get('points', []))

    if wave_count1 == 0 or wave_count2 == 0:
        return 0.0

    phase_diff = abs((wave_count1 % 5) - (wave_count2 % 5))
    return 1.0 - (phase_diff / 5.0)


# Alias for backward compatibility
calculate_confidence = calculate_base_confidence


def validate_pattern_relationships(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate relationships between multiple patterns.

    Args:
        patterns: List of pattern dictionaries

    Returns:
        Dictionary containing relationship analysis
    """
    if len(patterns) < 2:
        return {'relationships': [], 'total_patterns': len(patterns)}

    relationships = []

    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            pattern1 = patterns[i]
            pattern2 = patterns[j]

            # Check for time overlap
            points1 = pattern1.get('points', [])
            points2 = pattern2.get('points', [])

            if len(points1) >= 2 and len(points2) >= 2:
                overlap = calculate_time_overlap(points1[0], points1[-1],
                                                points2[0], points2[-1])

                # Determine relationship type based on overlap
                if overlap > 0.7:
                    rel_type = 'confirmation'
                elif overlap > 0.3:
                    rel_type = 'supporting'
                elif overlap > 0:
                    rel_type = 'conflict'
                else:
                    rel_type = 'independent'

                strength = calculate_relationship_strength(pattern1, pattern2, rel_type)

                relationships.append({
                    'pattern1_index': i,
                    'pattern2_index': j,
                    'type': rel_type,
                    'strength': strength,
                    'overlap': overlap
                })

    return {
        'relationships': relationships,
        'total_patterns': len(patterns),
        'total_relationships': len(relationships)
    }


def are_patterns_related(pattern1: Dict[str, Any], pattern2: Dict[str, Any],
                        relationships: Dict[str, Any]) -> bool:
    """
    Check if two patterns are related based on relationship data.

    Args:
        pattern1: First pattern dictionary
        pattern2: Second pattern dictionary
        relationships: Relationship dictionary

    Returns:
        True if patterns are related, False otherwise
    """
    # Check time overlap
    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])

    if len(points1) >= 2 and len(points2) >= 2:
        overlap = calculate_time_overlap(points1[0], points1[-1],
                                        points2[0], points2[-1])
        if overlap > 0.5:
            return True

    # Check relationship strength
    for rel in relationships.get('relationships', []):
        if rel.get('strength', 0) > 0.6:
            return True

    return False


def merge_pattern_data(pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge data from two patterns.

    Args:
        pattern1: First pattern dictionary
        pattern2: Second pattern dictionary

    Returns:
        Merged pattern dictionary
    """
    merged = pattern1.copy()

    # Merge confidence scores (average)
    conf1 = pattern1.get('confidence', 0.5)
    conf2 = pattern2.get('confidence', 0.5)
    merged['confidence'] = (conf1 + conf2) / 2

    # Extend wave points if compatible
    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])

    if points1 and points2:
        # If patterns are consecutive, extend points
        if points1[-1] == points2[0]:
            merged['points'] = points1 + points2[1:]
        else:
            # Keep the longer pattern's points
            merged['points'] = points1 if len(points1) >= len(points2) else points2

    return merged 