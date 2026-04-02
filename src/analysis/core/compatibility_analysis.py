"""
Pattern Compatibility and Analysis Metrics
============================================

This module provides analysis functions for pattern compatibility,
including wave structure, Fibonacci ratios, and volume pattern alignment.

Main Functions:
---------------
analyze_wave_structure_compatibility : Compare wave structures
analyze_fibonacci_relationships : Analyze Fibonacci ratio relationships
analyze_volume_pattern_alignment : Compare volume patterns
calculate_complexity_score : Measure pattern complexity
calculate_complex_alignment_score : Overall alignment score

Examples:
---------
>>> from compatibility_analysis import analyze_wave_structure_compatibility
>>> score = analyze_wave_structure_compatibility(pattern1, pattern2)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def analyze_wave_structure_compatibility(pattern1: Dict, pattern2: Dict) -> float:
    """
    Analyze compatibility of wave structures between patterns.

    Parameters
    ----------
    pattern1, pattern2 : Dict
        Pattern dictionaries with 'points' and 'wave_type' keys

    Returns
    -------
    float
        Compatibility score [0, 1]

    Notes
    -----
    Considers wave count similarity, phase alignment, and pattern type match.
    Patterns with similar structures have higher compatibility.
    """
    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])

    if len(points1) < 3 or len(points2) < 3:
        return 0.0

    try:
        # Check wave count compatibility
        wave_count1 = len(points1)
        wave_count2 = len(points2)

        # Patterns with similar wave counts are more compatible
        wave_count_diff = abs(wave_count1 - wave_count2)
        wave_count_compatibility = max(0, 1 - wave_count_diff / 5)

        # Check wave phase alignment (where in the 5-wave cycle)
        phase1 = wave_count1 % 5
        phase2 = wave_count2 % 5
        phase_alignment = 1 - abs(phase1 - phase2) / 5

        # Check pattern type compatibility
        type1 = pattern1.get('wave_type', 'unknown')
        type2 = pattern2.get('wave_type', 'unknown')
        type_compatibility = 1.0 if type1 == type2 else 0.5

        return (wave_count_compatibility * 0.4 +
                phase_alignment * 0.4 +
                type_compatibility * 0.2)

    except Exception as e:
        logger.warning(f"Error analyzing wave structure compatibility: {e}")
        return 0.0


def analyze_fibonacci_relationships(pattern1: Dict, pattern2: Dict,
                                  df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze Fibonacci relationships between patterns.

    Parameters
    ----------
    pattern1, pattern2 : Dict
        Pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - fibonacci_score: Overall Fibonacci compatibility [0, 1]
        - relationships: List of Fibonacci ratios for each pattern

    Notes
    -----
    Compares wave retracements and extensions between patterns.
    Similar Fibonacci ratios indicate compatible pattern structures.
    """
    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])

    if len(points1) < 5 or len(points2) < 5:
        return {'fibonacci_score': 0.0, 'relationships': []}

    try:
        prices1 = df[column].iloc[points1].values
        prices2 = df[column].iloc[points2].values

        # Calculate Fibonacci ratios for each pattern
        fib_ratios1 = calculate_fibonacci_ratios(prices1)
        fib_ratios2 = calculate_fibonacci_ratios(prices2)

        # Compare Fibonacci relationships
        fib_compatibility = compare_fibonacci_ratios(fib_ratios1, fib_ratios2)

        return {
            'fibonacci_score': fib_compatibility,
            'relationships': [
                {'pattern1_ratios': fib_ratios1},
                {'pattern2_ratios': fib_ratios2}
            ]
        }

    except (KeyError, IndexError) as e:
        logger.warning(f"Error analyzing Fibonacci relationships: {e}")
        return {'fibonacci_score': 0.0, 'relationships': []}


def analyze_volume_pattern_alignment(pattern1: Dict, pattern2: Dict,
                                   df: pd.DataFrame) -> float:
    """
    Analyze volume pattern alignment between patterns.

    Parameters
    ----------
    pattern1, pattern2 : Dict
        Pattern dictionaries
    df : pd.DataFrame
        Price data with 'volume' column

    Returns
    -------
    float
        Volume alignment score [0, 1]

    Notes
    -----
    Compares volume characteristics: average volume, volume trend,
    and volume consistency. Returns 0.5 if volume data unavailable.
    """
    if 'volume' not in df.columns:
        return 0.5  # Neutral if no volume data

    points1 = pattern1.get('points', [])
    points2 = pattern2.get('points', [])

    if len(points1) < 3 or len(points2) < 3:
        return 0.5

    try:
        # Calculate volume characteristics for each pattern
        vol1 = calculate_volume_characteristics(df, points1)
        vol2 = calculate_volume_characteristics(df, points2)

        # Compare volume patterns
        volume_alignment = compare_volume_patterns(vol1, vol2)

        return volume_alignment

    except Exception as e:
        logger.warning(f"Error analyzing volume pattern alignment: {e}")
        return 0.5


def calculate_complexity_score(pattern1: Dict, pattern2: Dict) -> float:
    """
    Calculate complexity score for pattern relationship.

    Parameters
    ----------
    pattern1, pattern2 : Dict
        Pattern dictionaries with 'points' key

    Returns
    -------
    float
        Complexity score [0, 1]

    Notes
    -----
    More complex patterns (more points) have higher scores.
    Normalized by 5 points (typical Elliott Wave pattern).
    """
    points1 = len(pattern1.get('points', []))
    points2 = len(pattern2.get('points', []))

    # More complex patterns have higher scores
    complexity1 = min(points1 / 5, 1.0)
    complexity2 = min(points2 / 5, 1.0)

    return (complexity1 + complexity2) / 2


def calculate_complex_alignment_score(relationships: Dict[str, Any]) -> float:
    """
    Calculate comprehensive alignment score considering all relationship factors.

    Parameters
    ----------
    relationships : Dict[str, Any]
        Pattern relationships with confirmations, conflicts, nested patterns

    Returns
    -------
    float
        Overall alignment score [0, 1]

    Notes
    -----
    Weights confirmations more heavily than conflicts (2x weight).
    Adjusts score based on average confirmation strength.
    Returns 0.5 if no relationships detected.
    """
    confirmations = len(relationships.get('confirmations', []))
    conflicts = len(relationships.get('conflicts', []))
    nested = len(relationships.get('nested_patterns', []))

    if confirmations + conflicts + nested == 0:
        return 0.5

    try:
        # Weight confirmations more heavily than conflicts
        alignment_score = (
            (confirmations * 2 + nested) /
            (confirmations * 2 + conflicts + nested)
        )

        # Adjust based on confirmation strength
        if confirmations > 0:
            avg_confirmation_strength = np.mean([
                c['strength'] for c in relationships['confirmations']
            ])
            alignment_score *= (0.5 + avg_confirmation_strength * 0.5)

        return min(alignment_score, 1.0)

    except Exception as e:
        logger.warning(f"Error calculating alignment score: {e}")
        return 0.5


# Helper functions

def calculate_fibonacci_ratios(prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate Fibonacci ratios for a price sequence.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices at wave points

    Returns
    -------
    Dict[str, float]
        Dictionary with Fibonacci ratios:
        - wave2_retracement: Wave 2 retracement ratio
        - wave3_extension: Wave 3 extension ratio
        - wave4_retracement: Wave 4 retracement ratio

    Notes
    -----
    Requires at least 5 prices for full analysis.
    Returns empty dict if insufficient data.
    """
    if len(prices) < 5:
        return {}

    ratios = {}

    try:
        # Wave 2 retracement
        if prices[1] != prices[0]:
            wave2_retr = abs((prices[2] - prices[1]) / (prices[1] - prices[0]))
            ratios['wave2_retracement'] = wave2_retr

        # Wave 3 extension
        if prices[1] != prices[0]:
            wave3_ext = abs((prices[3] - prices[2]) / (prices[1] - prices[0]))
            ratios['wave3_extension'] = wave3_ext

        # Wave 4 retracement
        if prices[3] != prices[2]:
            wave4_retr = abs((prices[4] - prices[3]) / (prices[3] - prices[2]))
            ratios['wave4_retracement'] = wave4_retr

    except (ZeroDivisionError, IndexError) as e:
        logger.warning(f"Error calculating Fibonacci ratios: {e}")

    return ratios


def compare_fibonacci_ratios(ratios1: Dict[str, float],
                            ratios2: Dict[str, float]) -> float:
    """
    Compare Fibonacci ratios between patterns.

    Parameters
    ----------
    ratios1, ratios2 : Dict[str, float]
        Fibonacci ratio dictionaries to compare

    Returns
    -------
    float
        Similarity score [0, 1]

    Notes
    -----
    Compares common ratio keys. Closer ratios result in higher similarity.
    Returns 0.5 if no common ratios or empty dicts.
    """
    if not ratios1 or not ratios2:
        return 0.5

    common_keys = set(ratios1.keys()) & set(ratios2.keys())
    if not common_keys:
        return 0.5

    similarities = []
    for key in common_keys:
        ratio1 = ratios1[key]
        ratio2 = ratios2[key]

        # Calculate similarity (closer ratios = higher similarity)
        diff = abs(ratio1 - ratio2)
        max_ratio = max(ratio1, ratio2)
        similarity = max(0, 1 - diff / max_ratio) if max_ratio > 0 else 0
        similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.5


def calculate_volume_characteristics(df: pd.DataFrame,
                                    points: List[int]) -> Dict[str, float]:
    """
    Calculate volume characteristics for a pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'volume' column
    points : List[int]
        Wave point indices

    Returns
    -------
    Dict[str, float]
        Dictionary with volume metrics:
        - avg_volume: Average volume
        - volume_trend: Linear trend (slope)
        - volume_consistency: Inverse of coefficient of variation

    Notes
    -----
    Returns zeros if volume data unavailable or insufficient points.
    """
    if 'volume' not in df.columns or len(points) < 3:
        return {'avg_volume': 0, 'volume_trend': 0, 'volume_consistency': 0}

    try:
        volumes = df['volume'].iloc[points].values

        avg_volume = np.mean(volumes)
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        volume_consistency = (
            1 - np.std(volumes) / avg_volume
            if avg_volume > 0 else 0
        )

        return {
            'avg_volume': avg_volume,
            'volume_trend': volume_trend,
            'volume_consistency': volume_consistency
        }

    except Exception as e:
        logger.warning(f"Error calculating volume characteristics: {e}")
        return {'avg_volume': 0, 'volume_trend': 0, 'volume_consistency': 0}


def compare_volume_patterns(vol1: Dict[str, float],
                           vol2: Dict[str, float]) -> float:
    """
    Compare volume patterns between two patterns.

    Parameters
    ----------
    vol1, vol2 : Dict[str, float]
        Volume characteristic dictionaries

    Returns
    -------
    float
        Volume pattern similarity [0, 1]

    Notes
    -----
    Compares average volumes (40%), trends (40%), and consistency (20%).
    Returns 0.5 if insufficient data.
    """
    if not vol1 or not vol2:
        return 0.5

    try:
        # Compare average volumes
        max_avg = max(vol1['avg_volume'], vol2['avg_volume'])
        avg_vol_similarity = (
            1 - abs(vol1['avg_volume'] - vol2['avg_volume']) / max_avg
            if max_avg > 0 else 0
        )

        # Compare volume trends
        max_trend = max(abs(vol1['volume_trend']), abs(vol2['volume_trend']))
        trend_similarity = (
            1 - abs(vol1['volume_trend'] - vol2['volume_trend']) / max_trend
            if max_trend > 0 else 0
        )

        # Compare volume consistency
        consistency_similarity = 1 - abs(
            vol1['volume_consistency'] - vol2['volume_consistency']
        )

        return (avg_vol_similarity * 0.4 +
                trend_similarity * 0.4 +
                consistency_similarity * 0.2)

    except Exception as e:
        logger.warning(f"Error comparing volume patterns: {e}")
        return 0.5
