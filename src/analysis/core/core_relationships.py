"""
Core Pattern Relationship Detection
====================================

This module handles the primary multi-pattern relationship analysis,
detecting confirmations, conflicts, and nested patterns across timeframes.

Main Functions:
---------------
detect_multiple_pattern_relationships : Analyze multi-timeframe relationships
analyze_complex_pattern_pair : Compare individual patterns
determine_complex_relationship_type : Classify relationship type

Examples:
---------
>>> from core_relationships import detect_multiple_pattern_relationships
>>> relationships = detect_multiple_pattern_relationships(patterns_dict, df, column='close')
>>> signals = relationships['trading_signals']
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

from .utils import calculate_time_overlap, get_pattern_direction
from .compatibility_analysis import (
    analyze_wave_structure_compatibility,
    analyze_fibonacci_relationships,
    analyze_volume_pattern_alignment,
    calculate_complexity_score,
    calculate_complex_alignment_score
)
from .signal_generation import (
    generate_trading_signals,
    assess_pattern_risks
)

logger = logging.getLogger(__name__)


def detect_multiple_pattern_relationships(patterns_dict: Dict[str, Dict],
                                        df: pd.DataFrame,
                                        column: str = 'close') -> Dict[str, Any]:
    """
    Analyze relationships between multiple Elliott Wave patterns.

    Parameters
    ----------
    patterns_dict : Dict[str, Dict]
        Dictionary containing patterns from different timeframes
        Format: {'day': {...}, 'week': {...}, 'month': {...}}
    df : pd.DataFrame
        DataFrame with price data
    column : str, default 'close'
        Column name for price data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - confirmations: List of pattern confirmations
        - conflicts: List of pattern conflicts
        - nested_patterns: List of nested pattern relationships
        - pattern_alignment: Alignment metrics
        - trading_signals: Generated trading signals
        - risk_assessments: Risk analysis
        - alignment_score: Overall alignment score [0, 1]

    Notes
    -----
    This function performs pairwise analysis of patterns across timeframes
    to identify confirmations, conflicts, and generate actionable signals.
    """
    relationships = {
        'confirmations': [],
        'conflicts': [],
        'nested_patterns': [],
        'pattern_alignment': {},
        'trading_signals': [],
        'risk_assessments': [],
        'alignment_score': 0.5
    }

    try:
        timeframes = list(patterns_dict.keys())

        if not timeframes:
            logger.warning("No timeframes found in patterns_dict")
            return relationships

        logger.info(f"Analyzing pattern relationships across {len(timeframes)} timeframes")

        # Analyze pairwise relationships
        for i, tf1 in enumerate(timeframes):
            for j, tf2 in enumerate(timeframes[i+1:], i+1):
                patterns1 = patterns_dict[tf1].get('multiple_patterns', [])
                patterns2 = patterns_dict[tf2].get('multiple_patterns', [])

                logger.debug(f"Comparing {len(patterns1)} patterns from {tf1} "
                           f"with {len(patterns2)} patterns from {tf2}")

                for p1 in patterns1:
                    for p2 in patterns2:
                        try:
                            relationship = analyze_complex_pattern_pair(
                                p1, p2, tf1, tf2, df, column
                            )

                            if relationship['type'] == 'confirmation':
                                relationships['confirmations'].append(relationship)
                            elif relationship['type'] == 'conflict':
                                relationships['conflicts'].append(relationship)
                            elif relationship['type'] == 'nested':
                                relationships['nested_patterns'].append(relationship)

                        except Exception as e:
                            logger.warning(f"Error analyzing pattern pair: {e}")
                            continue

        # Generate trading signals based on pattern relationships
        relationships['trading_signals'] = generate_trading_signals(
            relationships, patterns_dict
        )

        # Assess risk based on pattern conflicts
        relationships['risk_assessments'] = assess_pattern_risks(
            relationships, patterns_dict
        )

        # Calculate overall alignment score
        relationships['alignment_score'] = calculate_complex_alignment_score(
            relationships
        )

        logger.info(f"Found {len(relationships['confirmations'])} confirmations, "
                   f"{len(relationships['conflicts'])} conflicts, "
                   f"alignment_score={relationships['alignment_score']:.2f}")

    except Exception as e:
        logger.error(f"Error in multi-pattern relationship analysis: {e}", exc_info=True)

    return relationships


def analyze_complex_pattern_pair(pattern1: Dict, pattern2: Dict,
                               timeframe1: str, timeframe2: str,
                               df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze complex relationship between two patterns from different timeframes.

    Parameters
    ----------
    pattern1, pattern2 : Dict
        Pattern dictionaries to compare
    timeframe1, timeframe2 : str
        Timeframe identifiers for each pattern
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    Dict[str, Any]
        Relationship analysis including:
        - type: 'confirmation', 'conflict', 'nested', etc.
        - overlap_ratio: Time overlap between patterns
        - directional_agreement: bool
        - wave_structure_compatibility: float [0, 1]
        - fibonacci_relationships: Dict
        - volume_pattern_alignment: float [0, 1]
        - strength: Relationship strength
        - complexity_score: Pattern complexity measure

    Notes
    -----
    Uses multiple dimensions of analysis: time overlap, direction,
    wave structure, Fibonacci ratios, and volume patterns.
    """
    # Basic relationship analysis
    overlap_ratio = calculate_time_overlap(
        pattern1['start_date'], pattern1['end_date'],
        pattern2['start_date'], pattern2['end_date']
    )

    direction1 = get_pattern_direction(pattern1)
    direction2 = get_pattern_direction(pattern2)
    directional_agreement = direction1 == direction2

    # Advanced analysis
    wave_structure_compatibility = analyze_wave_structure_compatibility(
        pattern1, pattern2
    )
    fibonacci_relationships = analyze_fibonacci_relationships(
        pattern1, pattern2, df, column
    )
    volume_pattern_alignment = analyze_volume_pattern_alignment(
        pattern1, pattern2, df
    )

    # Determine relationship type and strength
    relationship_type, strength = determine_complex_relationship_type(
        overlap_ratio, directional_agreement, wave_structure_compatibility,
        fibonacci_relationships, volume_pattern_alignment
    )

    return {
        'type': relationship_type,
        'pattern1': {'timeframe': timeframe1, 'confidence': pattern1['confidence']},
        'pattern2': {'timeframe': timeframe2, 'confidence': pattern2['confidence']},
        'overlap_ratio': overlap_ratio,
        'directional_agreement': directional_agreement,
        'wave_structure_compatibility': wave_structure_compatibility,
        'fibonacci_relationships': fibonacci_relationships,
        'volume_pattern_alignment': volume_pattern_alignment,
        'strength': strength,
        'complexity_score': calculate_complexity_score(pattern1, pattern2)
    }


def determine_complex_relationship_type(overlap_ratio: float,
                                      directional_agreement: bool,
                                      wave_structure_compatibility: float,
                                      fibonacci_relationships: Dict[str, Any],
                                      volume_pattern_alignment: float) -> Tuple[str, float]:
    """
    Determine the type and strength of relationship between patterns.

    Parameters
    ----------
    overlap_ratio : float
        Time overlap between patterns [0, 1]
    directional_agreement : bool
        Whether patterns agree on direction
    wave_structure_compatibility : float
        Wave structure similarity [0, 1]
    fibonacci_relationships : Dict
        Fibonacci analysis results
    volume_pattern_alignment : float
        Volume pattern similarity [0, 1]

    Returns
    -------
    Tuple[str, float]
        (relationship_type, strength)
        Types: 'confirmation', 'partial_confirmation', 'independent',
               'partial_conflict', 'conflict'

    Notes
    -----
    Uses weighted composite score to classify relationship.
    Higher scores indicate stronger confirmations.
    """
    fib_score = fibonacci_relationships.get('fibonacci_score', 0.0)

    # Calculate composite score
    composite_score = (
        overlap_ratio * 0.3 +
        (1.0 if directional_agreement else 0.0) * 0.2 +
        wave_structure_compatibility * 0.2 +
        fib_score * 0.2 +
        volume_pattern_alignment * 0.1
    )

    # Determine relationship type based on composite score
    if composite_score > 0.7:
        relationship_type = 'confirmation'
        strength = composite_score
    elif composite_score > 0.5:
        relationship_type = 'partial_confirmation'
        strength = composite_score * 0.8
    elif composite_score > 0.3:
        relationship_type = 'independent'
        strength = composite_score * 0.5
    elif composite_score > 0.1:
        relationship_type = 'partial_conflict'
        strength = -composite_score * 0.3
    else:
        relationship_type = 'conflict'
        strength = -composite_score

    return relationship_type, strength
