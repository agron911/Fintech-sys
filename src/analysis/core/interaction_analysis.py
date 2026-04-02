"""
Pattern Interaction and Multi-Timeframe Analysis
=================================================

This module analyzes pattern interactions across timeframes,
identifying clusters, divergences, convergences, nested structures,
and generating composite forecasts.

Main Functions:
---------------
analyze_pattern_interactions : Main analysis function
identify_pattern_clusters : Group patterns by similar targets
identify_pattern_divergences : Find conflicting patterns
identify_pattern_convergences : Find aligned patterns
analyze_nested_structures : Analyze nested pattern relationships
identify_fractal_relationships : Find self-similar patterns
generate_composite_forecast : Generate integrated forecast

Examples:
---------
>>> from interaction_analysis import analyze_pattern_interactions
>>> interactions = analyze_pattern_interactions(df, patterns_data, column='close')
>>> forecast = interactions['composite_forecast']
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

from .utils import check_wave_phase_alignment

logger = logging.getLogger(__name__)


def analyze_pattern_interactions(df: pd.DataFrame,
                               patterns_data: Dict[str, Any],
                               column: str = 'close') -> Dict[str, Any]:
    """
    Analyze interactions and relationships between multiple Elliott Wave patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    patterns_data : Dict[str, Any]
        Dictionary containing detected patterns from all timeframes
    column : str, default 'close'
        Price column to analyze

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - pattern_clusters: List of pattern clusters
        - divergences: List of pattern conflicts
        - convergences: List of pattern alignments
        - nested_structures: List of nested pattern relationships
        - fractal_relationships: List of fractal similarities
        - composite_forecast: Integrated forecast from all patterns

    Notes
    -----
    This function integrates analysis from multiple timeframes to provide
    a comprehensive view of pattern relationships and generate composite
    forecasts with higher confidence.
    """
    interactions = {
        'pattern_clusters': [],
        'divergences': [],
        'convergences': [],
        'nested_structures': [],
        'fractal_relationships': [],
        'composite_forecast': {}
    }

    try:
        # Extract all patterns from different timeframes
        all_patterns = extract_all_patterns(patterns_data)

        if not all_patterns:
            logger.warning("No patterns found in patterns_data")
            return interactions

        logger.info(f"Analyzing interactions for {len(all_patterns)} patterns")

        # 1. Identify Pattern Clusters (multiple patterns pointing to same outcome)
        clusters = identify_pattern_clusters(all_patterns, df, column)
        interactions['pattern_clusters'] = clusters
        logger.debug(f"Found {len(clusters)} pattern clusters")

        # 2. Find Pattern Divergences (conflicting signals)
        divergences = identify_pattern_divergences(all_patterns, df, column)
        interactions['divergences'] = divergences
        logger.debug(f"Found {len(divergences)} pattern divergences")

        # 3. Find Pattern Convergences (aligned signals)
        convergences = identify_pattern_convergences(all_patterns, df, column)
        interactions['convergences'] = convergences
        logger.debug(f"Found {len(convergences)} pattern convergences")

        # 4. Analyze Nested Structures (patterns within patterns)
        nested = analyze_nested_structures(all_patterns, df, column)
        interactions['nested_structures'] = nested
        logger.debug(f"Found {len(nested)} nested structures")

        # 5. Identify Fractal Relationships
        fractals = identify_fractal_relationships(all_patterns, df, column)
        interactions['fractal_relationships'] = fractals
        logger.debug(f"Found {len(fractals)} fractal relationships")

        # 6. Generate Composite Forecast
        forecast = generate_composite_forecast(interactions, df, column)
        interactions['composite_forecast'] = forecast
        logger.info(f"Generated composite forecast with confidence "
                   f"{forecast.get('overall_confidence', 0):.2f}")

    except Exception as e:
        logger.error(f"Error in pattern interaction analysis: {e}", exc_info=True)

    return interactions


def identify_pattern_clusters(patterns: List[Dict], df: pd.DataFrame,
                            column: str) -> List[Dict[str, Any]]:
    """
    Identify clusters of patterns that suggest similar price targets.

    Parameters
    ----------
    patterns : List[Dict]
        List of pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    List[Dict[str, Any]]
        List of cluster dictionaries sorted by strength

    Notes
    -----
    Clusters are formed when multiple patterns project similar target prices
    (within 5% tolerance). Stronger clusters have more patterns, higher
    confidence, and multiple timeframes represented.
    """
    clusters = []

    # Group patterns by target zones
    target_zones = {}

    for pattern in patterns:
        try:
            # Calculate pattern target based on wave structure
            target = calculate_pattern_target(pattern, df, column)

            if target['price'] <= 0:
                continue

            # Find or create cluster for this target zone
            cluster_found = False
            for zone_key, cluster in target_zones.items():
                if abs(target['price'] - cluster['center']) / cluster['center'] < 0.05:  # 5% tolerance
                    cluster['patterns'].append(pattern)
                    cluster['confidence_sum'] += pattern.get('confidence', 0)
                    cluster['timeframes'].add(pattern.get('timeframe', 'unknown'))
                    cluster_found = True
                    break

            if not cluster_found:
                zone_key = f"zone_{len(target_zones)}"
                target_zones[zone_key] = {
                    'center': target['price'],
                    'patterns': [pattern],
                    'confidence_sum': pattern.get('confidence', 0),
                    'timeframes': set([pattern.get('timeframe', 'unknown')])
                }

        except Exception as e:
            logger.warning(f"Error processing pattern for clustering: {e}")
            continue

    # Convert to cluster list with analysis
    for zone_key, cluster_data in target_zones.items():
        if len(cluster_data['patterns']) >= 2:  # Only clusters with 2+ patterns
            cluster_analysis = {
                'target_price': cluster_data['center'],
                'pattern_count': len(cluster_data['patterns']),
                'average_confidence': cluster_data['confidence_sum'] / len(cluster_data['patterns']),
                'timeframes_involved': list(cluster_data['timeframes']),
                'cluster_strength': calculate_cluster_strength(cluster_data),
                'patterns': cluster_data['patterns']
            }
            clusters.append(cluster_analysis)

    return sorted(clusters, key=lambda x: x['cluster_strength'], reverse=True)


def identify_pattern_divergences(patterns: List[Dict], df: pd.DataFrame,
                               column: str) -> List[Dict[str, Any]]:
    """
    Identify divergences between patterns (conflicting directional signals).

    Parameters
    ----------
    patterns : List[Dict]
        List of pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    List[Dict[str, Any]]
        List of divergence dictionaries

    Notes
    -----
    Divergences occur when overlapping patterns have opposite directional
    biases. These conflicts require careful analysis to determine which
    pattern is likely to dominate.
    """
    divergences = []

    for i, pattern1 in enumerate(patterns):
        for pattern2 in patterns[i+1:]:
            try:
                # Check if patterns overlap in time
                if patterns_overlap_in_time(pattern1, pattern2):
                    # Calculate directional bias for each pattern
                    bias1 = get_pattern_directional_bias(pattern1, df, column)
                    bias2 = get_pattern_directional_bias(pattern2, df, column)

                    # Check for divergence (opposite signs)
                    if bias1 * bias2 < 0:
                        divergence = {
                            'pattern1': {
                                'timeframe': pattern1.get('timeframe'),
                                'bias': bias1,
                                'confidence': pattern1.get('confidence', 0)
                            },
                            'pattern2': {
                                'timeframe': pattern2.get('timeframe'),
                                'bias': bias2,
                                'confidence': pattern2.get('confidence', 0)
                            },
                            'severity': abs(bias1 - bias2),
                            'resolution_probability': calculate_divergence_resolution(
                                pattern1, pattern2, df, column
                            )
                        }
                        divergences.append(divergence)

            except Exception as e:
                logger.warning(f"Error analyzing pattern divergence: {e}")
                continue

    return divergences


def identify_pattern_convergences(patterns: List[Dict], df: pd.DataFrame,
                                column: str) -> List[Dict[str, Any]]:
    """
    Identify convergences between patterns (aligned signals).

    Parameters
    ----------
    patterns : List[Dict]
        List of pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    List[Dict[str, Any]]
        List of convergence dictionaries sorted by combined confidence

    Notes
    -----
    Convergences occur when overlapping patterns align in direction, target,
    and wave phase. Strong convergences (alignment > 0.7) significantly
    increase forecast confidence.
    """
    convergences = []

    for i, pattern1 in enumerate(patterns):
        for pattern2 in patterns[i+1:]:
            try:
                if patterns_overlap_in_time(pattern1, pattern2):
                    # Check alignment in multiple dimensions
                    alignment = calculate_pattern_alignment(pattern1, pattern2, df, column)

                    if alignment['score'] > 0.7:  # Strong alignment threshold
                        convergence = {
                            'patterns': [pattern1, pattern2],
                            'alignment_score': alignment['score'],
                            'alignment_type': alignment['type'],
                            'combined_confidence': min(
                                pattern1.get('confidence', 0) + pattern2.get('confidence', 0),
                                0.95  # Cap at 95%
                            ),
                            'target_confluence': alignment.get('target_confluence'),
                            'timeframes': [
                                pattern1.get('timeframe'),
                                pattern2.get('timeframe')
                            ]
                        }
                        convergences.append(convergence)

            except Exception as e:
                logger.warning(f"Error analyzing pattern convergence: {e}")
                continue

    return sorted(convergences, key=lambda x: x['combined_confidence'], reverse=True)


def analyze_nested_structures(patterns: List[Dict], df: pd.DataFrame,
                            column: str) -> List[Dict[str, Any]]:
    """
    Analyze patterns that are nested within larger patterns.

    Parameters
    ----------
    patterns : List[Dict]
        List of pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    List[Dict[str, Any]]
        List of nested structure analyses

    Notes
    -----
    Nested structures occur when smaller timeframe patterns form subwaves
    of larger timeframe patterns. This is a fundamental aspect of Elliott
    Wave fractal nature.
    """
    nested_structures = []

    # Sort patterns by timeframe scale (largest first)
    sorted_patterns = sorted(
        patterns,
        key=lambda p: get_timeframe_scale(p.get('timeframe', 'day')),
        reverse=True
    )

    for i, larger_pattern in enumerate(sorted_patterns):
        for smaller_pattern in sorted_patterns[i+1:]:
            try:
                if is_pattern_contained(smaller_pattern, larger_pattern):
                    # Analyze the nested relationship
                    nested_analysis = {
                        'parent_pattern': {
                            'timeframe': larger_pattern.get('timeframe'),
                            'wave_position': get_current_wave_position(larger_pattern)
                        },
                        'child_pattern': {
                            'timeframe': smaller_pattern.get('timeframe'),
                            'wave_position': get_current_wave_position(smaller_pattern)
                        },
                        'nesting_type': determine_nesting_type(
                            larger_pattern, smaller_pattern, df, column
                        ),
                        'alignment': check_nested_alignment(
                            larger_pattern, smaller_pattern, df, column
                        ),
                        'fractal_confidence': calculate_fractal_confidence(
                            larger_pattern, smaller_pattern
                        )
                    }
                    nested_structures.append(nested_analysis)

            except Exception as e:
                logger.warning(f"Error analyzing nested structure: {e}")
                continue

    return nested_structures


def identify_fractal_relationships(patterns: List[Dict], df: pd.DataFrame,
                                 column: str) -> List[Dict[str, Any]]:
    """
    Identify fractal (self-similar) relationships between patterns.

    Parameters
    ----------
    patterns : List[Dict]
        List of pattern dictionaries
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    List[Dict[str, Any]]
        List of fractal relationship analyses

    Notes
    -----
    Fractal relationships indicate patterns at different scales that have
    similar wave structures and ratios. High fractal similarity (> 0.75)
    increases confidence in the pattern interpretation.
    """
    fractal_relationships = []

    for i, pattern1 in enumerate(patterns):
        for pattern2 in patterns[i+1:]:
            try:
                # Check for fractal similarity
                similarity = calculate_fractal_similarity(pattern1, pattern2, df, column)

                if similarity['score'] > 0.75:
                    relationship = {
                        'pattern1': {
                            'timeframe': pattern1.get('timeframe'),
                            'scale': get_pattern_scale(pattern1, df, column)
                        },
                        'pattern2': {
                            'timeframe': pattern2.get('timeframe'),
                            'scale': get_pattern_scale(pattern2, df, column)
                        },
                        'similarity_score': similarity['score'],
                        'scale_ratio': similarity['scale_ratio'],
                        'phase_alignment': similarity['phase_alignment'],
                        'fractal_dimension': calculate_fractal_dimension(
                            [pattern1, pattern2], df, column
                        )
                    }
                    fractal_relationships.append(relationship)

            except Exception as e:
                logger.warning(f"Error analyzing fractal relationship: {e}")
                continue

    return fractal_relationships


def generate_composite_forecast(interactions: Dict[str, Any],
                              df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Generate a composite forecast based on all pattern interactions.

    Parameters
    ----------
    interactions : Dict[str, Any]
        Dictionary containing all pattern interaction analyses
    df : pd.DataFrame
        Price data
    column : str
        Price column name

    Returns
    -------
    Dict[str, Any]
        Composite forecast containing:
        - primary_scenario: Most likely outcome
        - alternative_scenarios: Top 3 alternative outcomes
        - confidence_levels: Confidence breakdown
        - key_levels: Important price levels
        - time_projections: Time-based projections
        - overall_confidence: Aggregate confidence score

    Notes
    -----
    The composite forecast integrates all pattern types with weighted
    contributions. Clusters and convergences have positive weight,
    while divergences reduce confidence.
    """
    forecast = {
        'primary_scenario': {},
        'alternative_scenarios': [],
        'confidence_levels': {},
        'key_levels': [],
        'time_projections': {},
        'overall_confidence': 0.0
    }

    try:
        # Weight different interaction types
        weights = {
            'clusters': 0.3,
            'convergences': 0.25,
            'nested': 0.2,
            'fractals': 0.15,
            'divergences': -0.1  # Negative weight for conflicts
        }

        # Calculate weighted forecast scenarios
        scenarios = []

        # From clusters
        for cluster in interactions.get('pattern_clusters', []):
            scenarios.append({
                'type': 'cluster',
                'target': cluster['target_price'],
                'confidence': cluster['average_confidence'] * weights['clusters'],
                'timeframes': cluster['timeframes_involved']
            })

        # From convergences
        for convergence in interactions.get('convergences', []):
            if convergence.get('target_confluence'):
                scenarios.append({
                    'type': 'convergence',
                    'target': convergence['target_confluence'],
                    'confidence': convergence['combined_confidence'] * weights['convergences'],
                    'timeframes': convergence['timeframes']
                })

        # Sort scenarios by confidence
        scenarios.sort(key=lambda x: x['confidence'], reverse=True)

        if scenarios:
            # Primary scenario is highest confidence
            forecast['primary_scenario'] = scenarios[0]
            logger.info(f"Primary scenario: {scenarios[0]['type']}, "
                       f"target={scenarios[0]['target']:.2f}, "
                       f"confidence={scenarios[0]['confidence']:.2f}")

            # Alternative scenarios (top 3)
            forecast['alternative_scenarios'] = scenarios[1:4]

        # Calculate key levels from all patterns
        forecast['key_levels'] = calculate_key_levels_from_interactions(
            interactions, df, column
        )

        # Time projections
        forecast['time_projections'] = calculate_time_projections(
            interactions, df
        )

        # Overall confidence
        forecast['overall_confidence'] = calculate_overall_forecast_confidence(
            interactions
        )

    except Exception as e:
        logger.error(f"Error generating composite forecast: {e}", exc_info=True)

    return forecast


# ==================== Helper Functions ====================

def calculate_pattern_target(pattern: Dict, df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate price target for a pattern using Fibonacci extensions."""
    wave_points = pattern.get('points', [])
    if len(wave_points) < 3:
        return {'price': df[column].iloc[-1] if len(df) > 0 else 0, 'confidence': 0}

    try:
        prices = df[column].iloc[wave_points].values

        if len(prices) >= 5:
            # Project based on wave 1 and current position
            wave1_length = abs(prices[1] - prices[0])

            if wave1_length == 0:
                return {'price': prices[-1], 'confidence': 0.3}

            # Common target is 1.618 * wave1 from wave4
            direction = np.sign(prices[1] - prices[0])
            projection = prices[4] + (1.618 * wave1_length * direction)

            return {'price': projection, 'confidence': pattern.get('confidence', 0.5)}

        return {'price': prices[-1], 'confidence': 0.3}

    except (KeyError, IndexError) as e:
        logger.warning(f"Error calculating pattern target: {e}")
        return {'price': df[column].iloc[-1] if len(df) > 0 else 0, 'confidence': 0}


def calculate_cluster_strength(cluster_data: Dict) -> float:
    """Calculate strength of a pattern cluster."""
    pattern_count_score = min(len(cluster_data['patterns']) / 5, 1.0)
    confidence_score = cluster_data['confidence_sum'] / len(cluster_data['patterns'])
    timeframe_diversity = len(cluster_data['timeframes']) / 3  # Normalize by max 3 timeframes

    return (pattern_count_score * 0.3 +
            confidence_score * 0.5 +
            timeframe_diversity * 0.2)


def patterns_overlap_in_time(pattern1: Dict, pattern2: Dict) -> bool:
    """Check if two patterns overlap in time."""
    start1, end1 = pattern1.get('start_date'), pattern1.get('end_date')
    start2, end2 = pattern2.get('start_date'), pattern2.get('end_date')

    if not all([start1, end1, start2, end2]):
        return False

    return not (end1 < start2 or end2 < start1)


def get_pattern_directional_bias(pattern: Dict, df: pd.DataFrame, column: str) -> float:
    """Get directional bias of pattern. Returns -1 (bearish) to +1 (bullish)."""
    wave_points = pattern.get('points', [])
    if len(wave_points) < 2:
        return 0

    try:
        prices = df[column].iloc[wave_points].values
        net_change = prices[-1] - prices[0]
        volatility = np.std(prices) if len(prices) > 1 else 1

        return np.tanh(net_change / volatility) if volatility > 0 else 0

    except (KeyError, IndexError):
        return 0


def calculate_pattern_alignment(pattern1: Dict, pattern2: Dict,
                              df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate multi-dimensional alignment between patterns."""
    alignment = {
        'score': 0,
        'type': 'none',
        'target_confluence': None
    }

    try:
        # Directional alignment
        bias1 = get_pattern_directional_bias(pattern1, df, column)
        bias2 = get_pattern_directional_bias(pattern2, df, column)
        directional_score = 1 - abs(bias1 - bias2) / 2

        # Target alignment
        target1 = calculate_pattern_target(pattern1, df, column)
        target2 = calculate_pattern_target(pattern2, df, column)

        target_score = 0
        if target1['price'] > 0 and target2['price'] > 0:
            target_diff = abs(target1['price'] - target2['price']) / target1['price']
            target_score = 1 - min(target_diff / 0.1, 1)  # 10% tolerance

            if target_score > 0.8:
                alignment['target_confluence'] = (target1['price'] + target2['price']) / 2

        # Phase alignment (where in the wave sequence)
        phase_score = check_wave_phase_alignment(pattern1, pattern2)

        # Combined score
        alignment['score'] = (directional_score * 0.4 +
                             target_score * 0.4 +
                             phase_score * 0.2)

        # Determine alignment type
        if alignment['score'] > 0.8:
            alignment['type'] = 'strong'
        elif alignment['score'] > 0.6:
            alignment['type'] = 'moderate'
        elif alignment['score'] > 0.4:
            alignment['type'] = 'weak'

    except Exception as e:
        logger.warning(f"Error calculating pattern alignment: {e}")

    return alignment


def get_timeframe_scale(timeframe: str) -> int:
    """Get numerical scale for timeframe comparison. Returns minutes per candle."""
    scales = {
        'minute': 1,
        'hour': 60,
        'day': 1440,
        'week': 10080,
        'month': 43200,
        'recent': 1440,
        'medium': 10080,
        'extended': 43200,
        'long_term': 86400
    }
    return scales.get(timeframe.lower(), 1440)


def is_pattern_contained(smaller: Dict, larger: Dict) -> bool:
    """Check if smaller pattern is contained within larger pattern timeframe."""
    if not all([smaller.get('start_date'), smaller.get('end_date'),
                larger.get('start_date'), larger.get('end_date')]):
        return False

    return (larger['start_date'] <= smaller['start_date'] and
            larger['end_date'] >= smaller['end_date'])


def get_current_wave_position(pattern: Dict) -> str:
    """Get current wave position from pattern based on point count."""
    wave_points = pattern.get('points', [])
    wave_count = len(wave_points)

    if wave_count == 0:
        return 'undefined'
    elif wave_count <= 2:
        return 'wave_1'
    elif wave_count == 3:
        return 'wave_2'
    elif wave_count == 4:
        return 'wave_3'
    elif wave_count == 5:
        return 'wave_4'
    elif wave_count == 6:
        return 'wave_5'
    else:
        return 'extended'


def determine_nesting_type(larger: Dict, smaller: Dict,
                         df: pd.DataFrame, column: str) -> str:
    """Determine the type of nesting relationship between patterns."""
    larger_pos = get_current_wave_position(larger)
    smaller_pos = get_current_wave_position(smaller)

    # Check if smaller pattern is a subwave of larger
    if 'wave_3' in larger_pos and 'wave_5' in smaller_pos:
        return 'subwave_completion'
    elif 'wave_2' in larger_pos or 'wave_4' in larger_pos:
        return 'corrective_substructure'
    else:
        return 'fractal_subdivision'


def check_nested_alignment(larger: Dict, smaller: Dict,
                         df: pd.DataFrame, column: str) -> float:
    """Check alignment between nested patterns. Returns alignment score [0, 1]."""
    try:
        # Get wave directions
        larger_direction = get_pattern_directional_bias(larger, df, column)
        smaller_direction = get_pattern_directional_bias(smaller, df, column)

        # For proper nesting, directions should align in impulse waves
        larger_pos = get_current_wave_position(larger)

        if larger_pos in ['wave_1', 'wave_3', 'wave_5']:
            # Impulse waves - should have same direction
            return 1.0 if larger_direction * smaller_direction > 0 else 0.0
        else:
            # Corrective waves - can have opposite direction
            return 0.5

    except Exception:
        return 0.5


def calculate_fractal_confidence(larger: Dict, smaller: Dict) -> float:
    """Calculate confidence in fractal relationship."""
    # Base confidence on pattern confidences
    base_confidence = (larger.get('confidence', 0) + smaller.get('confidence', 0)) / 2

    # Adjust for scale difference
    larger_scale = get_timeframe_scale(larger.get('timeframe', 'day'))
    smaller_scale = get_timeframe_scale(smaller.get('timeframe', 'day'))
    scale_ratio = larger_scale / smaller_scale if smaller_scale > 0 else 1

    # Ideal scale ratios are around 5-10
    if 5 <= scale_ratio <= 10:
        scale_factor = 1.0
    elif 3 <= scale_ratio <= 15:
        scale_factor = 0.8
    else:
        scale_factor = 0.5

    return base_confidence * scale_factor


def calculate_fractal_similarity(pattern1: Dict, pattern2: Dict,
                               df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate fractal similarity between patterns."""
    similarity = {
        'score': 0,
        'scale_ratio': 1,
        'phase_alignment': 0
    }

    try:
        # Get pattern scales
        scale1 = get_pattern_scale(pattern1, df, column)
        scale2 = get_pattern_scale(pattern2, df, column)

        if scale2 > 0:
            similarity['scale_ratio'] = scale1 / scale2

        # Check wave structure similarity
        points1 = pattern1.get('points', [])
        points2 = pattern2.get('points', [])

        if len(points1) >= 3 and len(points2) >= 3:
            # Compare wave ratios
            ratios1 = calculate_wave_ratios(points1, df, column)
            ratios2 = calculate_wave_ratios(points2, df, column)

            # Calculate similarity score
            ratio_diffs = []
            for key in set(ratios1.keys()) & set(ratios2.keys()):
                diff = abs(ratios1[key] - ratios2[key])
                ratio_diffs.append(1 - min(diff, 1))

            if ratio_diffs:
                similarity['score'] = np.mean(ratio_diffs)

        # Phase alignment
        similarity['phase_alignment'] = check_wave_phase_alignment(pattern1, pattern2)

    except Exception as e:
        logger.warning(f"Error calculating fractal similarity: {e}")

    return similarity


def get_pattern_scale(pattern: Dict, df: pd.DataFrame, column: str) -> float:
    """Get the scale (price range * time range) of a pattern."""
    wave_points = pattern.get('points', [])
    if len(wave_points) < 2:
        return 0

    try:
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]

        price_range = max(prices) - min(prices)
        time_range = (dates[-1] - dates[0]).days

        return price_range * time_range

    except (KeyError, IndexError):
        return 0


def calculate_wave_ratios(points: List[int], df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate characteristic wave ratios for pattern comparison."""
    ratios = {}

    if len(points) < 3:
        return ratios

    try:
        prices = df[column].iloc[points].values

        # Calculate key ratios
        if len(prices) >= 3:
            wave1 = abs(prices[1] - prices[0])
            wave2 = abs(prices[2] - prices[1])
            if wave1 > 0:
                ratios['wave2_to_wave1'] = wave2 / wave1

        if len(prices) >= 5:
            wave3 = abs(prices[3] - prices[2])
            wave4 = abs(prices[4] - prices[3])
            if wave1 > 0:
                ratios['wave3_to_wave1'] = wave3 / wave1
            if wave3 > 0:
                ratios['wave4_to_wave3'] = wave4 / wave3

    except (KeyError, IndexError):
        pass

    return ratios


def calculate_fractal_dimension(patterns: List[Dict], df: pd.DataFrame, column: str) -> float:
    """Calculate approximate fractal dimension of pattern set."""
    scales = []
    counts = []

    for pattern in patterns:
        scale = get_pattern_scale(pattern, df, column)
        if scale > 0:
            scales.append(scale)
            counts.append(len(pattern.get('points', [])))

    if len(scales) >= 2:
        try:
            # Log-log regression
            log_scales = np.log(scales)
            log_counts = np.log(counts)

            # Simple linear regression
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                return abs(slope)
        except Exception:
            pass

    return 1.5  # Default fractal dimension


def calculate_divergence_resolution(pattern1: Dict, pattern2: Dict,
                                  df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate probability of divergence resolution in favor of each pattern."""
    conf1 = pattern1.get('confidence', 0.5)
    conf2 = pattern2.get('confidence', 0.5)

    # Larger timeframes typically dominate
    scale1 = get_timeframe_scale(pattern1.get('timeframe', 'day'))
    scale2 = get_timeframe_scale(pattern2.get('timeframe', 'day'))

    timeframe_factor = scale1 / (scale1 + scale2)

    # Recent momentum
    momentum1 = calculate_pattern_momentum(pattern1, df, column)
    momentum2 = calculate_pattern_momentum(pattern2, df, column)

    momentum_factor = abs(momentum1) / (abs(momentum1) + abs(momentum2) + 0.001)

    # Combined probability
    prob1 = (conf1 * 0.3 + timeframe_factor * 0.4 + momentum_factor * 0.3)
    prob2 = 1 - prob1

    return {
        'pattern1_probability': prob1,
        'pattern2_probability': prob2,
        'likely_winner': 'pattern1' if prob1 > prob2 else 'pattern2'
    }


def calculate_pattern_momentum(pattern: Dict, df: pd.DataFrame, column: str) -> float:
    """Calculate momentum of pattern (rate of price change over time)."""
    wave_points = pattern.get('points', [])
    if len(wave_points) < 2:
        return 0

    try:
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]

        price_change = prices[-1] - prices[0]
        time_change = (dates[-1] - dates[0]).days

        if time_change > 0:
            return price_change / time_change
    except (KeyError, IndexError):
        pass

    return 0


def calculate_key_levels_from_interactions(interactions: Dict[str, Any],
                                         df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    """Extract key price levels from pattern interactions."""
    levels = []

    # From clusters
    for cluster in interactions.get('pattern_clusters', []):
        levels.append({
            'price': cluster['target_price'],
            'type': 'cluster_target',
            'strength': cluster['cluster_strength']
        })

    # From convergences
    for convergence in interactions.get('convergences', []):
        if convergence.get('target_confluence'):
            levels.append({
                'price': convergence['target_confluence'],
                'type': 'confluence_target',
                'strength': convergence['combined_confidence']
            })

    # Remove duplicates and merge strength
    unique_levels = []
    for level in levels:
        is_duplicate = False
        for existing in unique_levels:
            if abs(level['price'] - existing['price']) / existing['price'] < 0.01:
                # Merge strength
                existing['strength'] = max(existing['strength'], level['strength'])
                is_duplicate = True
                break

        if not is_duplicate:
            unique_levels.append(level)

    return sorted(unique_levels, key=lambda x: x['strength'], reverse=True)


def calculate_time_projections(interactions: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate time projections from pattern interactions."""
    projections = {
        'primary_completion': None,
        'alternative_completions': [],
        'critical_dates': []
    }

    # Analyze nested structures for timing
    for nested in interactions.get('nested_structures', []):
        if nested['nesting_type'] == 'subwave_completion':
            # Estimate completion based on parent pattern
            parent_completion = estimate_pattern_completion_time(
                nested['parent_pattern'], df
            )
            if parent_completion:
                projections['critical_dates'].append({
                    'date': parent_completion,
                    'significance': 'major_pattern_completion',
                    'confidence': nested['fractal_confidence']
                })

    # Sort by date
    projections['critical_dates'].sort(key=lambda x: x['date'])

    # Set primary completion
    if projections['critical_dates']:
        projections['primary_completion'] = projections['critical_dates'][0]['date']

    return projections


def estimate_pattern_completion_time(pattern_info: Dict, df: pd.DataFrame) -> pd.Timestamp:
    """Estimate when a pattern might complete based on current wave position."""
    wave_position = pattern_info.get('wave_position', '')

    # Average duration per wave (simplified)
    if 'wave_5' in wave_position:
        # Pattern nearly complete
        return df.index[-1] + pd.Timedelta(days=5)
    elif 'wave_4' in wave_position:
        return df.index[-1] + pd.Timedelta(days=15)
    elif 'wave_3' in wave_position:
        return df.index[-1] + pd.Timedelta(days=30)
    else:
        return df.index[-1] + pd.Timedelta(days=60)


def calculate_overall_forecast_confidence(interactions: Dict[str, Any]) -> float:
    """Calculate overall confidence in the composite forecast."""
    confidence_factors = []

    # Pattern clusters add confidence
    cluster_count = len(interactions.get('pattern_clusters', []))
    confidence_factors.append(min(cluster_count / 3, 1.0))

    # Convergences add confidence
    convergence_count = len(interactions.get('convergences', []))
    confidence_factors.append(min(convergence_count / 5, 1.0))

    # Divergences reduce confidence
    divergence_count = len(interactions.get('divergences', []))
    confidence_factors.append(max(1 - divergence_count / 10, 0))

    # Fractal relationships add confidence
    fractal_count = len(interactions.get('fractal_relationships', []))
    confidence_factors.append(min(fractal_count / 4, 1.0))

    # Weight and combine
    weights = [0.3, 0.3, 0.2, 0.2]
    overall = sum(f * w for f, w in zip(confidence_factors, weights))

    return min(overall * 0.9, 0.95)  # Cap at 95%


def extract_all_patterns(patterns_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all patterns from the patterns data structure."""
    all_patterns = []

    # Handle both old and new structure
    if 'patterns_by_timeframe' in patterns_data:
        for timeframe, tf_data in patterns_data['patterns_by_timeframe'].items():
            for pattern in tf_data.get('multiple_patterns', []):
                pattern['timeframe'] = timeframe
                all_patterns.append(pattern)
    elif 'multiple_patterns' in patterns_data:
        all_patterns = patterns_data['multiple_patterns']

    return all_patterns
