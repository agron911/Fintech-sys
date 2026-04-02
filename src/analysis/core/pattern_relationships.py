"""
Multi-Pattern Relationship and Interaction Analysis (REFACTORED)
=================================================================

This module provides a consolidated public API for pattern relationship analysis,
integrating multi-timeframe pattern relationships, signal generation, and
composite forecasting across multiple specialized submodules.

Features:
---------
1. Multi-Timeframe Pattern Relationships:
   - Pattern confirmations and conflicts
   - Wave structure compatibility analysis
   - Fibonacci relationship analysis
   - Volume pattern alignment
   - Trading signal generation
   - Risk assessment

2. Pattern Interactions and Forecasting:
   - Pattern clusters (multiple patterns pointing to same outcome)
   - Pattern divergences (conflicting signals)
   - Pattern convergences (aligned signals)
   - Nested structures (patterns within patterns)
   - Fractal relationships (self-similar patterns)
   - Composite forecasting

Main Functions:
---------------
detect_multiple_pattern_relationships : Analyze multi-timeframe relationships
analyze_pattern_interactions : Analyze pattern interactions and clustering
generate_trading_signals : Generate trading signals from pattern relationships
generate_composite_forecast : Generate integrated forecast from all patterns
assess_pattern_risks : Assess risks based on pattern conflicts

Module Architecture:
--------------------
- core_relationships.py : Core multi-pattern analysis
- compatibility_analysis.py : Wave structure, Fibonacci, volume analysis
- signal_generation.py : Trading signals and risk assessment
- interaction_analysis.py : Clustering, divergences, fractals, forecasting

Examples:
---------
>>> from pattern_relationships import detect_multiple_pattern_relationships
>>> relationships = detect_multiple_pattern_relationships(patterns_dict, df, column='close')
>>> signals = relationships['trading_signals']
>>>
>>> from pattern_relationships import analyze_pattern_interactions
>>> interactions = analyze_pattern_interactions(df, patterns_data, column='close')
>>> forecast = interactions['composite_forecast']
"""

# Import public API from submodules
from .core_relationships import (
    detect_multiple_pattern_relationships,
    analyze_complex_pattern_pair,
    determine_complex_relationship_type
)

from .interaction_analysis import (
    analyze_pattern_interactions,
    identify_pattern_clusters,
    identify_pattern_divergences,
    identify_pattern_convergences,
    analyze_nested_structures,
    identify_fractal_relationships,
    generate_composite_forecast
)

from .signal_generation import (
    generate_trading_signals,
    assess_pattern_risks
)

from .compatibility_analysis import (
    analyze_wave_structure_compatibility,
    analyze_fibonacci_relationships,
    analyze_volume_pattern_alignment,
    calculate_complexity_score,
    calculate_complex_alignment_score
)

# Expose all public functions
__all__ = [
    'detect_multiple_pattern_relationships',
    'analyze_complex_pattern_pair',
    'determine_complex_relationship_type',
    'analyze_pattern_interactions',
    'identify_pattern_clusters',
    'identify_pattern_divergences',
    'identify_pattern_convergences',
    'analyze_nested_structures',
    'identify_fractal_relationships',
    'generate_composite_forecast',
    'generate_trading_signals',
    'assess_pattern_risks',
    'analyze_wave_structure_compatibility',
    'analyze_fibonacci_relationships',
    'analyze_volume_pattern_alignment',
    'calculate_complexity_score',
    'calculate_complex_alignment_score',
]
