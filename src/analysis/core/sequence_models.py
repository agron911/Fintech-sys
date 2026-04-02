"""
Wave Sequence Data Models
=========================

This module provides data models for Elliott Wave sequence analysis,
including wave sequences, multi-pattern sequences, and pattern relationships.

Classes
-------
PatternRelationship : Enum
    Types of relationships between patterns
WaveSequence : dataclass
    Enhanced wave sequence with scoring and validation
MultiPatternSequence : dataclass
    Sequence supporting multiple patterns and relationships

Examples
--------
>>> from sequence_models import WaveSequence, PatternRelationship
>>> wave = WaveSequence(
...     points=np.array([0, 10, 15, 30, 35, 50]),
...     sequence_type='impulse',
...     confidence=0.75,
...     validation_details={'wave2_retracement': 0.618}
... )
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PatternRelationship(Enum):
    """
    Types of relationships between patterns.

    Attributes
    ----------
    PARENT_CHILD : str
        One pattern is a subwave of another (fractal relationship)
    SIBLING : str
        Patterns exist at the same hierarchical level
    CONFLICTING : str
        Patterns suggest opposing outcomes
    SUPPORTING : str
        Patterns reinforce the same outcome
    INDEPENDENT : str
        Patterns exist independently without interaction

    Notes
    -----
    Pattern relationships affect composite confidence calculations.
    PARENT_CHILD and SUPPORTING relationships increase confidence,
    while CONFLICTING relationships decrease it.
    """
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    CONFLICTING = "conflicting"
    SUPPORTING = "supporting"
    INDEPENDENT = "independent"


@dataclass
class WaveSequence:
    """
    Enhanced wave sequence with scoring and validation.

    Attributes
    ----------
    points : np.ndarray
        Array of wave point indices in the price data
    sequence_type : str
        Type of wave sequence: 'impulse' or 'corrective'
    confidence : float
        Confidence score [0, 1] for this wave sequence
    validation_details : Dict[str, Any]
        Dictionary containing validation metrics and details
    alternative_points : List[np.ndarray], optional
        List of alternative wave point arrays with lower confidence

    Methods
    -------
    __post_init__()
        Initialize alternative_points list if not provided

    Examples
    --------
    >>> wave_seq = WaveSequence(
    ...     points=np.array([0, 10, 15, 30, 35, 50]),
    ...     sequence_type='impulse',
    ...     confidence=0.75,
    ...     validation_details={
    ...         'wave2_retracement': 0.618,
    ...         'wave3_extension': 1.618
    ...     }
    ... )
    >>> print(f"Confidence: {wave_seq.confidence}")
    """
    points: np.ndarray
    sequence_type: str  # 'impulse' or 'corrective'
    confidence: float
    validation_details: Dict[str, Any]
    alternative_points: List[np.ndarray] = None

    def __post_init__(self):
        """Initialize alternative_points list if not provided."""
        if self.alternative_points is None:
            self.alternative_points = []


@dataclass
class MultiPatternSequence:
    """
    Enhanced sequence supporting multiple patterns and relationships.

    This class manages a primary pattern along with related patterns,
    tracking their relationships and calculating composite confidence.

    Attributes
    ----------
    primary_pattern : WaveSequence
        The main wave sequence pattern
    related_patterns : List[WaveSequence]
        List of related wave patterns
    relationships : Dict[str, PatternRelationship]
        Mapping of pattern IDs to their relationship types
    composite_confidence : float
        Overall confidence considering all patterns
    interaction_score : float
        Score representing pattern interaction strength
    validation_summary : Dict[str, Any]
        Summary of validation across all patterns

    Methods
    -------
    add_related_pattern(pattern, relationship)
        Add a related pattern with its relationship type
    _recalculate_composite_confidence()
        Recalculate composite confidence based on all patterns

    Examples
    --------
    >>> primary = WaveSequence(...)
    >>> multi_seq = MultiPatternSequence(primary_pattern=primary)
    >>> supporting = WaveSequence(...)
    >>> multi_seq.add_related_pattern(supporting, PatternRelationship.SUPPORTING)
    >>> print(f"Composite confidence: {multi_seq.composite_confidence}")

    Notes
    -----
    Composite confidence is calculated using weighted averaging:
    - PARENT_CHILD: 1.2x weight
    - SUPPORTING: 1.1x weight
    - SIBLING: 1.0x weight
    - INDEPENDENT: 0.9x weight
    - CONFLICTING: 0.7x weight
    """
    primary_pattern: 'WaveSequence'
    related_patterns: List['WaveSequence'] = field(default_factory=list)
    relationships: Dict[str, PatternRelationship] = field(default_factory=dict)
    composite_confidence: float = 0.0
    interaction_score: float = 0.0
    validation_summary: Dict[str, Any] = field(default_factory=dict)

    def add_related_pattern(self, pattern: 'WaveSequence',
                          relationship: PatternRelationship) -> None:
        """
        Add a related pattern with its relationship type.

        Parameters
        ----------
        pattern : WaveSequence
            The wave sequence to add as a related pattern
        relationship : PatternRelationship
            Type of relationship between this pattern and the primary

        Notes
        -----
        Automatically recalculates composite confidence after adding pattern.
        Pattern ID is generated as "{sequence_type}_{count}".
        """
        self.related_patterns.append(pattern)
        pattern_id = f"{pattern.sequence_type}_{len(self.related_patterns)}"
        self.relationships[pattern_id] = relationship
        self._recalculate_composite_confidence()

        logger.debug(f"Added {relationship.value} pattern, "
                    f"new composite confidence: {self.composite_confidence:.2f}")

    def _recalculate_composite_confidence(self) -> None:
        """
        Recalculate composite confidence based on all patterns.

        Uses weighted averaging where relationship type determines weight.
        More supportive relationships increase confidence, conflicting
        relationships decrease it.

        Notes
        -----
        Composite confidence is capped at 0.95 to maintain realism.
        If no related patterns exist, composite confidence equals
        primary pattern confidence.
        """
        if not self.related_patterns:
            self.composite_confidence = self.primary_pattern.confidence
            return

        # Weight relationships
        relationship_weights = {
            PatternRelationship.PARENT_CHILD: 1.2,
            PatternRelationship.SUPPORTING: 1.1,
            PatternRelationship.SIBLING: 1.0,
            PatternRelationship.INDEPENDENT: 0.9,
            PatternRelationship.CONFLICTING: 0.7
        }

        # Calculate weighted confidence
        total_weight = 1.0  # Primary pattern
        weighted_confidence = self.primary_pattern.confidence

        for i, pattern in enumerate(self.related_patterns):
            pattern_id = f"{pattern.sequence_type}_{i+1}"
            rel_type = self.relationships.get(
                pattern_id,
                PatternRelationship.INDEPENDENT
            )
            weight = relationship_weights[rel_type]

            total_weight += weight
            weighted_confidence += pattern.confidence * weight

        self.composite_confidence = min(weighted_confidence / total_weight, 0.95)

        logger.debug(f"Recalculated composite confidence: {self.composite_confidence:.2f} "
                    f"from {len(self.related_patterns) + 1} patterns")
