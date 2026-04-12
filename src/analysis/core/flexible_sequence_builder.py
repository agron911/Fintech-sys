import numpy as np
import pandas as pd
import itertools
from typing import Tuple, List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import warnings
import hashlib
from functools import lru_cache
from enum import Enum
warnings.filterwarnings('ignore')

class PatternRelationship(Enum):
    """Types of relationships between patterns"""
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    CONFLICTING = "conflicting"
    SUPPORTING = "supporting"
    INDEPENDENT = "independent"

@dataclass
class WaveSequence:
    """Enhanced wave sequence with scoring and validation."""
    points: np.ndarray
    sequence_type: str  # 'impulse' or 'corrective'
    confidence: float
    validation_details: Dict[str, Any]
    alternative_points: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.alternative_points is None:
            self.alternative_points = []

@dataclass
class MultiPatternSequence:
    """Enhanced sequence supporting multiple patterns and relationships"""
    primary_pattern: 'WaveSequence'
    related_patterns: List['WaveSequence'] = field(default_factory=list)
    relationships: Dict[str, PatternRelationship] = field(default_factory=dict)
    composite_confidence: float = 0.0
    interaction_score: float = 0.0
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_related_pattern(self, pattern: 'WaveSequence', relationship: PatternRelationship):
        """Add a related pattern with its relationship type"""
        self.related_patterns.append(pattern)
        pattern_id = f"{pattern.sequence_type}_{len(self.related_patterns)}"
        self.relationships[pattern_id] = relationship
        self._recalculate_composite_confidence()
    
    def _recalculate_composite_confidence(self):
        """Recalculate composite confidence based on all patterns"""
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
            rel_type = self.relationships.get(pattern_id, PatternRelationship.INDEPENDENT)
            weight = relationship_weights[rel_type]
            
            total_weight += weight
            weighted_confidence += pattern.confidence * weight
        
        self.composite_confidence = min(weighted_confidence / total_weight, 0.95)

class FlexibleSequenceBuilder:
    """
    Enhanced sequence builder that handles real-world wave patterns with flexibility.
    """
    def __init__(self, 
                 impulse_points: int = 6,  # Start + 5 wave endpoints
                 corrective_points: int = 4,  # Start + 3 wave endpoints
                 max_alternatives: int = 5,
                 min_confidence_threshold: float = 0.2):
        self.impulse_points = impulse_points
        self.corrective_points = corrective_points
        self.max_alternatives = max_alternatives
        self.min_confidence_threshold = min_confidence_threshold
    
    def build_wave_sequence_flexible(self, 
                                   df: pd.DataFrame, 
                                   labeled_points: List[Tuple[int, str]], 
                                   start_idx: int, 
                                   end_idx: int, 
                                   wave_type: str = 'impulse',
                                   column: str = 'close') -> Optional[WaveSequence]:
        expected_count = self.impulse_points if wave_type == 'impulse' else self.corrective_points
        candidate_points = [pt for pt in labeled_points if start_idx <= pt[0] <= end_idx]
        if len(candidate_points) < expected_count:
            return None
        valid_sequences = self._generate_valid_sequences(
            df, candidate_points, expected_count, wave_type, column
        )
        if not valid_sequences:
            return self._try_relaxed_sequence(
                df, candidate_points, expected_count, wave_type, column
            )
        best_sequence = max(valid_sequences, key=lambda x: x.confidence)
        alternatives = [seq.points for seq in valid_sequences[1:self.max_alternatives+1]]
        best_sequence.alternative_points = alternatives
        return best_sequence
    def _generate_valid_sequences(self,
                                df: pd.DataFrame,
                                candidate_points: List[Tuple[int, str]],
                                expected_count: int,
                                wave_type: str,
                                column: str) -> List[WaveSequence]:
        valid_sequences = []

        # PERFORMANCE OPTIMIZATION: Add early termination to avoid processing millions of combinations
        max_combinations_to_check = 50000  # Limit total combinations checked
        combinations_checked = 0

        # PERFORMANCE OPTIMIZATION: Early exit if we find enough high-confidence patterns
        target_sequences = 5  # Target number of good sequences
        high_confidence_threshold = 0.7  # What we consider "high confidence"

        for combo in itertools.combinations(range(len(candidate_points)), expected_count):
            combinations_checked += 1

            # PERFORMANCE: Stop if we've checked too many combinations
            if combinations_checked > max_combinations_to_check:
                break

            # PERFORMANCE: Early exit if we have enough high-quality patterns
            if len(valid_sequences) >= target_sequences:
                high_conf_count = sum(1 for seq in valid_sequences if seq.confidence >= high_confidence_threshold)
                if high_conf_count >= 3:  # At least 3 high-confidence patterns
                    break

            sequence = [candidate_points[i] for i in combo]
            # Enforce alternation and correct starting type for impulse
            if self._is_alternating(sequence):
                if wave_type == 'impulse' and len(sequence) > 1:
                    prices = df[column].iloc[[pt[0] for pt in sequence]].values
                    trend_up = prices[1] > prices[0]
                    expected_start = 'trough' if trend_up else 'peak'
                    if sequence[0][1] != expected_start:
                        continue
                wave_points = np.array([pt[0] for pt in sequence])
                confidence, details = self._score_sequence(df, wave_points, wave_type, column)
                if confidence >= self.min_confidence_threshold:
                    valid_sequences.append(WaveSequence(
                        points=wave_points,
                        sequence_type=wave_type,
                        confidence=confidence,
                        validation_details=details
                    ))

        # Only try extended sequences if we don't have enough good patterns and haven't exhausted our budget
        if len(valid_sequences) < 3 and len(candidate_points) > expected_count and combinations_checked < max_combinations_to_check:
            valid_sequences.extend(
                self._try_extended_sequences(df, candidate_points, expected_count, wave_type, column, max_combinations_to_check - combinations_checked)
            )
        return sorted(valid_sequences, key=lambda x: x.confidence, reverse=True)
    def _try_extended_sequences(self,
                              df: pd.DataFrame,
                              candidate_points: List[Tuple[int, str]],
                              expected_count: int,
                              wave_type: str,
                              column: str,
                              remaining_budget: int = 10000) -> List[WaveSequence]:
        extended_sequences = []
        combinations_checked = 0

        for extra in range(1, min(4, len(candidate_points) - expected_count + 1)):
            extended_count = expected_count + extra
            for combo in itertools.combinations(range(len(candidate_points)), extended_count):
                combinations_checked += 1

                # PERFORMANCE: Stop if budget exhausted
                if combinations_checked > remaining_budget:
                    return extended_sequences

                sequence = [candidate_points[i] for i in combo]
                if self._is_alternating(sequence):
                    best_subset = self._find_best_subset(
                        df, sequence, expected_count, wave_type, column
                    )
                    if best_subset:
                        extended_sequences.append(best_subset)

                        # PERFORMANCE: Stop if we found enough good extended sequences
                        if len(extended_sequences) >= 2:
                            return extended_sequences

        return extended_sequences
    def _find_best_subset(self,
                         df: pd.DataFrame,
                         extended_sequence: List[Tuple[int, str]],
                         target_count: int,
                         wave_type: str,
                         column: str) -> Optional[WaveSequence]:
        best_sequence = None
        best_confidence = 0
        first_point = extended_sequence[0]
        last_point = extended_sequence[-1]
        middle_points = extended_sequence[1:-1]
        for combo in itertools.combinations(range(len(middle_points)), target_count - 2):
            subset = [first_point] + [middle_points[i] for i in combo] + [last_point]
            if self._is_alternating(subset):
                wave_points = np.array([pt[0] for pt in subset])
                confidence, details = self._score_sequence(df, wave_points, wave_type, column)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_sequence = WaveSequence(
                        points=wave_points,
                        sequence_type=wave_type,
                        confidence=confidence,
                        validation_details=details
                    )
        return best_sequence
    def _try_relaxed_sequence(self,
                            df: pd.DataFrame,
                            candidate_points: List[Tuple[int, str]],
                            expected_count: int,
                            wave_type: str,
                            column: str) -> Optional[WaveSequence]:
        relaxed_sequences = []
        for combo in itertools.combinations(range(len(candidate_points)), expected_count):
            sequence = [candidate_points[i] for i in combo]
            wave_points = np.array([pt[0] for pt in sequence])
            alternation_score = self._calculate_alternation_score(sequence)
            if alternation_score >= 0.7:
                confidence, details = self._score_sequence(df, wave_points, wave_type, column)
                confidence *= alternation_score
                if confidence >= self.min_confidence_threshold * 0.5:
                    relaxed_sequences.append(WaveSequence(
                        points=wave_points,
                        sequence_type=wave_type,
                        confidence=confidence,
                        validation_details={**details, 'relaxed_alternation': True}
                    ))
        if relaxed_sequences:
            return max(relaxed_sequences, key=lambda x: x.confidence)
        if len(candidate_points) >= expected_count - 1:
            wave_points = np.array([pt[0] for pt in candidate_points])
            confidence, details = self._score_sequence(df, wave_points, wave_type, column)
            confidence *= 0.5
            if confidence >= self.min_confidence_threshold * 0.3:
                return WaveSequence(
                    points=wave_points,
                    sequence_type=wave_type,
                    confidence=confidence,
                    validation_details={**details, 'relaxed_count': True}
                )
        return None
    def _score_sequence(self, 
                       df: pd.DataFrame, 
                       wave_points: np.ndarray, 
                       wave_type: str,
                       column: str) -> Tuple[float, Dict[str, Any]]:
        if len(wave_points) < 3:
            return 0.0, {'error': 'insufficient_points'}
        prices = df[column].iloc[wave_points].values
        dates = df.index[wave_points]
        score = 0.0
        details = {}
        try:
            progression_score = self._score_wave_progression(prices, wave_type)
            score += progression_score * 0.4
            details['progression_score'] = progression_score
            proportion_score = self._score_wave_proportions(prices, wave_type)
            score += proportion_score * 0.3
            details['proportion_score'] = proportion_score
            trend_score = self._score_trend_consistency(prices, wave_type)
            score += trend_score * 0.2
            details['trend_score'] = trend_score
            time_score = self._score_time_relationships(dates, wave_type)
            score += time_score * 0.1
            details['time_score'] = time_score
            if wave_type == 'impulse' and len(wave_points) == 6:
                if len(prices) >= 4:
                    wave_1 = abs(prices[1] - prices[0])
                    wave_3 = abs(prices[3] - prices[2])
                    if wave_3 > wave_1 * 1.5:
                        score += 0.1
                        details['extended_wave_3'] = True
            score = min(score, 1.0)
            if len(wave_points) < (self.impulse_points if wave_type == 'impulse' else self.corrective_points):
                score *= 0.8
                details['short_sequence_penalty'] = True
            # Add penalty for sequences that are too perfect
            if score > 0.95:
                score = 0.95 - (score - 0.95) * 0.5  # Penalize over-confidence
            # Add reality check - real market data rarely produces perfect patterns
            reality_adjustment = 0.85  # Maximum realistic confidence
            score = min(score * reality_adjustment, 1.0)
            # Post-validation: additional quality checks
            quality_metrics = validate_wave_sequence_quality(df, wave_points, column)
            score += quality_metrics.get('quality_score', 0)
            details['quality_metrics'] = quality_metrics
        except Exception as e:
            details['scoring_error'] = str(e)
            score = 0.1
        return score, details
    def _score_wave_progression(self, prices: np.ndarray, wave_type: str) -> float:
        if wave_type == 'impulse' and len(prices) >= 4:
            score = 0.0
            wave_1 = prices[1] - prices[0]
            trend_up = wave_1 > 0
            if len(prices) > 3:
                wave_3 = prices[3] - prices[2]
                if (wave_3 > 0) == trend_up:
                    score += 0.4
                    if abs(wave_3) >= abs(wave_1):
                        score += 0.3
                wave_2 = prices[2] - prices[1]
                if (wave_2 > 0) != trend_up:
                    score += 0.2
                if len(prices) > 4:
                    wave_4 = prices[4] - prices[3]
                    if (wave_4 > 0) != trend_up:
                        score += 0.1
            return score
        elif wave_type == 'corrective':
            if len(prices) >= 3:
                return 0.7
        return 0.5
    def _score_wave_proportions(self, prices: np.ndarray, wave_type: str) -> float:
        if len(prices) < 4:
            return 0.5
        score = 0.0
        if wave_type == 'impulse':
            waves = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
            wave_sizes = [abs(w) for w in waves]
            if len(wave_sizes) >= 3:
                if wave_sizes[2] != min(wave_sizes[:3]):
                    score += 0.5
                impulse_waves = [0, 2]
                if len(wave_sizes) > 4:
                    impulse_waves.append(4)
                impulse_sizes = [wave_sizes[i] for i in impulse_waves if i < len(wave_sizes)]
                if len(impulse_sizes) > 1 and wave_sizes[2] == max(impulse_sizes):
                    score += 0.3
                if len(waves) >= 2 and waves[0] != 0:
                    wave_2_retracement = abs(waves[1] / waves[0])
                    if 0.3 <= wave_2_retracement <= 0.8:
                        score += 0.2
        return min(score, 1.0)
    def _score_trend_consistency(self, prices: np.ndarray, wave_type: str) -> float:
        if len(prices) < 3:
            return 0.5
        if wave_type == 'impulse':
            net_move = prices[-1] - prices[0]
            total_movement = sum(abs(prices[i+1] - prices[i]) for i in range(len(prices) - 1))
            if total_movement > 0:
                efficiency = abs(net_move) / total_movement
                return min(efficiency * 2, 1.0)
        return 0.6
    def _score_time_relationships(self, dates: pd.DatetimeIndex, wave_type: str) -> float:
        if len(dates) < 3:
            return 0.5
        durations = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
        if len(durations) >= 2:
            avg_duration = np.mean(durations)
            duration_std = np.std(durations)
            if avg_duration > 0:
                consistency = 1 - (duration_std / avg_duration)
                return max(0.3, min(consistency, 1.0))
        return 0.5
    def _calculate_alternation_score(self, sequence: List[Tuple[int, str]]) -> float:
        if len(sequence) < 2:
            return 1.0
        alternations = 0
        total_transitions = len(sequence) - 1
        for i in range(total_transitions):
            if sequence[i][1] != sequence[i+1][1]:
                alternations += 1
        return alternations / total_transitions if total_transitions > 0 else 0.0
    def _is_alternating(self, sequence: List[Tuple[int, str]]) -> bool:
        if len(sequence) < 2:
            return True
        return all(
            sequence[i][1] != sequence[i+1][1] 
            for i in range(len(sequence) - 1)
        )

class PatternCache:
    def __init__(self):
        self.cache = {}
    def get_cache_key(self, df: pd.DataFrame, params: Dict) -> str:
        """Generate unique key for caching"""
        data_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:8]
        param_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]
        return f"{data_hash}_{param_hash}"
    @staticmethod
    @lru_cache(maxsize=100)
    def detect_patterns_cached(cache_key: str, *args, **kwargs):
        """Cached pattern detection. Implement your detection logic here or call the actual detection function."""
        # Example: return expensive_detection_function(*args, **kwargs)
        pass

class PatternTracker:
    def __init__(self):
        self.patterns = []
    def record_pattern(self, symbol: str, wave_data: Dict, timestamp: pd.Timestamp):
        """Record detected pattern for future validation"""
        pattern_record = {
            'symbol': symbol,
            'timestamp': timestamp,
            'wave_points': wave_data['impulse_wave'],
            'confidence': wave_data['confidence'],
            'wave_type': wave_data['wave_type'],
            'predicted_next': self._predict_next_move(wave_data)
        }
        self.patterns.append(pattern_record)
    def validate_predictions(self, df: pd.DataFrame, lookback_days: int = 30):
        """Check how accurate past predictions were"""
        results = []
        for pattern in self.patterns:
            if (pd.Timestamp.now() - pattern['timestamp']).days > lookback_days:
                actual_move = self._get_actual_move(df, pattern)
                accuracy = self._calculate_accuracy(pattern['predicted_next'], actual_move)
                results.append({
                    'pattern': pattern,
                    'accuracy': accuracy,
                    'actual_vs_predicted': (actual_move, pattern['predicted_next'])
                })
        return results
    def _predict_next_move(self, wave_data: Dict):
        # Stub: implement your prediction logic based on wave_data
        return None
    def _get_actual_move(self, df: pd.DataFrame, pattern: Dict):
        # Stub: implement logic to get actual move after pattern timestamp
        return None
    def _calculate_accuracy(self, predicted, actual):
        # Stub: implement accuracy calculation
        return None

def build_wave_sequence_enhanced(df: pd.DataFrame, 
                               labeled_points: List[Tuple[int, str]], 
                               start_idx: int, 
                               end_idx: int, 
                               wave_type: str = 'impulse',
                               column: str = 'close',
                               max_depth: int = 2) -> Optional[Dict[str, Any]]:
    builder = FlexibleSequenceBuilder()
    sequence_result = builder.build_wave_sequence_flexible(
        df, labeled_points, start_idx, end_idx, wave_type, column
    )
    if sequence_result is None:
        return None
    sequence_points = [(int(idx), 'peak' if i % 2 == 0 else 'trough') 
                      for i, idx in enumerate(sequence_result.points)]
    subwaves = []
    if max_depth > 0 and sequence_result.confidence > 0.5:
        subwaves = build_subwaves_flexible(
            df, labeled_points, sequence_result.points, wave_type, column, max_depth - 1
        )
    return {
        'sequence': sequence_points,
        'subwaves': subwaves,
        'confidence': sequence_result.confidence,
        'validation_details': sequence_result.validation_details,
        'alternatives': sequence_result.alternative_points
    }

def build_subwaves_flexible(df: pd.DataFrame,
                          labeled_points: List[Tuple[int, str]],
                          parent_points: np.ndarray,
                          parent_type: str,
                          column: str,
                          max_depth: int) -> List[Optional[Dict[str, Any]]]:
    if max_depth <= 0 or len(parent_points) < 2:
        return []
    subwaves = []
    builder = FlexibleSequenceBuilder(min_confidence_threshold=0.1)
    for i in range(len(parent_points) - 1):
        sub_start = parent_points[i]
        sub_end = parent_points[i + 1]
        if parent_type == 'impulse':
            sub_type = 'impulse' if i % 2 == 0 else 'corrective'
        else:
            sub_type = 'corrective'
        if sub_end - sub_start > 5:
            sub_result = builder.build_wave_sequence_flexible(
                df, labeled_points, sub_start, sub_end, sub_type, column
            )
            if sub_result and sub_result.confidence > 0.1:
                subwave_points = [(int(idx), 'peak' if j % 2 == 0 else 'trough') 
                                for j, idx in enumerate(sub_result.points)]
                subwaves.append({
                    'sequence': subwave_points,
                    'subwaves': [],
                    'confidence': sub_result.confidence,
                    'validation_details': sub_result.validation_details
                })
            else:
                subwaves.append(None)
        else:
            subwaves.append(None)
    return subwaves

# Test function moved to tests/test_flexible_sequence_builder.py for proper separation of concerns

def validate_wave_sequence_quality(df: pd.DataFrame, wave_points: np.ndarray, 
                                 column: str = 'close') -> Dict[str, Any]:
    """Additional quality checks for wave sequences"""
    prices = df[column].iloc[wave_points].values
    # Check for reasonable wave relationships
    quality_metrics = {
        'wave_1_to_3_ratio': abs((prices[3] - prices[2]) / (prices[1] - prices[0])) if len(prices) > 3 else 0,
        'wave_2_depth': abs((prices[2] - prices[1]) / (prices[1] - prices[0])) if len(prices) > 2 else 0,
        'wave_4_depth': abs((prices[4] - prices[3]) / (prices[3] - prices[2])) if len(prices) > 4 else 0,
        'total_progress': (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 and prices[0] != 0 else 0,
        'wave_efficiency': abs(prices[-1] - prices[0]) / sum(abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)) if len(prices) > 1 and sum(abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)) != 0 else 0
    }
    # Score the quality
    quality_score = 0.0
    # Wave 3 should be 0.618 to 2.618 times wave 1
    if 0.618 <= quality_metrics['wave_1_to_3_ratio'] <= 2.618:
        quality_score += 0.3
    # Wave 2 should retrace 0.236 to 0.786
    if 0.236 <= quality_metrics['wave_2_depth'] <= 0.786:
        quality_score += 0.2
    # Wave efficiency (directional movement vs total movement)
    if quality_metrics['wave_efficiency'] > 0.3:
        quality_score += 0.2
    quality_metrics['quality_score'] = quality_score
    return quality_metrics 

class EnhancedSequenceBuilder:
    """Enhanced builder supporting multiple simultaneous patterns"""
    
    def __init__(self, 
                 impulse_points: int = 6,
                 corrective_points: int = 4,
                 max_alternatives: int = 5,
                 min_confidence_threshold: float = 0.2,
                 enable_multi_pattern: bool = True):
        self.impulse_points = impulse_points
        self.corrective_points = corrective_points
        self.max_alternatives = max_alternatives
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_multi_pattern = enable_multi_pattern
        self.pattern_cache = {}
        
    def build_multi_pattern_sequence(self,
                                   df: pd.DataFrame,
                                   labeled_points: List[Tuple[int, str]],
                                   start_idx: int,
                                   end_idx: int,
                                   column: str = 'close',
                                   max_patterns: int = 5) -> Optional[MultiPatternSequence]:
        """
        Build multiple pattern sequences with relationship analysis
        """
        # Find primary pattern
        primary_sequence = self.build_wave_sequence_flexible(
            df, labeled_points, start_idx, end_idx, 'impulse', column
        )
        
        if not primary_sequence:
            return None
        
        # Create multi-pattern sequence
        multi_sequence = MultiPatternSequence(primary_pattern=primary_sequence)
        
        if not self.enable_multi_pattern:
            return multi_sequence
        
        # Find related patterns
        related_patterns = self._find_related_patterns(
            df, labeled_points, primary_sequence, column, max_patterns
        )
        
        # Add related patterns with relationships
        for pattern, relationship in related_patterns:
            multi_sequence.add_related_pattern(pattern, relationship)
        
        # Validate pattern interactions
        multi_sequence.validation_summary = self._validate_pattern_interactions(
            df, multi_sequence, column
        )
        
        # Calculate interaction score
        multi_sequence.interaction_score = self._calculate_interaction_score(
            multi_sequence
        )
        
        return multi_sequence
    
    def build_wave_sequence_flexible(self,
                                   df: pd.DataFrame,
                                   labeled_points: List[Tuple[int, str]],
                                   start_idx: int,
                                   end_idx: int,
                                   sequence_type: str,
                                   column: str = 'close') -> Optional['WaveSequence']:
        """
        Build a flexible wave sequence with enhanced validation
        """
        # Filter labeled points within range
        filtered_points = [
            (idx, label) for idx, label in labeled_points
            if start_idx <= idx <= end_idx
        ]
        
        if len(filtered_points) < 3:
            return None
        
        # Extract indices and labels
        indices = [idx for idx, _ in filtered_points]
        labels = [label for _, label in filtered_points]
        
        # Build sequence based on type
        if sequence_type == 'impulse':
            return self._build_impulse_sequence(df, indices, labels, column)
        elif sequence_type == 'corrective':
            return self._build_corrective_sequence(df, indices, labels, column)
        else:
            return None
    
    def _build_impulse_sequence(self,
                               df: pd.DataFrame,
                               indices: List[int],
                               labels: List[str],
                               column: str) -> Optional['WaveSequence']:
        """Build impulse wave sequence"""
        if len(indices) < 5:
            return None
        
        # Find impulse pattern (5 waves)
        for i in range(len(indices) - 4):
            wave_points = indices[i:i+5]
            wave_labels = labels[i:i+5]
            
            # Validate impulse structure
            if self._validate_impulse_structure(df, wave_points, wave_labels, column):
                confidence = self._calculate_impulse_confidence(df, wave_points, column)
                
                if confidence >= self.min_confidence_threshold:
                    return WaveSequence(
                        points=wave_points,
                        labels=wave_labels,
                        sequence_type='impulse',
                        confidence=confidence
                    )
        
        return None
    
    def _build_corrective_sequence(self,
                                  df: pd.DataFrame,
                                  indices: List[int],
                                  labels: List[str],
                                  column: str) -> Optional['WaveSequence']:
        """Build corrective wave sequence"""
        if len(indices) < 3:
            return None
        
        # Find corrective pattern (3 waves)
        for i in range(len(indices) - 2):
            wave_points = indices[i:i+3]
            wave_labels = labels[i:i+3]
            
            # Validate corrective structure
            if self._validate_corrective_structure(df, wave_points, wave_labels, column):
                confidence = self._calculate_corrective_confidence(df, wave_points, column)
                
                if confidence >= self.min_confidence_threshold:
                    return WaveSequence(
                        points=wave_points,
                        labels=wave_labels,
                        sequence_type='corrective',
                        confidence=confidence
                    )
        
        return None
    
    def _find_related_patterns(self,
                             df: pd.DataFrame,
                             labeled_points: List[Tuple[int, str]],
                             primary: 'WaveSequence',
                             column: str,
                             max_patterns: int) -> List[Tuple['WaveSequence', PatternRelationship]]:
        """Find patterns related to the primary pattern"""
        related = []
        
        # 1. Look for parent patterns (containing primary)
        parent_patterns = self._find_parent_patterns(
            df, labeled_points, primary, column
        )
        related.extend([(p, PatternRelationship.PARENT_CHILD) for p in parent_patterns])
        
        # 2. Look for child patterns (within primary)
        child_patterns = self._find_child_patterns(
            df, labeled_points, primary, column
        )
        related.extend([(p, PatternRelationship.PARENT_CHILD) for p in child_patterns])
        
        # 3. Look for sibling patterns (same level, adjacent)
        sibling_patterns = self._find_sibling_patterns(
            df, labeled_points, primary, column
        )
        related.extend([(p, PatternRelationship.SIBLING) for p in sibling_patterns])
        
        # 4. Look for supporting patterns (confirming direction)
        supporting_patterns = self._find_supporting_patterns(
            df, labeled_points, primary, column
        )
        related.extend([(p, PatternRelationship.SUPPORTING) for p in supporting_patterns])
        
        # Sort by confidence and limit
        related.sort(key=lambda x: x[0].confidence, reverse=True)
        return related[:max_patterns]
    
    def _find_parent_patterns(self,
                            df: pd.DataFrame,
                            labeled_points: List[Tuple[int, str]],
                            child: 'WaveSequence',
                            column: str) -> List['WaveSequence']:
        """Find larger patterns that contain the child pattern"""
        parent_patterns = []
        child_start = child.points[0]
        child_end = child.points[-1]
        
        # Look for patterns that start before and end after child
        extended_start = max(0, child_start - 100)
        extended_end = min(len(df) - 1, child_end + 100)
        
        # Try different wave types
        for wave_type in ['impulse', 'corrective']:
            parent = self.build_wave_sequence_flexible(
                df, labeled_points, extended_start, extended_end, wave_type, column
            )
            
            if parent and self._contains_pattern(parent, child):
                parent_patterns.append(parent)
        
        return parent_patterns
    
    def _find_child_patterns(self,
                           df: pd.DataFrame,
                           labeled_points: List[Tuple[int, str]],
                           parent: 'WaveSequence',
                           column: str) -> List['WaveSequence']:
        """Find smaller patterns within the parent pattern"""
        child_patterns = []
        
        # Check each wave segment for sub-patterns
        for i in range(len(parent.points) - 1):
            segment_start = parent.points[i]
            segment_end = parent.points[i + 1]
            
            # Skip very small segments
            if segment_end - segment_start < 10:
                continue
            
            # Look for patterns within this segment
            for wave_type in ['impulse', 'corrective']:
                child = self.build_wave_sequence_flexible(
                    df, labeled_points, segment_start, segment_end, wave_type, column
                )
                
                if child and child.confidence >= self.min_confidence_threshold * 0.7:
                    child_patterns.append(child)
        
        return child_patterns
    
    def _find_sibling_patterns(self,
                             df: pd.DataFrame,
                             labeled_points: List[Tuple[int, str]],
                             pattern: 'WaveSequence',
                             column: str) -> List['WaveSequence']:
        """Find patterns at the same level, before or after the given pattern"""
        sibling_patterns = []
        pattern_start = pattern.points[0]
        pattern_end = pattern.points[-1]
        
        # Look before pattern
        if pattern_start > 50:
            before_end = pattern_start - 1
            before_start = max(0, before_end - (pattern_end - pattern_start))
            
            sibling_before = self.build_wave_sequence_flexible(
                df, labeled_points, before_start, before_end, 
                pattern.sequence_type, column
            )
            
            if sibling_before:
                sibling_patterns.append(sibling_before)
        
        # Look after pattern
        if pattern_end < len(df) - 50:
            after_start = pattern_end + 1
            after_end = min(len(df) - 1, after_start + (pattern_end - pattern_start))
            
            sibling_after = self.build_wave_sequence_flexible(
                df, labeled_points, after_start, after_end,
                pattern.sequence_type, column
            )
            
            if sibling_after:
                sibling_patterns.append(sibling_after)
        
        return sibling_patterns
    
    def _find_supporting_patterns(self,
                                df: pd.DataFrame,
                                labeled_points: List[Tuple[int, str]],
                                pattern: 'WaveSequence',
                                column: str) -> List['WaveSequence']:
        """Find patterns that support (confirm) the main pattern"""
        supporting_patterns = []
        
        # Look for patterns with similar directional bias
        pattern_direction = self._get_pattern_direction(df, pattern, column)
        
        # Search in overlapping regions
        search_ranges = [
            (max(0, pattern.points[0] - 50), min(len(df) - 1, pattern.points[-1] + 50)),
            (pattern.points[0], pattern.points[-1])
        ]
        
        for start, end in search_ranges:
            for wave_type in ['impulse', 'corrective']:
                candidate = self.build_wave_sequence_flexible(
                    df, labeled_points, start, end, wave_type, column
                )
                
                if candidate and candidate != pattern:
                    candidate_direction = self._get_pattern_direction(df, candidate, column)
                    
                    # Same direction = supporting
                    if pattern_direction * candidate_direction > 0:
                        supporting_patterns.append(candidate)
        
        return supporting_patterns
    
    def _contains_pattern(self, parent: 'WaveSequence', child: 'WaveSequence') -> bool:
        """Check if parent pattern contains child pattern"""
        return (parent.points[0] <= child.points[0] and 
                parent.points[-1] >= child.points[-1])
    
    def _get_pattern_direction(self, df: pd.DataFrame, 
                             pattern: 'WaveSequence', column: str) -> float:
        """Get overall direction of pattern (1 for up, -1 for down)"""
        prices = df[column].iloc[pattern.points].values
        if len(prices) < 2:
            return 0
        
        net_change = prices[-1] - prices[0]
        return 1 if net_change > 0 else -1 if net_change < 0 else 0
    
    def _validate_pattern_interactions(self,
                                     df: pd.DataFrame,
                                     multi_sequence: MultiPatternSequence,
                                     column: str) -> Dict[str, Any]:
        """Validate interactions between patterns"""
        validation = {
            'conflicts': [],
            'confirmations': [],
            'warnings': [],
            'overall_validity': True
        }
        
        # Check for conflicts
        primary_direction = self._get_pattern_direction(
            df, multi_sequence.primary_pattern, column
        )
        
        for i, related in enumerate(multi_sequence.related_patterns):
            pattern_id = f"{related.sequence_type}_{i+1}"
            relationship = multi_sequence.relationships[pattern_id]
            
            if relationship == PatternRelationship.CONFLICTING:
                validation['conflicts'].append({
                    'pattern': pattern_id,
                    'reason': 'Marked as conflicting relationship'
                })
            
            # Check directional conflicts
            related_direction = self._get_pattern_direction(df, related, column)
            
            if (relationship in [PatternRelationship.SUPPORTING, PatternRelationship.PARENT_CHILD] 
                and primary_direction * related_direction < 0):
                validation['warnings'].append({
                    'pattern': pattern_id,
                    'reason': 'Direction mismatch with primary pattern'
                })
        
        # Check for confirmations
        for i, related in enumerate(multi_sequence.related_patterns):
            if multi_sequence.relationships.get(f"{related.sequence_type}_{i+1}") == PatternRelationship.SUPPORTING:
                validation['confirmations'].append({
                    'pattern': f"{related.sequence_type}_{i+1}",
                    'confidence_boost': related.confidence * 0.1
                })
        
        # Overall validity
        validation['overall_validity'] = len(validation['conflicts']) == 0
        
        return validation
    
    def _calculate_interaction_score(self, multi_sequence: MultiPatternSequence) -> float:
        """Calculate overall interaction score for pattern group"""
        score = 0.0
        
        # Base score from primary pattern
        score += multi_sequence.primary_pattern.confidence * 0.5
        
        # Bonus for supporting patterns
        supporting_count = sum(
            1 for rel in multi_sequence.relationships.values()
            if rel == PatternRelationship.SUPPORTING
        )
        score += min(supporting_count * 0.1, 0.3)
        
        # Bonus for parent-child relationships (fractal confirmation)
        parent_child_count = sum(
            1 for rel in multi_sequence.relationships.values()
            if rel == PatternRelationship.PARENT_CHILD
        )
        score += min(parent_child_count * 0.05, 0.15)
        
        # Penalty for conflicts
        conflict_count = sum(
            1 for rel in multi_sequence.relationships.values()
            if rel == PatternRelationship.CONFLICTING
        )
        score -= conflict_count * 0.15
        
        # Validation bonus
        if multi_sequence.validation_summary.get('overall_validity', False):
            score += 0.05
        
        return min(max(score, 0.0), 1.0)
    
    def _validate_impulse_structure(self,
                                   df: pd.DataFrame,
                                   points: List[int],
                                   labels: List[str],
                                   column: str) -> bool:
        """Validate impulse wave structure"""
        if len(points) != 5:
            return False
        
        prices = df[column].iloc[points].values
        
        # Check basic impulse rules
        # Wave 2 should not retrace more than 100% of wave 1
        wave1 = abs(prices[1] - prices[0])
        wave2 = abs(prices[2] - prices[1])
        if wave2 > wave1:
            return False
        
        # Wave 3 should not be the shortest
        wave3 = abs(prices[3] - prices[2])
        wave4 = abs(prices[4] - prices[3])
        if wave3 < wave1 or wave3 < wave4:
            return False
        
        # Wave 4 should not overlap wave 1
        if (prices[3] > prices[1] and prices[4] < prices[1]) or \
           (prices[3] < prices[1] and prices[4] > prices[1]):
            return False
        
        return True
    
    def _validate_corrective_structure(self,
                                      df: pd.DataFrame,
                                      points: List[int],
                                      labels: List[str],
                                      column: str) -> bool:
        """Validate corrective wave structure"""
        if len(points) != 3:
            return False
        
        prices = df[column].iloc[points].values
        
        # Check that wave B doesn't retrace more than 100% of wave A
        wave_a = abs(prices[1] - prices[0])
        wave_b = abs(prices[2] - prices[1])
        
        if wave_b > wave_a:
            return False
        
        return True
    
    def _calculate_impulse_confidence(self,
                                    df: pd.DataFrame,
                                    points: List[int],
                                    column: str) -> float:
        """Calculate confidence for impulse pattern"""
        prices = df[column].iloc[points].values
        
        # Base confidence
        confidence = 0.5
        
        # Check Fibonacci relationships
        wave1 = abs(prices[1] - prices[0])
        wave2 = abs(prices[2] - prices[1])
        wave3 = abs(prices[3] - prices[2])
        wave4 = abs(prices[4] - prices[3])
        
        # Wave 3 to wave 1 ratio (should be 1.618 or 2.618)
        if wave1 > 0:
            ratio = wave3 / wave1
            if 1.5 <= ratio <= 2.7:
                confidence += 0.2
            elif 0.8 <= ratio <= 1.2:
                confidence += 0.1
        
        # Wave 2 to wave 1 ratio (should be 0.618 or 0.5)
        if wave1 > 0:
            ratio = wave2 / wave1
            if 0.5 <= ratio <= 0.7:
                confidence += 0.15
            elif 0.3 <= ratio <= 0.5:
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _calculate_corrective_confidence(self,
                                       df: pd.DataFrame,
                                       points: List[int],
                                       column: str) -> float:
        """Calculate confidence for corrective pattern"""
        prices = df[column].iloc[points].values
        
        # Base confidence
        confidence = 0.4
        
        # Check Fibonacci relationships
        wave_a = abs(prices[1] - prices[0])
        wave_b = abs(prices[2] - prices[1])
        
        # Wave B to wave A ratio (should be 0.618 or 0.5)
        if wave_a > 0:
            ratio = wave_b / wave_a
            if 0.5 <= ratio <= 0.7:
                confidence += 0.2
            elif 0.3 <= ratio <= 0.5:
                confidence += 0.15
        
        return min(confidence, 0.9)

class MultiTimeframeSequenceBuilder:
    """Build sequences across multiple timeframes simultaneously"""
    
    def __init__(self, base_builder: EnhancedSequenceBuilder):
        self.base_builder = base_builder
        self.timeframe_patterns = {}
        
    def build_multi_timeframe_sequences(self,
                                      df_dict: Dict[str, pd.DataFrame],
                                      column: str = 'close',
                                      reference_date: pd.Timestamp = None) -> Dict[str, Any]:
        """
        Build sequences across multiple timeframes
        
        Args:
            df_dict: Dictionary of DataFrames by timeframe
            column: Price column to analyze
            reference_date: Date to align patterns around
        
        Returns:
            Dictionary of multi-timeframe analysis
        """
        if reference_date is None:
            reference_date = max(df.index.max() for df in df_dict.values())
        
        # Build patterns for each timeframe
        timeframe_sequences = {}
        
        for timeframe, df in df_dict.items():
            sequences = self._build_timeframe_sequences(df, timeframe, column)
            timeframe_sequences[timeframe] = sequences
        
        # Analyze cross-timeframe relationships
        relationships = self._analyze_cross_timeframe_relationships(
            timeframe_sequences, reference_date
        )
        
        # Build composite view
        composite = self._build_composite_view(timeframe_sequences, relationships)
        
        return {
            'timeframe_sequences': timeframe_sequences,
            'relationships': relationships,
            'composite': composite,
            'reference_date': reference_date
        }
    
    def _build_timeframe_sequences(self,
                                 df: pd.DataFrame,
                                 timeframe: str,
                                 column: str) -> List[MultiPatternSequence]:
        """Build sequences for a single timeframe"""
        # Detect peaks and troughs
        from .peaks import detect_peaks_troughs_enhanced
        peaks, troughs = detect_peaks_troughs_enhanced(df, column)
        
        # Create labeled points
        labeled_points = (
            [(int(p), 'peak') for p in peaks] +
            [(int(t), 'trough') for t in troughs]
        )
        labeled_points.sort(key=lambda x: x[0])
        
        sequences = []
        
        # Try multiple starting points
        for start_idx in range(0, len(df) - 50, 50):
            for end_offset in [100, 200, 500]:
                end_idx = min(start_idx + end_offset, len(df) - 1)
                
                sequence = self.base_builder.build_multi_pattern_sequence(
                    df, labeled_points, start_idx, end_idx, column
                )
                
                if sequence and sequence.composite_confidence > 0.3:
                    sequences.append(sequence)
        
        # Remove duplicates and sort by confidence
        sequences = self._deduplicate_sequences(sequences)
        sequences.sort(key=lambda s: s.composite_confidence, reverse=True)
        
        return sequences[:5]  # Top 5 sequences per timeframe
    
    def _analyze_cross_timeframe_relationships(self,
                                             timeframe_sequences: Dict[str, List[MultiPatternSequence]],
                                             reference_date: pd.Timestamp) -> Dict[str, Any]:
        """Analyze relationships between patterns across timeframes"""
        relationships = {
            'alignments': [],
            'divergences': [],
            'nested': [],
            'confluence_zones': []
        }
        
        timeframes = list(timeframe_sequences.keys())
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i+1:]:
                sequences1 = timeframe_sequences[tf1]
                sequences2 = timeframe_sequences[tf2]
                
                for seq1 in sequences1:
                    for seq2 in sequences2:
                        relationship = self._analyze_sequence_pair(
                            seq1, seq2, tf1, tf2, reference_date
                        )
                        
                        if relationship['type'] == 'aligned':
                            relationships['alignments'].append(relationship)
                        elif relationship['type'] == 'divergent':
                            relationships['divergences'].append(relationship)
                        elif relationship['type'] == 'nested':
                            relationships['nested'].append(relationship)
        
        # Identify confluence zones
        relationships['confluence_zones'] = self._identify_confluence_zones(
            relationships['alignments']
        )
        
        return relationships
    
    def _analyze_sequence_pair(self,
                             seq1: MultiPatternSequence,
                             seq2: MultiPatternSequence,
                             tf1: str,
                             tf2: str,
                             reference_date: pd.Timestamp) -> Dict[str, Any]:
        """Analyze relationship between two sequences from different timeframes"""
        # Compare primary patterns
        pattern1 = seq1.primary_pattern
        pattern2 = seq2.primary_pattern
        
        # Check time overlap
        overlap = self._calculate_time_overlap(pattern1, pattern2)
        
        # Check directional alignment
        direction_match = self._check_directional_alignment(pattern1, pattern2)
        
        # Determine relationship type
        if overlap > 0.7 and direction_match:
            rel_type = 'aligned'
        elif overlap > 0.5 and not direction_match:
            rel_type = 'divergent'
        elif self._is_nested(pattern1, pattern2, tf1, tf2):
            rel_type = 'nested'
        else:
            rel_type = 'independent'
        
        return {
            'type': rel_type,
            'timeframe1': tf1,
            'timeframe2': tf2,
            'confidence1': seq1.composite_confidence,
            'confidence2': seq2.composite_confidence,
            'overlap': overlap,
            'direction_match': direction_match,
            'combined_strength': self._calculate_combined_strength(seq1, seq2, rel_type)
        }
    
    def _build_composite_view(self,
                            timeframe_sequences: Dict[str, List[MultiPatternSequence]],
                            relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Build composite view combining all timeframes"""
        composite = {
            'primary_trend': None,
            'confirmation_level': 0.0,
            'key_patterns': [],
            'trading_bias': None,
            'risk_levels': []
        }
        
        # Identify primary trend from alignments
        if relationships['alignments']:
            strongest_alignment = max(
                relationships['alignments'],
                key=lambda a: a['combined_strength']
            )
            
            # Get the higher timeframe pattern as primary
            tf1, tf2 = strongest_alignment['timeframe1'], strongest_alignment['timeframe2']
            if self._get_timeframe_priority(tf1) > self._get_timeframe_priority(tf2):
                primary_tf = tf1
            else:
                primary_tf = tf2
            
            if timeframe_sequences[primary_tf]:
                composite['primary_trend'] = {
                    'timeframe': primary_tf,
                    'pattern': timeframe_sequences[primary_tf][0],
                    'support_count': len([
                        a for a in relationships['alignments']
                        if primary_tf in [a['timeframe1'], a['timeframe2']]
                    ])
                }
        
        # Calculate confirmation level
        alignment_score = len(relationships['alignments']) / max(
            len(relationships['alignments']) + len(relationships['divergences']), 1
        )
        composite['confirmation_level'] = alignment_score
        
        # Determine trading bias
        if composite['confirmation_level'] > 0.7:
            composite['trading_bias'] = 'strong'
        elif composite['confirmation_level'] > 0.5:
            composite['trading_bias'] = 'moderate'
        else:
            composite['trading_bias'] = 'weak'
        
        # Identify key patterns
        all_patterns = []
        for tf, sequences in timeframe_sequences.items():
            for seq in sequences[:2]:  # Top 2 from each timeframe
                all_patterns.append({
                    'timeframe': tf,
                    'sequence': seq,
                    'relevance': seq.composite_confidence * self._get_timeframe_priority(tf)
                })
        
        all_patterns.sort(key=lambda p: p['relevance'], reverse=True)
        composite['key_patterns'] = all_patterns[:5]
        
        return composite
    
    def _deduplicate_sequences(self,
                             sequences: List[MultiPatternSequence]) -> List[MultiPatternSequence]:
        """Remove duplicate sequences"""
        unique_sequences = []
        
        for seq in sequences:
            is_duplicate = False
            
            for existing in unique_sequences:
                if self._sequences_are_similar(seq, existing):
                    # Keep the one with higher confidence
                    if seq.composite_confidence > existing.composite_confidence:
                        unique_sequences.remove(existing)
                        unique_sequences.append(seq)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sequences.append(seq)
        
        return unique_sequences
    
    def _sequences_are_similar(self,
                             seq1: MultiPatternSequence,
                             seq2: MultiPatternSequence,
                             threshold: float = 0.8) -> bool:
        """Check if two sequences are similar"""
        points1 = set(seq1.primary_pattern.points)
        points2 = set(seq2.primary_pattern.points)
        
        if not points1 or not points2:
            return False
        
        overlap = len(points1 & points2)
        similarity = overlap / min(len(points1), len(points2))
        
        return similarity > threshold
    
    def _calculate_time_overlap(self,
                              pattern1: WaveSequence,
                              pattern2: WaveSequence) -> float:
        """Calculate time overlap between patterns"""
        start1, end1 = pattern1.points[0], pattern1.points[-1]
        start2, end2 = pattern2.points[0], pattern2.points[-1]
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        total_length = max(end1, end2) - min(start1, start2)
        
        return overlap_length / total_length if total_length > 0 else 0
    
    def _check_directional_alignment(self,
                                   pattern1: WaveSequence,
                                   pattern2: WaveSequence) -> bool:
        """Check if patterns have same directional bias"""
        # Simple check based on start and end points
        direction1 = pattern1.points[-1] > pattern1.points[0]
        direction2 = pattern2.points[-1] > pattern2.points[0]
        
        return direction1 == direction2
    
    def _is_nested(self,
                  pattern1: WaveSequence,
                  pattern2: WaveSequence,
                  tf1: str,
                  tf2: str) -> bool:
        """Check if one pattern is nested within another"""
        # Higher timeframe should contain lower
        if self._get_timeframe_priority(tf1) > self._get_timeframe_priority(tf2):
            larger, smaller = pattern1, pattern2
        else:
            larger, smaller = pattern2, pattern1
        
        return (larger.points[0] <= smaller.points[0] and
                larger.points[-1] >= smaller.points[-1])
    
    def _calculate_combined_strength(self,
                                   seq1: MultiPatternSequence,
                                   seq2: MultiPatternSequence,
                                   rel_type: str) -> float:
        """Calculate combined strength of two sequences"""
        base_strength = (seq1.composite_confidence + seq2.composite_confidence) / 2
        
        # Adjust based on relationship type
        if rel_type == 'aligned':
            return base_strength * 1.2
        elif rel_type == 'nested':
            return base_strength * 1.1
        elif rel_type == 'divergent':
            return base_strength * 0.7
        else:
            return base_strength * 0.9
    
    def _identify_confluence_zones(self,
                                 alignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify zones where multiple patterns align"""
        # Group alignments by strength
        strong_alignments = [a for a in alignments if a['combined_strength'] > 0.6]
        
        confluence_zones = []
        
        # Simple clustering of strong alignments
        for alignment in strong_alignments:
            zone = {
                'alignments': [alignment],
                'strength': alignment['combined_strength'],
                'timeframes': {alignment['timeframe1'], alignment['timeframe2']}
            }
            
            # Check if this can be merged with existing zones
            merged = False
            for existing_zone in confluence_zones:
                if zone['timeframes'] & existing_zone['timeframes']:
                    existing_zone['alignments'].extend(zone['alignments'])
                    existing_zone['timeframes'] |= zone['timeframes']
                    existing_zone['strength'] = max(existing_zone['strength'], zone['strength'])
                    merged = True
                    break
            
            if not merged:
                confluence_zones.append(zone)
        
        return sorted(confluence_zones, key=lambda z: z['strength'], reverse=True)
    
    def _get_timeframe_priority(self, timeframe: str) -> float:
        """Get priority/weight for a timeframe"""
        priorities = {
            'month': 1.0,
            'week': 0.7,
            'day': 0.5,
            'hour': 0.3,
            'minute': 0.1
        }
        return priorities.get(timeframe.lower(), 0.5) 