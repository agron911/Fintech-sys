import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from .fib_utils import validate_fibonacci_relationships, FibonacciRatios
from .trendlines import analyze_diagonal_trend_lines
from .volume import analyze_diagonal_volume_pattern, validate_volume_patterns
from .alternation import analyze_diagonal_wave_alternation, validate_alternation_principle

@dataclass
class ValidationConfig:
    """Centralized configuration for all wave validations"""
    strict_mode: bool = False
    fibonacci_tolerance: float = 0.15
    wave2_max_retracement: float = 0.9
    wave3_min_ratio: float = 1.0
    overlap_allowed: bool = True
    acceptance_threshold: float = 0.3
    reality_adjustment: float = 0.85
    
    # Confidence weights
    direction_weight: float = 0.3
    wave2_weight: float = 0.2
    wave3_weight: float = 0.25
    overlap_weight: float = 0.15
    fibonacci_weight: float = 0.25
    
    @classmethod
    def strict_config(cls):
        """Create strict mode configuration"""
        return cls(
            strict_mode=True,
            fibonacci_tolerance=0.05,
            wave2_max_retracement=0.786,
            wave3_min_ratio=1.1,
            overlap_allowed=False,
            acceptance_threshold=0.4,
            reality_adjustment=0.9
        )
    
    @classmethod
    def relaxed_config(cls):
        """Create relaxed mode configuration"""
        return cls(
            strict_mode=False,
            fibonacci_tolerance=0.2,
            wave2_max_retracement=0.95,
            wave3_min_ratio=0.9,
            overlap_allowed=True,
            acceptance_threshold=0.25,
            reality_adjustment=0.8
        )

class WaveEqualityChecker:
    """Consolidated wave equality and relationship analysis"""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
    
    def check_wave_equality(self, waves: Dict[int, float]) -> Dict[str, Any]:
        """
        CONSOLIDATED: Single function for all wave equality checks
        Replaces check_wave_equality() and check_wave_equality_simple() from complex_patterns.py
        """
        result = {
            'has_equality': False,
            'equal_pairs': [],
            'confidence': 0.0,
            'relationships': {}
        }
        
        # Standard equality checks with confidence weights
        equality_checks = [
            (1, 5, 0.8, "wave_1_5_equality"),  # Most common and important
            (1, 3, 0.6, "wave_1_3_equality"),  # Less common
            (3, 5, 0.5, "wave_3_5_equality")   # Rare but possible
        ]
        
        max_confidence = 0.0
        
        for wave1, wave2, base_confidence, relationship_name in equality_checks:
            if wave1 in waves and wave2 in waves and waves[wave1] != 0:
                ratio = abs(waves[wave2] / waves[wave1])
                deviation = abs(1 - ratio)
                
                if deviation <= self.tolerance:
                    result['equal_pairs'].append((wave1, wave2))
                    result['has_equality'] = True
                    
                    # Confidence decreases with deviation
                    confidence = base_confidence * (1 - deviation / self.tolerance)
                    max_confidence = max(max_confidence, confidence)
                    
                    result['relationships'][relationship_name] = {
                        'ratio': ratio,
                        'deviation': deviation,
                        'confidence': confidence
                    }
        
        result['confidence'] = max_confidence
        return result
    
    def check_wave_extensions(self, waves: Dict[int, float]) -> Dict[str, Any]:
        """
        CONSOLIDATED: Check for extended waves (consolidates similar logic from multiple files)
        """
        result = {
            'has_extension': False,
            'extended_wave': None,
            'extension_ratio': 0.0,
            'confidence': 0.0
        }
        
        # Only check impulse waves
        impulse_waves = {k: v for k, v in waves.items() if k in [1, 3, 5]}
        
        if len(impulse_waves) < 2:
            return result
        
        wave_lengths = {k: abs(v) for k, v in impulse_waves.items()}
        max_wave = max(wave_lengths.items(), key=lambda x: x[1])
        max_wave_num, max_length = max_wave
        
        other_lengths = [length for num, length in wave_lengths.items() 
                        if num != max_wave_num]
        
        if other_lengths:
            avg_other = np.mean(other_lengths)
            extension_ratio = max_length / avg_other if avg_other > 0 else 0
            
            # Extension threshold
            if extension_ratio >= 1.618:
                result.update({
                    'has_extension': True,
                    'extended_wave': max_wave_num,
                    'extension_ratio': extension_ratio,
                    'confidence': min(0.9, 0.5 + (extension_ratio - 1.618) * 0.2)
                })
        
        return result
    
    def check_corrective_alternation(self, waves: Dict[int, float], 
                                   wave_points: np.ndarray = None) -> Dict[str, Any]:
        """Check alternation between corrective waves 2 and 4"""
        result = {
            'has_alternation': False,
            'alternation_type': 'none',
            'confidence': 0.0
        }
        
        if 2 not in waves or 4 not in waves:
            return result
        
        wave_2_length = abs(waves[2])
        wave_4_length = abs(waves[4])
        
        if wave_2_length == 0 or wave_4_length == 0:
            return result
        
        # Length alternation
        length_ratio = wave_4_length / wave_2_length
        
        # Time alternation (if wave_points provided)
        time_ratio = 1.0
        if wave_points is not None and len(wave_points) >= 5:
            wave_2_time = wave_points[2] - wave_points[1]
            wave_4_time = wave_points[4] - wave_points[3]
            if wave_2_time > 0:
                time_ratio = wave_4_time / wave_2_time
        
        # Good alternation criteria
        has_length_alt = length_ratio > 1.618 or length_ratio < 0.618
        has_time_alt = time_ratio > 1.618 or time_ratio < 0.618
        
        if has_length_alt or has_time_alt:
            result.update({
                'has_alternation': True,
                'alternation_type': 'good',
                'confidence': 0.15,
                'length_ratio': length_ratio,
                'time_ratio': time_ratio
            })
        elif 0.8 <= length_ratio <= 1.2:  # Poor alternation
            result.update({
                'alternation_type': 'poor',
                'confidence': 0.0
            })
        else:  # Moderate alternation
            result.update({
                'alternation_type': 'moderate', 
                'confidence': 0.05
            })
        
        return result

def _validate_wave_directions_consolidated(waves: Dict[int, float], 
                                         config: ValidationConfig) -> Dict[str, Any]:
    """Consolidated wave direction validation"""
    trend_up = waves[1] > 0
    expected_directions = {
        1: trend_up, 2: not trend_up, 3: trend_up, 
        4: not trend_up, 5: trend_up
    }
    
    correct = sum(1 for wave_num, expected in expected_directions.items()
                  if wave_num in waves and (waves[wave_num] > 0) == expected)
    
    total_waves = min(len(waves), 5)
    confidence = (correct / total_waves) * 0.3  # Max 0.3
    
    return {
        'valid': correct >= (total_waves if config.strict_mode else total_waves - 1),
        'confidence': confidence,
        'correct_directions': correct,
        'total_waves': total_waves,
        'error': 'invalid_directions' if correct < total_waves - 1 else None
    }

def _validate_wave_2_retracement_consolidated(waves: Dict[int, float], 
                                            config: ValidationConfig) -> Dict[str, Any]:
    """Consolidated Wave 2 retracement validation"""
    if waves[1] == 0:
        return {'valid': True, 'confidence': 0.0, 'retracement': 0}
    
    retracement = abs(waves[2] / waves[1])
    
    # Hard failure conditions
    if retracement > 1.0:
        return {
            'valid': False,
            'confidence': 0.0,
            'retracement': retracement,
            'error': 'wave2_100_percent_plus'
        }
    
    if retracement > config.wave2_max_retracement:
        return {
            'valid': False,
            'confidence': 0.0,
            'retracement': retracement,
            'error': 'wave2_excessive_retracement'
        }
    
    # Calculate confidence based on ideal ranges
    confidence = 0.0
    if 0.382 <= retracement <= 0.618:  # Ideal Fibonacci range
        confidence = 0.2
    elif 0.236 <= retracement <= 0.786:  # Good Fibonacci range
        confidence = 0.15
    elif retracement <= 0.9:  # Acceptable range
        confidence = 0.1
    
    return {
        'valid': True,
        'confidence': confidence,
        'retracement': retracement,
        'in_ideal_range': 0.382 <= retracement <= 0.618
    }

def _validate_wave_3_length_consolidated(waves: Dict[int, float], 
                                       config: ValidationConfig) -> Dict[str, Any]:
    """Consolidated Wave 3 length validation"""
    wave_lengths = [abs(waves.get(i, 0)) for i in [1, 3, 5] if i in waves]
    
    if len(wave_lengths) < 2:
        return {'valid': True, 'confidence': 0.0}
    
    wave_3_length = abs(waves[3])
    min_length = min(wave_lengths)
    max_length = max(wave_lengths)
    
    # Check if Wave 3 is shortest (violation)
    is_shortest = wave_3_length <= min_length * config.wave3_min_ratio
    
    if is_shortest:
        return {
            'valid': False,
            'confidence': 0.0,
            'wave_3_length': wave_3_length,
            'wave_lengths': wave_lengths,
            'error': 'wave3_is_shortest'
        }
    
    # Calculate confidence
    confidence = 0.0
    if wave_3_length == max_length:  # Wave 3 is longest (ideal)
        confidence = 0.25
    elif wave_3_length >= max_length * 0.95:  # Close to longest
        confidence = 0.2
    elif wave_3_length >= max_length * 0.9:   # Reasonably long
        confidence = 0.15
    else:  # Not shortest but not impressive
        confidence = 0.1
    
    return {
        'valid': True,
        'confidence': confidence,
        'wave_3_length': wave_3_length,
        'wave_lengths': wave_lengths,
        'is_longest': wave_3_length == max_length
    }

def _validate_wave_4_overlap_consolidated(prices: np.ndarray, 
                                        config: ValidationConfig) -> Dict[str, Any]:
    """Consolidated Wave 4 overlap validation"""
    if len(prices) < 5:
        return {'valid': True, 'confidence': 0.0}
    
    wave_1_high = max(prices[0], prices[1])
    wave_1_low = min(prices[0], prices[1])
    wave_4_end = prices[4]
    
    has_overlap = wave_1_low <= wave_4_end <= wave_1_high
    
    if has_overlap and not config.overlap_allowed:
        return {
            'valid': False,
            'confidence': 0.0,
            'has_overlap': True,
            'error': 'wave4_overlaps_wave1'
        }
    
    # Calculate confidence
    confidence = 0.15 if not has_overlap else 0.05
    
    # Check for minor overlap (might be acceptable)
    minor_overlap = False
    if has_overlap:
        wave_1_range = wave_1_high - wave_1_low
        if wave_1_range > 0:
            overlap_amount = min(abs(wave_4_end - wave_1_low), abs(wave_4_end - wave_1_high))
            overlap_pct = overlap_amount / wave_1_range
            minor_overlap = overlap_pct < 0.1
            
            if minor_overlap:
                confidence = 0.1  # Minor penalty for small overlap
    
    return {
        'valid': True,
        'confidence': confidence,
        'has_overlap': has_overlap,
        'minor_overlap': minor_overlap
    }

def _validate_fibonacci_consolidated(waves: Dict[int, float], dates: pd.DatetimeIndex, 
                                   config: ValidationConfig) -> Dict[str, Any]:
    """Consolidated Fibonacci relationship validation"""
    confidence = validate_fibonacci_relationships(
        waves, dates, tolerance=config.fibonacci_tolerance
    )
    
    return {
        'valid': True,  # Fibonacci always valid, just affects confidence
        'confidence': confidence,
        'tolerance_used': config.fibonacci_tolerance
    }

def validate_impulse_wave_rules(df: pd.DataFrame, wave_points: np.ndarray, 
                              column: str = 'close', strict_mode: bool = False,
                              validation_config: ValidationConfig = None) -> Tuple[bool, float, Dict[str, Any]]:
    """
    REFACTORED: Main impulse wave validation with consolidated logic
    This replaces multiple validation functions across different files
    """
    # Use provided config or create based on mode
    if validation_config is None:
        config = ValidationConfig.strict_config() if strict_mode else ValidationConfig()
    else:
        config = validation_config
    
    # Basic validation
    if len(wave_points) < 5:
        return False, 0.0, {"error": "insufficient_points", "config": config.__dict__}
    
    prices = df[column].iloc[wave_points].values
    dates = df.index[wave_points]
    
    # Calculate waves
    waves = {i+1: prices[i+1] - prices[i] for i in range(min(len(prices)-1, 5))}
    
    # Initialize result
    validation_results = {
        'config_used': config.__dict__,
        'waves': waves,
        'wave_analysis': {}
    }
    
    confidence = 0.0
    critical_failures = []
    
    # 1. CONSOLIDATED Wave Direction Validation
    direction_result = _validate_wave_directions_consolidated(waves, config)
    validation_results['wave_analysis']['directions'] = direction_result
    
    if not direction_result['valid'] and config.strict_mode:
        return False, 0.0, {**validation_results, "error": "direction_violation"}
    confidence += direction_result['confidence'] * config.direction_weight
    
    # 2. CONSOLIDATED Wave 2 Retracement
    wave2_result = _validate_wave_2_retracement_consolidated(waves, config)
    validation_results['wave_analysis']['wave2'] = wave2_result
    
    if not wave2_result['valid']:
        if config.strict_mode:
            return False, 0.0, {**validation_results, "error": wave2_result['error']}
        else:
            critical_failures.append('wave2_excessive')
    confidence += wave2_result['confidence'] * config.wave2_weight
    
    # 3. CONSOLIDATED Wave 3 Length Validation
    wave3_result = _validate_wave_3_length_consolidated(waves, config)
    validation_results['wave_analysis']['wave3'] = wave3_result
    
    if not wave3_result['valid']:
        if config.strict_mode:
            return False, 0.0, {**validation_results, "error": wave3_result['error']}
        else:
            critical_failures.append('wave3_shortest')
    confidence += wave3_result['confidence'] * config.wave3_weight
    
    # 4. CONSOLIDATED Wave 4 Overlap
    overlap_result = _validate_wave_4_overlap_consolidated(prices, config)
    validation_results['wave_analysis']['overlap'] = overlap_result
    
    if not overlap_result['valid'] and config.strict_mode:
        return False, 0.0, {**validation_results, "error": "wave4_overlap"}
    confidence += overlap_result['confidence'] * config.overlap_weight
    
    # 5. CONSOLIDATED Fibonacci Relationships
    fib_result = _validate_fibonacci_consolidated(waves, dates, config)
    validation_results['wave_analysis']['fibonacci'] = fib_result
    confidence += fib_result['confidence'] * config.fibonacci_weight
    
    # 6. CONSOLIDATED Wave Equality and Extensions Analysis
    equality_checker = WaveEqualityChecker(tolerance=0.1)
    equality_result = equality_checker.check_wave_equality(waves)
    extension_result = equality_checker.check_wave_extensions(waves)
    alternation_result = equality_checker.check_corrective_alternation(waves, wave_points)
    
    validation_results['wave_analysis'].update({
        'equality': equality_result,
        'extensions': extension_result,
        'alternation': alternation_result
    })
    
    # Add bonuses for good relationships
    confidence += equality_result['confidence'] * 0.1
    confidence += extension_result['confidence'] * 0.1
    confidence += alternation_result['confidence'] * 0.1
    
    # Apply reality adjustment and penalties
    if critical_failures:
        confidence *= (1 - len(critical_failures) * 0.15)  # Penalty for failures
    
    final_confidence = min(confidence * config.reality_adjustment, 1.0)
    
    # Final validation decision
    is_valid = final_confidence >= config.acceptance_threshold
    
    validation_results.update({
        'final_confidence': final_confidence,
        'is_valid': is_valid,
        'critical_failures': critical_failures,
        'base_confidence': confidence
    })
    
    return is_valid, final_confidence, validation_results

# LEGACY FUNCTION WRAPPERS (for backward compatibility)

def validate_wave_4_overlap(df: pd.DataFrame, wave_points: np.ndarray, 
                          column: str = 'close', allow_overlap: bool = True) -> Dict[str, Any]:
    """
    LEGACY WRAPPER: Maintains backward compatibility
    Use _validate_wave_4_overlap_consolidated for new code
    """
    prices = df[column].iloc[wave_points].values
    config = ValidationConfig(overlap_allowed=allow_overlap)
    result = _validate_wave_4_overlap_consolidated(prices, config)
    
    # Convert to legacy format
    return {
        "valid": result['valid'],
        "has_overlap": result.get('has_overlap', False),
        "minor_overlap": result.get('minor_overlap', False),
        "reason": result.get('error', 'no_overlap' if not result.get('has_overlap') else 'overlap_allowed')
    }

def validate_wave_directions(waves: Dict[int, float], strict: bool = True, 
                           timeframe: str = 'day') -> Tuple[bool, float]:
    """
    LEGACY WRAPPER: Maintains backward compatibility
    Use _validate_wave_directions_consolidated for new code
    """
    config = ValidationConfig(strict_mode=strict)
    result = _validate_wave_directions_consolidated(waves, config)
    return result['valid'], result['confidence']

def validate_diagonal_triangle(df: pd.DataFrame, wave_points: np.ndarray, 
                             column: str = 'close') -> Dict[str, Any]:
    """Specialized diagonal triangle validation - keep as is"""
    if len(wave_points) < 5:
        return {"is_valid_diagonal": False, "reason": "insufficient_points"}
    
    prices = df[column].iloc[wave_points].values
    indices = wave_points
    
    validation_result = {
        "is_valid_diagonal": False,
        "trend_lines_converge": False,
        "volume_diminishes": False,
        "wave_alternation": False,
        "confidence": 0.0
    }
    
    try:
        # Trend line analysis
        trend_analysis = analyze_diagonal_trend_lines(prices, indices)
        validation_result.update(trend_analysis)
        
        if not trend_analysis['valid_trend_lines']:
            validation_result["reason"] = "invalid_trend_lines"
            return validation_result
        
        # Volume analysis
        if 'volume' in df.columns:
            volume_analysis = analyze_diagonal_volume_pattern(df, wave_points)
            validation_result.update(volume_analysis)
        else:
            validation_result["volume_diminishes"] = True
        
        # Alternation analysis
        alternation_analysis = analyze_diagonal_wave_alternation(df, wave_points, column)
        validation_result.update(alternation_analysis)
        
        # Calculate final score
        validation_score = 0
        if trend_analysis['valid_trend_lines']: 
            validation_score += 0.4
        if validation_result['volume_diminishes']: 
            validation_score += 0.3
        if validation_result['wave_alternation']: 
            validation_score += 0.3
        
        validation_result["confidence"] = validation_score
        validation_result["is_valid_diagonal"] = validation_score >= 0.6
        
        if validation_result["is_valid_diagonal"]:
            validation_result["reason"] = "valid_diagonal_pattern"
        else:
            validation_result["reason"] = f"insufficient_diagonal_characteristics_{validation_score:.2f}"
            
    except Exception as e:
        validation_result["reason"] = f"analysis_error_{str(e)}"
    
    return validation_result

def validate_elliott_wave_pattern(df: pd.DataFrame, wave_points: np.ndarray,
                                 column: str = 'close', strict_mode: bool = False,
                                 pattern_type: str = 'impulse') -> Dict[str, Any]:
    """
    MAIN INTERFACE: Single function for all Elliott Wave validation
    Replaces multiple scattered validation functions
    """
    if pattern_type.lower() == 'impulse':
        is_valid, confidence, details = validate_impulse_wave_rules(
            df, wave_points, column, strict_mode
        )
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'pattern_type': 'impulse',
            'validation_details': details,
            'wave_points': wave_points,
            'prices': df[column].iloc[wave_points].values.tolist()
        }
    
    elif pattern_type.lower() == 'diagonal':
        diagonal_result = validate_diagonal_triangle(df, wave_points, column)
        
        return {
            'is_valid': diagonal_result['is_valid_diagonal'],
            'confidence': diagonal_result['confidence'],
            'pattern_type': 'diagonal',
            'validation_details': diagonal_result,
            'wave_points': wave_points,
            'prices': df[column].iloc[wave_points].values.tolist()
        }
    
    else:
        return {
            'is_valid': False,
            'confidence': 0.0,
            'error': f'Unknown pattern type: {pattern_type}',
            'pattern_type': pattern_type
        } 