import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# Imports from other core modules (will refine as needed)
from ..core.impulse import find_elliott_wave_pattern_enhanced
from ..core.corrective import detect_corrective_patterns
from ..core.models import WaveType

def detect_current_wave_position_enhanced(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    """
    Enhanced current wave position detection with comprehensive analysis.
    """
    try:
        # Analyze multiple timeframes for better accuracy
        timeframes = {
            'short': df.tail(250) if len(df) > 250 else df,  # ~1 year
            'medium': df.tail(500) if len(df) > 500 else df,  # ~2 years
            'long': df.tail(1000) if len(df) > 1000 else df   # ~4 years
        }

        analyses = {}

        for tf_name, tf_data in timeframes.items():
            if len(tf_data) >= 50:  # Minimum data requirement
                wave_data = find_elliott_wave_pattern_enhanced(tf_data, column)
                analyses[tf_name] = analyze_wave_position(tf_data, wave_data, column)

        # Combine analyses for final assessment
        final_assessment = combine_timeframe_analyses(analyses)

        return final_assessment

    except Exception as e:
        return {
            'position': 'Analysis Error',
            'confidence': 0.0,
            'forecast': f'Error in wave position analysis: {str(e)}',
            'wave_type': 'error',
            'details': {'error': str(e)}
        }


def analyze_wave_position(df: pd.DataFrame, wave_data: Dict[str, Any], 
                         column: str = 'close') -> Dict[str, Any]:
    """
    Analyze the current position within an Elliott Wave structure.
    """
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    corrective_wave = wave_data.get('corrective_wave', np.array([]))
    confidence = wave_data.get('confidence', 0.0)

    current_price = df[column].iloc[-1]
    current_date = df.index[-1]

    # Determine position based on wave structure
    if len(impulse_wave) >= 5:
        # We have a complete or near-complete impulse wave
        last_impulse_idx = impulse_wave[-1]
        last_impulse_date = df.index[last_impulse_idx]

        if len(corrective_wave) > 0:
            # We have corrective wave data
            return analyze_corrective_position_enhanced(
                df, impulse_wave, corrective_wave, current_date, current_price, confidence
            )
        else:
            # Likely at the end of impulse, beginning of correction
            return analyze_post_impulse_position(
                df, impulse_wave, current_date, current_price, confidence
            )

    elif len(impulse_wave) >= 3:
        # Partial impulse wave
        return analyze_partial_impulse_position(
            df, impulse_wave, current_date, current_price, confidence
        )

    else:
        # No clear wave structure
        return {
            'position': 'Unclear Structure',
            'confidence': 0.2,
            'forecast': 'No clear Elliott Wave pattern identified. Market may be in transition or consolidation.',
            'wave_type': 'transitional',
            'details': {'reason': 'insufficient_wave_structure'}
        }


def analyze_corrective_position_enhanced(df: pd.DataFrame, impulse_wave: np.ndarray, 
                                       corrective_wave: np.ndarray, current_date: pd.Timestamp,
                                       current_price: float, base_confidence: float) -> Dict[str, Any]:
    """
    Analyze position within corrective wave structure.
    """
    try:
        corrective_prices = df['close'].iloc[corrective_wave].values
        corrective_dates = df.index[corrective_wave]

        # Determine corrective wave position
        if len(corrective_wave) >= 3:
            # Complete A-B-C structure
            wave_a = corrective_prices[1] - corrective_prices[0]
            wave_b = corrective_prices[2] - corrective_prices[1] if len(corrective_prices) > 2 else 0

            # Check if we\'re past the corrective structure
            last_corrective_date = corrective_dates[-1]
            days_since_correction = (current_date - last_corrective_date).days

            if days_since_correction > 30:  # More than a month since last corrective point
                return {
                    'position': 'Post-Corrective (New Impulse Expected)',
                    'confidence': base_confidence * 0.8,
                    'forecast': 'Corrective pattern appears complete. New impulse wave likely beginning.',
                    'wave_type': 'transitional_to_impulse',
                    'details': {
                        'days_since_correction': days_since_correction,
                        'corrective_complete': True
                    }
                }
            else:
                return {
                    'position': 'Corrective Wave C (Final Phase)',
                    'confidence': base_confidence * 0.9,
                    'forecast': 'Final wave of corrective pattern. Expect completion soon and trend reversal.',
                    'wave_type': 'corrective_c',
                    'details': {
                        'corrective_phase': 'C',
                        'near_completion': True
                    }
                }

        elif len(corrective_wave) == 2:
            return {
                'position': 'Corrective Wave B',
                'confidence': base_confidence * 0.85,
                'forecast': 'In Wave B of corrective pattern. Expect reversal toward Wave C direction.',
                'wave_type': 'corrective_b',
                'details': {'corrective_phase': 'B'}
            }

        else:
            return {
                'position': 'Corrective Wave A',
                'confidence': base_confidence * 0.8,
                'forecast': 'Initial corrective wave in progress. Expect continuation of corrective pattern.',
                'wave_type': 'corrective_a',
                'details': {'corrective_phase': 'A'}
            }

    except Exception as e:
        return {
            'position': 'Corrective Analysis Error',
            'confidence': 0.3,
            'forecast': 'Error analyzing corrective position',
            'wave_type': 'error',
            'details': {'error': str(e)}
        }


def analyze_post_impulse_position(df: pd.DataFrame, impulse_wave: np.ndarray,
                                current_date: pd.Timestamp, current_price: float,
                                base_confidence: float) -> Dict[str, Any]:
    """
    Analyze position immediately after impulse wave completion.
    """
    try:
        last_impulse_idx = impulse_wave[-1]
        last_impulse_price = df['close'].iloc[last_impulse_idx]
        last_impulse_date = df.index[last_impulse_idx]

        days_since_impulse = (current_date - last_impulse_date).days
        price_change_since = (current_price - last_impulse_price) / last_impulse_price

        # Determine if correction has begun
        if abs(price_change_since) > 0.05 and days_since_impulse > 5:  # >5% change and >5 days
            correction_direction = "downward" if price_change_since < 0 else "upward"

            return {
                'position': f'Early Corrective Phase (Wave A)',
                'confidence': base_confidence * 0.85,
                'forecast': f'Corrective pattern beginning with {correction_direction} Wave A. Expect 3-wave A-B-C correction.',
                'wave_type': 'early_corrective',
                'details': {
                    'days_since_impulse': days_since_impulse,
                    'price_change_pct': price_change_since * 100,
                    'correction_direction': correction_direction
                }
            }
        else:
            return {
                'position': 'Post-Impulse Consolidation',
                'confidence': base_confidence * 0.7,
                'forecast': 'Recently completed impulse wave. Consolidation or early corrective phase.',
                'wave_type': 'post_impulse',
                'details': {
                    'days_since_impulse': days_since_impulse,
                    'price_change_pct': price_change_since * 100
                }
            }

    except Exception as e:
        return {
            'position': 'Post-Impulse Analysis Error',
            'confidence': 0.3,
            'forecast': 'Error analyzing post-impulse position',
            'wave_type': 'error',
            'details': {'error': str(e)}
        }


def analyze_partial_impulse_position(df: pd.DataFrame, impulse_wave: np.ndarray,
                                   current_date: pd.Timestamp, current_price: float,
                                   base_confidence: float) -> Dict[str, Any]:
    """
    Analyze position within a partial impulse wave structure.
    """
    try:
        wave_count = len(impulse_wave)
        impulse_prices = df['close'].iloc[impulse_wave].values

        if wave_count == 4:
            # Likely in Wave 5
            wave_4_price = impulse_prices[-1]
            price_change = (current_price - wave_4_price) / wave_4_price

            return {
                'position': 'Impulse Wave 5 (Final Wave)',
                'confidence': base_confidence * 0.9,
                'forecast': 'In final impulse wave. Expect completion and subsequent corrective pattern.',
                'wave_type': 'impulse_5',
                'details': {
                    'wave_count': wave_count,
                    'price_change_from_wave_4': price_change * 100
                }
            }

        elif wave_count == 3:
            # Likely in Wave 4 correction
            return {
                'position': 'Impulse Wave 4 (Correction)',
                'confidence': base_confidence * 0.85,
                'forecast': 'In Wave 4 corrective phase within impulse. Expect final Wave 5 push.',
                'wave_type': 'impulse_4',
                'details': {'wave_count': wave_count}
            }

        else:
            position_map = {2: 'Wave 2', 1: 'Wave 1'}
            wave_name = position_map.get(wave_count, f'Wave {wave_count}')

            return {
                'position': f'Impulse {wave_name}',
                'confidence': base_confidence * 0.8,
                'forecast': f'In {wave_name} of impulse sequence. Pattern still developing.',
                'wave_type': f'impulse_{wave_count}',
                'details': {'wave_count': wave_count}
            }

    except Exception as e:
        return {
            'position': 'Partial Impulse Analysis Error',
            'confidence': 0.3,
            'forecast': 'Error analyzing partial impulse position',
            'wave_type': 'error',
            'details': {'error': str(e)}
        }

def combine_timeframe_analyses(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple timeframe analyses for a comprehensive assessment.
    """
    if not analyses:
        return {
            'position': 'No Analysis Available',
            'confidence': 0.0,
            'forecast': 'Insufficient data for analysis',
            'wave_type': 'insufficient_data'
        }

    # Weight different timeframes
    weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}

    # Calculate weighted confidence
    weighted_confidence = 0.0
    total_weight = 0.0

    positions = []
    forecasts = []

    for tf_name, analysis in analyses.items():
        weight = weights.get(tf_name, 0.1)
        confidence = analysis.get('confidence', 0.0)

        weighted_confidence += confidence * weight
        total_weight += weight

        positions.append(f"{tf_name}: {analysis.get('position', 'Unknown')}")
        forecasts.append(analysis.get('forecast', ''))

    final_confidence = weighted_confidence / max(total_weight, 0.1)

    # Use the highest confidence analysis as primary
    primary_analysis = max(analyses.values(), key=lambda x: x.get('confidence', 0.0))

    # Combine forecasts intelligently
    primary_forecast = primary_analysis.get('forecast', '')

    return {
        'position': primary_analysis.get('position', 'Unknown'),
        'confidence': final_confidence,
        'forecast': primary_forecast,
        'wave_type': primary_analysis.get('wave_type', 'unknown'),
        'timeframe_details': {
            'primary_timeframe': max(analyses.keys(), key=lambda k: analyses[k].get('confidence', 0.0)),
            'all_positions': positions,
            'analysis_count': len(analyses)
        }
    } 