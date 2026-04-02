import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta

# Imports from other core modules (will refine as needed)
from src.analysis.core.impulse import find_elliott_wave_pattern_enhanced
from src.analysis.core.corrective_patterns import detect_corrective_patterns
from src.analysis.core.models import WaveType

def detect_current_wave_position_enhanced(df: pd.DataFrame, 
                                        wave_data: Dict[str, Any], 
                                        column: str = 'close') -> Dict[str, Any]:
    """
    Enhanced current position detection with multiple methods
    """
    current_price = df[column].iloc[-1]
    current_date = df.index[-1]
    
    position_info = {
        'current_wave': 'Unknown',
        'wave_progress': 0.0,
        'next_target': None,
        'key_levels': {},
        'confidence': 0.0,
        'analysis_method': 'none'
    }
    
    impulse_wave = wave_data.get('impulse_wave', np.array([]))
    
    if len(impulse_wave) == 0:
        return detect_position_without_pattern(df, column)
    
    # Method 1: Position relative to detected waves
    if len(impulse_wave) >= 2:
        position_info = analyze_position_in_waves(
            df, impulse_wave, current_price, current_date, column
        )
    
    # Method 2: Fibonacci projection
    fib_position = analyze_fibonacci_position(df, impulse_wave, current_price, column)
    if fib_position['confidence'] > position_info['confidence']:
        position_info.update(fib_position)
    
    # Method 3: Time-based analysis
    time_position = analyze_time_based_position(df, impulse_wave, current_date)
    position_info['time_analysis'] = time_position
    
    return position_info

def analyze_position_in_waves(df: pd.DataFrame, 
                            wave_points: np.ndarray,
                            current_price: float,
                            current_date: pd.Timestamp,
                            column: str) -> Dict[str, Any]:
    """Analyze current position within detected wave structure"""
    
    wave_prices = df[column].iloc[wave_points].values
    wave_dates = df.index[wave_points]
    
    # Determine which wave we're likely in
    last_wave_idx = wave_points[-1]
    last_wave_date = wave_dates[-1]
    days_since_last = (current_date - last_wave_date).days
    
    position_info = {
        'analysis_method': 'wave_structure'
    }
    
    if len(wave_points) >= 5:  # Complete 5-wave structure
        # Check if we're in a corrective phase
        wave_5_price = wave_prices[-1]
        
        # Calculate expected correction levels
        total_move = wave_5_price - wave_prices[0]
        fib_levels = {
            '0.236': wave_5_price - total_move * 0.236,
            '0.382': wave_5_price - total_move * 0.382,
            '0.500': wave_5_price - total_move * 0.5,
            '0.618': wave_5_price - total_move * 0.618
        }
        
        position_info['key_levels'] = fib_levels
        
        # Determine if we're still in wave 5 or starting correction
        if days_since_last < 20:  # Recent completion
            position_info['current_wave'] = 'Wave 5 completion / Early correction'
            position_info['confidence'] = 0.7
        else:
            # Check correction depth
            correction_depth = (wave_5_price - current_price) / total_move
            if 0 < correction_depth < 0.236:
                position_info['current_wave'] = 'Wave A of correction'
            elif 0.236 <= correction_depth < 0.5:
                position_info['current_wave'] = 'Wave B or C of correction'
            else:
                position_info['current_wave'] = 'Deep correction or new trend'
            position_info['confidence'] = 0.6
            
    elif len(wave_points) >= 3:  # Partial structure
        # We're likely in wave 3 or 4
        wave_3_price = wave_prices[2] if len(wave_prices) > 2 else wave_prices[-1]
        
        if current_price > wave_3_price:
            position_info['current_wave'] = 'Possible Wave 5 in progress'
            position_info['next_target'] = wave_3_price * 1.618  # Common extension
        else:
            position_info['current_wave'] = 'Wave 4 correction'
            position_info['next_target'] = wave_3_price * 0.618  # Support level
        
        position_info['confidence'] = 0.5
    
    return position_info

def analyze_fibonacci_position(df: pd.DataFrame,
                             wave_points: np.ndarray,
                             current_price: float,
                             column: str) -> Dict[str, Any]:
    """Analyze position using Fibonacci relationships"""
    if len(wave_points) < 2:
        return {'confidence': 0.0}
    
    wave_prices = df[column].iloc[wave_points].values
    
    # Calculate key Fibonacci levels
    wave_start = wave_prices[0]
    wave_high = max(wave_prices)
    wave_low = min(wave_prices)
    
    trend_up = wave_prices[-1] > wave_prices[0]
    
    if trend_up:
        range_size = wave_high - wave_low
        fib_0 = wave_low
        fib_100 = wave_high
    else:
        range_size = wave_high - wave_low
        fib_0 = wave_high
        fib_100 = wave_low
    
    # Calculate where current price sits in Fibonacci terms
    if range_size > 0:
        current_fib_level = abs(current_price - fib_0) / range_size
    else:
        current_fib_level = 0.5
    
    # Determine position based on Fibonacci level
    position_info = {
        'analysis_method': 'fibonacci',
        'confidence': 0.5
    }
    
    if 0.9 < current_fib_level < 1.1:
        position_info['current_wave'] = 'Near wave completion (100% level)'
    elif 0.55 < current_fib_level < 0.65:
        position_info['current_wave'] = 'Near 61.8% retracement - key decision point'
    elif 0.35 < current_fib_level < 0.4:
        position_info['current_wave'] = 'Near 38.2% retracement - shallow correction'
    elif current_fib_level > 1.618:
        position_info['current_wave'] = 'Extended beyond 161.8% - possible wave 3 or 5 extension'
    
    return position_info

def analyze_time_based_position(df: pd.DataFrame,
                              wave_points: np.ndarray,
                              current_date: pd.Timestamp) -> Dict[str, str]:
    """Analyze position based on time relationships"""
    if len(wave_points) < 2:
        return {'time_position': 'unknown'}
    
    wave_dates = df.index[wave_points]
    
    # Calculate wave durations
    wave_durations = []
    for i in range(len(wave_dates) - 1):
        duration = (wave_dates[i+1] - wave_dates[i]).days
        wave_durations.append(duration)
    
    # Average wave duration
    avg_duration = np.mean(wave_durations) if wave_durations else 30
    
    # Time since last wave point
    days_since_last = (current_date - wave_dates[-1]).days
    
    # Estimate position based on time
    if days_since_last < avg_duration * 0.3:
        time_position = "Early in current wave"
    elif days_since_last < avg_duration * 0.7:
        time_position = "Middle of current wave"
    elif days_since_last < avg_duration * 1.2:
        time_position = "Late in current wave - watch for reversal"
    else:
        time_position = "Overextended - new pattern may be forming"
    
    return {
        'time_position': time_position,
        'days_since_last_wave': days_since_last,
        'average_wave_duration': avg_duration
    }

def detect_position_without_pattern(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Detect position using alternative methods when no clear pattern exists"""
    current_price = df[column].iloc[-1]
    
    # Use simple trend and support/resistance analysis
    sma_20 = df[column].rolling(20).mean().iloc[-1]
    sma_50 = df[column].rolling(50).mean().iloc[-1]
    sma_200 = df[column].rolling(200).mean().iloc[-1]
    
    # Determine trend
    if current_price > sma_20 > sma_50 > sma_200:
        trend = "Strong Uptrend"
    elif current_price < sma_20 < sma_50 < sma_200:
        trend = "Strong Downtrend"
    else:
        trend = "Consolidation/Unclear"
    
    # Find recent support/resistance
    recent_high = df[column].iloc[-60:].max()
    recent_low = df[column].iloc[-60:].min()
    
    position = {
        'current_wave': f'No clear Elliott Wave - {trend}',
        'wave_progress': 0.0,
        'next_target': None,
        'key_levels': {
            'resistance': recent_high,
            'support': recent_low,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200
        },
        'confidence': 0.3,
        'analysis_method': 'trend_analysis'
    }
    
    return position

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