import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from src.analysis.core.flexible_sequence_builder import FlexibleSequenceBuilder
from src.analysis.core.validation import (
    validate_elliott_wave_pattern,
    ValidationConfig
)
from src.analysis.core.models import WaveType
from src.analysis.core.peaks import detect_peaks_troughs_enhanced

class SubwaveImportance(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"
    SKIP = "skip"

@dataclass
class SubwaveAnalysisConfig:
    def __init__(self, candlestick_type: str = 'day'):
        if candlestick_type == 'week':
            self.min_duration_days = 21  # 3 weeks minimum
            self.min_price_range_pct = 0.02  # 2% for weekly
            self.min_bars_count = 3  # 3 weekly bars minimum
        elif candlestick_type == 'month':
            self.min_duration_days = 60  # 2 months minimum
            self.min_price_range_pct = 0.05  # 5% for monthly
            self.min_bars_count = 2  # 2 monthly bars minimum
        else:  # day
            self.min_duration_days = 5
            self.min_price_range_pct = 0.01
            self.min_bars_count = 5
        
        self.wave_3_always_analyze = True
        self.wave_5_extension_threshold = 1.5
        self.confidence_penalty_missing = 0.05
        self.confidence_bonus_found = 0.03

class IntelligentSubwaveAnalyzer:
    def __init__(self, config: SubwaveAnalysisConfig = None):
        self.config = config or SubwaveAnalysisConfig()

    def analyze_subwaves_intelligently(self,
                                     df: pd.DataFrame,
                                     parent_wave_points: np.ndarray,
                                     parent_wave_type: str,
                                     labeled_points: List[Tuple[int, str]],
                                     column: str = 'close') -> Dict[str, Any]:
        if len(parent_wave_points) < 2:
            return {
                'subwave_results': [],
                'confidence_adjustment': 0.0,
                'analysis_summary': 'Insufficient parent wave points'
            }
        subwave_results = []
        confidence_adjustment = 0.0
        analysis_summary = {
            'waves_analyzed': 0,
            'waves_found': 0,
            'waves_skipped': 0,
            'critical_waves_missing': 0,
            'details': []
        }
        for i in range(len(parent_wave_points) - 1):
            segment_start = parent_wave_points[i]
            segment_end = parent_wave_points[i + 1]
            wave_number = i + 1
            importance = self._classify_wave_importance(
                wave_number, parent_wave_type, df, segment_start, segment_end, column
            )
            if importance == SubwaveImportance.SKIP:
                subwave_results.append(None)
                analysis_summary['waves_skipped'] += 1
                analysis_summary['details'].append(f"Wave {wave_number}: Skipped (too small/short)")
                continue
            subwave_result = self._analyze_single_subwave(
                df, segment_start, segment_end, wave_number,
                parent_wave_type, labeled_points, column, importance
            )
            subwave_results.append(subwave_result)
            analysis_summary['waves_analyzed'] += 1
            if subwave_result is not None:
                analysis_summary['waves_found'] += 1
                analysis_summary['details'].append(
                    f"Wave {wave_number}: Found subwave structure "
                    f"(confidence: {subwave_result.get('confidence', 0):.2f})"
                )
                if importance in [SubwaveImportance.CRITICAL, SubwaveImportance.IMPORTANT]:
                    confidence_adjustment += self.config.confidence_bonus_found
            else:
                analysis_summary['details'].append(f"Wave {wave_number}: No clear subwave structure")
                if importance == SubwaveImportance.CRITICAL:
                    analysis_summary['critical_waves_missing'] += 1
                    confidence_adjustment -= self.config.confidence_penalty_missing
                elif importance == SubwaveImportance.IMPORTANT:
                    confidence_adjustment -= self.config.confidence_penalty_missing * 0.5
        analysis_summary['success_rate'] = (
            analysis_summary['waves_found'] / max(analysis_summary['waves_analyzed'], 1)
        )
        return {
            'subwave_results': subwave_results,
            'confidence_adjustment': confidence_adjustment,
            'analysis_summary': analysis_summary
        }

    def _classify_wave_importance(self,
                                wave_number: int,
                                parent_type: str,
                                df: pd.DataFrame,
                                start_idx: int,
                                end_idx: int,
                                column: str) -> SubwaveImportance:
        duration_days = self._calculate_duration_days(df, start_idx, end_idx)
        price_range_pct = self._calculate_price_range_pct(df, start_idx, end_idx, column)
        bar_count = end_idx - start_idx
        if (duration_days < self.config.min_duration_days or 
            price_range_pct < self.config.min_price_range_pct or 
            bar_count < self.config.min_bars_count):
            return SubwaveImportance.SKIP
        if parent_type == 'impulse':
            return self._classify_impulse_wave_importance(
                wave_number, duration_days, price_range_pct, bar_count, 
                df, start_idx, end_idx, column
            )
        else:
            return self._classify_corrective_wave_importance(
                wave_number, duration_days, price_range_pct, bar_count
            )

    def _classify_impulse_wave_importance(self,
                                        wave_number: int,
                                        duration_days: int,
                                        price_range_pct: float,
                                        bar_count: int,
                                        df: pd.DataFrame,
                                        start_idx: int,
                                        end_idx: int,
                                        column: str) -> SubwaveImportance:
        if wave_number == 3:
            if self.config.wave_3_always_analyze and duration_days >= 15:
                return SubwaveImportance.CRITICAL
            elif duration_days >= 10:
                return SubwaveImportance.IMPORTANT
        elif wave_number == 1:
            if duration_days >= 30 or price_range_pct >= 0.15:
                return SubwaveImportance.IMPORTANT
            elif duration_days >= 15:
                return SubwaveImportance.OPTIONAL
        elif wave_number == 5:
            wave_5_extension = self._check_wave_5_extension(
                df, start_idx, end_idx, column
            )
            if wave_5_extension >= self.config.wave_5_extension_threshold:
                return SubwaveImportance.IMPORTANT
            elif duration_days >= 20:
                return SubwaveImportance.OPTIONAL
        elif wave_number in [2, 4]:
            if duration_days >= 25:
                return SubwaveImportance.IMPORTANT
            elif duration_days >= 15:
                return SubwaveImportance.OPTIONAL
        if duration_days >= 20 and price_range_pct >= 0.08:
            return SubwaveImportance.OPTIONAL
        return SubwaveImportance.SKIP

    def _classify_corrective_wave_importance(self,
                                           wave_number: int,
                                           duration_days: int,
                                           price_range_pct: float,
                                           bar_count: int) -> SubwaveImportance:
        if duration_days >= 30 and price_range_pct >= 0.1:
            return SubwaveImportance.IMPORTANT
        elif duration_days >= 20:
            return SubwaveImportance.OPTIONAL
        else:
            return SubwaveImportance.SKIP

    def _analyze_single_subwave(self,
                              df: pd.DataFrame,
                              start_idx: int,
                              end_idx: int,
                              wave_number: int,
                              parent_type: str,
                              labeled_points: List[Tuple[int, str]],
                              column: str,
                              importance: SubwaveImportance) -> Optional[Dict[str, Any]]:
        try:
            if parent_type == 'impulse':
                expected_subtype = 'impulse' if wave_number % 2 == 1 else 'corrective'
            else:
                expected_subtype = 'corrective'
            if importance == SubwaveImportance.CRITICAL:
                min_confidence = 0.15
            elif importance == SubwaveImportance.IMPORTANT:
                min_confidence = 0.1
            else:
                min_confidence = 0.05
            builder = FlexibleSequenceBuilder(min_confidence_threshold=min_confidence)
            result = builder.build_wave_sequence_flexible(
                df, labeled_points, start_idx, end_idx, expected_subtype, column
            )
            if result and result.confidence >= min_confidence:
                return {
                    'wave_number': wave_number,
                    'subwave_type': expected_subtype,
                    'points': result.points,
                    'confidence': result.confidence,
                    'importance': importance.value,
                    'validation_details': result.validation_details
                }
        except Exception as e:
            logger.debug(f"Subwave analysis error for Wave {wave_number}: {str(e)}")
        return None

    def _calculate_duration_days(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> int:
        try:
            start_date = df.index[start_idx]
            end_date = df.index[end_idx]
            return (end_date - start_date).days
        except:
            return end_idx - start_idx

    def _calculate_price_range_pct(self, df: pd.DataFrame, start_idx: int, end_idx: int, column: str) -> float:
        try:
            segment_prices = df[column].iloc[start_idx:end_idx+1]
            price_range = segment_prices.max() - segment_prices.min()
            start_price = df[column].iloc[start_idx]
            return price_range / abs(start_price) if start_price != 0 else 0.0
        except:
            return 0.0

    def _check_wave_5_extension(self, df: pd.DataFrame, start_idx: int, end_idx: int, column: str) -> float:
        try:
            wave_5_magnitude = abs(df[column].iloc[end_idx] - df[column].iloc[start_idx])
            return 1.2
        except:
            return 1.0

def build_wave_sequence_with_intelligent_subwaves(df: pd.DataFrame,
                                                 labeled_points: List[Tuple[int, str]],
                                                 start_idx: int,
                                                 end_idx: int,
                                                 wave_type: str = 'impulse',
                                                 column: str = 'close',
                                                 max_depth: int = 2) -> Optional[Dict[str, Any]]:
    builder = FlexibleSequenceBuilder()
    main_sequence = builder.build_wave_sequence_flexible(
        df, labeled_points, start_idx, end_idx, wave_type, column
    )
    if main_sequence is None:
        logger.debug(f"No main sequence found for {start_idx} to {end_idx}")
        return None
    logger.debug(f"Main sequence found: points={main_sequence.points}, confidence={main_sequence.confidence:.3f}")
    subwave_results = []
    confidence_adjustment = 0.0
    analysis_summary = {}
    if max_depth > 0 and len(main_sequence.points) >= 2:
        analyzer = IntelligentSubwaveAnalyzer()
        subwave_analysis = analyzer.analyze_subwaves_intelligently(
            df, main_sequence.points, wave_type, labeled_points, column
        )
        subwave_results = subwave_analysis['subwave_results']
        confidence_adjustment = subwave_analysis['confidence_adjustment']
        analysis_summary = subwave_analysis['analysis_summary']
        logger.debug(f"Subwave analysis: {analysis_summary}")
    adjusted_confidence = max(0.0, min(1.0, main_sequence.confidence + confidence_adjustment))
    sequence_points = [(int(idx), 'peak' if i % 2 == 0 else 'trough')
                      for i, idx in enumerate(main_sequence.points)]
    return {
        'sequence': sequence_points,
        'subwaves': subwave_results,
        'confidence': adjusted_confidence,
        'original_confidence': main_sequence.confidence,
        'confidence_adjustment': confidence_adjustment,
        'validation_details': {
            **main_sequence.validation_details,
            'subwave_analysis': analysis_summary,
            'intelligent_subwaves': True
        },
        'alternatives': main_sequence.alternative_points
    }

def validate_impulse_with_intelligent_subwaves(df: pd.DataFrame,
                                             wave_points: np.ndarray,
                                             column: str = 'close',
                                             subwave_results: List[Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
    """Validate impulse wave with intelligent subwave analysis"""
    # Use the new validation system with relaxed config
    config = ValidationConfig.relaxed_config()
    result = validate_elliott_wave_pattern(
        df, wave_points, column, strict_mode=False, pattern_type='impulse'
    )
    
    final_confidence = result['confidence']
    validation_details = result['validation_details']
    
    if result['is_valid'] and subwave_results:
        subwave_confidence_boost = 0.0
        subwave_insights = {
            'total_subwaves': len(subwave_results),
            'valid_subwaves': sum(1 for sw in subwave_results if sw is not None),
            'subwave_quality': []
        }
        
        for i, subwave in enumerate(subwave_results):
            if subwave is not None:
                sw_confidence = subwave.get('confidence', 0)
                sw_importance = subwave.get('importance', 'optional')
                
                if sw_importance == 'critical' and sw_confidence > 0.3:
                    subwave_confidence_boost += 0.1
                elif sw_importance == 'important' and sw_confidence > 0.2:
                    subwave_confidence_boost += 0.05
                elif sw_confidence > 0.1:
                    subwave_confidence_boost += 0.02
                
                subwave_insights['subwave_quality'].append({
                    'wave': i + 1,
                    'confidence': sw_confidence,
                    'importance': sw_importance
                })
        
        subwave_confidence_boost = min(subwave_confidence_boost, 0.2)
        final_confidence = min(result['confidence'] + subwave_confidence_boost, 1.0)
        # Reality adjustment already applied in validation.py — no second discount
        
        validation_details.update({
            'subwave_insights': subwave_insights,
            'subwave_confidence_boost': subwave_confidence_boost,
            'base_confidence': result['confidence']
        })
    else:
        final_confidence = result['confidence']  # Reality adjustment already in validation.py
    
    is_valid = final_confidence >= config.acceptance_threshold
    return is_valid, final_confidence, validation_details

def generate_impulse_candidates_with_intelligent_subwaves(df: pd.DataFrame,
                                                        peaks: np.ndarray,
                                                        troughs: np.ndarray,
                                                        start_type: str,
                                                        column: str = 'close',
                                                        min_points: int = 6,
                                                        max_points: int = 12,
                                                        config=None) -> List[Dict[str, Any]]:
    candidates = []
    labeled_points = (
        [(int(p), 'peak') for p in peaks] +
        [(int(t), 'trough') for t in troughs]
    )
    labeled_points.sort(key=lambda x: x[0])
    logger.debug(f"Labeled points: {len(labeled_points)} (peaks: {len(peaks)}, troughs: {len(troughs)})")
    cutoff_date = df.index.max() - pd.DateOffset(years=5)
    recent_indices = df.index >= cutoff_date
    valid_start_points = [sp for sp in (troughs if start_type == 'trough' else peaks)
                         if sp < len(df) and sp in df[recent_indices].index.get_indexer(df.index)]
    valid_start_points = valid_start_points[-8:]  # More permissive: last 8
    logger.debug(f"Start type: {start_type}, valid_start_points: {valid_start_points}")
    for start_idx in valid_start_points:
        min_range = max(min_points * 8, 50)
        max_range = min(len(df) - start_idx, 400)
        for range_mult in [0.4, 0.6, 0.8, 1.0]:
            end_idx = int(start_idx + max_range * range_mult)
            end_idx = min(end_idx, len(df) - 1)
            if end_idx - start_idx < min_range:
                continue
            logger.debug(f"Testing sequence: {start_idx} to {end_idx} (len={end_idx - start_idx})")
            analyzer = IntelligentSubwaveAnalyzer(config=config)
            result = build_wave_sequence_with_intelligent_subwaves(
                df, labeled_points, start_idx, end_idx, 'impulse', column, max_depth=1
            )
            if result is not None:
                sequence = result['sequence']
                confidence = result.get('confidence', 0.0)
                logger.debug(f"Candidate sequence: {sequence}, confidence: {confidence:.3f}")
                if len(sequence) >= 5 and confidence > 0.10:  # Minimum quality threshold
                    wave_points = np.array([point[0] for point in sequence])
                    is_valid, final_confidence, validation_details = validate_impulse_with_intelligent_subwaves(
                        df, wave_points, column, result.get('subwaves', [])
                    )
                    logger.debug(f"Validation: is_valid={is_valid}, final_confidence={final_confidence:.3f}")
                    if is_valid or final_confidence > 0.15:  # Meaningful confidence floor
                        wave_type = WaveType.IMPULSE
                        if validation_details.get('is_diagonal', False):
                            wave_type = WaveType.DIAGONAL_ENDING
                        candidate = {
                            "wave_points": wave_points,
                            "confidence": final_confidence,
                            "wave_type": wave_type.value,
                            "validation_details": validation_details,
                            "subwaves": result.get('subwaves', []),
                            "subwave_analysis": result.get('validation_details', {}).get('subwave_analysis', {}),
                            "intelligent_analysis": True
                        }
                        logger.debug(f"Added candidate: confidence={final_confidence:.3f}")
                        candidates.append(candidate)
                    else:
                        logger.debug(f"Rejected candidate: final_confidence={final_confidence:.3f}, reason={validation_details.get('error', 'low_confidence')}")
                else:
                    logger.debug(f"Rejected sequence: confidence={confidence:.3f}, len={len(sequence)}")
            else:
                logger.debug(f"No result for sequence {start_idx} to {end_idx}")
    logger.debug(f"Total candidates found: {len(candidates)}")
    return candidates

def test_intelligent_subwave_system(df: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("INTELLIGENT SUBWAVE ANALYSIS SYSTEM TEST")
    logger.info("=" * 80)
    peaks, troughs = detect_peaks_troughs_enhanced(df, column)
    logger.info(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
    candidates = []
    for start_type in ['trough', 'peak']:
        type_candidates = generate_impulse_candidates_with_intelligent_subwaves(
            df, peaks, troughs, start_type, column
        )
        candidates.extend(type_candidates)
    if candidates:
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        logger.info(f"\nBest candidate found:")
        logger.info(f"- Confidence: {best_candidate['confidence']:.3f}")
        logger.info(f"- Wave points: {best_candidate['wave_points']}")
        logger.info(f"- Subwaves found: {len([sw for sw in best_candidate['subwaves'] if sw is not None])}")
        subwave_analysis = best_candidate.get('subwave_analysis', {})
        logger.info(f"- Subwave success rate: {subwave_analysis.get('success_rate', 0):.2%}")
        logger.info(f"- Critical waves missing: {subwave_analysis.get('critical_waves_missing', 0)}")
        return {
            'total_candidates': len(candidates),
            'best_candidate': best_candidate,
            'intelligent_analysis': True
        }
    else:
        logger.info("No valid candidates found with intelligent analysis")
        return {'total_candidates': 0, 'best_candidate': None} 