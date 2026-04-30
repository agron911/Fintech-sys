"""
Trading Signal and Risk Assessment Generation
==============================================

This module generates trading signals and risk assessments based on
pattern relationships and conflict analysis.

Main Functions:
---------------
generate_trading_signals : Generate buy/sell/caution signals
assess_pattern_risks : Identify and assess pattern-based risks

Examples:
---------
>>> from signal_generation import generate_trading_signals
>>> signals = generate_trading_signals(relationships, patterns_dict)
"""

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def generate_trading_signals(relationships: Dict[str, Any],
                           patterns_dict: Dict[str, Dict],
                           regime_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Generate trading signals based on pattern relationships.

    Parameters
    ----------
    relationships : Dict[str, Any]
        Pattern relationship analysis results
    patterns_dict : Dict[str, Dict]
        All patterns by timeframe
    regime_info : Dict[str, Any], optional
        Market regime detection output from regime.detect_regime().
        When provided, signal confidence is adjusted:
        - 'trending' regime boosts trend-following (strong_buy/strong_sell) by +10%
        - 'ranging' regime boosts caution/mixed signals, penalizes directional by -10%
        - 'volatile' regime reduces all confidence by -8%

    Returns
    -------
    List[Dict[str, Any]]
        List of trading signal dictionaries containing:
        - type: 'strong_buy', 'strong_sell', 'caution', 'mixed'
        - confidence: Signal confidence [0, 1]
        - reason: Explanation of signal
        - timeframes: Involved timeframes
        - regime: Market regime label (if regime_info provided)

    Notes
    -----
    Strong signals require 2+ confirmations with >0.7 average strength.
    Conflicts trigger caution signals. Mixed signals indicate uncertainty.
    """
    signals = []

    confirmations = relationships.get('confirmations', [])
    conflicts = relationships.get('conflicts', [])

    # Extract regime context for confidence adjustments
    regime_type = regime_info.get('regime', 'unknown') if regime_info else 'unknown'

    try:
        # Strong confirmation signal
        if len(confirmations) >= 2:
            avg_strength = np.mean([c['strength'] for c in confirmations])
            if avg_strength > 0.7:
                # Determine direction from actual pattern direction, not just agreement
                first_conf = confirmations[0]
                is_bullish = first_conf.get('direction', 'neutral') == 'bullish'

                confidence = avg_strength
                # Regime adjustment: trending markets favor directional signals
                if regime_type == 'trending':
                    confidence = min(confidence * 1.10, 1.0)
                elif regime_type == 'ranging':
                    confidence *= 0.90
                elif regime_type == 'volatile':
                    confidence *= 0.92

                signal = {
                    'type': 'strong_buy' if is_bullish else 'strong_sell',
                    'confidence': confidence,
                    'reason': f'Multiple pattern confirmations ({len(confirmations)})',
                    'timeframes': list(set(
                        [c['pattern1']['timeframe'] for c in confirmations] +
                        [c['pattern2']['timeframe'] for c in confirmations]
                    ))
                }
                if regime_type != 'unknown':
                    signal['regime'] = regime_type
                signals.append(signal)
                logger.info(f"Generated strong signal: {'buy' if is_bullish else 'sell'}, "
                           f"confidence={confidence:.2f}, regime={regime_type}")

        # Conflict warning signal
        if len(conflicts) >= 2:
            caution_confidence = 0.8
            # Ranging/volatile regimes reinforce caution
            if regime_type in ('ranging', 'volatile'):
                caution_confidence = min(caution_confidence * 1.10, 1.0)

            signal = {
                'type': 'caution',
                'confidence': caution_confidence,
                'reason': f'Pattern conflicts detected ({len(conflicts)})',
                'timeframes': list(set(
                    [c['pattern1']['timeframe'] for c in conflicts] +
                    [c['pattern2']['timeframe'] for c in conflicts]
                ))
            }
            if regime_type != 'unknown':
                signal['regime'] = regime_type
            signals.append(signal)
            logger.warning(f"Pattern conflicts detected: {len(conflicts)}")

        # Mixed signal
        if len(confirmations) >= 1 and len(conflicts) >= 1:
            mixed_confidence = 0.6
            if regime_type == 'volatile':
                mixed_confidence *= 0.92

            signal = {
                'type': 'mixed',
                'confidence': mixed_confidence,
                'reason': 'Mixed pattern signals - use additional confirmation',
                'confirmations': len(confirmations),
                'conflicts': len(conflicts)
            }
            if regime_type != 'unknown':
                signal['regime'] = regime_type
            signals.append(signal)
            logger.info("Mixed signals detected - caution advised")

    except Exception as e:
        logger.error(f"Error generating trading signals: {e}", exc_info=True)

    return signals


def assess_pattern_risks(relationships: Dict[str, Any],
                        patterns_dict: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Assess risks based on pattern relationships.

    Parameters
    ----------
    relationships : Dict[str, Any]
        Pattern relationship analysis
    patterns_dict : Dict[str, Dict]
        All patterns by timeframe

    Returns
    -------
    List[Dict[str, Any]]
        List of risk assessment dictionaries containing:
        - level: 'high', 'medium', 'low'
        - type: Risk category
        - description: Risk explanation
        - mitigation: Suggested mitigation strategy

    Notes
    -----
    Assesses multiple risk types:
    - Pattern conflicts (3+ conflicts = high risk)
    - Low alignment across timeframes (<0.3 = medium risk)
    - Timeframe divergence (2+ cross-timeframe conflicts = medium risk)
    """
    risks = []

    conflicts = relationships.get('conflicts', [])
    alignment_score = relationships.get('alignment_score', 0.5)

    try:
        # High conflict risk
        if len(conflicts) >= 3:
            risks.append({
                'level': 'high',
                'type': 'pattern_conflicts',
                'description': f'Multiple pattern conflicts ({len(conflicts)}) detected',
                'mitigation': 'Wait for pattern resolution or use smaller position sizes'
            })
            logger.warning(f"High risk: {len(conflicts)} pattern conflicts")

        # Low alignment risk
        if alignment_score < 0.3:
            risks.append({
                'level': 'medium',
                'type': 'low_alignment',
                'description': 'Low pattern alignment across timeframes',
                'mitigation': 'Seek additional technical confirmation'
            })
            logger.info(f"Medium risk: Low alignment score {alignment_score:.2f}")

        # Timeframe divergence risk
        timeframes = list(patterns_dict.keys())
        if len(timeframes) >= 3:
            timeframe_conflicts = sum(
                1 for c in conflicts
                if c['pattern1']['timeframe'] != c['pattern2']['timeframe']
            )
            if timeframe_conflicts >= 2:
                risks.append({
                    'level': 'medium',
                    'type': 'timeframe_divergence',
                    'description': 'Conflicting signals across multiple timeframes',
                    'mitigation': 'Focus on higher timeframe signals'
                })
                logger.info(f"Medium risk: {timeframe_conflicts} timeframe conflicts")

    except Exception as e:
        logger.error(f"Error assessing pattern risks: {e}", exc_info=True)

    return risks
