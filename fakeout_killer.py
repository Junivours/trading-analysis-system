# Fakeout Killer Engine - Advanced Fakeout Detection and Protection
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FakeoutKillerEngine:
    """Advanced fakeout detection and protection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fakeout_patterns = {
            'VOLUME_DIVERGENCE': {'weight': 0.3, 'threshold': 0.7},
            'PATTERN_FAILURE': {'weight': 0.25, 'threshold': 0.6},
            'MOMENTUM_WEAKNESS': {'weight': 0.2, 'threshold': 0.65},
            'SUPPORT_RESISTANCE_BREAK': {'weight': 0.25, 'threshold': 0.8}
        }
    
    def analyze_fakeout_probability(self, symbol: str, pattern_data: Dict, 
                                  price_data: List[Dict], indicators: Dict) -> Dict[str, Any]:
        """Analyze probability of fakeout and generate protection signals"""
        try:
            if not price_data or len(price_data) < 20:
                return self._fallback_analysis(symbol, indicators)
            
            # Analyze different fakeout indicators
            volume_analysis = self._analyze_volume_divergence(price_data, indicators)
            pattern_analysis = self._analyze_pattern_reliability(pattern_data, price_data)
            momentum_analysis = self._analyze_momentum_weakness(price_data, indicators)
            sr_analysis = self._analyze_support_resistance(price_data, indicators)
            
            # Calculate overall fakeout probability
            fakeout_probability = self._calculate_fakeout_probability(
                volume_analysis, pattern_analysis, momentum_analysis, sr_analysis
            )
            
            # Generate protection signals
            protection_signals = self._generate_protection_signals(
                volume_analysis, pattern_analysis, momentum_analysis, sr_analysis
            )
            
            # Determine protection level
            protection_level = self._determine_protection_level(fakeout_probability)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                fakeout_probability, protection_signals, protection_level
            )
            
            return {
                'fakeout_probability': int(fakeout_probability),
                'fakeout_type': self._classify_fakeout_type(fakeout_probability),
                'protection_signals': protection_signals,
                'protection_level': protection_level,
                'confidence_score': self._calculate_confidence(volume_analysis, pattern_analysis),
                'recommendations': recommendations,
                'killer_active': fakeout_probability > 60,
                'analysis_details': {
                    'volume_score': volume_analysis['score'],
                    'pattern_score': pattern_analysis['score'],
                    'momentum_score': momentum_analysis['score'],
                    'sr_score': sr_analysis['score']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fakeout analysis failed for {symbol}: {e}")
            return self._fallback_analysis(symbol, indicators)
    
    def _analyze_volume_divergence(self, price_data: List[Dict], indicators: Dict) -> Dict[str, Any]:
        """Analyze volume divergence patterns"""
        try:
            if len(price_data) < 10:
                return {'score': 0.5, 'divergence_detected': False, 'strength': 'UNKNOWN'}
            
            # Get recent volume data
            recent_volumes = [p.get('volume', 0) for p in price_data[-5:]]
            historical_volumes = [p.get('volume', 0) for p in price_data[-20:-5]]
            
            if not recent_volumes or not historical_volumes:
                return {'score': 0.5, 'divergence_detected': False, 'strength': 'UNKNOWN'}
            
            recent_avg = np.mean(recent_volumes)
            historical_avg = np.mean(historical_volumes)
            
            # Check for volume divergence
            volume_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
            
            # Volume divergence scoring
            if volume_ratio < 0.6:  # Low volume on move = potential fakeout
                score = 0.8
                divergence_detected = True
                strength = 'STRONG'
            elif volume_ratio < 0.8:
                score = 0.6
                divergence_detected = True
                strength = 'MODERATE'
            elif volume_ratio > 2.0:  # Very high volume = likely genuine
                score = 0.2
                divergence_detected = False
                strength = 'CONFIRMING'
            else:
                score = 0.4
                divergence_detected = False
                strength = 'NEUTRAL'
            
            return {
                'score': score,
                'divergence_detected': divergence_detected,
                'strength': strength,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            return {'score': 0.5, 'divergence_detected': False, 'strength': 'ERROR'}
    
    def _analyze_pattern_reliability(self, pattern_data: Dict, price_data: List[Dict]) -> Dict[str, Any]:
        """Analyze reliability of detected patterns"""
        try:
            detected_patterns = pattern_data.get('detected_patterns', [])
            
            if not detected_patterns:
                return {'score': 0.3, 'reliability': 'LOW', 'pattern_count': 0}
            
            # Analyze pattern quality
            pattern_count = len(detected_patterns)
            
            # Check for conflicting patterns (sign of potential fakeout)
            bullish_patterns = sum(1 for p in detected_patterns if p.get('direction') == 'BULLISH')
            bearish_patterns = sum(1 for p in detected_patterns if p.get('direction') == 'BEARISH')
            
            # Pattern reliability scoring
            if pattern_count > 3 and abs(bullish_patterns - bearish_patterns) <= 1:
                # Too many conflicting patterns = likely fakeout
                score = 0.75
                reliability = 'CONFLICTING'
            elif pattern_count == 1:
                # Single clear pattern = more reliable
                score = 0.25
                reliability = 'HIGH'
            elif pattern_count <= 2:
                # Few patterns = moderate reliability
                score = 0.4
                reliability = 'MODERATE'
            else:
                # Multiple patterns but not conflicting
                score = 0.6
                reliability = 'MIXED'
            
            return {
                'score': score,
                'reliability': reliability,
                'pattern_count': pattern_count,
                'bullish_count': bullish_patterns,
                'bearish_count': bearish_patterns
            }
            
        except Exception as e:
            return {'score': 0.5, 'reliability': 'ERROR', 'pattern_count': 0}
    
    def _analyze_momentum_weakness(self, price_data: List[Dict], indicators: Dict) -> Dict[str, Any]:
        """Analyze momentum weakness indicators"""
        try:
            if len(price_data) < 10:
                return {'score': 0.5, 'weakness_detected': False, 'trend': 'UNKNOWN'}
            
            # Get RSI and momentum indicators
            rsi = indicators.get('current_rsi_14', 50)
            
            # Calculate price momentum
            closes = [p['close'] for p in price_data[-10:]]
            if len(closes) < 5:
                return {'score': 0.5, 'weakness_detected': False, 'trend': 'UNKNOWN'}
            
            short_momentum = (closes[-1] - closes[-3]) / closes[-3] * 100
            medium_momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            
            # Momentum divergence analysis
            momentum_divergence = False
            weakness_score = 0.3
            
            # Check for momentum weakness
            if abs(short_momentum) < 1 and abs(medium_momentum) < 2:
                # Weak momentum on move = potential fakeout
                weakness_score = 0.7
                momentum_divergence = True
                trend = 'WEAKENING'
            elif (rsi > 70 and short_momentum < 0) or (rsi < 30 and short_momentum > 0):
                # RSI divergence
                weakness_score = 0.6
                momentum_divergence = True
                trend = 'DIVERGING'
            else:
                trend = 'STRONG' if abs(short_momentum) > 2 else 'MODERATE'
            
            return {
                'score': weakness_score,
                'weakness_detected': momentum_divergence,
                'trend': trend,
                'short_momentum': short_momentum,
                'rsi': rsi
            }
            
        except Exception as e:
            return {'score': 0.5, 'weakness_detected': False, 'trend': 'ERROR'}
    
    def _analyze_support_resistance(self, price_data: List[Dict], indicators: Dict) -> Dict[str, Any]:
        """Analyze support/resistance break quality"""
        try:
            if len(price_data) < 20:
                return {'score': 0.5, 'break_quality': 'UNKNOWN', 'level_strength': 'UNKNOWN'}
            
            # Get recent price action
            closes = [p['close'] for p in price_data[-20:]]
            highs = [p['high'] for p in price_data[-20:]]
            lows = [p['low'] for p in price_data[-20:]]
            
            current_price = closes[-1]
            
            # Find potential support/resistance levels
            resistance_level = max(highs[-10:-1])  # Recent high
            support_level = min(lows[-10:-1])  # Recent low
            
            # Check break quality
            break_quality_score = 0.4
            
            # Analyze if we're near a break
            near_resistance = abs(current_price - resistance_level) / resistance_level < 0.02
            near_support = abs(current_price - support_level) / support_level < 0.02
            
            if near_resistance or near_support:
                # Check for weak break (potential fakeout)
                price_range = resistance_level - support_level
                if price_range > 0:
                    break_strength = abs(current_price - (resistance_level if near_resistance else support_level)) / price_range
                    
                    if break_strength < 0.005:  # Very small break = likely fakeout
                        break_quality_score = 0.8
                        break_quality = 'WEAK_BREAK'
                        level_strength = 'STRONG'
                    elif break_strength < 0.02:  # Small break
                        break_quality_score = 0.6
                        break_quality = 'MODERATE_BREAK'
                        level_strength = 'MODERATE'
                    else:  # Strong break
                        break_quality_score = 0.2
                        break_quality = 'STRONG_BREAK'
                        level_strength = 'WEAK'
                else:
                    break_quality = 'UNCLEAR'
                    level_strength = 'UNCLEAR'
            else:
                break_quality = 'NO_BREAK'
                level_strength = 'HOLDING'
            
            return {
                'score': break_quality_score,
                'break_quality': break_quality,
                'level_strength': level_strength,
                'resistance_level': resistance_level,
                'support_level': support_level,
                'current_price': current_price
            }
            
        except Exception as e:
            return {'score': 0.5, 'break_quality': 'ERROR', 'level_strength': 'ERROR'}
    
    def _calculate_fakeout_probability(self, volume_analysis: Dict, pattern_analysis: Dict,
                                     momentum_analysis: Dict, sr_analysis: Dict) -> float:
        """Calculate overall fakeout probability"""
        try:
            # Weighted scoring
            weights = {
                'volume': 0.3,
                'pattern': 0.25,
                'momentum': 0.2,
                'sr': 0.25
            }
            
            total_score = (
                volume_analysis['score'] * weights['volume'] +
                pattern_analysis['score'] * weights['pattern'] +
                momentum_analysis['score'] * weights['momentum'] +
                sr_analysis['score'] * weights['sr']
            )
            
            # Convert to percentage
            probability = total_score * 100
            return min(95, max(5, probability))
            
        except Exception as e:
            return 50.0  # Default moderate probability
    
    def _generate_protection_signals(self, volume_analysis: Dict, pattern_analysis: Dict,
                                   momentum_analysis: Dict, sr_analysis: Dict) -> List[Dict]:
        """Generate protection signals based on analysis"""
        signals = []
        
        try:
            # Volume confirmation signal
            if volume_analysis['divergence_detected']:
                signals.append({
                    'signal': 'VOLUME_DIVERGENCE',
                    'status': 'ACTIVE',
                    'strength': volume_analysis['strength'],
                    'description': 'Volume divergence detected - potential fakeout'
                })
            else:
                signals.append({
                    'signal': 'VOLUME_CONFIRMATION',
                    'status': 'ACTIVE' if volume_analysis['score'] < 0.4 else 'INACTIVE',
                    'strength': 'HIGH' if volume_analysis['score'] < 0.3 else 'MODERATE',
                    'description': 'Volume supports the move'
                })
            
            # Pattern reliability signal
            signals.append({
                'signal': 'PATTERN_VERIFICATION',
                'status': 'ACTIVE' if pattern_analysis['reliability'] in ['HIGH', 'MODERATE'] else 'INACTIVE',
                'strength': pattern_analysis['reliability'],
                'description': f"Pattern reliability: {pattern_analysis['reliability']}"
            })
            
            # Momentum confirmation
            if momentum_analysis['weakness_detected']:
                signals.append({
                    'signal': 'MOMENTUM_WEAKNESS',
                    'status': 'ACTIVE',
                    'strength': 'HIGH' if momentum_analysis['score'] > 0.6 else 'MODERATE',
                    'description': 'Momentum weakness detected'
                })
            
            # Support/Resistance quality
            if sr_analysis['break_quality'] in ['WEAK_BREAK', 'MODERATE_BREAK']:
                signals.append({
                    'signal': 'WEAK_BREAK',
                    'status': 'ACTIVE',
                    'strength': 'HIGH' if sr_analysis['break_quality'] == 'WEAK_BREAK' else 'MODERATE',
                    'description': f"Break quality: {sr_analysis['break_quality']}"
                })
            
            return signals[:4]  # Limit to top 4 signals
            
        except Exception as e:
            return [{'signal': 'ERROR', 'status': 'INACTIVE', 'strength': 'LOW', 'description': 'Signal generation failed'}]
    
    def _determine_protection_level(self, fakeout_probability: float) -> str:
        """Determine protection level based on fakeout probability"""
        if fakeout_probability >= 75:
            return 'MAXIMUM'
        elif fakeout_probability >= 60:
            return 'HIGH'
        elif fakeout_probability >= 40:
            return 'MODERATE'
        elif fakeout_probability >= 25:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _classify_fakeout_type(self, probability: float) -> str:
        """Classify the type of fakeout based on probability"""
        if probability >= 80:
            return 'VERY_HIGH_PROBABILITY'
        elif probability >= 65:
            return 'HIGH_PROBABILITY'
        elif probability >= 45:
            return 'MEDIUM_PROBABILITY'
        elif probability >= 25:
            return 'LOW_PROBABILITY'
        else:
            return 'VERY_LOW_PROBABILITY'
    
    def _calculate_confidence(self, volume_analysis: Dict, pattern_analysis: Dict) -> int:
        """Calculate confidence in fakeout analysis"""
        try:
            base_confidence = 60
            
            # Volume analysis confidence boost
            if volume_analysis['strength'] in ['STRONG', 'CONFIRMING']:
                base_confidence += 15
            elif volume_analysis['strength'] == 'MODERATE':
                base_confidence += 8
            
            # Pattern analysis confidence boost
            if pattern_analysis['reliability'] == 'HIGH':
                base_confidence += 15
            elif pattern_analysis['reliability'] == 'MODERATE':
                base_confidence += 8
            
            return min(85, max(35, base_confidence))
            
        except Exception:
            return 60
    
    def _generate_recommendations(self, fakeout_probability: float, 
                                protection_signals: List[Dict], protection_level: str) -> List[str]:
        """Generate recommendations based on fakeout analysis"""
        recommendations = []
        
        try:
            # Main recommendation based on probability
            if fakeout_probability >= 70:
                recommendations.append(f"🚨 HIGH FAKEOUT RISK ({fakeout_probability:.0f}%) - Exercise extreme caution")
                recommendations.append("⏳ Wait for additional confirmation before entering")
            elif fakeout_probability >= 50:
                recommendations.append(f"⚠️ MODERATE FAKEOUT RISK ({fakeout_probability:.0f}%) - Use smaller position sizes")
                recommendations.append("📊 Require volume confirmation for entry")
            elif fakeout_probability >= 30:
                recommendations.append(f"✅ LOW FAKEOUT RISK ({fakeout_probability:.0f}%) - Proceed with normal caution")
            else:
                recommendations.append(f"🎯 VERY LOW FAKEOUT RISK ({fakeout_probability:.0f}%) - High confidence setup")
            
            # Protection level recommendation
            recommendations.append(f"🛡️ Protection level: {protection_level}")
            
            # Active signal recommendations
            active_signals = [s for s in protection_signals if s.get('status') == 'ACTIVE']
            if active_signals:
                recommendations.append(f"🔍 Active warnings: {len(active_signals)} signals detected")
            
            return recommendations[:4]  # Limit recommendations
            
        except Exception:
            return [f"Fakeout probability: {fakeout_probability:.0f}%", "Analysis recommendations unavailable"]
    
    def _fallback_analysis(self, symbol: str, indicators: Dict) -> Dict[str, Any]:
        """Fallback analysis when main fakeout analysis fails"""
        rsi = indicators.get('current_rsi_14', 50)
        
        # Simple RSI-based fakeout estimation
        if rsi > 75 or rsi < 25:
            probability = 60  # Higher fakeout risk at extremes
            protection_level = 'HIGH'
        else:
            probability = 35  # Lower risk in normal range
            protection_level = 'MODERATE'
        
        return {
            'fakeout_probability': probability,
            'fakeout_type': 'MODERATE_PROBABILITY',
            'protection_signals': [{
                'signal': 'BASIC_RSI_CHECK',
                'status': 'ACTIVE' if rsi > 70 or rsi < 30 else 'INACTIVE',
                'strength': 'MODERATE',
                'description': f'RSI at {rsi} level'
            }],
            'protection_level': protection_level,
            'confidence_score': 45,
            'recommendations': [
                f"Fallback fakeout analysis for {symbol}",
                f"RSI-based risk assessment: {probability}%",
                "Consider additional confirmation"
            ],
            'killer_active': probability > 50,
            'analysis_details': {
                'volume_score': 0.5,
                'pattern_score': 0.5,
                'momentum_score': 0.5,
                'sr_score': 0.5
            }
        }
