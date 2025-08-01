# Market DNA Engine - Advanced Market Personality Analysis
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MarketDNAEngine:
    """Advanced market personality and DNA analysis engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dna_patterns = {
            'AGGRESSIVE': {'threshold': 70, 'characteristics': ['high_volatility', 'quick_moves']},
            'CONSERVATIVE': {'threshold': 30, 'characteristics': ['low_volatility', 'slow_moves']},
            'ADAPTIVE': {'threshold': 50, 'characteristics': ['balanced_moves', 'moderate_volatility']}
        }
    
    def analyze_market_dna(self, symbol: str, price_data: List[Dict], 
                          volume_data: List[float], indicators: Dict) -> Dict[str, Any]:
        """Analyze market DNA and personality"""
        try:
            if not price_data or len(price_data) < 20:
                return self._fallback_analysis(symbol, indicators)
            
            # Calculate DNA metrics
            volatility = self._calculate_volatility(price_data)
            momentum = self._calculate_momentum(price_data)
            volume_profile = self._analyze_volume_profile(volume_data)
            
            # Determine market personality
            personality = self._determine_personality(volatility, momentum, volume_profile)
            
            # Generate DNA-based signals
            signals = self._generate_dna_signals(personality, indicators)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(volatility, momentum, volume_profile)
            
            return {
                'market_personality': personality['name'],
                'dna_type': personality['type'],
                'confidence_score': confidence,
                'personalized_signals': signals,
                'dna_patterns': {
                    'trend_dna': personality.get('trend_bias', 'NEUTRAL'),
                    'volume_dna': volume_profile['profile'],
                    'volatility_dna': volatility['level']
                },
                'recommendations': self._generate_recommendations(personality, signals),
                'dna_strength': personality.get('strength', 'MODERATE')
            }
            
        except Exception as e:
            self.logger.error(f"Market DNA analysis failed for {symbol}: {e}")
            return self._fallback_analysis(symbol, indicators)
    
    def _calculate_volatility(self, price_data: List[Dict]) -> Dict[str, Any]:
        """Calculate market volatility metrics"""
        try:
            closes = [p['close'] for p in price_data[-20:]]
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            if volatility > 0.4:
                level = 'HIGH'
            elif volatility > 0.2:
                level = 'MODERATE'
            else:
                level = 'LOW'
            
            return {'value': volatility, 'level': level}
        except:
            return {'value': 0.2, 'level': 'MODERATE'}
    
    def _calculate_momentum(self, price_data: List[Dict]) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        try:
            closes = [p['close'] for p in price_data[-14:]]
            if len(closes) < 14:
                return {'value': 0, 'direction': 'NEUTRAL'}
            
            momentum = (closes[-1] - closes[0]) / closes[0] * 100
            
            if momentum > 5:
                direction = 'STRONG_BULLISH'
            elif momentum > 2:
                direction = 'BULLISH'
            elif momentum < -5:
                direction = 'STRONG_BEARISH'
            elif momentum < -2:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            return {'value': momentum, 'direction': direction}
        except:
            return {'value': 0, 'direction': 'NEUTRAL'}
    
    def _analyze_volume_profile(self, volume_data: List[float]) -> Dict[str, Any]:
        """Analyze volume profile and patterns"""
        try:
            if len(volume_data) < 10:
                return {'profile': 'UNKNOWN', 'trend': 'NEUTRAL'}
            
            recent_avg = np.mean(volume_data[-5:])
            historical_avg = np.mean(volume_data[-20:-5]) if len(volume_data) >= 20 else np.mean(volume_data[:-5])
            
            if recent_avg > historical_avg * 1.5:
                profile = 'HIGH_VOLUME'
                trend = 'INCREASING'
            elif recent_avg < historical_avg * 0.7:
                profile = 'LOW_VOLUME'
                trend = 'DECREASING'
            else:
                profile = 'NORMAL_VOLUME'
                trend = 'STABLE'
            
            return {'profile': profile, 'trend': trend, 'ratio': recent_avg / historical_avg}
        except:
            return {'profile': 'NORMAL_VOLUME', 'trend': 'STABLE', 'ratio': 1.0}
    
    def _determine_personality(self, volatility: Dict, momentum: Dict, volume: Dict) -> Dict[str, Any]:
        """Determine market personality based on DNA analysis"""
        try:
            # Scoring system
            aggression_score = 0
            
            # Volatility contribution
            if volatility['level'] == 'HIGH':
                aggression_score += 40
            elif volatility['level'] == 'MODERATE':
                aggression_score += 20
            
            # Momentum contribution
            if 'STRONG' in momentum['direction']:
                aggression_score += 30
            elif momentum['direction'] in ['BULLISH', 'BEARISH']:
                aggression_score += 15
            
            # Volume contribution
            if volume['profile'] == 'HIGH_VOLUME':
                aggression_score += 20
            elif volume['profile'] == 'NORMAL_VOLUME':
                aggression_score += 10
            
            # Determine personality
            if aggression_score >= 70:
                return {
                    'name': '🔥 Aggressive Trader DNA',
                    'type': 'AGGRESSIVE',
                    'strength': 'HIGH',
                    'trend_bias': 'MOMENTUM_FOCUSED'
                }
            elif aggression_score >= 40:
                return {
                    'name': '⚖️ Balanced Trader DNA',
                    'type': 'ADAPTIVE',
                    'strength': 'MODERATE',
                    'trend_bias': 'TREND_FOLLOWING'
                }
            else:
                return {
                    'name': '🛡️ Conservative Trader DNA',
                    'type': 'CONSERVATIVE',
                    'strength': 'LOW',
                    'trend_bias': 'RISK_AVERSE'
                }
        except:
            return {
                'name': '⚖️ Balanced Trader DNA',
                'type': 'ADAPTIVE',
                'strength': 'MODERATE',
                'trend_bias': 'NEUTRAL'
            }
    
    def _generate_dna_signals(self, personality: Dict, indicators: Dict) -> List[Dict]:
        """Generate personalized trading signals based on DNA"""
        signals = []
        
        try:
            rsi = indicators.get('current_rsi_14', 50)
            personality_type = personality.get('type', 'ADAPTIVE')
            
            if personality_type == 'AGGRESSIVE':
                # Aggressive DNA - Quick entries/exits
                if rsi < 35:
                    signals.append({
                        'type': 'AGGRESSIVE_BUY',
                        'confidence': 75,
                        'reason': 'Aggressive DNA: Quick oversold bounce opportunity'
                    })
                elif rsi > 65:
                    signals.append({
                        'type': 'AGGRESSIVE_SELL',
                        'confidence': 70,
                        'reason': 'Aggressive DNA: Quick overbought exit signal'
                    })
            
            elif personality_type == 'CONSERVATIVE':
                # Conservative DNA - Safe entries
                if rsi < 25:
                    signals.append({
                        'type': 'SAFE_ACCUMULATE',
                        'confidence': 65,
                        'reason': 'Conservative DNA: Safe accumulation zone'
                    })
                elif rsi > 75:
                    signals.append({
                        'type': 'SAFE_REDUCE',
                        'confidence': 60,
                        'reason': 'Conservative DNA: Risk reduction recommended'
                    })
            
            else:  # ADAPTIVE
                # Balanced approach
                if 30 <= rsi <= 70:
                    signals.append({
                        'type': 'BALANCED_HOLD',
                        'confidence': 60,
                        'reason': 'Adaptive DNA: Balanced market conditions'
                    })
                else:
                    signals.append({
                        'type': 'ADAPTIVE_WAIT',
                        'confidence': 55,
                        'reason': 'Adaptive DNA: Wait for better conditions'
                    })
            
            return signals[:3]  # Limit to top 3 signals
            
        except Exception as e:
            return [{'type': 'ERROR', 'confidence': 0, 'reason': f'Signal generation failed: {e}'}]
    
    def _calculate_confidence(self, volatility: Dict, momentum: Dict, volume: Dict) -> int:
        """Calculate overall confidence in DNA analysis"""
        try:
            base_confidence = 50
            
            # Volatility confidence boost
            if volatility['level'] in ['HIGH', 'MODERATE']:
                base_confidence += 15
            
            # Momentum confidence boost
            if momentum['direction'] != 'NEUTRAL':
                base_confidence += 20
            
            # Volume confirmation
            if volume['profile'] != 'UNKNOWN':
                base_confidence += 10
            
            return min(85, max(30, base_confidence))
        except:
            return 50
    
    def _generate_recommendations(self, personality: Dict, signals: List[Dict]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        try:
            personality_type = personality.get('type', 'ADAPTIVE')
            
            if personality_type == 'AGGRESSIVE':
                recommendations.extend([
                    "🔥 Your aggressive DNA suggests quick decision-making",
                    "⚡ Focus on momentum breakouts and quick scalps",
                    "⏰ Use tight stops and quick profit-taking"
                ])
            elif personality_type == 'CONSERVATIVE':
                recommendations.extend([
                    "🛡️ Your conservative DNA suggests patience",
                    "📊 Focus on high-probability setups only",
                    "💰 Use wide stops and longer timeframes"
                ])
            else:
                recommendations.extend([
                    "⚖️ Your adaptive DNA suggests flexibility",
                    "📈 Adjust strategy based on market conditions",
                    "🎯 Balance risk and reward appropriately"
                ])
            
            # Add signal-based recommendations
            if signals:
                recommendations.append(f"💡 Current signal: {signals[0].get('reason', 'No specific signal')}")
            
            return recommendations[:4]  # Limit recommendations
            
        except:
            return ["DNA analysis recommendations temporarily unavailable"]
    
    def _fallback_analysis(self, symbol: str, indicators: Dict) -> Dict[str, Any]:
        """Fallback analysis when main DNA analysis fails"""
        rsi = indicators.get('current_rsi_14', 50)
        
        return {
            'market_personality': '⚖️ Standard Market DNA',
            'dna_type': 'ADAPTIVE',
            'confidence_score': 45,
            'personalized_signals': [{
                'type': 'STANDARD_SIGNAL',
                'confidence': 45,
                'reason': f"Standard analysis for {symbol}"
            }],
            'dna_patterns': {
                'trend_dna': 'NEUTRAL',
                'volume_dna': 'NORMAL',
                'volatility_dna': 'MODERATE'
            },
            'recommendations': [
                "Using fallback DNA analysis",
                f"RSI level: {rsi}",
                "Consider multiple timeframe analysis"
            ],
            'dna_strength': 'MODERATE'
        }
