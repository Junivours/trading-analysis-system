"""
🎯 TRADING SETUPS - Separate Datei
Generiert Trading Signale und Entry/Exit Points  
"""

import numpy as np
from typing import Dict, List, Tuple

class TradingSetupAnalyzer:
    """🎯 Professionelle Trading Setup Analyse"""
    
    def __init__(self):
        self.signal_weights = {
            'technical_indicators': 0.40,    # 40% Gewichtung
            'chart_patterns': 0.25,          # 25% Gewichtung  
            'risk_management': 0.20,         # 20% Gewichtung
            'market_sentiment': 0.15         # 15% Gewichtung
        }
        
        self.risk_reward_ratios = {
            'conservative': 2.0,   # 1:2 Risk/Reward
            'moderate': 1.5,       # 1:1.5 Risk/Reward
            'aggressive': 1.2      # 1:1.2 Risk/Reward
        }
    
    def analyze_trading_setup(self, current_price: float, market_data: List[Dict], 
                            technical_indicators: Dict, chart_patterns: Dict = None, 
                            liquidation_data: Dict = None) -> Dict:
        """🎯 Komplette Trading Setup Analyse"""
        try:
            if not market_data or len(market_data) < 20:
                return {'error': 'Nicht genügend Marktdaten für Trading Setup'}
            
            # 1. Technische Analyse auswerten
            tech_signal = self._analyze_technical_signals(technical_indicators)
            
            # 2. Chartmuster einbeziehen (falls verfügbar)
            pattern_signal = self._analyze_pattern_signals(chart_patterns) if chart_patterns else {'signal': 'NEUTRAL', 'confidence': 50}
            
            # 3. Risk Management berechnen
            risk_analysis = self._calculate_risk_management(current_price, market_data, liquidation_data)
            
            # 4. Market Sentiment bewerten
            sentiment_signal = self._analyze_market_sentiment(market_data, technical_indicators)
            
            # 5. Gesamt-Signal berechnen
            combined_signal = self._calculate_combined_signal(tech_signal, pattern_signal, risk_analysis, sentiment_signal)
            
            # 6. Trading Setup generieren
            trading_setup = self._generate_trading_setup(current_price, combined_signal, risk_analysis, market_data)
            
            return {
                'success': True,
                'current_price': current_price,
                'signals': {
                    'technical': tech_signal,
                    'patterns': pattern_signal,
                    'risk': risk_analysis,
                    'sentiment': sentiment_signal,
                    'combined': combined_signal
                },
                'trading_setup': trading_setup,
                'confidence': combined_signal['confidence'],
                'recommendation': self._generate_recommendation(combined_signal, risk_analysis)
            }
            
        except Exception as e:
            return {'error': f'Trading Setup Fehler: {str(e)}'}
    
    def _analyze_technical_signals(self, indicators: Dict) -> Dict:
        """📈 Analysiert technische Indikatoren"""
        try:
            signals = []
            bullish_count = 0
            bearish_count = 0
            total_weight = 0
            
            # RSI Signal
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                signals.append({'indicator': 'RSI', 'signal': 'BULLISH', 'strength': 80, 'reason': 'Überverkauft'})
                bullish_count += 2
            elif rsi < 40:
                signals.append({'indicator': 'RSI', 'signal': 'BULLISH', 'strength': 60, 'reason': 'Leicht überverkauft'})
                bullish_count += 1
            elif rsi > 70:
                signals.append({'indicator': 'RSI', 'signal': 'BEARISH', 'strength': 80, 'reason': 'Überkauft'})
                bearish_count += 2
            elif rsi > 60:
                signals.append({'indicator': 'RSI', 'signal': 'BEARISH', 'strength': 60, 'reason': 'Leicht überkauft'})
                bearish_count += 1
            else:
                signals.append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'strength': 50, 'reason': 'Neutral Zone'})
            total_weight += 1
            
            # MACD Signal
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal and macd > 0:
                signals.append({'indicator': 'MACD', 'signal': 'BULLISH', 'strength': 75, 'reason': 'Bullish Crossover'})
                bullish_count += 2
            elif macd > macd_signal:
                signals.append({'indicator': 'MACD', 'signal': 'BULLISH', 'strength': 60, 'reason': 'Positive Momentum'})
                bullish_count += 1
            elif macd < macd_signal and macd < 0:
                signals.append({'indicator': 'MACD', 'signal': 'BEARISH', 'strength': 75, 'reason': 'Bearish Crossover'})
                bearish_count += 2
            elif macd < macd_signal:
                signals.append({'indicator': 'MACD', 'signal': 'BEARISH', 'strength': 60, 'reason': 'Negative Momentum'})
                bearish_count += 1
            else:
                signals.append({'indicator': 'MACD', 'signal': 'NEUTRAL', 'strength': 50, 'reason': 'Kein klarer Trend'})
            total_weight += 1
            
            # Moving Averages Signal
            sma_9 = indicators.get('sma_9', 0)
            sma_20 = indicators.get('sma_20', 0)
            current_price = indicators.get('current_price', 0)
            
            if current_price > sma_9 > sma_20:
                signals.append({'indicator': 'SMA', 'signal': 'BULLISH', 'strength': 70, 'reason': 'Preis über beiden SMAs'})
                bullish_count += 2
            elif current_price > sma_9:
                signals.append({'indicator': 'SMA', 'signal': 'BULLISH', 'strength': 55, 'reason': 'Preis über SMA9'})
                bullish_count += 1
            elif current_price < sma_9 < sma_20:
                signals.append({'indicator': 'SMA', 'signal': 'BEARISH', 'strength': 70, 'reason': 'Preis unter beiden SMAs'})
                bearish_count += 2
            elif current_price < sma_9:
                signals.append({'indicator': 'SMA', 'signal': 'BEARISH', 'strength': 55, 'reason': 'Preis unter SMA9'})
                bearish_count += 1
            else:
                signals.append({'indicator': 'SMA', 'signal': 'NEUTRAL', 'strength': 50, 'reason': 'Mixed MA Signal'})
            total_weight += 1
            
            # Bollinger Bands Signal
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            if current_price <= bb_lower:
                signals.append({'indicator': 'BB', 'signal': 'BULLISH', 'strength': 75, 'reason': 'Preis am unteren BB'})
                bullish_count += 2
            elif current_price >= bb_upper:
                signals.append({'indicator': 'BB', 'signal': 'BEARISH', 'strength': 75, 'reason': 'Preis am oberen BB'})
                bearish_count += 2
            else:
                signals.append({'indicator': 'BB', 'signal': 'NEUTRAL', 'strength': 50, 'reason': 'Preis in BB Range'})
            total_weight += 1
            
            # Gesamtsignal berechnen
            if bullish_count > bearish_count + 1:
                overall_signal = 'BULLISH'
                confidence = min(90, 50 + (bullish_count - bearish_count) * 10)
            elif bearish_count > bullish_count + 1:
                overall_signal = 'BEARISH'
                confidence = min(90, 50 + (bearish_count - bullish_count) * 10)
            else:
                overall_signal = 'NEUTRAL'
                confidence = 50
            
            return {
                'signal': overall_signal,
                'confidence': confidence,
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'individual_signals': signals,
                'strength': abs(bullish_count - bearish_count)
            }
            
        except Exception as e:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'error': str(e)}
    
    def _analyze_pattern_signals(self, chart_patterns: Dict) -> Dict:
        """📊 Analysiert Chartmuster Signale"""
        try:
            if not chart_patterns or 'patterns' not in chart_patterns:
                return {'signal': 'NEUTRAL', 'confidence': 50}
            
            patterns = chart_patterns['patterns']
            if not patterns:
                return {'signal': 'NEUTRAL', 'confidence': 50}
            
            # Stärkstes Pattern finden
            strongest_pattern = max(patterns, key=lambda p: p.get('confidence', 0))
            
            pattern_signal = strongest_pattern.get('signal', 'NEUTRAL')
            pattern_confidence = strongest_pattern.get('confidence', 50)
            
            # Pattern Gewichtung (Chartmuster sind wichtig aber nicht alles)
            adjusted_confidence = min(85, pattern_confidence * 0.8)
            
            return {
                'signal': pattern_signal,
                'confidence': adjusted_confidence,
                'pattern_name': strongest_pattern.get('name', 'Unknown'),
                'pattern_type': strongest_pattern.get('pattern', 'unknown'),
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'error': str(e)}
    
    def _calculate_risk_management(self, current_price: float, market_data: List[Dict], liquidation_data: Dict = None) -> Dict:
        """🛡️ Berechnet Risk Management Parameter"""
        try:
            # Volatilität berechnen
            closes = [float(candle[4]) for candle in market_data[-20:]]
            volatility = np.std(closes) / np.mean(closes) * 100
            
            # ATR (Average True Range) approximation
            highs = [float(candle[2]) for candle in market_data[-14:]]
            lows = [float(candle[3]) for candle in market_data[-14:]]
            true_ranges = []
            
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            atr = np.mean(true_ranges)
            atr_percent = (atr / current_price) * 100
            
            # Risk Level bestimmen
            if volatility > 4 or atr_percent > 3:
                risk_level = 'HIGH'
                risk_score = 80
            elif volatility > 2 or atr_percent > 2:
                risk_level = 'MEDIUM'
                risk_score = 60
            else:
                risk_level = 'LOW'
                risk_score = 40
            
            # Liquidation Risk einbeziehen (falls verfügbar)
            liq_risk_factor = 1.0
            if liquidation_data and 'risk_assessment' in liquidation_data:
                liq_risk = liquidation_data['risk_assessment']['overall_risk_level']
                if liq_risk == 'EXTREME':
                    liq_risk_factor = 2.0
                elif liq_risk == 'HIGH':
                    liq_risk_factor = 1.5
                elif liq_risk == 'MEDIUM':
                    liq_risk_factor = 1.2
            
            # Angepasster Risk Score
            adjusted_risk_score = min(100, risk_score * liq_risk_factor)
            
            return {
                'risk_level': risk_level,
                'risk_score': adjusted_risk_score,
                'volatility_percent': volatility,
                'atr_percent': atr_percent,
                'liquidation_risk_factor': liq_risk_factor,
                'recommended_position_size': self._calculate_position_size(adjusted_risk_score),
                'stop_loss_distance': max(atr_percent * 1.5, 2.0)  # Mindestens 2%
            }
            
        except Exception as e:
            return {
                'risk_level': 'MEDIUM',
                'risk_score': 60,
                'error': str(e),
                'recommended_position_size': 1.0,
                'stop_loss_distance': 2.0
            }
    
    def _calculate_position_size(self, risk_score: float) -> float:
        """💰 Berechnet empfohlene Position Size"""
        # Je höher das Risiko, desto kleiner die Position
        if risk_score > 80:
            return 0.5  # 50% der normalen Position
        elif risk_score > 60:
            return 1.0  # Normale Position
        elif risk_score > 40:
            return 1.5  # 150% der normalen Position
        else:
            return 2.0  # 200% der normalen Position (bei niedrigem Risiko)
    
    def _analyze_market_sentiment(self, market_data: List[Dict], indicators: Dict) -> Dict:
        """💭 Analysiert Market Sentiment"""
        try:
            # Volume Analyse
            volumes = [float(candle[5]) for candle in market_data[-10:]]
            avg_volume = np.mean(volumes)
            recent_volume = volumes[-1]
            volume_ratio = recent_volume / avg_volume
            
            # Price Action Analyse
            closes = [float(candle[4]) for candle in market_data[-5:]]
            price_momentum = (closes[-1] - closes[0]) / closes[0] * 100
            
            # 24h Change einbeziehen
            change_24h = indicators.get('change_24h', 0)
            
            # Sentiment Score berechnen
            sentiment_score = 50  # Neutral Start
            
            # Volume Factor
            if volume_ratio > 1.5:
                sentiment_score += 15  # Hohes Volume = mehr Interesse
            elif volume_ratio < 0.7:
                sentiment_score -= 10  # Niedriges Volume = weniger Interesse
            
            # Price Momentum Factor  
            if price_momentum > 2:
                sentiment_score += 20  # Starker Aufwärtstrend
            elif price_momentum > 0.5:
                sentiment_score += 10  # Leichter Aufwärtstrend
            elif price_momentum < -2:
                sentiment_score -= 20  # Starker Abwärtstrend
            elif price_momentum < -0.5:
                sentiment_score -= 10  # Leichter Abwärtstrend
            
            # 24h Change Factor
            if change_24h > 5:
                sentiment_score += 15
            elif change_24h > 2:
                sentiment_score += 10
            elif change_24h < -5:
                sentiment_score -= 15
            elif change_24h < -2:
                sentiment_score -= 10
            
            # Sentiment Level bestimmen
            sentiment_score = max(0, min(100, sentiment_score))
            
            if sentiment_score > 70:
                sentiment = 'BULLISH'
            elif sentiment_score > 55:
                sentiment = 'SLIGHTLY_BULLISH'
            elif sentiment_score > 45:
                sentiment = 'NEUTRAL'
            elif sentiment_score > 30:
                sentiment = 'SLIGHTLY_BEARISH'
            else:
                sentiment = 'BEARISH'
            
            return {
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'volume_ratio': volume_ratio,
                'price_momentum': price_momentum,
                'change_24h': change_24h
            }
            
        except Exception as e:
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_score': 50,
                'error': str(e)
            }
    
    def _calculate_combined_signal(self, tech_signal: Dict, pattern_signal: Dict, 
                                 risk_analysis: Dict, sentiment_signal: Dict) -> Dict:
        """🎯 Berechnet kombiniertes Trading Signal"""
        try:
            # Signal-Werte normalisieren
            signals = {
                'technical': self._normalize_signal(tech_signal['signal'], tech_signal['confidence']),
                'patterns': self._normalize_signal(pattern_signal['signal'], pattern_signal['confidence']),
                'sentiment': self._normalize_signal_sentiment(sentiment_signal['sentiment'], sentiment_signal['sentiment_score'])
            }
            
            # Risk-adjustierte Gewichtung
            risk_factor = self._get_risk_factor(risk_analysis['risk_level'])
            
            # Gewichtete Summe berechnen
            weighted_score = 0
            total_weight = 0
            
            for signal_type, score in signals.items():
                weight = self.signal_weights.get(signal_type, 0.25)
                weighted_score += score * weight
                total_weight += weight
            
            # Risk Management Penalty/Bonus
            if risk_analysis['risk_level'] == 'HIGH':
                weighted_score *= 0.8  # 20% Penalty bei hohem Risiko
            elif risk_analysis['risk_level'] == 'LOW':
                weighted_score *= 1.1  # 10% Bonus bei niedrigem Risiko
            
            # Final Score normalisieren
            final_score = max(-100, min(100, weighted_score))
            
            # Signal bestimmen
            if final_score > 20:
                final_signal = 'BULLISH'
                confidence = min(95, 50 + abs(final_score) * 0.5)
            elif final_score > 5:
                final_signal = 'SLIGHTLY_BULLISH'
                confidence = min(70, 50 + abs(final_score) * 0.8)
            elif final_score > -5:
                final_signal = 'NEUTRAL'
                confidence = 50
            elif final_score > -20:
                final_signal = 'SLIGHTLY_BEARISH'
                confidence = min(70, 50 + abs(final_score) * 0.8)
            else:
                final_signal = 'BEARISH'
                confidence = min(95, 50 + abs(final_score) * 0.5)
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'score': final_score,
                'component_scores': signals,
                'risk_adjusted': True,
                'risk_factor': risk_factor
            }
            
        except Exception as e:
            return {
                'signal': 'NEUTRAL',
                'confidence': 50,
                'error': str(e)
            }
    
    def _normalize_signal(self, signal: str, confidence: float) -> float:
        """📊 Normalisiert Signal zu Score (-100 bis +100)"""
        base_score = 0
        
        if signal == 'BULLISH':
            base_score = 50
        elif signal == 'SLIGHTLY_BULLISH':
            base_score = 25
        elif signal == 'NEUTRAL':
            base_score = 0
        elif signal == 'SLIGHTLY_BEARISH':
            base_score = -25
        elif signal == 'BEARISH':
            base_score = -50
        
        # Confidence Factor anwenden
        confidence_factor = confidence / 100
        return base_score * confidence_factor
    
    def _normalize_signal_sentiment(self, sentiment: str, score: float) -> float:
        """💭 Normalisiert Sentiment zu Score (-100 bis +100)"""
        # Score bereits zwischen 0-100, zu -100 bis +100 konvertieren
        return (score - 50) * 2
    
    def _get_risk_factor(self, risk_level: str) -> float:
        """⚠️ Gibt Risk Factor zurück"""
        factors = {
            'LOW': 0.8,
            'MEDIUM': 1.0,
            'HIGH': 1.3
        }
        return factors.get(risk_level, 1.0)
    
    def _generate_trading_setup(self, current_price: float, combined_signal: Dict, 
                              risk_analysis: Dict, market_data: List[Dict]) -> Dict:
        """🎯 Generiert komplettes Trading Setup"""
        try:
            signal = combined_signal['signal']
            confidence = combined_signal['confidence']
            
            if signal in ['NEUTRAL']:
                return {
                    'action': 'HOLD',
                    'direction': 'NONE',
                    'confidence': confidence,
                    'reason': 'Keine klare Richtung erkennbar'
                }
            
            # Trading Direction bestimmen
            direction = 'LONG' if 'BULLISH' in signal else 'SHORT'
            
            # Stop Loss Distance
            stop_distance = risk_analysis.get('stop_loss_distance', 2.0)
            
            # Entry, Stop Loss, Take Profit berechnen
            if direction == 'LONG':
                entry_price = current_price
                stop_loss = entry_price * (1 - stop_distance / 100)
                
                # Risk/Reward basierend auf Confidence
                if confidence > 80:
                    risk_reward = self.risk_reward_ratios['aggressive']
                elif confidence > 60:
                    risk_reward = self.risk_reward_ratios['moderate']
                else:
                    risk_reward = self.risk_reward_ratios['conservative']
                
                take_profit = entry_price + (entry_price - stop_loss) * risk_reward
                
            else:  # SHORT
                entry_price = current_price
                stop_loss = entry_price * (1 + stop_distance / 100)
                
                if confidence > 80:
                    risk_reward = self.risk_reward_ratios['aggressive']
                elif confidence > 60:
                    risk_reward = self.risk_reward_ratios['moderate']
                else:
                    risk_reward = self.risk_reward_ratios['conservative']
                
                take_profit = entry_price - (stop_loss - entry_price) * risk_reward
            
            # Position Size
            position_size = risk_analysis.get('recommended_position_size', 1.0)
            
            # Risk Percentage (2% Standard, angepasst an Risiko)
            risk_percentage = 2.0
            if risk_analysis['risk_level'] == 'HIGH':
                risk_percentage = 1.0
            elif risk_analysis['risk_level'] == 'LOW':
                risk_percentage = 3.0
            
            return {
                'action': 'ENTER',
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'risk_percentage': risk_percentage,
                'risk_reward_ratio': risk_reward,
                'confidence': confidence,
                'stop_distance_percent': stop_distance,
                'expected_profit_percent': ((take_profit - entry_price) / entry_price * 100) if direction == 'LONG' else ((entry_price - take_profit) / entry_price * 100)
            }
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'error': str(e)
            }
    
    def _generate_recommendation(self, combined_signal: Dict, risk_analysis: Dict) -> Dict:
        """💡 Generiert Trading Empfehlung"""
        try:
            signal = combined_signal['signal']
            confidence = combined_signal['confidence']
            risk_level = risk_analysis['risk_level']
            
            # Base Recommendation
            if confidence > 75 and risk_level != 'HIGH':
                strength = 'STRONG'
            elif confidence > 60:
                strength = 'MODERATE'
            elif confidence > 45:
                strength = 'WEAK'
            else:
                strength = 'NO'
            
            # Risk Warnings
            warnings = []
            if risk_level == 'HIGH':
                warnings.append('🚨 Hohes Marktrisiko - reduzierte Position empfohlen')
            if confidence < 60:
                warnings.append('⚠️ Niedrige Confidence - vorsichtig agieren')
            
            # Action Recommendation
            if signal == 'BULLISH' and strength in ['STRONG', 'MODERATE']:
                action = f"{strength} BUY"
                emoji = '🚀'
            elif signal == 'BEARISH' and strength in ['STRONG', 'MODERATE']:
                action = f"{strength} SELL"
                emoji = '📉'
            else:
                action = 'HOLD'
                emoji = '⏸️'
            
            return {
                'action': action,
                'emoji': emoji,
                'strength': strength,
                'confidence': confidence,
                'risk_level': risk_level,
                'warnings': warnings,
                'summary': f"{emoji} {action} - {confidence}% confidence | Risk: {risk_level}"
            }
            
        except Exception as e:
            return {
                'action': 'HOLD',
                'emoji': '❓',
                'error': str(e)
            }

# Global instance
trading_analyzer = TradingSetupAnalyzer()
