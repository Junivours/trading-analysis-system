"""
💀 LIQUIDATION MAPS - Separate Datei  
Berechnet Liquidation Heatmaps und Risk Management
"""

import numpy as np
from typing import Dict, List, Tuple

class LiquidationMapAnalyzer:
    """💀 Professionelle Liquidation Heatmap Analyse"""
    
    def __init__(self):
        self.leverage_levels = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]
        self.risk_thresholds = {
            'LOW': 15,      # Unter 15% Entfernung = geringes Risiko
            'MEDIUM': 8,    # 8-15% = mittleres Risiko  
            'HIGH': 5,      # 5-8% = hohes Risiko
            'EXTREME': 2    # Unter 5% = extremes Risiko
        }
    
    def analyze_liquidation_risk(self, current_price: float, market_data: List[Dict]) -> Dict:
        """💀 Komplette Liquidation Risk Analyse"""
        try:
            if not market_data or len(market_data) < 20:
                return {'error': 'Nicht genügend Marktdaten für Liquidation-Analyse'}
            
            # Support/Resistance Levels berechnen
            support_resistance = self._calculate_support_resistance(market_data, current_price)
            
            # Liquidation Levels für alle Hebel berechnen
            liq_levels = self._calculate_liquidation_levels(current_price)
            
            # Volatilität und Trend analysieren
            volatility_analysis = self._analyze_volatility_trend(market_data)
            
            # Risk Assessment
            risk_assessment = self._assess_liquidation_risk(current_price, liq_levels, volatility_analysis)
            
            return {
                'success': True,
                'current_price': current_price,
                'liquidation_levels': liq_levels,
                'support_resistance': support_resistance,
                'volatility_analysis': volatility_analysis,
                'risk_assessment': risk_assessment,
                'recommended_leverage': self._recommend_leverage(risk_assessment),
                'danger_zones': self._identify_danger_zones(current_price, liq_levels)
            }
            
        except Exception as e:
            return {'error': f'Liquidation-Analyse Fehler: {str(e)}'}
    
    def _calculate_liquidation_levels(self, current_price: float) -> Dict:
        """💀 Berechnet Liquidation Levels für alle Hebel"""
        liquidation_data = {
            'long_liquidations': {},
            'short_liquidations': {},
            'all_levels': []
        }
        
        for leverage in self.leverage_levels:
            # Long Liquidation = Entry Price * (1 - 1/Leverage)
            long_liq = current_price * (1 - 1/leverage)
            
            # Short Liquidation = Entry Price * (1 + 1/Leverage)  
            short_liq = current_price * (1 + 1/leverage)
            
            # Entfernung in Prozent
            long_distance = ((current_price - long_liq) / current_price) * 100
            short_distance = ((short_liq - current_price) / current_price) * 100
            
            liquidation_data['long_liquidations'][f'{leverage}x'] = {
                'price': long_liq,
                'distance_percent': long_distance
            }
            
            liquidation_data['short_liquidations'][f'{leverage}x'] = {
                'price': short_liq,
                'distance_percent': short_distance
            }
            
            liquidation_data['all_levels'].append({
                'level': f'{leverage}x',
                'long_liquidation': long_liq,
                'short_liquidation': short_liq,
                'distance_long': long_distance,
                'distance_short': short_distance
            })
        
        return liquidation_data
    
    def _calculate_support_resistance(self, market_data: List[Dict], current_price: float) -> Dict:
        """📊 Berechnet Support und Resistance Levels"""
        try:
            # Extrahiere Preisdaten
            highs = [float(candle[2]) for candle in market_data[-50:]]
            lows = [float(candle[3]) for candle in market_data[-50:]]
            closes = [float(candle[4]) for candle in market_data[-50:]]
            
            # Support Levels (wichtige Tiefs)
            support_levels = []
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
            
            # Resistance Levels (wichtige Hochs)
            resistance_levels = []
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
            
            # Nächste Support/Resistance finden
            support_levels = [s for s in support_levels if s < current_price]
            resistance_levels = [r for r in resistance_levels if r > current_price]
            
            nearest_support = max(support_levels) if support_levels else min(lows)
            nearest_resistance = min(resistance_levels) if resistance_levels else max(highs)
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'all_supports': sorted(support_levels, reverse=True)[:3],
                'all_resistances': sorted(resistance_levels)[:3],
                'support_distance': ((current_price - nearest_support) / current_price) * 100,
                'resistance_distance': ((nearest_resistance - current_price) / current_price) * 100
            }
            
        except Exception as e:
            return {
                'nearest_support': current_price * 0.95,
                'nearest_resistance': current_price * 1.05,
                'error': str(e)
            }
    
    def _analyze_volatility_trend(self, market_data: List[Dict]) -> Dict:
        """📈 Analysiert Volatilität und Trend"""
        try:
            closes = [float(candle[4]) for candle in market_data[-20:]]
            
            # Volatilität (Standard Deviation)
            volatility = np.std(closes) / np.mean(closes) * 100
            
            # Trend (Lineare Regression)
            x = list(range(len(closes)))
            trend_slope = np.polyfit(x, closes, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(closes) * 100
            
            # Trend Direction
            if trend_slope > 0:
                trend_direction = 'bullish'
            elif trend_slope < 0:
                trend_direction = 'bearish'
            else:
                trend_direction = 'sideways'
            
            # Trend Classification
            if trend_strength > 2:
                trend_classification = 'strong_' + trend_direction
            elif trend_strength > 0.5:
                trend_classification = 'moderate_' + trend_direction
            else:
                trend_classification = 'weak_' + trend_direction
            
            return {
                'volatility_percent': volatility,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'trend_classification': trend_classification,
                'volatility_level': self._classify_volatility(volatility)
            }
            
        except Exception as e:
            return {
                'volatility_percent': 1.0,
                'trend_direction': 'sideways',
                'error': str(e)
            }
    
    def _classify_volatility(self, volatility: float) -> str:
        """📊 Klassifiziert Volatilität"""
        if volatility > 3:
            return 'EXTREME'
        elif volatility > 2:
            return 'HIGH'
        elif volatility > 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_liquidation_risk(self, current_price: float, liq_levels: Dict, volatility: Dict) -> Dict:
        """⚠️ Bewertet das Liquidation Risiko"""
        
        # Gefährlichste Liquidation Levels finden
        dangerous_long = []
        dangerous_short = []
        
        for level_data in liq_levels['all_levels']:
            # Long Positions - Risiko bei fallenden Preisen
            if level_data['distance_long'] < self.risk_thresholds['HIGH']:
                dangerous_long.append({
                    'leverage': level_data['level'],
                    'liquidation_price': level_data['long_liquidation'],
                    'distance': level_data['distance_long']
                })
            
            # Short Positions - Risiko bei steigenden Preisen  
            if level_data['distance_short'] < self.risk_thresholds['HIGH']:
                dangerous_short.append({
                    'leverage': level_data['level'],
                    'liquidation_price': level_data['short_liquidation'],
                    'distance': level_data['distance_short']
                })
        
        # Overall Risk Level
        min_long_distance = min([l['distance_long'] for l in liq_levels['all_levels'][:5]])  # Top 5 Hebel
        min_short_distance = min([l['distance_short'] for l in liq_levels['all_levels'][:5]])
        
        overall_risk = self._calculate_overall_risk(min_long_distance, min_short_distance, volatility)
        
        return {
            'overall_risk_level': overall_risk,
            'dangerous_long_levels': dangerous_long,
            'dangerous_short_levels': dangerous_short,
            'min_safe_distance_long': min_long_distance,
            'min_safe_distance_short': min_short_distance,
            'volatility_factor': volatility.get('volatility_percent', 1.0)
        }
    
    def _calculate_overall_risk(self, min_long_dist: float, min_short_dist: float, volatility: Dict) -> str:
        """🎯 Berechnet das Gesamt-Risiko"""
        
        min_distance = min(min_long_dist, min_short_dist)
        vol_level = volatility.get('volatility_percent', 1.0)
        
        # Risiko steigt mit hoher Volatilität
        adjusted_distance = min_distance - (vol_level * 2)
        
        if adjusted_distance < self.risk_thresholds['EXTREME']:
            return 'EXTREME'
        elif adjusted_distance < self.risk_thresholds['HIGH']:
            return 'HIGH'
        elif adjusted_distance < self.risk_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _recommend_leverage(self, risk_assessment: Dict) -> Dict:
        """🎯 Empfiehlt sicheren Hebel"""
        
        risk_level = risk_assessment['overall_risk_level']
        volatility = risk_assessment['volatility_factor']
        
        if risk_level == 'EXTREME' or volatility > 3:
            recommended = '2x'
            max_safe = '3x'
        elif risk_level == 'HIGH' or volatility > 2:
            recommended = '3x'
            max_safe = '5x'
        elif risk_level == 'MEDIUM' or volatility > 1.5:
            recommended = '5x'
            max_safe = '10x'
        else:
            recommended = '10x'
            max_safe = '20x'
        
        return {
            'recommended_leverage': recommended,
            'max_safe_leverage': max_safe,
            'avoid_above': self._get_danger_threshold(risk_level),
            'reasoning': f'Risk Level: {risk_level}, Volatility: {volatility:.1f}%'
        }
    
    def _get_danger_threshold(self, risk_level: str) -> str:
        """⚠️ Definiert gefährliche Hebel"""
        thresholds = {
            'EXTREME': '5x',
            'HIGH': '10x', 
            'MEDIUM': '25x',
            'LOW': '50x'
        }
        return thresholds.get(risk_level, '10x')
    
    def _identify_danger_zones(self, current_price: float, liq_levels: Dict) -> Dict:
        """🚨 Identifiziert Gefahrenzonen"""
        
        danger_zones = {
            'immediate_danger': [],  # Unter 3%
            'high_risk': [],         # 3-8%
            'medium_risk': []        # 8-15%
        }
        
        for level in liq_levels['all_levels']:
            long_dist = level['distance_long']
            short_dist = level['distance_short']
            
            # Long Position Gefahrenzonen
            if long_dist < 3:
                danger_zones['immediate_danger'].append({
                    'type': 'LONG',
                    'leverage': level['level'],
                    'liquidation_price': level['long_liquidation'],
                    'distance': long_dist
                })
            elif long_dist < 8:
                danger_zones['high_risk'].append({
                    'type': 'LONG',
                    'leverage': level['level'],
                    'liquidation_price': level['long_liquidation'],
                    'distance': long_dist
                })
            elif long_dist < 15:
                danger_zones['medium_risk'].append({
                    'type': 'LONG',
                    'leverage': level['level'],
                    'liquidation_price': level['long_liquidation'],
                    'distance': long_dist
                })
            
            # Short Position Gefahrenzonen
            if short_dist < 3:
                danger_zones['immediate_danger'].append({
                    'type': 'SHORT',
                    'leverage': level['level'],
                    'liquidation_price': level['short_liquidation'],
                    'distance': short_dist
                })
            elif short_dist < 8:
                danger_zones['high_risk'].append({
                    'type': 'SHORT',
                    'leverage': level['level'],
                    'liquidation_price': level['short_liquidation'],
                    'distance': short_dist
                })
            elif short_dist < 15:
                danger_zones['medium_risk'].append({
                    'type': 'SHORT',
                    'leverage': level['level'],
                    'liquidation_price': level['short_liquidation'],
                    'distance': short_dist
                })
        
        return danger_zones

# Global instance
liquidation_analyzer = LiquidationMapAnalyzer()
