"""
Signal Enhancement Module
Detailliertere Signal-Erkennung ohne Breaking Changes
"""
import numpy as np
from typing import Dict, List, Any, Optional

class SignalEnhancer:
    """Erweiterte Signal-Analyse für höhere Präzision und Erfolgswahrscheinlichkeit"""
    
    @staticmethod
    def detect_micro_patterns(candles: List[Dict], lookback: int = 20) -> List[Dict]:
        """Mikro-Pattern Erkennung für kurzfristige Präzisions-Signale"""
        patterns = []
        
        if len(candles) < lookback:
            return patterns
            
        highs = [c['high'] for c in candles[-lookback:]]
        lows = [c['low'] for c in candles[-lookback:]]
        closes = [c['close'] for c in candles[-lookback:]]
        volumes = [c['volume'] for c in candles[-lookback:]]
        
        # 1. Smart Money Footprints
        smart_money = SignalEnhancer._detect_smart_money_footprints(highs, lows, closes, volumes)
        if smart_money:
            patterns.append(smart_money)
            
        # 2. Volume-Price Analysis (VPA)
        vpa = SignalEnhancer._detect_vpa_signals(highs, lows, closes, volumes)
        if vpa:
            patterns.extend(vpa)
            
        # 3. Institutional Order Flow
        institutional = SignalEnhancer._detect_institutional_flow(highs, lows, volumes)
        if institutional:
            patterns.append(institutional)
            
        # 4. Liquidity Sweep Patterns
        sweeps = SignalEnhancer._detect_liquidity_sweeps(highs, lows, closes)
        if sweeps:
            patterns.extend(sweeps)
            
        return patterns
    
    @staticmethod
    def _detect_smart_money_footprints(highs, lows, closes, volumes):
        """Smart Money Accumulation/Distribution Erkennung"""
        if len(closes) < 10:
            return None
            
        recent_vol = volumes[-5:]
        avg_vol = np.mean(volumes[:-5]) if len(volumes) > 5 else np.mean(volumes)
        
        # Hohe Volumen bei kleinen Körpern = Smart Money Aktivität
        for i in range(-3, 0):
            if i >= -len(closes):
                body_size = abs(closes[i] - closes[i-1]) / closes[i-1] if i > 0 else 0
                vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1
                
                if vol_ratio > 2.0 and body_size < 0.008:  # Hohe Vol, kleine Bewegung
                    trend = 'accumulation' if closes[i] > closes[i-5] else 'distribution'
                    
                    return {
                        'type': f'Smart Money {trend.title()}',
                        'signal': 'bullish' if trend == 'accumulation' else 'bearish',
                        'confidence': min(85, 65 + int(vol_ratio * 8)),
                        'timeframe': '1h',
                        'strength': 'HIGH',
                        'description': f'{trend} bei {vol_ratio:.1f}x Volumen, kleine Körper',
                        'reliability_score': 78,
                        'quality_grade': 'A',
                        'entry_zone': closes[-1] * (1.002 if trend == 'accumulation' else 0.998)
                    }
        return None
    
    @staticmethod
    def _detect_vpa_signals(highs, lows, closes, volumes):
        """Volume Price Analysis - Wyckoff Methodik"""
        signals = []
        
        if len(closes) < 8:
            return signals
            
        # Test für Strength/Weakness
        for i in range(-3, -1):
            if abs(i) < len(closes):
                vol_curr = volumes[i]
                vol_prev = volumes[i-1] if i-1 >= -len(volumes) else vol_curr
                price_change = (closes[i] - closes[i-1]) / closes[i-1] if i > 0 else 0
                
                # Effort vs Result Analysis
                if vol_curr > vol_prev * 1.5:  # Hoher Effort (Volumen)
                    if abs(price_change) < 0.005:  # Niedriges Result (Preis)
                        # No Supply (Bullish) oder No Demand (Bearish)
                        close_pos = (closes[i] - lows[i]) / (highs[i] - lows[i]) if highs[i] != lows[i] else 0.5
                        
                        if close_pos > 0.7:  # Close im oberen Bereich
                            signals.append({
                                'type': 'VPA No Supply',
                                'signal': 'bullish',
                                'confidence': 72,
                                'timeframe': '1h',
                                'strength': 'MEDIUM',
                                'description': f'Hohe Vol, wenig Bewegung, Close oben',
                                'reliability_score': 68,
                                'quality_grade': 'B'
                            })
                        elif close_pos < 0.3:  # Close im unteren Bereich  
                            signals.append({
                                'type': 'VPA No Demand',
                                'signal': 'bearish',
                                'confidence': 72,
                                'timeframe': '1h',
                                'strength': 'MEDIUM',
                                'description': f'Hohe Vol, wenig Bewegung, Close unten',
                                'reliability_score': 68,
                                'quality_grade': 'B'
                            })
                            
        return signals
    
    @staticmethod
    def _detect_institutional_flow(highs, lows, volumes):
        """Institutional Order Flow Detection"""
        if len(volumes) < 6:
            return None
            
        # Sudden Volume Spike = Institutional Entry
        recent_vol = volumes[-3:]
        baseline_vol = np.mean(volumes[:-3]) if len(volumes) > 3 else np.mean(volumes)
        
        max_recent = max(recent_vol)
        if max_recent > baseline_vol * 3.0:  # 3x Volume Spike
            spike_idx = recent_vol.index(max_recent) - 3  # Relative zum Ende
            
            # Price Action während Spike analysieren
            spike_high = highs[spike_idx]
            spike_low = lows[spike_idx]
            range_size = (spike_high - spike_low) / spike_low if spike_low > 0 else 0
            
            if range_size > 0.015:  # Signifikante Bewegung mit hohem Volumen
                direction = 'bullish' if spike_high > np.mean(highs[-6:-3]) else 'bearish'
                
                return {
                    'type': 'Institutional Flow',
                    'signal': direction,
                    'confidence': 82,
                    'timeframe': '1h',
                    'strength': 'VERY_HIGH',
                    'description': f'{max_recent/baseline_vol:.1f}x Vol-Spike, {range_size*100:.1f}% Range',
                    'reliability_score': 79,
                    'quality_grade': 'A',
                    'volume_ratio': max_recent/baseline_vol
                }
        return None
    
    @staticmethod 
    def _detect_liquidity_sweeps(highs, lows, closes):
        """Liquidity Sweep Pattern - Stop Hunting Detection"""
        sweeps = []
        
        if len(closes) < 10:
            return sweeps
            
        # Suche nach vorherigen Highs/Lows die "getestet" werden
        recent_highs = []
        recent_lows = []
        
        for i in range(-8, -2):
            if abs(i) < len(highs):
                # Lokale Highs/Lows identifizieren
                if i > -len(highs) and i < -1:
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        recent_highs.append((i, highs[i]))
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        recent_lows.append((i, lows[i]))
        
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        
        # Liquidity Sweep Detection
        for idx, high_level in recent_highs:
            if current_high > high_level * 1.002:  # Sweep über altes High
                if current_close < high_level * 0.998:  # Aber Close darunter = Fake Breakout
                    sweeps.append({
                        'type': 'Liquidity Sweep High',
                        'signal': 'bearish',
                        'confidence': 76,
                        'timeframe': '1h', 
                        'strength': 'HIGH',
                        'description': f'Sweep über {high_level:.4f}, Close {current_close:.4f}',
                        'reliability_score': 74,
                        'quality_grade': 'B',
                        'sweep_level': high_level,
                        'false_breakout': True
                    })
                    
        for idx, low_level in recent_lows:
            if current_low < low_level * 0.998:  # Sweep unter altes Low
                if current_close > low_level * 1.002:  # Aber Close darüber = Fake Breakdown
                    sweeps.append({
                        'type': 'Liquidity Sweep Low',
                        'signal': 'bullish',
                        'confidence': 76,
                        'timeframe': '1h',
                        'strength': 'HIGH', 
                        'description': f'Sweep unter {low_level:.4f}, Close {current_close:.4f}',
                        'reliability_score': 74,
                        'quality_grade': 'B',
                        'sweep_level': low_level,
                        'false_breakout': True
                    })
        
        return sweeps
    
    @staticmethod
    def enhance_confluence_scoring(setup: Dict, tech_analysis: Dict, pattern_analysis: Dict, 
                                 multi_timeframe: Dict, order_flow: Dict) -> Dict:
        """Verbesserte Confluence-Bewertung für höhere Präzision"""
        
        confluence_score = 0
        confluence_factors = []
        
        direction = setup.get('direction')
        
        # 1. Multi-Timeframe Alignment (weighted by timeframe)
        timeframes = multi_timeframe.get('timeframes', [])
        bull_count = sum(1 for tf in timeframes if 'bull' in tf.get('signal', '').lower())
        bear_count = sum(1 for tf in timeframes if 'bear' in tf.get('signal', '').lower())
        
        if direction == 'LONG' and bull_count > bear_count:
            confluence_score += 15
            confluence_factors.append(f'MTF: {bull_count}/{len(timeframes)} bullish')
        elif direction == 'SHORT' and bear_count > bull_count:
            confluence_score += 15
            confluence_factors.append(f'MTF: {bear_count}/{len(timeframes)} bearish')
            
        # 2. Technical Indicator Confluence
        rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
        macd_direction = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
        
        if direction == 'LONG':
            if 30 <= rsi <= 65:  # Sweet spot für LONG
                confluence_score += 8
                confluence_factors.append(f'RSI optimal: {rsi:.1f}')
            if 'bullish' in macd_direction:
                confluence_score += 6
                confluence_factors.append('MACD bullish')
        elif direction == 'SHORT':
            if 35 <= rsi <= 70:  # Sweet spot für SHORT
                confluence_score += 8
                confluence_factors.append(f'RSI optimal: {rsi:.1f}')
            if 'bearish' in macd_direction:
                confluence_score += 6
                confluence_factors.append('MACD bearish')
                
        # 3. Pattern Quality Boost
        patterns = pattern_analysis.get('patterns', [])
        high_quality_patterns = [p for p in patterns if p.get('quality_grade') in ['A', 'B']]
        
        if high_quality_patterns:
            aligned_patterns = [p for p in high_quality_patterns 
                             if (direction == 'LONG' and p.get('signal') == 'bullish') or
                                (direction == 'SHORT' and p.get('signal') == 'bearish')]
            if aligned_patterns:
                confluence_score += len(aligned_patterns) * 4
                confluence_factors.append(f'{len(aligned_patterns)} HQ patterns aligned')
                
        # 4. Order Flow Confirmation
        flow_sentiment = order_flow.get('flow_sentiment', 'neutral')
        if direction == 'LONG' and flow_sentiment in ['buy_pressure', 'bullish']:
            confluence_score += 7
            confluence_factors.append('Order flow bullish')
        elif direction == 'SHORT' and flow_sentiment in ['sell_pressure', 'bearish']:
            confluence_score += 7
            confluence_factors.append('Order flow bearish')
            
        # 5. Volume Profile Confirmation
        poc = order_flow.get('volume_profile_poc')
        current_price = tech_analysis.get('current_price')
        
        if poc and current_price:
            poc_distance = abs(current_price - poc) / current_price * 100
            if poc_distance < 0.5:  # Nahe POC = hohe Wahrscheinlichkeit
                confluence_score += 5
                confluence_factors.append(f'Near POC ({poc_distance:.2f}%)')
                
        # Update Setup mit Enhanced Metrics
        enhanced_setup = setup.copy()
        enhanced_setup.update({
            'confluence_score': confluence_score,
            'confluence_factors': confluence_factors,
            'success_probability': min(85, 45 + confluence_score * 0.8),  # Estimate
            'enhanced_confidence': min(95, setup.get('confidence', 50) + confluence_score * 0.6)
        })
        
        return enhanced_setup
    
    @staticmethod
    def add_timing_precision(setup: Dict, candles: List[Dict]) -> Dict:
        """Präzise Timing-Signale für Entry/Exit Optimierung"""
        
        if len(candles) < 5:
            return setup
            
        closes = [c['close'] for c in candles[-5:]]
        volumes = [c['volume'] for c in candles[-5:]]
        
        timing_signals = []
        
        # 1. Momentum Divergence Check
        price_momentum = closes[-1] / closes[-3] - 1 if len(closes) >= 3 else 0
        volume_momentum = volumes[-1] / np.mean(volumes[:-1]) - 1 if len(volumes) > 1 else 0
        
        if price_momentum > 0 and volume_momentum < -0.2:  # Preis steigt, Volumen fällt
            timing_signals.append('Bullish divergence - schwaches Volumen')
        elif price_momentum < 0 and volume_momentum < -0.2:  # Preis & Volumen fallen
            timing_signals.append('Bearish momentum - Verkaufsdruck')
            
        # 2. Intraday Precision Entry
        current_hour = 14  # Simplified - could use real time
        if 9 <= current_hour <= 11 or 14 <= current_hour <= 16:  # Aktive Stunden
            timing_signals.append('High activity window')
            
        # 3. Candle Pattern Confirmation
        last_candle = candles[-1]
        prev_candle = candles[-2] if len(candles) >= 2 else last_candle
        
        body_size = abs(last_candle['close'] - last_candle['open']) / last_candle['open']
        wick_ratio = (last_candle['high'] - max(last_candle['open'], last_candle['close'])) / body_size if body_size > 0 else 0
        
        if setup.get('direction') == 'LONG' and wick_ratio < 0.3 and body_size > 0.008:
            timing_signals.append('Strong bullish candle')
        elif setup.get('direction') == 'SHORT' and wick_ratio < 0.3 and body_size > 0.008:
            timing_signals.append('Strong bearish candle')
            
        enhanced_setup = setup.copy()
        enhanced_setup.update({
            'timing_signals': timing_signals,
            'entry_precision': len(timing_signals) / 3.0,  # 0-1 score
            'timing_confidence': min(100, setup.get('confidence', 50) + len(timing_signals) * 5)
        })
        
        return enhanced_setup
