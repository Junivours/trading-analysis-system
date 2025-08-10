"""
📊 CHARTMUSTER ANALYSE - Separate Datei
Erkennt und analysiert alle Trading-Chartmuster
"""

import numpy as np
from typing import Dict, List, Tuple

class ChartPatternAnalyzer:
    """📊 Professionelle Chartmuster-Erkennung"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': 'Kopf-Schulter Formation',
            'double_top': 'Doppeltop',
            'double_bottom': 'Doppelboden', 
            'triangle_ascending': 'Aufsteigendes Dreieck',
            'triangle_descending': 'Absteigendes Dreieck',
            'triangle_symmetrical': 'Symmetrisches Dreieck',
            'flag_bull': 'Bullische Flagge',
            'flag_bear': 'Bärische Flagge',
            'wedge_rising': 'Steigender Keil',
            'wedge_falling': 'Fallender Keil',
            'cup_and_handle': 'Tasse mit Henkel',
            'inverse_head_shoulders': 'Umgekehrte Kopf-Schulter'
        }
    
    def analyze_patterns(self, market_data: List[Dict]) -> Dict:
        """🔍 Analysiert alle Chartmuster in den Marktdaten"""
        try:
            if len(market_data) < 50:
                return {'error': 'Nicht genügend Daten für Chartmuster-Analyse'}
            
            # Preisdaten extrahieren
            highs = [float(candle[2]) for candle in market_data]
            lows = [float(candle[3]) for candle in market_data]
            closes = [float(candle[4]) for candle in market_data]
            
            detected_patterns = []
            
            # 1. Kopf-Schulter Formation prüfen
            head_shoulders = self._detect_head_and_shoulders(highs, lows)
            if head_shoulders:
                detected_patterns.append(head_shoulders)
            
            # 2. Doppeltop/Doppelboden prüfen
            double_patterns = self._detect_double_patterns(highs, lows)
            detected_patterns.extend(double_patterns)
            
            # 3. Dreiecksformationen prüfen
            triangles = self._detect_triangles(highs, lows, closes)
            detected_patterns.extend(triangles)
            
            # 4. Flaggen und Keile prüfen
            flags_wedges = self._detect_flags_and_wedges(highs, lows, closes)
            detected_patterns.extend(flags_wedges)
            
            # 5. Tasse mit Henkel prüfen
            cup_handle = self._detect_cup_and_handle(lows, closes)
            if cup_handle:
                detected_patterns.append(cup_handle)
            
            return {
                'success': True,
                'patterns_detected': len(detected_patterns),
                'patterns': detected_patterns,
                'strongest_signal': self._get_strongest_pattern(detected_patterns)
            }
            
        except Exception as e:
            return {'error': f'Chartmuster-Analyse Fehler: {str(e)}'}
    
    def _detect_head_and_shoulders(self, highs: List[float], lows: List[float]) -> Dict:
        """👤 Erkennt Kopf-Schulter Formationen"""
        try:
            # Vereinfachte Kopf-Schulter Erkennung
            recent_highs = highs[-30:]  # Letzten 30 Kerzen
            
            if len(recent_highs) < 15:
                return None
            
            # Finde lokale Maxima
            peaks = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i-1] and 
                    recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and 
                    recent_highs[i] > recent_highs[i+2]):
                    peaks.append((i, recent_highs[i]))
            
            # Mindestens 3 Peaks für Kopf-Schulter
            if len(peaks) >= 3:
                # Sortiere nach Höhe
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                
                # Höchster Peak = Kopf, zwei niedrigere = Schultern
                head = peaks_sorted[0]
                shoulders = peaks_sorted[1:3]
                
                # Prüfe ob Formation stimmt
                left_shoulder = min(shoulders, key=lambda x: x[0])
                right_shoulder = max(shoulders, key=lambda x: x[0])
                
                if left_shoulder[0] < head[0] < right_shoulder[0]:
                    # Gültige Kopf-Schulter Formation
                    confidence = 75 + (head[1] - max(left_shoulder[1], right_shoulder[1])) / head[1] * 25
                    
                    return {
                        'pattern': 'head_and_shoulders',
                        'name': 'Kopf-Schulter Formation',
                        'confidence': min(95, max(60, confidence)),
                        'signal': 'BEARISH',
                        'target_price': head[1] * 0.95,  # 5% unter Kopf
                        'stop_loss': head[1] * 1.02,
                        'description': 'Bärische Umkehrformation - Verkaufssignal'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _detect_double_patterns(self, highs: List[float], lows: List[float]) -> List[Dict]:
        """📊 Erkennt Doppeltop und Doppelboden"""
        patterns = []
        
        try:
            # Doppeltop Erkennung
            recent_highs = highs[-20:]
            if len(recent_highs) >= 10:
                max_high = max(recent_highs)
                high_indices = [i for i, h in enumerate(recent_highs) if h > max_high * 0.98]
                
                if len(high_indices) >= 2:
                    first_high = high_indices[0]
                    last_high = high_indices[-1]
                    
                    if last_high - first_high > 5:  # Mindestabstand
                        patterns.append({
                            'pattern': 'double_top',
                            'name': 'Doppeltop',
                            'confidence': 70,
                            'signal': 'BEARISH',
                            'target_price': max_high * 0.96,
                            'stop_loss': max_high * 1.01,
                            'description': 'Bärisches Umkehrmuster - Verkaufssignal'
                        })
            
            # Doppelboden Erkennung
            recent_lows = lows[-20:]
            if len(recent_lows) >= 10:
                min_low = min(recent_lows)
                low_indices = [i for i, l in enumerate(recent_lows) if l < min_low * 1.02]
                
                if len(low_indices) >= 2:
                    first_low = low_indices[0]
                    last_low = low_indices[-1]
                    
                    if last_low - first_low > 5:  # Mindestabstand
                        patterns.append({
                            'pattern': 'double_bottom',
                            'name': 'Doppelboden',
                            'confidence': 70,
                            'signal': 'BULLISH',
                            'target_price': min_low * 1.04,
                            'stop_loss': min_low * 0.99,
                            'description': 'Bullisches Umkehrmuster - Kaufsignal'
                        })
            
        except Exception:
            pass
        
        return patterns
    
    def _detect_triangles(self, highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
        """📐 Erkennt Dreiecksformationen"""
        patterns = []
        
        try:
            if len(closes) < 20:
                return patterns
            
            recent_data = closes[-20:]
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Trend der Hochs und Tiefs berechnen
            x_highs = list(range(len(recent_highs)))
            x_lows = list(range(len(recent_lows)))
            
            # Lineare Regression für Trendlinien
            high_slope = np.polyfit(x_highs, recent_highs, 1)[0]
            low_slope = np.polyfit(x_lows, recent_lows, 1)[0]
            
            # Aufsteigendes Dreieck (horizontale Resistance, steigende Support)
            if abs(high_slope) < 0.1 and low_slope > 0.1:
                patterns.append({
                    'pattern': 'triangle_ascending',
                    'name': 'Aufsteigendes Dreieck',
                    'confidence': 65,
                    'signal': 'BULLISH',
                    'target_price': max(recent_highs) * 1.03,
                    'stop_loss': min(recent_lows) * 0.98,
                    'description': 'Bullisches Fortsetzungsmuster'
                })
            
            # Absteigendes Dreieck (fallende Resistance, horizontale Support)
            elif high_slope < -0.1 and abs(low_slope) < 0.1:
                patterns.append({
                    'pattern': 'triangle_descending',
                    'name': 'Absteigendes Dreieck',
                    'confidence': 65,
                    'signal': 'BEARISH',
                    'target_price': min(recent_lows) * 0.97,
                    'stop_loss': max(recent_highs) * 1.02,
                    'description': 'Bärisches Fortsetzungsmuster'
                })
            
            # Symmetrisches Dreieck (fallende Resistance, steigende Support)
            elif high_slope < -0.05 and low_slope > 0.05:
                patterns.append({
                    'pattern': 'triangle_symmetrical',
                    'name': 'Symmetrisches Dreieck',
                    'confidence': 60,
                    'signal': 'NEUTRAL',
                    'target_price': recent_data[-1],
                    'stop_loss': recent_data[-1] * 0.95,
                    'description': 'Fortsetzungsmuster - Richtung unbestimmt'
                })
            
        except Exception:
            pass
        
        return patterns
    
    def _detect_flags_and_wedges(self, highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
        """🚩 Erkennt Flaggen und Keile"""
        patterns = []
        
        try:
            if len(closes) < 15:
                return patterns
            
            # Letzten 15 Kerzen für Flaggen/Keil-Erkennung
            recent_closes = closes[-15:]
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Volatilität der letzten Kerzen prüfen
            volatility = np.std(recent_closes) / np.mean(recent_closes)
            
            # Bullische Flagge (kleine Konsolidierung nach starkem Anstieg)
            if len(closes) > 20:
                prev_period = closes[-25:-15]
                current_period = closes[-15:]
                
                # Starker vorheriger Anstieg
                prev_gain = (prev_period[-1] - prev_period[0]) / prev_period[0]
                current_range = (max(current_period) - min(current_period)) / np.mean(current_period)
                
                if prev_gain > 0.03 and current_range < 0.02:  # 3% Anstieg, dann 2% Range
                    patterns.append({
                        'pattern': 'flag_bull',
                        'name': 'Bullische Flagge',
                        'confidence': 70,
                        'signal': 'BULLISH',
                        'target_price': recent_closes[-1] * 1.05,
                        'stop_loss': min(recent_lows) * 0.98,
                        'description': 'Bullisches Fortsetzungsmuster'
                    })
                
                # Bärische Flagge (kleine Konsolidierung nach starkem Fall)
                elif prev_gain < -0.03 and current_range < 0.02:
                    patterns.append({
                        'pattern': 'flag_bear',
                        'name': 'Bärische Flagge',
                        'confidence': 70,
                        'signal': 'BEARISH',
                        'target_price': recent_closes[-1] * 0.95,
                        'stop_loss': max(recent_highs) * 1.02,
                        'description': 'Bärisches Fortsetzungsmuster'
                    })
            
        except Exception:
            pass
        
        return patterns
    
    def _detect_cup_and_handle(self, lows: List[float], closes: List[float]) -> Dict:
        """☕ Erkennt Tasse-mit-Henkel Formation"""
        try:
            if len(lows) < 30:
                return None
            
            # Suche nach U-förmiger Formation in den Tiefs
            recent_lows = lows[-30:]
            
            # Finde den tiefsten Punkt (Boden der Tasse)
            cup_bottom_idx = recent_lows.index(min(recent_lows))
            
            # Prüfe ob genug Daten vor und nach dem Boden vorhanden sind
            if cup_bottom_idx < 10 or cup_bottom_idx > len(recent_lows) - 10:
                return None
            
            # Tasse: U-förmige Formation
            left_side = recent_lows[:cup_bottom_idx]
            right_side = recent_lows[cup_bottom_idx:]
            
            # Henkel: kleine Korrektur am Ende
            if len(right_side) > 5:
                handle_lows = right_side[-8:]
                handle_correction = (max(handle_lows) - min(handle_lows)) / max(handle_lows)
                
                if 0.01 < handle_correction < 0.05:  # 1-5% Korrektur für Henkel
                    return {
                        'pattern': 'cup_and_handle',
                        'name': 'Tasse mit Henkel',
                        'confidence': 75,
                        'signal': 'BULLISH',
                        'target_price': closes[-1] * 1.06,
                        'stop_loss': min(handle_lows) * 0.97,
                        'description': 'Starkes bullisches Fortsetzungsmuster'
                    }
            
        except Exception:
            pass
        
        return None
    
    def _get_strongest_pattern(self, patterns: List[Dict]) -> Dict:
        """🏆 Findet das stärkste Chartmuster"""
        if not patterns:
            return {'pattern': 'none', 'signal': 'NEUTRAL'}
        
        # Sortiere nach Confidence
        strongest = max(patterns, key=lambda p: p.get('confidence', 0))
        return strongest

# Global instance
chart_analyzer = ChartPatternAnalyzer()
