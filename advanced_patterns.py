import numpy as np
import pandas as pd
from typing import List, Dict, Any

class AdvancedPatternDetector:
    """Advanced Chart Pattern Detection with Timeframe-Specific TP/SL"""
    def __init__(self):
        self.timeframe_multipliers = {
            '15m': {'tp_base': 0.5, 'sl_base': 0.3, 'volatility_adj': 1.2},
            '1h': {'tp_base': 1.0, 'sl_base': 0.5, 'volatility_adj': 1.0},
            '4h': {'tp_base': 2.0, 'sl_base': 0.8, 'volatility_adj': 0.8},
            '1d': {'tp_base': 3.5, 'sl_base': 1.2, 'volatility_adj': 0.6}
        }

    def detect_advanced_patterns(self, df: pd.DataFrame, timeframe: str, current_price: float) -> List[Dict[str, Any]]:
        patterns = []
        patterns += self._detect_double_patterns(df['high'].values, df['low'].values, df['close'].values, timeframe, current_price)
        patterns += self._detect_head_shoulders(df['high'].values, df['low'].values, df['close'].values, timeframe, current_price)
        patterns += self._detect_triangle_patterns(df['high'].values, df['low'].values, df['close'].values, timeframe, current_price)
        return patterns

    def _detect_double_patterns(self, highs, lows, closes, timeframe: str, current_price: float) -> List[Dict[str, Any]]:
        patterns = []
        # Double Top
        if len(highs) >= 30:
            recent_highs = highs[-25:]
            peaks = [i for i in range(3, len(recent_highs)-3) if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]]
            if len(peaks) >= 2:
                patterns.append({
                    'name': 'Double Top',
                    'direction': 'SHORT',
                    'confidence': 78,
                    'timeframe': timeframe,
                    'strength': 'HIGH',
                    'description': f'Double Top on {timeframe} - breakdown below ${current_price:.2f} expected'
                })
        # Double Bottom
        if len(lows) >= 30:
            recent_lows = lows[-25:]
            valleys = [i for i in range(3, len(recent_lows)-3) if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]]
            if len(valleys) >= 2:
                patterns.append({
                    'name': 'Double Bottom',
                    'direction': 'LONG',
                    'confidence': 78,
                    'timeframe': timeframe,
                    'strength': 'HIGH',
                    'description': f'Double Bottom on {timeframe} - breakout above ${current_price:.2f} expected'
                })
        return patterns

    def _detect_head_shoulders(self, highs, lows, closes, timeframe: str, current_price: float) -> List[Dict[str, Any]]:
        patterns = []
        # Dummy implementation
        return patterns

    def _detect_triangle_patterns(self, highs, lows, closes, timeframe: str, current_price: float) -> List[Dict[str, Any]]:
        patterns = []
        # Dummy implementation
        return patterns
