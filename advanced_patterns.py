# ==========================================
# 🔬 ADVANCED PATTERN RECOGNITION ENGINE
# Professional Trading Patterns like your colleague's dashboard
# ==========================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingPattern:
    """Advanced Trading Pattern Definition"""
    name: str
    confidence: float
    direction: str  # LONG, SHORT, NEUTRAL
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    timeframe_days: str
    pattern_type: str  # 'REVERSAL', 'CONTINUATION', 'BREAKOUT'
    validation_price: Optional[float] = None
    description: str = ""

class AdvancedPatternEngine:
    """🔬 Advanced Pattern Recognition like your colleague's dashboard"""
    
    def __init__(self):
        self.min_pattern_confidence = 40.0  # Gelockert von 45.0
        self.patterns_detected = []
    
    def detect_all_patterns(self, df: pd.DataFrame, current_price: float, symbol: str) -> List[TradingPattern]:
        """Detect all advanced patterns like in the dashboard"""
        patterns = []
        
        try:
            # 🎯 DOUBLE BOTTOM PATTERN (like 85.5% in dashboard)
            double_bottom = self._detect_double_bottom(df, current_price, symbol)
            if double_bottom:
                patterns.append(double_bottom)
            
            # 📉 BEARISH FAIR VALUE GAP (like 71.1% & 76.2% in dashboard)
            fair_value_gaps = self._detect_fair_value_gaps(df, current_price, symbol)
            patterns.extend(fair_value_gaps)
            
            # 💥 SUPER BREAKDOWN PATTERN (like 73.7% in dashboard)
            super_breakdown = self._detect_super_breakdown(df, current_price, symbol)
            if super_breakdown:
                patterns.append(super_breakdown)
            
            # 🚀 BREAKOUT PATTERNS
            breakout_patterns = self._detect_breakout_patterns(df, current_price, symbol)
            patterns.extend(breakout_patterns)
            
            # Sort by confidence (highest first)
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            logger.info(f"🔬 Advanced Pattern Detection: Found {len(patterns)} patterns for {symbol}")
            return patterns[:10]  # Top 10 patterns
            
        except Exception as e:
            logger.error(f"❌ Advanced pattern detection error: {e}")
            return []
    
    def _detect_double_bottom(self, df: pd.DataFrame, current_price: float, symbol: str) -> Optional[TradingPattern]:
        """🎯 Double Bottom Pattern Detection (like 85.5% confidence in dashboard)"""
        try:
            if len(df) < 50:
                return None
            
            # Find local lows
            lows = df['low'].rolling(window=5, center=True).min()
            low_indices = df.index[df['low'] == lows].tolist()
            
            if len(low_indices) < 2:
                return None
            
            # Check for double bottom formation
            recent_lows = low_indices[-10:]  # Last 10 lows
            if len(recent_lows) >= 2:
                low1_idx = recent_lows[-2]
                low2_idx = recent_lows[-1]
                
                low1_price = df.loc[low1_idx, 'low']
                low2_price = df.loc[low2_idx, 'low']
                
                # Double bottom criteria
                price_similarity = abs(low1_price - low2_price) / low1_price
                
                if price_similarity < 0.02:  # Within 2%
                    # Calculate pattern metrics
                    pattern_high = df.loc[low1_idx:low2_idx, 'high'].max()
                    neckline = pattern_high
                    
                    # Entry, target, stop loss
                    entry = current_price
                    target = neckline + (neckline - min(low1_price, low2_price)) * 0.8
                    stop_loss = min(low1_price, low2_price) * 0.99
                    
                    risk_reward = (target - entry) / (entry - stop_loss) if entry > stop_loss else 0
                    
                    # Confidence calculation
                    volume_confirmation = df.loc[low2_idx:, 'volume'].mean() > df.loc[low1_idx:low2_idx, 'volume'].mean()
                    confidence = 75.0 + (10.0 if volume_confirmation else 0) + min(risk_reward * 5, 10)
                    
                    if confidence >= self.min_pattern_confidence:
                        return TradingPattern(
                            name="Double Bottom",
                            confidence=min(confidence, 95.0),
                            direction="LONG",
                            entry_price=entry,
                            target_price=target,
                            stop_loss=stop_loss,
                            risk_reward=risk_reward,
                            timeframe_days="2-8",
                            pattern_type="REVERSAL",
                            validation_price=neckline,
                            description=f"Validated Double Bottom bei ${min(low1_price, low2_price):.2f}"
                        )
            
        except Exception as e:
            logger.error(f"Double bottom detection error: {e}")
        
        return None
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame, current_price: float, symbol: str) -> List[TradingPattern]:
        """📉 Fair Value Gap Detection (like 71.1% & 76.2% in dashboard)"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            # Look for gaps in the last 20 candles
            for i in range(len(df) - 20, len(df) - 2):
                prev_candle = df.iloc[i]
                current_candle = df.iloc[i + 1]
                next_candle = df.iloc[i + 2]
                
                # Bearish Fair Value Gap
                if (prev_candle['low'] > next_candle['high'] and 
                    current_candle['close'] < current_candle['open']):
                    
                    gap_top = prev_candle['low']
                    gap_bottom = next_candle['high']
                    gap_size = (gap_top - gap_bottom) / gap_bottom
                    
                    if gap_size > 0.005:  # Minimum 0.5% gap
                        entry = current_price
                        target = gap_bottom * 0.995
                        stop_loss = gap_top * 1.005
                        
                        risk_reward = (entry - target) / (stop_loss - entry) if stop_loss > entry else 0
                        
                        # Confidence based on gap size and market structure
                        confidence = 65.0 + min(gap_size * 1000, 15) + min(risk_reward * 5, 10)
                        
                        if confidence >= self.min_pattern_confidence:
                            patterns.append(TradingPattern(
                                name="Bearish Fair Value Gap",
                                confidence=min(confidence, 85.0),
                                direction="SHORT",
                                entry_price=entry,
                                target_price=target,
                                stop_loss=stop_loss,
                                risk_reward=risk_reward,
                                timeframe_days="1-5",
                                pattern_type="CONTINUATION",
                                validation_price=gap_top,
                                description=f"Validated Bearish FVG bei ${gap_top:.2f}"
                            ))
                
                # Bullish Fair Value Gap
                elif (prev_candle['high'] < next_candle['low'] and 
                      current_candle['close'] > current_candle['open']):
                    
                    gap_bottom = prev_candle['high']
                    gap_top = next_candle['low']
                    gap_size = (gap_top - gap_bottom) / gap_bottom
                    
                    if gap_size > 0.005:  # Minimum 0.5% gap
                        entry = current_price
                        target = gap_top * 1.005
                        stop_loss = gap_bottom * 0.995
                        
                        risk_reward = (target - entry) / (entry - stop_loss) if entry > stop_loss else 0
                        
                        confidence = 65.0 + min(gap_size * 1000, 15) + min(risk_reward * 5, 10)
                        
                        if confidence >= self.min_pattern_confidence:
                            patterns.append(TradingPattern(
                                name="Bullish Fair Value Gap",
                                confidence=min(confidence, 85.0),
                                direction="LONG",
                                entry_price=entry,
                                target_price=target,
                                stop_loss=stop_loss,
                                risk_reward=risk_reward,
                                timeframe_days="1-5",
                                pattern_type="CONTINUATION",
                                validation_price=gap_bottom,
                                description=f"Validated Bullish FVG bei ${gap_bottom:.2f}"
                            ))
        
        except Exception as e:
            logger.error(f"Fair value gap detection error: {e}")
        
        return patterns
    
    def _detect_super_breakdown(self, df: pd.DataFrame, current_price: float, symbol: str) -> Optional[TradingPattern]:
        """💥 Super Breakdown Pattern (like 73.7% in dashboard)"""
        try:
            if len(df) < 30:
                return None
            
            # Look for multiple pattern confluence
            recent_data = df.tail(30)
            
            # Check for bearish confluence
            ema_20 = recent_data['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = recent_data['close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema_20
            
            # Multiple bearish factors
            bearish_factors = 0
            
            # Factor 1: Price below EMAs
            if current_price < ema_20 < ema_50:
                bearish_factors += 1
            
            # Factor 2: Recent breakdown below support
            support_level = recent_data['low'].rolling(window=10).min().iloc[-1]
            if current_price < support_level * 1.005:
                bearish_factors += 1
            
            # Factor 3: Volume increase on breakdown
            recent_volume = recent_data['volume'].tail(5).mean()
            prev_volume = recent_data['volume'].head(25).mean()
            if recent_volume > prev_volume * 1.2:
                bearish_factors += 1
            
            # Factor 4: Bear Fair Value Gap present
            fair_value_gaps = self._detect_fair_value_gaps(df, current_price, symbol)
            bearish_gaps = [gap for gap in fair_value_gaps if gap.direction == "SHORT"]
            if bearish_gaps:
                bearish_factors += 1
            
            if bearish_factors >= 2:  # At least 2 confluences
                entry = current_price
                target = support_level * 0.95
                stop_loss = max(ema_20, recent_data['high'].tail(5).max()) * 1.01
                
                risk_reward = (entry - target) / (stop_loss - entry) if stop_loss > entry else 0
                
                # Confidence based on confluence factors
                confidence = 60.0 + (bearish_factors * 8) + min(risk_reward * 3, 15)
                
                if confidence >= self.min_pattern_confidence and risk_reward > 1.0:
                    return TradingPattern(
                        name="Super Breakdown",
                        confidence=min(confidence, 90.0),
                        direction="SHORT",
                        entry_price=entry,
                        target_price=target,
                        stop_loss=stop_loss,
                        risk_reward=risk_reward,
                        timeframe_days="2-14",
                        pattern_type="BREAKDOWN",
                        validation_price=support_level,
                        description=f"Massive {bearish_factors}-Pattern Confluence! Bearish Fair Value Gap, Bearish Fair Value Gap bestätigt sich gegenseitig"
                    )
        
        except Exception as e:
            logger.error(f"Super breakdown detection error: {e}")
        
        return None
    
    def _detect_breakout_patterns(self, df: pd.DataFrame, current_price: float, symbol: str) -> List[TradingPattern]:
        """🚀 Breakout Pattern Detection"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            # Double Bottom Breakout
            double_bottom = self._detect_double_bottom(df, current_price, symbol)
            if double_bottom and current_price > double_bottom.validation_price:
                breakout_pattern = TradingPattern(
                    name="Double Bottom Breakout",
                    confidence=double_bottom.confidence * 0.7,  # Slightly lower for breakout
                    direction="LONG",
                    entry_price=current_price,
                    target_price=double_bottom.target_price,
                    stop_loss=double_bottom.validation_price * 0.99,
                    risk_reward=double_bottom.risk_reward,
                    timeframe_days="1-5",
                    pattern_type="BREAKOUT",
                    validation_price=double_bottom.validation_price,
                    description="Bullisches Szenario basierend auf Double Bottom"
                )
                patterns.append(breakout_pattern)
        
        except Exception as e:
            logger.error(f"Breakout detection error: {e}")
        
        return patterns

class TradingScenarioEngine:
    """🎯 Trading Scenarios like in the dashboard"""
    
    def __init__(self):
        self.scenario_engine = AdvancedPatternEngine()
    
    def generate_trading_scenarios(self, patterns: List[TradingPattern], current_price: float) -> List[Dict[str, Any]]:
        """Generate detailed trading scenarios with risk management"""
        scenarios = []
        
        for pattern in patterns[:3]:  # Top 3 patterns like in dashboard
            scenario = self._create_scenario(pattern, current_price)
            scenarios.append(scenario)
        
        return scenarios
    
    def _create_scenario(self, pattern: TradingPattern, current_price: float) -> Dict[str, Any]:
        """Create detailed trading scenario with risk levels"""
        
        # Calculate position sizes for different risk levels
        conservative_size = self._calculate_position_size(pattern.risk_reward, "CONSERVATIVE")
        standard_size = self._calculate_position_size(pattern.risk_reward, "STANDARD")
        aggressive_size = self._calculate_position_size(pattern.risk_reward, "AGGRESSIVE")
        
        return {
            'name': pattern.name,
            'confidence': pattern.confidence,
            'direction': pattern.direction,
            'timeframe': pattern.timeframe_days,
            'description': pattern.description,
            'execution_price': pattern.entry_price,
            'risk_management': {
                'conservative': {
                    'position_size': conservative_size,
                    'target': pattern.target_price,
                    'stop_loss': pattern.stop_loss
                },
                'standard': {
                    'position_size': standard_size,
                    'target': pattern.target_price,
                    'stop_loss': pattern.stop_loss
                },
                'aggressive': {
                    'position_size': aggressive_size,
                    'target': pattern.target_price,
                    'stop_loss': pattern.stop_loss
                }
            },
            'risk_reward': pattern.risk_reward,
            'pattern_type': pattern.pattern_type
        }
    
    def _calculate_position_size(self, risk_reward: float, risk_level: str) -> str:
        """Calculate position size based on risk level"""
        base_sizes = {
            'CONSERVATIVE': "1-2%",
            'STANDARD': "2-4%", 
            'AGGRESSIVE': "4-8%"
        }
        
        # Adjust based on risk/reward ratio
        if risk_reward >= 3:
            multiplier = 1.5
        elif risk_reward >= 2:
            multiplier = 1.2
        else:
            multiplier = 0.8
        
        return base_sizes[risk_level]
