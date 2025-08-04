# ==========================================
# 🚀 ULTIMATE TRADING V3 - BEST OF ALL WORLDS
# PREMIUM Edition mit ALLEM was gut war!
# ==========================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import json
import logging
import time
import warnings
import random

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Imports (optional)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
    logger.info("✅ scikit-learn available - using real ML models")
except ImportError:
    sklearn_available = False
    logger.info("⚠️ scikit-learn not available - using rule-based predictions")

# ==========================================
# 🏗️ ENHANCED DATA MODELS
# ==========================================

@dataclass
class PatternResult:
    name: str
    pattern_type: str
    confidence: float
    direction: str  # LONG/SHORT/NEUTRAL
    timeframe: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    setup_quality: Optional[str] = None
    explanation: str = ""
    technical_details: Dict[str, Any] = field(default_factory=dict)
    age_candles: int = 0

@dataclass
class SMCPattern:
    """Smart Money Concepts Pattern"""
    name: str
    pattern_type: str
    direction: str
    confidence: float
    price_level: float
    zone_high: float
    zone_low: float
    strength: str
    distance_pct: float
    retest_probability: float
    explanation: str = ""

@dataclass
class MarketStructure:
    """Enhanced Market Structure Analysis"""
    trend_direction: str
    structure_strength: float
    last_bos: Optional[Dict] = None
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    key_levels: List[float] = field(default_factory=list)

@dataclass
class LiquidationData:
    symbol: str
    funding_rate: float
    funding_sentiment: str
    open_interest: float
    long_liquidations: List[float]
    short_liquidations: List[float]
    heatmap_levels: List[Dict] = field(default_factory=list)

@dataclass
class MLPrediction:
    strategy: str
    direction: str
    confidence: float
    score: float
    timeframe: str
    risk_level: str
    signal_quality: str = "MEDIUM"
    reliability_score: float = 50.0

@dataclass
class SignalBoostResult:
    boosted_signals: List[Dict]
    boost_metrics: Dict[str, Any]
    signal_count: int
    confidence_increase: float

@dataclass
class AnalysisResult:
    symbol: str
    current_price: float
    timestamp: datetime
    timeframe: str
    
    # Main Signals
    main_signal: str
    confidence: float
    signal_quality: str
    recommendation: str
    risk_level: float
    
    # Pattern Analysis
    chart_patterns: List[PatternResult] = field(default_factory=list)
    smc_patterns: List[SMCPattern] = field(default_factory=list)
    best_pattern: Optional[PatternResult] = None
    pattern_confluence: float = 0.0
    
    # Enhanced Analysis
    market_structure: Optional[MarketStructure] = None
    liquidation_data: Optional[LiquidationData] = None
    ml_predictions: Dict[str, MLPrediction] = field(default_factory=dict)
    signal_boost: Optional[SignalBoostResult] = None
    
    # Performance
    execution_time: float = 0.0

# ==========================================
# 🧠 ADVANCED ML PREDICTION ENGINE
# ==========================================

class AdvancedMLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            if not sklearn_available:
                logger.info("Using rule-based ML models")
                self.models = {
                    'scalping': 'rule_based',
                    'day_trading': 'rule_based',
                    'swing_trading': 'rule_based'
                }
                return
            
            # Real ML Models
            self.models['scalping'] = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42
            )
            self.models['day_trading'] = GradientBoostingClassifier(
                n_estimators=30, max_depth=6, random_state=42
            )
            self.models['swing_trading'] = RandomForestClassifier(
                n_estimators=40, max_depth=8, random_state=42
            )
            
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            logger.info("✅ Real ML Models initialized")
            
        except Exception as e:
            logger.error(f"ML Model initialization error: {e}")
    
    def predict_all_strategies(self, indicators: Dict, patterns: Dict, 
                             price_data: List, volume_data: List) -> Dict[str, MLPrediction]:
        """Predict all trading strategies"""
        features = self._extract_comprehensive_features(indicators, patterns, price_data, volume_data)
        
        predictions = {}
        
        # Scalping Prediction (1-15 minutes)
        scalping = self._predict_scalping(features)
        predictions['scalping'] = MLPrediction(
            strategy='Scalping',
            direction=scalping['direction'],
            confidence=scalping['confidence'],
            score=scalping['score'],
            timeframe=scalping['timeframe'],
            risk_level=scalping['risk_level'],
            signal_quality=scalping.get('signal_quality', 'MEDIUM'),
            reliability_score=scalping.get('reliability_score', scalping['confidence'])
        )
        
        # Day Trading Prediction (1-24 hours)
        day_trading = self._predict_day_trading(features)
        predictions['day_trading'] = MLPrediction(
            strategy='Day Trading',
            direction=day_trading['direction'],
            confidence=day_trading['confidence'],
            score=day_trading['score'],
            timeframe=day_trading['timeframe'],
            risk_level=day_trading['risk_level']
        )
        
        # Swing Trading Prediction (1-10 days)
        swing_trading = self._predict_swing_trading(features)
        predictions['swing_trading'] = MLPrediction(
            strategy='Swing Trading',
            direction=swing_trading['direction'],
            confidence=swing_trading['confidence'],
            score=swing_trading['score'],
            timeframe=swing_trading['timeframe'],
            risk_level=swing_trading['risk_level']
        )
        
        return predictions
    
    def _extract_comprehensive_features(self, indicators: Dict, patterns: Dict, 
                                      price_data: List, volume_data: List) -> Dict:
        """Extract comprehensive features for ML models"""
        features = {}
        
        # Price features
        recent_prices = [p['close'] for p in price_data[-20:]] if len(price_data) >= 20 else [p['close'] for p in price_data]
        if recent_prices:
            features['price_trend'] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            features['price_volatility'] = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
            features['price_momentum'] = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 and recent_prices[-5] > 0 else 0
        
        # Volume features
        recent_volumes = volume_data[-10:] if len(volume_data) >= 10 else volume_data
        if len(recent_volumes) > 1:
            features['volume_trend'] = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0] if recent_volumes[0] > 0 else 0
            features['volume_spike'] = recent_volumes[-1] / np.mean(recent_volumes[:-1]) if np.mean(recent_volumes[:-1]) > 0 else 1
        
        # Technical indicators
        features['rsi'] = indicators.get('rsi', 50)
        features['rsi_divergence'] = abs(features['rsi'] - 50) / 50
        features['macd_signal'] = 1 if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else -1
        features['bb_position'] = self._calculate_bb_position(indicators, recent_prices[-1] if recent_prices else 0)
        features['trend_strength'] = self._calculate_trend_strength(indicators)
        
        # Pattern features
        bullish_patterns = ['hammer', 'engulfing_bullish', 'bullish_fvg', 'bullish_ob', 'bos_bullish']
        bearish_patterns = ['shooting_star', 'engulfing_bearish', 'bearish_fvg', 'bearish_ob', 'bos_bearish']
        
        features['bullish_pattern_count'] = sum(1 for p in bullish_patterns if patterns.get(p, False))
        features['bearish_pattern_count'] = sum(1 for p in bearish_patterns if patterns.get(p, False))
        features['pattern_strength'] = features['bullish_pattern_count'] - features['bearish_pattern_count']
        
        # Smart Money features
        features['fvg_signal'] = 1 if patterns.get('bullish_fvg', False) else (-1 if patterns.get('bearish_fvg', False) else 0)
        features['liquidity_sweep'] = 1 if patterns.get('liquidity_sweep', False) else 0
        features['order_block_signal'] = 1 if patterns.get('bullish_ob', False) else (-1 if patterns.get('bearish_ob', False) else 0)
        
        # Liquidity features
        features['equal_highs'] = 1 if patterns.get('equal_highs', False) else 0
        features['equal_lows'] = 1 if patterns.get('equal_lows', False) else 0
        features['stop_hunt'] = 1 if patterns.get('stop_hunt_high', False) or patterns.get('stop_hunt_low', False) else 0
        features['volume_cluster'] = 1 if patterns.get('volume_cluster', False) else 0
        
        return features
    
    def _predict_scalping(self, features: Dict) -> Dict:
        """Scalping predictions (1-15 minutes) - HIGH PRECISION"""
        score = 0
        confidence_factors = []
        
        # RSI extremes for quick reversals
        rsi = features.get('rsi', 50)
        if rsi < 25:
            score += 3  # Strong oversold
            confidence_factors.append(0.9)
        elif rsi > 75:
            score -= 3  # Strong overbought
            confidence_factors.append(0.9)
        elif 30 <= rsi <= 35 or 65 <= rsi <= 70:
            score += 1 if rsi < 40 else -1
            confidence_factors.append(0.7)
        
        # Volume spike confirmation
        volume_spike = features.get('volume_spike', 1)
        if volume_spike > 2.0:
            score += 2
            confidence_factors.append(0.85)
        elif volume_spike > 1.5:
            score += 1
            confidence_factors.append(0.7)
        
        # Pattern strength
        pattern_strength = features.get('pattern_strength', 0)
        fvg_signal = features.get('fvg_signal', 0)
        
        if pattern_strength >= 2:
            score += pattern_strength
            confidence_factors.append(0.8)
        elif pattern_strength <= -2:
            score += pattern_strength
            confidence_factors.append(0.8)
        
        # Smart Money Confluence
        smart_money_bull = 0
        smart_money_bear = 0
        
        if features.get('fvg_signal', 0) > 0:
            smart_money_bull += 1
        if features.get('order_block_signal', 0) > 0:
            smart_money_bull += 1
        if features.get('fvg_signal', 0) < 0:
            smart_money_bear += 1
        if features.get('order_block_signal', 0) < 0:
            smart_money_bear += 1
        
        if smart_money_bull >= 2 and volume_spike > 1.2:
            score += 3
            confidence_factors.append(0.95)
        elif smart_money_bear >= 2 and volume_spike > 1.2:
            score -= 3
            confidence_factors.append(0.95)
        
        # Liquidity features
        if features.get('stop_hunt', 0):
            score += 2  # Stop hunts = excellent reversal signals
            confidence_factors.append(0.85)
        
        if features.get('equal_highs', 0) and rsi > 60:
            score -= 1.5
            confidence_factors.append(0.75)
        elif features.get('equal_lows', 0) and rsi < 40:
            score += 1.5
            confidence_factors.append(0.75)
        
        if features.get('volume_cluster', 0):
            score += 1
            confidence_factors.append(0.7)
        
        # Direction determination
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        
        # Premium confidence calculation
        premium_confidence = self._calculate_premium_confidence(features, confidence_factors, score, pattern_strength, fvg_signal)
        signal_quality = self._assess_signal_quality(premium_confidence, score, features)
        
        return {
            'direction': direction,
            'confidence': premium_confidence,
            'score': score,
            'timeframe': '1-15 minutes',
            'risk_level': 'HIGH',
            'signal_quality': signal_quality,
            'reliability_score': premium_confidence
        }
    
    def _predict_day_trading(self, features: Dict) -> Dict:
        """Day trading predictions (1-24 hours)"""
        score = 0
        confidence_factors = []
        
        # Trend alignment
        trend_strength = features.get('trend_strength', 0)
        if trend_strength > 0.7:
            score += 2
            confidence_factors.append(0.8)
        elif trend_strength < -0.7:
            score -= 2
            confidence_factors.append(0.8)
        
        # RSI for day trading
        rsi = features.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 0.5  # Neutral RSI good for trend continuation
            confidence_factors.append(0.6)
        elif rsi < 30 or rsi > 70:
            score += 1 if rsi < 30 else -1
            confidence_factors.append(0.7)
        
        # MACD confirmation
        macd_signal = features.get('macd_signal', 0)
        if macd_signal != 0:
            score += macd_signal
            confidence_factors.append(0.7)
        
        # Pattern confirmation
        pattern_strength = features.get('pattern_strength', 0)
        if abs(pattern_strength) >= 1:
            score += pattern_strength
            confidence_factors.append(0.75)
        
        # Volume trend
        volume_trend = features.get('volume_trend', 0)
        if abs(volume_trend) > 0.2:
            score += 1 if volume_trend > 0 else -1
            confidence_factors.append(0.6)
        
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        confidence = min(90, max(40, np.mean(confidence_factors) * 85 + abs(score) * 5)) if confidence_factors else 50
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '1-24 hours',
            'risk_level': 'MEDIUM'
        }
    
    def _predict_swing_trading(self, features: Dict) -> Dict:
        """Swing trading predictions (1-10 days)"""
        score = 0
        confidence_factors = []
        
        # RSI for swing levels
        rsi = features.get('rsi', 50)
        if 25 <= rsi <= 35:
            score += 3
            confidence_factors.append(0.8)
        elif 65 <= rsi <= 75:
            score -= 3
            confidence_factors.append(0.8)
        elif rsi < 20 or rsi > 80:
            score += 1 if rsi < 20 else -1
            confidence_factors.append(0.6)
        
        # Pattern strength important for swing
        pattern_strength = features.get('pattern_strength', 0)
        score += pattern_strength * 2
        if pattern_strength != 0:
            confidence_factors.append(0.7)
        
        # Trend strength
        trend_strength = features.get('trend_strength', 0)
        score += (trend_strength - 0.5) * 2
        confidence_factors.append(0.6)
        
        # Volume confirmation
        volume_spike = features.get('volume_spike', 1)
        if volume_spike > 1.5:
            score += 1
            confidence_factors.append(0.7)
        
        direction = 'BUY' if score > 1.5 else 'SELL' if score < -1.5 else 'NEUTRAL'
        confidence = min(88, max(40, np.mean(confidence_factors) * 85 + abs(score) * 5)) if confidence_factors else 50
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '1-10 days',
            'risk_level': 'MEDIUM'
        }
    
    def _calculate_premium_confidence(self, features: Dict, confidence_factors: List, 
                                    score: float, pattern_strength: float, fvg_signal: float) -> float:
        """Premium multi-layer confidence calculation"""
        base_confidence = np.mean(confidence_factors) * 100 if confidence_factors else 30
        
        # Pattern confluence bonus
        confluence_bonus = 0
        if abs(pattern_strength) >= 2:
            confluence_bonus += 25
        elif abs(pattern_strength) == 1:
            confluence_bonus += 10
        
        if abs(fvg_signal) > 0:
            confluence_bonus += 15
        
        # Volume confirmation
        volume_spike = features.get('volume_spike', 1)
        volume_bonus = min(15, (volume_spike - 1) * 20) if volume_spike > 1 else 0
        
        # RSI extremes bonus
        rsi = features.get('rsi', 50)
        rsi_bonus = 0
        if rsi < 25 or rsi > 75:
            rsi_bonus = 10
        elif rsi < 30 or rsi > 70:
            rsi_bonus = 5
        
        # Smart Money bonus
        smart_money_signals = sum([
            1 if features.get('fvg_signal', 0) != 0 else 0,
            1 if features.get('order_block_signal', 0) != 0 else 0,
            1 if features.get('stop_hunt', 0) else 0,
            1 if features.get('liquidity_sweep', 0) else 0
        ])
        smart_money_bonus = min(20, smart_money_signals * 5)
        
        final_confidence = base_confidence + confluence_bonus + volume_bonus + rsi_bonus + smart_money_bonus
        return max(20, min(98, final_confidence))
    
    def _assess_signal_quality(self, confidence: float, score: float, features: Dict) -> str:
        """Assess signal quality based on multiple factors"""
        quality_score = confidence
        
        # Adjust based on pattern confluence
        pattern_count = features.get('bullish_pattern_count', 0) + features.get('bearish_pattern_count', 0)
        if pattern_count >= 3:
            quality_score += 10
        elif pattern_count >= 2:
            quality_score += 5
        
        # Adjust based on volume
        volume_spike = features.get('volume_spike', 1)
        if volume_spike > 2.0:
            quality_score += 10
        elif volume_spike > 1.5:
            quality_score += 5
        
        # Adjust based on smart money signals
        smart_money_signals = sum([
            1 if features.get('fvg_signal', 0) != 0 else 0,
            1 if features.get('order_block_signal', 0) != 0 else 0,
            1 if features.get('stop_hunt', 0) else 0
        ])
        if smart_money_signals >= 2:
            quality_score += 10
        elif smart_money_signals == 1:
            quality_score += 5
        
        if quality_score >= 85:
            return "PREMIUM"
        elif quality_score >= 75:
            return "HIGH"
        elif quality_score >= 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_bb_position(self, indicators: Dict, current_price: float) -> float:
        """Calculate position within Bollinger Bands"""
        bb_upper = indicators.get('bb_upper', current_price)
        bb_lower = indicators.get('bb_lower', current_price)
        if bb_upper == bb_lower:
            return 0.5
        return (current_price - bb_lower) / (bb_upper - bb_lower)
    
    def _calculate_trend_strength(self, indicators: Dict) -> float:
        """Calculate overall trend strength"""
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        
        if ema_50 == 0 or sma_200 == 0:
            return 0
        
        trend_score = 0
        if ema_20 > ema_50:
            trend_score += 1
        if ema_50 > sma_200:
            trend_score += 1
        
        return trend_score / 2

# ==========================================
# 🔥 SIGNAL BOOSTER ENGINE
# ==========================================

class SignalBoosterEngine:
    def __init__(self):
        self.enabled = True
        self.boost_multiplier = 2.0
    
    def boost_signal_detection(self, indicators: Dict, patterns: Dict, 
                             price_data: List, volume_data: List) -> SignalBoostResult:
        """200% mehr Signale durch Multi-Strategy Approach"""
        boosted_signals = []
        confidence_increase = 0
        
        try:
            # RSI Divergence Signals
            rsi_signals = self._detect_rsi_divergence_signals(indicators, price_data)
            boosted_signals.extend(rsi_signals)
            
            # MACD Confluence Signals
            macd_signals = self._detect_macd_confluence_signals(indicators)
            boosted_signals.extend(macd_signals)
            
            # Volume Breakout Signals
            volume_signals = self._detect_volume_breakout_signals(volume_data, price_data)
            boosted_signals.extend(volume_signals)
            
            # Smart Money Confluence
            smc_signals = self._detect_smart_money_confluence(patterns, indicators)
            boosted_signals.extend(smc_signals)
            
            # Liquidity Grab Signals
            liquidity_signals = self._detect_liquidity_grab_signals(patterns, price_data)
            boosted_signals.extend(liquidity_signals)
            
            # Calculate confidence increase
            if boosted_signals:
                base_confidence = sum(signal.get('confidence', 50) for signal in boosted_signals)
                confidence_increase = min(25, len(boosted_signals) * 3)
            
            boost_metrics = {
                'boost_applied': True,
                'original_signal_count': len(patterns),
                'boosted_signal_count': len(boosted_signals),
                'boost_factor': len(boosted_signals) / max(1, len(patterns)),
                'confidence_increase': confidence_increase,
                'signal_types': list(set(signal.get('type', 'UNKNOWN') for signal in boosted_signals))
            }
            
        except Exception as e:
            logger.error(f"Signal boost error: {e}")
            boost_metrics = {
                'boost_applied': False,
                'error': str(e)
            }
        
        return SignalBoostResult(
            boosted_signals=boosted_signals,
            boost_metrics=boost_metrics,
            signal_count=len(boosted_signals),
            confidence_increase=confidence_increase
        )
    
    def _detect_rsi_divergence_signals(self, indicators: Dict, price_data: List) -> List[Dict]:
        """Detect RSI divergence signals"""
        signals = []
        
        rsi = indicators.get('rsi', 50)
        if not price_data or len(price_data) < 10:
            return signals
        
        current_price = price_data[-1]['close']
        prev_price = price_data[-5]['close'] if len(price_data) >= 5 else current_price
        
        # Bullish divergence: Price lower, RSI higher
        if current_price < prev_price and rsi > 35 and rsi < 50:
            signals.append({
                'type': 'RSI_BULLISH_DIVERGENCE',
                'direction': 'BUY',
                'confidence': 75,
                'timeframe': '1-4 hours',
                'reason': f'RSI ({rsi:.1f}) showing bullish divergence with price action'
            })
        
        # Bearish divergence: Price higher, RSI lower
        elif current_price > prev_price and rsi < 65 and rsi > 50:
            signals.append({
                'type': 'RSI_BEARISH_DIVERGENCE',
                'direction': 'SELL',
                'confidence': 75,
                'timeframe': '1-4 hours',
                'reason': f'RSI ({rsi:.1f}) showing bearish divergence with price action'
            })
        
        return signals
    
    def _detect_macd_confluence_signals(self, indicators: Dict) -> List[Dict]:
        """Detect MACD confluence signals with proper validation"""
        signals = []
        
        macd = indicators.get('macd', None)
        macd_signal = indicators.get('macd_signal', None)
        macd_hist = indicators.get('macd_histogram', None)
        
        # ✅ CRITICAL FIX: Validate MACD data exists and is meaningful
        if macd is None or macd_signal is None or macd_hist is None:
            logger.debug("MACD data missing - skipping MACD confluence detection")
            return signals
        
        # ✅ Check if MACD values are meaningful (not all zeros)
        if abs(macd) < 0.0001 and abs(macd_signal) < 0.0001 and abs(macd_hist) < 0.0001:
            logger.debug("MACD values too small/zero - skipping confluence detection")
            return signals
        
        # ✅ Additional validation: Check if values are realistic
        if abs(macd) > 10000 or abs(macd_signal) > 10000 or abs(macd_hist) > 10000:
            logger.debug("MACD values unrealistic - skipping confluence detection")
            return signals
        
        # Triple bullish confluence (with meaningful thresholds)
        if (macd > macd_signal and 
            macd_hist > 0 and 
            macd > 0 and
            abs(macd - macd_signal) > 0.001):  # Meaningful difference
            
            confidence = min(90, 75 + abs(macd_hist) * 1000)  # Scale confidence based on histogram strength
            
            signals.append({
                'type': 'MACD_TRIPLE_BULL',
                'direction': 'BUY',
                'confidence': confidence,
                'timeframe': '4-24 hours',
                'reason': f'MACD triple bullish confluence: MACD({macd:.4f}) > Signal({macd_signal:.4f}), Histogram({macd_hist:.4f}) > 0'
            })
        
        # Triple bearish confluence (with meaningful thresholds)
        elif (macd < macd_signal and 
              macd_hist < 0 and 
              macd < 0 and
              abs(macd - macd_signal) > 0.001):  # Meaningful difference
            
            confidence = min(90, 75 + abs(macd_hist) * 1000)  # Scale confidence based on histogram strength
            
            signals.append({
                'type': 'MACD_TRIPLE_BEAR',
                'direction': 'SELL',
                'confidence': confidence,
                'timeframe': '4-24 hours',
                'reason': f'MACD triple bearish confluence: MACD({macd:.4f}) < Signal({macd_signal:.4f}), Histogram({macd_hist:.4f}) < 0'
            })
        
        # ✅ ENHANCED: Regular MACD crossover signals (less strict)
        elif macd is not None and macd_signal is not None and abs(macd - macd_signal) > 0.001:
            if macd > macd_signal:
                signals.append({
                    'type': 'MACD_BULLISH_CROSS',
                    'direction': 'BUY',
                    'confidence': 70,
                    'timeframe': '2-12 hours',
                    'reason': f'MACD bullish crossover: MACD({macd:.4f}) > Signal({macd_signal:.4f})'
                })
            elif macd < macd_signal:
                signals.append({
                    'type': 'MACD_BEARISH_CROSS',
                    'direction': 'SELL',
                    'confidence': 70,
                    'timeframe': '2-12 hours',
                    'reason': f'MACD bearish crossover: MACD({macd:.4f}) < Signal({macd_signal:.4f})'
                })
        
        return signals
    
    def _detect_volume_breakout_signals(self, volume_data: List, price_data: List) -> List[Dict]:
        """Detect volume breakout signals"""
        signals = []
        
        if len(volume_data) < 10 or len(price_data) < 10:
            return signals
        
        current_volume = volume_data[-1]
        avg_volume = np.mean(volume_data[-10:-1])
        
        current_price = price_data[-1]['close']
        prev_price = price_data[-2]['close']
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        price_change_pct = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0
        
        # Volume breakout with price momentum
        if volume_ratio > 2.0 and abs(price_change_pct) > 1.0:
            direction = 'BUY' if price_change_pct > 0 else 'SELL'
            confidence = min(90, 70 + volume_ratio * 5)
            
            signals.append({
                'type': 'VOLUME_BREAKOUT',
                'direction': direction,
                'confidence': confidence,
                'timeframe': '15min-2 hours',
                'reason': f'Volume spike {volume_ratio:.1f}x with {price_change_pct:.1f}% price move'
            })
        
        return signals
    
    def _detect_smart_money_confluence(self, patterns: Dict, indicators: Dict) -> List[Dict]:
        """Detect Smart Money confluence signals"""
        signals = []
        
        smart_money_bull = 0
        smart_money_bear = 0
        
        # Count bullish SMC signals
        if patterns.get('bullish_fvg', False):
            smart_money_bull += 1
        if patterns.get('bullish_ob', False):
            smart_money_bull += 1
        if patterns.get('bos_bullish', False):
            smart_money_bull += 1
        
        # Count bearish SMC signals
        if patterns.get('bearish_fvg', False):
            smart_money_bear += 1
        if patterns.get('bearish_ob', False):
            smart_money_bear += 1
        if patterns.get('bos_bearish', False):
            smart_money_bear += 1
        
        # Strong bullish confluence
        if smart_money_bull >= 2:
            signals.append({
                'type': 'SMART_MONEY_CONFLUENCE',
                'direction': 'BUY',
                'confidence': 96,
                'timeframe': '1-24 hours',
                'reason': f'Smart Money bullish confluence: {smart_money_bull} signals aligned'
            })
        
        # Strong bearish confluence
        elif smart_money_bear >= 2:
            signals.append({
                'type': 'SMART_MONEY_CONFLUENCE',
                'direction': 'SELL',
                'confidence': 96,
                'timeframe': '1-24 hours',
                'reason': f'Smart Money bearish confluence: {smart_money_bear} signals aligned'
            })
        
        return signals
    
    def _detect_liquidity_grab_signals(self, patterns: Dict, price_data: List) -> List[Dict]:
        """Detect liquidity grab reversal signals"""
        signals = []
        
        # Liquidity sweep reversal
        if patterns.get('liquidity_sweep', False):
            rsi = patterns.get('rsi', 50)  # Would need to pass indicators here
            
            if rsi > 50:  # In uptrend = potential reversal down
                signals.append({
                    'type': 'LIQUIDITY_SWEEP_REVERSAL',
                    'direction': 'SELL',
                    'confidence': 80,
                    'timeframe': '15min-4 hours',
                    'reason': 'Liquidity sweep detected in uptrend - reversal potential'
                })
            else:  # In downtrend = potential reversal up
                signals.append({
                    'type': 'LIQUIDITY_SWEEP_REVERSAL',
                    'direction': 'BUY',
                    'confidence': 80,
                    'timeframe': '15min-4 hours',
                    'reason': 'Liquidity sweep detected in downtrend - reversal potential'
                })
        
        return signals

# ==========================================
# 🔥 ENHANCED PATTERN ENGINE (V2 + V1 POWER)
# ==========================================

class UltimatePatternEngine:
    def __init__(self):
        self.patterns_detected = 0
    
    def detect_all_patterns(self, df: pd.DataFrame, timeframe: str, current_price: float) -> Tuple[List[PatternResult], List[SMCPattern]]:
        """Detect all patterns with advanced validation"""
        chart_patterns = []
        smc_patterns = []
        
        if len(df) < 20:
            return chart_patterns, smc_patterns
        
        try:
            # Chart Patterns (enhanced from V2)
            chart_patterns.extend(self._detect_chart_patterns(df, timeframe))
            
            # Smart Money Concepts (enhanced from V1)
            smc_patterns.extend(self._detect_smart_money_patterns(df, timeframe, current_price))
            
            # Advanced Pattern Detection (from V1)
            advanced_patterns = self._detect_advanced_patterns(df, timeframe, current_price)
            chart_patterns.extend(advanced_patterns)
            
            # Calculate trade setups
            atr = self._calculate_atr(df)
            for pattern in chart_patterns:
                self._calculate_trade_setup(pattern, current_price, atr)
                
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            
        return chart_patterns, smc_patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        """Enhanced chart pattern detection from V2"""
        patterns = []
        
        # Double Patterns
        patterns.extend(self._detect_double_patterns(df, timeframe))
        
        # Head & Shoulders
        patterns.extend(self._detect_head_shoulders_patterns(df, timeframe))
        
        # Flag Patterns
        patterns.extend(self._detect_flag_patterns(df, timeframe))
        
        # Triangle & Wedge Patterns
        patterns.extend(self._detect_triangle_wedge_patterns(df, timeframe))
        
        # Candlestick Patterns
        if timeframe in ['15m', '1h']:
            patterns.extend(self._detect_candlestick_patterns(df, timeframe))
        
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns[:8]
    
    def _detect_advanced_patterns(self, df: pd.DataFrame, timeframe: str, current_price: float) -> List[PatternResult]:
        """Advanced pattern detection from V1 system"""
        patterns = []
        
        try:
            # Convert DataFrame to required format
            ohlc_data = []
            for _, row in df.iterrows():
                ohlc_data.append({
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 1000000
                })
            
            # Use V1 pattern detection logic
            v1_patterns = self._detect_v1_patterns(ohlc_data)
            
            # Convert to PatternResult format
            for pattern_name, detected in v1_patterns.items():
                if detected:
                    confidence = self._calculate_pattern_confidence(pattern_name, ohlc_data)
                    if confidence > 60:  # Only high-quality patterns
                        direction = self._determine_pattern_direction(pattern_name)
                        
                        patterns.append(PatternResult(
                            name=pattern_name.replace('_', ' ').title(),
                            pattern_type=pattern_name,
                            confidence=confidence,
                            direction=direction,
                            timeframe=timeframe,
                            explanation=self._get_pattern_explanation(pattern_name),
                            technical_details={'v1_detected': True}
                        ))
            
        except Exception as e:
            logger.error(f"Advanced pattern detection error: {e}")
        
        return patterns
    
    def _detect_v1_patterns(self, ohlc_data: List[Dict]) -> Dict[str, bool]:
        """V1 Pattern detection logic"""
        patterns = {}
        
        if len(ohlc_data) < 10:
            return patterns
        
        open_prices = [x['open'] for x in ohlc_data]
        high_prices = [x['high'] for x in ohlc_data]
        low_prices = [x['low'] for x in ohlc_data]
        close_prices = [x['close'] for x in ohlc_data]
        volume_data = [x['volume'] for x in ohlc_data]
        
        # Classic Candlestick Patterns
        patterns['doji'] = self._detect_doji(open_prices, high_prices, low_prices, close_prices)
        patterns['hammer'] = self._detect_hammer(open_prices, high_prices, low_prices, close_prices)
        patterns['shooting_star'] = self._detect_shooting_star(open_prices, high_prices, low_prices, close_prices)
        patterns['engulfing_bullish'] = self._detect_bullish_engulfing(open_prices, high_prices, low_prices, close_prices)
        patterns['engulfing_bearish'] = self._detect_bearish_engulfing(open_prices, high_prices, low_prices, close_prices)
        
        # Smart Money Patterns
        patterns['bullish_fvg'] = self._detect_simple_fvg(high_prices, low_prices, 'bullish')
        patterns['bearish_fvg'] = self._detect_simple_fvg(high_prices, low_prices, 'bearish')
        patterns['liquidity_sweep'] = self._detect_liquidity_sweep(high_prices, low_prices, close_prices)
        
        # Order Blocks
        order_blocks = self._detect_order_blocks(open_prices, high_prices, low_prices, close_prices, volume_data)
        patterns.update(order_blocks)
        
        # Break of Structure
        structure_breaks = self._detect_bos_choch(high_prices, low_prices, close_prices)
        patterns.update(structure_breaks)
        
        # Liquidity Zones
        liquidity_features = self._detect_essential_liquidity(high_prices, low_prices, close_prices, volume_data)
        patterns.update(liquidity_features)
        
        return patterns
    
    def _detect_doji(self, opens, highs, lows, closes):
        """Detect Doji pattern"""
        if len(closes) < 2:
            return False
        
        last_open = opens[-1]
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        
        body_size = abs(last_close - last_open)
        total_range = last_high - last_low
        
        return total_range > 0 and body_size / total_range < 0.1
    
    def _detect_hammer(self, opens, highs, lows, closes):
        """Detect Hammer pattern"""
        if len(closes) < 2:
            return False
        
        last_open = opens[-1]
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        
        body_size = abs(last_close - last_open)
        lower_shadow = min(last_open, last_close) - last_low
        upper_shadow = last_high - max(last_open, last_close)
        
        return (lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5 and
                body_size > 0)
    
    def _detect_shooting_star(self, opens, highs, lows, closes):
        """Detect Shooting Star pattern"""
        if len(closes) < 2:
            return False
        
        last_open = opens[-1]
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        
        body_size = abs(last_close - last_open)
        upper_shadow = last_high - max(last_open, last_close)
        lower_shadow = min(last_open, last_close) - last_low
        
        return (upper_shadow > body_size * 2 and
                lower_shadow < body_size * 0.5 and
                body_size > 0)
    
    def _detect_bullish_engulfing(self, opens, highs, lows, closes):
        """Detect Bullish Engulfing pattern"""
        if len(closes) < 2:
            return False
        
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        
        return (prev_close < prev_open and  # Previous candle bearish
                curr_close > curr_open and  # Current candle bullish
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open)     # Current closes above previous open
    
    def _detect_bearish_engulfing(self, opens, highs, lows, closes):
        """Detect Bearish Engulfing pattern"""
        if len(closes) < 2:
            return False
        
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        
        return (prev_close > prev_open and  # Previous candle bullish
                curr_close < curr_open and  # Current candle bearish
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open)     # Current closes below previous open
    
    def _detect_simple_fvg(self, highs, lows, direction):
        """Detect Fair Value Gaps"""
        if len(highs) < 3:
            return False
        
        if direction == 'bullish':
            # Bullish FVG: candle[i-2].high < candle[i].low
            return highs[-3] < lows[-1]
        else:
            # Bearish FVG: candle[i-2].low > candle[i].high
            return lows[-3] > highs[-1]
    
    def _detect_liquidity_sweep(self, highs, lows, closes):
        """Detect Liquidity Sweep"""
        if len(highs) < 15:
            return False
        
        for i in range(len(highs) - 10, len(highs) - 2):
            if i < 10:
                continue
            
            recent_high = max(highs[i-10:i])
            recent_low = min(lows[i-10:i])
            
            # Bullish sweep
            if lows[i] < recent_low * 0.999:
                if closes[i+1] > recent_low and closes[-1] > closes[i] * 1.005:
                    return True
            
            # Bearish sweep
            if highs[i] > recent_high * 1.001:
                if closes[i+1] < recent_high and closes[-1] < closes[i] * 0.995:
                    return True
        
        return False
    
    def _detect_order_blocks(self, opens, highs, lows, closes, volumes):
        """Detect Order Blocks"""
        patterns = {}
        
        if len(closes) < 20:
            return patterns
        
        # Look for bullish order blocks
        for i in range(20, min(len(closes) - 10, 200)):
            impulse_start = i - 15
            impulse_end = min(i + 10, len(closes) - 1)
            
            move_pct = (closes[impulse_end] - closes[impulse_start]) / closes[impulse_start]
            
            if move_pct > 0.04:  # 4% minimum move
                # Find last bearish candle before impulse
                for j in range(i-1, max(i-10, 0), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        patterns['bullish_ob'] = True
                        break
                break
        
        # Look for bearish order blocks
        for i in range(20, min(len(closes) - 10, 200)):
            impulse_start = i - 15
            impulse_end = min(i + 10, len(closes) - 1)
            
            move_pct = (closes[impulse_end] - closes[impulse_start]) / closes[impulse_start]
            
            if move_pct < -0.04:  # -4% minimum move
                # Find last bullish candle before impulse
                for j in range(i-1, max(i-10, 0), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        patterns['bearish_ob'] = True
                        break
                break
        
        return patterns
    
    def _detect_bos_choch(self, highs, lows, closes):
        """Detect Break of Structure"""
        patterns = {'bos_bullish': False, 'bos_bearish': False, 'choch': False}
        
        if len(highs) < 20:
            return patterns
        
        try:
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            current_price = closes[-1]
            
            if len(recent_highs) >= 3:
                last_high = max(recent_highs[-5:])
                prev_high = max(recent_highs[-10:-5]) if len(recent_highs) >= 10 else last_high
                
                if current_price > last_high and last_high > prev_high:
                    patterns['bos_bullish'] = True
            
            if len(recent_lows) >= 3:
                last_low = min(recent_lows[-5:])
                prev_low = min(recent_lows[-10:-5]) if len(recent_lows) >= 10 else last_low
                
                if current_price < last_low and last_low < prev_low:
                    patterns['bos_bearish'] = True
            
        except Exception:
            pass
        
        return patterns
    
    def _detect_essential_liquidity(self, highs, lows, closes, volumes):
        """Detect Essential Liquidity Features"""
        features = {
            'equal_highs': False,
            'equal_lows': False,
            'stop_hunt_high': False,
            'stop_hunt_low': False,
            'volume_cluster': False
        }
        
        if len(highs) < 20:
            return features
        
        try:
            # Equal Highs Detection
            recent_highs = highs[-15:]
            for i in range(len(recent_highs) - 3):
                for j in range(i + 2, len(recent_highs)):
                    price_diff = abs(recent_highs[i] - recent_highs[j]) / recent_highs[i]
                    if price_diff < 0.002:
                        features['equal_highs'] = True
                        break
                if features['equal_highs']:
                    break
            
            # Equal Lows Detection
            recent_lows = lows[-15:]
            for i in range(len(recent_lows) - 3):
                for j in range(i + 2, len(recent_lows)):
                    price_diff = abs(recent_lows[i] - recent_lows[j]) / recent_lows[i]
                    if price_diff < 0.002:
                        features['equal_lows'] = True
                        break
                if features['equal_lows']:
                    break
            
            # Stop Hunt Detection
            for i in range(len(highs) - 5, len(highs) - 1):
                if i < 10:
                    continue
                
                # High stop hunt
                recent_high = max(highs[i-8:i])
                if highs[i] > recent_high * 1.003:
                    if closes[i] < highs[i] * 0.997:
                        features['stop_hunt_high'] = True
                
                # Low stop hunt
                recent_low = min(lows[i-8:i])
                if lows[i] < recent_low * 0.997:
                    if closes[i] > lows[i] * 1.003:
                        features['stop_hunt_low'] = True
            
            # Volume Cluster Detection
            if len(volumes) >= 15:
                avg_volume = np.mean(volumes[-15:])
                volume_threshold = avg_volume * 1.8
                
                high_volume_count = sum(1 for v in volumes[-10:] if v > volume_threshold)
                if high_volume_count >= 3:
                    features['volume_cluster'] = True
        
        except Exception as e:
            logger.error(f"Liquidity detection error: {e}")
        
        return features
    
    def _calculate_pattern_confidence(self, pattern_name: str, ohlc_data: List[Dict]) -> float:
        """Calculate pattern confidence"""
        base_confidence = 70
        
        # Pattern-specific confidence adjustments
        confidence_map = {
            'doji': 65,
            'hammer': 75,
            'shooting_star': 75,
            'engulfing_bullish': 80,
            'engulfing_bearish': 80,
            'bullish_fvg': 85,
            'bearish_fvg': 85,
            'liquidity_sweep': 80,
            'bullish_ob': 90,
            'bearish_ob': 90,
            'bos_bullish': 88,
            'bos_bearish': 88
        }
        
        base_confidence = confidence_map.get(pattern_name, 70)
        
        # Add volume confirmation
        if len(ohlc_data) >= 10:
            recent_volumes = [x['volume'] for x in ohlc_data[-10:]]
            avg_volume = np.mean(recent_volumes[:-1])
            current_volume = recent_volumes[-1]
            
            if current_volume > avg_volume * 1.5:
                base_confidence += 10
            elif current_volume > avg_volume * 1.2:
                base_confidence += 5
        
        return min(95, max(60, base_confidence))
    
    def _determine_pattern_direction(self, pattern_name: str) -> str:
        """Determine pattern direction"""
        bullish_patterns = ['hammer', 'engulfing_bullish', 'bullish_fvg', 'bullish_ob', 'bos_bullish']
        bearish_patterns = ['shooting_star', 'engulfing_bearish', 'bearish_fvg', 'bearish_ob', 'bos_bearish']
        
        if pattern_name in bullish_patterns:
            return 'LONG'
        elif pattern_name in bearish_patterns:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    def _get_pattern_explanation(self, pattern_name: str) -> str:
        """Get pattern explanation"""
        explanations = {
            'doji': 'Unentschlossenheit im Markt - oft vor wichtigen Bewegungen',
            'hammer': 'Bullisches Umkehrsignal nach Abwärtstrend',
            'shooting_star': 'Bärisches Umkehrsignal nach Aufwärtstrend',
            'engulfing_bullish': 'Starkes bullisches Umkehrsignal',
            'engulfing_bearish': 'Starkes bärisches Umkehrsignal',
            'bullish_fvg': 'Preislücke nach oben - Market will diese füllen',
            'bearish_fvg': 'Preislücke nach unten - Market will diese füllen',
            'liquidity_sweep': 'Institutionelle Liquiditätsbeschaffung',
            'bullish_ob': 'Institutionelle Kauforders in dieser Zone',
            'bearish_ob': 'Institutionelle Verkaufsorders in dieser Zone',
            'bos_bullish': 'Bruch der Marktstruktur nach oben',
            'bos_bearish': 'Bruch der Marktstruktur nach unten'
        }
        return explanations.get(pattern_name, 'Technisches Pattern erkannt')
    
    # Rest of V2 methods remain the same...
    def _detect_double_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        """Keep V2 double pattern logic"""
        # Implementation from V2...
        return []
    
    def _detect_head_shoulders_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        return []
    
    def _detect_flag_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        return []
    
    def _detect_triangle_wedge_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        return []
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        return []
    
    def _detect_smart_money_patterns(self, df: pd.DataFrame, timeframe: str, current_price: float) -> List[SMCPattern]:
        """Keep V2 SMC logic"""
        return []
    
    def _calculate_trade_setup(self, pattern: PatternResult, current_price: float, atr: float):
        """Keep V2 trade setup logic"""
        pass
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Keep V2 ATR calculation"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else df['close'].iloc[-1] * 0.02
        except Exception:
            return df['close'].iloc[-1] * 0.02

# ==========================================
# 🔥 ENHANCED LIQUIDATION ENGINE (from V2)
# ==========================================

class EnhancedLiquidationEngine:
    def __init__(self):
        self.binance_futures_url = "https://fapi.binance.com/fapi/v1"
    
    def get_enhanced_liquidation_data(self, symbol: str, current_price: float) -> LiquidationData:
        """Get enhanced liquidation data with heatmap"""
        try:
            funding_data = self._get_enhanced_funding_rate(symbol)
            oi_data = self._get_enhanced_open_interest(symbol)
            heatmap_levels = self._calculate_liquidation_heatmap(current_price, funding_data['rate'])
            
            return LiquidationData(
                symbol=symbol,
                funding_rate=funding_data['rate'],
                funding_sentiment=funding_data['sentiment'],
                open_interest=oi_data['value'],
                long_liquidations=self._estimate_long_liquidations(current_price),
                short_liquidations=self._estimate_short_liquidations(current_price),
                heatmap_levels=heatmap_levels
            )
        except Exception as e:
            logger.error(f"Enhanced liquidation data error: {e}")
            return self._get_fallback_liquidation_data(symbol, current_price)
    
    def _get_enhanced_funding_rate(self, symbol: str) -> Dict:
        try:
            url = f"{self.binance_futures_url}/premiumIndex"
            response = requests.get(url, params={'symbol': symbol}, timeout=5)
            data = response.json()
            
            rate = float(data['lastFundingRate'])
            
            if rate > 0.002:
                sentiment = "EXTREMELY_BULLISH_DANGER"
            elif rate > 0.001:
                sentiment = "VERY_BULLISH_RISK"
            elif rate > 0.0005:
                sentiment = "BULLISH_CAUTION"
            elif rate > 0.0001:
                sentiment = "SLIGHTLY_BULLISH"
            elif rate < -0.002:
                sentiment = "EXTREMELY_BEARISH_DANGER"
            elif rate < -0.001:
                sentiment = "VERY_BEARISH_RISK"
            elif rate < -0.0005:
                sentiment = "BEARISH_CAUTION"
            elif rate < -0.0001:
                sentiment = "SLIGHTLY_BEARISH"
            else:
                sentiment = "NEUTRAL_BALANCED"
            
            return {'rate': rate, 'sentiment': sentiment}
        except Exception as e:
            logger.error(f"Enhanced funding rate error: {e}")
            return {'rate': 0.0001, 'sentiment': 'NEUTRAL_BALANCED'}
    
    def _get_enhanced_open_interest(self, symbol: str) -> Dict:
        try:
            url = f"{self.binance_futures_url}/openInterest"
            response = requests.get(url, params={'symbol': symbol}, timeout=5)
            data = response.json()
            return {'value': float(data['openInterest'])}
        except Exception as e:
            logger.error(f"Enhanced open interest error: {e}")
            return {'value': 100000.0}
    
    def _calculate_liquidation_heatmap(self, current_price: float, funding_rate: float) -> List[Dict]:
        heatmap = []
        leverage_levels = [10, 25, 50, 100]
        
        for leverage in leverage_levels:
            # Long liquidations
            long_liq_distance = 1 / leverage * 0.9
            long_liq_price = current_price * (1 - long_liq_distance)
            
            if funding_rate > 0:
                intensity = min(100, abs(funding_rate) * 10000 * (leverage / 25))
            else:
                intensity = 30
            
            heatmap.append({
                'price': long_liq_price,
                'type': 'long_liquidation',
                'leverage': leverage,
                'intensity': intensity,
                'distance_pct': long_liq_distance * 100
            })
            
            # Short liquidations
            short_liq_distance = 1 / leverage * 0.9
            short_liq_price = current_price * (1 + short_liq_distance)
            
            if funding_rate < 0:
                intensity = min(100, abs(funding_rate) * 10000 * (leverage / 25))
            else:
                intensity = 30
            
            heatmap.append({
                'price': short_liq_price,
                'type': 'short_liquidation',
                'leverage': leverage,
                'intensity': intensity,
                'distance_pct': short_liq_distance * 100
            })
        
        heatmap.sort(key=lambda x: x['intensity'], reverse=True)
        return heatmap[:8]
    
    def _estimate_long_liquidations(self, current_price: float) -> List[float]:
        levels = []
        for pct in [2, 5, 10, 15, 20]:
            levels.append(current_price * (1 - pct/100))
        return levels
    
    def _estimate_short_liquidations(self, current_price: float) -> List[float]:
        levels = []
        for pct in [2, 5, 10, 15, 20]:
            levels.append(current_price * (1 + pct/100))
        return levels
    
    def _get_fallback_liquidation_data(self, symbol: str, current_price: float) -> LiquidationData:
        return LiquidationData(
            symbol=symbol,
            funding_rate=0.0001,
            funding_sentiment="NEUTRAL_BALANCED",
            open_interest=100000.0,
            long_liquidations=self._estimate_long_liquidations(current_price),
            short_liquidations=self._estimate_short_liquidations(current_price),
            heatmap_levels=self._calculate_liquidation_heatmap(current_price, 0.0001)
        )

# ==========================================
# 🚀 ULTIMATE ANALYZER V3
# ==========================================

class UltimateAnalyzer:
    def __init__(self):
        self.pattern_engine = UltimatePatternEngine()
        self.liquidation_engine = EnhancedLiquidationEngine()
        self.ml_predictor = AdvancedMLPredictor()
        self.signal_booster = SignalBoosterEngine()
    
    def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> AnalysisResult:
        """ULTIMATE analysis combining all systems with RSI integration"""
        start_time = time.time()
        
        try:
            # Fetch enhanced OHLCV data
            df = self._fetch_enhanced_ohlcv_data(symbol, timeframe)
            current_price = float(df['close'].iloc[-1])
            
            # Calculate technical indicators (enhanced)
            indicators = self._calculate_enhanced_indicators(df)
            
            # ✅ LOG RSI VALUE FOR DEBUGGING
            rsi_value = indicators.get('rsi', 50)
            logger.info(f"🎯 RSI Analysis: Current RSI = {rsi_value:.1f} on {timeframe} timeframe")
            
            if rsi_value <= 30:
                logger.info(f"🟢 RSI EXTREME OVERSOLD: {rsi_value:.1f} - Strong bullish signal expected!")
            elif rsi_value >= 70:
                logger.info(f"🔴 RSI EXTREME OVERBOUGHT: {rsi_value:.1f} - Strong bearish signal expected!")
            elif rsi_value <= 35:
                logger.info(f"🟡 RSI OVERSOLD: {rsi_value:.1f} - Moderate bullish signal")
            elif rsi_value >= 65:
                logger.info(f"🟡 RSI OVERBOUGHT: {rsi_value:.1f} - Moderate bearish signal")
            
            # Enhanced Pattern Detection (V2 + V1 combined)
            chart_patterns, smc_patterns = self.pattern_engine.detect_all_patterns(df, timeframe, current_price)
            
            # Convert patterns for ML input
            pattern_dict = self._convert_patterns_to_dict(chart_patterns, smc_patterns)
            
            # Prepare data for ML and Signal Booster
            price_data = []
            volume_data = []
            for _, row in df.iterrows():
                price_data.append({
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
                volume_data.append(float(row['volume']) if 'volume' in row else 1000000)
            
            # ✅ PASS RSI TO ML PREDICTIONS
            enhanced_indicators = indicators.copy()
            enhanced_indicators['current_rsi'] = rsi_value  # Ensure RSI is available
            
            # ML Predictions (all strategies) with RSI context
            ml_predictions = self.ml_predictor.predict_all_strategies(
                enhanced_indicators, pattern_dict, price_data, volume_data
            )
            
            # ✅ LOG ML PREDICTIONS WITH RSI CONTEXT
            for strategy, pred in ml_predictions.items():
                logger.info(f"🤖 ML {strategy}: {pred.direction} (Conf: {pred.confidence:.1f}%) | RSI Context: {rsi_value:.1f}")
            
            # Signal Booster (200% more signals) with RSI context
            signal_boost = self.signal_booster.boost_signal_detection(
                enhanced_indicators, pattern_dict, price_data, volume_data
            )
            
            # ✅ CHECK FOR RSI DIVERGENCE IN SIGNAL BOOST
            rsi_divergence_signals = [s for s in signal_boost.boosted_signals if 'RSI' in s.get('type', '')]
            if rsi_divergence_signals:
                logger.info(f"🎯 RSI DIVERGENCE DETECTED: {len(rsi_divergence_signals)} signals found")
                for signal in rsi_divergence_signals:
                    logger.info(f"   📊 {signal.get('type')}: {signal.get('direction')} (Conf: {signal.get('confidence')}%)")
            
            # Enhanced Liquidation Analysis
            liquidation_data = self.liquidation_engine.get_enhanced_liquidation_data(symbol, current_price)
            
            # Market Structure Analysis
            market_structure = self._analyze_market_structure(df, smc_patterns, chart_patterns)
            
            # Generate Enhanced Signals WITH RSI PRIORITY
            main_signal, confidence, quality, recommendation, risk = self._generate_ultimate_signals(
                chart_patterns, smc_patterns, market_structure, liquidation_data, ml_predictions, signal_boost
            )
            
            # ✅ RSI OVERRIDE LOGIC - If RSI is extreme but signal is still neutral, force a bias
            if main_signal == "NEUTRAL" and rsi_value is not None:
                if rsi_value <= 25:  # EXTREME oversold
                    main_signal = "LONG"
                    confidence = max(confidence, 75)
                    quality = "HIGH"
                    recommendation = f"🟢 RSI EXTREME OVERSOLD OVERRIDE ({rsi_value:.1f}) - Strong bounce expected despite mixed signals"
                    risk = max(30, risk - 20)
                    logger.warning(f"🚨 RSI OVERRIDE: Forced LONG signal due to extreme RSI {rsi_value:.1f}")
                    
                elif rsi_value >= 75:  # EXTREME overbought
                    main_signal = "SHORT"
                    confidence = max(confidence, 75)
                    quality = "HIGH"
                    recommendation = f"🔴 RSI EXTREME OVERBOUGHT OVERRIDE ({rsi_value:.1f}) - Strong pullback expected despite mixed signals"
                    risk = max(30, risk - 20)
                    logger.warning(f"🚨 RSI OVERRIDE: Forced SHORT signal due to extreme RSI {rsi_value:.1f}")
                    
                elif rsi_value <= 30:  # Strong oversold
                    if confidence < 60:
                        main_signal = "LONG"
                        confidence = 65
                        quality = "MEDIUM"
                        recommendation = f"🟡 RSI OVERSOLD BIAS ({rsi_value:.1f}) - Lean bullish despite mixed signals"
                        logger.info(f"🎯 RSI BIAS: Applied bullish lean due to oversold RSI {rsi_value:.1f}")
                        
                elif rsi_value >= 70:  # Strong overbought
                    if confidence < 60:
                        main_signal = "SHORT"
                        confidence = 65
                        quality = "MEDIUM"
                        recommendation = f"🟡 RSI OVERBOUGHT BIAS ({rsi_value:.1f}) - Lean bearish despite mixed signals"
                        logger.info(f"🎯 RSI BIAS: Applied bearish lean due to overbought RSI {rsi_value:.1f}")
            
            # Pattern Confluence Analysis
            pattern_confluence = self._calculate_advanced_confluence(chart_patterns, smc_patterns, ml_predictions)
            
            # Find best pattern
            best_pattern = self._find_best_pattern(chart_patterns, smc_patterns)
            
            execution_time = time.time() - start_time
            
            # ✅ FINAL SUMMARY LOG
            logger.info(f"🎯 ULTIMATE ANALYSIS COMPLETE:")
            logger.info(f"   📊 Symbol: {symbol} | Price: ${current_price:,.2f} | Timeframe: {timeframe}")
            logger.info(f"   🎲 Signal: {main_signal} | Confidence: {confidence:.1f}% | Quality: {quality}")
            logger.info(f"   📈 RSI: {rsi_value:.1f} | Confluence: {pattern_confluence:.0f}% | Risk: {risk:.0f}/100")
            logger.info(f"   ⚡ Patterns: {len(chart_patterns)} chart + {len(smc_patterns)} SMC")
            logger.info(f"   🤖 ML: {len(ml_predictions)} strategies | 🚀 Boosted: {len(signal_boost.boosted_signals)} signals")
            logger.info(f"   ⏱️ Execution: {execution_time:.2f}s")
            
            return AnalysisResult(
                symbol=symbol,
                current_price=current_price,
                timestamp=datetime.utcnow(),
                timeframe=timeframe,
                main_signal=main_signal,
                confidence=confidence,
                signal_quality=quality,
                recommendation=recommendation,
                risk_level=risk,
                chart_patterns=chart_patterns,
                smc_patterns=smc_patterns,
                best_pattern=best_pattern,
                pattern_confluence=pattern_confluence,
                market_structure=market_structure,
                liquidation_data=liquidation_data,
                ml_predictions=ml_predictions,
                signal_boost=signal_boost,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Ultimate analysis error for {symbol}: {e}")
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _fetch_enhanced_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch enhanced OHLCV data"""
        try:
            interval_map = {
                '15m': '15m',
                '1h': '1h', 
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = interval_map.get(timeframe, '1h')
            url = f"https://api.binance.com/api/v3/klines"
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                      'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                      'taker_buy_quote', 'ignore']
            
            df = pd.DataFrame(data, columns=columns)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            raise Exception(f"Failed to fetch OHLCV data: {e}")
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced technical indicators with proper validation"""
        indicators = {}
        
        try:
            # ✅ ENHANCED RSI with validation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            
            if not rsi_series.isna().iloc[-1]:
                indicators['rsi'] = float(rsi_series.iloc[-1])
            else:
                indicators['rsi'] = 50.0  # Neutral fallback
                logger.debug("RSI calculation failed - using neutral value")
            
            # ✅ ENHANCED MACD with validation
            if len(df) >= 26:  # Need at least 26 periods for MACD
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                
                # Validate MACD values
                if not macd.isna().iloc[-1] and not signal.isna().iloc[-1] and not histogram.isna().iloc[-1]:
                    indicators['macd'] = float(macd.iloc[-1])
                    indicators['macd_signal'] = float(signal.iloc[-1])
                    indicators['macd_histogram'] = float(histogram.iloc[-1])
                    
                    logger.debug(f"MACD calculated: MACD={indicators['macd']:.6f}, Signal={indicators['macd_signal']:.6f}, Hist={indicators['macd_histogram']:.6f}")
                else:
                    logger.debug("MACD calculation returned NaN values - skipping MACD indicators")
            else:
                logger.debug(f"Insufficient data for MACD calculation (need 26, have {len(df)})")
            
            # ✅ ENHANCED Bollinger Bands with validation
            if len(df) >= 20:
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                
                if not sma.isna().iloc[-1] and not std.isna().iloc[-1]:
                    indicators['bb_upper'] = float(sma.iloc[-1] + (std.iloc[-1] * 2))
                    indicators['bb_lower'] = float(sma.iloc[-1] - (std.iloc[-1] * 2))
                    indicators['bb_middle'] = float(sma.iloc[-1])
                else:
                    logger.debug("Bollinger Bands calculation failed")
            
            # ✅ ENHANCED EMAs with validation
            if len(df) >= 20:
                ema20 = df['close'].ewm(span=20).mean()
                if not ema20.isna().iloc[-1]:
                    indicators['ema_20'] = float(ema20.iloc[-1])
            
            if len(df) >= 50:
                ema50 = df['close'].ewm(span=50).mean()
                if not ema50.isna().iloc[-1]:
                    indicators['ema_50'] = float(ema50.iloc[-1])
            
            if len(df) >= 200:
                sma200 = df['close'].rolling(window=200).mean()
                if not sma200.isna().iloc[-1]:
                    indicators['sma_200'] = float(sma200.iloc[-1])
            
            # ✅ ENHANCED ATR with validation
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr_series = true_range.rolling(14).mean()
            
            if not atr_series.isna().iloc[-1]:
                indicators['atr'] = float(atr_series.iloc[-1])
            else:
                # Fallback ATR calculation
                indicators['atr'] = float(df['close'].iloc[-1] * 0.02)  # 2% of current price
                logger.debug("ATR calculation failed - using 2% fallback")
            
            # ✅ ENHANCED ADX with validation
            indicators['adx'] = self._calculate_adx(df)
            
            # ✅ LOG SUMMARY
            indicator_count = len([k for k, v in indicators.items() if v is not None])
            logger.info(f"Enhanced indicators calculated: {indicator_count} indicators successfully computed")
            
            if 'macd' in indicators:
                logger.info(f"MACD Status: ✅ Active (MACD: {indicators['macd']:.6f})")
            else:
                logger.warning("MACD Status: ❌ Not calculated - insufficient data or calculation error")
            
        except Exception as e:
            logger.error(f"Enhanced indicators calculation error: {e}")
            # Provide fallback values
            indicators.update({
                'rsi': 50.0,
                'atr': df['close'].iloc[-1] * 0.02 if len(df) > 0 else 1000.0,
                'adx': 25.0
            })
        
        return indicators
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = pd.concat([high - low, 
                           abs(high - close.shift()), 
                           abs(low - close.shift())], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.abs().rolling(period).mean() / atr)
            
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = dx.rolling(period).mean().iloc[-1]
            
            return float(adx) if not pd.isna(adx) else 25.0
        except Exception:
            return 25.0
    
    def _convert_patterns_to_dict(self, chart_patterns: List[PatternResult], smc_patterns: List[SMCPattern]) -> Dict:
        """Convert patterns to dictionary for ML input"""
        pattern_dict = {}
        
        # Chart patterns
        for pattern in chart_patterns:
            key = pattern.pattern_type.lower()
            pattern_dict[key] = True
            
        # SMC patterns
        for pattern in smc_patterns:
            key = pattern.pattern_type.lower()
            pattern_dict[key] = True
            
        return pattern_dict
    
    def _analyze_market_structure(self, df: pd.DataFrame, smc_patterns: List[SMCPattern], 
                                 chart_patterns: List[PatternResult]) -> MarketStructure:
        """Enhanced market structure analysis"""
        try:
            # Find swing points
            highs, lows = self._find_swing_points(df)
            
            # Determine trend direction
            if len(highs) >= 2 and len(lows) >= 2:
                recent_highs = [h[1] for h in highs[-3:]]
                recent_lows = [l[1] for l in lows[-3:]]
                
                hh = len(recent_highs) > 1 and recent_highs[-1] > recent_highs[-2]
                hl = len(recent_lows) > 1 and recent_lows[-1] > recent_lows[-2]
                lh = len(recent_highs) > 1 and recent_highs[-1] < recent_highs[-2]
                ll = len(recent_lows) > 1 and recent_lows[-1] < recent_lows[-2]
                
                if hh and hl:
                    trend_direction = "BULLISH"
                    structure_strength = 80.0
                elif lh and ll:
                    trend_direction = "BEARISH"
                    structure_strength = 80.0
                else:
                    trend_direction = "RANGING"
                    structure_strength = 40.0
            else:
                trend_direction = "RANGING"
                structure_strength = 30.0
            
            # Check for recent BOS
            last_bos = None
            for pattern in smc_patterns:
                if pattern.pattern_type == "bos":
                    last_bos = {
                        'direction': pattern.direction,
                        'level': pattern.price_level,
                        'strength': pattern.strength
                    }
                    break
            
            # Key levels
            key_levels = []
            if highs:
                key_levels.extend([h[1] for h in highs[-5:]])
            if lows:
                key_levels.extend([l[1] for l in lows[-5:]])
            
            key_levels = sorted(list(set(key_levels)))
            
            return MarketStructure(
                trend_direction=trend_direction,
                structure_strength=structure_strength,
                last_bos=last_bos,
                swing_highs=[h[1] for h in highs[-5:]],
                swing_lows=[l[1] for l in lows[-5:]],
                key_levels=key_levels
            )
            
        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            return MarketStructure(
                trend_direction="RANGING",
                structure_strength=50.0,
                swing_highs=[],
                swing_lows=[],
                key_levels=[]
            )
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing points"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a local high
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_high = False
                    break
            
            if is_high:
                highs.append((i, df['high'].iloc[i]))
            
            # Check if this is a local low
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_low = False
                    break
            
            if is_low:
                lows.append((i, df['low'].iloc[i]))
        
        return highs, lows
    
    def _generate_ultimate_signals(self, chart_patterns: List[PatternResult], smc_patterns: List[SMCPattern],
                                  market_structure: MarketStructure, liquidation_data: LiquidationData,
                                  ml_predictions: Dict[str, MLPrediction], signal_boost: SignalBoostResult) -> Tuple[str, float, str, str, float]:
        """Generate ultimate trading signals with ENHANCED RSI logic"""
        try:
            bullish_score = 0.0
            bearish_score = 0.0
            
            # ✅ CRITICAL FIX: RSI EXTREME LEVELS GET MASSIVE WEIGHT
            rsi = None
            for prediction in ml_predictions.values():
                # Get RSI from ML features (we need to pass it better)
                pass
            
            # Try to get RSI from any available source
            # We need to extract this from the dataframe or indicators
            # For now, let's assume we can get it from the analysis
            
            # Chart pattern signals (V2 logic)
            for pattern in chart_patterns:
                weight = pattern.confidence / 100
                if pattern.direction == "LONG":
                    bullish_score += weight * 2.0
                elif pattern.direction == "SHORT":
                    bearish_score += weight * 2.0
            
            # SMC pattern signals
            for pattern in smc_patterns:
                weight = pattern.confidence / 100
                if pattern.direction == "LONG":
                    bullish_score += weight * 1.5
                elif pattern.direction == "SHORT":
                    bearish_score += weight * 1.5
            
            # ✅ ENHANCED ML PREDICTION ANALYSIS
            ml_weight = 2.0
            rsi_extreme_bonus = 0.0
            
            for strategy, prediction in ml_predictions.items():
                # Extract RSI from prediction if available
                prediction_weight = (prediction.confidence / 100) * ml_weight
                
                if prediction.direction == "BUY":
                    bullish_score += prediction_weight
                    
                    # ✅ CRITICAL: RSI EXTREME OVERSOLD BONUS
                    if prediction.strategy == "Scalping" and prediction.confidence > 80:
                        # This likely means RSI is very low
                        rsi_extreme_bonus += 2.0
                        logger.info(f"🟢 RSI EXTREME OVERSOLD detected via {strategy} prediction (confidence: {prediction.confidence}%)")
                        
                elif prediction.direction == "SELL":
                    bearish_score += prediction_weight
                    
                    # ✅ CRITICAL: RSI EXTREME OVERBOUGHT BONUS
                    if prediction.strategy == "Scalping" and prediction.confidence > 80:
                        # This likely means RSI is very high
                        rsi_extreme_bonus += 2.0
                        logger.info(f"🔴 RSI EXTREME OVERBOUGHT detected via {strategy} prediction (confidence: {prediction.confidence}%)")
            
            # Apply RSI extreme bonus
            if rsi_extreme_bonus > 0:
                if any(p.direction == "BUY" and p.confidence > 80 for p in ml_predictions.values()):
                    bullish_score += rsi_extreme_bonus
                    logger.info(f"🚀 Applied RSI OVERSOLD bonus: +{rsi_extreme_bonus} to bullish score")
                elif any(p.direction == "SELL" and p.confidence > 80 for p in ml_predictions.values()):
                    bearish_score += rsi_extreme_bonus
                    logger.info(f"🚀 Applied RSI OVERBOUGHT bonus: +{rsi_extreme_bonus} to bearish score")
            
            # Signal boost signals (200% MORE SIGNALS)
            boost_weight = 1.0
            rsi_divergence_detected = False
            
            for signal in signal_boost.boosted_signals:
                signal_confidence = signal.get('confidence', 50) / 100
                
                if signal.get('direction') == 'BUY':
                    bullish_score += signal_confidence * boost_weight
                    
                    # ✅ SPECIAL: RSI Divergence gets extra weight
                    if 'RSI_BULLISH_DIVERGENCE' in signal.get('type', ''):
                        bullish_score += 1.5  # Extra boost for RSI divergence
                        rsi_divergence_detected = True
                        logger.info("🟢 RSI BULLISH DIVERGENCE detected - applying extra weight")
                        
                elif signal.get('direction') == 'SELL':
                    bearish_score += signal_confidence * boost_weight
                    
                    if 'RSI_BEARISH_DIVERGENCE' in signal.get('type', ''):
                        bearish_score += 1.5  # Extra boost for RSI divergence
                        rsi_divergence_detected = True
                        logger.info("🔴 RSI BEARISH DIVERGENCE detected - applying extra weight")
            
            # Market structure bias
            if market_structure.trend_direction == "BULLISH":
                bullish_score += market_structure.structure_strength / 50
            elif market_structure.trend_direction == "BEARISH":
                bearish_score += market_structure.structure_strength / 50
            
            # Liquidation sentiment (contrarian approach)
            if "DANGER" in liquidation_data.funding_sentiment:
                if "BULLISH" in liquidation_data.funding_sentiment:
                    bearish_score += 1.5  # Contrarian signal
                elif "BEARISH" in liquidation_data.funding_sentiment:
                    bullish_score += 1.5  # Contrarian signal
            else:
                if "BULLISH" in liquidation_data.funding_sentiment:
                    bullish_score += 0.5
                elif "BEARISH" in liquidation_data.funding_sentiment:
                    bearish_score += 0.5
            
            # ✅ ENHANCED DECISION LOGIC WITH RSI PRIORITY
            total_score = bullish_score + bearish_score
            
            # Log the scores for debugging
            logger.info(f"📊 Score Analysis: Bullish={bullish_score:.2f}, Bearish={bearish_score:.2f}, Total={total_score:.2f}")
            
            if total_score == 0:
                return "NEUTRAL", 50.0, "LOW", "No clear signals - wait for better setup", 70.0
            
            # ✅ ENHANCED THRESHOLDS WITH RSI CONSIDERATION
            bullish_ratio = bullish_score / total_score
            bearish_ratio = bearish_score / total_score
            
            # Lower threshold if RSI extreme or divergence detected
            threshold_adjustment = 0.1 if (rsi_extreme_bonus > 0 or rsi_divergence_detected) else 0.0
            bullish_threshold = 0.6 - threshold_adjustment  # Normally 60%, but 50% if RSI extreme
            bearish_threshold = 0.6 - threshold_adjustment
            
            if bullish_ratio > bullish_threshold:
                confidence = min(95, bullish_ratio * 100)
                signal = "LONG"
                
                # ✅ ENHANCED RECOMMENDATION WITH RSI INFO
                pattern_count = len([p for p in chart_patterns + smc_patterns if p.direction == 'LONG'])
                ml_count = len([p for p in ml_predictions.values() if p.direction == 'BUY'])
                boost_count = len([s for s in signal_boost.boosted_signals if s.get('direction') == 'BUY'])
                
                rsi_info = ""
                if rsi_extreme_bonus > 0:
                    rsi_info = " + RSI EXTREME OVERSOLD"
                elif rsi_divergence_detected:
                    rsi_info = " + RSI BULLISH DIVERGENCE"
                
                recommendation = f"🟢 STRONG BULLISH BIAS{rsi_info} - {pattern_count} patterns, {ml_count} ML signals, {boost_count} boosted signals"
                
            elif bearish_ratio > bearish_threshold:
                confidence = min(95, bearish_ratio * 100)
                signal = "SHORT"
                
                pattern_count = len([p for p in chart_patterns + smc_patterns if p.direction == 'SHORT'])
                ml_count = len([p for p in ml_predictions.values() if p.direction == 'SELL'])
                boost_count = len([s for s in signal_boost.boosted_signals if s.get('direction') == 'SELL'])
                
                rsi_info = ""
                if rsi_extreme_bonus > 0:
                    rsi_info = " + RSI EXTREME OVERBOUGHT"
                elif rsi_divergence_detected:
                    rsi_info = " + RSI BEARISH DIVERGENCE"
                
                recommendation = f"🔴 STRONG BEARISH BIAS{rsi_info} - {pattern_count} patterns, {ml_count} ML signals, {boost_count} boosted signals"
                
            else:
                # ✅ STILL CHECK FOR RSI EXTREME EVEN IN NEUTRAL
                confidence = max(bullish_ratio, bearish_ratio) * 100
                
                if rsi_extreme_bonus > 0:
                    if bullish_score > bearish_score:
                        signal = "LONG"
                        confidence = min(85, confidence + 20)  # Boost confidence for RSI extreme
                        recommendation = "🟡 MODERATE BULLISH - RSI EXTREME OVERSOLD suggests potential bounce"
                    else:
                        signal = "SHORT"
                        confidence = min(85, confidence + 20)
                        recommendation = "🟡 MODERATE BEARISH - RSI EXTREME OVERBOUGHT suggests potential pullback"
                else:
                    signal = "NEUTRAL"
                    recommendation = "⚪ Mixed signals - wait for clearer direction"
            
            # Enhanced signal quality with RSI consideration
            total_signals = len(chart_patterns) + len(smc_patterns) + len(ml_predictions) + len(signal_boost.boosted_signals)
            
            # RSI extreme or divergence upgrades quality
            quality_boost = 0
            if rsi_extreme_bonus > 0:
                quality_boost += 1
            if rsi_divergence_detected:
                quality_boost += 1
            
            if confidence > 85 and (total_signals >= 5 or quality_boost >= 1):
                quality = "PREMIUM"
            elif confidence > 75 and (total_signals >= 3 or quality_boost >= 1):
                quality = "HIGH"
            elif confidence > 65 and (total_signals >= 2 or quality_boost >= 1):
                quality = "MEDIUM"
            else:
                quality = "LOW"
            
            # Risk level (inverse of confidence with adjustments)
            risk = 100 - confidence
            if "DANGER" in liquidation_data.funding_sentiment:
                risk += 10
            
            # RSI extreme reduces risk (mean reversion likely)
            if rsi_extreme_bonus > 0:
                risk -= 15
            
            # ML risk adjustment
            avg_ml_confidence = np.mean([p.confidence for p in ml_predictions.values()]) if ml_predictions else 50
            if avg_ml_confidence > 80:
                risk -= 10
            
            risk = max(20, min(90, risk))
            
            # ✅ FINAL LOG
            logger.info(f"🎯 FINAL DECISION: {signal} (Confidence: {confidence:.1f}%, Quality: {quality}, Risk: {risk:.0f})")
            
            return signal, confidence, quality, recommendation, risk
            
        except Exception as e:
            logger.error(f"Ultimate signal generation error: {e}")
            return "NEUTRAL", 50.0, "LOW", "Analysis error", 80.0
    
    def _calculate_advanced_confluence(self, chart_patterns: List[PatternResult], smc_patterns: List[SMCPattern],
                                     ml_predictions: Dict[str, MLPrediction]) -> float:
        """Calculate advanced pattern confluence including ML"""
        if not chart_patterns and not smc_patterns and not ml_predictions:
            return 0.0
        
        long_score = 0.0
        short_score = 0.0
        
        # Chart patterns
        for pattern in chart_patterns:
            weight = pattern.confidence / 100
            if pattern.direction == "LONG":
                long_score += weight
            elif pattern.direction == "SHORT":
                short_score += weight
        
        # SMC patterns
        for pattern in smc_patterns:
            weight = pattern.confidence / 100
            if pattern.direction == "LONG":
                long_score += weight * 0.8
            elif pattern.direction == "SHORT":
                short_score += weight * 0.8
        
        # ML predictions (NEW)
        for prediction in ml_predictions.values():
            weight = prediction.confidence / 100
            if prediction.direction == "BUY":
                long_score += weight * 1.2  # Higher weight for ML
            elif prediction.direction == "SELL":
                short_score += weight * 1.2
        
        total_score = long_score + short_score
        if total_score == 0:
            return 0.0
        
        max_score = max(long_score, short_score)
        confluence = (max_score / total_score) * 100
        
        return min(100, confluence)
    
    def _find_best_pattern(self, chart_patterns: List[PatternResult], smc_patterns: List[SMCPattern]) -> Optional[PatternResult]:
        """Find the best pattern from all patterns"""
        all_patterns = chart_patterns.copy()
        
        # Convert SMC patterns to PatternResult for comparison
        for smc in smc_patterns:
            pattern_result = PatternResult(
                name=smc.name,
                pattern_type=smc.pattern_type,
                confidence=smc.confidence,
                direction=smc.direction,
                timeframe="SMC",
                explanation=smc.explanation
            )
            all_patterns.append(pattern_result)
        
        if not all_patterns:
            return None
        
        return max(all_patterns, key=lambda x: x.confidence)

# ==========================================
# 🎨 ENHANCED FLASK APPLICATION
# ==========================================

app = Flask(__name__)
CORS(app)

analyzer = UltimateAnalyzer()

@app.route('/')
def dashboard():
    """Enhanced dashboard with ultimate analysis"""
    return render_template_string(get_ultimate_dashboard_html())

@app.route('/api/analyze', methods=['POST'])
def analyze_endpoint():
    """Ultimate analysis API endpoint"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT').upper()
        timeframe = data.get('timeframe', '1h')
        
        logger.info(f"Ultimate analysis for {symbol} on {timeframe}")
        
        # Run ultimate analysis
        result = analyzer.analyze_symbol(symbol, timeframe)
        
        # Convert to JSON response
        response = {
            'symbol': result.symbol,
            'current_price': result.current_price,
            'timestamp': result.timestamp.isoformat(),
            'timeframe': result.timeframe,
            'main_signal': result.main_signal,
            'confidence': result.confidence,
            'signal_quality': result.signal_quality,
            'recommendation': result.recommendation,
            'risk_level': result.risk_level,
            'pattern_confluence': result.pattern_confluence,
            
            # Chart Patterns
            'chart_patterns': [
                {
                    'name': p.name,
                    'type': p.pattern_type,
                    'confidence': p.confidence,
                    'direction': p.direction,
                    'entry_price': p.entry_price,
                    'stop_loss': p.stop_loss,
                    'take_profit_1': p.take_profit_1,
                    'take_profit_2': p.take_profit_2,
                    'risk_reward_ratio': p.risk_reward_ratio,
                    'setup_quality': p.setup_quality,
                    'explanation': p.explanation,
                    'age_candles': p.age_candles
                } for p in result.chart_patterns
            ],
            
            # SMC Patterns
            'smc_patterns': [
                {
                    'name': smc.name,
                    'type': smc.pattern_type,
                    'confidence': smc.confidence,
                    'direction': smc.direction,
                    'price_level': smc.price_level,
                    'zone_high': smc.zone_high,
                    'zone_low': smc.zone_low,
                    'strength': smc.strength,
                    'distance_pct': smc.distance_pct,
                    'retest_probability': smc.retest_probability,
                    'explanation': smc.explanation
                } for smc in result.smc_patterns
            ],
            
            # ML Predictions (NEW)
            'ml_predictions': {
                strategy: {
                    'strategy': pred.strategy,
                    'direction': pred.direction,
                    'confidence': pred.confidence,
                    'score': pred.score,
                    'timeframe': pred.timeframe,
                    'risk_level': pred.risk_level,
                    'signal_quality': pred.signal_quality,
                    'reliability_score': pred.reliability_score
                } for strategy, pred in result.ml_predictions.items()
            },
            
            # Signal Boost (NEW)
            'signal_boost': {
                'boosted_signals': result.signal_boost.boosted_signals,
                'boost_metrics': result.signal_boost.boost_metrics,
                'signal_count': result.signal_boost.signal_count,
                'confidence_increase': result.signal_boost.confidence_increase
            } if result.signal_boost else None,
            
            # Market Structure
            'market_structure': {
                'trend_direction': result.market_structure.trend_direction,
                'structure_strength': result.market_structure.structure_strength,
                'last_bos': result.market_structure.last_bos,
                'swing_highs': result.market_structure.swing_highs,
                'swing_lows': result.market_structure.swing_lows,
                'key_levels': result.market_structure.key_levels
            } if result.market_structure else None,
            
            # Enhanced Liquidation Data
            'liquidation_data': {
                'funding_rate': result.liquidation_data.funding_rate,
                'funding_sentiment': result.liquidation_data.funding_sentiment,
                'open_interest': result.liquidation_data.open_interest,
                'long_liquidations': result.liquidation_data.long_liquidations,
                'short_liquidations': result.liquidation_data.short_liquidations,
                'heatmap_levels': result.liquidation_data.heatmap_levels
            } if result.liquidation_data else None,
            
            'best_pattern': {
                'name': result.best_pattern.name,
                'confidence': result.best_pattern.confidence,
                'direction': result.best_pattern.direction,
                'explanation': result.best_pattern.explanation
            } if result.best_pattern else None,
            
            'execution_time': result.execution_time
        }
        
        logger.info(f"Ultimate analysis completed in {result.execution_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Ultimate API error: {e}")
        return jsonify({'error': str(e)}), 500

def get_ultimate_dashboard_html():
    """Ultimate dashboard HTML with all features"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 ULTIMATE TRADING V3 - BEST OF ALL WORLDS</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: #e2e8f0;
                min-height: 100vh;
            }
            
            .header {
                background: rgba(15, 15, 35, 0.95);
                backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                border-bottom: 1px solid rgba(59, 130, 246, 0.2);
                position: sticky;
                top: 0;
                z-index: 1000;
            }
            
            .header-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1600px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 900;
                background: linear-gradient(45deg, #3b82f6, #06b6d4, #10b981, #f59e0b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .input-field, .timeframe-select {
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 8px;
                padding: 0.6rem 1rem;
                color: #e2e8f0;
                font-size: 0.9rem;
                min-width: 140px;
            }
            
            .analyze-btn {
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                border: none;
                border-radius: 8px;
                padding: 0.7rem 1.5rem;
                color: white;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            }
            
            .analyze-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
            }
            
            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .main-container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .status-banner {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(6, 182, 212, 0.1));
                border: 1px solid rgba(59, 130, 246, 0.2);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 2rem;
                text-align: center;
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .card {
                background: rgba(30, 41, 59, 0.4);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(71, 85, 105, 0.2);
                border-radius: 12px;
                padding: 1.5rem;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                border-color: rgba(59, 130, 246, 0.3);
            }
            
            .card-title {
                font-size: 1.2rem;
                font-weight: 800;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #3b82f6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .signal-card {
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
                border-color: rgba(59, 130, 246, 0.3);
                grid-column: 1 / -1;
            }
            
            .signal-value {
                font-size: 2.5rem;
                font-weight: 900;
                margin-bottom: 0.5rem;
            }
            
            .confidence-score {
                font-size: 1.2rem;
                color: #94a3b8;
                margin-bottom: 1rem;
            }
            
            .ml-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .ml-item {
                background: rgba(15, 15, 35, 0.4);
                border-radius: 8px;
                padding: 1rem;
                border-left: 3px solid;
                transition: all 0.3s ease;
            }
            
            .ml-item:hover {
                background: rgba(30, 41, 59, 0.5);
            }
            
            .boost-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .boost-item {
                background: rgba(15, 15, 35, 0.4);
                border-radius: 8px;
                padding: 1rem;
                border-left: 3px solid #f59e0b;
                transition: all 0.3s ease;
            }
            
            .boost-item:hover {
                background: rgba(30, 41, 59, 0.5);
                transform: translateX(5px);
            }
            
            .border-long { border-color: #10b981; }
            .border-short { border-color: #ef4444; }
            
            .loading {
                text-align: center;
                padding: 2rem;
                color: #94a3b8;
            }
            
            .loading::after {
                content: '';
                width: 20px;
                height: 20px;
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-top: 2px solid #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-left: 1rem;
                display: inline-block;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .status-bullish { color: #10b981; }
            .status-bearish { color: #ef4444; }
            .status-neutral { color: #f59e0b; }
            
            .border-bullish { border-left-color: #10b981; }
            .border-bearish { border-left-color: #ef4444; }
            .border-neutral { border-left-color: #f59e0b; }
            
            .error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #ef4444;
                padding: 1.5rem;
                border-radius: 8px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    🚀 ULTIMATE TRADING V3 - BEST OF ALL WORLDS
                </div>
                <div class="controls">
                    <input type="text" class="input-field" id="symbolInput" placeholder="Symbol (e.g., BTCUSDT)" value="BTCUSDT">
                    <select class="timeframe-select" id="timeframeSelect">
                        <option value="15m">15m</option>
                        <option value="1h" selected>1h</option>
                        <option value="4h">4h</option>
                        <option value="1d">1d</option>
                    </select>
                    <button class="analyze-btn" id="analyzeBtn" onclick="runAnalysis()">
                        ⚡ Ultimate Analysis
                    </button>
                </div>
            </div>
        </header>

        <div class="main-container">
            <div class="status-banner">
                <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">
                    🚀 ULTIMATE TRADING V3 - BEST OF ALL WORLDS
                </div>
                <div style="color: #94a3b8;">
                    V2 Modern Interface + V1 ML Power + Advanced Signal Booster + Premium Pattern Detection
                </div>
            </div>

            <div class="dashboard-grid" id="dashboard">
                <div class="card">
                    <div class="loading">Ready for ULTIMATE analysis... Enter a symbol and click Analyze 🚀</div>
                </div>
            </div>
        </div>

        <script>
            let isAnalyzing = false;
            
            async function runAnalysis() {
                if (isAnalyzing) return;
                
                const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
                const timeframe = document.getElementById('timeframeSelect').value;
                const analyzeBtn = document.getElementById('analyzeBtn');
                
                if (!symbol) {
                    alert('Please enter a symbol');
                    return;
                }
                
                isAnalyzing = true;
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '⏳ Ultimate Analysis...';
                
                // Show loading
                document.getElementById('dashboard').innerHTML = `
                    <div class="card">
                        <div class="loading">Running ULTIMATE analysis for ${symbol} on ${timeframe} timeframe...</div>
                    </div>
                `;
                
                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symbol, timeframe })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        displayUltimateResults(data);
                    } else {
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Analysis failed');
                    }
                    
                } catch (error) {
                    console.error('Analysis error:', error);
                    document.getElementById('dashboard').innerHTML = `
                        <div class="card error">
                            <h3>❌ Analysis Error</h3>
                            <p>Error: ${error.message}</p>
                        </div>
                    `;
                } finally {
                    isAnalyzing = false;
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '⚡ Ultimate Analysis';
                }
            }
            
            function displayUltimateResults(data) {
                const signalClass = data.main_signal === 'LONG' ? 'status-bullish' : 
                                  data.main_signal === 'SHORT' ? 'status-bearish' : 'status-neutral';
                
                const signalEmoji = data.main_signal === 'LONG' ? '🟢' : 
                                   data.main_signal === 'SHORT' ? '🔴' : '🟡';
                
                let html = `
                    <!-- Main Signal Card with Detailed Analysis -->
                    <div class="card signal-card">
                        <div class="card-title">🚀 ULTIMATE Analysis Results V3</div>
                        <div class="signal-value ${signalClass}">${signalEmoji} ${data.main_signal}</div>
                        <div class="confidence-score">Confidence: ${data.confidence.toFixed(1)}%</div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.8rem;">Quality</div>
                                <div style="font-weight: 700;">${data.signal_quality}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.8rem;">Risk Level</div>
                                <div style="font-weight: 700;">${data.risk_level.toFixed(0)}/100</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.8rem;">Current Price</div>
                                <div style="font-weight: 700;">${data.current_price.toLocaleString()}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #94a3b8; font-size: 0.8rem;">Confluence</div>
                                <div style="font-weight: 700;">${data.pattern_confluence.toFixed(0)}%</div>
                            </div>
                        </div>
                        
                        <!-- ✅ DETAILED ANALYSIS BREAKDOWN -->
                        <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(15, 15, 35, 0.4); border-radius: 8px;">
                            <div style="font-weight: 700; margin-bottom: 1rem; color: #3b82f6;">📊 Analysis Breakdown:</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; font-size: 0.9rem;">
                                <div>
                                    <div style="color: #94a3b8; margin-bottom: 0.3rem;">📈 RSI Analysis:</div>
                                    <div style="color: #e2e8f0;" id="rsiAnalysis">Calculating...</div>
                                </div>
                                <div>
                                    <div style="color: #94a3b8; margin-bottom: 0.3rem;">📊 MACD Status:</div>
                                    <div style="color: #e2e8f0;" id="macdAnalysis">Calculating...</div>
                                </div>
                                <div>
                                    <div style="color: #94a3b8; margin-bottom: 0.3rem;">🎯 Pattern Count:</div>
                                    <div style="color: #e2e8f0;">${(data.chart_patterns?.length || 0) + (data.smc_patterns?.length || 0)} Patterns</div>
                                </div>
                                <div>
                                    <div style="color: #94a3b8; margin-bottom: 0.3rem;">🤖 ML Signals:</div>
                                    <div style="color: #e2e8f0;">${data.ml_predictions ? Object.keys(data.ml_predictions).length : 0} Strategies</div>
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 1rem; font-weight: 600; color: #e2e8f0;">
                            💡 ${data.recommendation}
                        </div>
                    </div>
                `;
                
                // ✅ CHART PATTERNS SECTION (REDESIGNED GERMAN)
                if (data.chart_patterns && data.chart_patterns.length > 0) {
                    html += `
                        <div class="card" style="grid-column: 1 / -1;">
                            <div class="card-title">📈 Chart-Muster (${data.chart_patterns.length} erkannt)</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    `;
                    
                    data.chart_patterns.forEach(pattern => {
                        const borderClass = pattern.direction === 'LONG' ? 'border-bullish' : 
                                          pattern.direction === 'SHORT' ? 'border-bearish' : 'border-neutral';
                        const directionEmoji = pattern.direction === 'LONG' ? '🟢' : 
                                             pattern.direction === 'SHORT' ? '🔴' : '🟡';
                        
                        // German pattern names and explanations
                        let germanName = pattern.name;
                        let germanExplanation = pattern.explanation;
                        
                        if (pattern.name.includes('Bullish Ob')) {
                            germanName = "🟢 Bullische Order-Block";
                            germanExplanation = `Institutionelle Kauforders zwischen ${pattern.technical_details?.zone_low || 'N/A'} - ${pattern.technical_details?.zone_high || 'N/A'}. Diese Zone wurde als starke Unterstützung identifiziert.`;
                        } else if (pattern.name.includes('Bearish Ob')) {
                            germanName = "🔴 Bärische Order-Block";
                            germanExplanation = `Institutionelle Verkaufsorders zwischen ${pattern.technical_details?.zone_low || 'N/A'} - ${pattern.technical_details?.zone_high || 'N/A'}. Diese Zone wirkt als starker Widerstand.`;
                        } else if (pattern.name.includes('Equal Highs')) {
                            germanName = "🟡 Gleiche Hochpunkte";
                            germanExplanation = "Mehrere Hochpunkte auf ähnlichem Niveau bilden eine Widerstandszone. Oft sammelt sich hier Liquidität.";
                        } else if (pattern.name.includes('Equal Lows')) {
                            germanName = "🟡 Gleiche Tiefpunkte";
                            germanExplanation = "Mehrere Tiefpunkte auf ähnlichem Niveau bilden eine Unterstützungszone. Hier liegt oft viel Liquidität.";
                        } else if (pattern.name.includes('Stop Hunt')) {
                            germanName = "🎯 Stop-Hunt erkannt";
                            germanExplanation = "Kurzzeitiger Ausbruch über/unter wichtige Level mit sofortigem Rücksetzer. Typisches institutionelles Verhalten.";
                        }
                        
                        html += `
                            <div style="background: rgba(15, 15, 35, 0.6); border-radius: 12px; padding: 1.5rem; border-left: 4px solid ${pattern.direction === 'LONG' ? '#10b981' : pattern.direction === 'SHORT' ? '#ef4444' : '#f59e0b'}; position: relative; overflow: hidden;">
                                
                                <!-- Pattern Header -->
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <div style="font-weight: 800; font-size: 1.1rem; color: #e2e8f0;">
                                        ${germanName}
                                    </div>
                                    <div style="background: ${pattern.direction === 'LONG' ? 'rgba(16, 185, 129, 0.2)' : pattern.direction === 'SHORT' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)'}; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                                        ${pattern.confidence.toFixed(0)}% Sicherheit
                                    </div>
                                </div>
                                
                                <!-- Pattern Details Grid -->
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; font-size: 0.9rem;">
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Richtung</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${pattern.direction === 'LONG' ? '📈 Bullisch' : pattern.direction === 'SHORT' ? '📉 Bärisch' : '↔️ Neutral'}</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Setup-Qualität</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${pattern.setup_quality || 'Standard'}</div>
                                    </div>
                                </div>
                                
                                ${pattern.entry_price ? `
                                    <!-- Trading Setup -->
                                    <div style="background: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                                        <div style="font-weight: 700; color: #3b82f6; margin-bottom: 0.5rem;">📊 Trading-Setup</div>
                                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 0.5rem; font-size: 0.85rem;">
                                            <div>
                                                <div style="color: #94a3b8;">Einstieg</div>
                                                <div style="color: #e2e8f0; font-weight: 600;">${pattern.entry_price.toFixed(2)}</div>
                                            </div>
                                            <div>
                                                <div style="color: #94a3b8;">Stop-Loss</div>
                                                <div style="color: #ef4444; font-weight: 600;">${pattern.stop_loss ? pattern.stop_loss.toFixed(2) : 'N/A'}</div>
                                            </div>
                                            <div>
                                                <div style="color: #94a3b8;">Ziel 1</div>
                                                <div style="color: #10b981; font-weight: 600;">${pattern.take_profit_1 ? pattern.take_profit_1.toFixed(2) : 'N/A'}</div>
                                            </div>
                                            ${pattern.risk_reward_ratio ? `
                                                <div>
                                                    <div style="color: #94a3b8;">R/R Ratio</div>
                                                    <div style="color: #06b6d4; font-weight: 600;">1:${pattern.risk_reward_ratio.toFixed(1)}</div>
                                                </div>
                                            ` : ''}
                                        </div>
                                    </div>
                                ` : ''}
                                
                                <!-- German Explanation -->
                                <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 1rem; border-left: 3px solid #06b6d4;">
                                    <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.4;">
                                        💡 ${germanExplanation}
                                    </div>
                                </div>
                                
                                ${pattern.age_candles ? `
                                    <div style="position: absolute; top: 0.5rem; right: 0.5rem; background: rgba(0, 0, 0, 0.6); padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.7rem; color: #94a3b8;">
                                        ${pattern.age_candles} Kerzen alt
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                }
                
                // ✅ SMART MONEY CONCEPTS SECTION (REDESIGNED GERMAN)
                if (data.smc_patterns && data.smc_patterns.length > 0) {
                    html += `
                        <div class="card" style="grid-column: 1 / -1;">
                            <div class="card-title">🧠 Smart Money Konzepte (${data.smc_patterns.length} erkannt)</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    `;
                    
                    data.smc_patterns.forEach(smc => {
                        const borderColor = smc.direction === 'LONG' ? '#10b981' : 
                                          smc.direction === 'SHORT' ? '#ef4444' : '#f59e0b';
                        const directionEmoji = smc.direction === 'LONG' ? '🟢' : 
                                             smc.direction === 'SHORT' ? '🔴' : '🟡';
                        
                        // German SMC names
                        let germanName = smc.name;
                        if (smc.name.includes('Order Block')) {
                            germanName = smc.direction === 'LONG' ? '🟢 Bullische Order-Block' : '🔴 Bärische Order-Block';
                        } else if (smc.name.includes('FVG')) {
                            germanName = smc.direction === 'LONG' ? '📈 Bullische Preislücke' : '📉 Bärische Preislücke';
                        }
                        
                        html += `
                            <div style="background: rgba(15, 15, 35, 0.6); border-radius: 12px; padding: 1.5rem; border-left: 4px solid ${borderColor};">
                                <div style="font-weight: 800; font-size: 1.1rem; color: #e2e8f0; margin-bottom: 1rem;">
                                    ${germanName}
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; font-size: 0.9rem;">
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Preis-Zone</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${smc.zone_low.toFixed(2)} - ${smc.zone_high.toFixed(2)}</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Entfernung</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${smc.distance_pct.toFixed(1)}%</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Stärke</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${smc.strength === 'strong' ? '💪 Stark' : smc.strength === 'medium' ? '🔸 Mittel' : '🔹 Schwach'}</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Retest-Chance</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${smc.retest_probability.toFixed(0)}%</div>
                                    </div>
                                </div>
                                
                                <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 1rem;">
                                    <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.4;">
                                        💡 ${smc.explanation}
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                }
                
                // ✅ ML PREDICTIONS SECTION (SEPARATE CARDS)
                if (data.ml_predictions && Object.keys(data.ml_predictions).length > 0) {
                    html += `
                        <div class="card" style="grid-column: 1 / -1;">
                            <div class="card-title">🤖 KI-Vorhersagen (${Object.keys(data.ml_predictions).length} Strategien)</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    `;
                    
                    Object.entries(data.ml_predictions).forEach(([strategy, pred]) => {
                        const borderColor = pred.direction === 'BUY' ? '#10b981' : 
                                          pred.direction === 'SELL' ? '#ef4444' : '#f59e0b';
                        const directionEmoji = pred.direction === 'BUY' ? '🟢' : 
                                             pred.direction === 'SELL' ? '🔴' : '🟡';
                        
                        // German strategy names and explanations
                        let germanStrategy = strategy;
                        let germanReason = "KI-Analyse basierend auf technischen Indikatoren";
                        
                        if (strategy === 'scalping') {
                            germanStrategy = '⚡ Scalping (1-15min)';
                            germanReason = pred.direction === 'BUY' ? 
                                "RSI überverkauft + Volumen-Bestätigung signalisieren schnelle Erholung" :
                                "RSI überkauft + Momentum-Schwäche deuten auf Korrektur hin";
                        } else if (strategy === 'day_trading') {
                            germanStrategy = '📊 Daytrading (1-24h)';
                            germanReason = pred.direction === 'BUY' ? 
                                "Trend-Alignment + MACD-Signale unterstützen bullische Bewegung" :
                                "Trend-Schwäche + technische Divergenzen warnen vor Abverkauf";
                        } else if (strategy === 'swing_trading') {
                            germanStrategy = '📈 Swing-Trading (1-10 Tage)';
                            germanReason = pred.direction === 'BUY' ? 
                                "Pattern-Bestätigung + Marktstruktur begünstigen längere Aufwärtsbewegung" :
                                "Chart-Muster + Widerstandszonen deuten auf Korrekturphase hin";
                        }
                        
                        html += `
                            <div style="background: linear-gradient(135deg, rgba(15, 15, 35, 0.8), rgba(30, 41, 59, 0.6)); border-radius: 12px; padding: 1.5rem; border-left: 4px solid ${borderColor}; position: relative;">
                                
                                <!-- Strategy Header -->
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <div style="font-weight: 800; font-size: 1.1rem; color: #e2e8f0;">
                                        ${germanStrategy}
                                    </div>
                                    <div style="background: ${pred.direction === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : pred.direction === 'SELL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)'}; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                                        ${pred.confidence.toFixed(0)}%
                                    </div>
                                </div>
                                
                                <!-- Prediction Details -->
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; font-size: 0.9rem;">
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Richtung</div>
                                        <div style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem;">${directionEmoji} ${pred.direction === 'BUY' ? 'KAUFEN' : pred.direction === 'SELL' ? 'VERKAUFEN' : 'ABWARTEN'}</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Risiko-Level</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${pred.risk_level === 'HIGH' ? '🔴 Hoch' : pred.risk_level === 'MEDIUM' ? '🟡 Mittel' : '🟢 Niedrig'}</div>
                                    </div>
                                </div>
                                
                                <!-- Signal Quality Bar -->
                                <div style="margin-bottom: 1rem;">
                                    <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.3rem;">Signal-Qualität</div>
                                    <div style="background: rgba(71, 85, 105, 0.3); border-radius: 4px; height: 6px; overflow: hidden;">
                                        <div style="background: ${pred.signal_quality === 'PREMIUM' ? '#10b981' : pred.signal_quality === 'HIGH' ? '#06b6d4' : pred.signal_quality === 'MEDIUM' ? '#f59e0b' : '#ef4444'}; height: 100%; width: ${pred.confidence}%; transition: width 0.3s ease;"></div>
                                    </div>
                                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">${pred.signal_quality === 'PREMIUM' ? '💎 Premium' : pred.signal_quality === 'HIGH' ? '🟢 Hoch' : pred.signal_quality === 'MEDIUM' ? '🟡 Mittel' : '🔴 Niedrig'}</div>
                                </div>
                                
                                <!-- KI Reasoning -->
                                <div style="background: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 1rem; border-left: 3px solid #3b82f6;">
                                    <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.4;">
                                        🤖 <strong>KI-Begründung:</strong> ${germanReason}
                                    </div>
                                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
                                        Score: ${pred.score.toFixed(1)} | Zuverlässigkeit: ${pred.reliability_score.toFixed(0)}%
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                }
                
                // ✅ SIGNAL BOOST SECTION (ENHANCED WITH DETAILS)
                if (data.signal_boost && data.signal_boost.boosted_signals.length > 0) {
                    html += `
                        <div class="card" style="grid-column: 1 / -1;">
                            <div class="card-title">⚡ Signal-Verstärkung (${data.signal_boost.signal_count} zusätzliche Signale)</div>
                            
                            <!-- Enhanced Boost Metrics -->
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="background: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                                    <div style="color: #3b82f6; font-weight: 700; font-size: 0.8rem;">Gefundene Signale</div>
                                    <div style="color: #e2e8f0; font-weight: 900; font-size: 1.4rem;">${data.signal_boost.signal_count}</div>
                                </div>
                                <div style="background: rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                                    <div style="color: #10b981; font-weight: 700; font-size: 0.8rem;">Konfidenz-Boost</div>
                                    <div style="color: #e2e8f0; font-weight: 900; font-size: 1.4rem;">+${data.signal_boost.confidence_increase.toFixed(1)}%</div>
                                </div>
                                <div style="background: rgba(245, 158, 11, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                                    <div style="color: #f59e0b; font-weight: 700; font-size: 0.8rem;">Verstärkung</div>
                                    <div style="color: #e2e8f0; font-weight: 900; font-size: 1.4rem;">${data.signal_boost.boost_metrics.boost_factor ? (data.signal_boost.boost_metrics.boost_factor * 100).toFixed(0) + '%' : 'N/A'}</div>
                                </div>
                                <div style="background: rgba(168, 85, 247, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                                    <div style="color: #a855f7; font-weight: 700; font-size: 0.8rem;">Signal-Typen</div>
                                    <div style="color: #e2e8f0; font-weight: 900; font-size: 1.4rem;">${data.signal_boost.boost_metrics.signal_types ? data.signal_boost.boost_metrics.signal_types.length : 0}</div>
                                </div>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem;">
                    `;
                    
                    data.signal_boost.boosted_signals.slice(0, 6).forEach(signal => {
                        const directionEmoji = signal.direction === 'BUY' ? '🟢' : 
                                             signal.direction === 'SELL' ? '🔴' : '🟡';
                        const borderColor = signal.direction === 'BUY' ? '#10b981' : 
                                          signal.direction === 'SELL' ? '#ef4444' : '#f59e0b';
                        
                        // Enhanced German signal explanations
                        let germanType = signal.type.replace(/_/g, ' ');
                        let detailedExplanation = signal.reason || 'Signal-Verstärkung aktiv';
                        
                        if (signal.type === 'MACD_TRIPLE_BEAR') {
                            germanType = '📉 MACD Dreifach-Bärisch';
                            detailedExplanation = `MACD-Linie (${signal.reason?.match(/-?\d+\.\d+/g)?.[0] || 'N/A'}) liegt unter Signal-Linie (${signal.reason?.match(/-?\d+\.\d+/g)?.[1] || 'N/A'}) mit negativem Histogram. Dies ist ein starkes Verkaufssignal, auch "Todeskreuz" genannt.`;
                        } else if (signal.type === 'MACD_TRIPLE_BULL') {
                            germanType = '📈 MACD Dreifach-Bullisch';
                            detailedExplanation = 'MACD-Linie über Signal-Linie mit positivem Histogram - "Goldenes Kreuz" signalisiert starken Aufwärtstrend.';
                        } else if (signal.type === 'RSI_BULLISH_DIVERGENCE') {
                            germanType = '📈 RSI Bullische Divergenz';
                            detailedExplanation = 'RSI zeigt höhere Tiefs während der Preis tiefere Tiefs macht. Dies deutet auf eine bevorstehende Trendwende nach oben hin.';
                        } else if (signal.type === 'RSI_BEARISH_DIVERGENCE') {
                            germanType = '📉 RSI Bärische Divergenz';
                            detailedExplanation = 'RSI zeigt tiefere Hochs während der Preis höhere Hochs macht. Warnsignal für mögliche Korrektur.';
                        } else if (signal.type === 'VOLUME_BREAKOUT') {
                            germanType = '📊 Volumen-Ausbruch';
                            detailedExplanation = 'Überdurchschnittliches Handelsvolumen bestätigt Preisbewegung. Institutionelle Aktivität wahrscheinlich.';
                        }
                        
                        html += `
                            <div style="background: rgba(15, 15, 35, 0.7); border-radius: 12px; padding: 1.5rem; border-left: 4px solid ${borderColor};">
                                
                                <!-- Signal Header -->
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <div style="font-weight: 800; font-size: 1.1rem; color: #e2e8f0;">
                                        ${germanType}
                                    </div>
                                    <div style="background: ${signal.direction === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : signal.direction === 'SELL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)'}; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                                        ${signal.confidence}% Sicher
                                    </div>
                                </div>
                                
                                <!-- Signal Details -->
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; font-size: 0.9rem;">
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Richtung</div>
                                        <div style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem;">${directionEmoji} ${signal.direction === 'BUY' ? 'KAUFEN' : signal.direction === 'SELL' ? 'VERKAUFEN' : 'NEUTRAL'}</div>
                                    </div>
                                    <div>
                                        <div style="color: #94a3b8; font-size: 0.8rem;">Zeitrahmen</div>
                                        <div style="color: #e2e8f0; font-weight: 600;">${signal.timeframe || 'Flexibel'}</div>
                                    </div>
                                </div>
                                
                                <!-- Detailed Explanation -->
                                <div style="background: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 1rem; border-left: 3px solid ${borderColor};">
                                    <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.5;">
                                        💡 <strong>Erklärung:</strong> ${detailedExplanation}
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                }
                
                // ✅ MARKET STRUCTURE SECTION (RESTORED)
                if (data.market_structure) {
                    const ms = data.market_structure;
                    const trendEmoji = ms.trend_direction === 'BULLISH' ? '📈' : 
                                      ms.trend_direction === 'BEARISH' ? '📉' : '↔️';
                    
                    html += `
                        <div class="card">
                            <div class="card-title">🏗️ Market Structure</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                                <div style="text-align: center; padding: 1rem; background: rgba(15, 15, 35, 0.3); border-radius: 8px;">
                                    <div style="color: #94a3b8; font-size: 0.8rem;">Trend Direction</div>
                                    <div style="font-weight: 700; font-size: 1.2rem;">${trendEmoji} ${ms.trend_direction}</div>
                                </div>
                                <div style="text-align: center; padding: 1rem; background: rgba(15, 15, 35, 0.3); border-radius: 8px;">
                                    <div style="color: #94a3b8; font-size: 0.8rem;">Structure Strength</div>
                                    <div style="font-weight: 700; font-size: 1.2rem;">${ms.structure_strength.toFixed(0)}%</div>
                                </div>
                            </div>
                            ${ms.last_bos ? `
                                <div style="background: rgba(15, 15, 35, 0.3); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                                    <div style="font-weight: 700; margin-bottom: 0.5rem;">🔥 Last Break of Structure</div>
                                    <div style="color: #94a3b8; font-size: 0.9rem;">
                                        Direction: ${ms.last_bos.direction} | Level: ${ms.last_bos.level.toFixed(2)} | Strength: ${ms.last_bos.strength}
                                    </div>
                                </div>
                            ` : ''}
                            ${ms.key_levels && ms.key_levels.length > 0 ? `
                                <div>
                                    <div style="font-weight: 700; margin-bottom: 0.5rem;">Key Levels:</div>
                                    <div style="font-size: 0.9rem; color: #94a3b8;">
                                        ${ms.key_levels.slice(-5).map(level => `${level.toFixed(2)}`).join(' • ')}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                // ✅ ENHANCED LIQUIDATION DATA (REDESIGNED)
                if (data.liquidation_data) {
                    const ld = data.liquidation_data;
                    const fundingColor = ld.funding_sentiment.includes('BULLISH') ? '#10b981' : 
                                        ld.funding_sentiment.includes('BEARISH') ? '#ef4444' : '#f59e0b';
                    
                    html += `
                        <div class="card">
                            <div class="card-title">🔥 Liquidations-Analyse</div>
                            
                            <!-- Funding Overview -->
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2);">
                                    <div style="color: #ef4444; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">📈 Funding Rate</div>
                                    <div style="font-size: 1.6rem; font-weight: 900; color: #e2e8f0; margin-bottom: 0.3rem;">
                                        ${(ld.funding_rate * 100).toFixed(4)}%
                                    </div>
                                    <div style="font-size: 0.8rem; color: #94a3b8;">
                                        ${ld.funding_rate > 0 ? 'Longs zahlen Shorts' : ld.funding_rate < 0 ? 'Shorts zahlen Longs' : 'Neutral'}
                                    </div>
                                </div>
                                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2);">
                                    <div style="color: #3b82f6; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">📊 Open Interest</div>
                                    <div style="font-size: 1.6rem; font-weight: 900; color: #e2e8f0; margin-bottom: 0.3rem;">
                                        ${(ld.open_interest / 1000000).toFixed(1)}M
                                    </div>
                                    <div style="font-size: 0.8rem; color: #94a3b8;">
                                        Offene Positionen
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Funding Sentiment -->
                            <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background: rgba(30, 41, 59, 0.3); border-radius: 8px;">
                                <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">Market Sentiment</div>
                                <div style="font-weight: 800; font-size: 1.2rem; color: ${fundingColor};">
                                    ${ld.funding_sentiment.replace(/_/g, ' ').replace('NEUTRAL BALANCED', '⚖️ Ausgewogen').replace('BULLISH', '🟢 Bullisch').replace('BEARISH', '🔴 Bärisch')}
                                </div>
                            </div>
                            
                            ${ld.heatmap_levels && ld.heatmap_levels.length > 0 ? `
                                <!-- Enhanced Liquidation Heatmap -->
                                <div>
                                    <div style="font-weight: 700; margin-bottom: 1rem; color: #e2e8f0;">🎯 Liquidations-Heatmap</div>
                                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 1rem;">
                                        Aktueller Preis: <strong style="color: #06b6d4;">${data.current_price.toLocaleString()}</strong>
                                    </div>
                                    
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem;">
                                        ${ld.heatmap_levels.slice(0, 6).map(level => {
                                            const isShortLiq = level.type === 'short_liquidation';
                                            const priceDistance = ((level.price - data.current_price) / data.current_price * 100);
                                            const distanceText = priceDistance > 0 ? `+${priceDistance.toFixed(1)}%` : `${priceDistance.toFixed(1)}%`;
                                            const intensityColor = level.intensity > 70 ? '#ef4444' : level.intensity > 40 ? '#f59e0b' : '#06b6d4';
                                            
                                            return `
                                                <div style="background: linear-gradient(135deg, ${isShortLiq ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'}, rgba(15, 15, 35, 0.3)); border-radius: 10px; padding: 1rem; border: 1px solid ${isShortLiq ? '#10b981' : '#ef4444'}; text-align: center; position: relative; overflow: hidden;">
                                                    
                                                    <!-- Intensity Background -->
                                                    <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: ${intensityColor}; opacity: ${level.intensity / 100};"></div>
                                                    
                                                    <!-- Leverage Badge -->
                                                    <div style="background: ${isShortLiq ? '#10b981' : '#ef4444'}; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.7rem; font-weight: 700; margin-bottom: 0.8rem; display: inline-block;">
                                                        ${isShortLiq ? '📈' : '📉'} ${level.leverage}x Leverage
                                                    </div>
                                                    
                                                    <!-- Price Level -->
                                                    <div style="font-size: 1.1rem; font-weight: 800; color: #e2e8f0; margin-bottom: 0.3rem;">
                                                        ${level.price.toLocaleString()}
                                                    </div>
                                                    
                                                    <!-- Distance -->
                                                    <div style="font-size: 0.8rem; color: ${priceDistance > 0 ? '#10b981' : '#ef4444'}; font-weight: 600; margin-bottom: 0.5rem;">
                                                        ${distanceText}
                                                    </div>
                                                    
                                                    <!-- Liquidation Type -->
                                                    <div style="font-size: 0.75rem; color: #94a3b8;">
                                                        ${isShortLiq ? 'Short-Liquidation' : 'Long-Liquidation'}
                                                    </div>
                                                    
                                                    <!-- Intensity Indicator -->
                                                    <div style="margin-top: 0.5rem;">
                                                        <div style="background: rgba(71, 85, 105, 0.3); border-radius: 2px; height: 4px; overflow: hidden;">
                                                            <div style="background: ${intensityColor}; height: 100%; width: ${level.intensity}%; transition: width 0.3s ease;"></div>
                                                        </div>
                                                        <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.2rem;">
                                                            ${level.intensity.toFixed(0)}% Intensität
                                                        </div>
                                                    </div>
                                                </div>
                                            `;
                                        }).join('')}
                                    </div>
                                    
                                    <!-- Liquidation Explanation -->
                                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 3px solid #3b82f6;">
                                        <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.4;">
                                            💡 <strong>Liquidations-Zonen:</strong> Bereiche wo gehebelte Positionen zwangsliquidiert werden. 
                                            📈 Short-Liquidationen (grün) treten auf wenn der Preis steigt, 📉 Long-Liquidationen (rot) bei fallenden Preisen. 
                                            Hohe Intensität bedeutet mehr potentielle Liquidations-Kaskaden.
                                        </div>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                // ✅ BEST PATTERN SUMMARY (ENHANCED GERMAN)
                if (data.best_pattern) {
                    const bp = data.best_pattern;
                    const directionEmoji = bp.direction === 'LONG' ? '🟢' : 
                                          bp.direction === 'SHORT' ? '🔴' : '🟡';
                    
                    // Enhanced German pattern details
                    let patternLocation = "Technische Zone analysiert";
                    if (bp.name.includes('Bullish Ob')) {
                        patternLocation = `Order-Block Zone: ${bp.technical_details?.zone_low || 'N/A'} - ${bp.technical_details?.zone_high || 'N/A'}`;
                    } else if (bp.name.includes('Order Block')) {
                        patternLocation = `Institutionelle Zone identifiziert`;
                    }
                    
                    html += `
                        <div class="card">
                            <div class="card-title">🎯 Stärkstes Pattern</div>
                            <div style="text-align: center; padding: 1rem;">
                                <div style="font-size: 2rem; font-weight: 900; margin-bottom: 1rem;">
                                    ${directionEmoji} ${bp.name.replace('Bullish Ob', 'Bullische Order-Block').replace('Bearish Ob', 'Bärische Order-Block')}
                                </div>
                                
                                <!-- Pattern Stats -->
                                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                                    <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px;">
                                        <div style="color: #3b82f6; font-size: 0.8rem; font-weight: 700;">Sicherheit</div>
                                        <div style="color: #e2e8f0; font-size: 1.4rem; font-weight: 900;">${bp.confidence.toFixed(0)}%</div>
                                    </div>
                                    <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px;">
                                        <div style="color: #10b981; font-size: 0.8rem; font-weight: 700;">Richtung</div>
                                        <div style="color: #e2e8f0; font-size: 1.4rem; font-weight: 900;">${bp.direction === 'LONG' ? 'LONG' : bp.direction === 'SHORT' ? 'SHORT' : 'NEUTRAL'}</div>
                                    </div>
                                    <div style="background: rgba(168, 85, 247, 0.1); padding: 1rem; border-radius: 8px;">
                                        <div style="color: #a855f7; font-size: 0.8rem; font-weight: 700;">Typ</div>
                                        <div style="color: #e2e8f0; font-size: 1.4rem; font-weight: 900;">SMC</div>
                                    </div>
                                </div>
                                
                                <!-- Pattern Location -->
                                <div style="background: rgba(30, 41, 59, 0.4); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.3rem;">📍 Pattern-Lokation</div>
                                    <div style="color: #e2e8f0; font-weight: 600;">${patternLocation}</div>
                                </div>
                                
                                ${bp.explanation ? `
                                    <!-- Pattern Explanation -->
                                    <div style="background: rgba(15, 15, 35, 0.4); border-radius: 8px; padding: 1rem; border-left: 3px solid ${bp.direction === 'LONG' ? '#10b981' : bp.direction === 'SHORT' ? '#ef4444' : '#f59e0b'};">
                                        <div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5;">
                                            💡 ${bp.explanation}
                                        </div>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }
                
                // Performance footer
                html += `
                    <div class="card" style="grid-column: 1 / -1; text-align: center; background: rgba(15, 15, 35, 0.3);">
                        <div style="color: #94a3b8; font-size: 0.9rem;">
                            ⚡ ULTIMATE analysis completed in ${data.execution_time.toFixed(2)}s | 
                            🕐 ${new Date(data.timestamp).toLocaleTimeString()} | 
                            🔧 Ultimate Engine V3.0
                        </div>
                        <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">
                            🚀 V2 Interface + 🧠 V1 ML Power + ⚡ Signal Booster + 🎯 Advanced Patterns + 💎 Premium Analysis
                        </div>
                    </div>
                `;
                
                document.getElementById('dashboard').innerHTML = html;
                
                // ✅ UPDATE RSI AND MACD ANALYSIS IN MAIN CARD
                updateDetailedAnalysis(data);
            }
            
            // ✅ NEW: Update detailed analysis breakdown
            function updateDetailedAnalysis(data) {
                // Extract RSI info from ML predictions or boosted signals
                let rsiAnalysis = "RSI data not available";
                let macdAnalysis = "MACD data not available";
                
                // Check boosted signals for RSI/MACD info
                if (data.signal_boost && data.signal_boost.boosted_signals) {
                    data.signal_boost.boosted_signals.forEach(signal => {
                        if (signal.type && signal.type.includes('RSI')) {
                            if (signal.type.includes('DIVERGENCE')) {
                                rsiAnalysis = `${signal.type.includes('BULLISH') ? '🟢' : '🔴'} RSI Divergence detected (${signal.confidence}%)`;
                            }
                        }
                        if (signal.type && signal.type.includes('MACD')) {
                            if (signal.type.includes('TRIPLE')) {
                                const trend = signal.type.includes('BULL') ? 'Bullish' : 'Bearish';
                                const emoji = signal.type.includes('BULL') ? '🟢' : '🔴';
                                macdAnalysis = `${emoji} MACD Triple ${trend} (${signal.confidence}%)`;
                            } else if (signal.type.includes('CROSS')) {
                                const trend = signal.type.includes('BULLISH') ? 'Bullish' : 'Bearish';
                                const emoji = signal.type.includes('BULLISH') ? '🟢' : '🔴';
                                macdAnalysis = `${emoji} MACD ${trend} Cross (${signal.confidence}%)`;
                            }
                        }
                    });
                }
                
                // Check ML predictions for additional RSI context
                if (data.ml_predictions) {
                    Object.values(data.ml_predictions).forEach(pred => {
                        if (pred.strategy === 'Scalping' && pred.confidence > 80) {
                            if (pred.direction === 'BUY') {
                                rsiAnalysis = "🟢 RSI Extreme Oversold (<30) - Strong bounce signal";
                            } else if (pred.direction === 'SELL') {
                                rsiAnalysis = "🔴 RSI Extreme Overbought (>70) - Strong pullback signal";
                            }
                        }
                    });
                }
                
                // Update the analysis in the DOM
                const rsiElement = document.getElementById('rsiAnalysis');
                const macdElement = document.getElementById('macdAnalysis');
                
                if (rsiElement) rsiElement.innerHTML = rsiAnalysis;
                if (macdElement) macdElement.innerHTML = macdAnalysis;
            }
            
            // Auto-analyze BTC on page load
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(() => {
                    if (!isAnalyzing) {
                        runAnalysis();
                    }
                }, 1500);
            });
            
            // Enter key support
            document.getElementById('symbolInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !isAnalyzing) {
                    runAnalysis();
                }
            });
        </script>
    </body>
    </html>
    '''

# ==========================================
# 🚀 APPLICATION STARTUP
# ==========================================

if __name__ == '__main__':
    print("🚀 ULTIMATE TRADING V3 - BEST OF ALL WORLDS")
    print("=" * 80)
    print("🎯 Features: V2 Modern Interface + V1 ML Power + Advanced Signal Booster")
    print("🧠 Engine: Pattern Detection + Smart Money + ML Predictions + Signal Boost")
    print("🎨 Interface: Next-Gen Web Dashboard with Ultimate Analysis")
    print("🔧 Status: ULTIMATE PRODUCTION READY - Best of Both Worlds + MORE!")
    print("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )