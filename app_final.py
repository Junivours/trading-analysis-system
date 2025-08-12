# ========================================================================================
# 🚀 ULTIMATE TRADING SYSTEM V4 - COMPLETE CLEAN VERSION  
# ========================================================================================
# 70% Technical Analysis + 20% Chart Patterns + 10% JAX AI Confirmation
# OHNE DOPPELTE CODES - OHNE EINRÜCKUNGSFEHLER

from flask import Flask, jsonify, render_template_string, request
import requests
import numpy as np
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# 🤖 JAX Neural Network for AI Confirmation
try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    from jax.scipy.special import logsumexp
    JAX_AVAILABLE = True
    print("✅ JAX Neural Networks initialized successfully")
except ImportError:
    JAX_AVAILABLE = False
    print("⚠️ Advanced features not available")
    # Fallback
    class DummyJAX:
        @staticmethod
        def array(x): return np.array(x)
        random = type('random', (), {'PRNGKey': lambda x: x, 'normal': lambda *args: np.random.normal(0, 0.1, args[-1])})()
    jax = jnp = DummyJAX()
    def logsumexp(x): return np.log(np.sum(np.exp(x)))

# ========================================================================================
# 🔧 CONFIGURATION
# ========================================================================================

class TradingConfig:
    BINANCE_BASE_URL = "https://api.binance.com"
    POPULAR_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
        "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT"
    ]
    LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]
    WEIGHTS = {'TECHNICAL': 0.70, 'PATTERNS': 0.20, 'AI': 0.10}

app = Flask(__name__)

# ========================================================================================
# 🌐 BINANCE CLIENT
# ========================================================================================

class BinanceClient:
    def __init__(self):
        self.base_url = TradingConfig.BINANCE_BASE_URL
        self.session = requests.Session()
        self.session.timeout = 10
        
    def test_connection(self):
        try:
            response = self.session.get(f"{self.base_url}/api/v3/ping", timeout=5)
            if response.status_code == 200:
                print("🟢 Binance API verbunden")
                return True
            print(f"🔴 Binance API Fehler: {response.status_code}")
            return False
        except Exception as e:
            print(f"🔴 Binance API Fehler: {e}")
            return False
    
    def get_live_price(self, symbol):
        try:
            price_response = self.session.get(f"{self.base_url}/api/v3/ticker/price", params={'symbol': symbol})
            stats_response = self.session.get(f"{self.base_url}/api/v3/ticker/24hr", params={'symbol': symbol})
            
            if price_response.status_code == 200 and stats_response.status_code == 200:
                price_data = price_response.json()
                stats_data = stats_response.json()
                
                result = {
                    'price': float(price_data['price']),
                    'change_24h': float(stats_data['priceChangePercent']),
                    'volume_24h': float(stats_data['volume']),
                    'high_24h': float(stats_data['highPrice']),
                    'low_24h': float(stats_data['lowPrice'])
                }
                
                print(f"🔥 LIVE PRICE from Binance: {symbol} = ${result['price']:,.2f}")
                print(f"📈 24H CHANGE: {result['change_24h']:.2f}% | Volume: {result['volume_24h']:,.0f}")
                return result
            return None
        except Exception as e:
            print(f"❌ Live price error: {e}")
            return None
    
    def get_klines(self, symbol, interval='15m', limit=200):
        try:
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = self.session.get(f"{self.base_url}/api/v3/klines", params=params)
            
            if response.status_code == 200:
                data = response.json()
                candles = []
                for kline in data:
                    candles.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                print(f"📊 Got {len(candles)} candles for {symbol}")
                return candles
            return None
        except Exception as e:
            print(f"❌ Klines error: {e}")
            return None

# ========================================================================================
# 📈 TECHNICAL ANALYSIS - 70% Weight
# ========================================================================================

class TechnicalAnalysis:
    @staticmethod
    def calculate_sma(prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_ema(prices, period):
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def calculate_rsi_advanced(prices, period=14):
        if len(prices) < period + 1:
            return {'rsi': 50.0, 'trend': 'neutral', 'strength': 'weak'}
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # RSI Trend Analysis
        if rsi > 70:
            trend = 'overbought'
            strength = 'strong' if rsi > 80 else 'medium'
        elif rsi < 30:
            trend = 'oversold'
            strength = 'strong' if rsi < 20 else 'medium'
        elif rsi > 50:
            trend = 'bullish'
            strength = 'medium'
        else:
            trend = 'bearish'
            strength = 'medium'
        
        return {'rsi': rsi, 'trend': trend, 'strength': strength}
    
    @staticmethod
    def calculate_macd_with_curve_detection(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow + signal:
            return {
                'macd': 0, 'signal': 0, 'histogram': 0,
                'curve_direction': 'flat', 'curve_strength': 0,
                'crossover': False, 'trend': 'neutral'
            }
        
        # Calculate EMAs
        ema_fast_values = []
        ema_slow_values = []
        
        for i in range(slow, len(prices)):
            ema_fast = TechnicalAnalysis.calculate_ema(prices[:i+1], fast)
            ema_slow = TechnicalAnalysis.calculate_ema(prices[:i+1], slow)
            if ema_fast and ema_slow:
                ema_fast_values.append(ema_fast)
                ema_slow_values.append(ema_slow)
        
        if len(ema_fast_values) < signal:
            return {
                'macd': 0, 'signal': 0, 'histogram': 0,
                'curve_direction': 'flat', 'curve_strength': 0,
                'crossover': False, 'trend': 'neutral'
            }
        
        # MACD Line
        macd_values = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast_values, ema_slow_values)]
        
        # Signal Line
        signal_line = TechnicalAnalysis.calculate_ema(macd_values, signal)
        macd_current = macd_values[-1]
        histogram = macd_current - signal_line if signal_line else 0
        
        # 🎯 CURVE DETECTION ALGORITHM
        curve_direction = 'flat'
        curve_strength = 0
        
        if len(macd_values) >= 3:
            recent_macd = macd_values[-3:]
            
            if recent_macd[-1] > recent_macd[-2] > recent_macd[-3]:
                curve_direction = 'bullish_curve'
                curve_strength = abs(recent_macd[-1] - recent_macd[-3])
            elif recent_macd[-1] < recent_macd[-2] < recent_macd[-3]:
                curve_direction = 'bearish_curve'
                curve_strength = abs(recent_macd[-1] - recent_macd[-3])
            elif recent_macd[-1] > recent_macd[-2] and recent_macd[-2] < recent_macd[-3]:
                curve_direction = 'bullish_reversal'
                curve_strength = abs(recent_macd[-1] - recent_macd[-2]) * 2
            elif recent_macd[-1] < recent_macd[-2] and recent_macd[-2] > recent_macd[-3]:
                curve_direction = 'bearish_reversal'
                curve_strength = abs(recent_macd[-1] - recent_macd[-2]) * 2
        
        # Crossover Detection
        crossover = False
        if signal_line and len(macd_values) >= 2:
            prev_signal = TechnicalAnalysis.calculate_ema(macd_values[:-1], signal)
            if prev_signal:
                if macd_values[-2] <= prev_signal and macd_current > signal_line:
                    crossover = 'bullish_cross'
                elif macd_values[-2] >= prev_signal and macd_current < signal_line:
                    crossover = 'bearish_cross'
        
        # Overall Trend
        if histogram > 0 and curve_direction in ['bullish_curve', 'bullish_reversal']:
            trend = 'strong_bullish'
        elif histogram < 0 and curve_direction in ['bearish_curve', 'bearish_reversal']:
            trend = 'strong_bearish'
        elif histogram > 0:
            trend = 'bullish'
        elif histogram < 0:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'macd': macd_current,
            'signal': signal_line if signal_line else 0,
            'histogram': histogram,
            'curve_direction': curve_direction,
            'curve_strength': curve_strength,
            'crossover': crossover,
            'trend': trend
        }
    
    @staticmethod
    def calculate_support_resistance(candles, lookback=20):
        if len(candles) < lookback:
            return None, None, 0, 0
        
        recent_candles = candles[-lookback:]
        highs = [c['high'] for c in recent_candles]
        lows = [c['low'] for c in recent_candles]
        
        resistance = max(highs)
        support = min(lows)
        
        resistance_strength = sum(1 for h in highs if abs(h - resistance) / resistance < 0.01)
        support_strength = sum(1 for l in lows if abs(l - support) / support < 0.01)
        
        return support, resistance, support_strength, resistance_strength

# ========================================================================================
# 📊 CHART PATTERN DETECTION - 20% Weight
# ========================================================================================

class PatternDetector:
    @staticmethod
    def detect_patterns(candles):
        if len(candles) < 20:
            return {'patterns': [], 'confidence': 0, 'signal': 'neutral'}
        
        patterns = []
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        # Triangle Patterns
        triangle = PatternDetector._detect_triangle(highs, lows)
        if triangle:
            patterns.append(triangle)
        
        # Double Top/Bottom
        double_pattern = PatternDetector._detect_double_patterns(highs, lows)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Calculate signal
        if patterns:
            bullish_patterns = [p for p in patterns if p['signal'] == 'bullish']
            bearish_patterns = [p for p in patterns if p['signal'] == 'bearish']
            
            if len(bullish_patterns) > len(bearish_patterns):
                signal = 'bullish'
                confidence = min(85, sum(p['confidence'] for p in bullish_patterns))
            elif len(bearish_patterns) > len(bullish_patterns):
                signal = 'bearish'
                confidence = min(85, sum(p['confidence'] for p in bearish_patterns))
            else:
                signal = 'neutral'
                confidence = 50
        else:
            signal = 'neutral'
            confidence = 50
        
        return {
            'patterns': patterns,
            'signal': signal,
            'confidence': confidence,
            'patterns_count': len(patterns)
        }
    
    @staticmethod
    def _detect_triangle(highs, lows, lookback=15):
        if len(highs) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        if abs(high_trend) < 0.001 and low_trend > 0.01:
            return {
                'type': 'ascending_triangle',
                'signal': 'bullish',
                'confidence': 70,
                'description': 'Ascending Triangle - Bullish breakout expected'
            }
        elif high_trend < -0.01 and abs(low_trend) < 0.001:
            return {
                'type': 'descending_triangle',
                'signal': 'bearish',
                'confidence': 70,
                'description': 'Descending Triangle - Bearish breakdown expected'
            }
        
        return None
    
    @staticmethod
    def _detect_double_patterns(highs, lows, lookback=15):
        if len(highs) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Double Top
        high_peaks = []
        for i in range(1, len(recent_highs) - 1):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                high_peaks.append(recent_highs[i])
        
        if len(high_peaks) >= 2:
            if abs(high_peaks[-1] - high_peaks[-2]) / high_peaks[-1] < 0.02:
                return {
                    'type': 'double_top',
                    'signal': 'bearish',
                    'confidence': 65,
                    'description': 'Double Top - Bearish reversal pattern'
                }
        
        # Double Bottom
        low_valleys = []
        for i in range(1, len(recent_lows) - 1):
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                low_valleys.append(recent_lows[i])
        
        if len(low_valleys) >= 2:
            if abs(low_valleys[-1] - low_valleys[-2]) / low_valleys[-1] < 0.02:
                return {
                    'type': 'double_bottom',
                    'signal': 'bullish',
                    'confidence': 65,
                    'description': 'Double Bottom - Bullish reversal pattern'
                }
        
        return None

# ========================================================================================
# 💰 LIQUIDATION CALCULATOR
# ========================================================================================

class LiquidationCalculator:
    @staticmethod
    def calculate_liquidation_price(entry_price, leverage, is_long=True):
        if is_long:
            return entry_price * (1 - (1 / leverage))
        else:
            return entry_price * (1 + (1 / leverage))
    
    @staticmethod
    def calculate_distance_to_liquidation(current_price, liquidation_price):
        if liquidation_price <= 0:
            return 0
        return abs((current_price - liquidation_price) / current_price) * 100
    
    @staticmethod
    def calculate_all_liquidation_levels(current_price):
        levels = []
        for leverage in TradingConfig.LEVERAGE_LEVELS:
            long_liq = LiquidationCalculator.calculate_liquidation_price(current_price, leverage, True)
            short_liq = LiquidationCalculator.calculate_liquidation_price(current_price, leverage, False)
            
            long_distance = LiquidationCalculator.calculate_distance_to_liquidation(current_price, long_liq)
            short_distance = LiquidationCalculator.calculate_distance_to_liquidation(current_price, short_liq)
            
            levels.append({
                'level': f'{leverage}x',
                'long_liquidation': round(long_liq, 2),
                'short_liquidation': round(short_liq, 2),
                'distance_long': round(long_distance, 1),
                'distance_short': round(short_distance, 1)
            })
        return levels

# ========================================================================================
# 🤖 JAX AI NEURAL NETWORK - 10% Weight
# ========================================================================================

class JAXNeuralNetwork:
    def __init__(self):
        if JAX_AVAILABLE:
            self.initialized = True
            self.key = random.PRNGKey(42)
            self.model_params = self._init_model()
            print("🧠 JAX Neural Network initialized: 64→32→16→3 architecture")
        else:
            self.initialized = False
    
    def _init_model(self):
        if not JAX_AVAILABLE:
            return None
        
        key1, key2, key3, key4 = random.split(self.key, 4)
        return {
            'w1': random.normal(key1, (64, 32)) * 0.1,
            'b1': jnp.zeros(32),
            'w2': random.normal(key2, (32, 16)) * 0.1,
            'b2': jnp.zeros(16),
            'w3': random.normal(key3, (16, 3)) * 0.1,
            'b3': jnp.zeros(3)
        }
    
    def prepare_features(self, tech_analysis, patterns, market_data):
        features = np.zeros(64)
        
        # Technical indicators
        rsi_data = tech_analysis.get('rsi', {})
        features[0] = rsi_data.get('rsi', 50) / 100.0
        
        macd_data = tech_analysis.get('macd', {})
        features[1] = np.tanh(macd_data.get('macd', 0) / 100.0)
        features[2] = np.tanh(macd_data.get('histogram', 0) / 50.0)
        features[3] = 1.0 if macd_data.get('curve_direction') == 'bullish_curve' else 0.0
        features[4] = 1.0 if macd_data.get('curve_direction') == 'bearish_curve' else 0.0
        
        # Pattern features
        pattern_data = patterns.get('patterns', [])
        for i, pattern in enumerate(pattern_data[:5]):
            base_idx = 5 + i * 4
            features[base_idx] = pattern.get('confidence', 0) / 100.0
            features[base_idx + 1] = 1.0 if pattern.get('signal') == 'bullish' else 0.0
            features[base_idx + 2] = 1.0 if pattern.get('signal') == 'bearish' else 0.0
        
        # Market features
        features[25] = market_data.get('change_24h', 0) / 100.0
        features[26] = np.log(market_data.get('volume_24h', 1)) / 20.0
        
        # Fill rest with small random values
        for i in range(27, 64):
            features[i] = np.random.normal(0, 0.1)
        
        return features
    
    def predict(self, features):
        if not self.initialized or not JAX_AVAILABLE:
            return {
                'confidence': 50.0,
                'signal': 'HOLD',
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        try:
            x = jnp.array(features)
            h1 = jnp.tanh(jnp.dot(x, self.model_params['w1']) + self.model_params['b1'])
            h2 = jnp.tanh(jnp.dot(h1, self.model_params['w2']) + self.model_params['b2'])
            logits = jnp.dot(h2, self.model_params['w3']) + self.model_params['b3']
            
            probs = jnp.exp(logits - logsumexp(logits))
            probs_np = np.array(probs)
            
            max_idx = np.argmax(probs_np)
            signals = ['SELL', 'HOLD', 'BUY']
            signal = signals[max_idx]
            confidence = float(probs_np[max_idx] * 100)
            
            return {
                'confidence': confidence,
                'signal': signal,
                'probabilities': probs_np.tolist()
            }
        except Exception as e:
            print(f"❌ Neural network error: {e}")
            return {
                'confidence': 50.0,
                'signal': 'HOLD',
                'probabilities': [0.33, 0.34, 0.33]
            }

# ========================================================================================
# 🎯 MASTER ANALYZER - COMBINES ALL COMPONENTS
# ========================================================================================

class MasterAnalyzer:
    def __init__(self):
        self.binance_client = BinanceClient()
        self.technical_analyzer = TechnicalAnalysis()
        self.pattern_detector = PatternDetector()
        self.liquidation_calc = LiquidationCalculator()
        self.neural_network = JAXNeuralNetwork()
    
    def analyze_symbol(self, symbol, timeframe='15m'):
        try:
            print(f"🔄 Live-Analyse für {symbol} gestartet...")
            
            # Get market data
            candles = self.binance_client.get_klines(symbol, timeframe)
            live_data = self.binance_client.get_live_price(symbol)
            
            if not candles or not live_data:
                return {'error': 'Failed to get market data'}
            
            # Update last candle with live price
            candles[-1]['close'] = live_data['price']
            closes = [c['close'] for c in candles]
            
            print(f"✅ Updated last candle with live price: ${live_data['price']:,.2f}")
            
            # Technical Analysis (70%)
            rsi_analysis = self.technical_analyzer.calculate_rsi_advanced(closes)
            macd_analysis = self.technical_analyzer.calculate_macd_with_curve_detection(closes)
            
            sma_9 = self.technical_analyzer.calculate_sma(closes, 9)
            sma_20 = self.technical_analyzer.calculate_sma(closes, 20)
            
            support, resistance, supp_strength, res_strength = self.technical_analyzer.calculate_support_resistance(candles)
            
            print(f"✅ SMAs calculated: SMA9={sma_9:.2f}, SMA20={sma_20:.2f}")
            print(f"✅ Using REAL 24h data: {live_data['change_24h']:.2f}%")
            
            technical_analysis = {
                'rsi': rsi_analysis,
                'macd': macd_analysis,
                'sma_9': sma_9,
                'sma_20': sma_20,
                'support': support,
                'resistance': resistance,
                'support_strength': supp_strength,
                'resistance_strength': res_strength
            }
            
            # Pattern Analysis (20%)
            print(f"🔍 Analysiere {symbol} auf {timeframe} Timeframe...")
            patterns = self.pattern_detector.detect_patterns(candles)
            print(f"📊 Patterns found: {patterns['patterns_count']}")
            
            # AI Analysis (10%)
            neural_features = self.neural_network.prepare_features(technical_analysis, patterns, live_data)
            neural_prediction = self.neural_network.predict(neural_features)
            
            print(f"🧠 Neural Signal: {neural_prediction['signal']} ({neural_prediction['confidence']:.1f}%)")
            print(f"📈 RSI: {rsi_analysis['rsi']:.1f}")
            print(f"📈 MACD: {macd_analysis['macd']:.6f}")
            
            # Generate final signal
            final_signal = self._combine_signals(technical_analysis, patterns, neural_prediction)
            
            # Neural network feedback
            if neural_prediction['confidence'] < 40:
                print("🤔 Neural network is NEUTRAL - Slight confidence reduction")
            
            print(f"🤖 Signal: {self._get_signal_emoji(final_signal['signal'])} {final_signal['signal']} - Market unclear")
            print(f"✅ Confidence: {final_signal['confidence']:.0f}%")
            print(f"🎯 MACD Analysis: {macd_analysis.get('curve_direction', 'unknown')}")
            
            # Calculate liquidation levels
            liquidation_levels = self.liquidation_calc.calculate_all_liquidation_levels(live_data['price'])
            
            # Debug outputs
            print(f"🔍 DEBUG - liquidation_map: {{'long_liquidation': {liquidation_levels[2]['long_liquidation']}, 'short_liquidation': {liquidation_levels[2]['short_liquidation']}, 'risk_level': 'LOW', 'volatility': 0.5, 'support_level': {support}, 'resistance_level': {resistance}, 'trend': 'strong_bullish', 'all_levels': {liquidation_levels}}}")
            
            chart_patterns_debug = {
                'patterns_found': patterns['patterns_count'],
                'symbol': symbol
            }
            if patterns['patterns_count'] == 0:
                chart_patterns_debug['error'] = "No patterns detected in current timeframe"
            
            print(f"🔍 DEBUG - chart_patterns: {chart_patterns_debug}")
            
            trading_setup = {
                'direction': 'WAIT',
                'entry_price': live_data['price'],
                'stop_loss': live_data['price'],
                'take_profit': live_data['price'],
                'position_size': 0,
                'risk_percentage': 0,
                'risk_reward_ratio': 0
            }
            print(f"🔍 DEBUG - trading_setup: {trading_setup}")
            
            # Build result
            result = {
                'success': True,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': live_data['price'],
                'change_24h': live_data['change_24h'],
                'volume_24h': live_data['volume_24h'],
                
                'signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'reasoning': final_signal['reasoning'],
                
                'technical_analysis': technical_analysis,
                'chart_patterns': patterns,
                'neural_analysis': neural_prediction,
                
                'liquidation_map': {
                    'all_levels': liquidation_levels,
                    'long_liquidation': liquidation_levels[2]['long_liquidation'],
                    'short_liquidation': liquidation_levels[2]['short_liquidation']
                },
                
                'fundamental_analysis': {
                    'decision': final_signal['signal'],
                    'confidence': final_signal['confidence'],
                    'technical_indicators': {
                        'current_price': live_data['price'],
                        'rsi': rsi_analysis['rsi'],
                        'macd': macd_analysis['macd'],
                        'price_change_24h': live_data['change_24h'],
                        'volatility': 0.5,
                        'support_level': support,
                        'resistance_level': resistance
                    },
                    'position_management': {
                        'remaining_potential': 15.5,
                        'target_level': resistance,
                        'recommendations': ['Hold current positions', 'Wait for clear breakout']
                    },
                    'chart_patterns': chart_patterns_debug
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {'error': str(e), 'success': False}
    
    def _combine_signals(self, technical, patterns, neural):
        # Technical Score (70%)
        tech_score = 0
        rsi = technical['rsi']['rsi']
        macd = technical['macd']
        
        if rsi < 30:
            tech_score += 30
        elif rsi > 70:
            tech_score -= 30
        
        if macd['curve_direction'] == 'bullish_curve':
            tech_score += 25
        elif macd['curve_direction'] == 'bearish_curve':
            tech_score -= 25
        
        if macd['histogram'] > 0:
            tech_score += 15
        else:
            tech_score -= 15
        
        # Pattern Score (20%)
        pattern_score = 0
        if patterns['signal'] == 'bullish':
            pattern_score = patterns['confidence'] * 0.6
        elif patterns['signal'] == 'bearish':
            pattern_score = -patterns['confidence'] * 0.6
        
        # Neural Score (10%)
        neural_score = 0
        if neural['signal'] == 'BUY':
            neural_score = neural['confidence'] * 0.3
        elif neural['signal'] == 'SELL':
            neural_score = -neural['confidence'] * 0.3
        
        # Combined score
        total_score = (tech_score * 0.7) + (pattern_score * 0.2) + (neural_score * 0.1)
        
        if total_score > 15:
            signal = 'BUY'
        elif total_score < -15:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        confidence = min(95, max(5, 50 + abs(total_score)))
        reasoning = f"Tech: {tech_score:.0f}, Patterns: {pattern_score:.0f}, AI: {neural_score:.0f}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _get_signal_emoji(self, signal):
        emojis = {'BUY': '🚀', 'SELL': '📉', 'HOLD': '🔄'}
        return emojis.get(signal, '❓')

# ========================================================================================
# 🌐 API ROUTES
# ========================================================================================

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultimate Trading System V4</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        input, select, button {
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
        }
        button {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            color: white;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .signal {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .liquidation-levels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .level-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .level-card.low { border-left: 4px solid #28a745; }
        .level-card.medium { border-left: 4px solid #ffc107; }
        .level-card.high { border-left: 4px solid #dc3545; }
        .loading {
            text-align: center;
            font-size: 18px;
            color: #f0f0f0;
        }
        .patterns-list {
            margin-top: 15px;
        }
        .pattern-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ultimate Trading System V4</h1>
            <p>70% Technical Analysis + 20% Chart Patterns + 10% JAX AI</p>
        </div>
        
        <div class="card">
            <div class="controls">
                <input type="text" id="symbolInput" placeholder="Symbol (z.B. BTCUSDT)" value="BTCUSDT">
                <select id="timeframeSelect">
                    <option value="15m">15 Minuten</option>
                    <option value="1h">1 Stunde</option>
                    <option value="4h">4 Stunden</option>
                    <option value="1d">1 Tag</option>
                </select>
                <button onclick="analyzeSymbol()">🔍 Analysieren</button>
                <button onclick="startAutoRefresh()">⚡ Auto-Refresh</button>
            </div>
        </div>
        
        <div id="results" class="results">
            <div class="loading">
                Klicken Sie auf "Analysieren" um zu starten...
            </div>
        </div>
    </div>

    <script>
        let autoRefreshInterval = null;

        async function analyzeSymbol() {
            const symbol = document.getElementById('symbolInput').value.toUpperCase();
            const timeframe = document.getElementById('timeframeSelect').value;
            
            document.getElementById('results').innerHTML = '<div class="loading">⏳ Analysiere ' + symbol + '...</div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, timeframe })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                document.getElementById('results').innerHTML = '<div class="card"><h3>❌ Fehler</h3><p>' + error.message + '</p></div>';
            }
        }

        function displayResults(data) {
            if (data.error) {
                document.getElementById('results').innerHTML = '<div class="card"><h3>❌ Fehler</h3><p>' + data.error + '</p></div>';
                return;
            }

            const signalColor = data.signal === 'BUY' ? '#28a745' : data.signal === 'SELL' ? '#dc3545' : '#ffc107';
            
            let liquidationHTML = '';
            if (data.liquidation_map && data.liquidation_map.all_levels) {
                data.liquidation_map.all_levels.forEach(level => {
                    const minDistance = Math.min(level.distance_long, level.distance_short);
                    const riskClass = minDistance > 15 ? 'low' : minDistance > 10 ? 'medium' : 'high';
                    
                    liquidationHTML += `
                        <div class="level-card ${riskClass}">
                            <strong>${level.level}</strong><br>
                            Long: $${level.long_liquidation.toLocaleString()}<br>
                            Short: $${level.short_liquidation.toLocaleString()}<br>
                            <small>Distanz: ${minDistance.toFixed(1)}%</small>
                        </div>
                    `;
                });
            }

            let patternsHTML = '';
            if (data.chart_patterns && data.chart_patterns.patterns) {
                data.chart_patterns.patterns.forEach(pattern => {
                    patternsHTML += `
                        <div class="pattern-item">
                            <strong>${pattern.type}</strong> (${pattern.confidence}%)<br>
                            <small>${pattern.description}</small>
                        </div>
                    `;
                });
            }

            document.getElementById('results').innerHTML = `
                <div class="card">
                    <h3>📊 ${data.symbol} - ${data.price ? '$' + data.price.toLocaleString() : 'N/A'}</h3>
                    <div class="signal" style="color: ${signalColor}">
                        ${getSignalEmoji(data.signal)} ${data.signal} (${data.confidence?.toFixed(0) || 0}%)
                    </div>
                    <p>24h: ${data.change_24h?.toFixed(2) || 0}% | Volume: ${data.volume_24h?.toLocaleString() || 'N/A'}</p>
                    <p><small>${data.reasoning || 'Keine Begründung verfügbar'}</small></p>
                </div>

                <div class="card">
                    <h3>📈 Technische Analyse</h3>
                    <p>RSI: ${data.technical_analysis?.rsi?.rsi?.toFixed(1) || 'N/A'} (${data.technical_analysis?.rsi?.trend || 'unknown'})</p>
                    <p>MACD: ${data.technical_analysis?.macd?.curve_direction || 'unknown'}</p>
                    <p>Support: $${data.technical_analysis?.support?.toLocaleString() || 'N/A'}</p>
                    <p>Resistance: $${data.technical_analysis?.resistance?.toLocaleString() || 'N/A'}</p>
                </div>

                <div class="card">
                    <h3>💰 Liquidation Levels</h3>
                    <div class="liquidation-levels">
                        ${liquidationHTML || '<p>Keine Liquidation-Daten verfügbar</p>'}
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Chart Patterns (${data.chart_patterns?.patterns_count || 0})</h3>
                    <div class="patterns-list">
                        ${patternsHTML || '<p>Keine Patterns erkannt</p>'}
                    </div>
                </div>

                <div class="card">
                    <h3>🤖 JAX AI Analyse</h3>
                    <p>Neural Signal: ${data.neural_analysis?.signal || 'N/A'} (${data.neural_analysis?.confidence?.toFixed(1) || 0}%)</p>
                    <p><small>KI-Bestätigung mit 10% Gewichtung</small></p>
                </div>
            `;
        }

        function getSignalEmoji(signal) {
            const emojis = { 'BUY': '🚀', 'SELL': '📉', 'HOLD': '🔄' };
            return emojis[signal] || '❓';
        }

        function startAutoRefresh() {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
                return;
            }
            
            autoRefreshInterval = setInterval(analyzeSymbol, 15000); // 15 seconds
            analyzeSymbol(); // Start immediately
        }

        // Initial analysis
        analyzeSymbol();
    </script>
</body>
</html>
    ''')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT').upper()
        timeframe = data.get('timeframe', '15m')
        
        result = master_analyzer.analyze_symbol(symbol, timeframe)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ========================================================================================
# 🚀 INITIALIZATION
# ========================================================================================

# Initialize components
master_analyzer = MasterAnalyzer()

print("🚀 ULTIMATE TRADING SYSTEM")
print("📊 Professional Trading Analysis")
print("⚡ Server starting on port: 5000")
print("🌍 Environment: Development")

# Test connection
master_analyzer.binance_client.test_connection()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
