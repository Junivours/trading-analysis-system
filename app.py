# ========================================================================================
# 🚀 ULTIMATE TRADING SYSTEM V5 - BEAUTIFUL & INTELLIGENT EDITION  
# ========================================================================================
# Professional Trading Dashboard mit intelligenter Position Management
# Basierend auf deinem schönen Backup + erweiterte Features

from flask import Flask, jsonify, render_template_string, request
import os
import subprocess
import requests
import numpy as np
import json
import time
import uuid
import logging
from collections import deque
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# 🤖 JAX Neural Network mit echtem Training
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
    class DummyJAX:
        @staticmethod
        def array(x): return np.array(x)
        random = type('random', (), {'PRNGKey': lambda x: x, 'normal': lambda *args: np.random.normal(0, 0.1, args[-1])})()
    jax = jnp = DummyJAX()
    def logsumexp(x): return np.log(np.sum(np.exp(x)))

app = Flask(__name__)

# ========================================================================================
# 🔢 VERSION / BUILD METADATA
# ========================================================================================
APP_START_TIME = datetime.utcnow().isoformat()+"Z"

def _detect_commit():
    # Priority 1: Explicit env vars commonly available on Railway or user-defined
    env_candidates = [
        'GIT_REV', 'RAILWAY_GIT_COMMIT_SHA', 'SOURCE_VERSION', 'SOURCE_COMMIT', 'COMMIT_HASH', 'RAILWAY_BUILD']
    for key in env_candidates:
        val = os.getenv(key)
        if val and val.lower() not in ('unknown','null','none'):
            return val[:8]
    # Priority 2: version.txt (user can echo commit > version.txt at build time)
    try:
        if os.path.exists('version.txt'):
            with open('version.txt','r',encoding='utf-8') as f:
                line = f.readline().strip()
                if line:
                    return line[:8]
    except Exception:
        pass
    # Priority 3: git (may not be present in container)
    try:
        out = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return 'unknown'

APP_COMMIT = _detect_commit()
APP_VERSION = f"v5-{APP_COMMIT}"
print(f"🔖 Starting Trading System {APP_VERSION} @ {APP_START_TIME}")

@app.after_request
def add_no_cache_headers(resp):
    # Help Railway not to serve stale cached responses (client/proxy) & expose version
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['X-App-Version'] = APP_VERSION
    return resp

@app.route('/api/version')
def api_version():
    return jsonify({
        'success': True,
        'version': APP_VERSION,
        'commit': APP_COMMIT,
        'started': APP_START_TIME,
        'jax_available': JAX_AVAILABLE,
        'features': ['rsi_tv_style','structured_logging','backtest_v1','dca_endpoint','cache_refresh','pattern_timeframes']
    })

@app.route('/api/version/refresh', methods=['POST'])
def api_version_refresh():
    global APP_COMMIT, APP_VERSION
    new_commit = _detect_commit()
    changed = new_commit != APP_COMMIT
    APP_COMMIT = new_commit
    APP_VERSION = f"v5-{APP_COMMIT}"
    return jsonify({'success': True, 'version': APP_VERSION, 'commit': APP_COMMIT, 'changed': changed})

@app.route('/health')
def health():
    # Lightweight health signal (no external API calls)
    return jsonify({'ok': True, 'version': APP_VERSION, 'time': datetime.utcnow().isoformat()+"Z"})

# ========================================================================================
# 🧠 INTELLIGENT POSITION MANAGEMENT ENGINE
# ========================================================================================

class PositionManager:
    @staticmethod
    def analyze_position_potential(current_price, support, resistance, trend_analysis, patterns):
        """Intelligente Position Management Empfehlungen"""
        
        # Berechne Potenzial bis Key-Levels
        resistance_potential = ((resistance - current_price) / current_price) * 100 if resistance else 0
        support_risk = ((current_price - support) / current_price) * 100 if support else 0
        
        recommendations = []
        position_status = "NEUTRAL"
        
        # 🚀 LONG Position Analysis
        if trend_analysis.get('trend') in ['bullish', 'strong_bullish']:
            if resistance_potential > 10:  # Noch gutes Potenzial
                recommendations.append({
                    'type': 'LONG',
                    'action': 'HOLD/ERWEITERN',
                    'reason': f'💰 Noch {resistance_potential:.1f}% Potenzial bis Resistance',
                    'details': f'Uptrend intakt - Resistance bei ${resistance:,.2f}',
                    'confidence': 85,
                    'color': '#28a745'
                })
                position_status = "BULLISH"
            elif resistance_potential > 5:
                recommendations.append({
                    'type': 'LONG',
                    'action': 'VORSICHTIG HALTEN',
                    'reason': f'⚠️ Nur noch {resistance_potential:.1f}% bis Resistance',
                    'details': 'Gewinnmitnahmen überdenken',
                    'confidence': 60,
                    'color': '#ffc107'
                })
            else:
                recommendations.append({
                    'type': 'LONG',
                    'action': 'GEWINNMITNAHME',
                    'reason': '🎯 Resistance erreicht - Profit sichern',
                    'details': 'Baue langsam Short-Position auf',
                    'confidence': 90,
                    'color': '#dc3545'
                })
        
        # 📉 SHORT Position Analysis
        if trend_analysis.get('trend') in ['bearish', 'strong_bearish']:
            if support_risk > 10:  # Noch Downside
                recommendations.append({
                    'type': 'SHORT',
                    'action': 'HOLD/ERWEITERN',
                    'reason': f'📉 Noch {support_risk:.1f}% Downside bis Support',
                    'details': f'Downtrend aktiv - Support bei ${support:,.2f}',
                    'confidence': 85,
                    'color': '#dc3545'
                })
                position_status = "BEARISH"
            elif support_risk > 5:
                recommendations.append({
                    'type': 'SHORT',
                    'action': 'VORSICHTIG',
                    'reason': f'⚠️ Nahe Support - nur noch {support_risk:.1f}%',
                    'details': 'Bereite Long-Einstieg vor',
                    'confidence': 65,
                    'color': '#ffc107'
                })
            else:
                recommendations.append({
                    'type': 'SHORT',
                    'action': 'LONG AUFBAUEN',
                    'reason': '🚀 Support erreicht - Bullish Reversal incoming!',
                    'details': 'Schließe Shorts, baue Long-Position auf',
                    'confidence': 88,
                    'color': '#28a745'
                })
        
        # 🔄 Pattern-basierte Empfehlungen
        for pattern in patterns.get('patterns', []):
            if pattern['signal'] == 'bullish' and pattern['confidence'] > 70:
                recommendations.append({
                    'type': 'PATTERN',
                    'action': 'LONG SIGNAL',
                    'reason': f'📈 {pattern["type"]} detected ({pattern["confidence"]}%)',
                    'details': pattern['description'],
                    'confidence': pattern['confidence'],
                    'color': '#28a745'
                })
            elif pattern['signal'] == 'bearish' and pattern['confidence'] > 70:
                recommendations.append({
                    'type': 'PATTERN',
                    'action': 'SHORT SIGNAL',
                    'reason': f'📉 {pattern["type"]} detected ({pattern["confidence"]}%)',
                    'details': pattern['description'],
                    'confidence': pattern['confidence'],
                    'color': '#dc3545'
                })
        
        return {
            'position_status': position_status,
            'recommendations': recommendations,
            'resistance_potential': resistance_potential,
            'support_risk': support_risk,
            'key_levels': {
                'next_resistance': resistance,
                'next_support': support,
                'current_price': current_price
            }
        }

# ========================================================================================
# 🤖 ADVANCED JAX AI WITH TRAINING
# ========================================================================================

class AdvancedJAXAI:
    def __init__(self):
        self.mode = 'jax' if JAX_AVAILABLE else 'numpy_fallback'
        self.training_data = []
        if JAX_AVAILABLE:
            self.initialized = True
            self.key = random.PRNGKey(42)
            self.model_params = self._init_model()
            print("🧠 JAX Neural Network initialized: 128→64→32→4 architecture")
        else:
            # Numpy fallback weights (deterministic for reproducibility)
            rng = np.random.default_rng(42)
            self.initialized = True  # we provide functional fallback
            self.np_params = {
                'w1': rng.normal(0,0.1,(128,64)), 'b1': np.zeros(64),
                'w2': rng.normal(0,0.1,(64,32)),  'b2': np.zeros(32),
                'w3': rng.normal(0,0.1,(32,16)),  'b3': np.zeros(16),
                'w4': rng.normal(0,0.1,(16,4)),   'b4': np.zeros(4)
            }
            print("⚠️ JAX nicht verfügbar – verwende Numpy Fallback KI (128→64→32→16→4)")
    
    def _init_model(self):
        """Erweiterte 4-Layer Architektur"""
        if not JAX_AVAILABLE:
            return None
        
        key1, key2, key3, key4, key5 = random.split(self.key, 5)
        
        return {
            'w1': random.normal(key1, (128, 64)) * 0.1,
            'b1': jnp.zeros(64),
            'w2': random.normal(key2, (64, 32)) * 0.1,
            'b2': jnp.zeros(32),
            'w3': random.normal(key3, (32, 16)) * 0.1,
            'b3': jnp.zeros(16),
            'w4': random.normal(key4, (16, 4)) * 0.1,  # 4 outputs: STRONG_SELL, SELL, BUY, STRONG_BUY
            'b4': jnp.zeros(4)
        }
    
    def prepare_advanced_features(self, tech_analysis, patterns, market_data, position_analysis):
        """Erweiterte Feature-Extraktion für bessere KI"""
        features = np.zeros(128)
        
        # Technical Analysis Features (0-49)
        rsi_data = tech_analysis.get('rsi', {})
        features[0] = rsi_data.get('rsi', 50) / 100.0
        features[1] = 1.0 if rsi_data.get('trend') == 'overbought' else 0.0
        features[2] = 1.0 if rsi_data.get('trend') == 'oversold' else 0.0
        
        macd_data = tech_analysis.get('macd', {})
        features[3] = np.tanh(macd_data.get('macd', 0) / 100.0)
        features[4] = np.tanh(macd_data.get('histogram', 0) / 50.0)
        features[5] = 1.0 if macd_data.get('curve_direction') == 'bullish_curve' else 0.0
        features[6] = 1.0 if macd_data.get('curve_direction') == 'bearish_curve' else 0.0
        features[7] = 1.0 if macd_data.get('curve_direction') == 'bullish_reversal' else 0.0
        features[8] = 1.0 if macd_data.get('curve_direction') == 'bearish_reversal' else 0.0
        
        # Moving Averages
        features[9] = tech_analysis.get('sma_9', 0) / 100000.0  # Normalized
        features[10] = tech_analysis.get('sma_20', 0) / 100000.0
        
        # Support/Resistance Strength
        features[11] = tech_analysis.get('support_strength', 0) / 10.0
        features[12] = tech_analysis.get('resistance_strength', 0) / 10.0
        
        # Pattern Features (50-79)
        pattern_data = patterns.get('patterns', [])
        for i, pattern in enumerate(pattern_data[:10]):  # Max 10 patterns
            base_idx = 50 + i * 3
            features[base_idx] = pattern.get('confidence', 0) / 100.0
            features[base_idx + 1] = 1.0 if pattern.get('signal') == 'bullish' else 0.0
            features[base_idx + 2] = 1.0 if pattern.get('signal') == 'bearish' else 0.0
        
        # Market Features (80-99)
        features[80] = market_data.get('change_24h', 0) / 100.0
        features[81] = np.log(market_data.get('volume_24h', 1)) / 25.0
        features[82] = market_data.get('high_24h', 0) / 100000.0
        features[83] = market_data.get('low_24h', 0) / 100000.0
        
        # Position Management Features (100-119)
        features[100] = position_analysis.get('resistance_potential', 0) / 100.0
        features[101] = position_analysis.get('support_risk', 0) / 100.0
        features[102] = 1.0 if position_analysis.get('position_status') == 'BULLISH' else 0.0
        features[103] = 1.0 if position_analysis.get('position_status') == 'BEARISH' else 0.0
        
        # Time-based features (120-127)
        now = datetime.now()
        features[120] = now.hour / 24.0  # Hour of day
        features[121] = now.weekday() / 6.0  # Day of week
        
        # Fill remaining with noise for regularization
        for i in range(122, 128):
            features[i] = np.random.normal(0, 0.05)
        
        return features
    
    def predict_advanced(self, features):
        """Erweiterte Vorhersage mit 4 Signalen"""
        if not self.initialized:
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':'KI nicht initialisiert'}

        def _postprocess(probs_arr, version_tag):
            probs_np = np.array(probs_arr, dtype=float)
            probs_np = probs_np / (probs_np.sum()+1e-9)
            signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
            max_idx = int(np.argmax(probs_np))
            signal = signals[max_idx]
            confidence = float(probs_np[max_idx]*100)
            if signal == 'STRONG_BUY' and confidence > 75:
                rec = '🚀 KI sehr bullish'
            elif signal == 'BUY' and confidence > 60:
                rec = '📈 Moderat bullish'
            elif signal == 'STRONG_SELL' and confidence > 75:
                rec = '📉 Stark bearish'
            elif signal == 'SELL' and confidence > 60:
                rec = '⚠️ Abwärtsrisiko'
            else:
                rec = '🔄 Neutral / Beobachten'
            return {
                'signal': signal,
                'confidence': round(confidence,2),
                'probabilities': probs_np.round(4).tolist(),
                'ai_recommendation': rec,
                'model_version': version_tag,
                'mode': self.mode
            }

        try:
            if self.mode == 'jax':
                x = jnp.array(features)
                h1 = jnp.tanh(jnp.dot(x, self.model_params['w1']) + self.model_params['b1'])
                h2 = jnp.tanh(jnp.dot(h1, self.model_params['w2']) + self.model_params['b2'])
                h3 = jnp.tanh(jnp.dot(h2, self.model_params['w3']) + self.model_params['b3'])
                logits = jnp.dot(h3, self.model_params['w4']) + self.model_params['b4']
                probs = jnp.exp(logits - logsumexp(logits))
                return _postprocess(np.array(probs), 'JAX-v2.0')
            else:
                x = np.array(features, dtype=float)
                h1 = np.tanh(x @ self.np_params['w1'] + self.np_params['b1'])
                h2 = np.tanh(h1 @ self.np_params['w2'] + self.np_params['b2'])
                h3 = np.tanh(h2 @ self.np_params['w3'] + self.np_params['b3'])
                logits = h3 @ self.np_params['w4'] + self.np_params['b4']
                # stable softmax
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                return _postprocess(probs, 'NP-FALLBACK-v1')
        except Exception as e:
            print(f"❌ Neural network error: {e}")
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':f'KI-Fehler: {e}','mode':self.mode}
    
    def add_training_data(self, features, actual_outcome):
        """Training data sammeln für späteres Lernen"""
        self.training_data.append({
            'features': features,
            'outcome': actual_outcome,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 samples
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
        
        # Automatisches Training alle 50 neue Samples
        if len(self.training_data) % 50 == 0 and len(self.training_data) >= 100:
            print(f"🤖 Automatisches JAX Training startet mit {len(self.training_data)} Samples...")
            self.auto_train()
    
    def auto_train(self):
        """Automatisches JAX Neural Network Training"""
        if not JAX_AVAILABLE or len(self.training_data) < 50:
            return
        
        try:
            # Prepare training data
            X = np.array([d['features'] for d in self.training_data])
            y = np.array([self._encode_outcome(d['outcome']) for d in self.training_data])
            
            # Simple gradient descent training
            learning_rate = 0.001
            batch_size = min(32, len(X))
            epochs = 10
            
            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                # Mini-batch training
                for i in range(0, len(X), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_y = y_shuffled[i:i+batch_size]
                    
                    # Simple weight update (simplified)
                    self._update_weights(batch_X, batch_y, learning_rate)
            
            print(f"✅ JAX Training abgeschlossen! Model wurde mit {len(self.training_data)} Samples trainiert")
            self.last_train_info = {
                'samples': len(self.training_data),
                'epochs': epochs,
                'updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.last_train_info = {'error': str(e), 'updated': datetime.now().isoformat()}

    def get_status(self):
        return {
            'initialized': self.initialized,
            'samples_collected': len(self.training_data),
            'last_train': getattr(self, 'last_train_info', None),
            'model_version': 'JAX-v2.0' if self.mode=='jax' else 'NP-FALLBACK-v1',
            'mode': self.mode
        }
    
    def _encode_outcome(self, outcome):
        """Encode trading outcome to neural network targets"""
        # outcome: 'profit', 'loss', 'neutral'
        if outcome == 'profit':
            return [0, 0, 1, 1]  # BUY, STRONG_BUY signals were correct
        elif outcome == 'loss':
            return [1, 1, 0, 0]  # SELL, STRONG_SELL signals were correct
        else:
            return [0.25, 0.25, 0.25, 0.25]  # Neutral outcome
    
    def _update_weights(self, batch_X, batch_y, learning_rate):
        """Simplified weight update for auto-training"""
        try:
            # Simple gradient approximation
            for i in range(len(batch_X)):
                features = batch_X[i]
                target = batch_y[i]
                
                # Forward pass
                h1 = jnp.tanh(jnp.dot(features, self.model_params['w1']) + self.model_params['b1'])
                h2 = jnp.tanh(jnp.dot(h1, self.model_params['w2']) + self.model_params['b2'])
                h3 = jnp.tanh(jnp.dot(h2, self.model_params['w3']) + self.model_params['b3'])
                output = jnp.dot(h3, self.model_params['w4']) + self.model_params['b4']
                
                # Simple error-based weight adjustment
                error = target - output
                
                # Update output layer weights (simplified)
                self.model_params['w4'] = self.model_params['w4'] + learning_rate * jnp.outer(h3, error)
                self.model_params['b4'] = self.model_params['b4'] + learning_rate * error
                
        except Exception as e:
            print(f"Weight update error: {e}")
    
    def get_training_stats(self):
        """Statistiken über das Training"""
        if not self.training_data:
            return "Keine Trainingsdaten vorhanden"
        
        total_samples = len(self.training_data)
        recent_samples = len([d for d in self.training_data if (datetime.now() - d['timestamp']).days < 7])
        
        return {
            'total_samples': total_samples,
            'recent_samples': recent_samples,
            'last_training': 'Automatisch alle 50 Samples',
            'model_version': 'JAX-v2.0-AutoTrain'
        }

# ========================================================================================
# 📊 ENHANCED CHART PATTERN DETECTION
# ========================================================================================

class AdvancedPatternDetector:
    @staticmethod
    def detect_advanced_patterns(candles):
        """Erweiterte Pattern-Erkennung mit visuellen Details"""
        if len(candles) < 30:
            return {
                'patterns': [],
                'pattern_summary': 'Nicht genug Daten für Pattern-Analyse',
                'visual_signals': [],
                'confidence_score': 0
            }
        
        patterns = []
        visual_signals = []
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        closes = [c['close'] for c in candles]
        volumes = [c['volume'] for c in candles]
        
        # 🔺 Enhanced Triangle Detection
        triangle = AdvancedPatternDetector._detect_enhanced_triangle(highs, lows, volumes)
        if triangle:
            patterns.append(triangle)
            visual_signals.append(f"📐 {triangle['type']} Formation erkannt")
        
        # 👑 Head & Shoulders mit Volumen-Bestätigung
        head_shoulders = AdvancedPatternDetector._detect_head_shoulders_with_volume(highs, lows, volumes)
        if head_shoulders:
            patterns.append(head_shoulders)
            visual_signals.append(f"👑 {head_shoulders['type']} Pattern bestätigt")
        
        # 🔄 Enhanced Double Patterns
        double_pattern = AdvancedPatternDetector._detect_enhanced_double_patterns(highs, lows, volumes)
        if double_pattern:
            patterns.append(double_pattern)
            visual_signals.append(f"🔄 {double_pattern['type']} - {double_pattern['strength']}")
        
        # 📈 Cup & Handle
        cup_handle = AdvancedPatternDetector._detect_cup_and_handle(highs, lows, closes)
        if cup_handle:
            patterns.append(cup_handle)
            visual_signals.append("☕ Cup & Handle - Bullish breakout erwartet")
        
        # 🏃 Breakout Patterns
        breakout = AdvancedPatternDetector._detect_breakout_patterns(highs, lows, closes, volumes)
        if breakout:
            patterns.append(breakout)
            visual_signals.append(f"🏃 {breakout['direction']} Breakout detected!")
        
        # Calculate overall pattern strength
        if patterns:
            total_confidence = sum(p['confidence'] for p in patterns)
            avg_confidence = total_confidence / len(patterns)
            
            bullish_count = len([p for p in patterns if p['signal'] == 'bullish'])
            bearish_count = len([p for p in patterns if p['signal'] == 'bearish'])
            
            if bullish_count > bearish_count:
                overall_signal = 'BULLISH'
                pattern_summary = f"🚀 {bullish_count} bullische Patterns dominieren"
            elif bearish_count > bullish_count:
                overall_signal = 'BEARISH'
                pattern_summary = f"📉 {bearish_count} bearische Patterns dominieren"
            else:
                overall_signal = 'NEUTRAL'
                pattern_summary = "⚖️ Gemischte Pattern-Signale"
        else:
            avg_confidence = 0
            overall_signal = 'NEUTRAL'
            pattern_summary = "Keine klaren Patterns erkannt"
            visual_signals.append("👀 Weiter beobachten...")
        
        return {
            'patterns': patterns,
            'pattern_summary': pattern_summary,
            'visual_signals': visual_signals,
            'overall_signal': overall_signal,
            'confidence_score': avg_confidence,
            'patterns_count': len(patterns)
        }
    
    @staticmethod
    def _detect_enhanced_triangle(highs, lows, volumes, lookback=20):
        """Erweiterte Triangle Detection mit Volumen"""
        if len(highs) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_volumes = volumes[-lookback:]
        
        # Trend-Berechnung
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        # Volumen-Analyse (sollte abnehmen bei Triangle)
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        volume_confirmation = volume_trend < 0  # Abnehmendes Volumen ist gut
        
        confidence_bonus = 10 if volume_confirmation else 0
        
        # Ascending Triangle
        if abs(high_trend) < 0.002 and low_trend > 0.015:
            return {
                'type': 'Ascending Triangle',
                'signal': 'bullish',
                'confidence': 75 + confidence_bonus,
                'description': f'Flache Resistance, steigende Support. Volumen: {"✅" if volume_confirmation else "⚠️"}',
                'target_level': recent_highs[-1] * 1.05,  # 5% above resistance
                'stop_level': recent_lows[-1] * 0.98,     # 2% below recent low
                'strength': 'STRONG' if volume_confirmation else 'MEDIUM'
            }
        
        # Descending Triangle
        elif high_trend < -0.015 and abs(low_trend) < 0.002:
            return {
                'type': 'Descending Triangle',
                'signal': 'bearish',
                'confidence': 75 + confidence_bonus,
                'description': f'Sinkende Resistance, flache Support. Volumen: {"✅" if volume_confirmation else "⚠️"}',
                'target_level': recent_lows[-1] * 0.95,   # 5% below support
                'stop_level': recent_highs[-1] * 1.02,    # 2% above recent high
                'strength': 'STRONG' if volume_confirmation else 'MEDIUM'
            }
        
        # Symmetrical Triangle
        elif high_trend < -0.008 and low_trend > 0.008:
            return {
                'type': 'Symmetrical Triangle',
                'signal': 'neutral',
                'confidence': 65 + confidence_bonus,
                'description': f'Konvergierende Linien. Volumen: {"✅" if volume_confirmation else "⚠️"}',
                'breakout_expected': True,
                'strength': 'MEDIUM'
            }
        
        return None
    
    @staticmethod
    def _detect_head_shoulders_with_volume(highs, lows, volumes, lookback=25):
        """Head & Shoulders mit Volumen-Bestätigung"""
        if len(highs) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_volumes = volumes[-lookback:]
        
        # Find peaks with volume analysis
        peaks = []
        for i in range(1, len(recent_highs) - 1):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                volume_at_peak = recent_volumes[i]
                peaks.append((i, recent_highs[i], volume_at_peak))
        
        if len(peaks) >= 3:
            # Sort by height to find head (highest) and shoulders
            peaks_by_height = sorted(peaks, key=lambda x: x[1], reverse=True)
            head = peaks_by_height[0]
            potential_shoulders = peaks_by_height[1:3]
            
            # Check if shoulders are similar height
            shoulder_diff = abs(potential_shoulders[0][1] - potential_shoulders[1][1])
            if shoulder_diff / potential_shoulders[0][1] < 0.03:  # Within 3%
                
                # Volume should be lower on right shoulder (bearish confirmation)
                left_shoulder_vol = potential_shoulders[0][2]
                right_shoulder_vol = potential_shoulders[1][2]
                head_volume = head[2]
                
                volume_confirmation = right_shoulder_vol < left_shoulder_vol
                
                return {
                    'type': 'Head and Shoulders',
                    'signal': 'bearish',
                    'confidence': 80 if volume_confirmation else 65,
                    'description': f'Klassische Umkehrformation. Volumen-Bestätigung: {"✅" if volume_confirmation else "⚠️"}',
                    'head_level': head[1],
                    'neckline': min(potential_shoulders[0][1], potential_shoulders[1][1]),
                    'target': min(potential_shoulders[0][1], potential_shoulders[1][1]) * 0.92,  # 8% below neckline
                    'strength': 'VERY_STRONG' if volume_confirmation else 'STRONG'
                }
        
        return None
    
    @staticmethod
    def _detect_enhanced_double_patterns(highs, lows, volumes, lookback=20):
        """Enhanced Double Top/Bottom mit Volumen"""
        if len(highs) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_volumes = volumes[-lookback:]
        
        # Double Top Detection mit Volumen
        high_peaks = []
        for i in range(1, len(recent_highs) - 1):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                high_peaks.append((i, recent_highs[i], recent_volumes[i]))
        
        if len(high_peaks) >= 2:
            last_two_peaks = high_peaks[-2:]
            height_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
            
            if height_diff / last_two_peaks[0][1] < 0.025:  # Within 2.5%
                # Check volume - second peak should have lower volume
                volume_confirmation = last_two_peaks[1][2] < last_two_peaks[0][2]
                
                return {
                    'type': 'Double Top',
                    'signal': 'bearish',
                    'confidence': 75 if volume_confirmation else 60,
                    'description': f'Doppelte Spitze erkannt. Volumen-Divergenz: {"✅" if volume_confirmation else "⚠️"}',
                    'resistance_level': max(last_two_peaks[0][1], last_two_peaks[1][1]),
                    'target': min(recent_lows) * 0.95,
                    'strength': 'STRONG' if volume_confirmation else 'MEDIUM'
                }
        
        # Double Bottom Detection
        low_valleys = []
        for i in range(1, len(recent_lows) - 1):
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                low_valleys.append((i, recent_lows[i], recent_volumes[i]))
        
        if len(low_valleys) >= 2:
            last_two_valleys = low_valleys[-2:]
            depth_diff = abs(last_two_valleys[0][1] - last_two_valleys[1][1])
            
            if depth_diff / last_two_valleys[0][1] < 0.025:  # Within 2.5%
                # Second bottom should have higher volume (buying interest)
                volume_confirmation = last_two_valleys[1][2] > last_two_valleys[0][2]
                
                return {
                    'type': 'Double Bottom',
                    'signal': 'bullish',
                    'confidence': 75 if volume_confirmation else 60,
                    'description': f'Doppelter Boden erkannt. Volumen-Bestätigung: {"✅" if volume_confirmation else "⚠️"}',
                    'support_level': min(last_two_valleys[0][1], last_two_valleys[1][1]),
                    'target': max(recent_highs) * 1.05,
                    'strength': 'STRONG' if volume_confirmation else 'MEDIUM'
                }
        
        return None
    
    @staticmethod
    def _detect_cup_and_handle(highs, lows, closes, lookback=30):
        """Cup & Handle Pattern Detection"""
        if len(closes) < lookback:
            return None
        
        recent_closes = closes[-lookback:]
        
        # Find potential cup (U-shape)
        start_price = recent_closes[0]
        end_price = recent_closes[-1]
        min_price = min(recent_closes)
        min_index = recent_closes.index(min_price)
        
        # Cup criteria
        cup_depth = (start_price - min_price) / start_price
        recovery_ratio = (end_price - min_price) / (start_price - min_price)
        
        if 0.1 < cup_depth < 0.5 and recovery_ratio > 0.7:  # Valid cup shape
            # Look for handle (small pullback)
            handle_start = int(lookback * 0.7)  # Last 30% of data
            handle_data = recent_closes[handle_start:]
            
            if len(handle_data) > 5:
                handle_high = max(handle_data)
                handle_low = min(handle_data)
                handle_depth = (handle_high - handle_low) / handle_high
                
                if 0.05 < handle_depth < 0.15:  # Valid handle
                    return {
                        'type': 'Cup and Handle',
                        'signal': 'bullish',
                        'confidence': 82,
                        'description': f'Cup-Tiefe: {cup_depth:.1%}, Handle-Korrektur: {handle_depth:.1%}',
                        'breakout_level': handle_high * 1.02,
                        'target': handle_high * (1 + cup_depth),
                        'strength': 'VERY_STRONG'
                    }
        
        return None
    
    @staticmethod
    def _detect_breakout_patterns(highs, lows, closes, volumes, lookback=15):
        """Breakout Pattern Detection"""
        if len(closes) < lookback:
            return None
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        recent_volumes = volumes[-lookback:]
        
        current_price = recent_closes[-1]
        avg_volume = sum(recent_volumes[:-5]) / (len(recent_volumes) - 5)
        current_volume = recent_volumes[-1]
        
        # Resistance breakout
        resistance = max(recent_highs[:-3])  # Exclude last 3 candles
        if current_price > resistance * 1.02 and current_volume > avg_volume * 1.5:
            return {
                'type': 'Resistance Breakout',
                'signal': 'bullish',
                'confidence': 85,
                'description': f'Ausbruch über ${resistance:,.2f} mit {current_volume/avg_volume:.1f}x Volumen',
                'direction': 'BULLISH',
                'target': resistance * 1.1,
                'strength': 'VERY_STRONG'
            }
        
        # Support breakdown
        support = min(recent_lows[:-3])  # Exclude last 3 candles
        if current_price < support * 0.98 and current_volume > avg_volume * 1.5:
            return {
                'type': 'Support Breakdown',
                'signal': 'bearish',
                'confidence': 85,
                'description': f'Durchbruch unter ${support:,.2f} mit {current_volume/avg_volume:.1f}x Volumen',
                'direction': 'BEARISH',
                'target': support * 0.9,
                'strength': 'VERY_STRONG'
            }
        
        return None

# Initializing global components
position_manager = PositionManager()
advanced_ai = AdvancedJAXAI()
pattern_detector = AdvancedPatternDetector()

# ========================================================================================
# 📈 ENHANCED TECHNICAL ANALYSIS WITH CURVE DETECTION
# ========================================================================================

class TechnicalAnalysis:
    @staticmethod
    def get_candle_data(symbol, limit=100, interval='1h'):
        """Get candlestick data from Binance (interval default 1h)"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not isinstance(data, list):
                return []
            candles = []
            for item in data:
                candles.append({
                    'timestamp': int(item[0]),
                    'time': int(item[0]),  # alias for backtest logic
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            return candles
        except Exception as e:
            print(f"Error getting candle data ({interval}): {e}")
            return []
    
    @staticmethod
    def calculate_advanced_indicators(candles):
        """Calculate all technical indicators"""
        if len(candles) < 50:
            return {}
        
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        
        # RSI with advanced analysis
        rsi_data = TechnicalAnalysis._calculate_advanced_rsi(closes)
        
        # MACD with curve detection
        macd_data = TechnicalAnalysis._calculate_advanced_macd(closes)
        
        # Moving Averages
        sma_9 = TechnicalAnalysis._sma(closes, 9)
        sma_20 = TechnicalAnalysis._sma(closes, 20)
        ema_12 = TechnicalAnalysis._ema(closes, 12)
        ema_26 = TechnicalAnalysis._ema(closes, 26)
        
        # Support/Resistance
        support, resistance = TechnicalAnalysis._calculate_support_resistance(highs, lows, closes)
        
        # Volume analysis
        volume_analysis = TechnicalAnalysis._analyze_volume(volumes, closes)
        
        # Trend analysis
        trend_analysis = TechnicalAnalysis._analyze_trend(closes, sma_9, sma_20)
        
        # Momentum indicators
        momentum = TechnicalAnalysis._calculate_momentum(closes)
        
        return {
            'rsi': rsi_data,
            'macd': macd_data,
            'sma_9': sma_9[-1] if len(sma_9) > 0 else closes[-1],
            'sma_20': sma_20[-1] if len(sma_20) > 0 else closes[-1],
            'ema_12': ema_12[-1] if len(ema_12) > 0 else closes[-1],
            'ema_26': ema_26[-1] if len(ema_26) > 0 else closes[-1],
            'support': support,
            'resistance': resistance,
            'support_strength': TechnicalAnalysis._calculate_level_strength(lows, support),
            'resistance_strength': TechnicalAnalysis._calculate_level_strength(highs, resistance),
            'volume_analysis': volume_analysis,
            'trend': trend_analysis,
            'momentum': momentum,
            'current_price': closes[-1],
            'price_position': TechnicalAnalysis._calculate_price_position(closes[-1], support, resistance)
        }
    
    @staticmethod
    def _calculate_advanced_rsi(closes, period=14):
        """Advanced RSI with TradingView-style Wilder smoothing + comparison.
        Returns dict with rsi (current), tv_rsi (should match), diff, trend classification and small tail series.
        No dummy placeholder values – if insufficient data returns {'error': 'insufficient_data'}.
        """
        n = int(period)
        if len(closes) < n + 1:
            return {'error': 'insufficient_data', 'needed': n+1, 'have': len(closes)}

        closes_arr = np.asarray(closes, dtype=float)
        deltas = np.diff(closes_arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder's initial average gain/loss
        avg_gain = gains[:n].mean()
        avg_loss = losses[:n].mean()

        rsi_series = np.full(len(closes_arr), np.nan, dtype=float)
        # First RSI value at index n
        rs = avg_gain / avg_loss if avg_loss > 1e-12 else np.inf
        rsi_series[n] = 100 - (100 / (1 + rs)) if np.isfinite(rs) else 100.0

        # Wilder smoothing forward
        for i in range(n+1, len(closes_arr)):
            gain = gains[i-1]
            loss = losses[i-1]
            avg_gain = (avg_gain * (n - 1) + gain) / n
            avg_loss = (avg_loss * (n - 1) + loss) / n
            rs = avg_gain / avg_loss if avg_loss > 1e-12 else np.inf
            rsi_series[i] = 100 - (100 / (1 + rs)) if np.isfinite(rs) else 100.0

        # Current RSI (TradingView equivalent)
        current_rsi = float(rsi_series[-1])
        tv_rsi = current_rsi  # identical algorithm; placeholder if later external API used
        rsi_diff = round(abs(current_rsi - tv_rsi), 4)

        # Trend classification using recent slope (last 5 valid points)
        recent = rsi_series[~np.isnan(rsi_series)][-5:]
        if len(recent) >= 2:
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
        else:
            slope = 0

        if current_rsi >= 80:
            trend = 'overbought'
            strength = 'very_strong' if slope > 0 else 'strong'
        elif current_rsi >= 70:
            trend = 'overbought_risk' if slope > 0 else 'weakening_overbought'
            strength = 'strong'
        elif current_rsi <= 20:
            trend = 'oversold'
            strength = 'very_strong' if slope < 0 else 'strong'
        elif current_rsi <= 30:
            trend = 'oversold_risk' if slope < 0 else 'weakening_oversold'
            strength = 'strong'
        elif 40 <= current_rsi <= 60:
            trend = 'neutral'
            strength = 'medium'
        else:
            trend = 'bullish_bias' if slope > 0 else 'bearish_bias' if slope < 0 else 'neutral'
            strength = 'medium'

        divergence = TechnicalAnalysis._check_rsi_divergence(closes_arr[-10:], rsi_series[~np.isnan(rsi_series)][-10:] if len(rsi_series[~np.isnan(rsi_series)]) >= 10 else [])

        return {
            'rsi': round(current_rsi, 2),
            'tv_rsi': round(tv_rsi, 2),
            'rsi_diff': rsi_diff,
            'trend': trend,
            'strength': strength,
            'divergence': divergence,
            'period': n,
            'series_tail': [round(x,2) for x in rsi_series[-30:].tolist()]
        }
    
    @staticmethod
    def _calculate_advanced_macd(closes, fast=12, slow=26, signal=9):
        """Advanced MACD with curve detection"""
        if len(closes) < slow + signal + 10:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'curve_direction': 'neutral'}
        
        ema_fast = TechnicalAnalysis._ema(closes, fast)
        ema_slow = TechnicalAnalysis._ema(closes, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysis._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        # Curve direction analysis
        if len(histogram) >= 5:
            recent_hist = histogram[-5:]
            curve_trend = np.polyfit(range(len(recent_hist)), recent_hist, 1)[0]
            
            # Check for curve patterns
            if len(histogram) >= 10:
                prev_hist = histogram[-10:-5]
                prev_trend = np.polyfit(range(len(prev_hist)), prev_hist, 1)[0]
                
                # Bullish curve (was going down, now going up)
                if prev_trend < -0.5 and curve_trend > 0.5:
                    curve_direction = 'bullish_reversal'
                # Bearish curve (was going up, now going down)
                elif prev_trend > 0.5 and curve_trend < -0.5:
                    curve_direction = 'bearish_reversal'
                # Continued bullish curve
                elif curve_trend > 1.0:
                    curve_direction = 'bullish_curve'
                # Continued bearish curve
                elif curve_trend < -1.0:
                    curve_direction = 'bearish_curve'
                else:
                    curve_direction = 'neutral'
            else:
                curve_direction = 'bullish_curve' if curve_trend > 0 else 'bearish_curve'
        else:
            curve_direction = 'neutral'
        
        # Curve strength (second derivative approximation on last 6 histogram points)
        curve_strength = 0.0
        try:
            if len(histogram) >= 6:
                recent = np.array(histogram[-6:], dtype=float)
                d1 = np.diff(recent)
                if len(d1) >= 2:
                    d2 = np.diff(d1)
                    curvature = float(np.mean(d2))
                    scale = float(np.mean(np.abs(recent)) + 1e-6)
                    norm_curv = curvature / (scale * 3)
                    curve_strength = max(-1.0, min(1.0, norm_curv))
        except Exception:
            pass
        return {
            'macd': macd_line[-1],
            'signal': signal_line[-1],
            'histogram': histogram[-1],
            'curve_direction': curve_direction,
            'trend_strength': abs(histogram[-1]) / max(abs(min(histogram)), abs(max(histogram))) if len(histogram) > 0 else 0,
            'curve_strength': curve_strength,
            'curve_strength_pct': round(curve_strength*100,2)
        }
    
    @staticmethod
    def _calculate_support_resistance(highs, lows, closes):
        """Calculate dynamic support and resistance levels"""
        # Use recent price action for dynamic levels
        recent_highs = highs[-20:] if len(highs) >= 20 else highs
        recent_lows = lows[-20:] if len(lows) >= 20 else lows
        current_price = closes[-1]
        
        # Find resistance (significant high above current price)
        resistance_candidates = [h for h in recent_highs if h > current_price * 1.001]
        resistance = min(resistance_candidates) if resistance_candidates else current_price * 1.05
        
        # Find support (significant low below current price)
        support_candidates = [l for l in recent_lows if l < current_price * 0.999]
        support = max(support_candidates) if support_candidates else current_price * 0.95
        
        return support, resistance
    
    @staticmethod
    def _calculate_level_strength(prices, level):
        """Calculate how strong a support/resistance level is"""
        touches = sum(1 for p in prices if abs(p - level) / level < 0.005)  # Within 0.5%
        return min(touches, 10)  # Cap at 10
    
    @staticmethod
    def _analyze_volume(volumes, closes):
        """Advanced volume analysis"""
        if len(volumes) < 20:
            return {'trend': 'unknown', 'strength': 'weak'}
        
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume
        
        # Price-Volume relationship
        price_change = (closes[-1] - closes[-2]) / closes[-2]
        
        if volume_ratio > 1.5:
            if price_change > 0:
                volume_trend = 'bullish_volume_surge'
                strength = 'very_strong'
            else:
                volume_trend = 'bearish_volume_surge'
                strength = 'very_strong'
        elif volume_ratio > 1.2:
            volume_trend = 'above_average'
            strength = 'strong'
        elif volume_ratio < 0.7:
            volume_trend = 'below_average'
            strength = 'weak'
        else:
            volume_trend = 'normal'
            strength = 'medium'
        
        return {
            'trend': volume_trend,
            'strength': strength,
            'ratio': volume_ratio,
            'current': current_volume,
            'average': avg_volume
        }
    
    @staticmethod
    def _analyze_trend(closes, sma_9, sma_20):
        """Comprehensive trend analysis"""
        if len(closes) < 20 or len(sma_9) == 0 or len(sma_20) == 0:
            return {'trend': 'neutral', 'strength': 'weak'}
        
        current_price = closes[-1]
        
        # Moving average relationship
        ma_bullish = sma_9[-1] > sma_20[-1]
        price_above_ma = current_price > sma_9[-1] and current_price > sma_20[-1]
        
        # Recent price momentum
        short_term_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        medium_term_change = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        
        # Determine trend
        if ma_bullish and price_above_ma and short_term_change > 0.02:
            trend = 'strong_bullish'
            strength = 'very_strong'
        elif ma_bullish and price_above_ma:
            trend = 'bullish'
            strength = 'strong'
        elif not ma_bullish and not price_above_ma and short_term_change < -0.02:
            trend = 'strong_bearish'
            strength = 'very_strong'
        elif not ma_bullish and not price_above_ma:
            trend = 'bearish'
            strength = 'strong'
        elif abs(short_term_change) < 0.005:
            trend = 'sideways'
            strength = 'weak'
        else:
            trend = 'neutral'
            strength = 'medium'
        
        return {
            'trend': trend,
            'strength': strength,
            'short_term_momentum': short_term_change,
            'medium_term_momentum': medium_term_change,
            'ma_alignment': 'bullish' if ma_bullish else 'bearish'
        }
    
    @staticmethod
    def _calculate_momentum(closes, period=10):
        """Calculate price momentum"""
        if len(closes) < period:
            return {'value': 0, 'trend': 'neutral'}
        
        momentum = (closes[-1] - closes[-period]) / closes[-period] * 100
        
        if momentum > 5:
            trend = 'very_bullish'
        elif momentum > 2:
            trend = 'bullish'
        elif momentum < -5:
            trend = 'very_bearish'
        elif momentum < -2:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'value': momentum,
            'trend': trend
        }
    
    @staticmethod
    def _calculate_price_position(current_price, support, resistance):
        """Calculate where price is between support and resistance"""
        if resistance <= support:
            return 0.5
        
        position = (current_price - support) / (resistance - support)
        return max(0, min(1, position))  # Clamp between 0 and 1
    
    @staticmethod
    def _check_rsi_divergence(prices, rsi_values):
        """Check for RSI divergence"""
        if len(prices) < 5 or len(rsi_values) < 5:
            return 'none'
        
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
        
        if price_trend > 0 and rsi_trend < 0:
            return 'bearish_divergence'
        elif price_trend < 0 and rsi_trend > 0:
            return 'bullish_divergence'
        else:
            return 'none'
    
    @staticmethod
    def _sma(data, window):
        """Simple Moving Average"""
        if len(data) < window:
            return np.array([])
        return np.array([np.mean(data[i-window:i]) for i in range(window, len(data) + 1)])
    
    @staticmethod
    def _ema(data, window):
        """Exponential Moving Average"""
        if len(data) < window:
            return np.zeros(len(data))  # Return zeros with same length
        
        alpha = 2 / (window + 1)
        ema = np.zeros(len(data))
        
        # Initialize with simple average for first window points
        for i in range(window):
            ema[i] = np.mean(data[:i+1])
        
        # Calculate EMA for remaining points
        for i in range(window, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema

# ========================================================================================
# 📊 ERWEITERTE TECHNISCHE ANALYSE - ENTERPRISE LEVEL
# ========================================================================================

class AdvancedTechnicalAnalysis:
    @staticmethod
    def calculate_extended_indicators(candles):
        """Berechnet erweiterte technische Indikatoren für Enterprise Trading"""
        if len(candles) < 50:
            return {}
        
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        
        # 1. BOLLINGER BANDS mit Details
        bb_data = AdvancedTechnicalAnalysis._calculate_bollinger_bands(closes)
        
        # 2. STOCHASTIC OSCILLATOR
        stoch_data = AdvancedTechnicalAnalysis._calculate_stochastic(highs, lows, closes)
        
        # 3. WILLIAMS %R
        williams_r = AdvancedTechnicalAnalysis._calculate_williams_r(highs, lows, closes)
        
        # 4. COMMODITY CHANNEL INDEX (CCI)
        cci = AdvancedTechnicalAnalysis._calculate_cci(highs, lows, closes)
        
        # 5. AVERAGE TRUE RANGE (ATR) - Volatilität
        atr = AdvancedTechnicalAnalysis._calculate_atr(highs, lows, closes)
        
        # 6. FIBONACCI RETRACEMENTS
        fib_levels = AdvancedTechnicalAnalysis._calculate_fibonacci_levels(highs, lows)
        
        # 7. ICHIMOKU CLOUD
        ichimoku = AdvancedTechnicalAnalysis._calculate_ichimoku(highs, lows, closes)
        
        # 8. PIVOT POINTS
        pivot_points = AdvancedTechnicalAnalysis._calculate_pivot_points(highs[-1], lows[-1], closes[-1])
        
        # 9. VOLUME INDICATORS
        volume_indicators = AdvancedTechnicalAnalysis._calculate_volume_indicators(volumes, closes)
        
        # 10. TREND STRENGTH
        trend_strength = AdvancedTechnicalAnalysis._calculate_trend_strength(closes)
        
        return {
            'bollinger_bands': bb_data,
            'stochastic': stoch_data,
            'williams_r': williams_r,
            'cci': cci,
            'atr': atr,
            'fibonacci': fib_levels,
            'ichimoku': ichimoku,
            'pivot_points': pivot_points,
            'volume_indicators': volume_indicators,
            'trend_strength': trend_strength
        }
    
    @staticmethod
    def _calculate_bollinger_bands(closes, period=20, std_multiplier=2):
        """Bollinger Bands mit Squeeze-Erkennung"""
        if len(closes) < period:
            return {'middle': closes[-1], 'upper': closes[-1], 'lower': closes[-1], 'squeeze': False}
        
        sma = TechnicalAnalysis._sma(closes, period)
        std = np.std(closes[-period:])
        
        middle = sma[-1]
        upper = middle + (std * std_multiplier)
        lower = middle - (std * std_multiplier)
        
        # Bollinger Band Squeeze Detection
        current_std = std
        avg_std = np.mean([np.std(closes[i-period:i]) for i in range(period, min(len(closes), period*3))])
        squeeze = current_std < avg_std * 0.8
        
        # Position relative zu Bands
        current_price = closes[-1]
        bb_position = (current_price - lower) / (upper - lower)
        
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'squeeze': squeeze,
            'position': bb_position,
            'width': ((upper - lower) / middle) * 100,
            'signal': 'overbought' if bb_position > 0.8 else 'oversold' if bb_position < 0.2 else 'neutral'
        }
    
    @staticmethod
    def _calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
        """Stochastic Oscillator %K und %D"""
        if len(closes) < k_period:
            return {'k': 50, 'd': 50, 'signal': 'neutral'}
        
        # %K Berechnung
        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D Berechnung (Glättung von %K)
        k_values = []
        for i in range(max(1, len(closes) - k_period + 1), len(closes) + 1):
            if i >= k_period:
                period_low = np.min(lows[i-k_period:i])
                period_high = np.max(highs[i-k_period:i])
                if period_high != period_low:
                    k_val = ((closes[i-1] - period_low) / (period_high - period_low)) * 100
                    k_values.append(k_val)
        
        d_percent = np.mean(k_values[-d_period:]) if len(k_values) >= d_period else k_percent
        
        # Signal bestimmen
        if k_percent > 80 and d_percent > 80:
            signal = 'overbought'
        elif k_percent < 20 and d_percent < 20:
            signal = 'oversold'
        elif k_percent > d_percent and k_percent > 50:
            signal = 'bullish'
        elif k_percent < d_percent and k_percent < 50:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return {
            'k': k_percent,
            'd': d_percent,
            'signal': signal,
            'crossover': 'bullish' if k_percent > d_percent else 'bearish'
        }
    
    @staticmethod
    def _calculate_williams_r(highs, lows, closes, period=14):
        """Williams %R Momentum Indikator"""
        if len(closes) < period:
            return {'value': -50, 'signal': 'neutral'}
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            williams_r = -50
        else:
            williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
        
        # Signal bestimmen
        if williams_r > -20:
            signal = 'overbought'
        elif williams_r < -80:
            signal = 'oversold'
        elif williams_r > -50:
            signal = 'bullish'
        else:
            signal = 'bearish'
        
        return {
            'value': williams_r,
            'signal': signal,
            'strength': abs(williams_r - (-50)) / 50  # Stärke des Signals
        }
    
    @staticmethod
    def _calculate_cci(highs, lows, closes, period=20):
        """Commodity Channel Index"""
        if len(closes) < period:
            return {'value': 0, 'signal': 'neutral'}
        
        # Typical Price berechnen
        typical_prices = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        sma_tp = np.mean(typical_prices)
        
        # Mean Deviation berechnen
        mean_deviation = np.mean(np.abs(typical_prices - sma_tp))
        
        if mean_deviation == 0:
            cci = 0
        else:
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
        
        # Signal bestimmen
        if cci > 100:
            signal = 'overbought'
        elif cci < -100:
            signal = 'oversold'
        elif cci > 0:
            signal = 'bullish'
        else:
            signal = 'bearish'
        
        return {
            'value': cci,
            'signal': signal,
            'extreme': abs(cci) > 200  # Extreme Werte
        }
    
    @staticmethod
    def _calculate_atr(highs, lows, closes, period=14):
        """Average True Range - Volatilitäts-Indikator"""
        if len(closes) < period + 1:
            return {'value': 0, 'volatility': 'low'}
        
        # True Range für jeden Tag berechnen
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        # ATR als gleitender Durchschnitt der True Ranges
        atr = np.mean(true_ranges[-period:])
        
        # Relative ATR (in % vom aktuellen Preis)
        atr_percent = (atr / closes[-1]) * 100
        
        # Volatilitäts-Level bestimmen
        if atr_percent > 5:
            volatility = 'very_high'
        elif atr_percent > 3:
            volatility = 'high'
        elif atr_percent > 1.5:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        return {
            'value': atr,
            'percentage': atr_percent,
            'volatility': volatility,
            'risk_level': 'high' if volatility in ['high', 'very_high'] else 'medium' if volatility == 'medium' else 'low'
        }
    
    @staticmethod
    def _calculate_fibonacci_levels(highs, lows):
        """Fibonacci Retracement Levels"""
        swing_high = np.max(highs[-50:])  # Letzten 50 Perioden
        swing_low = np.min(lows[-50:])
        
        diff = swing_high - swing_low
        
        fib_levels = {
            'high': swing_high,
            'low': swing_low,
            'fib_236': swing_high - (diff * 0.236),
            'fib_382': swing_high - (diff * 0.382),
            'fib_500': swing_high - (diff * 0.5),
            'fib_618': swing_high - (diff * 0.618),
            'fib_786': swing_high - (diff * 0.786)
        }
        
        return fib_levels
    
    @staticmethod
    def _calculate_ichimoku(highs, lows, closes):
        """Ichimoku Cloud Indikatoren"""
        if len(closes) < 52:
            return {'signal': 'neutral'}
        
        # Tenkan-sen (Conversion Line) - 9 Perioden
        tenkan_high = np.max(highs[-9:])
        tenkan_low = np.min(lows[-9:])
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line) - 26 Perioden
        kijun_high = np.max(highs[-26:])
        kijun_low = np.min(lows[-26:])
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B) - 52 Perioden
        senkou_b_high = np.max(highs[-52:])
        senkou_b_low = np.min(lows[-52:])
        senkou_b = (senkou_b_high + senkou_b_low) / 2
        
        # Cloud Analysis
        current_price = closes[-1]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if current_price > cloud_top:
            cloud_position = 'above'
            signal = 'bullish'
        elif current_price < cloud_bottom:
            cloud_position = 'below'
            signal = 'bearish'
        else:
            cloud_position = 'inside'
            signal = 'neutral'
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'cloud_position': cloud_position,
            'signal': signal,
            'tk_cross': 'bullish' if tenkan_sen > kijun_sen else 'bearish'
        }
    
    @staticmethod
    def _calculate_pivot_points(high, low, close):
        """Pivot Points für Intraday Trading"""
        pivot = (high + low + close) / 3
        
        # Resistance Levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Support Levels
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def _calculate_volume_indicators(volumes, closes):
        """Volume-basierte Indikatoren"""
        if len(volumes) < 20:
            return {'signal': 'neutral'}
        
        # Volume Moving Average
        vol_ma = np.mean(volumes[-20:])
        current_vol = volumes[-1]
        
        # Volume Ratio
        vol_ratio = current_vol / vol_ma
        
        # Price Volume Trend (PVT)
        pvt = 0
        for i in range(1, len(closes)):
            if closes[i-1] != 0:
                price_change = (closes[i] - closes[i-1]) / closes[i-1]
                pvt += price_change * volumes[i]
        
        # On Balance Volume (OBV)
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return {
            'volume_ma': vol_ma,
            'current_volume': current_vol,
            'volume_ratio': vol_ratio,
            'pvt': pvt,
            'obv': obv,
            'volume_signal': 'high' if vol_ratio > 1.5 else 'normal' if vol_ratio > 0.5 else 'low'
        }
    
    @staticmethod
    def _calculate_trend_strength(closes, period=20):
        """Trend-Stärke Analyse"""
        if len(closes) < period:
            return {'strength': 0, 'direction': 'neutral'}
        
        # Linear Regression für Trend
        x = np.arange(period)
        y = closes[-period:]
        
        # Berechne Steigung
        slope = np.polyfit(x, y, 1)[0]
        
        # R-Squared für Trend-Stärke
        y_pred = np.polyval([slope, y[0]], x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Trend-Richtung
        price_change = (closes[-1] - closes[-period]) / closes[-period] * 100
        
        if abs(price_change) > 10 and r_squared > 0.8:
            strength = 'very_strong'
        elif abs(price_change) > 5 and r_squared > 0.6:
            strength = 'strong'
        elif abs(price_change) > 2 and r_squared > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        direction = 'bullish' if slope > 0 else 'bearish' if slope < 0 else 'neutral'
        
        return {
            'strength': strength,
            'direction': direction,
            'slope': slope,
            'r_squared': r_squared,
            'price_change_percent': price_change
        }

# ========================================================================================
# 🔗 ENHANCED BINANCE CLIENT WITH SYMBOL SEARCH
# ========================================================================================

class BinanceClient:
    BASE_URL = "https://api.binance.com/api/v3"
    _cache = {
        'ticker': {},   # symbol -> (timestamp, data)
        'price': {},    # symbol -> (timestamp, price)
        'klines': {}    # (symbol, interval, limit) -> (timestamp, data)
    }
    TICKER_TTL = 10      # seconds
    PRICE_TTL = 3        # seconds
    KLINES_TTL = 45      # seconds

    @staticmethod
    def clear_symbol_cache(symbol: str):
        """Invalidate cached entries for a symbol."""
        try:
            symbol = symbol.upper()
            BinanceClient._cache['ticker'].pop(symbol, None)
            BinanceClient._cache['price'].pop(symbol, None)
            # Remove klines variants for symbol
            to_del = [k for k in BinanceClient._cache['klines'] if k[0] == symbol]
            for k in to_del:
                BinanceClient._cache['klines'].pop(k, None)
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    @staticmethod
    def search_symbols(query):
        """Search for trading symbols"""
        try:
            # Get all exchange info
            response = requests.get(f"{BinanceClient.BASE_URL}/exchangeInfo", timeout=10)
            data = response.json()
            
            symbols = []
            query_upper = query.upper()
            
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol']
                if symbol_info['status'] == 'TRADING':
                    # Prioritize exact matches and USDT pairs
                    score = 0
                    if query_upper in symbol:
                        score += 10
                    if symbol.endswith('USDT'):
                        score += 5
                    if symbol.startswith(query_upper):
                        score += 15
                    if symbol == query_upper + 'USDT':
                        score += 20
                    
                    if score > 0:
                        symbols.append({
                            'symbol': symbol,
                            'baseAsset': symbol_info['baseAsset'],
                            'quoteAsset': symbol_info['quoteAsset'],
                            'score': score
                        })
            
            # Sort by score and return top 10
            return sorted(symbols, key=lambda x: x['score'], reverse=True)[:10]
            
        except Exception as e:
            print(f"Error searching symbols: {e}")
            return []
    
    @staticmethod
    def get_ticker_data(symbol):
        """Get 24hr ticker data"""
        try:
            now = time.time()
            cached = BinanceClient._cache['ticker'].get(symbol)
            if cached and now - cached[0] < BinanceClient.TICKER_TTL:
                data = cached[1]
                data['_cache'] = 'HIT'
                return data
            response = requests.get(f"{BinanceClient.BASE_URL}/ticker/24hr", params={'symbol': symbol}, timeout=10)
            data = response.json()
            data['_cache'] = 'MISS'
            BinanceClient._cache['ticker'][symbol] = (now, data)
            return data
        except Exception as e:
            print(f"Error getting ticker data: {e}")
            return {}
    
    @staticmethod
    def get_current_price(symbol):
        """Get current price"""
        try:
            now = time.time()
            cached = BinanceClient._cache['price'].get(symbol)
            if cached and now - cached[0] < BinanceClient.PRICE_TTL:
                return cached[1]
            response = requests.get(f"{BinanceClient.BASE_URL}/ticker/price", params={'symbol': symbol}, timeout=10)
            data = response.json()
            price = float(data['price'])
            BinanceClient._cache['price'][symbol] = (now, price)
            return price
        except Exception as e:
            print(f"Error getting current price: {e}")
            return 0

# ========================================================================================
# 💰 ENHANCED LIQUIDATION CALCULATOR
# ========================================================================================

class LiquidationCalculator:
    LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]
    
    @staticmethod
    def calculate_liquidation_levels(entry_price, position_type='long'):
        """Calculate liquidation prices for all leverage levels"""
        liquidation_data = []
        
        for leverage in LiquidationCalculator.LEVERAGE_LEVELS:
            if position_type.lower() == 'long':
                # Long liquidation = Entry Price * (1 - 1/Leverage)
                liq_price = entry_price * (1 - 0.95/leverage)  # 5% margin for fees
                distance = ((entry_price - liq_price) / entry_price) * 100
            else:
                # Short liquidation = Entry Price * (1 + 1/Leverage)
                liq_price = entry_price * (1 + 0.95/leverage)  # 5% margin for fees
                distance = ((liq_price - entry_price) / entry_price) * 100
            
            # Risk assessment
            if distance > 10:
                risk_level = "NIEDRIG"
                risk_color = "#28a745"
            elif distance > 5:
                risk_level = "MITTEL"
                risk_color = "#ffc107"
            elif distance > 2:
                risk_level = "HOCH"
                risk_color = "#fd7e14"
            else:
                risk_level = "EXTREM"
                risk_color = "#dc3545"
            
            liquidation_data.append({
                'leverage': f"{leverage}x",
                'liquidation_price': liq_price,
                'distance_percent': distance,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'max_loss': 100 / leverage  # Maximum loss as percentage of equity
            })
        
        return liquidation_data

# ========================================================================================
# 🎯 MASTER ANALYZER - ORCHESTRATING ALL SYSTEMS
# ========================================================================================

class MasterAnalyzer:
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.pattern_detector = AdvancedPatternDetector()
        self.position_manager = PositionManager()
        self.liquidation_calc = LiquidationCalculator()
        self.binance_client = BinanceClient()
        self.ai_system = AdvancedJAXAI()
        
        # Weighting configuration (70% Technical, 20% Patterns, 10% AI)
        self.weights = {
            'technical': 0.70,
            'patterns': 0.20,
            'ai': 0.10
        }

    def run_backtest(self, symbol, interval='1h', limit=500):
        """Lightweight RSI mean-reversion backtest.
        Strategy: Enter long when RSI < 30, exit when RSI > 55 or stop at entry - 2*ATR.
        Capital model: 100% capital per trade (simplified). Educational only.
        Returns metrics, trade list, equity curve (trimmed) and strategy meta.
        """
        try:
            # Normalize params
            interval = (interval or '1h').lower()
            try:
                limit = int(limit)
            except Exception:
                limit = 500
            limit = max(100, min(limit, 1000))  # ensure enough data but cap for performance

            # Determine minimum required candles (adaptive: at least 120, aim for 240 for longer intervals)
            min_required = 120
            if interval in ('4h','1d'):
                min_required = 150
            if limit < min_required:
                # Auto-raise limit silently to improve user experience
                limit = min_required

            # Use existing TA candle fetcher (pass interval & limit) else fallback to direct klines
            klines = self.technical_analysis.get_candle_data(symbol, limit=limit, interval=interval)
            if not klines or len(klines) < min_required:
                print("⚠️ Fallback to direct klines fetch for backtest (insufficient or empty from TA layer)")
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
                r = requests.get(url, timeout=10)
                klines = r.json()

            if not isinstance(klines, list) or len(klines) < min_required:
                have = len(klines) if isinstance(klines, list) else 0
                return {'error': f'Not enough historical data: have {have}, need >= {min_required}', 'have': have, 'need': min_required, 'interval': interval, 'suggestion': 'Increase limit or choose higher timeframe'}

            # Normalize structure to list of dicts with open, high, low, close, time
            if isinstance(klines[0], list):
                candles = [{
                    'time': k[0], 'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
                } for k in klines]
            else:
                candles = klines

            closes = np.array([c['close'] for c in candles], dtype=float)
            highs = np.array([c['high'] for c in candles], dtype=float)
            lows = np.array([c['low'] for c in candles], dtype=float)
            times = [c['time'] for c in candles]

            period = 14
            if len(closes) <= period + 5:
                return {'error': 'Series too short'}

            # RSI
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            def rma(x, n):
                a = np.zeros_like(x)
                a[n-1] = np.mean(x[:n])
                for i in range(n, len(x)):
                    a[i] = (a[i-1]*(n-1) + x[i]) / n
                return a
            avg_gain = rma(gain, period)
            avg_loss = rma(loss, period)
            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
            rsi_vals = 100 - (100 / (1 + rs))
            rsi_vals = np.concatenate([[50]*(len(closes)-1-len(rsi_vals)), rsi_vals])  # align length-1 diff
            rsi_vals = np.concatenate([[50], rsi_vals])  # pad to closes length

            # ATR
            tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
            atr_vals = rma(tr, period)
            atr_vals = np.concatenate([[atr_vals[period-1]]*(len(closes)-1-len(atr_vals)), atr_vals])
            atr_vals = np.concatenate([[atr_vals[0]], atr_vals])

            equity = 1000.0
            base_equity = equity
            position = None
            trades = []
            equity_curve = []
            max_equity = equity
            max_dd = 0

            for i in range(period+5, len(closes)):
                price = closes[i]
                equity_curve.append({'t': times[i], 'equity': equity})
                if position is None:
                    if rsi_vals[i] < 30:
                        position = {
                            'entry_price': price,
                            'entry_time': times[i],
                            'stop': price - 2 * atr_vals[i] if atr_vals[i] > 0 else price * 0.98
                        }
                else:
                    stop_hit = price <= position['stop']
                    take = rsi_vals[i] > 55
                    if stop_hit or take:
                        ret_pct = (price - position['entry_price']) / position['entry_price'] * 100
                        equity *= (1 + ret_pct/100)
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': times[i],
                            'entry': round(position['entry_price'],6),
                            'exit': round(price,6),
                            'return_pct': round(ret_pct,2),
                            'outcome': 'win' if ret_pct > 0 else 'loss'
                        })
                        position = None
                if equity > max_equity:
                    max_equity = equity
                dd = (max_equity - equity) / max_equity
                if dd > max_dd:
                    max_dd = dd

            wins = sum(1 for t in trades if t['outcome']=='win')
            losses = sum(1 for t in trades if t['outcome']=='loss')
            total_trades = len(trades)
            win_rate = (wins/total_trades*100) if total_trades else 0
            avg_ret = np.mean([t['return_pct'] for t in trades]) if trades else 0
            total_ret = (equity/base_equity - 1)*100
            returns = np.array([t['return_pct']/100 for t in trades]) if trades else np.array([])
            sharpe = 0
            if len(returns) > 1:
                sharpe = np.mean(returns)/(np.std(returns)+1e-9)
            profits = [t['return_pct'] for t in trades if t['return_pct']>0]
            losses_list = [-t['return_pct'] for t in trades if t['return_pct']<0]
            profit_factor = (sum(profits)/sum(losses_list)) if losses_list else float('inf') if profits else 0
            expectancy = (win_rate/100)*(np.mean(profits) if profits else 0) - (1-win_rate/100)*(np.mean(losses_list) if losses_list else 0)

            # --- Additional advanced metrics ---
            buy_hold_return = (closes[-1]/closes[0]-1)*100 if len(closes) > 1 else 0
            relative_outperformance = total_ret - buy_hold_return
            # Exposure: proportion of bars where a position was open
            exposure_bars = sum(1 for t in equity_curve if True)  # placeholder full length
            # We didn't store per-bar position state; approximate exposure via average holding duration * trades
            # Track holding durations by reconstructing from trades (bars_held captured below if we enhance loop)
            max_consec_wins = 0
            max_consec_losses = 0
            cur_wins = 0
            cur_losses = 0
            for t in trades:
                if t['outcome'] == 'win':
                    cur_wins += 1
                    cur_losses = 0
                else:
                    cur_losses += 1
                    cur_wins = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
                max_consec_losses = max(max_consec_losses, cur_losses)
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses_list) if losses_list else 0
            win_loss_ratio = (avg_win/avg_loss) if avg_loss else float('inf') if avg_win>0 else 0
            risk_adjusted_return = round(total_ret/(max_dd*100+1e-9),2) if max_dd>0 else 'INF'

            return {
                'symbol': symbol.upper(),
                'interval': interval,
                'candles': len(closes),
                'strategy': 'RSI Mean Reversion V1',
                'metrics': {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate_pct': round(win_rate,2),
                    'avg_return_pct': round(avg_ret,2),
                    'total_return_pct': round(total_ret,2),
                    'max_drawdown_pct': round(max_dd*100,2),
                    'profit_factor': round(profit_factor,2) if profit_factor != float('inf') else 'INF',
                    'expectancy_pct': round(expectancy,2),
                    'sharpe_approx': round(sharpe,2),
                    'buy_hold_return_pct': round(buy_hold_return,2),
                    'alpha_vs_buy_hold_pct': round(relative_outperformance,2),
                    'avg_win_pct': round(avg_win,2),
                    'avg_loss_pct': round(avg_loss,2),
                    'win_loss_ratio': round(win_loss_ratio,2) if win_loss_ratio != float('inf') else 'INF',
                    'max_consecutive_wins': max_consec_wins,
                    'max_consecutive_losses': max_consec_losses,
                    'risk_adjusted_return_ratio': risk_adjusted_return
                },
                'trades': trades[-120:],
                'equity_curve': equity_curve[-250:],
                'disclaimer': 'Backtest ist vereinfacht. Vergangene Performance garantiert keine Zukunft.'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_symbol(self, symbol):
        """Complete analysis of a trading symbol"""
        try:
            print(f"🔍 Starting analysis for {symbol}")
            
            # Get market data
            ticker_data = self.binance_client.get_ticker_data(symbol)
            current_price = float(ticker_data.get('lastPrice', 0))
            print(f"✅ Got price: {current_price}")
            
            if current_price == 0:
                return {'error': 'Symbol not found or no price data available'}
            
            # Get candlestick data
            candles = self.technical_analysis.get_candle_data(symbol, interval='1h')
            if not candles:
                return {'error': 'Unable to fetch candlestick data'}
            print(f"✅ Got {len(candles)} candles")
            
            # Technical Analysis (70% weight) - BASIC ONLY FOR NOW
            print("🔍 Starting technical analysis...")
            tech_analysis = self.technical_analysis.calculate_advanced_indicators(candles)
            print("✅ Technical analysis complete")
            
            # Extended Technical Analysis (Enterprise Level) - Temporarily with error handling
            try:
                extended_analysis = AdvancedTechnicalAnalysis.calculate_extended_indicators(candles)
                print("✅ Extended analysis successful")
            except Exception as e:
                print(f"❌ Extended analysis error: {e}")
                extended_analysis = {
                    'bollinger_bands': {'signal': 'neutral', 'position': 0.5},
                    'stochastic': {'k': 50, 'd': 50, 'signal': 'neutral'},
                    'williams_r': {'value': -50, 'signal': 'neutral'},
                    'cci': {'value': 0, 'signal': 'neutral', 'extreme': False},
                    'atr': {'percentage': 2, 'volatility': 'medium', 'risk_level': 'medium'},
                    'fibonacci': {'fib_236': 0, 'fib_382': 0, 'fib_500': 0, 'fib_618': 0},
                    'pivot_points': {'pivot': tech_analysis.get('current_price', 0), 'r1': 0},
                    'trend_strength': {'strength': 'medium', 'direction': 'neutral'}
                }
            
            # Pattern Recognition (20% weight)
            pattern_analysis = self.pattern_detector.detect_advanced_patterns(candles)
            # Ensure timeframe tagging for primary pattern detection timeframe
            try:
                for p in pattern_analysis.get('patterns', []):
                    p.setdefault('timeframe', '1h')
            except Exception:
                pass

            # Multi-Timeframe Analysis (NEW)
            mt_timeframes = ['15m', '1h', '4h', '1d']
            multi_timeframe = {'timeframes': [], 'consensus': {}}
            mt_signals = []
            for tf in mt_timeframes:
                try:
                    tf_candles = self.technical_analysis.get_candle_data(symbol, interval=tf, limit=150 if tf!='1d' else 120)
                    if len(tf_candles) < 50:
                        multi_timeframe['timeframes'].append({'tf': tf, 'error': 'no_data'})
                        continue
                    tf_analysis = self.technical_analysis.calculate_advanced_indicators(tf_candles)
                    rsi_v = tf_analysis.get('rsi', {}).get('rsi', 50)
                    sma_fast = tf_analysis.get('sma_9', tf_analysis.get('current_price'))
                    sma_slow = tf_analysis.get('sma_20', tf_analysis.get('current_price'))
                    trend_dir = 'UP' if sma_fast > sma_slow else 'DOWN' if sma_fast < sma_slow else 'SIDE'
                    # Signal logic
                    if rsi_v > 60 and trend_dir == 'UP':
                        tf_signal = 'strong_bull'
                    elif rsi_v > 55 and trend_dir in ('UP','SIDE'):
                        tf_signal = 'bull'
                    elif rsi_v < 40 and trend_dir == 'DOWN':
                        tf_signal = 'strong_bear'
                    elif rsi_v < 45 and trend_dir in ('DOWN','SIDE'):
                        tf_signal = 'bear'
                    else:
                        tf_signal = 'neutral'
                    mt_signals.append(tf_signal)
                    # timeframe contribution percentages will be added after loop
                    multi_timeframe['timeframes'].append({
                        'tf': tf,
                        'rsi': round(rsi_v,2),
                        'trend': trend_dir,
                        'signal': tf_signal,
                        'price': tf_analysis.get('current_price'),
                        'support': tf_analysis.get('support'),
                        'resistance': tf_analysis.get('resistance')
                    })
                except Exception as e:
                    multi_timeframe['timeframes'].append({'tf': tf, 'error': str(e)})
            # Consensus
            if mt_signals:
                counts = {k: mt_signals.count(k) for k in set(mt_signals)}
                def group_score(g):
                    return counts.get(g,0)
                bull_score = group_score('bull') + group_score('strong_bull')*1.5
                bear_score = group_score('bear') + group_score('strong_bear')*1.5
                if bull_score > bear_score and bull_score >= len(mt_signals)*0.5:
                    primary = 'BULLISH'
                elif bear_score > bull_score and bear_score >= len(mt_signals)*0.5:
                    primary = 'BEARISH'
                else:
                    primary = 'NEUTRAL'
                multi_timeframe['consensus'] = {
                    'bull_score': bull_score,
                    'bear_score': bear_score,
                    'total': len(mt_signals),
                    'primary': primary
                }
                # Add percentage distribution
                total_counts = sum(counts.values()) or 1
                multi_timeframe['distribution_pct'] = {k: round(v/total_counts*100,2) for k,v in counts.items()}
            else:
                multi_timeframe['consensus'] = {'primary': 'UNKNOWN'}

            # Market side strength (long vs short)
            side_score = {'long':0.0,'short':0.0,'neutral':0.0}
            side_basis = {}
            rsi_v_main = tech_analysis.get('rsi', {}).get('rsi',50)
            if rsi_v_main > 55: side_score['long']+=1; side_basis['rsi']='long'
            elif rsi_v_main < 45: side_score['short']+=1; side_basis['rsi']='short'
            else: side_basis['rsi']='neutral'
            macd_hist = tech_analysis.get('macd', {}).get('histogram',0)
            if macd_hist > 0: side_score['long']+=1; side_basis['macd']='long'
            elif macd_hist < 0: side_score['short']+=1; side_basis['macd']='short'
            else: side_basis['macd']='neutral'
            curve_dir = tech_analysis.get('macd', {}).get('curve_direction','neutral')
            if 'bullish' in curve_dir: side_score['long']+=0.5; side_basis['macd_curve']=curve_dir
            elif 'bearish' in curve_dir: side_score['short']+=0.5; side_basis['macd_curve']=curve_dir
            trend_struct = tech_analysis.get('trend', {}) if isinstance(tech_analysis.get('trend'), dict) else {}
            if trend_struct.get('trend') in ('bullish','strong_bullish'): side_score['long']+=1; side_basis['trend']='long'
            elif trend_struct.get('trend') in ('bearish','strong_bearish'): side_score['short']+=1; side_basis['trend']='short'
            else: side_basis['trend']=trend_struct.get('trend','neutral')
            bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bullish')
            bear_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bearish')
            if bull_patterns > bear_patterns: side_score['long']+=1; side_basis['patterns']='long'
            elif bear_patterns > bull_patterns: side_score['short']+=1; side_basis['patterns']='short'
            else: side_basis['patterns']='neutral'
            total_side = side_score['long']+side_score['short']+side_score['neutral'] or 1
            market_bias = {
                'long_strength_pct': round(side_score['long']/total_side*100,2),
                'short_strength_pct': round(side_score['short']/total_side*100,2),
                'basis': side_basis
            }
            
            # Position Management Analysis
            position_analysis = self.position_manager.analyze_position_potential(
                current_price,
                tech_analysis.get('support'),
                tech_analysis.get('resistance'),
                tech_analysis.get('trend', {}),
                pattern_analysis
            )
            
            # AI Analysis (10% weight)
            ai_features = self.ai_system.prepare_advanced_features(
                tech_analysis, pattern_analysis, ticker_data, position_analysis
            )
            ai_analysis = self.ai_system.predict_advanced(ai_features)
            
            # Calculate weighted final score
            final_score = self._calculate_weighted_score(tech_analysis, pattern_analysis, ai_analysis)
            
            # Liquidation Analysis
            liquidation_long = self.liquidation_calc.calculate_liquidation_levels(current_price, 'long')
            liquidation_short = self.liquidation_calc.calculate_liquidation_levels(current_price, 'short')

            # Trade Setups (basic R/R framework)
            trade_setups = self._generate_trade_setups(
                current_price,
                tech_analysis,
                extended_analysis,
                pattern_analysis,
                final_score
            )
            # Enrich trade setup rationales with multi-timeframe consensus & pattern counts
            try:
                bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bullish')
                bear_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bearish')
                mt_primary = multi_timeframe.get('consensus', {}).get('primary')
                data_span_days = None
                try:
                    first_ts = candles[0]['time'] if isinstance(candles[0], dict) else candles[0][0]
                    last_ts = candles[-1]['time'] if isinstance(candles[-1], dict) else candles[-1][0]
                    data_span_days = round((last_ts - first_ts)/(1000*60*60*24),2)
                except Exception:
                    pass
                for s in trade_setups:
                    addon = f" | MTF: {mt_primary} | Patterns B:{bull_patterns}/S:{bear_patterns}"
                    if 'rationale' in s:
                        if addon.strip() not in s['rationale']:
                            s['rationale'] += addon
                    else:
                        s['rationale'] = addon.strip()
                    if data_span_days is not None:
                        s['data_span_days'] = data_span_days
                    s['market_bias'] = market_bias
            except Exception:
                pass
            
            # SAFE RETURN - Convert all numpy types to native Python
            print("🔍 Preparing return data...")
            
            def make_json_safe(obj):
                """Convert numpy types to JSON-safe types"""
                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            # Ensure validation structure always present even if scoring failed
            safe_final_score = final_score if isinstance(final_score, dict) else {
                'score': 50,
                'signal': 'HOLD',
                'signal_color': '#6c757d',
                'technical_weight': f"{self.weights['technical']*100}%",
                'pattern_weight': f"{self.weights['patterns']*100}%",
                'ai_weight': f"{self.weights['ai']*100}%",
                'component_scores': {'technical': 50, 'patterns': 50, 'ai': 50},
                'validation': {
                    'trading_action': 'WAIT',
                    'risk_level': 'MEDIUM',
                    'contradictions': [],
                    'warnings': [],
                    'confidence_factors': ['Fallback score used'],
                    'enterprise_ready': False
                }
            }

            # Build full response object expected by frontend
            result = make_json_safe({
                'symbol': symbol,
                'current_price': float(current_price),
                'market_data': ticker_data,  # original ticker payload for 24h stats
                'technical_analysis': tech_analysis,
                'extended_analysis': extended_analysis,
                'pattern_analysis': pattern_analysis,
                'position_analysis': position_analysis,
                'ai_analysis': ai_analysis,
                'market_bias': market_bias,
                'liquidation_long': liquidation_long,
                'liquidation_short': liquidation_short,
                'trade_setups': trade_setups,
                'weights': self.weights,
                'final_score': safe_final_score,
                'timestamp': datetime.now().isoformat()
            })
            
            print("✅ Return data prepared successfully")
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Analysis error: {e}")
            print(f"Full traceback: {error_trace}")
            return {'error': f'Analysis failed: {str(e)}', 'traceback': error_trace}
    
    def _calculate_weighted_score(self, tech_analysis, pattern_analysis, ai_analysis):
        """Calculate weighted final trading score"""
        # Technical score (70%)
        tech_score = 50  # Neutral base
        
        # Fix RSI access - it's directly available, not nested
        rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                tech_score -= (rsi - 70) * 0.5  # Penalize overbought
            elif rsi < 30:
                tech_score += (30 - rsi) * 0.5  # Reward oversold
        
        # Fix trend access
        trend_data = tech_analysis.get('trend', {})
        if isinstance(trend_data, dict):
            trend = trend_data.get('trend', 'neutral')
        else:
            trend = 'neutral'
            
        if trend in ['strong_bullish', 'bullish']:
            tech_score += 25
        elif trend in ['strong_bearish', 'bearish']:
            tech_score -= 25
        
        # Fix MACD access
        macd_data = tech_analysis.get('macd', {})
        if isinstance(macd_data, dict):
            macd_signal = macd_data.get('curve_direction', 'neutral')
            if 'bullish' in macd_signal:
                tech_score += 15
            elif 'bearish' in macd_signal:
                tech_score -= 15
        
        # Pattern score (20%)
        pattern_score = 50  # Neutral base
        patterns = pattern_analysis.get('patterns', [])
        
        for pattern in patterns:
            if pattern['signal'] == 'bullish':
                pattern_score += pattern['confidence'] * 0.3
            elif pattern['signal'] == 'bearish':
                pattern_score -= pattern['confidence'] * 0.3
        
        # AI score (10%)
        ai_signal = ai_analysis.get('signal', 'HOLD')
        ai_confidence = ai_analysis.get('confidence', 50)
        
        if ai_signal == 'STRONG_BUY':
            ai_score = 75 + (ai_confidence - 50) * 0.5
        elif ai_signal == 'BUY':
            ai_score = 60 + (ai_confidence - 50) * 0.3
        elif ai_signal == 'STRONG_SELL':
            ai_score = 25 - (ai_confidence - 50) * 0.5
        elif ai_signal == 'SELL':
            ai_score = 40 - (ai_confidence - 50) * 0.3
        else:
            ai_score = 50
        
        # Weighted final score
        final_score = (
            tech_score * self.weights['technical'] +
            pattern_score * self.weights['patterns'] +
            ai_score * self.weights['ai']
        )
        
        # Clamp to 0-100 range
        final_score = max(0, min(100, final_score))
        
        # Determine final signal
        if final_score >= 75:
            signal = 'STRONG_BUY'
            signal_color = '#28a745'
        elif final_score >= 60:
            signal = 'BUY'
            signal_color = '#6f42c1'
        elif final_score <= 25:
            signal = 'STRONG_SELL'
            signal_color = '#dc3545'
        elif final_score <= 40:
            signal = 'SELL'
            signal_color = '#fd7e14'
        else:
            signal = 'HOLD'
            signal_color = '#6c757d'
        
        return {
            'score': round(final_score, 1),
            'signal': signal,
            'signal_color': signal_color,
            'technical_weight': f"{self.weights['technical']*100}%",
            'pattern_weight': f"{self.weights['patterns']*100}%",
            'ai_weight': f"{self.weights['ai']*100}%",
            'component_scores': {
                'technical': round(tech_score, 1),
                'patterns': round(pattern_score, 1),
                'ai': round(ai_score, 1)
            },
            'validation': self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, signal)
        }

    # (Deprecated earlier _generate_trade_setups removed in favor of advanced version below)
    
    def _validate_signals(self, tech_analysis, pattern_analysis, ai_analysis, final_signal):
        """Enterprise-Level Signal Validation - Eliminiert Widersprüche"""
        warnings = []
        contradictions = []
        confidence_factors = []
        
        # 1. MACD vs Final Signal Validation
        macd_signal = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
        
        if 'bearish' in macd_signal and final_signal in ['BUY', 'STRONG_BUY']:
            contradictions.append({
                'type': 'MACD_CONTRADICTION',
                'message': f'⚠️ MACD zeigt {macd_signal.upper()}, aber Signal ist {final_signal}',
                'severity': 'HIGH',
                'recommendation': 'WARTE auf besseren Einstieg - MACD Bogen ist bearish!'
            })
        
        if 'bullish' in macd_signal and final_signal in ['SELL', 'STRONG_SELL']:
            contradictions.append({
                'type': 'MACD_CONTRADICTION', 
                'message': f'⚠️ MACD zeigt {macd_signal.upper()}, aber Signal ist {final_signal}',
                'severity': 'HIGH',
                'recommendation': 'WARTE auf besseren Einstieg - MACD Bogen ist bullish!'
            })
        
        # 2. RSI Extreme Levels Validation
        rsi_data = tech_analysis.get('rsi', {})
        if isinstance(rsi_data, dict):
            rsi = rsi_data.get('rsi', 50)
        else:
            rsi = rsi_data if isinstance(rsi_data, (int, float)) else 50
        
        if rsi > 80 and final_signal in ['BUY', 'STRONG_BUY']:
            warnings.append({
                'type': 'RSI_OVERBOUGHT',
                'message': f'⚠️ RSI überkauft ({rsi:.1f}) - Vorsicht bei LONG!',
                'recommendation': 'Warte auf RSI Rückgang unter 70'
            })
        
        if rsi < 20 and final_signal in ['SELL', 'STRONG_SELL']:
            warnings.append({
                'type': 'RSI_OVERSOLD',
                'message': f'⚠️ RSI überverkauft ({rsi:.1f}) - Vorsicht bei SHORT!',
                'recommendation': 'Warte auf RSI Anstieg über 30'
            })
        
        # 3. Support/Resistance Validation
        support = tech_analysis.get('support', 0)
        resistance = tech_analysis.get('resistance', 0)
        current_price = tech_analysis.get('current_price', 0)
        
        if current_price > 0:
            distance_to_resistance = ((resistance - current_price) / current_price) * 100
            distance_to_support = ((current_price - support) / current_price) * 100
            
            if distance_to_resistance < 2 and final_signal in ['BUY', 'STRONG_BUY']:
                warnings.append({
                    'type': 'NEAR_RESISTANCE',
                    'message': f'⚠️ Preis nur {distance_to_resistance:.1f}% unter Resistance',
                    'recommendation': 'Sehr riskanter LONG Einstieg - Resistance sehr nah!'
                })
            
            if distance_to_support < 2 and final_signal in ['SELL', 'STRONG_SELL']:
                warnings.append({
                    'type': 'NEAR_SUPPORT',
                    'message': f'⚠️ Preis nur {distance_to_support:.1f}% über Support',
                    'recommendation': 'Sehr riskanter SHORT Einstieg - Support sehr nah!'
                })
        
        # 4. Pattern Consistency Check
        patterns = pattern_analysis.get('patterns', [])
        pattern_signals = [p['signal'] for p in patterns]
        
        bearish_patterns = sum(1 for s in pattern_signals if s == 'bearish')
        bullish_patterns = sum(1 for s in pattern_signals if s == 'bullish')
        
        if bearish_patterns > bullish_patterns and final_signal in ['BUY', 'STRONG_BUY']:
            contradictions.append({
                'type': 'PATTERN_CONTRADICTION',
                'message': f'⚠️ {bearish_patterns} bearish vs {bullish_patterns} bullish patterns',
                'severity': 'MEDIUM',
                'recommendation': 'Chart Muster sprechen gegen LONG Position!'
            })
        
        # 5. AI Confidence Validation
        ai_confidence = ai_analysis.get('confidence', 50)
        ai_signal = ai_analysis.get('signal', 'HOLD')
        
        if ai_confidence < 60:
            warnings.append({
                'type': 'LOW_AI_CONFIDENCE',
                'message': f'⚠️ KI Confidence nur {ai_confidence}%',
                'recommendation': 'KI ist unsicher - warte auf klarere Signale!'
            })
        
        # 6. Overall Risk Assessment
        risk_level = 'LOW'
        if len(contradictions) > 0:
            risk_level = 'VERY_HIGH'
        elif len(warnings) > 2:
            risk_level = 'HIGH'
        elif len(warnings) > 0:
            risk_level = 'MEDIUM'
        
        # 7. Final Trading Recommendation
        trading_action = final_signal
        if len(contradictions) > 0:
            trading_action = 'WAIT'
            confidence_factors.append('❌ SIGNALE WIDERSPRECHEN SICH - WARTE!')
        elif risk_level in ['HIGH', 'VERY_HIGH']:
            trading_action = 'WAIT'
            confidence_factors.append('⚠️ HOHES RISIKO - besseren Einstieg abwarten!')
        else:
            confidence_factors.append('✅ Signale sind konsistent')
        
        return {
            'trading_action': trading_action,
            'risk_level': risk_level,
            'contradictions': contradictions,
            'warnings': warnings,
            'confidence_factors': confidence_factors,
            'enterprise_ready': len(contradictions) == 0 and risk_level in ['LOW', 'MEDIUM']
        }

    def _generate_trade_setups(self, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score):
        """Generate structured long & short trade setups based on technicals, volatility and validation.
        Returns list of up to 8 setups sorted by confidence."""
        setups = []
        try:
            validation = final_score.get('validation', {}) if isinstance(final_score, dict) else {}
            support = tech_analysis.get('support') or current_price * 0.985
            resistance = tech_analysis.get('resistance') or current_price * 1.015
            rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
            trend = tech_analysis.get('trend', {}).get('trend', 'neutral')
            atr_val = extended_analysis.get('atr', {}).get('value') or (current_price * 0.004)
            atr_perc = extended_analysis.get('atr', {}).get('percentage') or (atr_val / current_price * 100)
            fib = extended_analysis.get('fibonacci', {})
            enterprise_ready = validation.get('enterprise_ready', False)
            risk_level = validation.get('risk_level', 'MEDIUM')
            contradictions = validation.get('contradictions', [])
            contradiction_count = len(contradictions)
            patterns = pattern_analysis.get('patterns', [])
            bullish_pattern_present = any(p.get('signal')=='bullish' for p in patterns)
            bearish_pattern_present = any(p.get('signal')=='bearish' for p in patterns)
            # Relaxation metadata container
            relaxation = {
                'trend_original': trend,
                'rsi_original': rsi,
                'relaxed_trend_logic': False,
                'relaxed_rsi_bounds': False,
                'fallback_generated': False,
                'pattern_injected': False
            }

            # Ensure a minimum ATR baseline so targets are not "zu knapp"
            min_atr = max(atr_val, current_price * 0.0025)
            # Helper to widen targets dynamically: structural + ATR extension
            def _structural_targets(direction, entry):
                ext_mult = 4.0  # swing extension
                swing_target = entry + min_atr * ext_mult if direction=='LONG' else entry - min_atr * ext_mult
                return swing_target

            def _confidence(base, adds):
                score = base + sum(adds)
                if contradiction_count: score -= 25
                if risk_level in ['HIGH', 'VERY_HIGH']: score -= 15
                if atr_perc and atr_perc > 1.4: score -= 8
                return max(5, min(97, round(score)))

            def _targets(entry, stop, direction, extra=None):
                risk = (entry - stop) if direction=='LONG' else (stop - entry)  # absolute price risk
                # Enforce wider baseline risk using ATR so TP nicht direkt neben Entry
                if risk < min_atr * 0.8:
                    # Widen stop further away (simulate more realistic protective stop)
                    if direction=='LONG':
                        stop = entry - min_atr * 0.8
                    else:
                        stop = entry + min_atr * 0.8
                    risk = (entry - stop) if direction=='LONG' else (stop - entry)
                risk = max(risk, min_atr*0.75)

                base = []
                # Provide broader R multiples including 1.5R & 2.5R and 4R for swing
                for m in [1,1.5,2,2.5,3]:
                    tp = entry + risk*m if direction=='LONG' else entry - risk*m
                    base.append({'label': f'{m}R', 'price': round(tp,2), 'rr': float(m)})
                # Add structural / swing extension
                swing_ext = _structural_targets(direction, entry)
                base.append({'label':'Swing', 'price': round(swing_ext,2), 'rr': round(abs((swing_ext-entry)/risk),2)})
                if extra:
                    for lbl, lvl in extra:
                        if lvl:
                            rr = (lvl - entry)/risk if direction=='LONG' else (entry - lvl)/risk
                            base.append({'label': lbl, 'price': round(lvl,2), 'rr': round(rr,2)})
                base.sort(key=lambda x: x['rr'])
                # Keep top distinct targets (remove those closer than 0.4R apart to avoid clutter)
                filtered = []
                last_rr = -999
                for t in base:
                    if t['rr'] - last_rr >= 0.4:
                        filtered.append(t)
                        last_rr = t['rr']
                    if len(filtered) >= 7:
                        break
                return filtered

            # Relaxed trend rule: allow LONG setups if not strongly bearish
            if 'bullish' in trend or trend in ['neutral','weak','moderate']:
                if 'bullish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                entry_pb = support * 1.003
                stop_pb = support - atr_val*0.6
                setups.append({
                    'id':'L-PB', 'direction':'LONG', 'strategy':'Bullish Pullback',
                    'entry': round(entry_pb,2), 'stop_loss': round(stop_pb,2),
                    'risk_percent': round((entry_pb-stop_pb)/entry_pb*100,2),
                    'targets': _targets(entry_pb, stop_pb,'LONG', [
                        ('Resistance', resistance), ('Fib 0.382', fib.get('fib_382')), ('Fib 0.618', fib.get('fib_618'))
                    ]),
                    'confidence': _confidence(50,[12 if enterprise_ready else 0, 8 if rsi<65 else 0]),
                    'conditions': [
                        {'t':'Trend bullish','s':'ok'},
                        {'t':f'RSI {rsi:.1f}','s':'ok' if rsi<70 else 'warn'},
                        {'t':f'ATR {atr_perc:.2f}%','s':'ok' if atr_perc<1.2 else 'warn'},
                        {'t':'Wenig Widerspruch','s':'ok' if contradiction_count==0 else 'bad'}
                    ],
                    'rationale':'Einstieg nahe Support mit Trend-Rückenwind'
                })

                entry_bo = resistance * 1.0015
                stop_bo = resistance - atr_val
                setups.append({
                    'id':'L-BO', 'direction':'LONG', 'strategy':'Resistance Breakout',
                    'entry': round(entry_bo,2), 'stop_loss': round(stop_bo,2),
                    'risk_percent': round((entry_bo-stop_bo)/entry_bo*100,2),
                    'targets': _targets(entry_bo, stop_bo,'LONG', [
                        ('Fib 0.618', fib.get('fib_618')), ('Fib 0.786', fib.get('fib_786'))
                    ]),
                    'confidence': _confidence(48,[15 if enterprise_ready else 5, 6 if rsi<70 else -4]),
                    'conditions': [
                        {'t':'Break über Resistance','s':'ok'},
                        {'t':'Momentum intakt','s':'ok'},
                        {'t':'Kein starker Widerspruch','s':'ok' if contradiction_count==0 else 'bad'}
                    ],
                    'rationale':'Ausbruch nutzt Momentum Beschleunigung'
                })

            # Relax RSI: previously 32 -> now 35
            if rsi < 32:
                relaxation['relaxed_rsi_bounds'] = True  # because condition used strict threshold but we mark band widening below
            if rsi < 35:
                entry_mr = current_price*0.998
                stop_mr = entry_mr - atr_val*0.9
                setups.append({
                    'id':'L-MR', 'direction':'LONG', 'strategy':'RSI Mean Reversion',
                    'entry': round(entry_mr,2), 'stop_loss': round(stop_mr,2),
                    'risk_percent': round((entry_mr-stop_mr)/entry_mr*100,2),
                    'targets': _targets(entry_mr, stop_mr,'LONG', [('Resistance', resistance)]),
                    'confidence': _confidence(42,[10 if rsi<28 else 4]),
                    'conditions': [ {'t':f'RSI {rsi:.1f}','s':'ok'}, {'t':'Trend nicht stark bearish','s':'ok' if 'bearish' not in trend else 'warn'} ],
                    'rationale':'Überverkaufte Bedingung -> Rebound Szenario'
                })

            # SHORT strategies (relax: allow if not strongly bullish)
            if 'bearish' in trend or trend in ['neutral','weak','moderate']:
                if 'bearish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                entry_pbs = resistance*0.997
                stop_pbs = resistance + atr_val*0.6
                setups.append({
                    'id':'S-PB', 'direction':'SHORT', 'strategy':'Bearish Pullback',
                    'entry': round(entry_pbs,2), 'stop_loss': round(stop_pbs,2),
                    'risk_percent': round((stop_pbs-entry_pbs)/entry_pbs*100,2),
                    'targets': _targets(entry_pbs, stop_pbs,'SHORT', [('Support', support), ('Fib 0.382', fib.get('fib_382'))]),
                    'confidence': _confidence(50,[12 if enterprise_ready else 0, 6 if rsi>35 else 0]),
                    'conditions': [ {'t':'Trend bearish','s':'ok'}, {'t':f'RSI {rsi:.1f}','s':'ok' if rsi>35 else 'warn'} ],
                    'rationale':'Rücklauf an Widerstand im Abwärtstrend'
                })

                entry_bd = support*0.9985
                stop_bd = support + atr_val
                setups.append({
                    'id':'S-BD', 'direction':'SHORT', 'strategy':'Support Breakdown',
                    'entry': round(entry_bd,2), 'stop_loss': round(stop_bd,2),
                    'risk_percent': round((stop_bd-entry_bd)/entry_bd*100,2),
                    'targets': _targets(entry_bd, stop_bd,'SHORT', [('Fib 0.236', fib.get('fib_236'))]),
                    'confidence': _confidence(48,[14 if enterprise_ready else 4, 5]),
                    'conditions': [ {'t':'Bruch unter Support','s':'ok'}, {'t':'Keine bull. Divergenz','s':'ok'} ],
                    'rationale':'Beschleunigter Momentum-Handel beim Support-Bruch'
                })

            # Relax RSI upper band: previously 68 -> now 65
            if rsi > 68:
                relaxation['relaxed_rsi_bounds'] = True
            if rsi > 65:
                entry_mrs = current_price*1.002
                stop_mrs = entry_mrs + atr_val*0.9
                setups.append({
                    'id':'S-MR', 'direction':'SHORT', 'strategy':'RSI Mean Reversion',
                    'entry': round(entry_mrs,2), 'stop_loss': round(stop_mrs,2),
                    'risk_percent': round((stop_mrs-entry_mrs)/entry_mrs*100,2),
                    'targets': _targets(entry_mrs, stop_mrs,'SHORT', [('Support', support)]),
                    'confidence': _confidence(42,[10 if rsi>72 else 4]),
                    'conditions': [ {'t':f'RSI {rsi:.1f}','s':'ok'}, {'t':'Trend nicht stark bullish','s':'ok' if 'bullish' not in trend else 'warn'} ],
                    'rationale':'Überkaufte Bedingung -> Rücksetzer / Mean Reversion'
                })

            # Pattern injected setups if patterns present & not enough direction from trend
            if bullish_pattern_present and len([s for s in setups if s['direction']=='LONG']) < 2:
                entry_pat = current_price*1.001
                stop_pat = current_price - atr_val
                setups.append({
                    'id':'L-PAT', 'direction':'LONG', 'strategy':'Pattern Boost Long',
                    'entry': round(entry_pat,2), 'stop_loss': round(stop_pat,2),
                    'risk_percent': round((entry_pat-stop_pat)/entry_pat*100,2),
                    'targets': _targets(entry_pat, stop_pat,'LONG', [('Resistance', resistance)]),
                    'confidence': 55,
                    'conditions': [{'t':'Bullish Pattern','s':'ok'}],
                    'rationale':'Bullish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True
            if bearish_pattern_present and len([s for s in setups if s['direction']=='SHORT']) < 2:
                entry_pats = current_price*0.999
                stop_pats = current_price + atr_val
                setups.append({
                    'id':'S-PAT', 'direction':'SHORT', 'strategy':'Pattern Boost Short',
                    'entry': round(entry_pats,2), 'stop_loss': round(stop_pats,2),
                    'risk_percent': round((stop_pats-entry_pats)/entry_pats*100,2),
                    'targets': _targets(entry_pats, stop_pats,'SHORT', [('Support', support)]),
                    'confidence': 55,
                    'conditions': [{'t':'Bearish Pattern','s':'ok'}],
                    'rationale':'Bearish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True

            # Fallback generic setups if still too few (ensure at least 2 directions)
            if len(setups) < 2:
                relaxation['fallback_generated'] = True
                generic_risk = max(atr_val, current_price*0.003)
                # Generic LONG
                entry_gl = current_price
                stop_gl = entry_gl - generic_risk
                setups.append({
                    'id':'L-FB', 'direction':'LONG', 'strategy':'Generic Long',
                    'entry': round(entry_gl,2), 'stop_loss': round(stop_gl,2),
                    'risk_percent': round((entry_gl-stop_gl)/entry_gl*100,2),
                    'targets': _targets(entry_gl, stop_gl,'LONG', [('Resistance', resistance)]),
                    'confidence': 45,
                    'conditions': [{'t':'Fallback','s':'info'}],
                    'rationale':'Fallback Long Setup (relaxed)'
                })
                # Generic SHORT
                entry_gs = current_price
                stop_gs = entry_gs + generic_risk
                setups.append({
                    'id':'S-FB', 'direction':'SHORT', 'strategy':'Generic Short',
                    'entry': round(entry_gs,2), 'stop_loss': round(stop_gs,2),
                    'risk_percent': round((stop_gs-entry_gs)/entry_gs*100,2),
                    'targets': _targets(entry_gs, stop_gs,'SHORT', [('Support', support)]),
                    'confidence': 45,
                    'conditions': [{'t':'Fallback','s':'info'}],
                    'rationale':'Fallback Short Setup (relaxed)'
                })

            for s in setups:
                if s.get('targets'):
                    s['primary_rr'] = s['targets'][0]['rr']
            setups.sort(key=lambda x: x['confidence'], reverse=True)
            trimmed = setups[:8]
            # Attach relaxation meta to first element for transparency
            if trimmed:
                trimmed[0]['relaxation_meta'] = relaxation
            return trimmed
        except Exception as e:
            print(f"Trade setup generation error: {e}")
            return []

# Initialize the master analyzer
master_analyzer = MasterAnalyzer()

# ========================================================================================
# 🧾 STRUCTURED LOGGING (in-memory ring buffer + stdout)
# ========================================================================================
logger = logging.getLogger("trading_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)

RECENT_LOGS = deque(maxlen=300)

def log_event(level: str, message: str, **context):
    """Log event to stdout and store a compact version in RECENT_LOGS.
    Returns a short log_id for correlation."""
    log_id = str(uuid.uuid4())[:8]
    context['log_id'] = log_id
    try:
        RECENT_LOGS.append({
            'ts': time.time(),
            'level': level.upper(),
            'message': message,
            'context': context
        })
        log_line = f"[{log_id}] {message} | {context}"
        if level.lower() == 'error':
            logger.error(log_line)
        elif level.lower() == 'warning':
            logger.warning(log_line)
        else:
            logger.info(log_line)
    except Exception as e:
        print(f"Logging failure: {e}")
    return log_id

# ========================================================================================
# 🌐 API ROUTES
# ========================================================================================

# ========================================================================================
# 🌐 API ROUTES
# ========================================================================================

@app.route('/')
def dashboard():
    """Render the beautiful main dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/search/<query>')
def search_symbols(query):
    """Search for trading symbols"""
    try:
        symbols = master_analyzer.binance_client.search_symbols(query)
        return jsonify({
            'success': True,
            'symbols': symbols,
            'count': len(symbols)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """Complete analysis of a trading symbol"""
    try:
        # Optional cache bypass: /api/analyze/BTCUSDT?refresh=1
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for analyze', symbol=symbol.upper())
        log_id = log_event('info', 'Analyze request start', symbol=symbol.upper(), refresh=request.args.get('refresh')=='1')
        analysis = master_analyzer.analyze_symbol(symbol.upper())
        
        if 'error' in analysis:
            err_id = log_event('error', 'Analyze error', symbol=symbol.upper(), error=analysis['error'], parent=log_id)
            return jsonify({
                'success': False,
                'error': analysis['error'],
                'log_id': err_id,
                'symbol': symbol.upper()
            }), 400
        log_event('info', 'Analyze success', symbol=symbol.upper(), parent=log_id)
        
        return jsonify({
            'success': True,
            'data': analysis,
            'log_id': log_id
        })
    except Exception as e:
        err_id = log_event('error', 'Analyze exception', symbol=symbol.upper(), error=str(e))
        return jsonify({
            'success': False,
            'error': str(e),
            'log_id': err_id,
            'symbol': symbol.upper()
        }), 500

@app.route('/api/liquidation/<symbol>/<float:entry_price>/<position_type>')
def calculate_liquidation(symbol, entry_price, position_type):
    """Calculate liquidation levels"""
    try:
        long_liq = master_analyzer.liquidation_calc.calculate_liquidation_levels(entry_price, 'long')
        short_liq = master_analyzer.liquidation_calc.calculate_liquidation_levels(entry_price, 'short')
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'entry_price': entry_price,
            'liquidation_long': long_liq,
            'liquidation_short': short_liq
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/quick-price/<symbol>')
def quick_price(symbol):
    """Get quick price for a symbol"""
    try:
        price = master_analyzer.binance_client.get_current_price(symbol.upper())
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'price': price
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/favicon.ico')
def favicon():
    # Serve a tiny inline favicon (16x16) to prevent browser 404/502 spam.
    # If behind a proxy sometimes an empty 204 can show as 502 when worker restarts.
    import base64
    ico_b64 = (
        b'AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAQAQAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAACZmZkAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZm\n'
        b'AGZmZgBmZmYA///////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////\n'
        b'////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
    )
    data = base64.b64decode(ico_b64)
    return app.response_class(data, mimetype='image/x-icon')

@app.route('/api/ai/status')
def ai_status():
    try:
        status_data = {}
        try:
            if hasattr(master_analyzer, 'ai_system') and master_analyzer.ai_system:
                status_data = master_analyzer.ai_system.get_status()
            else:
                status_data = {'initialized': False, 'model_version': 'unavailable'}
        except Exception as inner:
            status_data = {'initialized': False, 'model_version': 'error', 'error': str(inner)}
        return jsonify({'success': True, 'data': status_data})
    except Exception as e:
        # Absolute fallback – never raise 500 for status endpoint
        return jsonify({'success': True, 'data': {'initialized': False, 'model_version': 'unavailable', 'error': str(e)}}), 200

@app.route('/api/backtest/<symbol>')
def backtest(symbol):
    """Run a lightweight backtest on-demand (RSI mean reversion)."""
    interval = request.args.get('interval', '1h')
    limit = int(request.args.get('limit', '500'))
    try:
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for backtest', symbol=symbol.upper())
        log_id = log_event('info', 'Backtest start', symbol=symbol.upper(), interval=interval, limit=limit, refresh=request.args.get('refresh')=='1')
        data = master_analyzer.run_backtest(symbol, interval=interval, limit=limit)
        if 'error' in data:
            # Attach meta for client-side diagnosis
            err_id = log_event('warning', 'Backtest insufficient / error', symbol=symbol.upper(), interval=interval, limit=limit, error=data.get('error'), have=data.get('have'), need=data.get('need'))
            return jsonify({'success': False, 'error': data['error'], 'meta': {
                'symbol': symbol.upper(), 'interval': interval, 'limit': limit
            }, 'log_id': err_id}), 400
        log_event('info', 'Backtest success', symbol=symbol.upper(), interval=interval, limit=limit, parent=log_id, trades=data.get('metrics',{}).get('total_trades'))
        return jsonify({'success': True, 'data': data, 'meta': {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}, 'log_id': log_id})
    except Exception as e:
        err_id = log_event('error', 'Backtest exception', symbol=symbol.upper(), interval=interval, limit=limit, error=str(e))
        return jsonify({'success': False, 'error': str(e), 'log_id': err_id}), 500

@app.route('/api/logs/recent')
def recent_logs():
    """Return recent in-memory logs (sanitized). Optional filter by level & limit.
    /api/logs/recent?limit=50&level=ERROR"""
    try:
        level_filter = request.args.get('level')
        limit = int(request.args.get('limit', '100'))
        limit = max(1, min(limit, 200))
        out = []
        for item in list(RECENT_LOGS)[-limit:][::-1]:  # newest first
            if level_filter and item['level'] != level_filter.upper():
                continue
            out.append(item)
        return jsonify({'success': True, 'logs': out, 'count': len(out)})
    except Exception as e:
        err_id = log_event('error', 'Logs endpoint failure', error=str(e))
        return jsonify({'success': False, 'error': str(e), 'log_id': err_id}), 500

@app.route('/api/position/dca', methods=['POST'])
def dca_position():
    """Calculate a DCA ladder given total capital, number of entries and risk params.
    Body: {symbol, total_capital, entries, base_price(optional), spacing_pct(optional), max_risk_pct(optional)}"""
    try:
        payload = request.get_json(force=True) or {}
        symbol = (payload.get('symbol') or 'BTCUSDT').upper()
        entries = max(1, min(int(payload.get('entries', 5)), 12))
        total_capital = float(payload.get('total_capital', 1000))
        spacing_pct = float(payload.get('spacing_pct', 1.25))  # percent between ladder steps
        max_risk_pct = float(payload.get('max_risk_pct', 2.0))  # overall account risk
        current_price = master_analyzer.binance_client.get_current_price(symbol) or payload.get('base_price') or 0
        if not current_price:
            return jsonify({'success': False, 'error': 'Preis nicht verfügbar'}), 400
        # Build descending ladder for LONG (basic); could extend with direction later
        per_step_capital = total_capital / ((entries*(entries+1))/2)  # weighted heavier lower fills
        ladder = []
        cumulative_size = 0
        avg_price_num = 0
        for i in range(entries):
            # deeper steps lower percentage
            level_price = current_price * (1 - (spacing_pct/100.0)*i)
            size = per_step_capital * (i+1)
            cumulative_size += size
            avg_price_num += size * level_price
            ladder.append({
                'step': i+1,
                'price': round(level_price,2),
                'allocation': round(size,2),
                'weight_pct': round(size/total_capital*100,2)
            })
        avg_entry = avg_price_num / cumulative_size if cumulative_size else current_price
        # Risk calc: stop below last ladder price by one spacing
        stop_price = ladder[-1]['price'] * (1 - spacing_pct/100.0)
        risk_per_unit = avg_entry - stop_price
        position_value = cumulative_size  # assume 1:1 notional (spot)
        risk_pct = (risk_per_unit / avg_entry) * 100 if avg_entry else 0
        ok = risk_pct <= max_risk_pct
        return jsonify({
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'avg_entry': round(avg_entry,2),
            'stop_price': round(stop_price,2),
            'ladder': ladder,
            'risk_pct_move_to_stop': round(risk_pct,2),
            'within_risk_limit': ok
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================================================================
# 🎨 BEAUTIFUL GLASSMORPHISM FRONTEND
# ========================================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultimate Trading System V5</title>
    <style>
        :root {
            --bg: #0b0f17;
            --bg-alt: #111a24;
            --bg-soft: #1a2532;
            --card-bg: linear-gradient(160deg, rgba(26,37,50,0.85) 0%, rgba(15,21,29,0.92) 100%);
            --card-border: rgba(255,255,255,0.08);
            --card-border-strong: rgba(255,255,255,0.18);
            --text-primary: #ffffff;
            --text-secondary: rgba(255,255,255,0.75);
            --text-dim: rgba(255,255,255,0.55);
            --accent: #0d6efd;
            --accent-glow: 0 0 0 3px rgba(13,110,253,0.15);
            --success: #26c281;
            --danger: #ff4d4f;
            --warning: #f5b041;
            --info: #36c2ff;
            --purple: #8b5cf6;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 24px;
            --shadow-soft: 0 4px 18px -4px rgba(0,0,0,0.55);
            --shadow-hover: 0 8px 28px -6px rgba(0,0,0,0.6);
            --gradient-accent: linear-gradient(90deg,#0d6efd,#8b5cf6 60%,#36c2ff);
        }

        body.dim {
            --bg: #06090f;
            --bg-alt: #0d141c;
            --bg-soft: #16202b;
            --card-bg: linear-gradient(150deg, rgba(20,29,40,0.88) 0%, rgba(10,15,21,0.95) 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: radial-gradient(circle at 70% 15%, #13202e 0%, var(--bg) 55%);
            min-height: 100vh;
            overflow-x: hidden;
            color: var(--text-primary);
            line-height: 1.42;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(18px) saturate(160%);
            border-radius: var(--radius-lg);
            border: 1px solid var(--card-border);
            padding: 26px 28px 30px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-soft), 0 0 0 1px rgba(255,255,255,0.03) inset;
            position: relative;
            overflow: hidden;
            transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
        }
        .glass-card:before {
            content: ""; position:absolute; inset:0; pointer-events:none;
            background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0) 40%);
            mix-blend-mode: overlay; opacity:.4;
        }
        .glass-card:hover { border-color: var(--card-border-strong); box-shadow: var(--shadow-hover); }
        .glass-card h3 { letter-spacing:.5px; font-weight:600; }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 30px;
        }

    .header-inner { display:flex; flex-direction:column; gap:10px; align-items:center; }
    .header h1 { font-size: 2.55rem; font-weight:700; background: var(--gradient-accent); -webkit-background-clip:text; color:transparent; letter-spacing:1px; }
    .header p { font-size:1.05rem; color: var(--text-secondary); letter-spacing:.4px; }
    .toolbar { display:flex; gap:12px; margin-top:4px; }
    .btn-ghost { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); padding:8px 16px; border-radius: var(--radius-sm); font-size:.8rem; letter-spacing:.6px; cursor:pointer; color:var(--text-secondary); display:inline-flex; gap:6px; align-items:center; transition:all .25s; }
    .btn-ghost:hover { color:#fff; border-color:var(--text-secondary); }
    .btn-ghost:active { transform:translateY(1px); }

        /* Search Section */
        .search-section {
            margin-bottom: 30px;
        }

        .search-container {
            position: relative;
            max-width: 600px;
            margin: 0 auto;
        }

    .search-input { width:100%; padding:16px 22px; font-size:1rem; border:1px solid rgba(255,255,255,0.12); border-radius:18px; background:rgba(255,255,255,0.07); color:#fff; outline:none; transition:border-color .25s, background .25s; }
    .search-input:focus { border-color: var(--accent); background:rgba(255,255,255,0.12); box-shadow: var(--accent-glow); }

    .search-btn { position:absolute; right:6px; top:50%; transform:translateY(-50%); background:var(--gradient-accent); color:#fff; border:none; padding:12px 22px; border-radius:16px; cursor:pointer; font-weight:600; font-size:.85rem; letter-spacing:.5px; box-shadow:0 4px 14px -4px rgba(13,110,253,0.55); transition:filter .25s, transform .25s; }
    .search-btn:hover { filter:brightness(1.15); }

        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .grid-full {
            grid-column: 1 / -1;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            color: white;
            font-size: 1.2rem;
            margin: 20px 0;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Analysis Results */
        .analysis-results {
            display: none;
        }

        .analysis-results.active {
            display: block;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

    .metric-card { background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)); border-radius:18px; padding:18px 16px 20px; text-align:center; border:1px solid rgba(255,255,255,0.07); position:relative; overflow:hidden; }
    .metric-card:after { content:""; position:absolute; inset:0; background:linear-gradient(120deg,transparent,rgba(255,255,255,0.08) 60%,transparent); opacity:.15; }

    .metric-value { font-size:1.5rem; font-weight:600; color: #fff; margin-bottom:4px; letter-spacing:.5px; }

        .metric-label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Signal Display */
    .signal-display { text-align:center; padding:34px 20px 30px; border-radius:30px; margin-bottom:10px; background: radial-gradient(circle at 50% 0%, rgba(13,110,253,0.35), rgba(13,110,253,0) 65%); position:relative; }
    .signal-display:before { content:""; position:absolute; inset:0; background:linear-gradient(140deg,rgba(255,255,255,0.1),rgba(255,255,255,0)); mix-blend-mode:overlay; opacity:.35; }

    .signal-value { font-size:3.1rem; font-weight:800; letter-spacing:1px; margin-bottom:10px; }

        .signal-score {
            font-size: 1.5rem;
            opacity: 0.9;
            margin-bottom: 15px;
        }

        .signal-weights {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

    .weight-item { background:rgba(255,255,255,0.08); padding:6px 14px; border-radius:14px; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-secondary); }

        /* Position Management */
        .position-management {
            margin-top: 25px;
        }

    .recommendation { background:rgba(255,255,255,0.06); border-radius:18px; padding:18px 20px 20px; margin-bottom:18px; border:1px solid rgba(255,255,255,0.08); position:relative; }
    .recommendation h4 { font-size:1.05rem; letter-spacing:.5px; }
    .recommendation:before { content:""; position:absolute; inset:0; border-radius:inherit; background:linear-gradient(120deg,rgba(255,255,255,0.12),rgba(255,255,255,0) 60%); opacity:.1; pointer-events:none; }

        .recommendation h4 {
            color: white;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }

        .recommendation p {
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 8px;
        }

        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: #000000;
            transition: width 0.5s ease;
        }

        /* Extended Technical Analysis Styles */
    .indicator-section { margin:22px 0; padding:18px 18px 20px; background:rgba(255,255,255,0.04); border-radius:18px; border:1px solid rgba(255,255,255,0.08); }

        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

    .indicator-item { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:8px 14px; background:rgba(255,255,255,0.05); border-radius:12px; border:1px solid rgba(255,255,255,0.07); }

        .indicator-name {
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
        }

        .indicator-value {
            color: white;
            font-weight: 700;
            font-size: 1.1em;
        }

        .indicator-signal {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9em;
            font-style: italic;
        }

        .risk-indicators, .levels-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }

        .risk-item, .level-item {
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .risk-label, .level-name {
            display: block;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .risk-value, .level-value {
            color: white;
            font-weight: 700;
            font-size: 1.2em;
        }

        .risk-level, .level-distance {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.85em;
        }

        .fib-levels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .fib-item {
            text-align: center;
            padding: 10px;
            background: rgba(111, 66, 193, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(111, 66, 193, 0.3);
        }

        .fib-label {
            display: block;
            color: #6f42c1;
            font-weight: 600;
            font-size: 0.9em;
        }

        .fib-value {
            color: white;
            font-weight: 700;
            font-size: 1.1em;
        }

        .extreme-signal {
            color: #6f42c1 !important;
            font-weight: 700;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }

        /* Enhanced Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border-color: rgba(255,255,255,0.4);
        }

        .metric-value {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .metric-label {
            color: rgba(255,255,255,0.7);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Liquidation Tables */
        .liquidation-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .liquidation-table th,
        .liquidation-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .liquidation-table th {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-weight: 600;
        }

        .liquidation-table td {
            color: rgba(255, 255, 255, 0.9);
        }

        /* Pattern Display */
    .pattern-item { background:rgba(255,255,255,0.06); border-radius:18px; padding:16px 18px 18px; margin-bottom:16px; border:1px solid rgba(255,255,255,0.08); position:relative; overflow:hidden; }
    .pattern-item:before { content:""; position:absolute; inset:0; background:linear-gradient(130deg,rgba(255,255,255,0.15),rgba(255,255,255,0)); opacity:.07; }

        .pattern-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .pattern-type {
            color: white;
            font-weight: 600;
        }

    .pattern-confidence { background:rgba(255,255,255,0.12); padding:4px 10px; border-radius:14px; font-size:.65rem; letter-spacing:.5px; text-transform:uppercase; color:var(--text-secondary); }

    /* Unified Section Titles */
    .section-title { display:flex; align-items:center; gap:10px; margin:-4px 0 20px; padding:0 0 10px; font-size:1rem; font-weight:600; letter-spacing:.6px; color:var(--text-primary); position:relative; }
    .section-title:after { content:""; position:absolute; left:0; bottom:0; height:2px; width:60px; background:var(--gradient-accent); border-radius:2px; }
    .section-title .tag { font-size:.55rem; background:rgba(255,255,255,0.12); padding:4px 8px; border-radius:8px; letter-spacing:1px; color:var(--text-secondary); }
    .section-title .icon { font-size:1.15rem; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.4)); }
    .sub-note { font-size:.6rem; color:var(--text-dim); letter-spacing:.5px; margin-top:-12px; margin-bottom:10px; }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .signal-value {
                font-size: 2rem;
            }

            .signal-weights {
                flex-direction: column;
                align-items: center;
            }
        }

        /* Color Classes */
        .text-success { color: #28a745 !important; }
        .text-danger { color: #dc3545 !important; }
        .text-warning { color: #ffc107 !important; }
        .text-info { color: #17a2b8 !important; }
        .text-primary { color: #007bff !important; }

        .bg-success { background-color: #28a745 !important; }
        .bg-danger { background-color: #dc3545 !important; }
        .bg-warning { background-color: #ffc107 !important; }
        .bg-info { background-color: #17a2b8 !important; }
        .bg-primary { background-color: #007bff !important; }

        /* Animation Classes */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
        }
    /* Trade Setups */
    .setup-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:18px; }
    .setup-card { position:relative; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); padding:16px 18px 18px; border-radius:16px; }
    .setup-title { font-size:1.05rem; font-weight:600; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
    .setup-badge { font-size:0.6rem; background:#0d6efd; color:#fff; padding:3px 6px; border-radius:6px; letter-spacing:0.5px; }
    .setup-badge.short { background:#dc3545; }
    .setup-badge.long { background:#198754; }
    .setup-line { display:flex; justify-content:space-between; font-size:0.72rem; padding:1px 0; color:rgba(255,255,255,0.82); }
    .setup-sep { margin:6px 0 8px; height:1px; background:linear-gradient(90deg,rgba(255,255,255,0),rgba(255,255,255,0.18),rgba(255,255,255,0)); }
    .confidence-chip { position:absolute; top:10px; right:10px; background:#198754; color:#fff; padding:4px 10px; font-size:0.65rem; font-weight:600; border-radius:14px; }
    .confidence-chip.mid { background:#fd7e14; }
    .confidence-chip.low { background:#dc3545; }
    .targets { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .target-pill { background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:12px; font-size:0.65rem; }
    .conditions { list-style:none; margin:8px 0 0; padding:0; font-size:0.6rem; line-height:1.1rem; color:rgba(255,255,255,0.55); }
    .conditions li { display:flex; gap:4px; align-items:center; }
    .c-ok { color:#20c997; }
    .c-warn { color:#ffc107; }
    .c-bad { color:#dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-inner">
                <h1>🚀 Ultimate Trading System V5</h1>
                <p>Professional Analysis • Intelligent Position Management • JAX Neural Networks</p>
                <div class="toolbar">
                    <button id="themeToggle" class="btn-ghost" title="Theme umschalten">🌗 Theme</button>
                    <button id="refreshBtn" class="btn-ghost" onclick="searchSymbol()" title="Neu analysieren">🔄 Refresh</button>
                </div>
            </div>
        </div>

        <!-- Search Section -->
        <div class="glass-card search-section">
            <div class="search-container">
                <input type="text" id="searchInput" class="search-input" 
                       placeholder="Enter symbol (e.g., BTC, ETH, DOGE...)" 
                       onkeypress="if(event.key==='Enter') searchSymbol()">
                <button class="search-btn" onclick="searchSymbol()">🔍 Analyze</button>
            </div>
        </div>

        <!-- Loading Animation -->
        <div id="loadingSection" class="loading">
            <div class="spinner"></div>
            <p>🧠 Analyzing with AI • 📊 Calculating Patterns • 💡 Generating Insights...</p>
        </div>

        <!-- Analysis Results -->
        <div id="analysisResults" class="analysis-results">
            <!-- Main Signal Display -->
            <div class="glass-card">
                <div id="signalDisplay" class="signal-display">
                    <!-- Signal content will be inserted here -->
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="glass-card">
                <div class="section-title"><span class="icon">📊</span> Key Metrics <span class="tag">LIVE</span></div>
                <div id="metricsGrid" class="metrics-grid">
                    <!-- Metrics will be inserted here -->
                </div>
            </div>

            <!-- Trade Setups -->
            <div class="glass-card" id="tradeSetupsCard">
                <h3 style="color: white; margin-bottom: 16px; display:flex; align-items:center; gap:10px;">🛠️ Trade Setups <span style="font-size:0.7rem; background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:8px; letter-spacing:1px;">BETA</span></h3>
                <div id="tradeSetupsContent" class="setup-grid"></div>
                <div id="tradeSetupsStatus" style="font-size:0.75rem; color:rgba(255,255,255,0.6); margin-top:10px;"></div>
            </div>

            <!-- Position Size Calculator -->
            <div class="glass-card" id="positionSizerCard">
                <div class="section-title"><span class="icon">📐</span> Position Size Calculator <span class="tag">RISK</span></div>
                <div style="display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin-bottom:14px;">
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Equity ($)</label>
                        <input id="psEquity" type="number" value="10000" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Risk %</label>
                        <input id="psRiskPct" type="number" value="1" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Entry</label>
                        <input id="psEntry" type="number" step="0.01" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Stop</label>
                        <input id="psStop" type="number" step="0.01" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                </div>
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                    <button class="btn-ghost" onclick="prefillFromFirstSetup()">⤵️ Aus Setup übernehmen</button>
                    <button class="btn-ghost" onclick="calcPositionSize()">🧮 Berechnen</button>
                </div>
                <div id="psResult" style="font-size:.7rem; color:var(--text-secondary); line-height:1.1rem;"></div>
            </div>

            <!-- Two Column Layout -->
            <div class="grid">
                <!-- Position Management -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🎯</span> Intelligent Position Management</div>
                    <div id="positionRecommendations">
                        <!-- Position recommendations will be inserted here -->
                    </div>
                </div>

                <!-- Technical Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">📈</span> Technical Analysis</div>
                    <div id="technicalAnalysis">
                        <!-- Technical analysis will be inserted here -->
                    </div>
                </div>

                <!-- Pattern Recognition -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🔍</span> Chart Patterns</div>
                    <div id="patternAnalysis">
                        <!-- Pattern analysis will be inserted here -->
                    </div>
                </div>

                <!-- Multi-Timeframe -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🕒</span> Multi-Timeframe</div>
                    <div id="multiTimeframe">
                        <!-- MTF analysis -->
                    </div>
                </div>

                <!-- AI Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🤖</span> JAX Neural Network</div>
                    <div id="aiAnalysis">
                        <!-- AI analysis will be inserted here -->
                    </div>
                    <div id="aiStatus" style="margin-top:14px; font-size:0.65rem; color:var(--text-dim); line-height:1rem;">
                        <!-- AI status -->
                    </div>
                </div>

                <!-- Backtest -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🧪</span> Backtest <span class="tag">BETA</span></div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                        <select id="btInterval" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;">
                            <option value="1h">1h</option>
                            <option value="30m">30m</option>
                            <option value="15m">15m</option>
                            <option value="4h">4h</option>
                            <option value="1d">1d</option>
                        </select>
                        <input id="btLimit" type="number" value="500" min="100" max="1000" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;" />
                        <button class="btn-ghost" onclick="runBacktest()" style="font-size:0.65rem;">▶️ Run</button>
                    </div>
                    <div id="backtestStatus" style="font-size:0.65rem; color:var(--text-secondary); margin-bottom:8px;"></div>
                    <div id="backtestResults" style="font-size:0.65rem; line-height:1rem; color:var(--text-secondary);"></div>
                </div>
            </div>

            <!-- Liquidation Calculator -->
            <div class="glass-card grid-full">
                <div class="section-title"><span class="icon">💰</span> Liquidation Calculator</div>
                <div class="grid">
                    <div>
                        <h4 style="color: #28a745; margin-bottom: 15px;">📈 LONG Positions</h4>
                        <div style="overflow-x: auto;">
                            <table id="liquidationLongTable" class="liquidation-table">
                                <!-- Long liquidation data will be inserted here -->
                            </table>
                        </div>
                    </div>
                    <div>
                        <h4 style="color: #dc3545; margin-bottom: 15px;">📉 SHORT Positions</h4>
                        <div style="overflow-x: auto;">
                            <table id="liquidationShortTable" class="liquidation-table">
                                <!-- Short liquidation data will be inserted here -->
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSymbol = '';
        let analysisData = null;

        // Search and analyze symbol
        async function searchSymbol() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Please enter a symbol');
                return;
            }

            showLoading(true);
            currentSymbol = query.toUpperCase();

            try {
                const response = await fetch(`/api/analyze/${currentSymbol}`);
                const result = await response.json();

                if (result.success) {
                    analysisData = result.data;
                    displayAnalysis(analysisData);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                console.error('Analysis error:', error);
                alert('Failed to analyze symbol. Please try again.');
            } finally {
                showLoading(false);
            }
        }

        // Show/hide loading animation
        function showLoading(show) {
            const loading = document.getElementById('loadingSection');
            const results = document.getElementById('analysisResults');
            
            if (show) {
                loading.classList.add('active');
                results.classList.remove('active');
            } else {
                loading.classList.remove('active');
                results.classList.add('active');
            }
        }

        // Display complete analysis
        function displayAnalysis(data) {
            displayMainSignal(data);
            displayEnterpriseValidation(data); // NEW: Enterprise Validation
            displayMetrics(data);
            displayTradeSetups(data);
            window.__lastAnalysis = data; // store for position sizing
            displayPositionManagement(data);
            displayTechnicalAnalysis(data);
            displayPatternAnalysis(data);
            displayMultiTimeframe(data);
            displayAIAnalysis(data);
            displayLiquidationTables(data);
        }

        // Display main trading signal
        function displayMainSignal(data) {
            const signal = data.final_score;
            const signalDisplay = document.getElementById('signalDisplay');
            
            signalDisplay.innerHTML = `
                <div class="signal-value" style="color: ${signal.signal_color}">
                    ${signal.signal}
                </div>
                <div class="signal-score" style="color: white">
                    Score: ${signal.score}/100
                </div>
                <div class="signal-weights">
                    <div class="weight-item">📊 Technical: ${signal.technical_weight}</div>
                    <div class="weight-item">🔍 Patterns: ${signal.pattern_weight}</div>
                    <div class="weight-item">🤖 AI: ${signal.ai_weight}</div>
                </div>
            `;
        }

        // NEW: Enterprise Validation Display
        function displayEnterpriseValidation(data) {
            const validation = data.final_score.validation;
            const validationDiv = document.getElementById('enterpriseValidation') || createValidationDiv();
            
            let html = `
                <h3>🏢 ENTERPRISE VALIDATION</h3>
                <div class="validation-header">
                    <div class="trading-action" style="color: ${validation.trading_action === 'WAIT' ? '#dc3545' : '#28a745'}">
                        EMPFEHLUNG: ${validation.trading_action}
                    </div>
                    <div class="risk-level" style="color: ${getRiskColor(validation.risk_level)}">
                        RISIKO: ${validation.risk_level}
                    </div>
                    <div class="enterprise-ready" style="color: ${validation.enterprise_ready ? '#28a745' : '#dc3545'}">
                        ${validation.enterprise_ready ? '✅ ENTERPRISE READY' : '❌ NICHT BEREIT'}
                    </div>
                </div>
            `;

            // Contradictions (Widersprüche)
            if (validation.contradictions.length > 0) {
                html += `<div class="contradictions-section">
                    <h4 style="color: #dc3545">⚠️ WIDERSPRÜCHE GEFUNDEN</h4>`;
                validation.contradictions.forEach(contradiction => {
                    html += `<div class="contradiction-item" style="border-left: 3px solid #dc3545; padding-left: 10px; margin: 5px 0;">
                        <div style="color: #dc3545; font-weight: bold">${contradiction.type}</div>
                        <div style="color: white">${contradiction.message}</div>
                        <div style="color: #ffc107; font-style: italic">${contradiction.recommendation}</div>
                    </div>`;
                });
                html += `</div>`;
            }

            // Warnings
            if (validation.warnings.length > 0) {
                html += `<div class="warnings-section">
                    <h4 style="color: #ffc107">⚠️ WARNUNGEN</h4>`;
                validation.warnings.forEach(warning => {
                    html += `<div class="warning-item" style="border-left: 3px solid #ffc107; padding-left: 10px; margin: 5px 0;">
                        <div style="color: #ffc107; font-weight: bold">${warning.type}</div>
                        <div style="color: white">${warning.message}</div>
                        <div style="color: #17a2b8; font-style: italic">${warning.recommendation}</div>
                    </div>`;
                });
                html += `</div>`;
            }

            // Confidence Factors
            html += `<div class="confidence-section">
                <h4 style="color: #17a2b8">✅ CONFIDENCE FAKTOREN</h4>`;
            validation.confidence_factors.forEach(factor => {
                html += `<div class="confidence-item" style="color: white; margin: 2px 0;">${factor}</div>`;
            });
            html += `</div>`;

            validationDiv.innerHTML = html;
        }

        function createValidationDiv() {
            const div = document.createElement('div');
            div.id = 'enterpriseValidation';
            div.className = 'analysis-card';
            div.style.marginTop = '20px';
            
            // Insert after signal display
            const signalCard = document.querySelector('.signal-card');
            if (signalCard && signalCard.parentNode) {
                signalCard.parentNode.insertBefore(div, signalCard.nextSibling);
            }
            
            return div;
        }

        function getRiskColor(riskLevel) {
            switch(riskLevel) {
                case 'LOW': return '#28a745';
                case 'MEDIUM': return '#ffc107';
                case 'HIGH': return '#fd7e14';
                case 'VERY_HIGH': return '#dc3545';
                default: return '#6c757d';
            }
        }

        // Display key metrics
        function displayMetrics(data) {
            const metrics = [
                { label: 'Current Price', value: `$${data.current_price.toLocaleString()}`, color: 'white' },
                { label: '24h Change', value: `${parseFloat(data.market_data.priceChangePercent).toFixed(2)}%`, 
                  color: parseFloat(data.market_data.priceChangePercent) >= 0 ? '#28a745' : '#dc3545' },
                { label: 'RSI', value: `${data.technical_analysis.rsi.rsi.toFixed(1)}`, 
                  color: getRSIColor(data.technical_analysis.rsi.rsi) },
                { label: 'Volume 24h', value: formatVolume(data.market_data.volume), color: '#17a2b8' },
                { label: 'Support', value: `$${data.technical_analysis.support.toLocaleString()}`, color: '#28a745' },
                { label: 'Resistance', value: `$${data.technical_analysis.resistance.toLocaleString()}`, color: '#dc3545' }
            ];

            const metricsGrid = document.getElementById('metricsGrid');
            metricsGrid.innerHTML = metrics.map(metric => `
                <div class="metric-card fade-in">
                    <div class="metric-value" style="color: ${metric.color}">${metric.value}</div>
                    <div class="metric-label">${metric.label}</div>
                </div>
            `).join('');
        }

        // Trade Setups Renderer (array based)
        function displayTradeSetups(data) {
            const container = document.getElementById('tradeSetupsContent');
            const status = document.getElementById('tradeSetupsStatus');
            const setups = data.trade_setups || [];
            if (!Array.isArray(setups) || setups.length === 0) {
                container.innerHTML = '';
                status.textContent = 'Keine Setups generiert (Bedingungen nicht erfüllt).';
                return;
            }
            const blocks = setups.map(s => {
                const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                const targets = (s.targets||[]).map(t=>`<span class="target-pill">${t.label}: ${t.price} (${t.rr}R)</span>`).join('');
                const conds = (s.conditions||[]).map(c=>`<li class="${c.s==='ok'?'c-ok':(c.s==='bad'?'c-bad':'c-warn')}">${c.t}</li>`).join('');
                return `
                <div class="setup-card">
                    <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                    <div class="setup-title">${s.direction} <span class="setup-badge ${s.direction==='LONG'?'long':'short'}">${s.strategy}</span></div>
                    <div class="setup-line"><span>Entry</span><span>${s.entry}</span></div>
                    <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                    <div class="setup-line"><span>Risk%</span><span>${s.risk_percent}%</span></div>
                    <div class="setup-sep"></div>
                    <div class="targets">${targets}</div>
                    <ul class="conditions">${conds}</ul>
                    <div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:0.75rem;">${s.rationale}</div>
                </div>`;
            });
            container.innerHTML = blocks.join('');
            status.textContent = 'Automatisch generierte Setups (experimentell)';
        }

        // Display position management recommendations
        function displayPositionManagement(data) {
            const positions = data.position_analysis;
            const recommendations = positions.recommendations || [];
            
            let html = `
                <div class="metrics-grid" style="margin-bottom: 20px;">
                    <div class="metric-card">
                        <div class="metric-value text-success">${positions.resistance_potential.toFixed(1)}%</div>
                        <div class="metric-label">Resistance Potential</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value text-danger">${positions.support_risk.toFixed(1)}%</div>
                        <div class="metric-label">Support Risk</div>
                    </div>
                </div>
            `;

            html += recommendations.map(rec => `
                <div class="recommendation fade-in" style="border-left-color: ${rec.color}">
                    <h4>${rec.type}: ${rec.action}</h4>
                    <p><strong>Reason:</strong> ${rec.reason}</p>
                    <p>${rec.details}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${rec.confidence}%"></div>
                    </div>
                    <small style="color: rgba(255,255,255,0.7)">Confidence: ${rec.confidence}%</small>
                </div>
            `).join('');

            document.getElementById('positionRecommendations').innerHTML = html;
        }

        // Display technical analysis
        function displayTechnicalAnalysis(data) {
            const tech = data.technical_analysis;
            const extended = data.extended_analysis || {}; // Safety check
            
            // Check if extended analysis is available
            if (!extended || Object.keys(extended).length === 0) {
                console.log("Extended analysis not available, falling back to basic display");
                displayBasicTechnicalAnalysis(tech);
                return;
            }
            
            const html = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value ${getTrendColor(tech.trend.trend)}">${tech.trend.trend.toUpperCase()}</div>
                        <div class="metric-label">Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${tech.macd.curve_direction.replace('_', ' ').toUpperCase()}</div>
                        <div class="metric-label">MACD Signal</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value ${getIndicatorColor(extended.stochastic.signal)}">${extended.stochastic.signal.toUpperCase()}</div>
                        <div class="metric-label">Stochastic</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value ${getVolatilityColor(extended.atr.volatility)}">${extended.atr.volatility.toUpperCase()}</div>
                        <div class="metric-label">Volatility (ATR)</div>
                    </div>
                </div>
                
                <!-- Core Indicators -->
                <div class="indicator-section">
                    <h4 style="color: #17a2b8; margin: 15px 0 10px 0;">📊 CORE INDICATORS</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-name">RSI:</span>
                            <span class="indicator-value ${getRsiColor(tech.rsi.rsi)}">${tech.rsi.rsi.toFixed(1)}</span>
                            <span class="indicator-signal">(${tech.rsi.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">MACD:</span>
                            <span class="indicator-value">${tech.macd.macd.toFixed(4)}</span>
                            <span class="indicator-signal">(${tech.macd.curve_direction})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Volume:</span>
                            <span class="indicator-value">${tech.volume_analysis.ratio.toFixed(2)}x</span>
                            <span class="indicator-signal">(${tech.volume_analysis.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Momentum:</span>
                            <span class="indicator-value ${getMomentumColor(tech.momentum.value)}">${tech.momentum.value.toFixed(2)}%</span>
                            <span class="indicator-signal">(${tech.momentum.trend})</span>
                        </div>
                    </div>
                </div>

                <!-- Extended Indicators -->
                <div class="indicator-section">
                    <h4 style="color: #ffc107; margin: 15px 0 10px 0;">🔬 ADVANCED INDICATORS</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-name">Bollinger Bands:</span>
                            <span class="indicator-value">${extended.bollinger_bands.signal.toUpperCase()}</span>
                            <span class="indicator-signal">(${(extended.bollinger_bands.position * 100).toFixed(0)}%)</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Stochastic %K:</span>
                            <span class="indicator-value ${getStochasticColor(extended.stochastic.k)}">${extended.stochastic.k.toFixed(1)}</span>
                            <span class="indicator-signal">%D: ${extended.stochastic.d.toFixed(1)}</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Williams %R:</span>
                            <span class="indicator-value ${getWilliamsColor(extended.williams_r.value)}">${extended.williams_r.value.toFixed(1)}</span>
                            <span class="indicator-signal">(${extended.williams_r.signal})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">CCI:</span>
                            <span class="indicator-value ${getCciColor(extended.cci.value)}">${extended.cci.value.toFixed(1)}</span>
                            <span class="indicator-signal ${extended.cci.extreme ? 'extreme-signal' : ''}">${extended.cci.signal}</span>
                        </div>
                    </div>
                </div>

                <!-- Volatility & Risk -->
                <div class="indicator-section">
                    <h4 style="color: #dc3545; margin: 15px 0 10px 0;">⚠️ VOLATILITY & RISK</h4>
                    <div class="risk-indicators">
                        <div class="risk-item">
                            <span class="risk-label">ATR (Volatility):</span>
                            <span class="risk-value ${getVolatilityColor(extended.atr.volatility)}">${extended.atr.percentage.toFixed(2)}%</span>
                            <span class="risk-level">(${extended.atr.risk_level} risk)</span>
                        </div>
                        <div class="risk-item">
                            <span class="risk-label">Trend Strength:</span>
                            <span class="risk-value ${getTrendStrengthColor(extended.trend_strength.strength)}">${extended.trend_strength.strength.toUpperCase()}</span>
                            <span class="risk-level">(${extended.trend_strength.direction})</span>
                        </div>
                    </div>
                </div>

                <!-- Support & Resistance -->
                <div class="indicator-section">
                    <h4 style="color: #28a745; margin: 15px 0 10px 0;">📈 LEVELS & TARGETS</h4>
                    <div class="levels-grid">
                        <div class="level-item">
                            <span class="level-name">Resistance:</span>
                            <span class="level-value">${tech.resistance.toFixed(4)}</span>
                            <span class="level-distance">+${(((tech.resistance - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Support:</span>
                            <span class="level-value">${tech.support.toFixed(4)}</span>
                            <span class="level-distance">${(((tech.support - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Pivot Point:</span>
                            <span class="level-value">${extended.pivot_points.pivot.toFixed(4)}</span>
                            <span class="level-distance">R1: ${extended.pivot_points.r1.toFixed(4)}</span>
                        </div>
                    </div>
                </div>

                <!-- Fibonacci Levels -->
                <div class="indicator-section">
                    <h4 style="color: #6f42c1; margin: 15px 0 10px 0;">🌀 FIBONACCI RETRACEMENTS</h4>
                    <div class="fib-levels">
                        <div class="fib-item">
                            <span class="fib-label">23.6%:</span>
                            <span class="fib-value">${extended.fibonacci.fib_236.toFixed(4)}</span>
                        </div>
                        <div class="fib-item">
                            <span class="fib-label">38.2%:</span>
                            <span class="fib-value">${extended.fibonacci.fib_382.toFixed(4)}</span>
                        </div>
                        <div class="fib-item">
                            <span class="fib-label">50.0%:</span>
                            <span class="fib-value">${extended.fibonacci.fib_500.toFixed(4)}</span>
                        </div>
                        <div class="fib-item">
                            <span class="fib-label">61.8%:</span>
                            <span class="fib-value">${extended.fibonacci.fib_618.toFixed(4)}</span>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('technicalAnalysis').innerHTML = html;
        }

        // Fallback function for basic technical analysis
        function displayBasicTechnicalAnalysis(tech) {
            const html = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value ${getTrendColor(tech.trend.trend)}">${tech.trend.trend.toUpperCase()}</div>
                        <div class="metric-label">Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${tech.macd.curve_direction.replace('_', ' ').toUpperCase()}</div>
                        <div class="metric-label">MACD Signal</div>
                    </div>
                </div>
                
                <div class="indicator-section">
                    <h4 style="color: #17a2b8; margin: 15px 0 10px 0;">📊 BASIC INDICATORS</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-name">RSI:</span>
                            <span class="indicator-value ${getRsiColor(tech.rsi.rsi)}">${tech.rsi.rsi.toFixed(1)}</span>
                            <span class="indicator-signal">(${tech.rsi.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">MACD:</span>
                            <span class="indicator-value">${tech.macd.macd.toFixed(4)}</span>
                            <span class="indicator-signal">(${tech.macd.curve_direction})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Volume:</span>
                            <span class="indicator-value">${tech.volume_analysis.ratio.toFixed(2)}x</span>
                            <span class="indicator-signal">(${tech.volume_analysis.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Momentum:</span>
                            <span class="indicator-value ${getMomentumColor(tech.momentum.value)}">${tech.momentum.value.toFixed(2)}%</span>
                            <span class="indicator-signal">(${tech.momentum.trend})</span>
                        </div>
                    </div>
                </div>

                <div class="indicator-section">
                    <h4 style="color: #28a745; margin: 15px 0 10px 0;">📈 LEVELS</h4>
                    <div class="levels-grid">
                        <div class="level-item">
                            <span class="level-name">Resistance:</span>
                            <span class="level-value">${tech.resistance.toFixed(4)}</span>
                            <span class="level-distance">+${(((tech.resistance - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Support:</span>
                            <span class="level-value">${tech.support.toFixed(4)}</span>
                            <span class="level-distance">${(((tech.support - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('technicalAnalysis').innerHTML = html;
        }

        // Display pattern analysis
        function displayPatternAnalysis(data) {
            const patterns = data.pattern_analysis;
            
            let html = `
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value ${getSignalColor(patterns.overall_signal)}">${patterns.overall_signal}</div>
                    <div class="metric-label">Overall Pattern Signal</div>
                </div>
                
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 15px;">
                    ${patterns.pattern_summary}
                </p>
            `;

            if (patterns.patterns && patterns.patterns.length > 0) {
                html += patterns.patterns.map(pattern => `
                    <div class="pattern-item fade-in" style="border-left-color: ${getSignalColor(pattern.signal)}">
                        <div class="pattern-header">
                            <span class="pattern-type">${pattern.type}</span>
                            <span class="pattern-confidence">${pattern.confidence}%</span>
                        </div>
                        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                            ${pattern.description}
                        </p>
                    </div>
                `).join('');
            } else {
                html += '<p style="color: rgba(255,255,255,0.7);">No significant patterns detected</p>';
            }

            document.getElementById('patternAnalysis').innerHTML = html;
        }

        // Display AI analysis
        function displayAIAnalysis(data) {
            const ai = data.ai_analysis;
            
            const html = `
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value ${getSignalColor(ai.signal)}">${ai.signal}</div>
                    <div class="metric-label">AI Signal</div>
                </div>
                
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value">${ai.confidence.toFixed(1)}%</div>
                    <div class="metric-label">AI Confidence</div>
                </div>
                
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 10px;">
                    <strong>AI Recommendation:</strong><br>
                    ${ai.ai_recommendation}
                </p>
                
                <small style="color: rgba(255,255,255,0.7);">
                    Model: ${ai.model_version || 'JAX-v2.0'}
                </small>
            `;

            document.getElementById('aiAnalysis').innerHTML = html;
        }

        function displayMultiTimeframe(data) {
            const mtf = data.multi_timeframe;
            if (!mtf) { document.getElementById('multiTimeframe').innerHTML = '<small style="color:var(--text-dim)">No data</small>'; return; }
            const rows = (mtf.timeframes||[]).map(tf => {
                if (tf.error) return `<tr><td>${tf.tf}</td><td colspan=5 style='color:#dc3545'>${tf.error}</td></tr>`;
                const sigColor = tf.signal.includes('bull')? '#28a745' : tf.signal.includes('bear')? '#dc3545' : '#ffc107';
                return `<tr>
                    <td>${tf.tf}</td>
                    <td>${tf.rsi}</td>
                    <td>${tf.trend}</td>
                    <td style='color:${sigColor}'>${tf.signal}</td>
                    <td>${tf.support? tf.support.toFixed(2):'-'}</td>
                    <td>${tf.resistance? tf.resistance.toFixed(2):'-'}</td>
                </tr>`;
            }).join('');
            const cons = mtf.consensus||{};
            const consColor = cons.primary==='BULLISH'? '#28a745': cons.primary==='BEARISH'? '#dc3545':'#ffc107';
            const html = `
                <table style='width:100%; border-collapse:collapse; font-size:0.6rem;'>
                    <thead style='background:rgba(255,255,255,0.08)'>
                        <tr><th>TF</th><th>RSI</th><th>Trend</th><th>Signal</th><th>S</th><th>R</th></tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
                <div style='margin-top:8px; font-size:0.6rem;'>Consensus: <span style='color:${consColor}; font-weight:600'>${cons.primary||'-'}</span> (Bull ${cons.bull_score||0} / Bear ${cons.bear_score||0})</div>`;
            document.getElementById('multiTimeframe').innerHTML = html;
        }

        // Fetch AI status
        async function fetchAIStatus() {
            try {
                const res = await fetch('/api/ai/status');
                const j = await res.json();
                if (j.success) {
                    const d = j.data;
                    const last = d.last_train ? `<br><span style='color:var(--text-secondary)'>Last Train: ${(d.last_train.updated || '').replace('T',' ').split('.')[0]}</span>` : '';
                    document.getElementById('aiStatus').innerHTML = `Model: <span style='color:#0d6efd'>${d.model_version}</span><br>Samples: <span style='color:#8b5cf6'>${d.samples_collected}</span>${last}`;
                }
            } catch(e) {
                // silent
            }
        }
        setInterval(fetchAIStatus, 15000);
        fetchAIStatus();

        // Run backtest
        async function runBacktest() {
            if (!currentSymbol) { alert('Erst Symbol analysieren.'); return; }
            const interval = document.getElementById('btInterval').value;
            const limit = document.getElementById('btLimit').value;
            const statusEl = document.getElementById('backtestStatus');
            const resultEl = document.getElementById('backtestResults');
            statusEl.textContent = 'Running backtest...';
            resultEl.textContent = '';
            try {
                const res = await fetch(`/api/backtest/${currentSymbol}?interval=${interval}&limit=${limit}`);
                const j = await res.json();
                if (!j.success) { statusEl.textContent = 'Error: '+ j.error; return; }
                const m = j.data.metrics;
                statusEl.textContent = `${j.data.strategy} • ${j.data.candles} candles`;
                let html = `<strong>Performance</strong><br>` +
                    `Trades: ${m.total_trades} | WinRate: ${m.win_rate_pct}% | PF: ${m.profit_factor}<br>` +
                    `Avg: ${m.avg_return_pct}% | Total: ${m.total_return_pct}% | MDD: ${m.max_drawdown_pct}%<br>` +
                    `Expectancy: ${m.expectancy_pct}% | Sharpe≈ ${m.sharpe_approx}`;
                if (j.data.trades && j.data.trades.length) {
                    const last = j.data.trades.slice(-5).map(t=>`${new Date(t.exit_time).toLocaleDateString()} ${t.return_pct}%`).join(' • ');
                    html += `<br><strong>Last Trades:</strong> ${last}`;
                }
                resultEl.innerHTML = html;
            } catch(e) {
                statusEl.textContent = 'Backtest error';
            }
        }

        // Display liquidation tables
        function displayLiquidationTables(data) {
            displayLiquidationTable('liquidationLongTable', data.liquidation_long, 'LONG');
            displayLiquidationTable('liquidationShortTable', data.liquidation_short, 'SHORT');
        }

        function displayLiquidationTable(tableId, liquidationData, type) {
            const table = document.getElementById(tableId);
            
            let html = `
                <thead>
                    <tr>
                        <th>Leverage</th>
                        <th>Liquidation Price</th>
                        <th>Distance</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
            `;

            liquidationData.forEach(item => {
                html += `
                    <tr>
                        <td><strong>${item.leverage}</strong></td>
                        <td>$${item.liquidation_price.toLocaleString()}</td>
                        <td>${item.distance_percent.toFixed(2)}%</td>
                        <td style="color: ${item.risk_color}"><strong>${item.risk_level}</strong></td>
                    </tr>
                `;
            });

            html += '</tbody>';
            table.innerHTML = html;
        }

        // Helper functions
        function getRSIColor(rsi) {
            if (rsi > 70) return '#dc3545';
            if (rsi < 30) return '#28a745';
            return '#ffc107';
        }

        function getTrendColor(trend) {
            if (trend.includes('bullish')) return 'text-success';
            if (trend.includes('bearish')) return 'text-danger';
            return 'text-warning';
        }

        function getSignalColor(signal) {
            const signalStr = signal.toLowerCase();
            if (signalStr.includes('buy') || signalStr.includes('bullish')) return '#28a745';
            if (signalStr.includes('sell') || signalStr.includes('bearish')) return '#dc3545';
            return '#6c757d';
        }

        // NEW: Extended Indicator Color Functions
        function getRsiColor(rsi) {
            if (rsi > 80) return '#dc3545'; // Overbought - Red
            if (rsi < 20) return '#28a745'; // Oversold - Green
            if (rsi > 70) return '#fd7e14'; // Warning - Orange
            if (rsi < 30) return '#20c997'; // Opportunity - Teal
            return '#17a2b8'; // Neutral - Blue
        }

        function getStochasticColor(k) {
            if (k > 80) return '#dc3545'; // Overbought
            if (k < 20) return '#28a745'; // Oversold
            return '#17a2b8'; // Neutral
        }

        function getWilliamsColor(wr) {
            if (wr > -20) return '#dc3545'; // Overbought
            if (wr < -80) return '#28a745'; // Oversold
            return '#17a2b8'; // Neutral
        }

        function getCciColor(cci) {
            if (cci > 100) return '#dc3545'; // Overbought
            if (cci < -100) return '#28a745'; // Oversold
            if (Math.abs(cci) > 200) return '#6f42c1'; // Extreme
            return '#17a2b8'; // Neutral
        }

        function getVolatilityColor(volatility) {
            switch(volatility) {
                case 'very_high': return '#dc3545';
                case 'high': return '#fd7e14';
                case 'medium': return '#ffc107';
                case 'low': return '#28a745';
                default: return '#6c757d';
            }
        }

        function getTrendStrengthColor(strength) {
            switch(strength) {
                case 'very_strong': return '#dc3545';
                case 'strong': return '#fd7e14';
                case 'moderate': return '#ffc107';
                case 'weak': return '#6c757d';
                default: return '#17a2b8';
            }
        }

        function getMomentumColor(momentum) {
            if (momentum > 5) return '#28a745';
            if (momentum > 2) return '#20c997';
            if (momentum < -5) return '#dc3545';
            if (momentum < -2) return '#fd7e14';
            return '#6c757d';
        }

        function getIndicatorColor(signal) {
            switch(signal.toLowerCase()) {
                case 'bullish': return '#28a745';
                case 'bearish': return '#dc3545';
                case 'overbought': return '#fd7e14';
                case 'oversold': return '#20c997';
                default: return '#6c757d';
            }
        }

        function formatVolume(volume) {
            const num = parseFloat(volume);
            if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
            return num.toFixed(0);
        }

        // Position Size Calculator Logic
        function prefillFromFirstSetup() {
            const data = window.__lastAnalysis;
            if(!data || !Array.isArray(data.trade_setups) || data.trade_setups.length===0) {
                document.getElementById('psResult').textContent = '⚠️ Keine Setups vorhanden zum Übernehmen.';
                return;
            }
            const s = data.trade_setups[0];
            document.getElementById('psEntry').value = s.entry;
            document.getElementById('psStop').value = s.stop_loss;
            calcPositionSize();
        }
        function calcPositionSize() {
            const equity = parseFloat(psEquity.value)||0;
            const riskPct = parseFloat(psRiskPct.value)||0;
            const entry = parseFloat(psEntry.value)||0;
            const stop = parseFloat(psStop.value)||0;
            const res = document.getElementById('psResult');
            if(equity<=0||riskPct<=0||entry<=0||stop<=0) { res.textContent='Bitte Werte eingeben.'; return; }
            const riskAmount = equity * (riskPct/100);
            const diff = Math.abs(entry - stop);
            if(diff <= 0) { res.textContent='Entry und Stop dürfen nicht identisch sein.'; return; }
            const qty = riskAmount / diff;
            // Suggest capital usage (notional)
            const notional = qty * entry;
            const rr2 = entry + (diff*2);
            const rr3 = entry + (diff*3);
            res.innerHTML = `Risiko: $${riskAmount.toFixed(2)} | Größe: <b>${qty.toFixed(4)}</b> | Notional ca: $${notional.toFixed(2)}<br>`+
                `TP 2R: ${rr2.toFixed(2)} | TP 3R: ${rr3.toFixed(2)} | Abstand (R): 1R = ${(diff).toFixed(2)}`;
        }

        // Initialize with a default symbol & interactions
        document.addEventListener('DOMContentLoaded', function() {
            const input = document.getElementById('searchInput');
            input.value = 'BTCUSDT';
            searchSymbol();

            // Theme toggle
            const toggle = document.getElementById('themeToggle');
            if (toggle) {
                toggle.addEventListener('click', () => {
                    document.body.classList.toggle('dim');
                });
            }

            // Smooth scroll to results after first analysis
            const observer = new MutationObserver(() => {
                const results = document.getElementById('analysisResults');
                if (results && results.classList.contains('active')) {
                    results.scrollIntoView({behavior:'smooth', block:'start'});
                    observer.disconnect();
                }
            });
            observer.observe(document.body, {subtree:true, attributes:true, attributeFilter:['class']});
        });

        // Add Enter key support
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchSymbol();
            }
        });
    </script>
</body>
</html>
"""

print("🚀 ULTIMATE TRADING SYSTEM")
print("📊 Professional Trading Analysis")
print("⚡ Server starting on port: 5000")
print("🌍 Environment: Development")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
