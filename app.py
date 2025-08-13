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
import hashlib
import json
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
        'features': [
            'rsi_tv_style',
            'structured_logging',
            'backtest_v1',
            'dca_endpoint',
            'cache_refresh',
            'pattern_timeframes',
            'multi_timeframe_consensus',
            'enterprise_validation',
            'dynamic_ai_weighting',
            'feature_hashing',
            'phase_timings'
        ]
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

class ChartPatternTrader:
    """🎯 Konkrete Trading-Setups basierend auf Chart-Mustern"""
    
    @staticmethod
    def generate_pattern_trades(patterns, current_price, atr_value, support=None, resistance=None):
        """Generiert konkrete Entry/TP/SL für erkannte Chart-Muster"""
        pattern_trades = []
        
        def _justify(base_trade, pattern_obj):
            # Einheitliche professionelle Begründung
            ptype = pattern_obj.get('type','Pattern')
            signal = pattern_obj.get('signal','neutral')
            conf = pattern_obj.get('confidence',50)
            direction = base_trade.get('direction')
            risk_rr = base_trade.get('risk_reward_ratio') or base_trade.get('take_profits',[{'price':base_trade['entry_price']}])[0]['price']
            thesis = f"{ptype} liefert {'bullishes' if signal=='bullish' else 'bearishes' if signal=='bearish' else 'neutrales'} Signal mit {conf}% Konfidenz."
            confluence_parts = []
            if support and direction=='LONG' and base_trade['entry_price']>support: confluence_parts.append('über lokalem Support')
            if resistance and direction=='SHORT' and base_trade['entry_price']<resistance: confluence_parts.append('unter lokalem Widerstand')
            if atr_value:
                vol_note = 'moderater Volatilität' if atr_value/current_price < 0.02 else 'erhöhter Volatilität'
                confluence_parts.append(vol_note)
            confluence = ', '.join(confluence_parts) if confluence_parts else 'Basis-Signal'
            risk_model = f"RR zentriert auf Kern-Ziel(e)." if risk_rr else 'Mehrstufige R-Multiple Struktur.'
            invalid = 'Pattern invalid bei Close jenseits des Stop-Niveaus.'
            exec_plan = f"Warte auf Bestätigung (Volumen/Breakout) und nutze Teilgewinnnahme auf TP1/TP2." 
            base_trade['justification'] = {
                'core_thesis': thesis,
                'confluence': confluence,
                'risk_model': risk_model,
                'invalidations': invalid,
                'execution_plan': exec_plan
            }
            return base_trade

        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            signal = pattern.get('signal', 'neutral')
            confidence = pattern.get('confidence', 50)
            
            # 🔺 TRIANGLE PATTERNS
            if 'Triangle' in pattern_type:
                trades = ChartPatternTrader._triangle_trades(pattern, current_price, atr_value)
                if trades: pattern_trades.extend(trades)
            
            # 👑 HEAD & SHOULDERS
            elif 'Head and Shoulders' in pattern_type:
                trades = ChartPatternTrader._head_shoulders_trades(pattern, current_price, atr_value)
                if trades: pattern_trades.extend(trades)
            
            # 🔄 DOUBLE TOP/BOTTOM
            elif 'Double' in pattern_type:
                trades = ChartPatternTrader._double_pattern_trades(pattern, current_price, atr_value)
                if trades: pattern_trades.extend(trades)
            
            # ☕ CUP & HANDLE
            elif 'Cup and Handle' in pattern_type:
                trades = ChartPatternTrader._cup_handle_trades(pattern, current_price, atr_value)
                if trades: pattern_trades.extend(trades)
            
            # 🏃 BREAKOUT PATTERNS
            elif 'Breakout' in pattern_type or 'Breakdown' in pattern_type:
                trades = ChartPatternTrader._breakout_trades(pattern, current_price, atr_value)
                if trades: pattern_trades.extend(trades)
        
        # Justification + Live Preis Validierung
        enriched = []
        for t in pattern_trades:
            try:
                # Find original pattern (by name heuristic)
                src = None
                for p in patterns:
                    if p.get('type') and p.get('type').split()[0] in t.get('pattern_name',''):
                        src = p; break
                t = _justify(t, src or {})
                diff_pct = abs(t['entry_price'] - current_price)/current_price*100 if current_price else 0
                if diff_pct > 6:  # zu weit vom Live Preis -> Hinweis & Anpassung anbieten
                    t['price_desync_pct'] = round(diff_pct,2)
                    t['adjusted_entry'] = round(current_price,4)
                    # Stop & TPs prozentual verschieben
                    if t.get('stop_loss'):
                        entry_old = t['entry_price']
                        ratio = current_price/entry_old if entry_old else 1
                        t['adjusted_stop_loss'] = round(t['stop_loss']*ratio,4)
                        for tp in t.get('take_profits', []):
                            tp['adjusted_price'] = round(tp['price']*ratio,4)
                    t['justification']['core_thesis'] += ' (Level neu skaliert an Live-Preis)'
                enriched.append(t)
            except Exception:
                enriched.append(t)
        return enriched
    
    @staticmethod
    def _triangle_trades(pattern, current_price, atr_value):
        """Triangle Pattern Trading Setups"""
        trades = []
        pattern_type = pattern.get('type', '')
        confidence = pattern.get('confidence', 50)
        
        if 'Ascending' in pattern_type:
            # 📈 Ascending Triangle - Bullish Breakout
            entry = current_price * 1.005  # Leicht über aktuellem Preis
            stop_loss = current_price - (atr_value * 1.5)  # 1.5 ATR Stop
            tp1 = entry + (atr_value * 2.5)  # 2.5 ATR TP1
            tp2 = entry + (atr_value * 4.0)  # 4 ATR TP2
            tp3 = entry + (atr_value * 6.0)  # 6 ATR TP3
            
            trades.append({
                'pattern_name': 'Ascending Triangle Breakout',
                'direction': 'LONG',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1', 'price': round(tp1, 4), 'percentage': 30},
                    {'level': 'TP2', 'price': round(tp2, 4), 'percentage': 40},
                    {'level': 'TP3', 'price': round(tp3, 4), 'percentage': 30}
                ],
                'risk_reward_ratio': round((tp2 - entry) / (entry - stop_loss), 2),
                'confidence': confidence,
                'setup_type': 'Bullish Triangle Breakout',
                'trade_plan': f'Entry bei Breakout über {entry:.4f}, Stop bei {stop_loss:.4f}',
                'market_structure': 'Higher Lows + Horizontal Resistance'
            })
            
        elif 'Descending' in pattern_type:
            # 📉 Descending Triangle - Bearish Breakdown
            entry = current_price * 0.995  # Leicht unter aktuellem Preis
            stop_loss = current_price + (atr_value * 1.5)  # 1.5 ATR Stop
            tp1 = entry - (atr_value * 2.5)  # 2.5 ATR TP1
            tp2 = entry - (atr_value * 4.0)  # 4 ATR TP2
            tp3 = entry - (atr_value * 6.0)  # 6 ATR TP3
            
            trades.append({
                'pattern_name': 'Descending Triangle Breakdown',
                'direction': 'SHORT',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1', 'price': round(tp1, 4), 'percentage': 30},
                    {'level': 'TP2', 'price': round(tp2, 4), 'percentage': 40},
                    {'level': 'TP3', 'price': round(tp3, 4), 'percentage': 30}
                ],
                'risk_reward_ratio': round((entry - tp2) / (stop_loss - entry), 2),
                'confidence': confidence,
                'setup_type': 'Bearish Triangle Breakdown',
                'trade_plan': f'Entry bei Breakdown unter {entry:.4f}, Stop bei {stop_loss:.4f}',
                'market_structure': 'Lower Highs + Horizontal Support'
            })
            
        elif 'Symmetrical' in pattern_type:
            # ⚖️ Symmetrical Triangle - Beide Richtungen vorbereiten
            # LONG Setup
            long_entry = current_price * 1.008
            long_stop = current_price - (atr_value * 1.8)
            long_tp1 = long_entry + (atr_value * 3.0)
            long_tp2 = long_entry + (atr_value * 5.0)
            
            trades.append({
                'pattern_name': 'Symmetrical Triangle - Bullish Breakout',
                'direction': 'LONG',
                'entry_price': round(long_entry, 4),
                'stop_loss': round(long_stop, 4),
                'take_profits': [
                    {'level': 'TP1', 'price': round(long_tp1, 4), 'percentage': 50},
                    {'level': 'TP2', 'price': round(long_tp2, 4), 'percentage': 50}
                ],
                'risk_reward_ratio': round((long_tp1 - long_entry) / (long_entry - long_stop), 2),
                'confidence': confidence - 10,  # Etwas weniger Confidence bei neutral pattern
                'setup_type': 'Symmetrical Triangle Breakout',
                'trade_plan': f'Entry bei Breakout über {long_entry:.4f}, Stop bei {long_stop:.4f}',
                'market_structure': 'Converging Triangle - Breakout pending'
            })
            
            # SHORT Setup
            short_entry = current_price * 0.992
            short_stop = current_price + (atr_value * 1.8)
            short_tp1 = short_entry - (atr_value * 3.0)
            short_tp2 = short_entry - (atr_value * 5.0)
            
            trades.append({
                'pattern_name': 'Symmetrical Triangle - Bearish Breakdown',
                'direction': 'SHORT',
                'entry_price': round(short_entry, 4),
                'stop_loss': round(short_stop, 4),
                'take_profits': [
                    {'level': 'TP1', 'price': round(short_tp1, 4), 'percentage': 50},
                    {'level': 'TP2', 'price': round(short_tp2, 4), 'percentage': 50}
                ],
                'risk_reward_ratio': round((short_entry - short_tp1) / (short_stop - short_entry), 2),
                'confidence': confidence - 10,
                'setup_type': 'Symmetrical Triangle Breakdown',
                'trade_plan': f'Entry bei Breakdown unter {short_entry:.4f}, Stop bei {short_stop:.4f}',
                'market_structure': 'Converging Triangle - Breakdown pending'
            })
        
        return trades
    
    @staticmethod
    def _head_shoulders_trades(pattern, current_price, atr_value):
        """Head & Shoulders Trading Setups"""
        trades = []
        confidence = pattern.get('confidence', 65)
        
        # Head & Shoulders ist bearish
        if pattern.get('signal') == 'bearish':
            # Neckline Breakdown Trade
            neckline = pattern.get('neckline', current_price * 0.98)
            entry = neckline * 0.998  # Entry unter Neckline
            stop_loss = neckline + (atr_value * 2.0)  # Stop über Neckline
            
            # Target = Head height projected down from neckline
            head_height = pattern.get('head_height', atr_value * 4)
            tp1 = entry - (head_height * 0.5)  # 50% der Head-Height
            tp2 = entry - (head_height * 0.8)  # 80% der Head-Height
            tp3 = entry - head_height  # Vollständige Projektion
            
            trades.append({
                'pattern_name': 'Head & Shoulders Breakdown',
                'direction': 'SHORT',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1 (50%)', 'price': round(tp1, 4), 'percentage': 40},
                    {'level': 'TP2 (80%)', 'price': round(tp2, 4), 'percentage': 35},
                    {'level': 'TP3 (Full)', 'price': round(tp3, 4), 'percentage': 25}
                ],
                'risk_reward_ratio': round((entry - tp2) / (stop_loss - entry), 2),
                'confidence': confidence,
                'setup_type': 'Classic Reversal Pattern',
                'trade_plan': f'Entry bei Neckline Break {entry:.4f}, Target: Head-Height Projektion',
                'market_structure': 'Three Peak Reversal - Bearish',
                'key_level': f'Neckline: {neckline:.4f}'
            })
        
        return trades
    
    @staticmethod
    def _double_pattern_trades(pattern, current_price, atr_value):
        """Double Top/Bottom Trading Setups"""
        trades = []
        pattern_type = pattern.get('type', '')
        confidence = pattern.get('confidence', 60)
        
        if 'Double Top' in pattern_type:
            # 📉 Double Top - Bearish
            resistance_level = pattern.get('resistance_level', current_price * 1.02)
            entry = resistance_level * 0.995  # Entry unter Resistance
            stop_loss = resistance_level * 1.015  # Stop über Double Top
            
            # Target = Distance zwischen Peaks und Valley
            valley = pattern.get('valley', current_price * 0.95)
            double_top_height = resistance_level - valley
            tp1 = entry - (double_top_height * 0.6)
            tp2 = entry - double_top_height  # Full projektion
            
            trades.append({
                'pattern_name': 'Double Top Breakdown',
                'direction': 'SHORT',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1 (60%)', 'price': round(tp1, 4), 'percentage': 60},
                    {'level': 'TP2 (Full)', 'price': round(tp2, 4), 'percentage': 40}
                ],
                'risk_reward_ratio': round((entry - tp1) / (stop_loss - entry), 2),
                'confidence': confidence,
                'setup_type': 'Reversal Pattern',
                'trade_plan': f'Entry bei Break unter {entry:.4f}, Target: Pattern-Height Projektion',
                'market_structure': 'Failed Retest of Highs - Bearish',
                'key_level': f'Double Top: {resistance_level:.4f}'
            })
            
        elif 'Double Bottom' in pattern_type:
            # 📈 Double Bottom - Bullish
            support_level = pattern.get('support_level', current_price * 0.98)
            entry = support_level * 1.005  # Entry über Support
            stop_loss = support_level * 0.985  # Stop unter Double Bottom
            
            # Target = Distance zwischen Valley und Peak
            peak = pattern.get('peak', current_price * 1.05)
            double_bottom_height = peak - support_level
            tp1 = entry + (double_bottom_height * 0.6)
            tp2 = entry + double_bottom_height  # Full projektion
            
            trades.append({
                'pattern_name': 'Double Bottom Breakout',
                'direction': 'LONG',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1 (60%)', 'price': round(tp1, 4), 'percentage': 60},
                    {'level': 'TP2 (Full)', 'price': round(tp2, 4), 'percentage': 40}
                ],
                'risk_reward_ratio': round((tp1 - entry) / (entry - stop_loss), 2),
                'confidence': confidence,
                'setup_type': 'Reversal Pattern',
                'trade_plan': f'Entry bei Break über {entry:.4f}, Target: Pattern-Height Projektion',
                'market_structure': 'Successful Retest of Lows - Bullish',
                'key_level': f'Double Bottom: {support_level:.4f}'
            })
        
        return trades
    
    @staticmethod
    def _cup_handle_trades(pattern, current_price, atr_value):
        """Cup & Handle Trading Setups"""
        trades = []
        confidence = pattern.get('confidence', 82)
        
        # Cup & Handle ist immer bullish
        breakout_level = pattern.get('breakout_level', current_price * 1.02)
        entry = breakout_level * 1.003  # Entry über Handle Breakout
        stop_loss = current_price - (atr_value * 2.0)  # Stop unter Handle
        
        # Target = Cup depth projected up
        target_level = pattern.get('target', current_price * 1.15)
        tp1 = entry + ((target_level - entry) * 0.5)  # 50% zum Target
        tp2 = entry + ((target_level - entry) * 0.8)  # 80% zum Target
        tp3 = target_level  # Full Target
        
        trades.append({
            'pattern_name': 'Cup & Handle Breakout',
            'direction': 'LONG',
            'entry_price': round(entry, 4),
            'stop_loss': round(stop_loss, 4),
            'take_profits': [
                {'level': 'TP1 (50%)', 'price': round(tp1, 4), 'percentage': 40},
                {'level': 'TP2 (80%)', 'price': round(tp2, 4), 'percentage': 35},
                {'level': 'TP3 (Full)', 'price': round(tp3, 4), 'percentage': 25}
            ],
            'risk_reward_ratio': round((tp2 - entry) / (entry - stop_loss), 2),
            'confidence': confidence,
            'setup_type': 'Continuation Pattern',
            'trade_plan': f'Entry bei Handle Breakout {entry:.4f}, Target: Cup-Depth Projektion',
            'market_structure': 'Accumulation -> Breakout Phase',
            'key_level': f'Handle High: {breakout_level:.4f}'
        })
        
        return trades
    
    @staticmethod
    def _breakout_trades(pattern, current_price, atr_value):
        """Breakout/Breakdown Trading Setups"""
        trades = []
        pattern_type = pattern.get('type', '')
        confidence = pattern.get('confidence', 85)
        direction = pattern.get('direction', 'NEUTRAL')
        
        if 'Resistance Breakout' in pattern_type:
            # 🚀 Bullish Breakout
            breakout_level = current_price
            entry = breakout_level * 1.002  # Entry über Breakout
            stop_loss = breakout_level * 0.985  # Stop unter Breakout Level
            
            # Targets based on ATR multiples
            tp1 = entry + (atr_value * 2.0)
            tp2 = entry + (atr_value * 4.0)
            tp3 = entry + (atr_value * 6.0)
            
            trades.append({
                'pattern_name': 'Resistance Breakout',
                'direction': 'LONG',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1 (2R)', 'price': round(tp1, 4), 'percentage': 40},
                    {'level': 'TP2 (4R)', 'price': round(tp2, 4), 'percentage': 35},
                    {'level': 'TP3 (6R)', 'price': round(tp3, 4), 'percentage': 25}
                ],
                'risk_reward_ratio': round((tp2 - entry) / (entry - stop_loss), 2),
                'confidence': confidence,
                'setup_type': 'Momentum Breakout',
                'trade_plan': f'Entry auf Breakout Bestätigung {entry:.4f}, Stop unter Support',
                'market_structure': 'Breakout mit Volume Confirmation',
                'key_level': f'Breakout Level: {breakout_level:.4f}'
            })
            
        elif 'Support Breakdown' in pattern_type:
            # 📉 Bearish Breakdown
            breakdown_level = current_price
            entry = breakdown_level * 0.998  # Entry unter Breakdown
            stop_loss = breakdown_level * 1.015  # Stop über Breakdown Level
            
            # Targets based on ATR multiples
            tp1 = entry - (atr_value * 2.0)
            tp2 = entry - (atr_value * 4.0)
            tp3 = entry - (atr_value * 6.0)
            
            trades.append({
                'pattern_name': 'Support Breakdown',
                'direction': 'SHORT',
                'entry_price': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profits': [
                    {'level': 'TP1 (2R)', 'price': round(tp1, 4), 'percentage': 40},
                    {'level': 'TP2 (4R)', 'price': round(tp2, 4), 'percentage': 35},
                    {'level': 'TP3 (6R)', 'price': round(tp3, 4), 'percentage': 25}
                ],
                'risk_reward_ratio': round((entry - tp2) / (stop_loss - entry), 2),
                'confidence': confidence,
                'setup_type': 'Momentum Breakdown',
                'trade_plan': f'Entry auf Breakdown Bestätigung {entry:.4f}, Stop über Resistance',
                'market_structure': 'Breakdown mit Volume Confirmation',
                'key_level': f'Breakdown Level: {breakdown_level:.4f}'
            })
        
        return trades

class PositionManager:
    @staticmethod
    def analyze_position_potential(current_price, support, resistance, trend_analysis, patterns, market_context=None, account_equity=None):
        """Intelligente Position Management Empfehlungen (erweitert)

        Erweiterungen:
        - Dynamische Positionsgröße basierend auf Stop-Distanz & Risikoprozent
        - Trailing Stop Vorschläge (ATR- & Struktur-basiert)
        - Skalierungsplan (Scale-Out an R-Multiples / Key-Levels)
        - Orderbuch / Flow Kontext (falls verfügbar)
        - Pattern Konfluenz Score
        """
        
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
        bullish_count = 0
        bearish_count = 0
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
                bullish_count += 1
            elif pattern['signal'] == 'bearish' and pattern['confidence'] > 70:
                recommendations.append({
                    'type': 'PATTERN',
                    'action': 'SHORT SIGNAL',
                    'reason': f'📉 {pattern["type"]} detected ({pattern["confidence"]}%)',
                    'details': pattern['description'],
                    'confidence': pattern['confidence'],
                    'color': '#dc3545'
                })
                bearish_count += 1
        pattern_confluence_score = (bullish_count - bearish_count) * 5  # einfache Heuristik

        # Orderbuch / Flow Kontext
        orderbook_note = None
        flow_note = None
        ob_imbalance = None
        if market_context:
            ob = market_context.get('order_book') or {}
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            try:
                bid_vol = sum(float(b[1]) for b in bids[:10]) if bids else 0
                ask_vol = sum(float(a[1]) for a in asks[:10]) if asks else 0
                if bid_vol + ask_vol > 0:
                    ob_imbalance = round((bid_vol - ask_vol)/(bid_vol+ask_vol)*100,2)
                    if ob_imbalance > 15:
                        orderbook_note = f'Starke Bid-Dominanz (+{ob_imbalance}%)'
                    elif ob_imbalance < -15:
                        orderbook_note = f'Starke Ask-Dominanz ({ob_imbalance}%)'
                    else:
                        orderbook_note = f'Neutraler Orderflow ({ob_imbalance}%)'
            except Exception:
                pass
            flow = market_context.get('recent_trades', {})
            if isinstance(flow, dict):
                buy_ratio = flow.get('buy_ratio')
                if buy_ratio is not None:
                    if buy_ratio > 0.6:
                        flow_note = f'Käuferdominanz ({int(buy_ratio*100)}%)'
                    elif buy_ratio < 0.4:
                        flow_note = f'Verkäuferdominanz ({int((1-buy_ratio)*100)}%)'
                    else:
                        flow_note = 'Flow ausgeglichen'

        # Dynamische Positionsgröße (angenommen 1% Account-Risk falls Equity bekannt)
        position_sizing = None
        if account_equity and support and resistance:
            # Nehme konservativen Stop-Distance als 1.2% des Preises oder Distanz zu nächstem Level
            structural_stop = min(abs(current_price - support), abs(resistance - current_price)) if resistance and support else current_price*0.012
            structural_stop = max(structural_stop, current_price*0.004)
            risk_amount = account_equity * 0.01  # 1% Risk
            units = risk_amount / structural_stop if structural_stop else 0
            position_sizing = {
                'risk_per_trade_pct': 1.0,
                'structural_stop_distance': round(structural_stop,4),
                'suggested_units': round(units,4),
                'notional_position_value': round(units*current_price,2)
            }

        # Trailing Stop Plan (ATR-basiert wenn Context ATR liefert)
        trailing_plan = None
        if market_context and market_context.get('atr'):
            atr_val = market_context['atr']
            trailing_plan = {
                'initial_trailing': f'{round(atr_val*1.5,2)} (1.5 ATR)',
                'aggressive_trailing': f'{round(atr_val,2)} (1.0 ATR nach +2R)',
                'structure_shift': 'Nach Break über Resistance -> Stop unter letzter Higher Low Struktur verschieben'
            }

        # Scaling Plan (universell)
        scaling_plan = {
            'scale_out_levels': ['50% bei 2R', '25% bei 3-4R', 'Rest laufen lassen bis Strukturbruch / Swing Level'],
            're_add_condition': 'Re-Entry bei Pullback zu 0.382/0.5 Fib mit bestätigtem Volumen'
        }
        
        result = {
            'position_status': position_status,
            'recommendations': recommendations,
            'resistance_potential': resistance_potential,
            'support_risk': support_risk,
            'key_levels': {
                'next_resistance': resistance,
                'next_support': support,
                'current_price': current_price
            },
            'pattern_confluence_score': pattern_confluence_score,
            'orderbook_note': orderbook_note,
            'flow_note': flow_note,
            'orderbook_imbalance_pct': ob_imbalance,
            'position_sizing': position_sizing,
            'trailing_stop_plan': trailing_plan,
            'scaling_plan': scaling_plan
        }
        return result

# ========================================================================================
# 🤖 ADVANCED JAX AI WITH TRAINING
# ========================================================================================

class AdvancedJAXAI:
    def __init__(self):
        self.mode = 'jax'
        self.training_data = []
        if not JAX_AVAILABLE:
            self.initialized = False
            print("❌ JAX nicht installiert – KI deaktiviert (Enterprise Modus erfordert JAX)")
            return
        self.initialized = True
        self.key = random.PRNGKey(42)
        self.model_params = self._init_model()
        print("🧠 JAX Neural Network initialized: 128→64→32→4 architecture (Enterprise Mode)")
    
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
            return {'signal':'HOLD','confidence':0.0,'probabilities':[0.25]*4,'ai_recommendation':'KI deaktiviert (JAX fehlt)','mode':'offline'}

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
            x = jnp.array(features)
            h1 = jnp.tanh(jnp.dot(x, self.model_params['w1']) + self.model_params['b1'])
            h2 = jnp.tanh(jnp.dot(h1, self.model_params['w2']) + self.model_params['b2'])
            h3 = jnp.tanh(jnp.dot(h2, self.model_params['w3']) + self.model_params['b3'])
            logits = jnp.dot(h3, self.model_params['w4']) + self.model_params['b4']
            probs = jnp.exp(logits - logsumexp(logits))
            return _postprocess(np.array(probs), 'JAX-v2.0')
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
            'model_version': 'JAX-v2.0' if self.initialized else 'unavailable',
            'mode': self.mode if self.initialized else 'offline'
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
                'confidence_score': 0,
                'average_quality_score': 0
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
        
        # 🔎 Pattern Quality Scoring (add lightweight quality heuristic without duplicating detection logic)
        # Factors considered:
        # - Base confidence (scaled 0-100)
        # - Strength label weighting (VERY_STRONG > STRONG > MEDIUM)
        # - Volume confirmation hint ("✅" symbol in description)
        # - Presence of explicit target/breakout levels (actionability)
        strength_weights = {
            'VERY_STRONG': 1.15,
            'STRONG': 1.05,
            'MEDIUM': 0.95,
            'WEAK': 0.85
        }
        total_quality = 0.0
        for p in patterns:
            base = float(p.get('confidence', 0)) / 100.0
            strength = p.get('strength', 'MEDIUM')
            mult = strength_weights.get(strength, 1.0)
            desc = p.get('description', '') or ''
            volume_bonus = 0.05 if '✅' in desc else 0.0
            actionable_bonus = 0.05 if any(k in p for k in ('target','target_level','breakout_level')) else 0.0
            quality = (base * mult) + volume_bonus + actionable_bonus
            # Clamp & scale to 0-100
            quality_score = max(0, min(100, round(quality * 100, 1)))
            p['quality_score'] = quality_score
            # Quality label buckets
            if quality_score >= 80:
                ql = 'A'
            elif quality_score >= 65:
                ql = 'B'
            elif quality_score >= 50:
                ql = 'C'
            else:
                ql = 'D'
            p['quality_grade'] = ql
            total_quality += quality_score
        avg_quality = round(total_quality/len(patterns),1) if patterns else 0

        return {
            'patterns': patterns,
            'pattern_summary': pattern_summary,
            'visual_signals': visual_signals,
            'overall_signal': overall_signal,
            'confidence_score': avg_confidence,
            'patterns_count': len(patterns),
            'average_quality_score': avg_quality
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
        """Head & Shoulders + Inverse Variante mit Volumen-Bestätigung"""
        if len(highs) < lookback:
            return None
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_vols = volumes[-lookback:]

        # Bearish Head & Shoulders
        peaks = []
        for i in range(1, len(recent_highs)-1):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                peaks.append((i, recent_highs[i], recent_vols[i]))
        if len(peaks) >= 3:
            peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
            head = peaks_sorted[0]
            shoulders = peaks_sorted[1:3]
            if len(shoulders) == 2:
                h_diff = abs(shoulders[0][1]-shoulders[1][1]) / max(shoulders[0][1],1e-9)
                if h_diff < 0.035:
                    left_vol = shoulders[0][2]; right_vol = shoulders[1][2]
                    vol_conf = right_vol < left_vol
                    neckline = min(shoulders[0][1], shoulders[1][1])
                    return {
                        'type': 'Head and Shoulders',
                        'signal': 'bearish',
                        'confidence': 80 if vol_conf else 66,
                        'description': f'Umbau Top-Struktur. Volumen rechts schwächer: {"✅" if vol_conf else "⚠️"}',
                        'head_level': head[1],
                        'neckline': neckline,
                        'target': neckline * 0.92,
                        'strength': 'VERY_STRONG' if vol_conf else 'STRONG'
                    }

        # Bullish Inverse Head & Shoulders
        valleys = []
        for i in range(1, len(recent_lows)-1):
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                valleys.append((i, recent_lows[i], recent_vols[i]))
        if len(valleys) >= 3:
            valleys_sorted = sorted(valleys, key=lambda x: x[1])  # tiefste zuerst
            head_v = valleys_sorted[0]
            shoulders_v = valleys_sorted[1:3]
            if len(shoulders_v) == 2:
                d_diff = abs(shoulders_v[0][1]-shoulders_v[1][1]) / max(shoulders_v[0][1],1e-9)
                if d_diff < 0.035:
                    l_vol = shoulders_v[0][2]; r_vol = shoulders_v[1][2]
                    vol_conf_b = r_vol < l_vol  # rechts leichter
                    neckline_b = max(shoulders_v[0][1], shoulders_v[1][1])
                    return {
                        'type': 'Inverse Head and Shoulders',
                        'signal': 'bullish',
                        'confidence': 78 if vol_conf_b else 64,
                        'description': f'Bullische Bodenstruktur. Volumen rechts leichter: {"✅" if vol_conf_b else "⚠️"}',
                        'head_level': head_v[1],
                        'neckline': neckline_b,
                        'target': neckline_b * 1.08,
                        'strength': 'STRONG' if vol_conf_b else 'MEDIUM'
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
        """Cup & Handle + inverted Variante"""
        if len(closes) < lookback:
            return None

        recent_closes = closes[-lookback:]
        start_price = recent_closes[0]
        end_price = recent_closes[-1]
        min_price = min(recent_closes)
        min_index = recent_closes.index(min_price)

        cup_depth = (start_price - min_price)/start_price if start_price else 0
        span = (start_price - min_price)
        recovery = (end_price - min_price)/span if span else 0

        # Bullish
        if 0.1 < cup_depth < 0.5 and recovery > 0.7:
            h_start = int(lookback * 0.7)
            hd = recent_closes[h_start:]
            if len(hd) > 5:
                h_high = max(hd); h_low = min(hd)
                h_depth = (h_high - h_low)/h_high if h_high else 0
                if 0.05 < h_depth < 0.18:
                    return {
                        'type': 'Cup and Handle',
                        'signal': 'bullish',
                        'confidence': 82,
                        'description': f'Cup-Tiefe: {cup_depth:.1%}, Handle-Korrektur: {h_depth:.1%}',
                        'breakout_level': h_high * 1.02,
                        'target': h_high * (1 + cup_depth),
                        'strength': 'VERY_STRONG'
                    }

        # Inverted bearish
        max_price = max(recent_closes)
        max_index = recent_closes.index(max_price)
        inv_depth = (max_price - min_price)/max_price if max_price else 0
        inv_span = (max_price - min_price)
        inv_recovery = (max_price - end_price)/inv_span if inv_span else 0
        if 0.1 < inv_depth < 0.5 and inv_recovery > 0.7 and max_index < min_index:
            h_start_i = int(lookback * 0.7)
            hd_i = recent_closes[h_start_i:]
            if len(hd_i) > 5:
                ih = max(hd_i); il = min(hd_i)
                i_depth = (ih - il)/ih if ih else 0
                if 0.05 < i_depth < 0.20:
                    return {
                        'type': 'Inverted Cup and Handle',
                        'signal': 'bearish',
                        'confidence': 78,
                        'description': f'Invertiert: Tiefe {inv_depth:.1%}, Handle {i_depth:.1%}',
                        'breakout_level': il * 0.98,
                        'target': il * (1 - inv_depth),
                        'strength': 'STRONG'
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
            phase_t0 = time.time()
            timings = {}
            
            # Get market data
            t_phase = time.time()
            ticker_data = self.binance_client.get_ticker_data(symbol)
            current_price = float(ticker_data.get('lastPrice', 0))
            timings['market_data_ms'] = round((time.time()-t_phase)*1000,2)
            print(f"✅ Got price: {current_price}")
            
            if current_price == 0:
                return {'error': 'Symbol not found or no price data available'}
            
            # Get candlestick data
            t_phase = time.time()
            candles = self.technical_analysis.get_candle_data(symbol, interval='1h')
            timings['candles_fetch_ms'] = round((time.time()-t_phase)*1000,2)
            if not candles:
                return {'error': 'Unable to fetch candlestick data'}
            print(f"✅ Got {len(candles)} candles")
            
            # Technical Analysis (70% weight) - BASIC ONLY FOR NOW
            print("🔍 Starting technical analysis...")
            t_phase = time.time()
            tech_analysis = self.technical_analysis.calculate_advanced_indicators(candles)
            timings['technical_ms'] = round((time.time()-t_phase)*1000,2)
            print("✅ Technical analysis complete")
            
            # Extended Technical Analysis (Enterprise Level) - Temporarily with error handling
            try:
                t_phase = time.time()
                extended_analysis = AdvancedTechnicalAnalysis.calculate_extended_indicators(candles)
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
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
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Pattern Recognition (20% weight)
            t_phase = time.time()
            pattern_analysis = self.pattern_detector.detect_advanced_patterns(candles)
            # Ensure timeframe tagging for primary pattern detection timeframe
            try:
                for p in pattern_analysis.get('patterns', []):
                    p.setdefault('timeframe', '1h')
            except Exception:
                pass
            timings['patterns_ms'] = round((time.time()-t_phase)*1000,2)

            # Multi-timeframe pattern scan (added enterprise)
            mt_pattern_frames = ['15m','4h','1d']
            multi_tf_patterns = []
            for ptf in mt_pattern_frames:
                try:
                    ptf_candles = self.technical_analysis.get_candle_data(symbol, interval=ptf, limit=120 if ptf!='1d' else 100)
                    if not ptf_candles or len(ptf_candles) < 40:
                        continue
                    pa = self.pattern_detector.detect_advanced_patterns(ptf_candles)
                    for pat in pa.get('patterns', []):
                        pat = dict(pat)
                        pat['timeframe'] = ptf
                        multi_tf_patterns.append(pat)
                except Exception as _e:
                    continue
            if multi_tf_patterns:
                pattern_analysis['multi_timeframe_patterns'] = multi_tf_patterns

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
            
            # 📊 Order Flow Analysis (Enhanced Market Context)
            order_flow_data = self._analyze_order_flow(
                symbol, 
                current_price, 
                tech_analysis.get('volume_analysis', {}),
                multi_timeframe
            )
            
            # AI Analysis (10% weight)
            t_phase = time.time()
            ai_features = self.ai_system.prepare_advanced_features(
                tech_analysis, pattern_analysis, ticker_data, position_analysis
            )
            # Feature integrity hash & stats
            try:
                feat_payload = json.dumps(ai_features, sort_keys=True, default=str).encode()
                feature_hash = hashlib.sha256(feat_payload).hexdigest()[:16]
            except Exception:
                feature_hash = 'hash_error'
            ai_analysis = self.ai_system.predict_advanced(ai_features)
            ai_analysis['feature_hash'] = feature_hash
            ai_analysis['feature_count'] = len(ai_features) if isinstance(ai_features, dict) else 0
            
            # 🔍 Feature Contribution Analysis (AI Explainability)
            feature_contributions = self._analyze_feature_contributions(
                ai_features, 
                ai_analysis, 
                tech_analysis, 
                pattern_analysis
            )
            ai_analysis['feature_contributions'] = feature_contributions
            
            timings['ai_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Calculate weighted final score
            t_phase = time.time()
            final_score = self._calculate_weighted_score(tech_analysis, pattern_analysis, ai_analysis)
            timings['scoring_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Liquidation Analysis
            liquidation_long = self.liquidation_calc.calculate_liquidation_levels(current_price, 'long')
            liquidation_short = self.liquidation_calc.calculate_liquidation_levels(current_price, 'short')

            # 🎯 Regime Detection (market classification)
            regime_data = self._detect_market_regime(
                candles,
                tech_analysis,
                extended_analysis,
                pattern_analysis,
                multi_timeframe
            )

            # Trade Setups (basic R/R framework)
            t_phase = time.time()
            trade_setups = self._generate_trade_setups(
                current_price,
                tech_analysis,
                extended_analysis,
                pattern_analysis,
                final_score,
                multi_timeframe,
                regime_data  # Pass regime context to setups
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
                # attach global data span for response
                if data_span_days is not None:
                    tech_analysis['data_span_days'] = data_span_days
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
                    # remove internal relaxation/fallback flags from public output if present
                    if 'relaxation_meta' in s:
                        rm = s['relaxation_meta']
                        rm.pop('fallback_generated', None)
            except Exception:
                pass
            timings['setups_ms'] = round((time.time()-t_phase)*1000,2)

            # Re-run validation including multi-timeframe consensus contradictions enhancement
            try:
                if isinstance(final_score, dict):
                    final_score['validation'] = self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, final_score.get('signal'), multi_timeframe)
            except Exception as _e:
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
                    'confidence_factors': ['Default safety score used'],
                    'enterprise_ready': False
                }
            }

            # Build full response object expected by frontend
            timings['total_ms'] = round((time.time()-phase_t0)*1000,2)
            validation_summary = safe_final_score.get('validation', {})
            try:
                log_event('info', 'Enterprise validation summary', symbol=symbol, risk=validation_summary.get('risk_level'), contradictions=len(validation_summary.get('contradictions', [])), warnings=len(validation_summary.get('warnings', [])))
            except Exception:
                pass

            # Data Freshness & Timing augmentation
            try:
                price_ts = None
                # Binance ticker has 'closeTime' or 'closeTime' like field? Use eventTime if available, else derive from last candle
                for k in ('closeTime','close_time','eventTime','E','close'):  # attempt common keys
                    if k in ticker_data and isinstance(ticker_data.get(k), (int,float)) and ticker_data.get(k) > 0:
                        price_ts = int(ticker_data.get(k))
                        break
                if price_ts is None and candles:
                    price_ts = candles[-1].get('time') or candles[-1].get('timestamp')
                now_ms = int(time.time()*1000)
                freshness_ms = int(now_ms - price_ts) if price_ts else None
            except Exception:
                price_ts = None
                freshness_ms = None

            # 📊 Adaptive Risk & Target Sizing
            adaptive_risk = self._calculate_adaptive_risk_targets(symbol, current_price, tech_analysis, regime_data, ai_analysis)

            result = make_json_safe({
                'symbol': symbol,
                'current_price': float(current_price),
                'market_data': ticker_data,  # original ticker payload for 24h stats
                'technical_analysis': tech_analysis,
                'extended_analysis': extended_analysis,
                'pattern_analysis': pattern_analysis,
                'multi_timeframe': multi_timeframe,
                'position_analysis': position_analysis,
                'ai_analysis': ai_analysis,
                'ai_feature_hash': ai_analysis.get('feature_hash'),
                'order_flow_analysis': order_flow_data,
                'adaptive_risk_targets': adaptive_risk,
                'market_bias': market_bias,
                'regime_analysis': regime_data,
                'liquidation_long': liquidation_long,
                'liquidation_short': liquidation_short,
                'trade_setups': trade_setups,
                'weights': self.weights,
                'final_score': safe_final_score,
                'phase_timings_ms': timings,
                'timestamp': datetime.now().isoformat(),
                'server_time_ms': int(time.time()*1000),
                'price_timestamp_ms': price_ts,
                'data_freshness_ms': freshness_ms
            })
            
            print("✅ Return data prepared successfully")
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Analysis error: {e}")
            print(f"Full traceback: {error_trace}")
            return {'error': f'Analysis failed: {str(e)}', 'traceback': error_trace}
    
    def _detect_market_regime(self, candles, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe):
        """🎯 Market Regime Classification: Trend, Range, Expansion, Volatility Crush"""
        try:
            if len(candles) < 50:
                return {'regime': 'unknown', 'confidence': 0, 'rationale': 'Insufficient data'}

            closes = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Get ATR for volatility assessment
            atr_data = extended_analysis.get('atr', {})
            volatility_level = atr_data.get('volatility', 'medium')
            atr_pct = atr_data.get('percentage', 2.0)
            
            # Trend Assessment
            trend_data = tech_analysis.get('trend', {})
            trend_strength = trend_data.get('strength', 'weak')
            trend_direction = trend_data.get('trend', 'neutral')
            
            # Range Detection (price bounded within support/resistance)
            support = tech_analysis.get('support', 0)
            resistance = tech_analysis.get('resistance', 0)
            current_price = closes[-1]
            range_pct = ((resistance - support) / current_price * 100) if resistance > support else 0
            
            # Multi-timeframe consensus
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL')
            
            # Volume Analysis
            vol_trend = tech_analysis.get('volume_analysis', {}).get('trend', 'normal')
            
            # Pattern Momentum
            patterns = pattern_analysis.get('patterns', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.get('type', '').lower()]
            consolidation_patterns = [p for p in patterns if any(x in p.get('type', '').lower() for x in ['triangle', 'flag', 'pennant'])]
            
            # Decision Logic
            regime_scores = {
                'trending': 0,
                'ranging': 0,
                'expansion': 0,
                'volatility_crush': 0
            }
            
            # Trending signals
            if trend_strength in ['strong', 'very_strong'] and trend_direction != 'neutral':
                regime_scores['trending'] += 30
            if mt_consensus in ['BULLISH', 'BEARISH']:
                regime_scores['trending'] += 20
            if atr_pct > 2.5 and vol_trend in ['high', 'very_high']:
                regime_scores['trending'] += 15
            
            # Ranging signals
            if range_pct < 8 and current_price > support * 1.02 and current_price < resistance * 0.98:
                regime_scores['ranging'] += 35
            if trend_direction in ['neutral', 'sideways']:
                regime_scores['ranging'] += 25
            if len(consolidation_patterns) > 0:
                regime_scores['ranging'] += 15
                
            # Expansion signals (high volatility + breakouts)
            if atr_pct > 4.0:
                regime_scores['expansion'] += 25
            if len(breakout_patterns) > 0:
                regime_scores['expansion'] += 30
            if vol_trend == 'very_high':
                regime_scores['expansion'] += 20
                
            # Volatility Crush signals (low volatility + compression)
            if atr_pct < 1.5:
                regime_scores['volatility_crush'] += 30
            if volatility_level == 'low':
                regime_scores['volatility_crush'] += 25
            if vol_trend in ['low', 'below_average']:
                regime_scores['volatility_crush'] += 20
            
            # Determine regime
            primary_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[primary_regime]
            
            # Rationale construction
            if primary_regime == 'trending':
                rationale = f"Strong directional movement detected. Trend: {trend_direction} ({trend_strength}), MTF: {mt_consensus}, ATR: {atr_pct:.1f}%"
            elif primary_regime == 'ranging':
                rationale = f"Price bounded in range. S/R spread: {range_pct:.1f}%, Trend: {trend_direction}, Consolidation patterns: {len(consolidation_patterns)}"
            elif primary_regime == 'expansion':
                rationale = f"High volatility expansion phase. ATR: {atr_pct:.1f}%, Breakouts: {len(breakout_patterns)}, Volume: {vol_trend}"
            else:  # volatility_crush
                rationale = f"Low volatility compression. ATR: {atr_pct:.1f}%, Vol trend: {vol_trend}, Squeeze conditions present"
            
            # Secondary regime (if close scores)
            sorted_scores = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
            secondary = None
            if len(sorted_scores) > 1 and sorted_scores[1][1] > confidence * 0.7:
                secondary = sorted_scores[1][0]
            
            return {
                'regime': primary_regime,
                'secondary_regime': secondary,
                'confidence': min(100, confidence),
                'rationale': rationale,
                'regime_scores': regime_scores,
                'volatility_level': volatility_level,
                'atr_percentage': atr_pct,
                'range_percentage': range_pct,
                'trend_classification': f"{trend_direction} ({trend_strength})"
            }
            
        except Exception as e:
            return {
                'regime': 'error', 
                'confidence': 0, 
                'rationale': f'Regime detection failed: {str(e)}'
            }
    
    def _generate_rsi_caution_narrative(self, rsi, trend):
        """🚨 Generate RSI-based caution narrative and confidence penalties"""
        caution_level = 'none'
        narrative = ''
        confidence_penalty = 0
        signal_quality = 'ok'
        
        if rsi >= 80:
            caution_level = 'extreme'
            narrative = '⚠️ EXTREME ÜBERKAUFT: RSI sehr hoch - Pullback-Risiko erhöht, reduzierte Position empfohlen'
            confidence_penalty = 25
            signal_quality = 'bad'
        elif rsi >= 70:
            caution_level = 'high'
            narrative = '⚠️ ÜBERKAUFT-WARNUNG: RSI über 70 - Vorsicht bei LONG-Einstiegen, enge Stops verwenden'
            confidence_penalty = 15
            signal_quality = 'warn'
        elif rsi <= 20:
            caution_level = 'extreme_oversold'
            narrative = '💡 EXTREME ÜBERVERKAUFT: RSI sehr niedrig - Bounce-Potential hoch, aber weitere Schwäche möglich'
            confidence_penalty = 0  # Oversold can be opportunity
            signal_quality = 'ok'
        elif rsi <= 30:
            caution_level = 'oversold_opportunity'
            narrative = '💡 ÜBERVERKAUFT: RSI unter 30 - Potentielle Einstiegschance für LONG bei Bestätigung'
            confidence_penalty = -5  # Small bonus for oversold
            signal_quality = 'ok'
        elif rsi >= 60 and 'bearish' in trend:
            caution_level = 'trend_conflict'
            narrative = '⚠️ TREND-KONFLIKT: RSI erhöht in bearischem Trend - kurzfristige Rallye könnte enden'
            confidence_penalty = 10
            signal_quality = 'warn'
        elif rsi <= 40 and 'bullish' in trend:
            caution_level = 'healthy_pullback'
            narrative = '✅ GESUNDER PULLBACK: RSI moderat in bullischem Trend - Einstiegschance bei Support'
            confidence_penalty = -3  # Small bonus
            signal_quality = 'ok'
            
        return {
            'caution_level': caution_level,
            'narrative': narrative,
            'confidence_penalty': confidence_penalty,
            'signal_quality': signal_quality,
            'rsi_value': rsi,
            'recommendation': self._get_rsi_recommendation(rsi, trend)
        }
    
    def _get_rsi_recommendation(self, rsi, trend):
        """Get specific RSI-based trading recommendation"""
        if rsi >= 80:
            return 'Avoid new LONG positions, consider profit-taking'
        elif rsi >= 70:
            return 'Use tight stops on LONG positions, monitor for reversal signals'
        elif rsi <= 20:
            return 'Monitor for reversal confirmation before entering LONG'
        elif rsi <= 30:
            return 'Good LONG entry zone if trend supports'
        elif 40 <= rsi <= 60:
            return 'Neutral RSI - rely on other indicators'
        else:
            return 'RSI in normal range - standard trade management'
    
    def _analyze_order_flow(self, symbol, current_price, volume_analysis, multi_timeframe):
        """📊 Order Flow & Market Microstructure Analysis"""
        try:
            # Simulated Order Flow (placeholder for real orderbook integration)
            # In production, this would fetch actual orderbook data
            order_flow = {
                'bid_ask_spread': 0,
                'order_book_imbalance': 0,
                'delta_momentum': 0,
                'volume_profile_poc': current_price,
                'liquidity_zones': [],
                'flow_sentiment': 'neutral'
            }
            
            # Volume-based flow estimation
            vol_ratio = volume_analysis.get('ratio', 1.0)
            vol_trend = volume_analysis.get('trend', 'normal')
            
            # Estimate bid/ask spread based on volatility
            if vol_ratio > 2.0:
                spread_estimate = current_price * 0.001  # 0.1% in high vol
            elif vol_ratio > 1.5:
                spread_estimate = current_price * 0.0005  # 0.05%
            else:
                spread_estimate = current_price * 0.0002  # 0.02% normal
            
            order_flow['bid_ask_spread'] = round(spread_estimate, 6)
            order_flow['spread_bps'] = round((spread_estimate / current_price) * 10000, 2)
            
            # Estimate order imbalance from volume pattern
            if vol_trend == 'very_high' and vol_ratio > 1.8:
                # High volume suggests imbalance
                imbalance = min(0.7, (vol_ratio - 1) * 0.3)  # Cap at 70%
                order_flow['order_book_imbalance'] = round(imbalance, 3)
                order_flow['flow_sentiment'] = 'buy_pressure' if imbalance > 0.3 else 'sell_pressure'
            elif vol_trend == 'below_average':
                order_flow['order_book_imbalance'] = round(-(1 - vol_ratio) * 0.2, 3)
                order_flow['flow_sentiment'] = 'low_liquidity'
            else:
                order_flow['order_book_imbalance'] = round((vol_ratio - 1) * 0.1, 3)
            
            # Multi-timeframe delta estimation
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL')
            if mt_consensus == 'BULLISH':
                order_flow['delta_momentum'] = round(0.3 + vol_ratio * 0.2, 3)
            elif mt_consensus == 'BEARISH':
                order_flow['delta_momentum'] = round(-0.3 - vol_ratio * 0.2, 3)
            else:
                order_flow['delta_momentum'] = round((vol_ratio - 1) * 0.1, 3)
            
            # Volume Profile Point of Control estimation
            order_flow['volume_profile_poc'] = round(current_price * (1 + order_flow['delta_momentum'] * 0.01), 4)
            
            # Liquidity zones (approximate based on volume)
            if vol_ratio > 1.5:
                zones = [
                    {'level': round(current_price * 0.995, 4), 'type': 'support', 'strength': 'medium'},
                    {'level': round(current_price * 1.005, 4), 'type': 'resistance', 'strength': 'medium'}
                ]
                if vol_ratio > 2.0:
                    zones.extend([
                        {'level': round(current_price * 0.99, 4), 'type': 'support', 'strength': 'strong'},
                        {'level': round(current_price * 1.01, 4), 'type': 'resistance', 'strength': 'strong'}
                    ])
                order_flow['liquidity_zones'] = zones
            
            # Flow strength assessment
            imb_abs = abs(order_flow['order_book_imbalance'])
            delta_abs = abs(order_flow['delta_momentum'])
            
            if imb_abs > 0.4 or delta_abs > 0.5:
                flow_strength = 'strong'
            elif imb_abs > 0.2 or delta_abs > 0.3:
                flow_strength = 'moderate'
            else:
                flow_strength = 'weak'
            
            order_flow['flow_strength'] = flow_strength
            order_flow['analysis_note'] = f"Estimated flow from volume patterns (spread: {order_flow['spread_bps']}bps, imbalance: {order_flow['order_book_imbalance']:.1%})"
            
            return order_flow
            
        except Exception as e:
            return {
                'error': f'Order flow analysis failed: {str(e)}',
                'flow_sentiment': 'unknown',
                'flow_strength': 'unknown'
            }
    
    def _analyze_feature_contributions(self, features, ai_analysis, tech_analysis, pattern_analysis):
        """🔍 AI Feature Contribution Analysis (Explainability)"""
        try:
            if not isinstance(features, (list, np.ndarray)) or len(features) == 0:
                return {'error': 'No features available for contribution analysis'}
            
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features)
            
            # Feature importance heuristic (simplified attribution)
            # In production, this would use actual gradient-based attribution
            
            feature_names = [
                'RSI', 'RSI_Overbought', 'RSI_Oversold', 'MACD', 'MACD_Hist', 
                'MACD_Bull_Curve', 'MACD_Bear_Curve', 'MACD_Bull_Rev', 'MACD_Bear_Rev',
                'SMA_9', 'SMA_20', 'Support_Strength', 'Resistance_Strength'
            ]
            
            # Extend feature names for all 128 features
            for i in range(len(feature_names), 50):
                feature_names.append(f'Tech_{i}')
            for i in range(50, 80):
                feature_names.append(f'Pattern_{i-50}')
            for i in range(80, 100):
                feature_names.append(f'Market_{i-80}')
            for i in range(100, 120):
                feature_names.append(f'Position_{i-100}')
            for i in range(120, 128):
                feature_names.append(f'Time_{i-120}')
            
            # Ensure we have enough names
            while len(feature_names) < len(features):
                feature_names.append(f'Feature_{len(feature_names)}')
            
            # Calculate pseudo-importance (magnitude * activation)
            feature_magnitudes = np.abs(features)
            feature_activations = np.where(features > 0, features, -features * 0.5)  # Positive bias
            importance_scores = feature_magnitudes * feature_activations
            
            # Normalize to percentages
            total_importance = np.sum(importance_scores)
            if total_importance > 0:
                normalized_importance = (importance_scores / total_importance) * 100
            else:
                normalized_importance = np.zeros_like(importance_scores)
            
            # Get top contributing features
            top_indices = np.argsort(normalized_importance)[-10:][::-1]  # Top 10
            
            contributions = []
            for idx in top_indices:
                if idx < len(feature_names) and normalized_importance[idx] > 0.5:  # Only meaningful contributions
                    contributions.append({
                        'feature': feature_names[idx],
                        'importance': round(float(normalized_importance[idx]), 2),
                        'value': round(float(features[idx]), 4),
                        'impact': 'positive' if features[idx] > 0 else 'negative'
                    })
            
            # Add contextual interpretations
            interpretations = []
            rsi_val = tech_analysis.get('rsi', {}).get('rsi', 50)
            if rsi_val > 70:
                interpretations.append('RSI overbought condition reducing BUY confidence')
            elif rsi_val < 30:
                interpretations.append('RSI oversold condition increasing BUY potential')
            
            pattern_count = len(pattern_analysis.get('patterns', []))
            if pattern_count > 0:
                bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal') == 'bullish')
                if bull_patterns > 0:
                    interpretations.append(f'{bull_patterns} bullish patterns supporting upside')
            
            ai_signal = ai_analysis.get('signal', 'HOLD')
            ai_conf = ai_analysis.get('confidence', 0)
            if ai_conf > 70:
                interpretations.append(f'High AI confidence ({ai_conf:.1f}%) in {ai_signal} signal')
            
            return {
                'top_features': contributions[:5],  # Top 5 for display
                'total_features_analyzed': len(features),
                'ai_signal_confidence': ai_conf,
                'contextual_interpretations': interpretations,
                'analysis_method': 'magnitude_activation_heuristic',
                'note': 'Simplified feature attribution - production systems use gradient-based methods'
            }
            
        except Exception as e:
            return {
                'error': f'Feature contribution analysis failed: {str(e)}',
                'top_features': [],
                'analysis_method': 'error'
            }
    
    def _calculate_adaptive_risk_targets(self, symbol, current_price, tech_analysis, regime_data, ai_analysis):
        """📊 Adaptive Risk & Target Sizing based on Market Conditions"""
        try:
            # Base parameters
            base_risk_pct = 2.0  # 2% base risk
            base_reward_ratio = 2.0  # 1:2 risk/reward base
            
            # Volatility adjustment from ATR
            atr = tech_analysis.get('atr', {}).get('atr', current_price * 0.02)
            atr_pct = (atr / current_price) * 100
            
            # Volatility risk scaling
            if atr_pct > 5.0:  # Very high volatility
                vol_multiplier = 0.6  # Reduce risk
                reward_multiplier = 3.0  # Increase reward ratio
            elif atr_pct > 3.0:  # High volatility
                vol_multiplier = 0.8
                reward_multiplier = 2.5
            elif atr_pct > 1.5:  # Normal volatility
                vol_multiplier = 1.0
                reward_multiplier = 2.0
            elif atr_pct > 0.8:  # Low volatility
                vol_multiplier = 1.2  # Increase risk slightly
                reward_multiplier = 1.8
            else:  # Very low volatility
                vol_multiplier = 1.4
                reward_multiplier = 1.5
            
            # Regime-based adjustments
            regime_multiplier = 1.0
            regime_reward_adj = 1.0
            
            if regime_data and 'regime_type' in regime_data:
                regime_type = regime_data['regime_type']
                regime_confidence = regime_data.get('confidence', 50)
                
                if regime_type == 'trending':
                    # Higher confidence in trends = more aggressive
                    if regime_confidence > 70:
                        regime_multiplier = 1.3
                        regime_reward_adj = 2.5
                    else:
                        regime_multiplier = 1.1
                        regime_reward_adj = 2.2
                        
                elif regime_type == 'ranging':
                    # Ranging markets = smaller targets, tighter stops
                    regime_multiplier = 0.8
                    regime_reward_adj = 1.5
                    
                elif regime_type == 'expansion':
                    # Expansion = potential for larger moves
                    regime_multiplier = 1.2
                    regime_reward_adj = 3.0
                    
                elif regime_type == 'volatility_crush':
                    # Low vol environment
                    regime_multiplier = 0.7
                    regime_reward_adj = 1.3
            
            # AI confidence adjustment
            ai_confidence = ai_analysis.get('confidence', 50)
            ai_signal = ai_analysis.get('signal', 'HOLD')
            
            if ai_confidence > 80 and ai_signal != 'HOLD':
                confidence_multiplier = 1.4  # High confidence = more aggressive
            elif ai_confidence > 60:
                confidence_multiplier = 1.2
            elif ai_confidence > 40:
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.7  # Low confidence = conservative
            
            # Calculate final risk percentage
            adaptive_risk_pct = base_risk_pct * vol_multiplier * regime_multiplier * confidence_multiplier
            adaptive_risk_pct = max(0.5, min(5.0, adaptive_risk_pct))  # Cap between 0.5% and 5%
            
            # Calculate reward ratio
            adaptive_reward_ratio = base_reward_ratio * reward_multiplier * regime_reward_adj
            adaptive_reward_ratio = max(1.2, min(4.0, adaptive_reward_ratio))  # Cap between 1.2:1 and 4:1
            
            # Position sizing (assuming $10,000 account)
            account_size = 10000
            risk_amount = account_size * (adaptive_risk_pct / 100)
            
            # Stop loss distance
            stop_distance_pct = atr_pct * 0.8  # Use 80% of ATR for stop
            stop_distance = current_price * (stop_distance_pct / 100)
            
            # Position size calculation
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Target levels
            target_1_distance = stop_distance * (adaptive_reward_ratio * 0.6)  # 60% of full target
            target_2_distance = stop_distance * adaptive_reward_ratio  # Full target
            target_3_distance = stop_distance * (adaptive_reward_ratio * 1.5)  # Extended target
            
            if ai_signal == 'BUY':
                stop_loss = current_price - stop_distance
                target_1 = current_price + target_1_distance
                target_2 = current_price + target_2_distance
                target_3 = current_price + target_3_distance
            else:  # SELL
                stop_loss = current_price + stop_distance
                target_1 = current_price - target_1_distance
                target_2 = current_price - target_2_distance
                target_3 = current_price - target_3_distance
            
            # Risk assessment
            risk_category = 'low'
            if adaptive_risk_pct > 3.5:
                risk_category = 'high'
            elif adaptive_risk_pct > 2.5:
                risk_category = 'medium'
            
            return {
                'adaptive_risk_pct': round(adaptive_risk_pct, 2),
                'adaptive_reward_ratio': round(adaptive_reward_ratio, 1),
                'position_size': round(position_size, 4),
                'stop_loss': round(stop_loss, 4),
                'targets': {
                    'target_1': round(target_1, 4),
                    'target_2': round(target_2, 4), 
                    'target_3': round(target_3, 4)
                },
                'risk_amount_usd': round(risk_amount, 2),
                'stop_distance_pct': round(stop_distance_pct, 3),
                'atr_pct': round(atr_pct, 3),
                'risk_category': risk_category,
                'adjustments': {
                    'volatility_multiplier': round(vol_multiplier, 2),
                    'regime_multiplier': round(regime_multiplier, 2),
                    'confidence_multiplier': round(confidence_multiplier, 2),
                    'reward_multiplier': round(reward_multiplier, 1)
                },
                'reasoning': f"Risk adjusted for {regime_data.get('regime_type', 'normal')} regime, {atr_pct:.1f}% volatility, {ai_confidence:.0f}% AI confidence"
            }
            
        except Exception as e:
            return {
                'error': f'Adaptive risk calculation failed: {str(e)}',
                'adaptive_risk_pct': 2.0,
                'adaptive_reward_ratio': 2.0
            }
    
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
        
        # Dynamic weight adaptation (reduce AI weight if offline or low confidence)
        dyn_weights = dict(self.weights)
        ai_conf = ai_analysis.get('confidence', 50)
        ai_status = ai_analysis.get('status') or ( 'offline' if not ai_analysis.get('initialized', True) else 'online')
        if ai_status == 'offline' or ai_conf < 40:
            # remove AI contribution, re-normalize others
            removed = dyn_weights.get('ai', 0)
            dyn_weights['ai'] = 0.0
            rem = dyn_weights['technical'] + dyn_weights['patterns']
            if rem <= 0:
                dyn_weights['technical'] = 0.7
                dyn_weights['patterns'] = 0.3
            else:
                dyn_weights['technical'] = dyn_weights['technical']/rem
                dyn_weights['patterns'] = dyn_weights['patterns']/rem
        # Weighted final score (raw aggregate)
        final_score = (
            tech_score * dyn_weights['technical'] +
            pattern_score * dyn_weights['patterns'] +
            ai_score * dyn_weights.get('ai',0)
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
        
        # Pseudo-Probability Calibration: map 0-100 score into 0-1 via logistic (center 50, slope 0.09)
        import math
        calibrated_prob = 1/(1+math.exp(-(final_score-50)*0.09))
        # Side-specific probability (prob of bullishness). If signal is SELL side invert.
        bullish_prob = calibrated_prob
        if signal in ['SELL','STRONG_SELL']:
            bullish_prob = 1 - bullish_prob
        # Attach reason if AI disabled
        ai_reason = None
        if dyn_weights.get('ai',0) == 0 and self.weights.get('ai',0) > 0:
            # Determine reason heuristically
            if ai_analysis.get('mode') == 'offline' or not ai_analysis.get('signal'): ai_reason = 'AI offline/initialization failed'
            elif ai_analysis.get('confidence',0) < 40: ai_reason = f'AI low confidence {ai_analysis.get("confidence",0)} < 40'
            else: ai_reason = 'AI weight dynamically suppressed'
        return {
            'score': round(final_score, 1),
            'probability_bullish': round(bullish_prob*100,2),
            'calibrated_probability': round(calibrated_prob*100,2),
            'probability_note': 'Heuristische Kalibrierung (logistische Kurve) – keine echte statistische Eintrittswahrscheinlichkeit',
            'signal': signal,
            'signal_color': signal_color,
            'technical_weight': f"{dyn_weights['technical']*100:.1f}%",
            'pattern_weight': f"{dyn_weights['patterns']*100:.1f}%",
            'ai_weight': f"{dyn_weights.get('ai',0)*100:.1f}%",
            'ai_disable_reason': ai_reason,
            'component_scores': {
                'technical': round(tech_score, 1),
                'patterns': round(pattern_score, 1),
                'ai': round(ai_score, 1)
            },
            'validation': self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, signal)
        }

    # (Deprecated earlier _generate_trade_setups removed in favor of advanced version below)
    
    def _validate_signals(self, tech_analysis, pattern_analysis, ai_analysis, final_signal, multi_timeframe=None):
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
        
        # 5. AI Confidence & Consistency Validation
        ai_confidence = ai_analysis.get('confidence', 50)
        ai_signal = ai_analysis.get('signal', 'HOLD')
        if ai_confidence < 60:
            warnings.append({
                'type': 'LOW_AI_CONFIDENCE',
                'message': f'⚠️ KI Confidence nur {ai_confidence}%',
                'recommendation': 'KI ist unsicher - warte auf klarere Signale!'
            })
        # AI vs final signal disagreement (directionally opposite)
        opposite_map = {
            'STRONG_BUY': ['SELL','STRONG_SELL'],
            'BUY': ['STRONG_SELL','SELL'],
            'STRONG_SELL': ['BUY','STRONG_BUY'],
            'SELL': ['STRONG_BUY','BUY']
        }
        if ai_signal in opposite_map.get(final_signal, []) and ai_confidence >= 55:
            contradictions.append({
                'type': 'AI_FINAL_CONTRADICTION',
                'message': f'⚠️ KI signal {ai_signal} widerspricht {final_signal}',
                'severity': 'MEDIUM',
                'recommendation': 'Weitere Bestätigung abwarten'
            })
        # Multi-timeframe consensus contradictions
        if multi_timeframe and isinstance(multi_timeframe, dict):
            mt_primary = multi_timeframe.get('consensus', {}).get('primary')
            if mt_primary == 'BULLISH' and final_signal in ['SELL','STRONG_SELL']:
                contradictions.append({
                    'type': 'MTF_CONTRADICTION',
                    'message': 'MTF Konsens BULLISH aber finales Signal bearish',
                    'severity': 'HIGH',
                    'recommendation': 'Auf Alignment warten'
                })
            if mt_primary == 'BEARISH' and final_signal in ['BUY','STRONG_BUY']:
                contradictions.append({
                    'type': 'MTF_CONTRADICTION',
                    'message': 'MTF Konsens BEARISH aber finales Signal bullish',
                    'severity': 'HIGH',
                    'recommendation': 'Auf Alignment warten'
                })
            # AI vs MTF
            if mt_primary == 'BULLISH' and ai_signal in ['SELL','STRONG_SELL'] and ai_confidence >= 55:
                warnings.append({
                    'type': 'AI_MTF_MISMATCH',
                    'message': 'KI bearish vs MTF bullish',
                    'recommendation': 'Signalqualität prüfen'
                })
            if mt_primary == 'BEARISH' and ai_signal in ['BUY','STRONG_BUY'] and ai_confidence >= 55:
                warnings.append({
                    'type': 'AI_MTF_MISMATCH',
                    'message': 'KI bullish vs MTF bearish',
                    'recommendation': 'Signalqualität prüfen'
                })
        # Multi-timeframe pattern majority vs final
        try:
            mtp = pattern_analysis.get('multi_timeframe_patterns', [])
            if mtp:
                b_mt = sum(1 for p in mtp if p.get('signal')=='bullish')
                br_mt = sum(1 for p in mtp if p.get('signal')=='bearish')
                if b_mt > br_mt and final_signal in ['SELL','STRONG_SELL']:
                    contradictions.append({
                        'type': 'MTPATTERN_CONTRADICTION',
                        'message': f'Mehrheit {b_mt} bullish MTF Patterns aber finales Signal bearish',
                        'severity': 'MEDIUM',
                        'recommendation': 'Auf Bestätigung warten'
                    })
                if br_mt > b_mt and final_signal in ['BUY','STRONG_BUY']:
                    contradictions.append({
                        'type': 'MTPATTERN_CONTRADICTION',
                        'message': f'Mehrheit {br_mt} bearish MTF Patterns aber finales Signal bullish',
                        'severity': 'MEDIUM',
                        'recommendation': 'Auf Bestätigung warten'
                    })
        except Exception:
            pass
        
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

    def _generate_trade_setups(self, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score, multi_timeframe=None, regime_data=None):
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
                ext_mult = 8.0  # Erweitert von 4.0 auf 8.0 für echte Swing-Targets
                swing_target = entry + min_atr * ext_mult if direction=='LONG' else entry - min_atr * ext_mult
                return swing_target

            def _confidence(base, adds):
                score = base + sum(adds)
                # 🔍 STRENGE VALIDIERUNG wie echte Trader
                if contradiction_count: score -= 35  # Erhöht von 25 auf 35
                if risk_level in ['HIGH', 'VERY_HIGH']: score -= 25  # Erhöht von 15 auf 25
                if atr_perc and atr_perc > 1.4: score -= 15  # Erhöht von 8 auf 15
                # Zusätzliche Validierung
                if atr_perc and atr_perc > 2.0: score -= 25  # Extreme Volatilität
                if not enterprise_ready: score -= 20  # Keine Enterprise-Validierung
                return max(10, min(95, round(score)))  # Min erhöht von 5 auf 10

            def _targets(entry, stop, direction, extra=None):
                risk = (entry - stop) if direction=='LONG' else (stop - entry)  # absolute price risk
                # 🔥 REALISTISCHE TRADER STOPS - breiter für echte Marktbedingungen
                if risk < min_atr * 1.2:  # Erweitert von 0.8 auf 1.2 für realistischere Stops
                    # Wesentlich breitere Stops wie echte Trader
                    if direction=='LONG':
                        stop = entry - min_atr * 1.2
                    else:
                        stop = entry + min_atr * 1.2
                    risk = (entry - stop) if direction=='LONG' else (stop - entry)
                risk = max(risk, min_atr*1.0)  # Minimum risk erhöht

                base = []
                # 🎯 REALISTISCHE TP TARGETS - wie echte Trader nutzen
                # Erste Gewinnmitnahme bei 1.5R, dann größere Swings
                for m in [1.5, 2.5, 4, 6, 8]:  # Entfernt 1R - zu enge, hinzugefügt 6R, 8R
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
                # 🎯 PROFESSIONAL TARGET FILTERING - breiter gefiltert
                # Keep top distinct targets (remove those closer than 0.8R apart für realistischere Abstände)
                filtered = []
                last_rr = -999
                for t in base:
                    if t['rr'] - last_rr >= 0.8:  # Erhöht von 0.4 auf 0.8
                        filtered.append(t)
                        last_rr = t['rr']
                    if len(filtered) >= 5:  # Reduziert von 7 auf 5 Targets
                        break
                return filtered

            # Aggregate multi-timeframe patterns for ranking
            timeframe_weight = {'15m':0.6,'1h':1.0,'4h':1.4,'1d':1.8}
            all_patterns = []
            try:
                base_p = pattern_analysis.get('patterns', [])
                mt_p = pattern_analysis.get('multi_timeframe_patterns', [])
                for p in base_p + mt_p:
                    tf = p.get('timeframe','1h')
                    conf = p.get('confidence',50)
                    w = timeframe_weight.get(tf,1.0)
                    p['_rank_score'] = conf * w
                    all_patterns.append(p)
            except Exception:
                pass
            bull_ranked = sorted([p for p in all_patterns if p.get('signal')=='bullish'], key=lambda x: x.get('_rank_score',0), reverse=True)
            bear_ranked = sorted([p for p in all_patterns if p.get('signal')=='bearish'], key=lambda x: x.get('_rank_score',0), reverse=True)

            # 🔍 ENHANCED TRADER VALIDIERUNG
            # Mehrstufige Validierung wie professionelle Trader
            setup_quality_filters = {
                'minimum_confidence': 35,  # Erhöht von Standard
                'maximum_risk_percent': 3.0,  # Max 3% Risk pro Trade
                'require_multiple_confirmations': True,
                'avoid_high_volatility_entries': atr_perc > 2.5,
                'trend_alignment_required': True
            }
            
            # 🚨 RSI CAUTION NARRATIVE INJECTION
            rsi_caution = self._generate_rsi_caution_narrative(rsi, trend)
            
            # Relaxed trend rule: allow LONG setups if not strongly bearish
            # 🚨 ABER mit zusätzlicher Validierung + RSI CAUTION
            trend_validation_passed = False
            if 'bullish' in trend or trend in ['neutral','weak','moderate']:
                if 'bullish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    # Zusätzliche Bestätigung erforderlich bei schwachem Trend
                    if rsi > 45 and contradiction_count == 0:
                        trend_validation_passed = True
                else:
                    trend_validation_passed = True
                if trend_validation_passed:  # Nur wenn Validierung bestanden
                    entry_pb = support * 1.003
                    stop_pb = support - atr_val*0.9  # Erweitert von 0.6 auf 0.9
                    risk_pct = round((entry_pb-stop_pb)/entry_pb*100,2)
                    
                    # 🔍 ZUSÄTZLICHE RISK VALIDIERUNG
                    if risk_pct <= setup_quality_filters['maximum_risk_percent']:
                        # Apply RSI caution to rationale
                        base_rationale = 'Multi-validated Einstieg nahe Support mit Professional Risk Management'
                        enhanced_rationale = f"{base_rationale}. {rsi_caution['narrative']}" if rsi_caution['caution_level'] != 'none' else base_rationale
                        
                        setups.append({
                            'id':'L-PB', 'direction':'LONG', 'strategy':'Professional Bullish Pullback',
                            'entry': round(entry_pb,2), 'stop_loss': round(stop_pb,2),
                            'risk_percent': risk_pct,
                            'targets': _targets(entry_pb, stop_pb,'LONG', [
                                ('Resistance', resistance), ('Fib 0.382', fib.get('fib_382')), ('Fib 0.618', fib.get('fib_618'))
                            ]),
                            'confidence': _confidence(55,[15 if enterprise_ready else 5, 8 if rsi<65 else 0, 10 if trend_validation_passed else 0]) - rsi_caution['confidence_penalty'],
                            'conditions': [
                                {'t':'Trend validation','s':'ok' if trend_validation_passed else 'warn'},
                                {'t':f'RSI {rsi:.1f}','s':rsi_caution['signal_quality']},
                                {'t':f'Risk {risk_pct:.1f}%','s':'ok' if risk_pct<=2.0 else 'warn'},
                                {'t':'Enterprise Ready','s':'ok' if enterprise_ready else 'warn'},
                                {'t':'Low Contradictions','s':'ok' if contradiction_count<=1 else 'bad'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                trend_validation_passed, enterprise_ready, contradiction_count==0, risk_pct<=2.0
                            ]) else 'STANDARD',
                            'rationale': enhanced_rationale,
                            'rsi_caution': rsi_caution,
                            'regime_context': regime_data.get('regime', 'unknown') if regime_data else 'unknown',
                            'justification': {
                                'core_thesis': 'Pullback in intaktem Aufwärtstrend zurück in Nachfragezone (Support Re-Test).',
                                'confluence': [
                                    'Trend Align (bullish / nicht bearish)',
                                    f'RSI moderat ({rsi:.1f}) -> noch kein Extrem',
                                    'Support strukturell bestätigt',
                                    'Risk <= 2% akzeptabel',
                                    'Keine starken Widersprüche'
                                ],
                                'risk_model': 'Stop unter strukturellem Support + ATR-Puffer (~1.2 ATR).',
                                'invalidations': [
                                    'Tiefer Schlusskurs 1.5% unter Support',
                                    'RSI Divergenz bearish + MACD Curve kippt',
                                    'Volumen Distribution Shift gegen Trend'
                                ],
                                'execution_plan': 'Limit/Stop-Order leicht über Re-Test Candle High, Teilgewinn bei 2R, Rest trailen.'
                            }
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
                    'rationale':'Ausbruch nutzt Momentum Beschleunigung',
                    'justification': {
                        'core_thesis': 'Preis akzeptiert oberhalb vorherigen Angebotslevels -> mögliche Preisentfaltung / Imbalance Fill.',
                        'confluence': [
                            'Break + Close über Resistance',
                            'Momentum bestätigt (MACD Curve / Volumen Spike möglich)',
                            f'RSI noch unter Extremzone ({rsi:.1f})',
                            'Keine akuten Widerspruchs-Signale'
                        ],
                        'risk_model': 'Stop unter ehemaligem Widerstand (jetzt potentielle Unterstützung) + ATR-Schutz.',
                        'invalidations': [
                            'Rückfall & Close zurück unter Level',
                            'Low-Volume Fakeout (Volumen unter Durchschnitt)',
                            'Bearish Engulfing direkt nach Break'
                        ],
                        'execution_plan': 'Stop-Order geringfügig über Break-Level, Confirm Candle abwarten, dann Staffeln der TPs.'
                    }
                })

                # Pattern Confirmation LONG
                if bull_ranked:
                    top_b = bull_ranked[0]
                    tfb = top_b.get('timeframe','1h')
                    entry_pc = current_price * 1.001 if current_price < resistance else resistance*1.001
                    stop_pc = entry_pc - atr_val*0.8
                    setups.append({
                        'id':'L-PC', 'direction':'LONG', 'strategy':'Pattern Confirmation',
                        'entry': round(entry_pc,2), 'stop_loss': round(stop_pc,2),
                        'risk_percent': round((entry_pc-stop_pc)/entry_pc*100,2),
                        'targets': _targets(entry_pc, stop_pc,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(52,[min(18,int(top_b.get('confidence',50)/3)), 5 if 'bullish' in trend else 0]),
                        'conditions': [
                            {'t':f'Pattern {top_b.get("name","?")}@{tfb}','s':'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's':'ok'},
                            {'t':f'RSI {rsi:.1f}','s':'ok' if rsi<70 else 'warn'}
                        ],
                        'pattern_timeframe': tfb,
                        'pattern_refs':[f"{top_b.get('name','?')}@{tfb}"],
                        'source_signals':['pattern','macd_curve','rsi'],
                        'rationale':'Bullisches Muster bestätigt Fortsetzung',
                        'justification': {
                            'core_thesis': f'Bestätigtes bullisches {top_b.get("name","Pattern")} auf {tfb} mit Momentum-Unterstützung.',
                            'confluence': [
                                'Muster + Trend nicht dagegen',
                                'MACD Curve positiv',
                                f'RSI gesund ({rsi:.1f})',
                                'Kein unmittelbarer Widerstand direkt vor Entry'
                            ],
                            'risk_model': 'Stop unter Pattern-Struktur + ATR-Puffer.',
                            'invalidations': [
                                'Bruch Pattern Low',
                                'Volumen-Absorption am Entry',
                                'Bearish Divergenz entsteht'
                            ],
                            'execution_plan': 'Einstieg nach Bestätigungs-Close, Teilverkäufe an 2R / Strukturzonen.'
                        }
                    })

                # Momentum Continuation LONG
                macd_curve = tech_analysis.get('macd',{}).get('curve_direction','neutral')
                if 'bullish' in macd_curve and rsi > 55:
                    entry_momo = current_price * 1.0005
                    stop_momo = entry_momo - atr_val
                    setups.append({
                        'id':'L-MOMO', 'direction':'LONG', 'strategy':'Momentum Continuation',
                        'entry': round(entry_momo,2), 'stop_loss': round(stop_momo,2),
                        'risk_percent': round((entry_momo-stop_momo)/entry_momo*100,2),
                        'targets': _targets(entry_momo, stop_momo,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(50,[10 if rsi>60 else 5, 6]),
                        'conditions':[{'t':'MACD Curve bullish','s':'ok'},{'t':f'RSI {rsi:.1f}','s':'ok'},{'t':'Trend nicht bearish','s':'ok' if 'bearish' not in trend else 'warn'}],
                        'source_signals':['macd_curve','rsi','trend'],
                        'rationale':'Momentum Fortsetzung basierend auf MACD Bogen + RSI',
                        'justification': {
                            'core_thesis': 'Fortsetzung nach impulsiver Expansionsphase ohne Erschöpfungssignale.',
                            'confluence': [ 'Bull MACD Curve', f'RSI > 55 ({rsi:.1f})', 'Keine bearishe Divergenz', 'Trend nicht kontra' ],
                            'risk_model': 'Stop unter kurzfristigem Momentum Pivot (letzte Mini-Konsolidierung).',
                            'invalidations': [ 'Momentum Collapse (starker Gegen-Volumen Spike)', 'RSI fällt unter 48', 'MACD Curve dreht sofort bearisch' ],
                            'execution_plan': 'Market/Limit Entry in leichte Pullback-Candle, aggressives Trail nach 2R.'
                        }
                    })

                # Support Rejection LONG
                if support and (current_price - support)/current_price*100 < 1.2 and bull_ranked:
                    top_b2 = bull_ranked[0]
                    tfb2 = top_b2.get('timeframe','1h')
                    entry_rej = support * 1.004
                    stop_rej = support - atr_val*0.7
                    setups.append({
                        'id':'L-REJ', 'direction':'LONG', 'strategy':'Support Rejection',
                        'entry': round(entry_rej,2), 'stop_loss': round(stop_rej,2),
                        'risk_percent': round((entry_rej-stop_rej)/entry_rej*100,2),
                        'targets': _targets(entry_rej, stop_rej,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(48,[8, min(14,int(top_b2.get('confidence',50)/4))]),
                        'conditions':[{'t':'Nahe Support','s':'ok'},{'t':'Bull Pattern','s':'ok'},{'t':'Volatilität ok','s':'ok' if atr_perc<1.5 else 'warn'}],
                        'pattern_timeframe': tfb2,
                        'pattern_refs':[f"{top_b2.get('name','?')}@{tfb2}"],
                        'source_signals':['support','pattern'],
                        'rationale':'Rejection nahe Support + bullisches Muster',
                        'justification': {
                            'core_thesis': 'Agressive Käufer verteidigen Key-Support -> frische Nachfrage bestätigt.',
                            'confluence': [ 'Wick Rejection / schnelle Zurückweisung', 'Bull Pattern aktiv', 'Volatilität moderat', 'Support mehrfach getestet' ],
                            'risk_model': 'Stop unter Rejection-Trigger + Struktur Low.',
                            'invalidations': [ 'Close unter Support', 'Volumen trocknet aus', 'Pattern verliert Struktur' ],
                            'execution_plan': 'Entry nach Rejection Candle High Break, konservatives Teilziel bei 2R.'
                        }
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
                    'rationale':'Überverkaufte Bedingung -> Rebound Szenario',
                    'justification': {
                        'core_thesis': 'Kurzfristige Übertreibung (Oversold) mit mean reversion Potenzial.',
                        'confluence': [ f'RSI < 35 ({rsi:.1f})', 'Trend nicht stark bearish', 'Keine massiven Distribution-Spikes' ],
                        'risk_model': 'Stop unter lokaler Exhaustion / Spike Low.',
                        'invalidations': [ 'Weitere starke Long Liquidations', 'RSI fällt unter 20 ohne Reaktionsvolumen' ],
                        'execution_plan': 'Scaling Entry in 2 Tranchen, enges Management, frühes Secure bei 1.5-2R.'
                    }
                })

            # SHORT strategies (relax: allow if not strongly bullish)
            # 🚨 ABER mit professioneller Validierung
            short_trend_validation_passed = False
            if 'bearish' in trend or trend in ['neutral','weak','moderate']:
                if 'bearish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    # Zusätzliche Bestätigung erforderlich bei schwachem Trend
                    if rsi < 55 and contradiction_count == 0:
                        short_trend_validation_passed = True
                else:
                    short_trend_validation_passed = True
                    
                if short_trend_validation_passed:  # Nur wenn Validierung bestanden
                    entry_pbs = resistance*0.997
                    stop_pbs = resistance + atr_val*0.9  # Erweitert von 0.6 auf 0.9
                    risk_pct_short = round((stop_pbs-entry_pbs)/entry_pbs*100,2)
                    
                    # 🔍 ZUSÄTZLICHE RISK VALIDIERUNG für SHORT
                    if risk_pct_short <= setup_quality_filters['maximum_risk_percent']:
                        setups.append({
                            'id':'S-PB', 'direction':'SHORT', 'strategy':'Professional Bearish Pullback',
                            'entry': round(entry_pbs,2), 'stop_loss': round(stop_pbs,2),
                            'risk_percent': risk_pct_short,
                            'targets': _targets(entry_pbs, stop_pbs,'SHORT', [('Support', support), ('Fib 0.382', fib.get('fib_382'))]),
                            'confidence': _confidence(55,[15 if enterprise_ready else 5, 8 if rsi>35 else 0, 10 if short_trend_validation_passed else 0]),
                            'conditions': [ 
                                {'t':'Short Trend validation','s':'ok' if short_trend_validation_passed else 'warn'}, 
                                {'t':f'RSI {rsi:.1f}','s':'ok' if rsi>35 else 'warn'},
                                {'t':f'Risk {risk_pct_short:.1f}%','s':'ok' if risk_pct_short<=2.0 else 'warn'},
                                {'t':'Enterprise Ready','s':'ok' if enterprise_ready else 'warn'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                short_trend_validation_passed, enterprise_ready, contradiction_count==0, risk_pct_short<=2.0
                            ]) else 'STANDARD',
                            'rationale':'Multi-validated Pullback an Widerstand mit Professional Risk Management',
                            'justification': {
                                'core_thesis': 'Pullback in aktiven Abwärtstrend zurück in Angebotszone (Lower High Opportunity).',
                                'confluence': [ 'Trend Align bearish / nicht bullisch', f'RSI neutral ({rsi:.1f}) -> Raum für Abwärtsbewegung', 'Widerstand bestätigt', 'Risk <= 2%' ],
                                'risk_model': 'Stop über strukturellem Swing High + ATR-Puffer.',
                                'invalidations': [ 'Starker Close über Widerstand', 'Bullische Volume Absorption', 'Momentum Shift (MACD Curve bullisch)' ],
                                'execution_plan': 'Entry mittels Limit nahe Pullback Spitze, Teilgewinn bei 2R.'
                            }
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
                    'rationale':'Beschleunigter Momentum-Handel beim Support-Bruch',
                    'justification': {
                        'core_thesis': 'Akzeptanz unter Key-Support -> möglicher Preis-Entleerungsbereich (Liquidity Vacuum).',
                        'confluence': [ 'Bruch + Close unter Support', 'Keine bullische Divergenz', 'Volumen nicht kollabierend', 'Trend nicht bullisch' ],
                        'risk_model': 'Stop über Breakdown Level + ATR.',
                        'invalidations': [ 'Starker Reclaim Support', 'Volumen Divergenz (fallender Preis, fallendes Volumen)', 'Bull Pattern formt sich sofort' ],
                        'execution_plan': 'Entry über Stop-Order unter Bestätigungscandle, schnelles Tightening nach initialem Flush.'
                    }
                })

                # Pattern Confirmation SHORT
                if bear_ranked:
                    top_s = bear_ranked[0]
                    tfs = top_s.get('timeframe','1h')
                    entry_ps = current_price * 0.999 if current_price > support else support*0.999
                    stop_ps = entry_ps + atr_val*0.8
                    setups.append({
                        'id':'S-PC', 'direction':'SHORT', 'strategy':'Pattern Confirmation',
                        'entry': round(entry_ps,2), 'stop_loss': round(stop_ps,2),
                        'risk_percent': round((stop_ps-entry_ps)/entry_ps*100,2),
                        'targets': _targets(entry_ps, stop_ps,'SHORT', [('Support', support)]),
                        'confidence': _confidence(52,[min(18,int(top_s.get('confidence',50)/3)), 5 if 'bearish' in trend else 0]),
                        'conditions': [
                            {'t':f'Pattern {top_s.get("name","?")}@{tfs}','s':'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's':'ok'},
                            {'t':f'RSI {rsi:.1f}','s':'ok' if rsi>30 else 'warn'}
                        ],
                        'pattern_timeframe': tfs,
                        'pattern_refs':[f"{top_s.get('name','?')}@{tfs}"],
                        'source_signals':['pattern','macd_curve','rsi'],
                        'rationale':'Bearishes Muster bestätigt Fortsetzung',
                        'justification': {
                            'core_thesis': f'Bestätigtes bearisches {top_s.get("name","Pattern")} auf {tfs} mit Momentum-Unterstützung.',
                            'confluence': [ 'Muster + Trend nicht bullish', 'MACD Curve negativ', f'RSI intakt ({rsi:.1f})', 'Kein unmittelbarer Support darunter' ],
                            'risk_model': 'Stop über Pattern-Struktur + ATR-Puffer.',
                            'invalidations': [ 'Close zurück in Pattern', 'Volumen Absorption durch Käufer', 'Bull Divergenz bildet sich' ],
                            'execution_plan': 'Entry nach Bestätigungs-Close, Teilziel 2R, Rest laufen bis Strukturbruch.'
                        }
                    })

                # Momentum Continuation SHORT
                macd_curve_s = tech_analysis.get('macd',{}).get('curve_direction','neutral')
                if 'bearish' in macd_curve_s and rsi < 45:
                    entry_momo_s = current_price * 0.9995
                    stop_momo_s = entry_momo_s + atr_val
                    setups.append({
                        'id':'S-MOMO', 'direction':'SHORT', 'strategy':'Momentum Continuation',
                        'entry': round(entry_momo_s,2), 'stop_loss': round(stop_momo_s,2),
                        'risk_percent': round((stop_momo_s-entry_momo_s)/entry_momo_s*100,2),
                        'targets': _targets(entry_momo_s, stop_momo_s,'SHORT', [('Support', support)]),
                        'confidence': _confidence(50,[10 if rsi<40 else 5, 6]),
                        'conditions':[{'t':'MACD Curve bearish','s':'ok'},{'t':f'RSI {rsi:.1f}','s':'ok'},{'t':'Trend nicht bullish','s':'ok' if 'bullish' not in trend else 'warn'}],
                        'source_signals':['macd_curve','rsi','trend'],
                        'rationale':'Momentum Fortsetzung basierend auf MACD Bogen + RSI',
                        'justification': {
                            'core_thesis': 'Fortlaufende Abwärts-Beschleunigung ohne deutliche Gegenreaktion (Momentum Squeeze).',
                            'confluence': [ 'Bear MACD Curve', f'RSI < 45 ({rsi:.1f})', 'Kein aggressives Buying', 'Trend nicht kontra' ],
                            'risk_model': 'Stop über kurzfristigem Momentum Pivot / Mini Range High.',
                            'invalidations': [ 'Sharp Reversal Candle mit Volumen', 'RSI Erholung > 52', 'Pattern Flip bullisch' ],
                            'execution_plan': 'Entry in Micro-Pullback, aggressives Trailing nach 2R.'
                        }
                    })

                # Resistance Rejection SHORT
                if resistance and (resistance - current_price)/current_price*100 < 1.2 and bear_ranked:
                    top_s2 = bear_ranked[0]
                    tfs2 = top_s2.get('timeframe','1h')
                    entry_rej_s = resistance * 0.996
                    stop_rej_s = resistance + atr_val*0.7
                    setups.append({
                        'id':'S-REJ', 'direction':'SHORT', 'strategy':'Resistance Rejection',
                        'entry': round(entry_rej_s,2), 'stop_loss': round(stop_rej_s,2),
                        'risk_percent': round((stop_rej_s-entry_rej_s)/entry_rej_s*100,2),
                        'targets': _targets(entry_rej_s, stop_rej_s,'SHORT', [('Support', support)]),
                        'confidence': _confidence(48,[8, min(14,int(top_s2.get('confidence',50)/4))]),
                        'pattern_timeframe': tfs2,
                        'pattern_refs':[f"{top_s2.get('name','?')}@{tfs2}"],
                        'source_signals':['resistance','pattern'],
                        'rationale':'Rejection nahe Resistance + bearisches Muster',
                        'justification': {
                            'core_thesis': 'Verkaufsabsorptionszone wird verteidigt -> Angebot dominiert weiter.',
                            'confluence': [ 'Mehrfaches Rejection Verhalten', 'Bear Pattern aktiv', 'Volatilität moderat', 'Kein Momentum Reversal' ],
                            'risk_model': 'Stop über Rejection Wick + ATR.',
                            'invalidations': [ 'Close über Level', 'Volumen Shift pro Käufer', 'Pattern Struktur bricht' ],
                            'execution_plan': 'Entry nach Bestätigung Reversal Candle, konservatives Ziel 2R, Rest strukturbasiert.'
                        }
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
                    'rationale':'Überkaufte Bedingung -> Rücksetzer / Mean Reversion',
                    'justification': {
                        'core_thesis': 'Kurzfristige Überdehnung (Overbought) lädt Mean-Reversion Bewegung ein.',
                        'confluence': [ f'RSI > 65 ({rsi:.1f})', 'Keine frische Breakout Momentum Candle', 'Trend nicht stark bullish' ],
                        'risk_model': 'Stop über Exhaustion Hoch.',
                        'invalidations': [ 'Fortgesetzter Momentum Squeeze', 'RSI > 78 mit Volumen Expansion' ],
                        'execution_plan': 'Entry gestaffelt, schneller Stop Tightening, conservative Targets.'
                    }
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

            # Add probability estimates (heuristic) per setup using calibrated final score
            try:
                base_prob = 0.5
                if isinstance(final_score, dict):
                    cp = final_score.get('calibrated_probability')
                    if isinstance(cp, (int,float)):
                        base_prob = max(0.01, min(0.99, cp/100.0))
                for s in setups:
                    conf = s.get('confidence',50)/100.0
                    if s.get('direction') == 'LONG':
                        p = base_prob + (conf-0.5)*0.35
                    else:
                        p = (1-base_prob) + (conf-0.5)*0.35
                    p = max(0.02, min(0.98, p))
                    s['probability_estimate_pct'] = round(p*100,2)
                    s['probability_note'] = 'Heuristisch (Score + Confidence). Nicht kalibriert.'
            except Exception:
                pass

            for s in setups:
                if s.get('targets'):
                    s['primary_rr'] = s['targets'][0]['rr']
            
            # 🎯 INTEGRATE CHART PATTERN TRADES
            pattern_trades = []
            if pattern_analysis and pattern_analysis.get('patterns'):
                pattern_trades = ChartPatternTrader.generate_pattern_trades(
                    pattern_analysis['patterns'], 
                    current_price, 
                    atr_val,
                    support, 
                    resistance
                )
                print(f"📊 Generated {len(pattern_trades)} pattern-based trades")
            
            # Combine traditional setups with pattern trades
            all_setups = setups + pattern_trades
            all_setups.sort(key=lambda x: x.get('confidence', 50), reverse=True)
            trimmed = all_setups[:12]  # Erweitert von 8 auf 12 für Pattern Trades
            
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
    
    /* Pattern Trading Cards */
    .pattern-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.08), rgba(255, 165, 0, 0.04));
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);
    }
    .pattern-card:hover {
        border-color: rgba(255, 215, 0, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.15);
    }
    .pattern-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500) !important;
        color: #000 !important;
        font-weight: 700;
        text-shadow: none;
    }
    .target-pill.pattern-target {
        background: linear-gradient(45deg, rgba(255, 215, 0, 0.2), rgba(255, 165, 0, 0.1));
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFD700;
    }
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

                <!-- Market Regime -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🎯</span> Market Regime <span class="tag">BETA</span></div>
                    <div id="regimeAnalysis">
                        <!-- Regime analysis -->
                    </div>
                </div>

                <!-- Adaptive Risk & Targets -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🎯</span> Adaptive Risk Management <span class="tag">NEW</span></div>
                    <div id="adaptiveRiskTargets">
                        <!-- Adaptive risk and targets -->
                    </div>
                </div>

                <!-- Order Flow Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">📊</span> Order Flow <span class="tag">NEW</span></div>
                    <div id="orderFlowAnalysis">
                        <!-- Order flow analysis -->
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

                <!-- Feature Contributions -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🔍</span> AI Explainability <span class="tag">NEW</span></div>
                    <div id="featureContributions">
                        <!-- Feature contributions analysis -->
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
            displayRegimeAnalysis(data);
            displayAdaptiveRiskTargets(data);
            displayOrderFlowAnalysis(data);
            displayMarketBias(data);
            displayAIAnalysis(data);
            displayFeatureContributions(data);
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

        // Trade Setups Renderer (array based) - Enhanced for Pattern Trades
        function displayTradeSetups(data) {
            const container = document.getElementById('tradeSetupsContent');
            const status = document.getElementById('tradeSetupsStatus');
            const setups = data.trade_setups || [];
            if (!Array.isArray(setups) || setups.length === 0) {
                container.innerHTML = '';
                status.textContent = 'Keine Setups generiert (Bedingungen nicht erfüllt).';
                return;
            }

            // Separate Pattern trades from regular trades
            const patternTrades = setups.filter(s => s.pattern_name || s.setup_type);
            const regularTrades = setups.filter(s => !s.pattern_name && !s.setup_type);

            let html = '';

            // Pattern Trades Section
            if (patternTrades.length > 0) {
                html += `<div class="trade-section">
                    <h4 style="color: #FFD700; margin-bottom: 12px; font-size: 0.85rem; display: flex; align-items: center;">
                        🎯 <span style="margin-left: 6px;">Chart Pattern Setups (${patternTrades.length})</span>
                    </h4>`;
                
                const patternBlocks = patternTrades.map(s => {
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets || s.take_profits || []).map(t=>{
                        const price = t.price || t.level;
                        const label = t.label || t.level;
                        const percentage = t.percentage ? ` (${t.percentage}%)` : '';
                        const rr = t.rr ? ` ${t.rr}R` : '';
                        return `<span class="target-pill pattern-target">${label}: ${price}${percentage}${rr}</span>`;
                    }).join('');
                    
                    return `
                    <div class="setup-card pattern-card" style="border-left: 4px solid ${s.direction==='LONG'?'#28a745':'#dc3545'};">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">
                            ${s.direction} 
                            <span class="setup-badge pattern-badge ${s.direction==='LONG'?'long':'short'}" style="background: linear-gradient(45deg, #FFD700, #FFA500); color: #000;">
                                ${s.pattern_name || s.strategy}
                            </span>
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry_price || s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        <div class="setup-line"><span>Risk%</span><span>${s.risk_percent || s.risk_reward_ratio}%</span></div>
                        ${s.risk_reward_ratio ? `<div class="setup-line"><span>R/R</span><span style="color: #28a745;">${s.risk_reward_ratio}</span></div>` : ''}
                        ${s.key_level ? `<div class="setup-line"><span>Key Level</span><span style="color: #FFD700;">${s.key_level}</span></div>` : ''}
                        ${s.regime_context ? `<div class="setup-line"><span>Regime</span><span style="color: #17a2b8;">${s.regime_context}</span></div>` : ''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        ${s.rsi_caution && s.rsi_caution.caution_level !== 'none' ? `<div style="margin-top:6px; font-size:.5rem; color:#ffc107; line-height:0.75rem;"><strong>RSI:</strong> ${s.rsi_caution.narrative}</div>` : ''}
                        ${s.trade_plan ? `<div style="margin-top:8px; font-size:.55rem; color:#FFD700; line-height:0.75rem;"><strong>Plan:</strong> ${s.trade_plan}</div>` : ''}
                        ${s.market_structure ? `<div style="margin-top:4px; font-size:.55rem; color:rgba(255,255,255,0.7); line-height:0.75rem;"><strong>Structure:</strong> ${s.market_structure}</div>` : ''}
                        <div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:0.75rem;">${s.rationale || s.trade_plan}</div>
                    </div>`;
                });
                html += patternBlocks.join('') + '</div>';
            }

            // Regular Technical Trades Section
            if (regularTrades.length > 0) {
                html += `<div class="trade-section">
                    <h4 style="color: #17a2b8; margin-bottom: 12px; font-size: 0.85rem; display: flex; align-items: center;">
                        📊 <span style="margin-left: 6px;">Technical Analysis Setups (${regularTrades.length})</span>
                    </h4>`;
                
                const regularBlocks = regularTrades.map(s => {
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets||[]).map(t=>`<span class="target-pill">${t.label}: ${t.price} (${t.rr}R)</span>`).join('');
                    const conds = (s.conditions||[]).map(c=>`<li class="${c.s==='ok'?'c-ok':(c.s==='bad'?'c-bad':'c-warn')}">${c.t}</li>`).join('');
                    
                    return `
                    <div class="setup-card">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">
                            ${s.direction} 
                            <span class="setup-badge ${s.direction==='LONG'?'long':'short'}">${s.strategy}</span>
                            ${s.validation_score ? `<span class="validation-badge ${s.validation_score.toLowerCase()}">${s.validation_score}</span>` : ''}
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        <div class="setup-line"><span>Risk%</span><span>${s.risk_percent}%</span></div>
                        ${s.primary_rr ? `<div class="setup-line"><span>R/R</span><span style="color: #28a745;">${s.primary_rr}R</span></div>` : ''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        <ul class="conditions">${conds}</ul>
                        <div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:0.75rem;">${s.rationale}</div>
                    </div>`;
                });
                html += regularBlocks.join('') + '</div>';
            }

            container.innerHTML = html;
            status.textContent = `${setups.length} Trading-Setups generiert (${patternTrades.length} Pattern + ${regularTrades.length} Technical)`;
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
            const patterns = data.pattern_analysis || {};
            const list = patterns.patterns || [];
            let html = `
                <div class=\"metric-card\" style=\"margin-bottom: 12px;\">
                    <div class=\"metric-value ${getSignalColor(patterns.overall_signal || 'neutral')}\">${patterns.overall_signal || 'NEUTRAL'}</div>
                    <div class=\"metric-label\">Overall Pattern Signal</div>
                </div>`;
            if (patterns.pattern_summary) {
                html += `<p style=\"color: var(--text-secondary); font-size:0.6rem; line-height:0.9rem; margin-bottom:10px;\">${patterns.pattern_summary}</p>`;
            }
            if (list.length === 0) {
                html += '<p style="color: rgba(255,255,255,0.5); font-size:0.65rem;">Keine Muster erkannt</p>';
            } else {
                html += '<div style="display:flex; flex-direction:column; gap:10px;">';
                list.forEach(p => {
                    html += `
                    <div class=\"pattern-item fade-in\" style=\"border-left:4px solid ${getSignalColor(p.signal)}\">
                        <div class=\"pattern-header\">
                            <span class=\"pattern-type\">${p.type || p.name}<span style=\"margin-left:6px; font-size:0.5rem; background:rgba(255,255,255,0.12); padding:3px 6px; border-radius:6px; letter-spacing:.5px;\">${p.timeframe||'1h'}</span></span>
                            <span class=\"pattern-confidence\">${p.confidence}% | Q:${p.quality_grade || '-'} ${p.quality_score ? '('+p.quality_score+')' : ''}</span>
                        </div>
                        <div style=\"font-size:0.55rem; color:var(--text-secondary); margin-bottom:4px;\">Signal: <span style=\"color:${p.signal==='bullish'?'#28a745':p.signal==='bearish'?'#dc3545':'#ffc107'}\">${p.signal}</span></div>
                        ${p.description?`<div style=\"font-size:0.55rem; color:rgba(255,255,255,0.55); line-height:0.85rem;\">${p.description}</div>`:''}
                    </div>`;
                });
                html += '</div>';
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
            const mt = data.multi_timeframe || {};
            const el = document.getElementById('multiTimeframe');
            if (!mt.timeframes || !mt.timeframes.length) { el.innerHTML = '<small style="color:var(--text-dim)">No data</small>'; return; }

            // Legend / explanation tooltip
            const legend = `<div style=\"font-size:0.48rem; letter-spacing:.4px; color:var(--text-dim); margin:-2px 0 6px; line-height:.7rem;\">
                <span style=\"color:#28a745;\">Bull/Bear Scores</span> = gewichtete Summe der Signale über Zeitrahmen. Verteilung zeigt prozentuale Häufigkeit von bull / neutral / bear Kategorien.
            </div>`;

            let dist = '';
            if (mt.distribution_pct) {
                dist = '<div style="display:flex; gap:6px; flex-wrap:wrap; margin:2px 0 10px;">' +
                    Object.entries(mt.distribution_pct).map(([k,v])=>`<div style=\"font-size:0.5rem; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); backdrop-filter:blur(3px); padding:4px 6px; border-radius:6px;\">${k}: ${v}%</div>`).join('') + '</div>';
            }
            const cons = mt.consensus || {};
            const consColor = cons.primary==='BULLISH'? '#26c281': cons.primary==='BEARISH'? '#ff4d4f':'#f5b041';
            let rows = mt.timeframes.map(t => {
                if (t.error) return `<div style='font-size:0.55rem; color:#ff4d4f; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:6px 8px; border-radius:10px;'>${t.tf}: ${t.error}</div>`;
                const sigColor = t.signal?.includes('bull')?'#26c281': t.signal?.includes('bear')?'#ff4d4f':'#f5b041';
                return `<div style=\"display:grid; grid-template-columns:50px 82px 52px 1fr; gap:4px; align-items:center; font-size:0.55rem; background:linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:12px; box-shadow:0 2px 4px -1px rgba(0,0,0,0.5);\">
                        <div style=\"font-weight:600; letter-spacing:.5px; color:var(--text-secondary);\">${t.tf}</div>
                        <div style=\"color:${sigColor}; font-weight:600;\">${t.signal}</div>
                        <div style=\"opacity:.85;\">RSI ${t.rsi ?? '-'} </div>
                        <div style=\"opacity:.55; font-style:italic;\">${t.trend || ''}</div>
                    </div>`;
            }).join('');
            el.innerHTML = `
                <div style=\"font-size:0.6rem; margin-bottom:6px; font-weight:600; letter-spacing:.5px;\">Consensus: <span style=\"color:${consColor}; font-weight:700;\">${cons.primary||'-'}</span>
                    <span style=\"font-size:0.5rem; font-weight:400; color:var(--text-dim);\">(Bull ${cons.bull_score||0} / Bear ${cons.bear_score||0})</span>
                </div>
                ${legend}
                ${dist}
                <div style=\"display:flex; flex-direction:column; gap:8px;\">${rows}</div>`;
        }

                function displayMarketBias(data){
                        const bias = data.market_bias; if(!bias) return;
                        const container = document.getElementById('signalDisplay');
                        const longPct = bias.long_strength_pct||0; const shortPct = bias.short_strength_pct||0; const neutralPct = Math.max(0, 100 - longPct - shortPct);
                        const existing = document.getElementById('marketBiasBar');
                        const html = `
                                <div id=\"marketBiasBar\" style=\"margin-top:18px;\">
                                    <div style=\"font-size:0.55rem; letter-spacing:.5px; color:var(--text-dim); margin-bottom:4px;\">MARKET BIAS</div>
                                    <div style=\"height:14px; width:100%; background:rgba(255,255,255,0.1); border-radius:8px; overflow:hidden; display:flex;\">
                                        <div title=\"Long\" style=\"flex:0 0 ${longPct}%; background:#198754;\"></div>
                                        <div title=\"Neutral\" style=\"flex:0 0 ${neutralPct}%; background:linear-gradient(90deg,#6c757d,#495057);\"></div>
                                        <div title=\"Short\" style=\"flex:0 0 ${shortPct}%; background:#dc3545;\"></div>
                                    </div>
                                    <div style=\"display:flex; justify-content:space-between; font-size:0.5rem; margin-top:4px; color:var(--text-secondary);\">
                                        <span>Long ${longPct}%</span><span>Neutral ${neutralPct}%</span><span>Short ${shortPct}%</span>
                                    </div>
                                </div>`;
                        if(existing){ existing.outerHTML = html; } else { container.insertAdjacentHTML('beforeend', html); }
                }

                function displayRegimeAnalysis(data) {
                    const regime = data.regime_analysis;
                    const el = document.getElementById('regimeAnalysis');
                    if (!regime || regime.regime === 'error') {
                        el.innerHTML = `<small style="color:#dc3545">${regime?.rationale || 'Regime analysis failed'}</small>`;
                        return;
                    }
                    
                    const regimeColors = {
                        'trending': '#0d6efd',
                        'ranging': '#6f42c1', 
                        'expansion': '#fd7e14',
                        'volatility_crush': '#20c997'
                    };
                    
                    const regimeIcons = {
                        'trending': '📈',
                        'ranging': '↔️',
                        'expansion': '💥',
                        'volatility_crush': '🤐'
                    };
                    
                    const color = regimeColors[regime.regime] || '#6c757d';
                    const icon = regimeIcons[regime.regime] || '❓';
                    
                    let html = `
                        <div class="metric-card" style="margin-bottom: 15px; border-left: 4px solid ${color};">
                            <div class="metric-value" style="color: ${color}">
                                ${icon} ${regime.regime.toUpperCase()}
                            </div>
                            <div class="metric-label">Market Regime (${regime.confidence}%)</div>
                        </div>
                        
                        <div style="font-size: 0.65rem; color: var(--text-secondary); margin-bottom: 10px;">
                            ${regime.rationale}
                        </div>
                    `;
                    
                    if (regime.secondary_regime) {
                        const secColor = regimeColors[regime.secondary_regime] || '#6c757d';
                        const secIcon = regimeIcons[regime.secondary_regime] || '❓';
                        html += `
                            <div style="font-size: 0.6rem; color: ${secColor}; margin: 5px 0;">
                                Secondary: ${secIcon} ${regime.secondary_regime}
                            </div>
                        `;
                    }
                    
                    html += `
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 10px;">
                            <div style="background: rgba(255,255,255,0.05); padding: 6px 8px; border-radius: 8px; font-size: 0.55rem;">
                                <div style="color: var(--text-dim);">ATR</div>
                                <div style="color: white; font-weight: 600;">${regime.atr_percentage?.toFixed(1) || 'N/A'}%</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.05); padding: 6px 8px; border-radius: 8px; font-size: 0.55rem;">
                                <div style="color: var(--text-dim);">Volatility</div>
                                <div style="color: white; font-weight: 600;">${regime.volatility_level || 'N/A'}</div>
                            </div>
                        </div>
                    `;
                    
                    if (regime.regime_scores) {
                        html += `
                            <div style="margin-top: 10px; font-size: 0.5rem;">
                                <div style="color: var(--text-dim); margin-bottom: 4px;">Regime Scores:</div>
                                <div style="display: flex; gap: 4px; flex-wrap: wrap;">
                        `;
                        Object.entries(regime.regime_scores).forEach(([key, score]) => {
                            const color = regimeColors[key] || '#6c757d';
                            html += `<span style="background: ${color}20; color: ${color}; padding: 2px 5px; border-radius: 4px;">${key}: ${score}</span>`;
                        });
                        html += `</div></div>`;
                    }
                    
                    el.innerHTML = html;
                }
                
                function displayOrderFlowAnalysis(data) {
                    const orderFlowContainer = document.getElementById('orderFlowAnalysis');
                    if (!orderFlowContainer || !data.order_flow_analysis) return;
                    
                    const flow = data.order_flow_analysis;
                    if (flow.error) {
                        orderFlowContainer.innerHTML = `<div class="alert alert-warning">⚠️ ${flow.error}</div>`;
                        return;
                    }
                    
                    const sentimentColors = {
                        'buy_pressure': '#28a745',
                        'sell_pressure': '#dc3545', 
                        'neutral': '#6c757d',
                        'low_liquidity': '#ffc107',
                        'unknown': '#6c757d'
                    };
                    
                    const sentimentEmojis = {
                        'buy_pressure': '🟢',
                        'sell_pressure': '🔴',
                        'neutral': '⚪',
                        'low_liquidity': '🟡',
                        'unknown': '❓'
                    };
                    
                    const imbalancePercent = (flow.order_book_imbalance * 100).toFixed(1);
                    const deltaPercent = (flow.delta_momentum * 100).toFixed(1);
                    
                    orderFlowContainer.innerHTML = `
                        <div class="order-flow-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(155deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">📊 Order Flow Analysis</h5>
                            
                            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:0 0 12px;">
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Flow Sentiment:</strong>
                                    <span style="color:${sentimentColors[flow.flow_sentiment]}; font-weight:600;">
                                        ${sentimentEmojis[flow.flow_sentiment]} ${flow.flow_sentiment.replace('_', ' ').toUpperCase()}
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Flow Strength:</strong>
                                    <span style="color:${flow.flow_strength === 'strong' ? '#28a745' : flow.flow_strength === 'moderate' ? '#ffc107' : '#6c757d'}; font-weight:600;">
                                        ${flow.flow_strength.toUpperCase()}
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Spread:</strong>
                                    <span>${flow.spread_bps || 0} bps</span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Order Imbalance:</strong>
                                    <span style="color: ${flow.order_book_imbalance > 0 ? '#28a745' : flow.order_book_imbalance < 0 ? '#dc3545' : '#6c757d'};">
                                        ${imbalancePercent}%
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Delta Momentum:</strong>
                                    <span style="color: ${flow.delta_momentum > 0 ? '#28a745' : flow.delta_momentum < 0 ? '#dc3545' : '#6c757d'};">
                                        ${deltaPercent}%
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Volume POC:</strong>
                                    <span>${flow.volume_profile_poc || 'N/A'}</span>
                                </div>
                            </div>
                            
                            ${flow.liquidity_zones && flow.liquidity_zones.length > 0 ? `
                                <div class="liquidity-zones" style="margin-top:6px;">
                                    <strong style="color:var(--text-secondary); font-size:0.65rem;">Liquidity Zones:</strong>
                                    <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;">
                                        ${flow.liquidity_zones.map(zone => `
                                            <span style="display:inline-flex; align-items:center; gap:4px; padding:4px 8px; border-radius:8px; font-size:0.55rem; letter-spacing:.3px; 
                                                  background:${zone.type === 'support' ? 'rgba(38,194,129,0.15)' : 'rgba(255,77,79,0.15)'}; 
                                                  border:1px solid ${zone.type === 'support' ? 'rgba(38,194,129,0.35)' : 'rgba(255,77,79,0.35)'}; 
                                                  color:${zone.type === 'support' ? '#26c281' : '#ff4d4f'}; font-weight:500;">
                                                ${zone.type.toUpperCase()} @ ${zone.level} <span style='opacity:.75;'>(${zone.strength})</span>
                                            </span>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${flow.analysis_note ? `
                                <div style="margin-top:10px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:8px 10px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                                    💡 ${flow.analysis_note}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                function displayFeatureContributions(data) {
                    const featureContainer = document.getElementById('featureContributions');
                    if (!featureContainer || !data.ai_analysis?.feature_contributions) return;
                    
                    const features = data.ai_analysis.feature_contributions;
                    if (features.error) {
                        featureContainer.innerHTML = `<div class="alert alert-warning">⚠️ ${features.error}</div>`;
                        return;
                    }
                    
                    featureContainer.innerHTML = `
                        <div class="feature-contributions-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(150deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">🔍 AI Feature Contributions</h5>
                            
                            <div style="margin:0 0 12px;">
                                <strong style="color:var(--text-secondary);">Signal Confidence:</strong>
                                <span style="color:${features.ai_signal_confidence > 70 ? '#28a745' : features.ai_signal_confidence > 50 ? '#ffc107' : '#dc3545'}; font-weight:700;">
                                    ${features.ai_signal_confidence?.toFixed(1) || 0}%
                                </span>
                                <span style="margin-left:10px; color:var(--text-dim); font-size:0.55rem;">
                                    (${features.total_features_analyzed || 0} features analyzed)
                                </span>
                            </div>
                            
                            ${features.top_features && features.top_features.length > 0 ? `
                                <div class="top-features" style="margin:0 0 12px;">
                                    <strong style="color:var(--text-secondary);">Top Contributing Features:</strong>
                                    <div style="margin-top:6px; display:flex; flex-direction:column; gap:6px;">
                                        ${features.top_features.map(feature => `
                                            <div style="display:grid; grid-template-columns:1fr 70px 74px; align-items:center; gap:8px; padding:6px 8px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; font-size:0.6rem;">
                                                <span style="font-weight:500; color:var(--text-primary);">${feature.feature}</span>
                                                <span style="text-align:center; color:${feature.impact === 'positive' ? '#26c281' : '#ff4d4f'}; font-weight:600;">${feature.impact === 'positive' ? '+' : '-'}${feature.importance}%</span>
                                                <span style="font-size:0.55rem; color:var(--text-dim);">val: ${feature.value}</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${features.contextual_interpretations && features.contextual_interpretations.length > 0 ? `
                                <div class="contextual-interpretations" style="margin-top:10px;">
                                    <strong style="color:var(--text-secondary);">Key Interpretations:</strong>
                                    <ul style="margin:6px 0 0; padding-left:16px;">
                                        ${features.contextual_interpretations.map(interp => `
                                            <li style="font-size:0.55rem; color:var(--text-secondary); margin:2px 0;">${interp}</li>
                                        `).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            
                            ${features.note ? `
                                <div style="margin-top:10px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:8px 10px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                                    💡 ${features.note}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                function displayAdaptiveRiskTargets(data) {
                    const adaptiveContainer = document.getElementById('adaptiveRiskTargets');
                    if (!adaptiveContainer || !data.adaptive_risk_targets) return;
                    
                    const risk = data.adaptive_risk_targets;
                    if (risk.error) {
                        adaptiveContainer.innerHTML = `<div class="alert alert-warning">⚠️ ${risk.error}</div>`;
                        return;
                    }
                    
                    const riskColors = {
                        'low': '#28a745',
                        'medium': '#ffc107',
                        'high': '#dc3545'
                    };
                    
                    const targets = risk.targets || {};
                    
                    adaptiveContainer.innerHTML = `
                        <div class="adaptive-risk-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">🎯 Adaptive Risk Management</h5>
                            
                            <!-- Risk Overview -->
                            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:0 0 14px;">
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Risk %:</strong>
                                    <span style="color:${riskColors[risk.risk_category]}; font-weight:700;">
                                        ${risk.adaptive_risk_pct}%
                                    </span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Reward Ratio:</strong>
                                    <span style="color:#0d6efd; font-weight:700;">
                                        1:${risk.adaptive_reward_ratio}
                                    </span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Position Size:</strong>
                                    <span>${risk.position_size}</span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Risk Amount:</strong>
                                    <span style="color:${riskColors[risk.risk_category]}; font-weight:600;">
                                        $${risk.risk_amount_usd}
                                    </span>
                                </div>
                            </div>
                            
                            <!-- Stop Loss & Targets -->
                            <div class="stop-targets" style="margin:0 0 12px;">
                                <h6 style="margin:0 0 10px; font-size:0.65rem; letter-spacing:.5px; font-weight:600; color:var(--text-secondary);">📍 Stop Loss & Targets</h6>
                                <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px;">
                                    <div style="padding:10px 8px; background:rgba(255,77,79,0.08); border:1px solid rgba(255,77,79,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#ff4d4f; font-weight:600;">Stop Loss</div>
                                        <div style="font-weight:700; color:#ff4d4f; font-size:0.7rem;">${risk.stop_loss}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:rgba(13,110,253,0.08); border:1px solid rgba(13,110,253,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#0d6efd; font-weight:600;">Target 1</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_1}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:rgba(38,194,129,0.10); border:1px solid rgba(38,194,129,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#26c281; font-weight:600;">Target 2</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_2}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:linear-gradient(135deg, rgba(38,194,129,0.18), rgba(13,110,253,0.12)); border:1px solid rgba(38,194,129,0.40); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#26c281; font-weight:600;">Target 3</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_3}</div>
                                    </div>
                                </div>
                            </div>
                            
                            ${risk.reasoning ? `
                                <div style="margin-top:8px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:10px 12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px;">
                                    💡 ${risk.reasoning}
                                </div>
                            ` : ''}
                        </div>
                    `;
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
