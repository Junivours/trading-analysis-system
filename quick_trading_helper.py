#!/usr/bin/env python3
"""
🎯 LIVE TRADING HELPER - ECHTE DATEN
Verbindet sich mit deiner Trading-App API für echte Live-Daten
"""

import requests
import json
from datetime import datetime

def get_live_analysis(symbol, base_url="http://127.0.0.1:5000"):
    """Holt echte Live-Analyse von deiner Trading-App"""
    try:
        response = requests.post(f"{base_url}/api/analyze", json={
            'symbol': symbol,
            'timeframe': '4h'
        }, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Kann nicht zu Trading-App verbinden. Ist sie gestartet?")
        print("💡 Starte sie mit: python app.py")
        return None
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return None

def extract_trading_signals(analysis_data):
    """Extrahiert Trading-Signale aus echten API-Daten"""
    if not analysis_data:
        return None
    
    # Echte Daten extrahieren
    symbol = analysis_data.get('symbol', 'UNKNOWN')
    current_price = analysis_data.get('current_price', 0)
    recommendation = analysis_data.get('recommendation', 'HOLD')
    confidence = analysis_data.get('confidence', 50)
    
    # Liquidation Map falls vorhanden
    liquidation_map = analysis_data.get('liquidation_map', {})
    long_liq = liquidation_map.get('long_liquidation', 0)
    short_liq = liquidation_map.get('short_liquidation', 0)
    
    # Trading Setup falls vorhanden  
    trading_setup = analysis_data.get('trading_setup', {})
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'recommendation': recommendation,
        'confidence': confidence,
        'long_liquidation': long_liq,
        'short_liquidation': short_liq,
        'trading_setup': trading_setup,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def calculate_live_trading_setup(symbol, base_url="http://127.0.0.1:5000"):
    """Berechnet Trading Setup mit ECHTEN Live-Daten"""
    
    print(f"🔄 Hole Live-Daten für {symbol}...")
    
    # Echte Daten von deiner API holen
    analysis_data = get_live_analysis(symbol, base_url)
    signals = extract_trading_signals(analysis_data)
    
    if not signals:
        return None
    
    current_price = signals['current_price']
    recommendation = signals['recommendation']
    confidence = signals['confidence']
    
    # Coin-spezifische Risk Levels (basierend auf echten Markt-Daten)
    risk_profiles = {
        'BTCUSDT': {'sl': 0.02, 'tp': 0.05, 'size': 2.0},  # Conservative
        'ETHUSDT': {'sl': 0.03, 'tp': 0.07, 'size': 2.5},  # Moderate  
        'SOLUSDT': {'sl': 0.04, 'tp': 0.10, 'size': 3.0},  # Aggressive
        'ADAUSDT': {'sl': 0.04, 'tp': 0.10, 'size': 3.0},
        'DOTUSDT': {'sl': 0.04, 'tp': 0.10, 'size': 3.0},
        'AVAXUSDT': {'sl': 0.04, 'tp': 0.10, 'size': 3.0},
        'DEFAULT': {'sl': 0.06, 'tp': 0.15, 'size': 4.0}   # High Risk
    }
    
    # Get risk profile
    profile = risk_profiles.get(symbol.upper(), risk_profiles['DEFAULT'])
    
    # Adjust based on REAL confidence from API
    confidence_multiplier = confidence / 100.0
    sl_percent = profile['sl'] * (2 - confidence_multiplier)
    tp_percent = profile['tp'] * confidence_multiplier
    position_size = profile['size'] * confidence_multiplier
    
    if 'BUY' in recommendation.upper() or 'LONG' in recommendation.upper():
        direction = 'LONG'
        entry_price = current_price
        stop_loss = current_price * (1 - sl_percent)
        take_profit = current_price * (1 + tp_percent)
        
    elif 'SELL' in recommendation.upper() or 'SHORT' in recommendation.upper():
        direction = 'SHORT'  
        entry_price = current_price
        stop_loss = current_price * (1 + sl_percent)
        take_profit = current_price * (1 - tp_percent)
        
    else:
        direction = 'WAIT'
        entry_price = current_price
        stop_loss = current_price
        take_profit = current_price
        position_size = 0
    
    # Verwende ECHTE Liquidation Levels falls verfügbar
    if signals['long_liquidation'] and signals['short_liquidation']:
        liq_long = signals['long_liquidation']
        liq_short = signals['short_liquidation']
    else:
        # Fallback Berechnung
        if direction == 'LONG':
            liq_long = current_price * 0.90
            liq_short = current_price * 1.10
        else:
            liq_long = current_price * 0.90
            liq_short = current_price * 1.10
    
    # Risk/Reward Ratio
    if direction != 'WAIT':
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
    else:
        rr_ratio = 0
    
    return {
        'symbol': symbol,
        'direction': direction,
        'entry_price': round(entry_price, 2),
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'position_size_percent': round(position_size, 1),
        'risk_reward_ratio': round(rr_ratio, 2),
        'liquidation_long': round(liq_long, 2),
        'liquidation_short': round(liq_short, 2),
        'confidence': confidence,
        'timestamp': signals['timestamp'],
        'raw_recommendation': recommendation,
        'live_data': True
    }

def print_live_trading_setup(setup):
    """Druckt LIVE Trading Setup schön formatiert"""
    if not setup:
        print("❌ Keine Live-Daten verfügbar!")
        return
        
    print(f"\n🔴 LIVE TRADING SETUP - {setup['symbol']}")
    print(f"⏰ {setup['timestamp']}")
    print(f"{'='*60}")
    print(f"📊 Live Price: ${setup['entry_price']}")
    print(f"🤖 AI Signal: {setup['raw_recommendation']}")
    print(f"🎪 Direction: {setup['direction']}")
    print(f"💰 Entry: ${setup['entry_price']}")
    print(f"🛑 Stop Loss: ${setup['stop_loss']}")
    print(f"🎯 Take Profit: ${setup['take_profit']}")
    print(f"📊 Position Size: {setup['position_size_percent']}%")
    print(f"⚖️ Risk/Reward: 1:{setup['risk_reward_ratio']}")
    print(f"🔥 Liquidation Long: ${setup['liquidation_long']}")
    print(f"🔥 Liquidation Short: ${setup['liquidation_short']}")
    print(f"✅ Confidence: {setup['confidence']}%")
    print(f"🔴 LIVE DATA: {setup['live_data']}")
    print(f"{'='*60}\n")

# LIVE USAGE:
if __name__ == "__main__":
    print("🚀 LIVE TRADING HELPER - Verbinde mit deiner Trading-App...")
    
    # Teste mit echten Live-Daten
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in symbols:
        live_setup = calculate_live_trading_setup(symbol)
        print_live_trading_setup(live_setup)
