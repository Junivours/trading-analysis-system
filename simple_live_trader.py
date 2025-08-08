#!/usr/bin/env python3
"""
🔴 SIMPLE LIVE TRADING HELPER
Einfache Lösung für echte Trading-Signale ohne komplexe Imports
"""

import requests
import json
from datetime import datetime

def get_binance_price(symbol):
    """Holt aktuellen Preis direkt von Binance"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
        return None
    except:
        return None

def get_binance_klines(symbol, interval='4h', limit=50):
    """Holt Kerzen-Daten direkt von Binance"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Konvertiere zu nutzbarem Format
            candles = []
            for kline in data:
                candles.append({
                    'open': float(kline[1]),
                    'high': float(kline[2]), 
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            return candles
        return None
    except:
        return None

def calculate_simple_rsi(prices, period=14):
    """Berechnet RSI"""
    if len(prices) < period + 1:
        return 50
        
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)

def calculate_simple_ema(prices, period):
    """Berechnet EMA"""
    if len(prices) < period:
        return sum(prices) / len(prices)
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def analyze_simple_signal(symbol):
    """Einfache Trading-Signal-Analyse mit echten Live-Daten"""
    
    print(f"🔄 Hole Live-Daten für {symbol}...")
    
    # 1. Aktueller Preis
    current_price = get_binance_price(symbol)
    if not current_price:
        print(f"❌ Kann Preis für {symbol} nicht laden")
        return None
    
    print(f"💰 Live-Preis: ${current_price:.2f}")
    
    # 2. Historische Daten für Indikatoren
    candles = get_binance_klines(symbol, '4h', 50)
    if not candles:
        print(f"❌ Kann Kerzen-Daten für {symbol} nicht laden")
        return None
    
    # 3. Technische Indikatoren berechnen
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    
    rsi = calculate_simple_rsi(closes)
    ema_12 = calculate_simple_ema(closes, 12)
    ema_26 = calculate_simple_ema(closes, 26)
    
    # 4. Volatility (vereinfacht)
    recent_prices = closes[-20:]
    volatility = (max(recent_prices) - min(recent_prices)) / min(recent_prices) * 100
    
    print(f"📈 RSI: {rsi}")
    print(f"📈 EMA12: ${ema_12:.2f}")
    print(f"📈 EMA26: ${ema_26:.2f}")
    print(f"📈 Volatility: {volatility:.2f}%")
    
    # 5. Signal-Logik (vereinfacht aber effektiv)
    signals = []
    confidence = 50
    
    # RSI Signale
    if rsi < 30:
        signals.append("BUY")
        confidence += 15
    elif rsi > 70:
        signals.append("SELL")
        confidence += 15
    
    # EMA Signale
    if current_price > ema_12 > ema_26:
        signals.append("BUY")
        confidence += 10
    elif current_price < ema_12 < ema_26:
        signals.append("SELL")
        confidence += 10
    
    # Trend Signal
    if closes[-1] > closes[-5]:  # 5-Kerzen Trend
        signals.append("BUY")
        confidence += 5
    elif closes[-1] < closes[-5]:
        signals.append("SELL")
        confidence += 5
    
    # Haupt-Signal bestimmen
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    
    if buy_count > sell_count:
        recommendation = "BUY"
    elif sell_count > buy_count:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
        confidence = max(40, confidence - 20)
    
    # Confidence begrenzen
    confidence = min(90, max(30, confidence))
    
    print(f"🤖 Signal: {recommendation}")
    print(f"✅ Confidence: {confidence}%")
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'recommendation': recommendation,
        'confidence': confidence,
        'rsi': rsi,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'volatility': volatility,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

def calculate_live_trading_setup(data):
    """Berechnet Trading Setup mit echten Live-Daten"""
    
    if not data:
        return None
    
    symbol = data['symbol']
    price = data['current_price']
    recommendation = data['recommendation']
    confidence = data['confidence']
    
    # Coin-spezifische Parameter
    if 'BTC' in symbol:
        sl_percent = 0.02  # 2%
        tp_percent = 0.05  # 5%
        size_percent = 2.0
    elif 'ETH' in symbol:
        sl_percent = 0.03  # 3%
        tp_percent = 0.07  # 7%
        size_percent = 2.5
    else:
        sl_percent = 0.04  # 4%
        tp_percent = 0.10  # 10%
        size_percent = 3.0
    
    # Confidence Anpassung
    conf_multiplier = confidence / 100.0
    sl_percent *= (2 - conf_multiplier)  # Higher confidence = tighter SL
    tp_percent *= conf_multiplier        # Higher confidence = higher TP
    size_percent *= conf_multiplier      # Higher confidence = bigger size
    
    # Trading Setup berechnen
    if recommendation == 'BUY':
        direction = 'LONG'
        entry = price
        stop_loss = price * (1 - sl_percent)
        take_profit = price * (1 + tp_percent)
    elif recommendation == 'SELL':
        direction = 'SHORT'
        entry = price
        stop_loss = price * (1 + sl_percent)
        take_profit = price * (1 - tp_percent)
    else:
        direction = 'WAIT'
        entry = price
        stop_loss = price
        take_profit = price
        size_percent = 0
    
    # Risk/Reward
    if direction != 'WAIT':
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = reward / risk if risk > 0 else 0
    else:
        rr_ratio = 0
    
    return {
        'direction': direction,
        'entry_price': round(entry, 2),
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'position_size': round(size_percent, 1),
        'risk_reward_ratio': round(rr_ratio, 2)
    }

def print_complete_live_report(symbol):
    """Kompletter Live-Trading-Report"""
    
    print("\n" + "="*60)
    print(f"🔴 LIVE TRADING REPORT - {symbol}")
    print("="*60)
    
    # 1. Live-Analyse
    analysis = analyze_simple_signal(symbol)
    if not analysis:
        print("❌ Analyse fehlgeschlagen!")
        return
    
    # 2. Trading Setup
    setup = calculate_live_trading_setup(analysis)
    
    # 3. Report ausgeben
    print(f"\n📊 MARKT-ANALYSE:")
    print(f"   Live-Preis: ${analysis['current_price']:.2f}")
    print(f"   Signal: {analysis['recommendation']}")
    print(f"   Confidence: {analysis['confidence']}%")
    print(f"   RSI: {analysis['rsi']}")
    print(f"   Volatility: {analysis['volatility']:.2f}%")
    
    if setup and setup['direction'] != 'WAIT':
        print(f"\n🎯 TRADING SETUP:")
        print(f"   Direction: {setup['direction']}")
        print(f"   Entry: ${setup['entry_price']}")
        print(f"   Stop Loss: ${setup['stop_loss']}")
        print(f"   Take Profit: ${setup['take_profit']}")
        print(f"   Position Size: {setup['position_size']}%")
        print(f"   Risk/Reward: 1:{setup['risk_reward_ratio']}")
        
        # Trading Empfehlung
        if analysis['confidence'] >= 70:
            print(f"\n💡 EMPFEHLUNG: 🟢 STARKES SIGNAL - Trade empfohlen!")
        elif analysis['confidence'] >= 60:
            print(f"\n💡 EMPFEHLUNG: 🟡 Moderates Signal - Vorsichtig traden")
        else:
            print(f"\n💡 EMPFEHLUNG: 🔴 Schwaches Signal - Warten")
    else:
        print(f"\n💡 EMPFEHLUNG: 🔴 WAIT - Kein klares Signal")
    
    print("="*60 + "\n")

def main():
    """Hauptfunktion"""
    print("🚀 SIMPLE LIVE TRADING HELPER")
    print("Mit echten Binance Live-Daten")
    print("-" * 40)
    
    # Standard oder User Input
    user_input = input("Coin eingeben (oder Enter für BTC): ").strip().upper()
    if user_input:
        if not user_input.endswith('USDT'):
            symbol = user_input + 'USDT'
        else:
            symbol = user_input
    else:
        symbol = 'BTCUSDT'
    
    print_complete_live_report(symbol)

if __name__ == "__main__":
    main()
