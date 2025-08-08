#!/usr/bin/env python3
"""
🔴 LIVE TRADING CONSOLE
Direkte Integration in deine Trading-App für echte Live-Signale
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your trading app functions
from app import (
    get_market_data_binance, 
    calculate_technical_indicators, 
    analyze_market_sentiment,
    get_coin_specific_setup,
    calculate_liquidation_zones
)

def get_live_trading_signal(symbol):
    """Holt echte Live-Trading-Signale direkt von der App-Engine"""
    
    print(f"🔄 Analysiere {symbol} mit LIVE-Daten...")
    
    try:
        # 1. Echte Marktdaten holen
        market_result = get_market_data_binance(symbol, '4h')
        if not market_result['success']:
            print(f"❌ Marktdaten-Fehler: {market_result.get('error', 'Unbekannt')}")
            return None
            
        candles = market_result['data']
        current_price = candles[-1]['close']
        
        print(f"📊 Live-Preis: ${current_price}")
        
        # 2. Technische Indikatoren berechnen
        tech_indicators = calculate_technical_indicators(candles)
        if 'error' in tech_indicators:
            print(f"❌ Indikator-Fehler: {tech_indicators['error']}")
            return None
            
        print(f"📈 RSI: {tech_indicators['rsi']:.1f}")
        print(f"📈 MACD: {tech_indicators['macd']:.6f}")
        
        # 3. Market Sentiment
        analysis_result = analyze_market_sentiment(
            symbol, candles, tech_indicators, current_price
        )
        
        recommendation = analysis_result.get('recommendation', 'HOLD')
        confidence = analysis_result.get('confidence', 50)
        
        print(f"🤖 AI-Signal: {recommendation}")
        print(f"✅ Confidence: {confidence}%")
        
        # 4. Trading Setup berechnen
        trading_setup = get_coin_specific_setup(
            symbol, current_price, tech_indicators, recommendation
        )
        
        # 5. Liquidation Zones
        liquidation_zones = calculate_liquidation_zones(symbol, current_price)
        main_liq = liquidation_zones[1] if len(liquidation_zones) > 1 else liquidation_zones[0]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'recommendation': recommendation,
            'confidence': confidence,
            'trading_setup': trading_setup,
            'liquidation_long': main_liq['long_liquidation'],
            'liquidation_short': main_liq['short_liquidation'],
            'tech_indicators': tech_indicators,
            'timestamp': market_result.get('timestamp', 'now')
        }
        
    except Exception as e:
        print(f"❌ Fehler bei Live-Analyse: {e}")
        return None

def print_live_trading_report(data):
    """Druckt kompletten Live-Trading-Report"""
    
    if not data:
        print("❌ Keine Live-Daten verfügbar!")
        return
        
    print("\n" + "="*70)
    print(f"🔴 LIVE TRADING REPORT - {data['symbol']}")
    print(f"⏰ {data['timestamp']}")
    print("="*70)
    
    # Markt-Info
    print(f"📊 Live-Preis: ${data['current_price']:.2f}")
    print(f"🤖 KI-Signal: {data['recommendation']}")
    print(f"✅ Confidence: {data['confidence']:.0f}%")
    
    # Technische Indikatoren
    tech = data['tech_indicators']
    print(f"\n📈 TECHNISCHE INDIKATOREN:")
    print(f"   RSI: {tech['rsi']:.1f}")
    print(f"   MACD: {tech['macd']:.6f}")
    print(f"   Volatility: {tech['volatility']:.2f}%")
    print(f"   ATR: {tech['atr']:.6f}")
    
    # Trading Setup
    setup = data['trading_setup']
    if setup:
        print(f"\n🎯 TRADING SETUP:")
        print(f"   Direction: {setup.get('direction', 'WAIT')}")
        print(f"   Entry: ${setup.get('entry_price', 0):.2f}")
        print(f"   Stop Loss: ${setup.get('stop_loss', 0):.2f}")
        print(f"   Take Profit: ${setup.get('take_profit', 0):.2f}")
        print(f"   Position Size: {setup.get('position_size', 0):.1f}%")
        print(f"   Risk/Reward: 1:{setup.get('risk_reward_ratio', 0):.2f}")
    
    # Liquidation Levels
    print(f"\n🔥 LIQUIDATION ZONES:")
    print(f"   Long Liquidation: ${data['liquidation_long']:.0f}")
    print(f"   Short Liquidation: ${data['liquidation_short']:.0f}")
    
    # Trading Empfehlung
    confidence = data['confidence']
    recommendation = data['recommendation']
    
    print(f"\n💡 TRADING EMPFEHLUNG:")
    if confidence >= 70:
        print(f"   🟢 STARKES SIGNAL - {recommendation}")
        print(f"   🎯 Empfehlung: Trade ausführen")
    elif confidence >= 60:
        print(f"   🟡 MODERATES SIGNAL - {recommendation}")  
        print(f"   ⚠️ Empfehlung: Vorsichtig traden")
    else:
        print(f"   🔴 SCHWACHES SIGNAL - {recommendation}")
        print(f"   🛑 Empfehlung: Warten auf bessere Gelegenheit")
    
    print("="*70 + "\n")

def main():
    """Hauptfunktion für Live-Trading-Console"""
    
    print("🚀 LIVE TRADING CONSOLE")
    print("Direkte Integration mit deiner Trading-App Engine")
    print("-" * 50)
    
    # Standard Coins
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Oder User Input
    user_symbol = input("Coin eingeben (oder Enter für BTC/ETH/SOL): ").strip().upper()
    if user_symbol:
        if not user_symbol.endswith('USDT'):
            user_symbol += 'USDT'
        symbols = [user_symbol]
    
    # Live-Analyse für alle Symbols
    for symbol in symbols:
        live_data = get_live_trading_signal(symbol)
        print_live_trading_report(live_data)

if __name__ == "__main__":
    main()
