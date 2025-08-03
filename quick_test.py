#!/usr/bin/env python3
"""
⚡ SCHNELLER MINI-TEST
Testet die wichtigsten Funktionen in unter 30 Sekunden
"""

import sys
import time
from datetime import datetime

def quick_test():
    print("⚡ SCHNELLER SYSTEM-TEST")
    print("=" * 40)
    
    # Test 1: Imports
    print("🔍 Teste Imports...")
    try:
        from binance_api import fetch_binance_data, test_binance_connection
        from ml_engine import RealMLTradingEngine
        from validation_suite import DataValidationSuite
        from real_data_enforcer import RealDataEnforcer
        print("✅ Alle Imports erfolgreich")
    except ImportError as e:
        print(f"❌ Import Fehler: {e}")
        return False
    
    # Test 2: Binance Verbindung
    print("🔍 Teste Binance Verbindung...")
    try:
        if test_binance_connection():
            print("✅ Binance API verbunden")
        else:
            print("❌ Binance Verbindung fehlgeschlagen")
            return False
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
        return False
    
    # Test 3: Daten abrufen
    print("🔍 Teste Daten abrufen...")
    try:
        data = fetch_binance_data('BTCUSDT', '1h', 50)
        if data and len(data) > 30:
            price = float(data[-1][4])
            print(f"✅ Daten erhalten - BTC Preis: ${price:,.2f}")
        else:
            print("❌ Nicht genug Daten erhalten")
            return False
    except Exception as e:
        print(f"❌ Datenfehler: {e}")
        return False
    
    # Test 4: Technische Indikatoren
    print("🔍 Teste technische Indikatoren...")
    try:
        enforcer = RealDataEnforcer()
        indicators = enforcer.get_real_technical_indicators('BTCUSDT')
        
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
            print(f"✅ RSI berechnet: {rsi:.2f}")
        else:
            print("❌ Indikatoren fehlgeschlagen")
            return False
    except Exception as e:
        print(f"❌ Indikator Fehler: {e}")
        return False
    
    # Test 5: ML Engine (ohne Training)
    print("🔍 Teste ML Engine...")
    try:
        ml_engine = RealMLTradingEngine()
        # Test ohne Training - nur Fallback
        predictions = ml_engine._fallback_predictions()
        
        if predictions and 'ensemble' in predictions:
            print("✅ ML Engine funktioniert")
        else:
            print("❌ ML Engine Fehler")
            return False
    except Exception as e:
        print(f"❌ ML Engine Fehler: {e}")
        return False
    
    print("\n🎉 ALLE SCHNELLTESTS BESTANDEN!")
    print("💡 Für vollständige Tests: python local_test.py")
    return True

if __name__ == "__main__":
    start = time.time()
    success = quick_test()
    duration = time.time() - start
    print(f"⏱️  Test dauerte: {duration:.1f} Sekunden")
    
    if success:
        print("🚀 System ist bereit für vollständige Tests!")
    else:
        print("⚠️  Probleme gefunden - prüfe die Konfiguration")
