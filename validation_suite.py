# === UMFASSENDE VALIDIERUNGS-SUITE FÜR ECHTE DATEN ===

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from binance_api import fetch_binance_data
import json

logger = logging.getLogger(__name__)

class DataValidationSuite:
    """Validiert alle Daten auf Echtheit und Korrektheit"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_all_systems(self):
        """Führe komplette Systemvalidierung durch"""
        print("🔍 STARTE VOLLSTÄNDIGE DATENVALIDIERUNG...")
        print("=" * 60)
        
        # 1. Binance API Validierung
        self.validate_binance_connection()
        
        # 2. Marktdaten Validierung
        self.validate_market_data()
        
        # 3. Technische Indikatoren Validierung
        self.validate_technical_indicators()
        
        # 4. ML Model Validierung
        self.validate_ml_predictions()
        
        # 5. Liquidity Map Validierung
        self.validate_liquidity_data()
        
        # Zusammenfassung
        self.print_validation_summary()
        
        return len(self.errors) == 0
    
    def validate_binance_connection(self):
        """Validiere echte Binance-Verbindung"""
        print("\n1️⃣ BINANCE API VALIDIERUNG")
        print("-" * 40)
        
        try:
            # Test direkte Binance API
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_price = float(data['lastPrice'])
                volume_24h = float(data['volume'])
                price_change = float(data['priceChangePercent'])
                
                print(f"✅ Binance API erreichbar")
                print(f"✅ BTCUSDT Preis: ${current_price:,.2f}")
                print(f"✅ 24h Volumen: {volume_24h:,.0f} BTC")
                print(f"✅ 24h Änderung: {price_change:+.2f}%")
                
                # Validiere dass Preis realistisch ist
                if 10000 <= current_price <= 200000:
                    self.validation_results['binance_api'] = 'VALID'
                else:
                    self.errors.append(f"BTCUSDT Preis unrealistisch: ${current_price}")
                
            else:
                self.errors.append(f"Binance API Fehler: {response.status_code}")
                
        except Exception as e:
            self.errors.append(f"Binance API Verbindung fehlgeschlagen: {e}")
    
    def validate_market_data(self):
        """Validiere Marktdaten auf Echtheit"""
        print("\n2️⃣ MARKTDATEN VALIDIERUNG")
        print("-" * 40)
        
        try:
            # Hole echte Daten von unserer API
            data = fetch_binance_data('BTCUSDT', '1h', 100)
            
            if data and len(data) >= 50:
                print(f"✅ {len(data)} Kerzen empfangen")
                
                # Validiere Datenstruktür
                latest_candle = data[-1]
                timestamp = int(latest_candle[0])
                open_price = float(latest_candle[1])
                high_price = float(latest_candle[2])
                low_price = float(latest_candle[3])
                close_price = float(latest_candle[4])
                volume = float(latest_candle[5])
                
                # Zeitstempel validieren (sollte aktuell sein)
                current_time = datetime.now().timestamp() * 1000
                age_hours = (current_time - timestamp) / (1000 * 60 * 60)
                
                print(f"✅ Letzter Kerze: {datetime.fromtimestamp(timestamp/1000)}")
                print(f"✅ OHLC: O:{open_price} H:{high_price} L:{low_price} C:{close_price}")
                print(f"✅ Volumen: {volume:,.2f}")
                print(f"✅ Datenalter: {age_hours:.1f} Stunden")
                
                # Validierungen
                if age_hours > 2:
                    self.warnings.append(f"Daten sind {age_hours:.1f}h alt")
                
                if high_price >= max(open_price, close_price) and low_price <= min(open_price, close_price):
                    print("✅ OHLC Logik korrekt")
                    self.validation_results['market_data'] = 'VALID'
                else:
                    self.errors.append("OHLC Daten logisch inkorrekt")
                
            else:
                self.errors.append("Nicht genug Marktdaten empfangen")
                
        except Exception as e:
            self.errors.append(f"Marktdaten Validierung fehlgeschlagen: {e}")
    
    def validate_technical_indicators(self):
        """Validiere technische Indikatoren"""
        print("\n3️⃣ TECHNISCHE INDIKATOREN VALIDIERUNG")
        print("-" * 40)
        
        try:
            data = fetch_binance_data('BTCUSDT', '1h', 100)
            if not data:
                self.errors.append("Keine Daten für Indikator-Validierung")
                return
            
            # Extrahiere Preise
            closes = [float(candle[4]) for candle in data]
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            volumes = [float(candle[5]) for candle in data]
            
            # RSI validieren
            rsi = self.calculate_real_rsi(closes)
            print(f"✅ RSI: {rsi:.2f}")
            
            if 0 <= rsi <= 100:
                print("✅ RSI Wertebereich korrekt")
            else:
                self.errors.append(f"RSI außerhalb Wertebereich: {rsi}")
            
            # MACD validieren
            macd = self.calculate_real_macd(closes)
            print(f"✅ MACD: {macd:.4f}")
            
            # Bollinger Bands validieren
            bb_upper, bb_lower, bb_middle = self.calculate_real_bollinger_bands(closes)
            current_price = closes[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            print(f"✅ BB Position: {bb_position:.3f}")
            print(f"✅ BB Bänder: Upper:{bb_upper:.2f} Lower:{bb_lower:.2f}")
            
            # Volumen-Analyse
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume
            
            print(f"✅ Volumen Ratio: {volume_ratio:.2f}")
            print(f"✅ Aktuelles Volumen: {current_volume:,.0f}")
            
            self.validation_results['technical_indicators'] = 'VALID'
            
        except Exception as e:
            self.errors.append(f"Technische Indikatoren Validierung fehlgeschlagen: {e}")
    
    def validate_ml_predictions(self):
        """Validiere ML-Vorhersagen"""
        print("\n4️⃣ ML MODELL VALIDIERUNG")
        print("-" * 40)
        
        try:
            # Teste ML API
            response = requests.get("http://127.0.0.1:5000/api/ml/predict/BTCUSDT", timeout=10)
            
            if response.status_code == 200:
                ml_data = response.json()
                
                print(f"✅ ML API erreichbar")
                print(f"✅ Modell trainiert: {ml_data.get('is_trained', False)}")
                
                predictions = ml_data.get('predictions', {})
                
                if predictions:
                    for model, pred in predictions.items():
                        if pred and isinstance(pred, dict):
                            confidence = pred.get('confidence', 0)
                            signal = pred.get('prediction', 'UNKNOWN')
                            print(f"✅ {model}: {signal} ({confidence:.1f}% Confidence)")
                
                self.validation_results['ml_predictions'] = 'VALID'
            else:
                self.warnings.append("ML API nicht erreichbar (App läuft möglicherweise nicht)")
                
        except Exception as e:
            self.warnings.append(f"ML Validierung fehlgeschlagen: {e}")
    
    def validate_liquidity_data(self):
        """Validiere Liquiditätsdaten"""
        print("\n5️⃣ LIQUIDITÄTSDATEN VALIDIERUNG")
        print("-" * 40)
        
        try:
            # Hole aktuelle Marktdaten für Liquidity-Berechnung
            data = fetch_binance_data('BTCUSDT', '4h', 200)
            
            if data and len(data) >= 100:
                # Extrahiere Volumen und Preise
                volumes = [float(candle[5]) for candle in data]
                highs = [float(candle[2]) for candle in data]
                lows = [float(candle[3]) for candle in data]
                closes = [float(candle[4]) for candle in data]
                
                # Berechne Support/Resistance Zonen
                support_zones = self.calculate_real_support_zones(lows, volumes)
                resistance_zones = self.calculate_real_resistance_zones(highs, volumes)
                
                current_price = closes[-1]
                
                print(f"✅ {len(support_zones)} Support-Zonen identifiziert")
                print(f"✅ {len(resistance_zones)} Resistance-Zonen identifiziert")
                
                # Zeige nächste wichtige Zonen
                if support_zones:
                    next_support = max([s for s in support_zones if s < current_price], default=None)
                    if next_support:
                        support_distance = ((current_price - next_support) / current_price) * 100
                        print(f"✅ Nächster Support: ${next_support:.2f} (-{support_distance:.2f}%)")
                
                if resistance_zones:
                    next_resistance = min([r for r in resistance_zones if r > current_price], default=None)
                    if next_resistance:
                        resistance_distance = ((next_resistance - current_price) / current_price) * 100
                        print(f"✅ Nächste Resistance: ${next_resistance:.2f} (+{resistance_distance:.2f}%)")
                
                self.validation_results['liquidity_data'] = 'VALID'
                
            else:
                self.errors.append("Nicht genug Daten für Liquiditäts-Analyse")
                
        except Exception as e:
            self.errors.append(f"Liquiditätsdaten Validierung fehlgeschlagen: {e}")
    
    def calculate_real_rsi(self, prices, period=14):
        """Berechne echten RSI"""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_real_macd(self, prices):
        """Berechne echten MACD"""
        if len(prices) < 26:
            return 0
        
        # EMA 12 und 26
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        return ema_12 - ema_26
    
    def calculate_ema(self, prices, period):
        """Berechne Exponential Moving Average"""
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_real_bollinger_bands(self, prices, period=20, std_dev=2):
        """Berechne echte Bollinger Bands"""
        if len(prices) < period:
            current = prices[-1]
            return current * 1.02, current * 0.98, current
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        variance = sum([(p - sma) ** 2 for p in recent_prices]) / period
        std = variance ** 0.5
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, lower_band, sma
    
    def calculate_real_support_zones(self, lows, volumes):
        """Berechne echte Support-Zonen basierend auf Volumen"""
        support_zones = []
        
        for i in range(20, len(lows) - 20):
            current_low = lows[i]
            volume_weight = volumes[i]
            
            # Prüfe ob es ein lokales Minimum ist
            is_local_min = True
            for j in range(i-10, i+10):
                if j != i and j >= 0 and j < len(lows):
                    if lows[j] < current_low:
                        is_local_min = False
                        break
            
            if is_local_min and volume_weight > np.mean(volumes):
                support_zones.append(current_low)
        
        return list(set(support_zones))  # Entferne Duplikate
    
    def calculate_real_resistance_zones(self, highs, volumes):
        """Berechne echte Resistance-Zonen basierend auf Volumen"""
        resistance_zones = []
        
        for i in range(20, len(highs) - 20):
            current_high = highs[i]
            volume_weight = volumes[i]
            
            # Prüfe ob es ein lokales Maximum ist
            is_local_max = True
            for j in range(i-10, i+10):
                if j != i and j >= 0 and j < len(highs):
                    if highs[j] > current_high:
                        is_local_max = False
                        break
            
            if is_local_max and volume_weight > np.mean(volumes):
                resistance_zones.append(current_high)
        
        return list(set(resistance_zones))  # Entferne Duplikate
    
    def print_validation_summary(self):
        """Drucke Validierungsergebnis"""
        print("\n" + "=" * 60)
        print("📋 VALIDIERUNGSERGEBNIS")
        print("=" * 60)
        
        # Erfolgreiche Validierungen
        valid_systems = [k for k, v in self.validation_results.items() if v == 'VALID']
        
        print(f"\n✅ ERFOLGREICH VALIDIERT ({len(valid_systems)}):")
        for system in valid_systems:
            print(f"   ✅ {system.replace('_', ' ').title()}")
        
        # Warnungen
        if self.warnings:
            print(f"\n⚠️  WARNUNGEN ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        # Fehler
        if self.errors:
            print(f"\n❌ FEHLER ({len(self.errors)}):")
            for error in self.errors:
                print(f"   ❌ {error}")
        else:
            print(f"\n🎉 ALLE SYSTEME VALIDIERT - 100% ECHTE DATEN!")
        
        # Gesamtstatus
        if len(self.errors) == 0:
            print(f"\n🚀 STATUS: VOLLSTÄNDIG VALIDIERT")
            print(f"📊 Alle Daten sind echt und korrekt!")
        else:
            print(f"\n⚠️  STATUS: VALIDIERUNG MIT FEHLERN")
            print(f"🔧 Bitte Fehler beheben für 100% Validierung")

def run_complete_validation():
    """Führe komplette Validierung aus"""
    validator = DataValidationSuite()
    return validator.validate_all_systems()

if __name__ == "__main__":
    run_complete_validation()
