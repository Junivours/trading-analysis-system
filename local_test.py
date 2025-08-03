#!/usr/bin/env python3
"""
🚀 LOKALER TEST FÜR DAS ML TRADING SYSTEM
Testet alle Komponenten lokal mit echten Daten
"""

import sys
import os
import time
from datetime import datetime
import json

# Lokale Imports
try:
    from binance_api import fetch_binance_data, test_binance_connection
    from ml_engine import RealMLTradingEngine
    from validation_suite import DataValidationSuite
    from real_data_enforcer import RealDataEnforcer
except ImportError as e:
    print(f"❌ Import Fehler: {e}")
    print("Stelle sicher, dass alle Module im gleichen Verzeichnis sind")
    sys.exit(1)

class LocalTester:
    """Lokaler Tester für das gesamte System"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_success(self, message):
        print(f"✅ {message}")
    
    def print_error(self, message):
        print(f"❌ {message}")
    
    def print_info(self, message):
        print(f"ℹ️  {message}")
    
    def test_binance_connection(self):
        """Test 1: Binance API Verbindung"""
        self.print_header("BINANCE API VERBINDUNG TESTEN")
        
        try:
            # Test Verbindung
            connection_ok = test_binance_connection()
            if connection_ok:
                self.print_success("Binance API Verbindung erfolgreich")
                self.results['binance_connection'] = True
            else:
                self.print_error("Binance API Verbindung fehlgeschlagen")
                self.results['binance_connection'] = False
                return False
            
            # Test Daten abrufen
            self.print_info("Teste Daten abrufen für BTCUSDT...")
            data = fetch_binance_data('BTCUSDT', '1h', 100)
            
            if data and len(data) > 50:
                self.print_success(f"Erfolgreich {len(data)} Datenpunkte abgerufen")
                self.print_info(f"Letzter Preis: ${float(data[-1][4]):,.2f}")
                self.results['binance_data'] = True
                return True
            else:
                self.print_error("Nicht genug Daten erhalten")
                self.results['binance_data'] = False
                return False
                
        except Exception as e:
            self.print_error(f"Binance Test fehlgeschlagen: {e}")
            self.results['binance_connection'] = False
            return False
    
    def test_ml_engine(self):
        """Test 2: ML Engine Training und Vorhersagen"""
        self.print_header("ML ENGINE TESTEN")
        
        try:
            # ML Engine initialisieren
            ml_engine = RealMLTradingEngine()
            self.print_info("ML Engine initialisiert")
            
            # Schnelles Training mit weniger Daten für Test
            self.print_info("Starte ML Training (kann 1-2 Minuten dauern)...")
            training_success = ml_engine.train_models('BTCUSDT', days_back=30)
            
            if training_success:
                self.print_success("ML Training erfolgreich abgeschlossen")
                self.results['ml_training'] = True
            else:
                self.print_error("ML Training fehlgeschlagen")
                self.results['ml_training'] = False
                return False
            
            # Test Vorhersage
            self.print_info("Teste ML Vorhersagen...")
            test_data = fetch_binance_data('BTCUSDT', '1h', 100)
            
            if test_data:
                predictions = ml_engine.predict(test_data)
                
                if predictions and 'ensemble' in predictions:
                    self.print_success("ML Vorhersagen erfolgreich generiert")
                    self.print_info(f"Ensemble Vorhersage: {predictions['ensemble']['prediction']}")
                    self.print_info(f"Konfidenz: {predictions['ensemble']['confidence']:.1f}%")
                    self.results['ml_predictions'] = True
                    return True
                else:
                    self.print_error("Keine gültigen Vorhersagen erhalten")
                    self.results['ml_predictions'] = False
                    return False
            else:
                self.print_error("Keine Testdaten für Vorhersagen")
                return False
                
        except Exception as e:
            self.print_error(f"ML Engine Test fehlgeschlagen: {e}")
            self.results['ml_training'] = False
            return False
    
    def test_validation_suite(self):
        """Test 3: Validation Suite"""
        self.print_header("VALIDATION SUITE TESTEN")
        
        try:
            validator = DataValidationSuite()
            self.print_info("Validation Suite initialisiert")
            
            # Test alle Validierungen
            validation_results = validator.validate_all_systems()
            
            if validation_results:
                self.print_success("Alle Validierungen erfolgreich")
                self.results['validation_suite'] = True
                return True
            else:
                self.print_error("Validierung fehlgeschlagen")
                self.results['validation_suite'] = False
                return False
                
        except Exception as e:
            self.print_error(f"Validation Suite Test fehlgeschlagen: {e}")
            self.results['validation_suite'] = False
            return False
    
    def test_real_data_enforcer(self):
        """Test 4: Real Data Enforcer"""
        self.print_header("REAL DATA ENFORCER TESTEN")
        
        try:
            enforcer = RealDataEnforcer()
            self.print_info("Real Data Enforcer initialisiert")
            
            # Test technische Indikatoren
            indicators = enforcer.get_real_technical_indicators('BTCUSDT', '1h', 100)
            
            if indicators and all(key in indicators for key in ['rsi', 'macd', 'bb_position']):
                self.print_success("Echte technische Indikatoren erhalten")
                self.print_info(f"RSI: {indicators['rsi']:.2f}")
                
                # MACD kann float oder dict sein
                if isinstance(indicators['macd'], dict):
                    self.print_info(f"MACD: {indicators['macd']['macd']:.4f}")
                else:
                    self.print_info(f"MACD: {indicators['macd']:.4f}")
                    
                self.results['real_data_enforcer'] = True
                return True
            else:
                self.print_error("Keine gültigen Indikatoren erhalten")
                self.results['real_data_enforcer'] = False
                return False
                
        except Exception as e:
            self.print_error(f"Real Data Enforcer Test fehlgeschlagen: {e}")
            self.results['real_data_enforcer'] = False
            return False
    
    def test_complete_workflow(self):
        """Test 5: Kompletter Workflow"""
        self.print_header("KOMPLETTER WORKFLOW TEST")
        
        try:
            symbol = 'ETHUSDT'  # Anderes Symbol für Variation
            self.print_info(f"Teste kompletten Workflow für {symbol}")
            
            # 1. Daten abrufen
            data = fetch_binance_data(symbol, '4h', 200)
            if not data:
                self.print_error("Keine Daten erhalten")
                return False
            
            # 2. ML Vorhersage
            ml_engine = RealMLTradingEngine()
            ml_engine.train_models(symbol, days_back=20)  # Schnelles Training
            predictions = ml_engine.predict(data)
            
            # 3. Technische Analyse
            enforcer = RealDataEnforcer()
            indicators = enforcer.get_real_technical_indicators(symbol, '4h', 100)
            
            # 4. Liquidity Analysis  
            liquidity = enforcer.get_real_liquidity_zones(symbol, '4h', 100)
            
            if predictions and indicators and liquidity:
                self.print_success("Kompletter Workflow erfolgreich")
                self.print_info(f"ML Prediction: {predictions.get('ensemble', {}).get('prediction', 'N/A')}")
                self.print_info(f"RSI: {indicators.get('rsi', 'N/A'):.2f}")
                self.print_info(f"Support Levels: {len(liquidity.get('support_levels', []))}")
                self.results['complete_workflow'] = True
                return True
            else:
                self.print_error("Workflow unvollständig")
                self.results['complete_workflow'] = False
                return False
                
        except Exception as e:
            self.print_error(f"Workflow Test fehlgeschlagen: {e}")
            self.results['complete_workflow'] = False
            return False
    
    def run_all_tests(self):
        """Führe alle Tests aus"""
        print(f"🚀 STARTE LOKALE TESTS - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python Version: {sys.version}")
        print(f"Arbeitsverzeichnis: {os.getcwd()}")
        
        # Test Sequenz
        tests = [
            ("Binance Verbindung", self.test_binance_connection),
            ("Validation Suite", self.test_validation_suite),
            ("Real Data Enforcer", self.test_real_data_enforcer),
            ("ML Engine", self.test_ml_engine),
            ("Kompletter Workflow", self.test_complete_workflow)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                self.print_error(f"Test '{test_name}' crashed: {e}")
        
        # Finale Zusammenfassung
        self.print_header("TEST ZUSAMMENFASSUNG")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"⏱️  Testdauer: {duration.total_seconds():.1f} Sekunden")
        print(f"📊 Bestanden: {passed_tests}/{total_tests} Tests")
        print(f"📈 Erfolgsrate: {passed_tests/total_tests*100:.1f}%")
        
        # Detaillierte Ergebnisse
        print(f"\n📋 DETAILLIERTE ERGEBNISSE:")
        for test_name, result in self.results.items():
            status = "✅ BESTANDEN" if result else "❌ FEHLGESCHLAGEN"
            print(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALLE TESTS BESTANDEN - SYSTEM IST BEREIT!")
            return True
        else:
            print(f"\n⚠️  EINIGE TESTS FEHLGESCHLAGEN - PRÜFE DIE LOGS")
            return False

if __name__ == "__main__":
    tester = LocalTester()
    success = tester.run_all_tests()
    
    # Exit Code für Automation
    sys.exit(0 if success else 1)
