# 🛡️ PROFESSIONELLE FEHLERBEHANDLUNG & STATUS-MANAGEMENT
# Robuste Fallback-Systeme und klare Status-Kommunikation

import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

class SystemStatus(Enum):
    """🚦 System-Status Definitionen"""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class DataSource(Enum):
    """📊 Datenquellen Status"""
    LIVE = "live"
    CACHED = "cached"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"

class RobustStatusManager:
    """🛡️ Robustes Status-Management System"""
    
    def __init__(self):
        self.component_status = {
            'binance_api': {'status': SystemStatus.ONLINE, 'last_check': time.time(), 'error_count': 0},
            'jax_ml': {'status': SystemStatus.ONLINE, 'last_check': time.time(), 'error_count': 0},
            'neural_engine': {'status': SystemStatus.ONLINE, 'last_check': time.time(), 'error_count': 0},
            'backtesting': {'status': SystemStatus.ONLINE, 'last_check': time.time(), 'error_count': 0},
            'cache_system': {'status': SystemStatus.ONLINE, 'last_check': time.time(), 'error_count': 0}
        }
        
        self.fallback_data = {}
        self.last_known_values = {}
        self.error_messages = {}
        
        print("🛡️ Robust Status Manager initialisiert")
    
    def update_component_status(self, component: str, status: SystemStatus, error_msg: str = None):
        """🔄 Komponenten-Status aktualisieren"""
        if component not in self.component_status:
            self.component_status[component] = {'status': status, 'last_check': time.time(), 'error_count': 0}
        
        old_status = self.component_status[component]['status']
        
        if status != SystemStatus.ONLINE:
            self.component_status[component]['error_count'] += 1
            if error_msg:
                self.error_messages[component] = error_msg
        else:
            self.component_status[component]['error_count'] = 0
            self.error_messages.pop(component, None)
        
        self.component_status[component]['status'] = status
        self.component_status[component]['last_check'] = time.time()
        
        # 📢 Status-Änderung loggen
        if old_status != status:
            print(f"🚦 {component}: {old_status.value} → {status.value}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """❤️ Gesamt-System-Gesundheit"""
        online_components = sum(1 for comp in self.component_status.values() if comp['status'] == SystemStatus.ONLINE)
        total_components = len(self.component_status)
        
        if online_components == total_components:
            overall_status = SystemStatus.ONLINE
        elif online_components >= total_components * 0.7:
            overall_status = SystemStatus.DEGRADED
        else:
            overall_status = SystemStatus.OFFLINE
        
        return {
            'overall_status': overall_status.value,
            'components': {
                name: {
                    'status': info['status'].value,
                    'last_check': info['last_check'],
                    'error_count': info['error_count'],
                    'uptime_pct': max(0, 100 - (info['error_count'] * 5))
                }
                for name, info in self.component_status.items()
            },
            'active_errors': self.error_messages.copy(),
            'health_score': (online_components / total_components) * 100
        }
    
    def store_fallback_data(self, key: str, data: Any, source: DataSource = DataSource.LIVE):
        """💾 Fallback-Daten speichern"""
        self.fallback_data[key] = {
            'data': data,
            'timestamp': time.time(),
            'source': source
        }
        
        # Letzte bekannte Werte aktualisieren (nur für Live-Daten)
        if source == DataSource.LIVE:
            self.last_known_values[key] = data
    
    def get_reliable_data(self, key: str, fresh_data: Any = None, max_age: int = 300) -> Dict[str, Any]:
        """🔍 Zuverlässige Daten abrufen mit Fallback-Logik"""
        current_time = time.time()
        
        # 1. Frische Daten verfügbar?
        if fresh_data is not None:
            self.store_fallback_data(key, fresh_data, DataSource.LIVE)
            return {
                'data': fresh_data,
                'source': DataSource.LIVE.value,
                'age_seconds': 0,
                'reliability': 100
            }
        
        # 2. Fallback-Daten prüfen
        if key in self.fallback_data:
            fallback_entry = self.fallback_data[key]
            age = current_time - fallback_entry['timestamp']
            
            if age <= max_age:
                reliability = max(50, 100 - (age / max_age * 50))
                return {
                    'data': fallback_entry['data'],
                    'source': DataSource.CACHED.value,
                    'age_seconds': int(age),
                    'reliability': int(reliability)
                }
        
        # 3. Letzte bekannte Werte
        if key in self.last_known_values:
            return {
                'data': self.last_known_values[key],
                'source': DataSource.FALLBACK.value,
                'age_seconds': -1,
                'reliability': 25
            }
        
        # 4. Keine Daten verfügbar
        return {
            'data': None,
            'source': DataSource.UNAVAILABLE.value,
            'age_seconds': -1,
            'reliability': 0
        }
    
    def get_user_friendly_message(self, component: str) -> str:
        """💬 Benutzerfreundliche Status-Nachrichten"""
        if component not in self.component_status:
            return "🔍 Komponente nicht gefunden"
        
        status = self.component_status[component]['status']
        error_count = self.component_status[component]['error_count']
        
        messages = {
            'binance_api': {
                SystemStatus.ONLINE: "🟢 Binance API verbunden",
                SystemStatus.DEGRADED: f"⚠️ Binance API instabil ({error_count} Fehler)",
                SystemStatus.OFFLINE: "🔴 Binance API nicht erreichbar - verwende gespeicherte Daten"
            },
            'jax_ml': {
                SystemStatus.ONLINE: "🧠 ML-System aktiv",
                SystemStatus.DEGRADED: "⚠️ ML-System teilweise verfügbar",
                SystemStatus.OFFLINE: "🔴 ML-Vorhersagen deaktiviert"
            },
            'neural_engine': {
                SystemStatus.ONLINE: "🤖 Neural Engine bereit",
                SystemStatus.DEGRADED: "⚠️ Neural Engine eingeschränkt",
                SystemStatus.OFFLINE: "🔴 LSTM-Prognosen nicht verfügbar"
            },
            'backtesting': {
                SystemStatus.ONLINE: "📊 Backtesting verfügbar",
                SystemStatus.DEGRADED: "⚠️ Backtesting eingeschränkt",
                SystemStatus.OFFLINE: "🔴 Backtesting-Modul fehlt"
            }
        }
        
        return messages.get(component, {}).get(status, f"❓ Status unbekannt: {status.value}")

class AdaptiveWeightManager:
    """⚖️ Dynamische Gewichtung basierend auf verfügbaren Systemen"""
    
    def __init__(self, status_manager: RobustStatusManager):
        self.status_manager = status_manager
        
        # 🎯 Standard-Gewichtungen
        self.base_weights = {
            'fundamental': 40,
            'technical': 40,
            'ml': 20
        }
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """⚖️ Adaptive Gewichtung basierend auf System-Status"""
        health = self.status_manager.get_system_health()
        components = health['components']
        
        # Verfügbare Systeme prüfen
        ml_available = (components.get('jax_ml', {}).get('status') == 'online' and 
                       components.get('neural_engine', {}).get('status') == 'online')
        api_available = components.get('binance_api', {}).get('status') == 'online'
        
        if ml_available and api_available:
            # 🟢 Alle Systeme verfügbar
            return self.base_weights.copy()
        elif not ml_available and api_available:
            # 🟡 ML nicht verfügbar
            return {
                'fundamental': 60,
                'technical': 40,
                'ml': 0
            }
        elif ml_available and not api_available:
            # 🟡 API nicht verfügbar, aber ML verfügbar
            return {
                'fundamental': 20,
                'technical': 30,
                'ml': 50
            }
        else:
            # 🔴 Minimal-Modus
            return {
                'fundamental': 70,
                'technical': 30,
                'ml': 0
            }
    
    def get_weight_explanation(self) -> str:
        """💬 Gewichtungs-Erklärung für UI"""
        weights = self.get_adaptive_weights()
        health = self.status_manager.get_system_health()
        
        if health['overall_status'] == 'online':
            return f"🟢 Vollanalyse: {weights['fundamental']}% Fundamental | {weights['technical']}% Technisch | {weights['ml']}% KI"
        elif health['overall_status'] == 'degraded':
            return f"🟡 Eingeschränkt: {weights['fundamental']}% Fundamental | {weights['technical']}% Technisch | {weights['ml']}% KI"
        else:
            return f"🔴 Fallback-Modus: {weights['fundamental']}% Fundamental | {weights['technical']}% Technisch"

# 🌟 Globale Instanzen
status_manager = RobustStatusManager()
weight_manager = AdaptiveWeightManager(status_manager)

def get_system_dashboard() -> Dict[str, Any]:
    """📊 System-Dashboard für Frontend"""
    health = status_manager.get_system_health()
    weights = weight_manager.get_adaptive_weights()
    
    return {
        'system_health': health,
        'adaptive_weights': weights,
        'weight_explanation': weight_manager.get_weight_explanation(),
        'user_messages': [
            status_manager.get_user_friendly_message(comp) 
            for comp in status_manager.component_status.keys()
        ]
    }

print("✅ Robuste Fehlerbehandlung und Status-Management bereit")
