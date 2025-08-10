# 🚀 PROFESSIONELLER CACHE MANAGER
# Optimierte API-Zugriffe mit intelligenter Cache-Schicht

import time
import json
from typing import Dict, Any, Optional
import threading
from datetime import datetime, timedelta

class SmartCacheManager:
    """🧠 Intelligenter Cache Manager für optimierte API-Zugriffe"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_lock = threading.Lock()
        
        # 🎯 Cache-Konfiguration
        self.cache_durations = {
            'price_data': 15,        # Live-Preise: 15 Sekunden
            'indicators': 60,        # Technische Indikatoren: 1 Minute
            'market_data': 30,       # Marktdaten: 30 Sekunden
            'ml_predictions': 120,   # ML-Vorhersagen: 2 Minuten
            'backtest_results': 300, # Backtest: 5 Minuten
            'kline_data': 45        # Kerzendaten: 45 Sekunden
        }
        
        print("🧠 Smart Cache Manager initialisiert")
        print(f"📊 Cache-Strategien: {len(self.cache_durations)} Kategorien")
    
    def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """📥 Daten aus Cache abrufen"""
        with self.cache_lock:
            full_key = f"{category}:{key}"
            
            if full_key not in self.cache:
                return None
            
            # ⏰ Cache-Alter prüfen
            cache_time = self.cache_timestamps.get(full_key, 0)
            max_age = self.cache_durations.get(category, 60)
            
            if time.time() - cache_time > max_age:
                # 🗑️ Veraltete Daten entfernen
                del self.cache[full_key]
                del self.cache_timestamps[full_key]
                return None
            
            return self.cache[full_key]
    
    def set(self, key: str, value: Any, category: str = 'default') -> None:
        """💾 Daten in Cache speichern"""
        with self.cache_lock:
            full_key = f"{category}:{key}"
            self.cache[full_key] = value
            self.cache_timestamps[full_key] = time.time()
    
    def invalidate(self, key: str = None, category: str = None) -> None:
        """🗑️ Cache invalidieren"""
        with self.cache_lock:
            if key and category:
                full_key = f"{category}:{key}"
                self.cache.pop(full_key, None)
                self.cache_timestamps.pop(full_key, None)
            elif category:
                # Gesamte Kategorie löschen
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{category}:")]
                for k in keys_to_remove:
                    self.cache.pop(k, None)
                    self.cache_timestamps.pop(k, None)
            else:
                # Alles löschen
                self.cache.clear()
                self.cache_timestamps.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """📊 Cache-Statistiken"""
        with self.cache_lock:
            stats = {
                'total_entries': len(self.cache),
                'categories': {},
                'oldest_entry': None,
                'cache_hit_potential': 0
            }
            
            current_time = time.time()
            category_counts = {}
            
            for full_key, timestamp in self.cache_timestamps.items():
                category = full_key.split(':', 1)[0]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                age = current_time - timestamp
                if stats['oldest_entry'] is None or age > stats['oldest_entry']:
                    stats['oldest_entry'] = age
            
            stats['categories'] = category_counts
            return stats
    
    def cleanup_expired(self) -> int:
        """🧹 Abgelaufene Einträge bereinigen"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = []
            
            for full_key, timestamp in self.cache_timestamps.items():
                category = full_key.split(':', 1)[0]
                max_age = self.cache_durations.get(category, 60)
                
                if current_time - timestamp > max_age:
                    expired_keys.append(full_key)
            
            for key in expired_keys:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            return len(expired_keys)

class APIOptimizer:
    """🚀 API-Zugriffe optimieren"""
    
    def __init__(self, cache_manager: SmartCacheManager):
        self.cache = cache_manager
        self.batch_requests = {}
        self.priority_queue = []
        
    def should_fetch_fresh(self, key: str, category: str) -> bool:
        """🔍 Prüfen ob frische Daten benötigt werden"""
        cached_data = self.cache.get(key, category)
        return cached_data is None
    
    def batch_symbol_requests(self, symbols: list) -> str:
        """📦 Mehrere Symbole in einem Request bündeln"""
        # Für Binance API: Mehrere Symbole als JSON Array
        return json.dumps(symbols)
    
    def prioritize_requests(self, requests: list) -> list:
        """🎯 Requests nach Priorität sortieren"""
        priority_order = {
            'price_data': 1,      # Höchste Priorität
            'market_data': 2,
            'indicators': 3,
            'ml_predictions': 4,
            'backtest_results': 5  # Niedrigste Priorität
        }
        
        return sorted(requests, key=lambda x: priority_order.get(x.get('category', 'default'), 999))

# 🌟 Globale Instanzen
cache_manager = SmartCacheManager()
api_optimizer = APIOptimizer(cache_manager)

def get_cache_status() -> Dict[str, Any]:
    """📊 Cache-Status für Frontend"""
    stats = cache_manager.get_cache_stats()
    return {
        'cache_active': True,
        'total_entries': stats['total_entries'],
        'categories': stats['categories'],
        'last_cleanup': datetime.now().isoformat()
    }

print("✅ Cache Manager und API Optimizer bereit")
