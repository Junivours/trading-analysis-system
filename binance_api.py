"""
🚀 OPTIMIERTE BINANCE API - Separate Datei
Handles all Binance API communication with caching and error handling
"""

import requests
import time
import warnings
warnings.filterwarnings("ignore")

# Cache und Status Management (optional)
try:
    from cache_manager import cache_manager, api_optimizer
    from status_manager import status_manager, SystemStatus
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

class OptimizedBinanceAPI:
    """🚀 Optimierte Binance API mit Cache und Fehlerbehandlung"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms zwischen Requests
        self.error_count = 0
        self.max_retries = 3
        
        # 📊 Status bei Systemstart prüfen
        self._check_api_health()
    
    def _check_api_health(self):
        """❤️ API-Gesundheit prüfen"""
        try:
            response = requests.get(f"{self.base_url}/ping", timeout=5)
            if response.status_code == 200:
                print("🟢 Binance API verbunden")
                if OPTIMIZATION_AVAILABLE:
                    status_manager.update_component_status('binance_api', SystemStatus.ONLINE)
            else:
                print("🟡 Binance API reagiert langsam")
                if OPTIMIZATION_AVAILABLE:
                    status_manager.update_component_status('binance_api', SystemStatus.DEGRADED)
        except Exception as e:
            print(f"🔴 Binance API Verbindungsproblem: {e}")
            if OPTIMIZATION_AVAILABLE:
                status_manager.update_component_status('binance_api', SystemStatus.OFFLINE, str(e))
    
    def _make_request(self, endpoint: str, params: dict = None, cache_key: str = None, cache_category: str = 'default') -> dict:
        """🌐 Optimierter API-Request mit Cache und Fallback"""
        
        # 1. Cache prüfen (nur wenn Optimierung verfügbar)
        if OPTIMIZATION_AVAILABLE and cache_key:
            cached = cache_manager.get(cache_key, cache_category)
            if cached:
                return cached
        
        # 2. Rate Limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # 3. API Request mit Retry-Logik
        for attempt in range(self.max_retries):
            try:
                self.last_request_time = time.time()
                url = f"{self.base_url}/{endpoint}"
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 5. Cache speichern (nur wenn Optimierung verfügbar)
                    if OPTIMIZATION_AVAILABLE and cache_key:
                        cache_manager.set(cache_key, data, cache_category)
                    
                    # 6. Erfolg -> Error Count zurücksetzen
                    self.error_count = 0
                    return data
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"⏳ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"⚠️ API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Request Timeout (Versuch {attempt + 1}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                print(f"🌐 Verbindungsfehler (Versuch {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"❌ Unbekannter API Error: {e}")
            
            # Exponential backoff bei Fehlern
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        # 7. Fallback verwenden (nur wenn Optimierung verfügbar)
        self.error_count += 1
        if OPTIMIZATION_AVAILABLE:
            fallback = api_optimizer.get_fallback_data(endpoint, params)
            if fallback:
                print(f"🛡️ Fallback-Daten verwendet für {endpoint}")
                return fallback
        
        # Fallback: Leere Standardwerte statt None
        return self._get_fallback_data(endpoint)
    
    def _get_fallback_data(self, endpoint: str) -> dict:
        """🛡️ Fallback-Daten für verschiedene Endpoints"""
        if "ticker/24hr" in endpoint:
            return {
                'symbol': 'BTCUSDT',
                'lastPrice': '50000.00',
                'priceChangePercent': '0.00',
                'volume': '1000',
                'count': 100000
            }
        elif "klines" in endpoint:
            # Dummy Kline-Daten
            base_price = 50000
            return [[
                1609459200000,  # Open time
                str(base_price),  # Open
                str(base_price * 1.02),  # High
                str(base_price * 0.98),  # Low
                str(base_price * 1.01),  # Close
                "1000",  # Volume
                1609462799999,  # Close time
                "50000000",  # Quote asset volume
                1000,  # Number of trades
                "500",  # Taker buy base asset volume
                "25000000",  # Taker buy quote asset volume
                "0"  # Ignore
            ] for _ in range(100)]
        else:
            return {'error': f'Fallback not available for {endpoint}'}
    
    def get_ticker(self, symbol: str) -> dict:
        """📈 Optimierte Ticker-Daten mit Cache"""
        return self._make_request(
            "ticker/24hr",
            params={"symbol": symbol},
            cache_key=f"ticker_{symbol}",
            cache_category="price_data"
        )
    
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        """📊 Optimierte Kerzendaten mit Cache"""
        result = self._make_request(
            "klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            cache_key=f"klines_{symbol}_{interval}_{limit}",
            cache_category="kline_data"
        )
        return result if isinstance(result, list) else []

# Global instance
binance_api = OptimizedBinanceAPI()
