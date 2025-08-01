"""
Real Binance API Integration - Echte Marktdaten
"""
import requests
import hashlib
import hmac
import time
import json
import logging
from urllib.parse import urlencode

logger = logging.getLogger('BinanceAPI')

class BinanceAPI:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
    
    def _generate_signature(self, params):
        """Generate signature for authenticated requests"""
        query_string = urlencode(params)
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, params=None, signed=False):
        """Make API request with error handling"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_klines(self, symbol='BTCUSDT', interval='1h', limit=100):
        """Get real kline/candlestick data"""
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)  # Binance limit
        }
        
        data = self._make_request('/klines', params)
        if not data:
            return []
        
        # Convert to our format
        formatted_data = []
        for kline in data:
            formatted_data.append({
                'timestamp': kline[0],
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': kline[6],
                'quote_volume': float(kline[7]),
                'trades': kline[8]
            })
        
        return formatted_data
    
    def get_24hr_ticker(self, symbol=None):
        """Get 24hr ticker statistics"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        return self._make_request('/ticker/24hr', params)
    
    def get_current_price(self, symbol='BTCUSDT'):
        """Get current price for symbol"""
        params = {'symbol': symbol.upper()}
        data = self._make_request('/ticker/price', params)
        
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_order_book(self, symbol='BTCUSDT', limit=100):
        """Get order book depth"""
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 5000)
        }
        
        return self._make_request('/depth', params)
    
    def get_recent_trades(self, symbol='BTCUSDT', limit=100):
        """Get recent trades"""
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        
        return self._make_request('/trades', params)
    
    def get_account_info(self):
        """Get account information (authenticated)"""
        return self._make_request('/account', signed=True)
    
    def test_connectivity(self):
        """Test API connectivity"""
        try:
            response = self._make_request('/ping')
            if response == {}:
                logger.info("✅ Binance API connection successful")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ API connectivity test failed: {e}")
            return False
    
    def get_server_time(self):
        """Get server time"""
        data = self._make_request('/time')
        if data and 'serverTime' in data:
            return data['serverTime']
        return None
    
    def get_exchange_info(self, symbol=None):
        """Get exchange trading rules and symbol information"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        return self._make_request('/exchangeInfo', params)

class RealMarketDataFetcher:
    """Replacement for mock data fetchers"""
    
    def __init__(self, api_key, secret_key):
        self.api = BinanceAPI(api_key, secret_key)
        self.cache = {}
        self.cache_duration = 30  # seconds
    
    def fetch_real_market_data(self, symbol='BTCUSDT', interval='1h', limit=200):
        """Fetch real market data instead of mock data"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_duration:
                return cached_data
        
        # Fetch fresh data
        logger.info(f"🔄 Fetching real market data for {symbol}")
        data = self.api.get_klines(symbol, interval, limit)
        
        if data:
            self.cache[cache_key] = (data, current_time)
            logger.info(f"✅ Retrieved {len(data)} real candles for {symbol}")
            return data
        else:
            logger.warning(f"⚠️ Failed to fetch data for {symbol}")
            return []
    
    def get_real_market_overview(self, symbols=None):
        """Get real market overview data"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
        overview = {}
        
        for symbol in symbols:
            try:
                ticker_data = self.api.get_24hr_ticker(symbol)
                if ticker_data:
                    overview[symbol] = {
                        'price': float(ticker_data.get('lastPrice', 0)),
                        'change_24h': float(ticker_data.get('priceChangePercent', 0)),
                        'volume_24h': float(ticker_data.get('volume', 0)),
                        'high_24h': float(ticker_data.get('highPrice', 0)),
                        'low_24h': float(ticker_data.get('lowPrice', 0))
                    }
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        return overview
    
    def get_real_volume_profile(self, symbol='BTCUSDT', limit=100):
        """Get real volume profile data"""
        trades = self.api.get_recent_trades(symbol, limit)
        if not trades:
            return {}
        
        volume_profile = {}
        for trade in trades:
            price = float(trade['price'])
            qty = float(trade['qty'])
            price_level = round(price, 2)
            
            if price_level not in volume_profile:
                volume_profile[price_level] = 0
            volume_profile[price_level] += qty
        
        return volume_profile
