# === BINANCE API MODULE ===
import requests
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def fetch_binance_data(symbol, interval='1h', limit=500):
    """
    Fetch OHLC data from Binance API
    Returns: List of OHLC data or None on error
    """
    try:
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max is 1000
        }
        
        logger.info(f"📡 Fetching Binance data: {symbol} {interval} (limit: {limit})")
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Fetched {len(data)} candles for {symbol}")
            return data
        else:
            logger.error(f"❌ Binance API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Binance API exception: {e}")
        return None

def fetch_24hr_ticker(symbol):
    """
    Fetch 24hr ticker data from Binance
    Returns: Dict with ticker data or None on error
    """
    try:
        url = 'https://api.binance.com/api/v3/ticker/24hr'
        params = {'symbol': symbol}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Normalize the data format
            normalized_data = {
                'symbol': data.get('symbol'),
                'last_price': float(data.get('lastPrice', 0)),
                'price_change_percent': float(data.get('priceChangePercent', 0)),
                'volume': float(data.get('volume', 0)),
                'high_price': float(data.get('highPrice', 0)),
                'low_price': float(data.get('lowPrice', 0)),
                'open_price': float(data.get('openPrice', 0)),
                'prev_close_price': float(data.get('prevClosePrice', 0))
            }
            
            logger.info(f"✅ Fetched 24hr ticker for {symbol}: ${normalized_data['last_price']}")
            return normalized_data
        else:
            logger.error(f"❌ Ticker API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Ticker API exception: {e}")
        return None

def get_exchange_info(symbol=None):
    """
    Fetch exchange info from Binance
    """
    try:
        url = 'https://api.binance.com/api/v3/exchangeInfo'
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Fetched exchange info")
            return data
        else:
            logger.error(f"❌ Exchange info error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Exchange info exception: {e}")
        return None

def test_binance_connection():
    """
    Test Binance API connection
    Returns: True if connection successful, False otherwise
    """
    try:
        url = 'https://api.binance.com/api/v3/ping'
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            logger.info("✅ Binance API connection successful")
            return True
        else:
            logger.error(f"❌ Binance ping failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Binance connection test failed: {e}")
        return False
