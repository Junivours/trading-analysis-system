import requests

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    """Fetch historical market data from Binance API."""
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "timestamp": entry[0],
                "open": float(entry[1]),
                "high": float(entry[2]),
                "low": float(entry[3]),
                "close": float(entry[4]),
                "volume": float(entry[5])
            }
            for entry in data
        ]
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return []