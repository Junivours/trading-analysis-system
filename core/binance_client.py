import os, time, requests

class BinanceClient:
    BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com/api/v3")
    _cache = {'ticker': {}, 'price': {}, 'klines': {}, 'exchangeInfo': {}}
    TICKER_TTL = 10; PRICE_TTL = 3; KLINES_TTL = 45
    EXINFO_TTL = 60*60  # 1 hour
    @staticmethod
    def clear_symbol_cache(symbol: str):
        try:
            symbol=symbol.upper();
            BinanceClient._cache['ticker'].pop(symbol, None)
            BinanceClient._cache['price'].pop(symbol, None)
            to_del=[k for k in BinanceClient._cache['klines'] if k[0]==symbol]
            for k in to_del: BinanceClient._cache['klines'].pop(k,None)
        except Exception as e: print(f"Cache clear error: {e}")
    @staticmethod
    def search_symbols(query):
        try:
            headers = {}
            if os.getenv('BINANCE_API_KEY'):
                headers['X-MBX-APIKEY'] = os.getenv('BINANCE_API_KEY')
            resp=requests.get(f"{BinanceClient.BASE_URL}/exchangeInfo",timeout=10, headers=headers); data=resp.json(); symbols=[]; q=query.upper()
            for s in data.get('symbols',[]):
                sym=s['symbol'];
                if s.get('status')=='TRADING':
                    score=0
                    if q in sym: score+=10
                    if sym.endswith('USDT'): score+=5
                    if sym.startswith(q): score+=15
                    if sym==q+'USDT': score+=20
                    if score>0: symbols.append({'symbol':sym,'baseAsset':s['baseAsset'],'quoteAsset':s['quoteAsset'],'score':score})
            return sorted(symbols,key=lambda x:x['score'],reverse=True)[:10]
        except Exception as e: print(f"Error searching symbols: {e}"); return []
    @staticmethod
    def get_ticker_data(symbol):
        try:
            now=time.time(); cached=BinanceClient._cache['ticker'].get(symbol)
            if cached and now-cached[0] < BinanceClient.TICKER_TTL:
                data=cached[1]; data['_cache']='HIT'; return data
            headers = {}
            if os.getenv('BINANCE_API_KEY'):
                headers['X-MBX-APIKEY'] = os.getenv('BINANCE_API_KEY')
            r=requests.get(f"{BinanceClient.BASE_URL}/ticker/24hr",params={'symbol':symbol},timeout=10, headers=headers); data=r.json(); data['_cache']='MISS'
            BinanceClient._cache['ticker'][symbol]=(now,data); return data
        except Exception as e: print(f"Error getting ticker data: {e}"); return {}
    @staticmethod
    def get_current_price(symbol):
        try:
            now=time.time(); cached=BinanceClient._cache['price'].get(symbol)
            if cached and now-cached[0] < BinanceClient.PRICE_TTL: return cached[1]
            headers = {}
            if os.getenv('BINANCE_API_KEY'):
                headers['X-MBX-APIKEY'] = os.getenv('BINANCE_API_KEY')
            r=requests.get(f"{BinanceClient.BASE_URL}/ticker/price",params={'symbol':symbol},timeout=10, headers=headers); data=r.json(); price=float(data['price'])
            BinanceClient._cache['price'][symbol]=(now,price); return price
        except Exception as e: print(f"Error getting current price: {e}"); return 0

    @staticmethod
    def get_symbol_filters(symbol: str):
        """Return important exchange filters and precisions for a given symbol.
        Keys: tickSize, stepSize, minNotional, minQty, maxQty, baseAssetPrecision, quoteAssetPrecision
        """
        try:
            symbol = symbol.upper()
            now = time.time()
            cache = BinanceClient._cache['exchangeInfo'].get(symbol)
            if cache and now - cache[0] < BinanceClient.EXINFO_TTL:
                return cache[1]
            # Fetch full exchangeInfo once, then find symbol
            headers = {}
            if os.getenv('BINANCE_API_KEY'):
                headers['X-MBX-APIKEY'] = os.getenv('BINANCE_API_KEY')
            resp = requests.get(f"{BinanceClient.BASE_URL}/exchangeInfo", timeout=15, headers=headers)
            data = resp.json()
            filters = {}
            for s in data.get('symbols', []):
                if s.get('symbol') == symbol:
                    filt_map = {f.get('filterType'): f for f in s.get('filters', [])}
                    price_filter = filt_map.get('PRICE_FILTER', {})
                    lot_filter = filt_map.get('LOT_SIZE', {})
                    min_notional = filt_map.get('MIN_NOTIONAL', {})
                    filters = {
                        'symbol': symbol,
                        'tickSize': float(price_filter.get('tickSize', '0.00000001')) if price_filter else 0.0,
                        'minPrice': float(price_filter.get('minPrice', '0')) if price_filter else 0.0,
                        'maxPrice': float(price_filter.get('maxPrice', '0')) if price_filter else 0.0,
                        'stepSize': float(lot_filter.get('stepSize', '0.00000001')) if lot_filter else 0.0,
                        'minQty': float(lot_filter.get('minQty', '0')) if lot_filter else 0.0,
                        'maxQty': float(lot_filter.get('maxQty', '0')) if lot_filter else 0.0,
                        'minNotional': float(min_notional.get('minNotional', '0')) if min_notional else 0.0,
                        'baseAssetPrecision': s.get('baseAssetPrecision', 8),
                        'quoteAssetPrecision': s.get('quoteAssetPrecision', 8)
                    }
                    break
            BinanceClient._cache['exchangeInfo'][symbol] = (now, filters)
            return filters
        except Exception as e:
            print(f"Error getting symbol filters: {e}")
            return {}
