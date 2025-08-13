import time, requests

class BinanceClient:
    BASE_URL = "https://api.binance.com/api/v3"
    _cache = {'ticker': {}, 'price': {}, 'klines': {}}
    TICKER_TTL = 10; PRICE_TTL = 3; KLINES_TTL = 45
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
            resp=requests.get(f"{BinanceClient.BASE_URL}/exchangeInfo",timeout=10); data=resp.json(); symbols=[]; q=query.upper()
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
            r=requests.get(f"{BinanceClient.BASE_URL}/ticker/24hr",params={'symbol':symbol},timeout=10); data=r.json(); data['_cache']='MISS'
            BinanceClient._cache['ticker'][symbol]=(now,data); return data
        except Exception as e: print(f"Error getting ticker data: {e}"); return {}
    @staticmethod
    def get_current_price(symbol):
        try:
            now=time.time(); cached=BinanceClient._cache['price'].get(symbol)
            if cached and now-cached[0] < BinanceClient.PRICE_TTL: return cached[1]
            r=requests.get(f"{BinanceClient.BASE_URL}/ticker/price",params={'symbol':symbol},timeout=10); data=r.json(); price=float(data['price'])
            BinanceClient._cache['price'][symbol]=(now,price); return price
        except Exception as e: print(f"Error getting current price: {e}"); return 0
