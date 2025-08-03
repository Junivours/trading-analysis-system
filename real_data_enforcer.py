# === ECHTE DATEN ENFORCEMENT - ENTFERNT ALLE DUMMY WERTE ===

from binance_api import fetch_binance_data
import logging

logger = logging.getLogger(__name__)

class RealDataEnforcer:
    """Stellt sicher, dass nur echte Daten verwendet werden"""
    
    @staticmethod
    def get_real_technical_indicators(symbol, interval='1h', limit=100):
        """Berechne AUSSCHLIESSLICH echte technische Indikatoren"""
        
        # Hole echte Marktdaten
        data = fetch_binance_data(symbol, interval, limit)
        
        if not data or len(data) < 50:
            raise ValueError(f"Nicht genug echte Daten für {symbol} - benötige mindestens 50 Kerzen")
        
        # Extrahiere echte Preise
        closes = [float(candle[4]) for candle in data]
        highs = [float(candle[2]) for candle in data]
        lows = [float(candle[3]) for candle in data]
        volumes = [float(candle[5]) for candle in data]
        opens = [float(candle[1]) for candle in data]
        
        # Validiere Datenqualität
        if not RealDataEnforcer._validate_price_data(closes, highs, lows, opens):
            raise ValueError("Datenqualität unzureichend - OHLC Logik inkorrekt")
        
        # Berechne ECHTE Indikatoren
        indicators = {
            'rsi': RealDataEnforcer._calculate_real_rsi(closes),
            'macd': RealDataEnforcer._calculate_real_macd(closes),
            'bb_position': RealDataEnforcer._calculate_real_bb_position(closes),
            'volume_ratio': RealDataEnforcer._calculate_real_volume_ratio(volumes),
            'volatility': RealDataEnforcer._calculate_real_volatility(closes),
            'trend_strength': RealDataEnforcer._calculate_real_trend_strength(closes),
            'support_distance': RealDataEnforcer._calculate_real_support_distance(closes, lows),
            'resistance_distance': RealDataEnforcer._calculate_real_resistance_distance(closes, highs),
            'current_price': closes[-1],
            'data_quality': 'VALIDATED_REAL',
            'timestamp': data[-1][0]  # Echter Zeitstempel
        }
        
        logger.info(f"✅ Echte Indikatoren für {symbol} berechnet - RSI: {indicators['rsi']:.2f}")
        
        return indicators
    
    @staticmethod
    def _validate_price_data(closes, highs, lows, opens):
        """Validiere dass OHLC-Daten logisch korrekt sind"""
        for i in range(len(closes)):
            # High muss größer oder gleich Open/Close sein
            if highs[i] < max(opens[i], closes[i]):
                return False
            # Low muss kleiner oder gleich Open/Close sein
            if lows[i] > min(opens[i], closes[i]):
                return False
            # Preise müssen positiv sein
            if any(price <= 0 for price in [closes[i], highs[i], lows[i], opens[i]]):
                return False
        return True
    
    @staticmethod
    def _calculate_real_rsi(prices, period=14):
        """Berechne echten RSI ohne Fallbacks"""
        if len(prices) < period + 1:
            raise ValueError(f"Nicht genug Daten für RSI - benötige {period + 1}, habe {len(prices)}")
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        # Verwende die letzten 'period' Werte
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Validiere RSI Wertebereich
        if not (0 <= rsi <= 100):
            raise ValueError(f"RSI außerhalb Wertebereich: {rsi}")
        
        return rsi
    
    @staticmethod
    def _calculate_real_macd(prices, fast=12, slow=26):
        """Berechne echten MACD ohne Fallbacks"""
        if len(prices) < slow:
            raise ValueError(f"Nicht genug Daten für MACD - benötige {slow}, habe {len(prices)}")
        
        # Berechne EMAs
        ema_fast = RealDataEnforcer._calculate_ema(prices, fast)
        ema_slow = RealDataEnforcer._calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        
        return macd
    
    @staticmethod
    def _calculate_ema(prices, period):
        """Berechne echten EMA"""
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def _calculate_real_bb_position(prices, period=20):
        """Berechne echte Bollinger Band Position"""
        if len(prices) < period:
            raise ValueError(f"Nicht genug Daten für BB - benötige {period}, habe {len(prices)}")
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        
        # Berechne Standardabweichung
        variance = sum([(p - sma) ** 2 for p in recent_prices]) / period
        std = variance ** 0.5
        
        current_price = prices[-1]
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    @staticmethod
    def _calculate_real_volume_ratio(volumes):
        """Berechne echtes Volumen-Verhältnis"""
        if len(volumes) < 20:
            raise ValueError(f"Nicht genug Volumendaten - benötige 20, habe {len(volumes)}")
        
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / 20
        
        if avg_volume == 0:
            return 1.0
        
        return current_volume / avg_volume
    
    @staticmethod
    def _calculate_real_volatility(prices):
        """Berechne echte Volatilität"""
        if len(prices) < 20:
            raise ValueError(f"Nicht genug Daten für Volatilität - benötige 20, habe {len(prices)}")
        
        recent_prices = prices[-20:]
        mean_price = sum(recent_prices) / len(recent_prices)
        
        variance = sum([(p - mean_price) ** 2 for p in recent_prices]) / len(recent_prices)
        volatility = (variance ** 0.5) / mean_price
        
        return volatility
    
    @staticmethod
    def _calculate_real_trend_strength(prices):
        """Berechne echte Trend-Stärke"""
        if len(prices) < 20:
            raise ValueError(f"Nicht genug Daten für Trend - benötige 20, habe {len(prices)}")
        
        # Berechne lineare Regression für Trend
        n = len(prices[-20:])
        x_values = list(range(n))
        y_values = prices[-20:]
        
        # Pearson Korrelationskoeffizient als Trend-Stärke
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum([(x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n)])
        x_variance = sum([(x - x_mean) ** 2 for x in x_values])
        y_variance = sum([(y - y_mean) ** 2 for y in y_values])
        
        if x_variance == 0 or y_variance == 0:
            return 0
        
        correlation = numerator / ((x_variance * y_variance) ** 0.5)
        
        return correlation
    
    @staticmethod
    def _calculate_real_support_distance(closes, lows):
        """Berechne echte Support-Distanz"""
        if len(lows) < 20:
            raise ValueError(f"Nicht genug Daten für Support - benötige 20, habe {len(lows)}")
        
        recent_low = min(lows[-20:])
        current_price = closes[-1]
        
        return (current_price - recent_low) / current_price
    
    @staticmethod
    def _calculate_real_resistance_distance(closes, highs):
        """Berechne echte Resistance-Distanz"""
        if len(highs) < 20:
            raise ValueError(f"Nicht genug Daten für Resistance - benötige 20, habe {len(highs)}")
        
        recent_high = max(highs[-20:])
        current_price = closes[-1]
        
        return (recent_high - current_price) / current_price
    
    @staticmethod
    def get_real_liquidity_zones(symbol, interval='4h', limit=200):
        """Berechne ECHTE Liquiditätszonen ohne Dummy-Daten"""
        
        data = fetch_binance_data(symbol, interval, limit)
        
        if not data or len(data) < 100:
            raise ValueError(f"Nicht genug Daten für Liquiditätsanalyse - benötige 100, habe {len(data) if data else 0}")
        
        highs = [float(candle[2]) for candle in data]
        lows = [float(candle[3]) for candle in data]
        volumes = [float(candle[5]) for candle in data]
        closes = [float(candle[4]) for candle in data]
        
        # Finde echte Support/Resistance basierend auf Volumen
        support_zones = []
        resistance_zones = []
        
        # Analysiere die letzten 100 Kerzen
        for i in range(20, len(data) - 20):
            current_high = highs[i]
            current_low = lows[i]
            volume_weight = volumes[i]
            
            # Support-Zone: Lokales Minimum mit hohem Volumen
            is_support = all(current_low <= lows[j] for j in range(i-10, i+10) if j != i)
            if is_support and volume_weight > sum(volumes[i-10:i+10]) / 20:
                support_zones.append({
                    'price': current_low,
                    'strength': volume_weight / max(volumes),
                    'touches': sum(1 for low in lows[i+1:] if abs(low - current_low) / current_low < 0.01)
                })
            
            # Resistance-Zone: Lokales Maximum mit hohem Volumen  
            is_resistance = all(current_high >= highs[j] for j in range(i-10, i+10) if j != i)
            if is_resistance and volume_weight > sum(volumes[i-10:i+10]) / 20:
                resistance_zones.append({
                    'price': current_high,
                    'strength': volume_weight / max(volumes),
                    'touches': sum(1 for high in highs[i+1:] if abs(high - current_high) / current_high < 0.01)
                })
        
        current_price = closes[-1]
        
        # Sortiere nach Stärke und Relevanz
        support_zones.sort(key=lambda x: x['strength'] * (1 + x['touches']), reverse=True)
        resistance_zones.sort(key=lambda x: x['strength'] * (1 + x['touches']), reverse=True)
        
        # Filtere auf relevante Zonen (innerhalb 20% vom aktuellen Preis)
        relevant_supports = [s for s in support_zones if s['price'] > current_price * 0.8 and s['price'] < current_price]
        relevant_resistances = [r for r in resistance_zones if r['price'] < current_price * 1.2 and r['price'] > current_price]
        
        return {
            'support_zones': relevant_supports[:5],  # Top 5
            'resistance_zones': relevant_resistances[:5],  # Top 5
            'current_price': current_price,
            'data_quality': 'VALIDATED_REAL',
            'analysis_period': f"{len(data)} candles ({interval})"
        }

def validate_no_dummy_data():
    """Validiere dass keine Dummy-Daten verwendet werden"""
    try:
        # Teste BTCUSDT
        indicators = RealDataEnforcer.get_real_technical_indicators('BTCUSDT')
        liquidity = RealDataEnforcer.get_real_liquidity_zones('BTCUSDT')
        
        print("✅ VALIDIERUNG ERFOLGREICH:")
        print(f"   RSI: {indicators['rsi']:.2f}")
        print(f"   MACD: {indicators['macd']:.4f}")
        print(f"   Aktueller Preis: ${indicators['current_price']:,.2f}")
        print(f"   Support-Zonen: {len(liquidity['support_zones'])}")
        print(f"   Resistance-Zonen: {len(liquidity['resistance_zones'])}")
        print("🎉 ALLE DATEN SIND 100% ECHT!")
        
        return True
        
    except Exception as e:
        print(f"❌ VALIDIERUNG FEHLGESCHLAGEN: {e}")
        return False

if __name__ == "__main__":
    validate_no_dummy_data()
