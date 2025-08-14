# 🚀 MEXC Trading Bot Konfiguration

## 1. Umgebungsvariablen setzen

Erstelle eine `.env` Datei oder setze die Variablen in deinem System:

```bash
# MEXC API Credentials (vom MEXC Account Dashboard)
MEXC_API_KEY=dein_mexc_api_key
MEXC_API_SECRET=dein_mexc_api_secret

# Optional: Binance Keys (falls du auch Binance nutzen möchtest)
BINANCE_API_KEY=dein_binance_api_key  
BINANCE_API_SECRET=dein_binance_api_secret
```

## 2. MEXC API Keys erstellen

1. Gehe zu [MEXC Account](https://www.mexc.com/account/api)
2. Erstelle einen neuen API Key
3. **Wichtig**: Aktiviere nur die nötigen Permissions:
   - ✅ **Spot Trading** (für Spot Orders)
   - ✅ **Futures Trading** (für Futures Orders) 
   - ✅ **Read** (für Account Info)
   - ❌ **Withdraw** (NICHT aktivieren für Sicherheit!)

## 3. Bot Konfiguration

### Paper Trading Test (Empfohlen zuerst):
```bash
curl -X POST http://localhost:5000/api/bot/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "exchange": "mexc",
    "interval": "1h", 
    "paper": true,
    "equity": 10000,
    "risk_pct": 0.5
  }'
```

### Live Trading (nur wenn API Keys gesetzt):
```bash
curl -X POST http://localhost:5000/api/bot/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SOLUSDT",
    "exchange": "mexc",
    "interval": "1h",
    "paper": false,
    "equity": 1000,
    "risk_pct": 0.5,
    "min_probability": 60,
    "min_rr": 1.5
  }'
```

## 4. Parameter Erklärung

| Parameter | Beschreibung | Standard | Empfehlung |
|-----------|-------------|----------|------------|
| `symbol` | Trading Pair (BTCUSDT, ETHUSDT, etc.) | BTCUSDT | Volatilitätsabhängig |
| `exchange` | "mexc" oder "binance" | binance | mexc |
| `interval` | Timeframe (1h, 4h, 1d) | 1h | 1h für aktive Signale |
| `paper` | Paper Trading (true/false) | true | true zum Testen |
| `equity` | Gesamtkapital in USDT | 10000 | Dein echtes Kapital |
| `risk_pct` | Risiko pro Trade in % | 0.5% | 0.3-1% je nach Risikotoleranz |
| `min_probability` | Min. Erfolgswahrscheinlichkeit | 54% | 55-65% für konservativ |
| `min_rr` | Min. Risk/Reward Ratio | 1.2 | 1.5+ für bessere Trades |

## 5. Sicherheitshinweise

⚠️ **Wichtig für Live Trading:**
- Starte IMMER mit Paper Trading (`"paper": true`)
- Verwende kleine Beträge zum Testen
- Setze niemals `risk_pct` über 1-2%
- Überwache die ersten Trades manuell
- API Keys nur mit nötigen Rechten (keine Withdraw!)

## 6. MEXC Besonderheiten

### Symbol Format:
- MEXC: `BTCUSDT`, `ETHUSDT` (wie Binance)
- Keine Unterschiede bei Standard-Pairs

### Futures vs Spot:
```bash
# Futures Trading (Standard)
{
  "exchange": "mexc",
  "futures": true    # automatisch aktiv
}

# Spot Trading  
{
  "exchange": "mexc",
  "futures": false   # manuell setzen wenn gewünscht
}
```

### Minimale Order Größen:
- MEXC hat andere Min-Order-Größen als Binance
- Der Bot berechnet automatisch die richtige Quantity
- Bei Fehlern prüfe die MEXC Symbol Info

## 7. Monitoring & Logs

### Bot Status prüfen:
```bash
curl http://localhost:5000/api/logs/recent?limit=50&level=INFO
```

### Account Balance (bei Live Trading):
```python
from core.trading.mexc_adapter import MEXCExchangeAdapter

adapter = MEXCExchangeAdapter(dry_run=False)
balance = adapter.get_account()
print(balance)
```

## 8. Troubleshooting

### Häufige Fehler:

**"API Key invalid":**
- Prüfe MEXC_API_KEY und MEXC_API_SECRET
- Stelle sicher, dass Trading Permission aktiviert ist

**"Insufficient balance":**
- Prüfe dein MEXC Account Balance
- Reduziere `equity` Parameter im Bot Call

**"Symbol not found":**
- Prüfe ob das Trading Pair auf MEXC existiert
- Format: BTCUSDT (ohne Bindestrich o.ä.)

**"Order size too small":**
- Erhöhe `min_notional` Parameter
- Oder erhöhe `equity` bzw. `risk_pct`

### Debug Mode:
```bash
# Ausführliche Logs
export LOG_LEVEL=DEBUG
python app.py
```

## 9. Python Script Beispiel

```python
import requests
import json

# MEXC Bot Run
def mexc_trading_bot(symbol="BTCUSDT", paper=True):
    url = "http://localhost:5000/api/bot/run"
    
    payload = {
        "symbol": symbol,
        "exchange": "mexc", 
        "interval": "1h",
        "paper": paper,
        "equity": 1000,
        "risk_pct": 0.5,
        "min_probability": 58,
        "min_rr": 1.3
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result['success']:
        print(f"✅ Bot run successful!")
        print(f"Exchange: {result['exchange']}")
        print(f"Paper Mode: {result['paper']}")
        print(f"Executed Trades: {len(result['data']['executed'])}")
    else:
        print(f"❌ Error: {result['error']}")
    
    return result

# Test run
mexc_trading_bot("SOLUSDT", paper=True)
```

## 10. Produktive Nutzung

1. **Phase 1**: Paper Trading für 1-2 Wochen
2. **Phase 2**: Sehr kleine Live Beträge (50-100 USDT)
3. **Phase 3**: Schrittweise Kapital erhöhen
4. **Phase 4**: Vollautomatisierung mit Monitoring

**Empfohlene Einstellungen für Anfang:**
```json
{
  "equity": 100,
  "risk_pct": 0.3,
  "min_probability": 60,
  "min_rr": 1.8
}
```
