# Railway Deployment Guide

Railway automatisch unterstützt Environment Variables! 

## 🚀 Railway Deployment Steps:

### 1. GitHub Repository erstellen
```bash
git init
git add .
git commit -m "Trading App with Binance API support"
git push origin main
```

### 2. Railway Project erstellen
- Gehe zu railway.app
- Connect GitHub Repository
- Deploy automatically

### 3. Environment Variables in Railway setzen
In Railway Dashboard → Variables Tab:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=false
RATE_LIMIT_REQUESTS_PER_MINUTE=1200
ENABLE_ACCOUNT_INFO=false
ENABLE_ORDER_BOOK_DEPTH=true
ENABLE_24H_TICKER_STATS=true
```

### 4. Port Configuration
Railway erkennt automatisch Flask Apps, aber stelle sicher dass der Port korrekt ist.

## ✅ Wichtige Dateien für Railway:

### requirements.txt
```
requests==2.32.4
pandas==2.3.1
numpy==2.3.2
Flask==3.1.1
flask-cors==6.0.1
scikit-learn==1.7.1
python-dotenv==1.1.1
```

### Procfile (optional)
```
web: python app_turbo.py
```

### runtime.txt (optional)
```
python-3.13.5
```

## 🔒 Sicherheit:
- ✅ .env ist in .gitignore
- ✅ API Keys nur in Railway Environment Variables
- ✅ Kein .env file im Repository

## 🎯 Deployment ohne .env:
JA! Railway verwendet seine eigenen Environment Variables.
Die App funktioniert automatisch ohne lokale .env Datei!
