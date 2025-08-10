# 🚀 RAILWAY DEPLOYMENT README

## Deployment auf Railway

### 1. Railway Account erstellen
- Gehe zu [railway.app](https://railway.app)
- Registriere dich mit GitHub Account

### 2. Neues Project erstellen
1. Klicke auf "New Project"
2. Wähle "Deploy from GitHub repo"
3. Wähle das Repository `Junivours/trading-analysis-system`

### 3. Deployment-Konfiguration
- Railway erkennt automatisch die `nixpacks.toml` und `railway.json`
- Das System startet automatisch mit `python app.py`
- Port wird automatisch von Railway konfiguriert

### 4. Environment Variables (Optional)
```
SECRET_KEY=your-secret-key
FLASK_ENV=production
BINANCE_API_KEY=optional
BINANCE_SECRET_KEY=optional
```

### 5. Deployment starten
- Railway deployt automatisch bei jedem Push zum main branch
- Erste Deployment dauert ca. 2-3 Minuten
- System ist dann unter einer Railway-URL verfügbar

### Features des Trading Systems:
- 🧠 KI-basierte Marktanalyse
- 📊 Live Trading Signale  
- 🎯 Pattern Recognition
- ⚡ Real-time Coin Search
- 📈 Advanced Backtesting
- 🔥 Liquidation Maps

### URLs nach Deployment:
- Live App: `https://your-app-name.railway.app`
- API Docs: `https://your-app-name.railway.app/docs`

### Support:
Bei Problemen checke die Railway Logs oder erstelle ein GitHub Issue.

---
**🚀 Trading System ready for production!**
