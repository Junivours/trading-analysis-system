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

#### Option A: Nixpacks (Standard)
Railway verwendet automatisch `nixpacks.toml` und `railway.json`

#### Option B: Docker (Falls Nixpacks Probleme macht)
1. Lösche `railway.json`
2. Benenne `railway-docker.json` zu `railway.json` um
3. Railway verwendet dann das `Dockerfile`

### 4. Nix-Probleme beheben (Falls notwendig)
Wenn der Fehler "extern verwaltete Umgebung" auftritt:
```bash
# Railway Settings → Environment Variables hinzufügen:
PIP_BREAK_SYSTEM_PACKAGES=true
PYTHONUSERBASE=/opt/venv
```

### 5. Environment Variables (Optional)
```
SECRET_KEY=your-secret-key
FLASK_ENV=production
PIP_BREAK_SYSTEM_PACKAGES=true
```

### 6. Deployment starten
- Railway deployt automatisch bei jedem Push zum main branch
- Erste Deployment dauert ca. 2-3 Minuten
- System ist dann unter einer Railway-URL verfügbar

### Features des Trading Systems:
- 🧠 Live Market Analysis
- 📊 Real-time Trading Signals  
- 🎯 Price Monitoring
- ⚡ Ultra-lightweight (nur Flask + requests)
- 📈 24h Market Data
- 🔥 No external dependencies

### URLs nach Deployment:
- Live App: `https://your-app-name.railway.app`
- Health Check: `https://your-app-name.railway.app/health`

### Troubleshooting:
1. **Nix-Fehler:** Nutze Docker-Option oder setze Environment Variables
2. **Build-Fehler:** Prüfe Railway Logs für Details
3. **API-Fehler:** Binance API funktioniert ohne Keys

### 📱 Teste lokal:
```bash
python app_railway.py
# Öffne http://localhost:5000
```

---
**🚀 Two deployment options - Nixpacks oder Docker!**
