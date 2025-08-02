# 🚀 DEPLOYMENT GUIDE

## GitHub zu Heroku Deployment

### Schritt 1: Repository vorbereiten
```bash
cd C:\Users\faruk\Downloads\Backuo
git init
git add .
git commit -m "Initial commit - Trading Analysis Pro v6.0"
```

### Schritt 2: GitHub Repository verbinden
```bash
git remote add origin https://github.com/Junivours/trading-analysis-system.git
git branch -M main
git push -u origin main
```

### Schritt 3: Heroku Deployment
```bash
# Heroku CLI installieren: https://devcenter.heroku.com/articles/heroku-cli

# Heroku Login
heroku login

# Neue Heroku App erstellen
heroku create trading-analysis-pro-2024

# Environment Variables setzen
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# App öffnen
heroku open
```

## Railway Deployment (Automatisch)

### Schritt 1: Railway Account
1. Gehe zu https://railway.app
2. Login mit GitHub Account
3. Klick "New Project"
4. Wähle "Deploy from GitHub repo"
5. Wähle `trading-analysis-system` Repository
6. Railway erkennt automatisch Flask App
7. Deployment startet automatisch

### Schritt 2: Domain Setup
1. Gehe zu Railway Dashboard
2. Klick auf dein Projekt
3. Tab "Settings" → "Domains"
4. Generiere Domain oder custom domain hinzufügen

## Vercel Deployment

### Schritt 1: vercel.json erstellen (bereits gemacht)
### Schritt 2: Vercel Login
```bash
npm i -g vercel
vercel login
```

### Schritt 3: Deploy
```bash
vercel --prod
```

## Environment Variables (für alle Plattformen)

```
PORT=8080
FLASK_ENV=production
PYTHONPATH=/app
```

## Wichtige URLs nach Deployment

- **Heroku**: https://trading-analysis-pro-2024.herokuapp.com
- **Railway**: https://trading-analysis-system-production.up.railway.app  
- **Vercel**: https://trading-analysis-system.vercel.app

## Testing der Deployment

### Health Check
```bash
curl https://your-app-url.herokuapp.com/api/status
```

### API Test
```bash
curl -X POST https://your-app-url.herokuapp.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","interval":"1h","limit":200}'
```

## Monitoring

- **Heroku**: `heroku logs --tail`
- **Railway**: Logs im Dashboard verfügbar
- **Vercel**: Logs im Dashboard verfügbar

## Troubleshooting

### Common Issues:
1. **Import Errors**: requirements.txt aktualisieren
2. **Port Issues**: PORT environment variable prüfen  
3. **Memory Limits**: Heroku Dyno Typ upgraden
4. **Timeout**: gunicorn timeout erhöhen

### Debug Commands:
```bash
# Heroku
heroku run python -c "import pandas_ta; print('pandas-ta OK')"
heroku ps:scale web=1
heroku restart

# Railway
# Logs im Dashboard einsehen
```

## Success Metrics

✅ App starts without errors
✅ API endpoints respond correctly  
✅ Technical analysis calculations work
✅ ML predictions generate
✅ UI loads properly
✅ Real-time data fetching works

Deployment sollte in 5-10 Minuten abgeschlossen sein!
