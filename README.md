# 🚀 Ultimate Trading System V4

**Professional Trading Analysis mit TradingView-kompatiblen Indikatoren**

## ✨ Features

### 📊 **70% Fundamental Analysis**
- Market Sentiment Analysis
- Volume Profile & Smart Money Flow
- Risk Management Assessment

### 📈 **20% Technical Analysis**
- **TradingView-kompatible RSI** (Wilder's Smoothing)
- **Präzise MACD** mit EMA-Berechnung
- Bollinger Bands, Stochastic, Support/Resistance

### 🎯 **Coin-spezifische Features**
- **💥 Dynamische Liquidation Map**
  - BTC: 5x-100x Leverage
  - ETH: 3x-75x Leverage  
  - Altcoins: 2x-50x Leverage
- **🎯 Trading Setups**
  - Coin-spezifische Stop Loss & Take Profit
  - Automatisches Position Sizing
  - Risk/Reward Berechnung

### 🤖 **10% ML Confirmation**
- JAX Neural Networks
- Multi-Timeframe Consensus

## 🚀 Deployment

Das System läuft auf Railway mit Docker:

```bash
# Lokale Entwicklung
python app.py

# Docker Build
docker build -t trading-system .
docker run -p 5000:5000 trading-system
```

## 📊 API Endpoints

- `POST /api/analyze` - Hauptanalyse
- `GET /api/liquidation_map/<symbol>` - Liquidation Levels
- `GET /api/trading_setup/<symbol>` - Trading Setups
- `POST /api/backtest` - Strategy Backtest

## 🎯 Supported Symbols

- **Bitcoin**: BTCUSDT (Conservative: 2% SL, 5% TP)
- **Ethereum**: ETHUSDT (Moderate: 3% SL, 7% TP)
- **Top Altcoins**: SOL, ADA, DOT, AVAX (Aggressive: 4% SL, 10% TP)
- **Andere Coins**: Alle Binance Pairs (High Risk: 6% SL, 15% TP)

## 🔧 Tech Stack

- **Backend**: Flask, NumPy
- **APIs**: Binance API (Live Data)
- **ML**: JAX (Google Research)
- **Deployment**: Docker + Railway
- **UI**: Glassmorphism Design

## ⚠️ Disclaimer

Nur für Bildungszwecke. Trading birgt hohe Risiken!

---
**🎯 Status: PRODUCTION READY** ✅

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)
