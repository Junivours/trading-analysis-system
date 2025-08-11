from flask import Flask, jsonify, request
import os
import json
import time
import random

app = Flask(__name__)

# Fallback-Daten für Demo-Zwecke
def get_demo_market_data(symbol="BTCUSDT"):
    """Generiert realistische Demo-Marktdaten"""
    base_price = 45000 if "BTC" in symbol else 100 if "SOL" in symbol else 2000
    return {
        'symbol': symbol,
        'price': round(base_price + random.uniform(-1000, 1000), 2),
        'change24h': round(random.uniform(-5, 5), 2),
        'volume': round(random.uniform(1000000, 10000000), 0),
        'high24h': round(base_price * 1.05, 2),
        'low24h': round(base_price * 0.95, 2)
    }

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Advanced Trading Intelligence Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: white;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .glass {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                padding: 30px;
                margin: 20px 0;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #00f5ff, #ff6b6b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .control-group {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .control-group h3 {
                color: #00f5ff;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            
            input, select, button {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: none;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 16px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            input::placeholder {
                color: rgba(255, 255, 255, 0.7);
            }
            
            button {
                background: linear-gradient(45deg, #00f5ff, #0099cc);
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 245, 255, 0.3);
            }
            
            .results {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            
            .result-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 25px;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }
            
            .result-card h3 {
                color: #ff6b6b;
                margin-bottom: 15px;
                font-size: 1.3em;
            }
            
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .metric:last-child {
                border-bottom: none;
            }
            
            .positive { color: #00ff88; }
            .negative { color: #ff4757; }
            .neutral { color: #ffa502; }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #00f5ff;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #00ff88;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .pattern-list {
                list-style: none;
                padding: 0;
            }
            
            .pattern-list li {
                background: rgba(255, 255, 255, 0.05);
                margin: 8px 0;
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid #00f5ff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Advanced Trading Intelligence</h1>
                <p><span class="status-indicator"></span>Railway Deployment - Live System</p>
            </div>
            
            <div class="glass">
                <div class="controls">
                    <div class="control-group">
                        <h3>📊 Market Analysis</h3>
                        <input type="text" id="symbolInput" placeholder="Enter symbol (e.g., BTCUSDT, SOLUSDT)" value="BTCUSDT">
                        <button onclick="analyzeMarket()">🔍 Analyze Market</button>
                        <button onclick="getTradingSignals()">📈 Get Trading Signals</button>
                    </div>
                    
                    <div class="control-group">
                        <h3>🎯 Pattern Scanner</h3>
                        <select id="timeframeSelect">
                            <option value="1h">1 Hour</option>
                            <option value="4h">4 Hours</option>
                            <option value="1d">1 Day</option>
                        </select>
                        <button onclick="scanPatterns()">🔄 Scan Patterns</button>
                        <button onclick="getLiquidationMap()">💀 Liquidation Map</button>
                    </div>
                    
                    <div class="control-group">
                        <h3>⚡ Quick Actions</h3>
                        <button onclick="getBestSetups()">🎯 Best Setups</button>
                        <button onclick="getMarketSentiment()">😊 Market Sentiment</button>
                        <button onclick="refreshData()">🔄 Refresh All</button>
                    </div>
                </div>
            </div>
            
            <div id="results" class="results">
                <!-- Results will be populated here -->
            </div>
        </div>
        
        <script>
            let currentSymbol = 'BTCUSDT';
            
            function showLoading(elementId) {
                const element = document.getElementById(elementId) || document.getElementById('results');
                element.innerHTML = '<div class="loading">🔄 Loading data...</div>';
            }
            
            function analyzeMarket() {
                const symbol = document.getElementById('symbolInput').value.toUpperCase() || 'BTCUSDT';
                currentSymbol = symbol;
                showLoading('results');
                
                fetch(`/api/market-analysis?symbol=${symbol}`)
                    .then(response => response.json())
                    .then(data => displayMarketAnalysis(data))
                    .catch(error => displayError('Market Analysis', error));
            }
            
            function getTradingSignals() {
                showLoading('results');
                fetch(`/api/trading-signals?symbol=${currentSymbol}`)
                    .then(response => response.json())
                    .then(data => displayTradingSignals(data))
                    .catch(error => displayError('Trading Signals', error));
            }
            
            function scanPatterns() {
                const timeframe = document.getElementById('timeframeSelect').value;
                showLoading('results');
                
                fetch(`/api/patterns?symbol=${currentSymbol}&timeframe=${timeframe}`)
                    .then(response => response.json())
                    .then(data => displayPatterns(data))
                    .catch(error => displayError('Pattern Scanner', error));
            }
            
            function getLiquidationMap() {
                showLoading('results');
                fetch(`/api/liquidation-map?symbol=${currentSymbol}`)
                    .then(response => response.json())
                    .then(data => displayLiquidationMap(data))
                    .catch(error => displayError('Liquidation Map', error));
            }
            
            function getBestSetups() {
                showLoading('results');
                fetch('/api/best-setups')
                    .then(response => response.json())
                    .then(data => displayBestSetups(data))
                    .catch(error => displayError('Best Setups', error));
            }
            
            function getMarketSentiment() {
                showLoading('results');
                fetch('/api/market-sentiment')
                    .then(response => response.json())
                    .then(data => displayMarketSentiment(data))
                    .catch(error => displayError('Market Sentiment', error));
            }
            
            function displayMarketAnalysis(data) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>📊 Market Analysis - ${data.symbol}</h3>
                        <div class="metric">
                            <span>Current Price:</span>
                            <span class="positive">$${data.price}</span>
                        </div>
                        <div class="metric">
                            <span>24h Change:</span>
                            <span class="${data.change24h >= 0 ? 'positive' : 'negative'}">${data.change24h >= 0 ? '+' : ''}${data.change24h}%</span>
                        </div>
                        <div class="metric">
                            <span>24h Volume:</span>
                            <span class="neutral">${data.volume.toLocaleString()}</span>
                        </div>
                        <div class="metric">
                            <span>24h High:</span>
                            <span class="positive">$${data.high24h}</span>
                        </div>
                        <div class="metric">
                            <span>24h Low:</span>
                            <span class="negative">$${data.low24h}</span>
                        </div>
                    </div>
                    <div class="result-card glass">
                        <h3>🎯 Technical Analysis</h3>
                        <div class="metric">
                            <span>Trend:</span>
                            <span class="positive">Bullish</span>
                        </div>
                        <div class="metric">
                            <span>RSI:</span>
                            <span class="neutral">65.2</span>
                        </div>
                        <div class="metric">
                            <span>Support:</span>
                            <span class="positive">$${(data.price * 0.95).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span>Resistance:</span>
                            <span class="negative">$${(data.price * 1.05).toFixed(2)}</span>
                        </div>
                    </div>
                `;
            }
            
            function displayTradingSignals(data) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>📈 Trading Signals</h3>
                        <div class="metric">
                            <span>Signal Strength:</span>
                            <span class="positive">Strong Buy</span>
                        </div>
                        <div class="metric">
                            <span>Entry Point:</span>
                            <span class="neutral">$${data.price}</span>
                        </div>
                        <div class="metric">
                            <span>Stop Loss:</span>
                            <span class="negative">$${(data.price * 0.97).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span>Take Profit:</span>
                            <span class="positive">$${(data.price * 1.08).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span>Risk/Reward:</span>
                            <span class="positive">1:2.67</span>
                        </div>
                    </div>
                `;
            }
            
            function displayPatterns(data) {
                const patterns = ['Ascending Triangle', 'Bull Flag', 'Cup and Handle', 'Support Breakout'];
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>🔄 Chart Patterns Detected</h3>
                        <ul class="pattern-list">
                            ${patterns.map(pattern => `<li>✅ ${pattern}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            function displayLiquidationMap(data) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>💀 Liquidation Heat Map</h3>
                        <div class="metric">
                            <span>Major Support:</span>
                            <span class="positive">$${(data.price * 0.92).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span>Liquidation Zone:</span>
                            <span class="negative">$${(data.price * 0.88).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span>Long Liquidations:</span>
                            <span class="negative">$2.3M</span>
                        </div>
                        <div class="metric">
                            <span>Short Liquidations:</span>
                            <span class="positive">$1.8M</span>
                        </div>
                    </div>
                `;
            }
            
            function displayBestSetups(data) {
                const setups = ['BTC/USDT - Bull Flag', 'ETH/USDT - Ascending Triangle', 'SOL/USDT - Breakout'];
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>🎯 Best Trading Setups</h3>
                        <ul class="pattern-list">
                            ${setups.map(setup => `<li>🚀 ${setup}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            function displayMarketSentiment(data) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>😊 Market Sentiment</h3>
                        <div class="metric">
                            <span>Overall Sentiment:</span>
                            <span class="positive">Bullish (78%)</span>
                        </div>
                        <div class="metric">
                            <span>Fear & Greed Index:</span>
                            <span class="neutral">65 (Greed)</span>
                        </div>
                        <div class="metric">
                            <span>Social Media Buzz:</span>
                            <span class="positive">High</span>
                        </div>
                        <div class="metric">
                            <span>Whale Activity:</span>
                            <span class="positive">Accumulating</span>
                        </div>
                    </div>
                `;
            }
            
            function displayError(feature, error) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result-card glass">
                        <h3>⚠️ ${feature} - Demo Mode</h3>
                        <p>Running in demo mode with simulated data.</p>
                        <p>Full API integration available in production environment.</p>
                    </div>
                `;
            }
            
            function refreshData() {
                analyzeMarket();
            }
            
            // Auto-load market analysis on page load
            window.addEventListener('load', function() {
                setTimeout(analyzeMarket, 1000);
            });
        </script>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'trading-system',
        'environment': 'railway'
    }), 200

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'message': 'Trading System API is running',
        'endpoints': ['/', '/health', '/api/status']
    }), 200

@app.route('/api/market-analysis')
def market_analysis():
    symbol = request.args.get('symbol', 'BTCUSDT')
    data = get_demo_market_data(symbol)
    return jsonify(data)

@app.route('/api/trading-signals')
def trading_signals():
    symbol = request.args.get('symbol', 'BTCUSDT')
    data = get_demo_market_data(symbol)
    
    signals = {
        'symbol': symbol,
        'signal': 'BUY',
        'strength': 'Strong',
        'entry_price': data['price'],
        'stop_loss': round(data['price'] * 0.97, 2),
        'take_profit': round(data['price'] * 1.08, 2),
        'risk_reward': '1:2.67',
        'confidence': 85
    }
    return jsonify(signals)

@app.route('/api/patterns')
def patterns():
    symbol = request.args.get('symbol', 'BTCUSDT')
    timeframe = request.args.get('timeframe', '1h')
    
    patterns = {
        'symbol': symbol,
        'timeframe': timeframe,
        'patterns_detected': [
            {'name': 'Ascending Triangle', 'confidence': 92},
            {'name': 'Bull Flag', 'confidence': 78},
            {'name': 'Cup and Handle', 'confidence': 65},
            {'name': 'Support Breakout', 'confidence': 88}
        ],
        'trend': 'Bullish',
        'score': 85
    }
    return jsonify(patterns)

@app.route('/api/liquidation-map')
def liquidation_map():
    symbol = request.args.get('symbol', 'BTCUSDT')
    data = get_demo_market_data(symbol)
    
    liquidations = {
        'symbol': symbol,
        'current_price': data['price'],
        'major_support': round(data['price'] * 0.92, 2),
        'liquidation_zone': round(data['price'] * 0.88, 2),
        'long_liquidations': '$2.3M',
        'short_liquidations': '$1.8M',
        'heat_map': [
            {'price': round(data['price'] * 0.95, 2), 'volume': 'High'},
            {'price': round(data['price'] * 0.90, 2), 'volume': 'Extreme'},
            {'price': round(data['price'] * 0.85, 2), 'volume': 'Critical'}
        ]
    }
    return jsonify(liquidations)

@app.route('/api/best-setups')
def best_setups():
    setups = {
        'top_setups': [
            {'pair': 'BTC/USDT', 'pattern': 'Bull Flag', 'score': 95},
            {'pair': 'ETH/USDT', 'pattern': 'Ascending Triangle', 'score': 88},
            {'pair': 'SOL/USDT', 'pattern': 'Breakout', 'score': 82},
            {'pair': 'ADA/USDT', 'pattern': 'Cup and Handle', 'score': 75}
        ],
        'market_condition': 'Bullish',
        'recommended_action': 'Long positions favored'
    }
    return jsonify(setups)

@app.route('/api/market-sentiment')
def market_sentiment():
    sentiment = {
        'overall_sentiment': 'Bullish',
        'sentiment_score': 78,
        'fear_greed_index': 65,
        'fear_greed_text': 'Greed',
        'social_media_buzz': 'High',
        'whale_activity': 'Accumulating',
        'indicators': {
            'rsi': 65.2,
            'macd': 'Bullish',
            'moving_averages': 'Above'
        }
    }
    return jsonify(sentiment)

if __name__ == '__main__':
    # Railway provides PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
