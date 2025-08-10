"""
🚀 ULTRA-LIGHTWEIGHT TRADING SYSTEM FOR RAILWAY
================================================================
Minimal version with only Flask and requests - no heavy dependencies
"""

import os
import time
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify

try:
    import requests
except ImportError:
    # Fallback if requests is not available
    import urllib.request
    import urllib.parse
    requests = None

def get_binance_data(url):
    """Universal function to get data from Binance API"""
    try:
        if requests:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        else:
            # Fallback using urllib
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode('utf-8')
                return json.loads(data)
    except Exception as e:
        print(f"API Error: {e}")
    return None

# Simple caching system
cache = {}
cache_timestamps = {}

def get_cached_data(key, max_age=60):
    """Simple cache with expiration"""
    if key in cache and key in cache_timestamps:
        if time.time() - cache_timestamps[key] < max_age:
            return cache[key]
    return None

def set_cached_data(key, data):
    """Store data in cache"""
    cache[key] = data
    cache_timestamps[key] = time.time()

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'trading-system-key')

# HTML Template for the trading dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Trading Intelligence System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0f1c 0%, #1a1f2e 30%, #2d3748 70%, #1a202c 100%);
            min-height: 100vh;
            color: #e2e8f0;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            text-align: center; 
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 16px;
        }
        .title {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #34d399, #a78bfa);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            padding: 20px;
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-2px); }
        .card-title { 
            font-size: 18px; 
            font-weight: 700; 
            margin-bottom: 15px;
            color: #60a5fa;
        }
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #60a5fa);
            color: white;
        }
        .btn-success {
            background: linear-gradient(135deg, #10b981, #34d399);
            color: white;
        }
        .btn-warning {
            background: linear-gradient(135deg, #f59e0b, #fbbf24);
            color: white;
        }
        .result {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            min-height: 100px;
        }
        .status { text-align: center; color: #94a3b8; padding: 20px; }
        .loading { color: #06b6d4; }
        .success { color: #10b981; }
        .error { color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🚀 Trading Intelligence System</div>
            <p style="color: #94a3b8; margin-top: 10px;">AI-Powered Market Analysis • Railway Deployment</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">📊 Market Analysis</div>
                <button class="btn btn-primary" onclick="analyzeMarket()">Analyze BTCUSDT</button>
                <div id="analysis-result" class="result">
                    <div class="status">Click analyze to start market intelligence</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🎯 Trading Signals</div>
                <button class="btn btn-success" onclick="getTradingSignals()">Get Signals</button>
                <div id="signals-result" class="result">
                    <div class="status">Get smart trading recommendations</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🔥 Market Data</div>
                <button class="btn btn-warning" onclick="getMarketData()">Live Data</button>
                <div id="data-result" class="result">
                    <div class="status">Fetch real-time market information</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeMarket() {
            const resultDiv = document.getElementById('analysis-result');
            resultDiv.innerHTML = '<div class="status loading">🧠 Analyzing market...</div>';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: 'BTCUSDT' })
                });
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="success">✅ Analysis Complete</div>
                        <p><strong>Price:</strong> $${data.price}</p>
                        <p><strong>24h Change:</strong> ${data.change}%</p>
                        <p><strong>Signal:</strong> ${data.signal}</p>
                        <p><strong>Confidence:</strong> ${data.confidence}%</p>
                    `;
                } else {
                    resultDiv.innerHTML = '<div class="error">❌ Analysis failed</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="error">❌ Connection error</div>';
            }
        }
        
        async function getTradingSignals() {
            const resultDiv = document.getElementById('signals-result');
            resultDiv.innerHTML = '<div class="status loading">🎯 Getting signals...</div>';
            
            try {
                const response = await fetch('/api/signals', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: 'BTCUSDT' })
                });
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="success">✅ Signals Ready</div>
                        <p><strong>Action:</strong> ${data.action}</p>
                        <p><strong>Entry:</strong> $${data.entry}</p>
                        <p><strong>Target:</strong> $${data.target}</p>
                        <p><strong>Stop Loss:</strong> $${data.stop_loss}</p>
                    `;
                } else {
                    resultDiv.innerHTML = '<div class="error">❌ Signal generation failed</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="error">❌ Connection error</div>';
            }
        }
        
        async function getMarketData() {
            const resultDiv = document.getElementById('data-result');
            resultDiv.innerHTML = '<div class="status loading">📊 Fetching data...</div>';
            
            try {
                const response = await fetch('/api/market_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: 'BTCUSDT' })
                });
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="success">✅ Data Retrieved</div>
                        <p><strong>Volume 24h:</strong> ${data.volume}</p>
                        <p><strong>High 24h:</strong> $${data.high}</p>
                        <p><strong>Low 24h:</strong> $${data.low}</p>
                        <p><strong>Last Update:</strong> ${data.timestamp}</p>
                    `;
                } else {
                    resultDiv.innerHTML = '<div class="error">❌ Data fetch failed</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="error">❌ Connection error</div>';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """Market analysis endpoint"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSDT')
        
        # Check cache first
        cache_key = f"analysis_{symbol}"
        cached_result = get_cached_data(cache_key, 60)  # 1 minute cache
        if cached_result:
            return jsonify(cached_result)
        
        # Get market data from Binance
        binance_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        market_data = get_binance_data(binance_url)
        
        if market_data:
            
            # Simple analysis
            price = float(market_data['lastPrice'])
            change = float(market_data['priceChangePercent'])
            volume = float(market_data['volume'])
            
            # Basic signal generation
            if change > 2:
                signal = "BULLISH"
                confidence = min(85, 60 + abs(change) * 2)
            elif change < -2:
                signal = "BEARISH" 
                confidence = min(85, 60 + abs(change) * 2)
            else:
                signal = "NEUTRAL"
                confidence = 50
            
            result = {
                'success': True,
                'symbol': symbol,
                'price': f"{price:,.2f}",
                'change': f"{change:+.2f}",
                'signal': signal,
                'confidence': int(confidence),
                'volume': f"{volume:,.0f}",
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            # Cache the result
            set_cached_data(cache_key, result)
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': 'Market data unavailable'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/signals', methods=['POST'])
def trading_signals():
    """Trading signals endpoint"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSDT')
        
        # Get current price
        binance_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        price_data = get_binance_data(binance_url)
        
        if price_data:
            current_price = float(price_data['price'])
            
            # Simple signal generation
            entry = current_price
            stop_loss = current_price * 0.97  # 3% stop loss
            target = current_price * 1.06     # 6% target
            
            return jsonify({
                'success': True,
                'action': 'BUY',
                'entry': f"{entry:,.2f}",
                'target': f"{target:,.2f}",
                'stop_loss': f"{stop_loss:,.2f}",
                'risk_reward': '1:2',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        else:
            return jsonify({'success': False, 'error': 'Price data unavailable'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market_data', methods=['POST'])
def market_data():
    """Market data endpoint"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSDT')
        
        # Get 24hr ticker data
        binance_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        ticker_data = get_binance_data(binance_url)
        
        if ticker_data:
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'volume': f"{float(ticker_data['volume']):,.0f}",
                'high': f"{float(ticker_data['highPrice']):,.2f}",
                'low': f"{float(ticker_data['lowPrice']):,.2f}",
                'trades': f"{ticker_data['count']:,}",
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        else:
            return jsonify({'success': False, 'error': 'Market data unavailable'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'service': 'trading-analysis-system'
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Trading Analysis System on port {port}")
    print(f"📊 Health check available at http://0.0.0.0:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=False)
