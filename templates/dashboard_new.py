def get_dashboard_template():
    """
    Enhanced Enterprise Dashboard Template - New 3-Column Layout
    
    Layout:
    - Left: Compact Recent Trades (3 rows)
    - Center: Detailed Trading Analysis with Setup Recommendations  
    - Right: Backtest + JAX Training Tools
    - Bottom: Liquidation Map with Support/Resistance Levels
    """
    
    return """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aurora Trading Control</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh;
            color: #e2e8f0;
            overflow-x: hidden;
        }
        
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }
        
        /* Topbar */
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 16px;
        }
        
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .brand h1 {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 24px;
            font-weight: 700;
        }
        
        .status-indicator {
            padding: 6px 12px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            border-radius: 20px;
            color: #10b981;
            font-size: 12px;
            font-weight: 600;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .user-avatar {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        /* Main 3-Column Layout */
        .main-content {
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            gap: 24px;
            margin-bottom: 24px;
        }
        
        /* Left Panel - Compact Recent Trades */
        .left-panel {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            height: fit-content;
        }
        
        /* Center Panel - Main Analysis */
        .center-panel {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        /* Right Panel - Tools */
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .tool-card {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        /* Compact Recent Trades */
        .compact-trades h3 {
            color: #a5b4fc;
            margin-bottom: 12px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .compact-trades table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        
        .compact-trades th,
        .compact-trades td {
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid rgba(51, 65, 85, 0.3);
        }
        
        .compact-trades th {
            background: rgba(99, 102, 241, 0.1);
            color: #a5b4fc;
            font-weight: 600;
            font-size: 11px;
        }
        
        .compact-trades .trade-buy { color: #10b981; font-weight: 600; }
        .compact-trades .trade-sell { color: #ef4444; font-weight: 600; }
        
        /* Trading Setup Analysis */
        .trading-analysis h3 {
            color: #a5b4fc;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .setup-recommendation {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border-left: 4px solid #6366f1;
        }
        
        .setup-type {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 8px;
        }
        
        .setup-type.long { color: #10b981; }
        .setup-type.short { color: #ef4444; }
        .setup-type.wait { color: #f59e0b; }
        
        .setup-reason {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.5;
        }
        
        /* Support/Resistance Levels */
        .sr-levels {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 16px;
            margin-top: 16px;
        }
        
        .level-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(51, 65, 85, 0.3);
        }
        
        .level-item:last-child {
            border-bottom: none;
        }
        
        .level-type {
            font-weight: 600;
            font-size: 12px;
        }
        
        .level-type.support { color: #10b981; }
        .level-type.resistance { color: #ef4444; }
        
        .level-price {
            color: #e2e8f0;
            font-weight: 600;
        }
        
        .level-strength {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        .level-strength.strong {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }
        
        .level-strength.weak {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        
        /* Buttons */
        button {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(139, 92, 246, 0.8));
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: 14px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
            background: linear-gradient(135deg, rgba(99, 102, 241, 1), rgba(139, 92, 246, 1));
        }
        
        button:active {
            transform: translateY(0);
        }
        
        /* Results Section */
        #results {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            margin-top: 24px;
            animation: slideUp 0.3s ease;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        #results h3 {
            color: #a5b4fc;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .analysis-table {
            overflow-x: auto;
        }
        
        .analysis-table table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            overflow: hidden;
        }
        
        .analysis-table th,
        .analysis-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }
        
        .analysis-table th {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .analysis-table td {
            color: #e2e8f0;
            font-size: 14px;
        }
        
        .signal-buy {
            color: #10b981;
            font-weight: 600;
        }
        
        .signal-sell {
            color: #ef4444;
            font-weight: 600;
        }
        
        .signal-hold {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .confidence-high {
            color: #10b981;
            font-weight: 600;
        }
        
        .confidence-medium {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .confidence-low {
            color: #ef4444;
            font-weight: 600;
        }
        
        /* Responsive Design */
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Topbar -->
        <div class="topbar">
            <div class="brand">
                <h1>Aurora Trading</h1>
                <div class="status-indicator">
                    Live: Connected
                </div>
            </div>
            <div class="user-info">
                <div class="user-avatar">JD</div>
            </div>
        </div>
        
        <!-- New 3-Column Layout -->
        <div class="main-content">
            <!-- Left Panel: Compact Recent Trades -->
            <div class="left-panel">
                <div class="compact-trades">
                    <h3><i class="fas fa-clock"></i> Recent Trades</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Pair</th>
                                <th>Side</th>
                                <th>Price</th>
                            </tr>
                        </thead>
                        <tbody id="recent-trades-body">
                            <tr>
                                <td colspan="3" style="text-align: center; color: #a5b4fc;">
                                    <i class="fas fa-spinner fa-spin"></i> Loading...
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Center Panel: Detailed Trading Analysis -->
            <div class="center-panel">
                <div class="trading-analysis">
                    <h3><i class="fas fa-chart-line"></i> Trading Setup Analysis</h3>
                    
                    <!-- Market Analysis Buttons (compact) -->
                    <div style="display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;">
                        <button onclick="testAnalysis()" style="padding: 8px 16px; font-size: 13px;">
                            <i class="fas fa-chart-area"></i> Analyze Market
                        </button>
                        <button onclick="testTradingSetup()" style="padding: 8px 16px; font-size: 13px;">
                            <i class="fas fa-bullseye"></i> Trading Setup
                        </button>
                        <button onclick="testLiquidationMap()" style="padding: 8px 16px; font-size: 13px;">
                            <i class="fas fa-map-marker-alt"></i> Liquidation Map
                        </button>
                    </div>
                    
                    <!-- Trading Recommendation -->
                    <div id="trading-recommendation">
                        <div class="setup-recommendation">
                            <div class="setup-type wait">⏳ Warte auf Analyse...</div>
                            <div class="setup-reason">
                                Klicke auf "Analyze Market" um eine detaillierte Marktanalyse zu erhalten.
                            </div>
                        </div>
                    </div>
                    
                    <!-- Support/Resistance Levels -->
                    <div id="sr-levels-container">
                        <h4 style="color: #a5b4fc; margin-bottom: 12px;">📊 Wichtige Support/Resistance Levels</h4>
                        <div class="sr-levels" id="sr-levels">
                            <div class="level-item">
                                <div>
                                    <span class="level-type support">SUPPORT</span>
                                </div>
                                <div class="level-price">Loading...</div>
                                <div class="level-strength strong">STRONG</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Tools -->
            <div class="right-panel">
                <!-- Backtest Tool -->
                <div class="tool-card">
                    <h4 style="color: #a5b4fc; margin-bottom: 12px;">📈 Backtest</h4>
                    <button onclick="testBacktest()" style="width: 100%; padding: 10px; font-size: 13px;">
                        <i class="fas fa-history"></i> Run Backtest
                    </button>
                    <div id="backtest-result" style="margin-top: 10px; font-size: 12px; color: #cbd5e1;">
                        Ready to backtest strategies
                    </div>
                </div>
                
                <!-- JAX Training Tool -->
                <div class="tool-card">
                    <h4 style="color: #a5b4fc; margin-bottom: 12px;">🧠 JAX Training</h4>
                    <button onclick="testJaxTraining()" style="width: 100%; padding: 10px; font-size: 13px; margin-bottom: 8px;">
                        <i class="fas fa-brain"></i> Train Model
                    </button>
                    <button onclick="testJaxPrediction()" style="width: 100%; padding: 10px; font-size: 13px;">
                        <i class="fas fa-crystal-ball"></i> Predict
                    </button>
                    <div id="jax-result" style="margin-top: 10px; font-size: 12px; color: #cbd5e1;">
                        Neural network ready
                    </div>
                </div>
                
                <!-- Multi-Asset Tool -->
                <div class="tool-card">
                    <h4 style="color: #a5b4fc; margin-bottom: 12px;">🌐 Multi-Asset</h4>
                    <button onclick="testMultiAsset()" style="width: 100%; padding: 10px; font-size: 13px;">
                        <i class="fas fa-globe"></i> Analyze Portfolio
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Bottom Section: Liquidation Map -->
        <div id="liquidation-map-section" style="display: none; margin-top: 24px;">
            <div style="background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(20px); border-radius: 16px; padding: 24px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h3 style="color: #a5b4fc; margin-bottom: 20px;">
                    <i class="fas fa-map-marker-alt"></i> Liquidation Levels Map
                </h3>
                <div id="liquidation-levels">
                    <!-- Liquidation levels will be populated here -->
                </div>
            </div>
        </div>
        
        <!-- Results section - DO NOT CHANGE IDs -->
        <div id="results" style="display:none;">
            <h3><i class="fas fa-chart-line"></i> Trading Analysis Results</h3>
            <div id="results-content">
                <!-- Analysis results will be displayed here as a table -->
                <div class="analysis-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Signal</th>
                                <th>Confidence</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="analysis-results-body">
                            <!-- Dynamic content will be inserted here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Enhanced Trading API Functions with Smart Recommendations
        function formatAnalysisResults(data) {
            const tbody = document.getElementById('analysis-results-body');
            tbody.innerHTML = '';
            
            if (data.success && data.market_analysis) {
                const analysis = data.market_analysis;
                
                // Update Trading Recommendation
                updateTradingRecommendation(analysis);
                
                // Update Support/Resistance Levels
                updateSRLevels(data);
                
                // Add analysis rows
                const addRow = (metric, value, signal, confidence, action) => {
                    const row = document.createElement('tr');
                    
                    // Apply signal colors
                    let signalClass = '';
                    if (signal === 'BUY' || signal === 'BULLISH') signalClass = 'signal-buy';
                    else if (signal === 'SELL' || signal === 'BEARISH') signalClass = 'signal-sell';
                    else signalClass = 'signal-hold';
                    
                    // Apply confidence colors
                    let confidenceClass = '';
                    if (confidence >= 80) confidenceClass = 'confidence-high';
                    else if (confidence >= 60) confidenceClass = 'confidence-medium';
                    else confidenceClass = 'confidence-low';
                    
                    row.innerHTML = `
                        <td>${metric}</td>
                        <td>${value}</td>
                        <td class="${signalClass}">${signal}</td>
                        <td class="${confidenceClass}">${confidence}%</td>
                        <td>${action}</td>
                    `;
                    tbody.appendChild(row);
                };
                
                // Technical indicators
                if (analysis.technical_analysis) {
                    const tech = analysis.technical_analysis;
                    if (tech.rsi) {
                        const rsiSignal = tech.rsi.value > 70 ? 'SELL' : tech.rsi.value < 30 ? 'BUY' : 'HOLD';
                        addRow('RSI', tech.rsi.value.toFixed(2), rsiSignal, tech.rsi.confidence || 75, rsiSignal);
                    }
                    if (tech.macd) {
                        const macdSignal = tech.macd.signal === 'bullish' ? 'BUY' : tech.macd.signal === 'bearish' ? 'SELL' : 'HOLD';
                        addRow('MACD', tech.macd.histogram?.toFixed(4) || 'N/A', macdSignal.toUpperCase(), tech.macd.confidence || 70, macdSignal);
                    }
                    if (tech.bollinger_bands) {
                        const bbSignal = tech.bollinger_bands.position;
                        addRow('Bollinger Bands', tech.bollinger_bands.squeeze_ratio?.toFixed(3) || 'N/A', bbSignal.toUpperCase(), 65, bbSignal);
                    }
                }
                
                // Market conditions
                if (analysis.market_conditions) {
                    const market = analysis.market_conditions;
                    addRow('Market Trend', market.trend || 'UNKNOWN', market.trend || 'HOLD', market.confidence || 60, market.trend || 'HOLD');
                    addRow('Volume Analysis', market.volume_trend || 'N/A', market.volume_signal || 'HOLD', 70, market.volume_signal || 'HOLD');
                }
                
                // Overall recommendation
                if (analysis.overall_score) {
                    const score = analysis.overall_score;
                    const signal = score > 7 ? 'BUY' : score < 3 ? 'SELL' : 'HOLD';
                    addRow('Overall Score', `${score}/10`, signal, Math.min(score * 10, 100), signal);
                }
                
                // Current price info
                if (data.ticker_data) {
                    const ticker = data.ticker_data;
                    const changeSignal = parseFloat(ticker.priceChangePercent) > 0 ? 'BULLISH' : 'BEARISH';
                    addRow('Current Price', `$${parseFloat(ticker.lastPrice).toFixed(2)}`, changeSignal, 100, 'MONITOR');
                    addRow('24h Change', `${ticker.priceChangePercent}%`, changeSignal, 100, 'MONITOR');
                    addRow('24h Volume', `${parseFloat(ticker.volume).toFixed(0)}`, 'INFO', 100, 'MONITOR');
                }
                
            } else {
                // Error or no data
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: #ef4444;">
                            ${data.error || 'No analysis data available'}
                        </td>
                    </tr>
                `;
            }
        }
        
        function updateTradingRecommendation(analysis) {
            const container = document.getElementById('trading-recommendation');
            
            // Determine overall recommendation
            let setupType = 'wait';
            let setupText = 'Warte noch bevor du eine Position eröffnest...';
            let setupReason = 'Marktbedingungen sind noch nicht optimal.';
            
            if (analysis.overall_score) {
                const score = analysis.overall_score;
                
                if (score >= 8) {
                    setupType = 'long';
                    setupText = '🚀 STARKER LONG AUFBAU';
                    setupReason = `Sehr starkes Kaufsignal (Score: ${score}/10). RSI unter 50, MACD bullish, starker Support Level erreicht. Langsamer Long-Aufbau empfohlen.`;
                } else if (score >= 6) {
                    setupType = 'long';
                    setupText = '📈 Langsamer Long Aufbau';
                    setupReason = `Moderates Kaufsignal (Score: ${score}/10). Technische Indikatoren zeigen positive Tendenz. Vorsichtiger Einstieg mit kleinen Positionen.`;
                } else if (score <= 2) {
                    setupType = 'short';
                    setupText = '📉 SHORT POSITION';
                    setupReason = `Starkes Verkaufssignal (Score: ${score}/10). RSI überkauft, MACD bearish, wichtige Resistance durchbrochen. Short-Position erwägen.`;
                } else if (score <= 4) {
                    setupType = 'short';
                    setupText = '🔻 Langsamer Short Aufbau';
                    setupReason = `Schwaches Verkaufssignal (Score: ${score}/10). Markt zeigt Schwäche, aber noch keine klare Richtung. Vorsichtige Short-Positionen möglich.`;
                }
            }
            
            container.innerHTML = `
                <div class="setup-recommendation">
                    <div class="setup-type ${setupType}">${setupText}</div>
                    <div class="setup-reason">${setupReason}</div>
                </div>
            `;
        }
        
        function updateSRLevels(data) {
            const container = document.getElementById('sr-levels');
            
            // Extract support/resistance from ticker data
            if (data.ticker_data) {
                const ticker = data.ticker_data;
                const currentPrice = parseFloat(ticker.lastPrice);
                const high24h = parseFloat(ticker.highPrice);
                const low24h = parseFloat(ticker.lowPrice);
                
                container.innerHTML = `
                    <div class="level-item">
                        <div><span class="level-type resistance">RESISTANCE</span></div>
                        <div class="level-price">$${high24h.toFixed(2)}</div>
                        <div class="level-strength strong">STRONG</div>
                    </div>
                    <div class="level-item">
                        <div><span class="level-type support">CURRENT</span></div>
                        <div class="level-price">$${currentPrice.toFixed(2)}</div>
                        <div class="level-strength">LIVE</div>
                    </div>
                    <div class="level-item">
                        <div><span class="level-type support">SUPPORT</span></div>
                        <div class="level-price">$${low24h.toFixed(2)}</div>
                        <div class="level-strength strong">STRONG</div>
                    </div>
                `;
            }
        }
        
        async function makeAPICall(endpoint, data = {}) {
            const resultsDiv = document.getElementById('results');
            const resultsContent = document.getElementById('results-content');
            
            try {
                resultsDiv.style.display = 'block';
                
                // Show loading in table
                const tbody = document.getElementById('analysis-results-body');
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: #a5b4fc;">
                            <i class="fas fa-spinner fa-spin"></i> Loading real Binance data...
                        </td>
                    </tr>
                `;
                
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Format results as table instead of raw JSON
                formatAnalysisResults(result);
                
            } catch (error) {
                const tbody = document.getElementById('analysis-results-body');
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: #ef4444;">
                            Error: ${error.message}
                        </td>
                    </tr>
                `;
            }
        }
        
        function testAnalysis() {
            makeAPICall('/api/analyze', { symbol: 'BTCUSDT' });
        }

        function testLiquidationMap() {
            // Show liquidation map section
            document.getElementById('liquidation-map-section').style.display = 'block';
            makeAPICall('/api/liquidation_map', { symbol: 'BTCUSDT' });
        }

        function testTradingSetup() {
            makeAPICall('/api/trading_setup', { symbol: 'BTCUSDT', timeframe: '1h' });
        }

        function testJaxTraining() {
            const resultDiv = document.getElementById('jax-result');
            resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
            makeAPICall('/api/jax_train', { symbol: 'BTCUSDT', interval: '1h', epochs: 50 });
        }

        function testBacktest() {
            const resultDiv = document.getElementById('backtest-result');
            resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Backtesting...';
            makeAPICall('/api/backtest', { symbol: 'BTCUSDT', start_date: '2024-01-01' });
        }

        function testMultiAsset() {
            makeAPICall('/api/multi_asset', { symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'] });
        }

        function testJaxPrediction() {
            const resultDiv = document.getElementById('jax-result');
            resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
            makeAPICall('/api/jax_predict', { symbol: 'BTCUSDT', interval: '1h' });
        }
        
        // Load real recent trades
        async function loadRecentTrades() {
            try {
                const response = await fetch('/api/recent_trades', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'] })
                });
                
                const result = await response.json();
                const tbody = document.getElementById('recent-trades-body');
                
                if (result.success && result.trades) {
                    tbody.innerHTML = '';
                    // Show only first 3 trades for compact view
                    result.trades.slice(0, 3).forEach(trade => {
                        const row = document.createElement('tr');
                        const sideClass = trade.isBuyerMaker ? 'trade-sell' : 'trade-buy';
                        const side = trade.isBuyerMaker ? 'SELL' : 'BUY';
                        
                        row.innerHTML = `
                            <td>${trade.symbol.replace('USDT', '')}</td>
                            <td class="${sideClass}">${side}</td>
                            <td>$${parseFloat(trade.price).toFixed(0)}</td>
                        `;
                        tbody.appendChild(row);
                    });
                } else {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="3" style="text-align: center; color: #ef4444;">
                                Error loading trades
                            </td>
                        </tr>
                    `;
                }
            } catch (error) {
                const tbody = document.getElementById('recent-trades-body');
                tbody.innerHTML = `
                    <tr>
                        <td colspan="3" style="text-align: center; color: #ef4444;">
                            Connection error
                        </td>
                    </tr>
                `;
            }
        }
        
        // Load recent trades when page loads
        window.addEventListener('load', function() {
            loadRecentTrades();
            // Refresh trades every 30 seconds
            setInterval(loadRecentTrades, 30000);
        });
    </script>
</body>
</html>
"""
