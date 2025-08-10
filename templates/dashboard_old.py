def get_dashboard_template():
    """
    Enterprise Dashboard Template (HTML/CSS/JS Version)
    
    Modern professional design inspired by Enterprise dashboards
    You can modify EVERYTHING in this template except:
    - JavaScript functions: testAnalysis(), testJaxTraining(), etc.
    - Element IDs: 'results', 'results-content'
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
            gap: 16px;
        }
        
        .brand-icon {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            padding: 12px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }
        
        .brand-text h1 {
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #f8fafc, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .brand-text p {
            font-size: 14px;
            opacity: 0.7;
            margin-top: 4px;
        }
        
        .topbar-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-badge {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ec4899, #f59e0b);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }
        
        /* Main Content Grid - 3 Column Layout */
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
        
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        /* Sidebar */
        .sidebar {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            height: fit-content;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
            color: #cbd5e1;
        }
        
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #f8fafc;
        }
        
        .nav-item.active {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
        }
        
        .system-status {
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .system-status h4 {
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .status-ok {
            color: #10b981;
            font-weight: 500;
        }
        
        /* Main Content */
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        /* KPI Cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .kpi-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.2s;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .kpi-header {
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        
        .kpi-change {
            font-size: 14px;
            color: #10b981;
        }
        
        /* Trading Cards */
        .trading-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }
        
        .card h3 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .button-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        
        button {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(139, 92, 246, 0.8));
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
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
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 16px;
            color: #a5b4fc;
        }
        
        #results-content {
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 500px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.3);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Activity Feed */
        .activity-feed {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 24px;
        }
        
        .activity-item {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .activity-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .activity-icon {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .activity-content {
            flex: 1;
        }
        
        .activity-time {
            font-size: 12px;
            opacity: 0.6;
        }
        
        .activity-text {
            font-size: 14px;
            margin-top: 4px;
        }
        
        /* Table Styles */
        .trades-table {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 24px;
        }
        
        .trades-table table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .trades-table th,
        .trades-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .trades-table th {
            font-size: 12px;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .trade-buy {
            color: #10b981;
        }
        
        .trade-sell {
            color: #ef4444;
        }
        
        .trade-profit {
            color: #10b981;
        }
        
        .trade-loss {
            color: #ef4444;
        }
        
        /* Pulse animation */
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.6);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.8);
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Topbar -->
        <div class="topbar">
            <div class="brand">
                <div class="brand-icon">
                    <i class="fas fa-bolt" style="color: white; font-size: 20px;"></i>
                </div>
                <div class="brand-text">
                    <h1>Aurora Trading Control</h1>
                    <p>Enterprise Dashboard · Real-time Insights · Secure</p>
                </div>
            </div>
            
            <div class="topbar-actions">
                <div class="status-badge pulse">
                    <i class="fas fa-circle" style="font-size: 8px; margin-right: 6px;"></i>
                    Live: Connected
                </div>
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
                    <div class="kpi-card">
                        <div class="kpi-header">Median Latency</div>
                        <div class="kpi-value">42ms</div>
                        <div class="kpi-change">-8.6%</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-header">System Uptime</div>
                        <div class="kpi-value">99.993%</div>
                        <div class="kpi-change">✅</div>
                    </div>
                </div>
                
                <!-- Trading Cards -->
                <div class="trading-cards">
                    <div class="card">
                        <h3><i class="fas fa-chart-bar"></i> Market Analysis</h3>
                        <div class="button-grid">
                            <button onclick="testAnalysis()">Market Analysis</button>
                            <button onclick="testLiquidationMap()">Liquidation Map</button>
                            <button onclick="testTradingSetup()">Trading Setup</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3><i class="fas fa-brain"></i> JAX Neural Networks</h3>
                        <div class="button-grid">
                            <button onclick="testJaxTraining()">JAX Training</button>
                            <button onclick="testJaxPrediction()">JAX Prediction</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3><i class="fas fa-chart-line"></i> Backtesting Engine</h3>
                        <div class="button-grid">
                            <button onclick="testBacktest()">Run Backtest</button>
                            <button onclick="testMultiAsset()">Multi-Asset Analysis</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Activity Feed -->
        <div class="activity-feed">
            <h3 style="margin-bottom: 16px; color: #a5b4fc;">Recent Activity</h3>
            <div class="activity-item">
                <div class="activity-icon" style="background: rgba(99, 102, 241, 0.6);">A</div>
                <div class="activity-content">
                    <div class="activity-time">Model retrained</div>
                    <div class="activity-text">Retrained "alpha-v2" on 500k samples · 12m ago</div>
                </div>
            </div>
            <div class="activity-item">
                <div class="activity-icon" style="background: rgba(16, 185, 129, 0.6);">S</div>
                <div class="activity-content">
                    <div class="activity-time">Worker scaled</div>
                    <div class="activity-text">Scaled down 2 workers due to low load · 45m ago</div>
                </div>
            </div>
            <div class="activity-item">
                <div class="activity-icon" style="background: rgba(239, 68, 68, 0.6);">E</div>
                <div class="activity-content">
                    <div class="activity-time">API latency spike</div>
                    <div class="activity-text">95th percentile latency 210ms · 2h ago</div>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades Table -->
        <div class="trades-table">
            <h3 style="margin-bottom: 16px; color: #a5b4fc;">
                <i class="fas fa-exchange-alt"></i> Recent Trades (Live)
            </h3>
            <table>
                <thead>
                    <tr>
                        <th>Pair</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="recent-trades-body">
                    <tr>
                        <td colspan="5" style="text-align: center; color: #a5b4fc;">
                            <i class="fas fa-spinner fa-spin"></i> Loading real trades...
                        </td>
                    </tr>
                </tbody>
            </table>
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
        // Trading API Functions - Enhanced for real data display
        function formatAnalysisResults(data) {
            const tbody = document.getElementById('analysis-results-body');
            tbody.innerHTML = '';
            
            if (data.success && data.market_analysis) {
                const analysis = data.market_analysis;
                
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
            makeAPICall('/api/liquidation_map', { symbol: 'BTCUSDT' });
        }
        
        function testJaxTraining() {
            makeAPICall('/api/jax_train', { symbol: 'BTCUSDT', interval: '1h', epochs: 10 });
        }
        
        function testTradingSetup() {
            makeAPICall('/api/trading_setup', { symbol: 'BTCUSDT', timeframe: '1h' });
        }
        
        function testBacktest() {
            makeAPICall('/api/backtest', { symbol: 'BTCUSDT', start_date: '2024-01-01' });
        }
        
        function testMultiAsset() {
            makeAPICall('/api/multi_asset', { symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'] });
        }
        
        function testJaxPrediction() {
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
                    result.trades.forEach(trade => {
                        const row = document.createElement('tr');
                        const sideClass = trade.isBuyerMaker ? 'trade-sell' : 'trade-buy';
                        const side = trade.isBuyerMaker ? 'SELL' : 'BUY';
                        
                        row.innerHTML = `
                            <td>${trade.symbol}</td>
                            <td class="${sideClass}">${side}</td>
                            <td>$${parseFloat(trade.price).toFixed(2)}</td>
                            <td>${parseFloat(trade.qty).toFixed(4)}</td>
                            <td>${new Date(trade.time).toLocaleTimeString()}</td>
                        `;
                        tbody.appendChild(row);
                    });
                } else {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="5" style="text-align: center; color: #ef4444;">
                                Unable to load recent trades
                            </td>
                        </tr>
                    `;
                }
            } catch (error) {
                const tbody = document.getElementById('recent-trades-body');
                tbody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: #ef4444;">
                            Error loading trades: ${error.message}
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
