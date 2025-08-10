from flask import Flask, jsonify, render_template_string, request
import requests
import numpy as np
import traceback
import os
import time
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 🚀 NEUE OPTIMIERUNGEN - Cache und Status Management
try:
    from utils.cache_manager import cache_manager, api_optimizer, get_cache_status
    from utils.status_manager import status_manager, weight_manager, get_system_dashboard, SystemStatus, DataSource
    OPTIMIZATION_AVAILABLE = True
    print("🚀 System-Optimierungen geladen: Cache + Status Management")
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ System-Optimierungen nicht verfügbar")

# 🤖 JAX Neural Network Dependencies (safe import)
try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    from jax.scipy.special import logsumexp
    JAX_AVAILABLE = True
    print("✅ JAX Neural Networks initialized successfully")
    
    # 📊 JAX-Status aktualisieren
    if OPTIMIZATION_AVAILABLE:
        status_manager.update_component_status('jax_ml', SystemStatus.ONLINE)
        
except ImportError as e:
    print(f"⚠️ JAX not available: {e}. Install with: pip install jax flax")
    JAX_AVAILABLE = False
    
    # JAX-Status aktualisieren
    if OPTIMIZATION_AVAILABLE:
        status_manager.update_component_status('jax_ml', SystemStatus.OFFLINE, str(e))
    
    # Create dummy jax/jnp for fallback
    class DummyJAX:
        @staticmethod
        def array(x): return np.array(x)
        random = type('random', (), {'PRNGKey': lambda x: x, 'normal': lambda *args: np.random.normal(0, 0.1, args[-1])})()
    jax = jnp = DummyJAX()

# ========================================================================================
# 🚀 ULTIMATE TRADING V3 - PROFESSIONAL AI-POWERED TRADING SYSTEM
# ========================================================================================

# ⚡ ADVANCED FEATURE IMPORTS - Railway Optimized Conditional Loading  
BACKTESTING_AVAILABLE = False
try:
    from core.backtesting_engine import AdvancedBacktestingEngine
    BACKTESTING_AVAILABLE = True
    print("🎯 Advanced Backtesting Engine loaded")
    if OPTIMIZATION_AVAILABLE:
        status_manager.update_component_status('backtesting', SystemStatus.ONLINE)
except ImportError as e:
    print(f"⚠️ Backtesting Engine nicht verfügbar: {e}")
    if OPTIMIZATION_AVAILABLE:
        status_manager.update_component_status('backtesting', SystemStatus.OFFLINE, str(e))

# ========================================================================================
# 🛠️ MODULAR IMPORTS - Separate Dateien für bessere Organisation
# ========================================================================================

# Import separate modules from organized folders
from core.binance_api import OptimizedBinanceAPI
from core.jax_neural import JAXNeuralEngine
from analysis.market_analysis import MarketAnalysisEngine
from analysis.chart_patterns import ChartPatternAnalyzer
from analysis.liquidation_maps import LiquidationMapAnalyzer  
from analysis.trading_setups import TradingSetupAnalyzer

# ========================================================================================
# 🎯 MAIN FLASK APPLICATION - Clean and Organized  
# ========================================================================================

app = Flask(__name__)

# 🚀 OPTIMIERTE API INITIALISIERUNG
binance_api = OptimizedBinanceAPI()

# Initialize JAX Neural Engine
jax_engine = JAXNeuralEngine()

# Initialize Advanced Features - Railway Optimized  
backtest_engine = None

if BACKTESTING_AVAILABLE:
    try:
        backtest_engine = AdvancedBacktestingEngine()
        print("✅ Backtesting Engine initialized")
    except Exception as e:
        print(f"❌ Backtesting Engine initialization failed: {e}")
        BACKTESTING_AVAILABLE = False

print(f"🔬 Advanced engines status: Backtesting={backtest_engine is not None}, JAX={jax_engine is not None}")

# 📊 Cache-System initialisieren (falls verfügbar)
if OPTIMIZATION_AVAILABLE:
    status_manager.update_component_status('cache_system', SystemStatus.ONLINE)
    print("🧠 Cache-System aktiv")
    
    # Periodische Cache-Bereinigung starten
    import threading
    def cleanup_cache():
        while True:
            time.sleep(300)  # Alle 5 Minuten
            cleaned = cache_manager.cleanup_expired()
            if cleaned > 0:
                print(f"🧹 {cleaned} abgelaufene Cache-Einträge bereinigt")
    
    cleanup_thread = threading.Thread(target=cleanup_cache, daemon=True)
    cleanup_thread.start()

# ========================================================================================
# 🎯 FUNDAMENTAL ANALYSIS ENGINE - Professional Trading Analysis
# ========================================================================================

class FundamentalAnalysisEngine:
    """🎯 Professional Fundamental Analysis - 70% Weight in Trading Decisions"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self._last_request_time = 0
        self.analysis_weights = {
            'volume_analysis': 0.30,
            'price_momentum': 0.25, 
            'orderbook_depth': 0.20,
            'trade_frequency': 0.15,
            'volatility_score': 0.10
        }
        
    def analyze_fundamentals(self, symbol: str) -> dict:
        """📊 Comprehensive fundamental analysis"""
        try:
            # Get market data
            ticker_data = binance_api.get_ticker_24hr(symbol)
            orderbook_data = binance_api.get_orderbook(symbol, 100)
            
            # Perform analysis
            analysis = self._perform_analysis(symbol, ticker_data, orderbook_data)
            
            return {
                'symbol': symbol,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
    
    def _perform_analysis(self, symbol: str, ticker_data: dict, orderbook_data: dict) -> dict:
        """📈 Core analysis logic"""
        
        # Volume Analysis (30%)
        volume_score = self._analyze_volume(ticker_data)
        
        # Price Momentum (25%)
        momentum_score = self._analyze_momentum(ticker_data)
        
        # Orderbook Depth (20%)
        depth_score = self._analyze_orderbook_depth(orderbook_data)
        
        # Trade Frequency (15%)
        frequency_score = self._analyze_trade_frequency(ticker_data)
        
        # Volatility (10%)
        volatility_score = self._analyze_volatility(ticker_data)
        
        # Calculate weighted score
        total_score = (
            volume_score * self.analysis_weights['volume_analysis'] +
            momentum_score * self.analysis_weights['price_momentum'] +
            depth_score * self.analysis_weights['orderbook_depth'] +
            frequency_score * self.analysis_weights['trade_frequency'] +
            volatility_score * self.analysis_weights['volatility_score']
        )
        
        return {
            'total_score': round(total_score, 2),
            'components': {
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'depth_score': depth_score,
                'frequency_score': frequency_score,
                'volatility_score': volatility_score
            },
            'recommendation': self._get_recommendation(total_score)
        }
    
    def _analyze_volume(self, ticker_data: dict) -> float:
        """📊 Volume analysis"""
        try:
            volume = float(ticker_data.get('volume', 0))
            if volume > 1000000:
                return 85.0
            elif volume > 100000:
                return 70.0
            else:
                return 50.0
        except:
            return 50.0
    
    def _analyze_momentum(self, ticker_data: dict) -> float:
        """📈 Price momentum analysis"""
        try:
            price_change = float(ticker_data.get('priceChangePercent', 0))
            if price_change > 5:
                return 90.0
            elif price_change > 0:
                return 70.0
            elif price_change > -5:
                return 50.0
            else:
                return 30.0
        except:
            return 50.0
    
    def _analyze_orderbook_depth(self, orderbook_data: dict) -> float:
        """📋 Orderbook depth analysis"""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if len(bids) > 50 and len(asks) > 50:
                return 80.0
            elif len(bids) > 20 and len(asks) > 20:
                return 65.0
            else:
                return 45.0
        except:
            return 50.0
    
    def _analyze_trade_frequency(self, ticker_data: dict) -> float:
        """💱 Trade frequency analysis"""
        try:
            count = int(ticker_data.get('count', 0))
            if count > 10000:
                return 85.0
            elif count > 1000:
                return 65.0
            else:
                return 45.0
        except:
            return 50.0
    
    def _analyze_volatility(self, ticker_data: dict) -> float:
        """📊 Volatility analysis"""
        try:
            high = float(ticker_data.get('highPrice', 0))
            low = float(ticker_data.get('lowPrice', 0))
            
            if high > 0 and low > 0:
                volatility = ((high - low) / low) * 100
                if 2 <= volatility <= 8:
                    return 75.0
                elif volatility < 2:
                    return 50.0
                else:
                    return 60.0
            return 50.0
        except:
            return 50.0
    
    def _get_recommendation(self, score: float) -> str:
        """🎯 Get trading recommendation"""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 50:
            return "NEUTRAL"
        elif score >= 35:
            return "SELL"
        else:
            return "STRONG_SELL"

# Initialize Fundamental Analysis Engine
fundamental_engine = FundamentalAnalysisEngine()

# ========================================================================================
# 🌐 FLASK ROUTES - API Endpoints
# ========================================================================================

@app.route('/favicon.ico')
def favicon():
    return '', 204

# Health check removed - using main app.py health endpoint

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
