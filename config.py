"""
🚀 PRODUCTION CONFIGURATION FOR RAILWAY DEPLOYMENT
==================================================
Optimized Flask configuration for Railway cloud deployment
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ultimate-trading-system-2025'
    
    # Flask settings
    DEBUG = False
    TESTING = False
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year cache for static files
    
    # ML/AI settings
    TF_CPP_MIN_LOG_LEVEL = '2'  # Reduce TensorFlow logging
    PYTHONUNBUFFERED = '1'
    
    # Trading system settings
    CACHE_ENABLED = True
    MAX_WORKERS = 1  # Railway limitation
    REQUEST_TIMEOUT = 300
    
    # API settings
    BINANCE_API_BASE = 'https://api.binance.com/api/v3'
    
class ProductionConfig(Config):
    """Production configuration for Railway"""
    DEBUG = False
    
    # Enhanced security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance optimizations
    CACHE_EXPIRY = {
        'price_data': 15,      # 15 seconds
        'indicators': 60,       # 1 minute
        'market_data': 30,      # 30 seconds
        'ml_predictions': 120,  # 2 minutes
        'backtest': 300,        # 5 minutes
        'klines': 45           # 45 seconds
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
