"""
Utility functions for data validation and sanitization
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def validate_price_data(price_data: List[Dict[str, Any]]) -> bool:
    """Validate price data format and content"""
    if not price_data or not isinstance(price_data, list):
        return False
    
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    
    for candle in price_data:
        if not isinstance(candle, dict):
            return False
        
        for field in required_fields:
            if field not in candle:
                return False
            
            try:
                value = float(candle[field])
                if value < 0 or not np.isfinite(value):
                    return False
            except (ValueError, TypeError):
                return False
    
    return True

def sanitize_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize indicator values"""
    sanitized = {}
    
    for key, value in indicators.items():
        if isinstance(value, (list, np.ndarray)):
            # Clean array values
            clean_array = np.nan_to_num(np.array(value), nan=0.0, posinf=0.0, neginf=0.0)
            sanitized[key] = clean_array.tolist()
        elif isinstance(value, (int, float, np.number)):
            # Clean single values
            if np.isnan(value) or np.isinf(value):
                sanitized[key] = 0.0
            else:
                sanitized[key] = float(value)
        else:
            sanitized[key] = value
    
    return sanitized

def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation for crypto pairs
    symbol = symbol.upper()
    if len(symbol) < 6 or len(symbol) > 12:
        return False
    
    # Must end with common quote currencies
    quote_currencies = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD', 'USD']
    return any(symbol.endswith(quote) for quote in quote_currencies)
