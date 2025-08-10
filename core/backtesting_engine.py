"""
🔬 PROFESSIONAL BACKTESTING ENGINE
====================================
- Multi-timeframe backtesting (1h, 4h, 1d)
- Performance metrics (Win Rate, Sharpe Ratio, Max Drawdown)
- Monte Carlo simulation
- Walk-forward analysis
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple
import statistics

class AdvancedBacktestingEngine:
    """🔬 Professional backtesting with advanced metrics"""
    
    def __init__(self):
        self.results = []
        self.trades = []
        self.portfolio_value = []
        self.drawdowns = []
        
    def fetch_historical_data(self, symbol: str, interval: str, lookback_days: int = 365) -> List[Dict]:
        """Fetch historical data from Binance"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            
            # Calculate timestamps
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            candles = []
            for candle in data:
                candles.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            print(f"📈 Fetched {len(candles)} candles for {symbol} ({interval})")
            return candles
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return []
    
    def calculate_technical_indicators(self, candles: List[Dict]) -> Dict:
        """Calculate technical indicators for backtesting"""
        if len(candles) < 50:
            return {}
            
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        volumes = [c['volume'] for c in candles]
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        # MACD calculation
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            exp1 = pd.Series(prices).ewm(span=fast).mean()
            exp2 = pd.Series(prices).ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            return macd_line.iloc[-1], signal_line.iloc[-1]
        
        # EMAs
        ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
        
        # RSI
        rsi = calculate_rsi(closes)
        
        # MACD
        macd, macd_signal = calculate_macd(closes)
        
        # Volatility
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * 100
        
        return {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'volatility': volatility,
            'current_price': closes[-1],
            'volume_avg': np.mean(volumes[-20:])
        }
    
    def generate_signal(self, indicators: Dict) -> Tuple[str, float]:
        """Generate trading signal using our confluence logic"""
        if not indicators:
            return "HOLD", 50
            
        # Extract indicators
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        ema_12 = indicators.get('ema_12', 0)
        ema_26 = indicators.get('ema_26', 0)
        current_price = indicators.get('current_price', 0)
        
        # Trend analysis
        trend_bullish = current_price > ema_12 > ema_26
        trend_bearish = current_price < ema_12 < ema_26
        
        # Signal scoring
        buy_score = 0
        sell_score = 0
        confidence = 50
        
        # RSI signals
        if rsi < 30:
            buy_score += 2
            confidence += 15
        elif rsi > 70:
            sell_score += 2
            confidence += 15
        
        # MACD signals
        if macd > 0 and trend_bullish:
            buy_score += 2
            confidence += 10
        elif macd < 0 and trend_bearish:
            sell_score += 2
            confidence += 10
        
        # Trend signals
        if trend_bullish:
            buy_score += 1
        elif trend_bearish:
            sell_score += 1
        
        # Confluence bonus
        confluence_count = 0
        if trend_bullish: confluence_count += 1
        if 30 <= rsi <= 70: confluence_count += 1
        if macd > 0: confluence_count += 1
        
        if confluence_count >= 2:
            if buy_score > sell_score:
                buy_score += 1
                confidence += 10
            elif sell_score > buy_score:
                sell_score += 1
                confidence += 10
        
        # Decision logic
        if buy_score > sell_score and rsi < 75:
            return "BUY", min(95, confidence)
        elif sell_score > buy_score and rsi > 25:
            return "SELL", min(95, confidence)
        else:
            return "HOLD", max(30, confidence - 20)
    
    def run_backtest(self, symbol: str, interval: str, initial_capital: float = 10000, 
                    lookback_days: int = 365, stop_loss: float = 0.05, 
                    take_profit: float = 0.10) -> Dict:
        """Run comprehensive backtest"""
        
        print(f"🔬 Starting backtest for {symbol} ({interval})")
        print(f"📊 Capital: ${initial_capital:,.2f} | Stop Loss: {stop_loss*100}% | Take Profit: {take_profit*100}%")
        
        # Fetch data
        candles = self.fetch_historical_data(symbol, interval, lookback_days)
        if len(candles) < 100:
            return {"error": "Insufficient data"}
        
        # Initialize
        capital = initial_capital
        position = None
        position_size = 0
        entry_price = 0
        trades = []
        portfolio_values = [initial_capital]
        
        # Walk through data
        for i in range(50, len(candles)):  # Start from index 50 for indicators
            current_candles = candles[:i+1]
            indicators = self.calculate_technical_indicators(current_candles)
            
            if not indicators:
                continue
                
            signal, confidence = self.generate_signal(indicators)
            current_price = indicators['current_price']
            timestamp = candles[i]['timestamp']
            
            # Handle existing position
            if position:
                # Check stop loss / take profit
                if position == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -stop_loss or pnl_pct >= take_profit or signal == "SELL":
                        # Close long position
                        exit_value = position_size * current_price
                        pnl = exit_value - (position_size * entry_price)
                        capital += exit_value
                        
                        trades.append({
                            'type': 'LONG',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'entry_time': entry_timestamp,
                            'exit_time': timestamp,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'reason': 'Stop Loss' if pnl_pct <= -stop_loss else 'Take Profit' if pnl_pct >= take_profit else 'Signal'
                        })
                        
                        position = None
                        position_size = 0
                        
                elif position == "SHORT":
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -stop_loss or pnl_pct >= take_profit or signal == "BUY":
                        # Close short position
                        pnl = position_size * (entry_price - current_price)
                        capital += position_size * entry_price + pnl
                        
                        trades.append({
                            'type': 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'entry_time': entry_timestamp,
                            'exit_time': timestamp,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'reason': 'Stop Loss' if pnl_pct <= -stop_loss else 'Take Profit' if pnl_pct >= take_profit else 'Signal'
                        })
                        
                        position = None
                        position_size = 0
            
            # Open new position
            if not position and confidence > 60:
                if signal == "BUY":
                    position = "LONG"
                    entry_price = current_price
                    entry_timestamp = timestamp
                    position_size = capital * 0.95 / current_price  # Use 95% of capital
                    capital *= 0.05  # Keep 5% as cash
                    
                elif signal == "SELL":
                    position = "SHORT"
                    entry_price = current_price
                    entry_timestamp = timestamp
                    position_size = capital * 0.95 / current_price
                    capital *= 0.05
            
            # Track portfolio value
            current_portfolio_value = capital
            if position and position_size > 0:
                if position == "LONG":
                    current_portfolio_value += position_size * current_price
                else:  # SHORT
                    current_portfolio_value += position_size * entry_price + position_size * (entry_price - current_price)
            
            portfolio_values.append(current_portfolio_value)
        
        # Calculate performance metrics
        return self.calculate_performance_metrics(trades, portfolio_values, initial_capital)
    
    def calculate_performance_metrics(self, trades: List[Dict], portfolio_values: List[float], 
                                    initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {
                "error": "No trades executed",
                "initial_capital": initial_capital,
                "final_capital": portfolio_values[-1] if portfolio_values else initial_capital
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        final_capital = portfolio_values[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # PnL metrics
        total_pnl = sum([t['pnl'] for t in trades])
        avg_win = statistics.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = statistics.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum([t['pnl'] for t in winning_trades])
        gross_loss = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        peak = initial_capital
        max_drawdown = 0
        drawdowns = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (simplified)
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        avg_return = statistics.mean(returns) if returns else 0
        return_std = statistics.stdev(returns) if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / return_std) * np.sqrt(252) if return_std > 0 else 0
        
        return {
            "summary": {
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "total_return_pct": total_return,
                "total_pnl": total_pnl
            },
            "trade_metrics": {
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate_pct": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor
            },
            "risk_metrics": {
                "max_drawdown_pct": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "volatility": return_std * np.sqrt(252) * 100 if return_std > 0 else 0
            },
            "trades": trades,
            "portfolio_curve": portfolio_values,
            "drawdown_curve": drawdowns
        }
    
    def monte_carlo_simulation(self, symbol: str, interval: str, simulations: int = 100) -> Dict:
        """Run Monte Carlo simulation for robustness testing"""
        
        print(f"🎲 Running Monte Carlo simulation ({simulations} runs)")
        
        results = []
        for i in range(simulations):
            # Add random noise to test robustness
            lookback_days = np.random.randint(200, 500)
            stop_loss = np.random.uniform(0.03, 0.08)
            take_profit = np.random.uniform(0.08, 0.15)
            
            result = self.run_backtest(symbol, interval, 10000, lookback_days, stop_loss, take_profit)
            
            if 'error' not in result:
                results.append(result['summary']['total_return_pct'])
            
            if (i + 1) % 10 == 0:
                print(f"🔄 Completed {i + 1}/{simulations} simulations")
        
        if not results:
            return {"error": "No successful simulations"}
        
        return {
            "simulations": len(results),
            "avg_return": statistics.mean(results),
            "median_return": statistics.median(results),
            "best_return": max(results),
            "worst_return": min(results),
            "std_deviation": statistics.stdev(results),
            "success_rate": len([r for r in results if r > 0]) / len(results) * 100,
            "all_returns": results
        }

if __name__ == "__main__":
    # Test the backtesting engine
    engine = AdvancedBacktestingEngine()
    
    print("🔬 BACKTESTING ENGINE TEST")
    print("=" * 50)
    
    # Single backtest
    result = engine.run_backtest("BTCUSDT", "1h", 10000, 180)  # 6 months
    
    if 'error' not in result:
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"💰 Initial Capital: ${result['summary']['initial_capital']:,.2f}")
        print(f"💰 Final Capital: ${result['summary']['final_capital']:,.2f}")
        print(f"📈 Total Return: {result['summary']['total_return_pct']:.2f}%")
        print(f"🎯 Win Rate: {result['trade_metrics']['win_rate_pct']:.1f}%")
        print(f"📊 Total Trades: {result['trade_metrics']['total_trades']}")
        print(f"💪 Profit Factor: {result['trade_metrics']['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {result['risk_metrics']['max_drawdown_pct']:.2f}%")
        print(f"📊 Sharpe Ratio: {result['risk_metrics']['sharpe_ratio']:.2f}")
    else:
        print(f"❌ Error: {result['error']}")
