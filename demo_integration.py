#!/usr/bin/env python3
"""
🚀 TRADING ANALYSIS SYSTEM - INTEGRATION EXAMPLE
Demonstrates TradingView-compatible RSI and multi-coin analysis features
"""

import requests
import json
import time
from datetime import datetime

class TradingAnalysisDemo:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        print("🚀 Trading Analysis System Demo")
        print("=" * 50)
    
    def get_available_coins(self):
        """Get list of available trading pairs"""
        try:
            response = requests.get(f"{self.base_url}/api/get_coin_list")
            data = response.json()
            
            if data['success']:
                print(f"✅ Found {data['total_coins']} available trading pairs")
                return [coin['symbol'] for coin in data['coins'][:10]]  # Top 10
            else:
                print(f"❌ Error getting coins: {data.get('error', 'Unknown error')}")
                return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Fallback
                
        except Exception as e:
            print(f"❌ Network error: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Fallback
    
    def analyze_rsi_single_coin(self, symbol="BTCUSDT"):
        """Demonstrate TradingView-compatible RSI analysis"""
        print(f"\n📈 RSI Analysis for {symbol}")
        print("-" * 30)
        
        try:
            response = requests.post(f"{self.base_url}/api/rsi_analysis", json={
                "symbol": symbol,
                "timeframes": ["1h", "4h", "1d"],
                "rsi_periods": [14, 21]
            })
            
            data = response.json()
            
            if data['success']:
                for timeframe, analysis in data['analysis'].items():
                    print(f"\n🕐 {timeframe} Timeframe:")
                    
                    for rsi_key, rsi_data in analysis.items():
                        if rsi_key.startswith('rsi_'):
                            period = rsi_data['period']
                            value = rsi_data['value']
                            signal = rsi_data['signal']
                            level = rsi_data['level']
                            divergence = rsi_data['divergence']
                            
                            print(f"   RSI-{period}: {value:.2f}")
                            print(f"   Signal: {signal}")
                            print(f"   Level: {level}")
                            
                            if divergence['divergence'] != 'None':
                                print(f"   🚨 Divergence: {divergence['divergence']} (confidence: {divergence['confidence']}%)")
                        
                        elif rsi_key == 'price_context':
                            price_info = rsi_data
                            print(f"   Current Price: ${price_info['current_price']:,.2f}")
                            print(f"   Price Change: {price_info['price_change_percent']:+.2f}%")
            else:
                print(f"❌ RSI Analysis failed: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def analyze_multi_coin(self, symbols=None):
        """Demonstrate multi-coin analysis with RSI"""
        if symbols is None:
            symbols = self.get_available_coins()[:5]  # Top 5 coins
        
        print(f"\n🌐 Multi-Coin Analysis")
        print("-" * 30)
        print(f"Analyzing: {', '.join(symbols)}")
        
        try:
            response = requests.post(f"{self.base_url}/api/enhanced_multi_coin", json={
                "symbols": symbols,
                "timeframes": ["1h", "4h"],
                "analysis_type": "rsi"
            })
            
            data = response.json()
            
            if data['success']:
                print(f"\n📊 Analysis Summary:")
                summary = data['summary']
                print(f"   Coins analyzed: {summary['total_coins_analyzed']}")
                print(f"   Timeframes: {summary['total_timeframes']}")
                
                if summary['top_performers']:
                    print(f"\n🚀 Top Performers (24h):")
                    for performer in summary['top_performers']:
                        print(f"   {performer['symbol']}: {performer['change_24h']:+.2f}%")
                
                if summary['bottom_performers']:
                    print(f"\n📉 Bottom Performers (24h):")
                    for performer in summary['bottom_performers']:
                        print(f"   {performer['symbol']}: {performer['change_24h']:+.2f}%")
                
                # Show detailed RSI analysis for each coin
                print(f"\n📈 RSI Analysis Results:")
                for symbol, timeframes in data['results'].items():
                    if 'error' not in timeframes:
                        print(f"\n{symbol}:")
                        
                        for tf, analysis in timeframes.items():
                            if 'rsi' in analysis and 'rsi_analysis' in analysis['rsi']:
                                rsi_data = analysis['rsi']['rsi_analysis']['rsi_14']
                                price_info = analysis['price_info']
                                
                                print(f"   {tf}: RSI-14: {rsi_data['value']:.1f} ({rsi_data['signal']}) | Price: ${price_info['current_price']:,.2f}")
                                
                                if rsi_data['value'] < 30:
                                    print(f"   🟢 {symbol} is OVERSOLD on {tf} - Potential BUY opportunity!")
                                elif rsi_data['value'] > 70:
                                    print(f"   🔴 {symbol} is OVERBOUGHT on {tf} - Potential SELL signal!")
            else:
                print(f"❌ Multi-coin analysis failed: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def find_trading_opportunities(self, symbols=None):
        """Find potential trading opportunities using RSI"""
        if symbols is None:
            symbols = self.get_available_coins()
        
        print(f"\n🎯 Trading Opportunity Scanner")
        print("-" * 30)
        
        opportunities = {
            'oversold': [],
            'overbought': [],
            'bullish_divergence': [],
            'bearish_divergence': []
        }
        
        for symbol in symbols:
            try:
                response = requests.post(f"{self.base_url}/api/rsi_analysis", json={
                    "symbol": symbol,
                    "timeframes": ["1h", "4h"],
                    "rsi_periods": [14]
                })
                
                data = response.json()
                
                if data['success']:
                    for timeframe, analysis in data['analysis'].items():
                        if 'rsi_14' in analysis:
                            rsi_data = analysis['rsi_14']
                            price_info = analysis['price_context']
                            
                            opportunity = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'rsi': rsi_data['value'],
                                'signal': rsi_data['signal'],
                                'price': price_info['current_price'],
                                'change_24h': price_info['price_change_percent']
                            }
                            
                            # Classify opportunities
                            if rsi_data['value'] < 30:
                                opportunities['oversold'].append(opportunity)
                            elif rsi_data['value'] > 70:
                                opportunities['overbought'].append(opportunity)
                            
                            # Check for divergences
                            divergence = rsi_data['divergence']
                            if divergence['divergence'] == 'Bullish' and divergence['confidence'] > 60:
                                opportunities['bullish_divergence'].append(opportunity)
                            elif divergence['divergence'] == 'Bearish' and divergence['confidence'] > 60:
                                opportunities['bearish_divergence'].append(opportunity)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error analyzing {symbol}: {e}")
                continue
        
        # Display opportunities
        print(f"\n🟢 OVERSOLD Opportunities (RSI < 30):")
        for opp in opportunities['oversold']:
            print(f"   {opp['symbol']} ({opp['timeframe']}): RSI {opp['rsi']:.1f} | Price: ${opp['price']:,.2f} | 24h: {opp['change_24h']:+.1f}%")
        
        print(f"\n🔴 OVERBOUGHT Signals (RSI > 70):")
        for opp in opportunities['overbought']:
            print(f"   {opp['symbol']} ({opp['timeframe']}): RSI {opp['rsi']:.1f} | Price: ${opp['price']:,.2f} | 24h: {opp['change_24h']:+.1f}%")
        
        print(f"\n📈 BULLISH Divergences:")
        for opp in opportunities['bullish_divergence']:
            print(f"   {opp['symbol']} ({opp['timeframe']}): RSI {opp['rsi']:.1f} | Price: ${opp['price']:,.2f}")
        
        print(f"\n📉 BEARISH Divergences:")
        for opp in opportunities['bearish_divergence']:
            print(f"   {opp['symbol']} ({opp['timeframe']}): RSI {opp['rsi']:.1f} | Price: ${opp['price']:,.2f}")
        
        # Summary
        total_opportunities = sum(len(opportunities[key]) for key in opportunities)
        print(f"\n📋 Summary: Found {total_opportunities} trading opportunities")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print(f"🕐 Demo started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Get available coins
        coins = self.get_available_coins()
        
        # 2. Analyze RSI for Bitcoin
        self.analyze_rsi_single_coin("BTCUSDT")
        
        # 3. Multi-coin analysis
        self.analyze_multi_coin(coins[:3])
        
        # 4. Find trading opportunities
        self.find_trading_opportunities(coins[:5])
        
        print(f"\n🎉 Demo completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✅ TradingView-compatible RSI calculations")
        print("   ✅ Multi-timeframe analysis (1h, 4h, 1d)")
        print("   ✅ RSI divergence detection")
        print("   ✅ Multi-coin processing")
        print("   ✅ Trading opportunity scanning")
        print("   ✅ Real-time market data integration")


if __name__ == "__main__":
    # Run the demo
    demo = TradingAnalysisDemo()
    demo.run_full_demo()