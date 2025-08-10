#!/bin/bash
# 🚀 AUTOMATIC RAILWAY DEPLOYMENT SCRIPT

echo "🚀 Starting Railway deployment for Trading Intelligence System..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project root."
    exit 1
fi

# Add all changes
echo "📦 Adding changes to git..."
git add .

# Create commit with timestamp
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo "💾 Creating commit..."
git commit -m "🚀 Railway deployment update - $timestamp"

# Push to GitHub (triggers Railway deployment)
echo "⬆️ Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "🌐 Your app will be available at:"
echo "   https://[your-project-name].railway.app"
echo ""
echo "📊 To monitor deployment:"
echo "   1. Go to railway.app"
echo "   2. Select your project"
echo "   3. Check the deployment logs"
echo ""
echo "⏱️ Deployment usually takes 2-3 minutes"
echo "🎯 Features available after deployment:"
echo "   • 🧠 AI-powered market analysis"
echo "   • 📊 Real-time trading signals"
echo "   • 🎯 Advanced pattern recognition"
echo "   • ⚡ Dynamic coin search & analysis"
echo "   • 📈 Multi-timeframe backtesting"
echo "   • 🔥 Liquidation risk mapping"
echo ""
echo "🚀 Happy Trading!"
