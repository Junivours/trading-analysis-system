#!/bin/bash
# 🚀 JAX AI Trading System - Quick Setup & Deploy Script

echo "🔥 JAX AI Trading System v4.0 Setup"
echo "====================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git remote add origin https://github.com/Junivours/trading-analysis-system.git
fi

# Add all files
echo "📁 Adding files to Git..."
git add .

# Commit with timestamp
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo "💾 Committing changes..."
git commit -m "🚀 JAX AI Trading System v4.0 - Deploy Ready ($timestamp)"

# Push to main
echo "🌐 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Repository updated successfully!"
echo ""
echo "🚀 Next Steps:"
echo "1. Go to Railway: https://railway.app"
echo "2. Connect your GitHub repo"
echo "3. Deploy automatically!"
echo ""
echo "🎯 Your JAX AI Trading System is ready for deployment!"
echo "Features: Neural Networks + Multi-TF Chart Patterns + Real Data"
