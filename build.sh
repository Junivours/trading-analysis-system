#!/bin/bash
# Railway Build Script für ULTIMATE Trading Analysis Pro

echo "🔥 ULTIMATE Trading Analysis Pro - Railway Build"
echo "✅ Starting installation..."

# Check Python
python --version
which python

# Check pip availability  
if command -v pip &> /dev/null; then
    echo "✅ pip found"
    pip --version
    pip install --upgrade pip
    pip install -r requirements.txt
elif python -m pip --version &> /dev/null; then
    echo "✅ pip module found"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "❌ pip not found, trying minimal installation"
    # Try to install packages individually
    python -c "
import subprocess
import sys

packages = ['flask>=2.0.0', 'flask-cors>=3.0.0', 'pandas>=1.5.0', 'numpy>=1.20.0', 'requests>=2.25.0']
for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f'✅ Installed {package}')
    except:
        print(f'❌ Failed to install {package}')
"
fi

echo "✅ Installation complete"
