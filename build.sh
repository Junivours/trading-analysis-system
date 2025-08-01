#!/bin/bash
#!/bin/bash
# 🛠️ Railway Build Script - Nix-Compatible Python Environment Setup
# Handles NixOS-specific Python/pip installation challenges

set -e  # Exit on any error

echo "� Starting Railway build process..."
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version check:"

# === PHASE 1: Python Environment Detection ===
PYTHON_CMD=""
PIP_CMD=""

# Find Python executable in Nix environment
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Found python3: $(which python3)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Found python: $(which python)"
else
    echo "❌ No Python found in PATH"
    # Try Nix store paths
    for python_path in /nix/store/*/bin/python3 /nix/store/*/bin/python; do
        if [ -x "$python_path" ]; then
            PYTHON_CMD="$python_path"
            echo "✅ Found Python in Nix store: $python_path"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ FATAL: No Python interpreter found!"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# === PHASE 2: Pip Detection and Setup ===
echo "📦 Setting up pip..."

# Method 1: Try python -m pip
if $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
    echo "✅ pip available via python module: $PIP_CMD"
# Method 2: Try direct pip command
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
    echo "✅ pip available directly: $(which pip)"
elif command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
    echo "✅ pip3 available: $(which pip3)"
# Method 3: Try Nix store paths
else
    echo "⚠️  pip not found in standard locations, searching Nix store..."
    for pip_path in /nix/store/*/bin/pip /nix/store/*/bin/pip3; do
        if [ -x "$pip_path" ]; then
            PIP_CMD="$pip_path"
            echo "✅ Found pip in Nix store: $pip_path"
            break
        fi
    done
fi

# === PHASE 3: Pip Installation Strategy ===
if [ -n "$PIP_CMD" ]; then
    echo "📦 Upgrading pip and essential packages..."
    
    # Upgrade core packages
    $PIP_CMD install --upgrade pip setuptools wheel --no-cache-dir || {
        echo "⚠️  Standard pip upgrade failed, trying alternative method..."
        $PYTHON_CMD -c "
import subprocess
import sys
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', '--no-cache-dir'])
    print('✅ Alternative pip upgrade successful')
except subprocess.CalledProcessError as e:
    print(f'⚠️  Alternative pip upgrade also failed: {e}')
    print('Continuing without upgrade...')
"
    }
    
    # Install requirements
    echo "📋 Installing requirements..."
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt --no-cache-dir || {
            echo "⚠️  Standard requirements installation failed, trying fallback..."
            
            # Try minimal requirements
            if [ -f "requirements-minimal.txt" ]; then
                echo "📋 Trying minimal requirements..."
                $PIP_CMD install -r requirements-minimal.txt --no-cache-dir || {
                    echo "⚠️  Minimal requirements failed, installing core packages individually..."
                    
                    # Install essential packages one by one
                    for package in flask flask-cors requests pandas numpy; do
                        echo "📦 Installing $package..."
                        $PIP_CMD install "$package" --no-cache-dir || echo "⚠️  Failed to install $package"
                    done
                }
            fi
        }
    else
        echo "⚠️  No requirements.txt found"
    fi
    
else
    echo "❌ CRITICAL: No pip installation method found!"
    echo "🔄 Attempting manual package installation..."
    
    # Try to install Flask directly with Python
    $PYTHON_CMD -c "
import sys
print('Attempting to verify Python packages...')
try:
    import flask
    print('✅ Flask is available')
except ImportError:
    print('❌ Flask not available')
    print('This may cause runtime issues.')

try:
    import requests
    print('✅ Requests is available')
except ImportError:
    print('❌ Requests not available')

try:
    import pandas
    print('✅ Pandas is available')
except ImportError:
    print('❌ Pandas not available')
    
print('Python path:', sys.path)
"
fi

# === PHASE 4: Validation ===
echo "🧪 Validating installation..."
$PYTHON_CMD -c "
import sys
print(f'✅ Python {sys.version}')
print(f'📍 Python executable: {sys.executable}')
print(f'📚 Python path: {sys.path}')

# Test critical imports
critical_packages = ['flask', 'requests']
for package in critical_packages:
    try:
        __import__(package)
        print(f'✅ {package} imported successfully')
    except ImportError as e:
        print(f'❌ {package} import failed: {e}')
        
print('🔍 Checking if app.py can be imported...')
try:
    import os
    if os.path.exists('app.py'):
        print('✅ app.py file exists')
    else:
        print('❌ app.py file not found!')
except Exception as e:
    print(f'❌ Error checking app.py: {e}')
"

echo "✅ Build script completed!"
echo "🚀 Ready for Railway deployment..."
