#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Railway Starter Script - Nix-Compatible Python Environment
Advanced Flask Application Launcher with Environment Validation
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_python_environment():
    """Comprehensive Python environment validation for Nix systems"""
    logger.info("🔍 Validating Python environment...")
    
    # Python version check
    python_version = sys.version_info
    logger.info(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    logger.info(f"📍 Python executable: {sys.executable}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.warning(f"⚠️  Python version {python_version.major}.{python_version.minor} may have compatibility issues")
    
    # Check Python path
    logger.info(f"📚 Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    return True

def check_critical_packages():
    """Check and report status of critical packages from Nix"""
    critical_packages = {
        'flask': 'Flask web framework',
        'requests': 'HTTP library', 
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library'
    }
    
    missing_packages = []
    
    logger.info("📦 Checking critical packages from Nix...")
    
    for package, description in critical_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✅ {package} {version} - {description} (available)")
        except ImportError as e:
            logger.error(f"❌ {package} - {description} - MISSING: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"⚠️  Missing Nix packages: {', '.join(missing_packages)}")
        logger.info("🔄 Attempting runtime pip installation...")
        
        # Try to install missing packages at runtime
        for package in missing_packages:
            try:
                logger.info(f"📦 Installing {package} via pip...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--no-cache-dir', package], 
                             check=True, capture_output=True, text=True)
                logger.info(f"✅ Successfully installed {package}")
                
                # Test the import
                try:
                    __import__(package)
                    logger.info(f"✅ {package} now available after pip install")
                except ImportError:
                    logger.warning(f"⚠️  {package} still not available after installation")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error installing {package}: {e}")
        
        # Re-check after installation attempts
        remaining_missing = []
        for package in missing_packages:
            try:
                __import__(package)
            except ImportError:
                remaining_missing.append(package)
        
        if remaining_missing:
            logger.error(f"❌ Still missing after installation: {', '.join(remaining_missing)}")
            # Return False only for Flask - other packages can be missing
            return 'flask' not in remaining_missing
        else:
            logger.info("✅ All packages now available!")
            return True
    else:
        logger.info("✅ All critical packages available from Nix environment!")
        return True

def setup_environment():
    """Set up environment variables for Railway deployment"""
    logger.info("🔧 Setting up environment...")
    
    # Railway PORT configuration  
    port = os.environ.get('PORT', '8080')
    logger.info(f"🌐 Port: {port}")
    
    # Flask configuration
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'production'
    
    # Python path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"📁 Added {current_dir} to Python path")
    
    # Disable Flask debug in production
    os.environ['FLASK_DEBUG'] = '0'
    
    return port

def verify_app_file():
    """Verify that app.py exists and can be imported"""
    logger.info("🔍 Verifying application file...")
    
    app_path = os.path.join(os.getcwd(), 'app.py')
    if not os.path.exists(app_path):
        logger.error(f"❌ app.py not found at {app_path}")
        return False
    
    logger.info(f"✅ app.py found at {app_path}")
    
    # Try to import the app module
    try:
        logger.info("🔄 Testing app.py import...")
        import app
        logger.info("✅ app.py imported successfully")
        
        # Check if Flask app exists
        if hasattr(app, 'app'):
            logger.info("✅ Flask app instance found")
            return True
        else:
            logger.error("❌ Flask app instance not found in app.py")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import app.py: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error importing app.py: {e}")
        return False

def start_flask_app(port):
    """Start the Flask application with proper configuration"""
    logger.info("� Starting Flask application...")
    
    try:
        # Import and configure the Flask app
        import app
        flask_app = app.app
        
        # Configure Flask for Railway
        flask_app.config['ENV'] = 'production'
        flask_app.config['DEBUG'] = False
        flask_app.config['TESTING'] = False
        
        logger.info(f"🌐 Starting server on 0.0.0.0:{port}")
        
        # Start the Flask application
        flask_app.run(
            host='0.0.0.0',
            port=int(port),
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import Flask app: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start Flask app: {e}")
        sys.exit(1)

def main():
    """Main application launcher"""
    logger.info("🎯 Railway Flask App Launcher Starting...")
    logger.info("=" * 50)
    
    try:
        # Step 1: Validate Python environment
        if not validate_python_environment():
            logger.error("❌ Python environment validation failed")
            sys.exit(1)
        
        # Step 2: Check critical packages
        packages_available = check_critical_packages()
        if not packages_available:
            logger.error("❌ Critical packages (especially Flask) missing")
            logger.error("❌ Cannot start web application without Flask")
            logger.error("❌ Deployment cannot continue")
            sys.exit(1)
        else:
            logger.info("✅ Sufficient packages available to start application")
        
        # Step 3: Setup environment
        port = setup_environment()
        
        # Step 4: Verify app file
        if not verify_app_file():
            logger.error("❌ Application file verification failed")
            sys.exit(1)
        
        # Step 5: Start Flask application
        logger.info("✅ All checks passed! Starting Flask application...")
        logger.info("=" * 50)
        
        start_flask_app(port)
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
