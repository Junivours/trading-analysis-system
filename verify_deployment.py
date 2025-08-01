#!/usr/bin/env python3
"""
Railway Deployment Verification Script
Tests if the application starts correctly
"""

import sys
import os
import subprocess
import time

def test_import():
    """Test if the app can be imported without errors"""
    try:
        print("📦 Testing imports...")
        import app
        print("✅ App imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    try:
        print("📋 Testing dependencies...")
        
        # Core dependencies
        import flask
        import pandas
        import numpy
        import requests
        print("✅ Core dependencies OK")
        
        # Optional dependencies
        try:
            import sklearn
            print("✅ scikit-learn available")
        except ImportError:
            print("⚠️  scikit-learn not available (optional)")
            
        try:
            import pandas_ta
            print("✅ pandas-ta available")
        except ImportError:
            print("⚠️  pandas-ta not available (optional)")
            
        return True
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    try:
        print("🌍 Testing environment...")
        
        # Check PORT
        port = os.environ.get('PORT', '8080')
        print(f"✅ Port: {port}")
        
        # Check Python version
        print(f"✅ Python: {sys.version}")
        
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🔥 Railway Deployment Verification")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("App Import", test_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 Running {name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} test failed")
    
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment!")
        return 0
    else:
        print("❌ Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
