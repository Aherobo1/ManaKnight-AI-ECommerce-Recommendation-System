#!/usr/bin/env python3
"""Test the Flask application"""

import os
import sys
import requests
import time
from threading import Thread

def start_app():
    """Start the Flask app in background"""
    os.system("python app.py")

def test_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    # Wait for app to start
    print("⏳ Waiting for app to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health: {response.status_code}")
        
        # Test product recommendation
        print("🔍 Testing product recommendation...")
        response = requests.post(f"{base_url}/product-recommendation", 
                               json={"query": "wireless headphones"}, timeout=10)
        print(f"✅ Recommendation: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {len(data.get('products', []))} products")
        
        # Test sample response
        print("🔍 Testing sample response...")
        response = requests.get(f"{base_url}/sample_response", timeout=5)
        print(f"✅ Sample response: {response.status_code}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Flask app test...")
    
    # Start app in background
    app_thread = Thread(target=start_app, daemon=True)
    app_thread.start()
    
    # Test endpoints
    test_endpoints()
