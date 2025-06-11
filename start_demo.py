#!/usr/bin/env python3
"""
Quick demo startup script for job interview
Initializes everything needed for the demo
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print demo banner"""
    print("=" * 60)
    print("🚀 E-COMMERCE AI RECOMMENDATION SYSTEM - DEMO")
    print("=" * 60)
    print("🎯 Job Interview Demo - Ready in 30 seconds!")
    print("📊 Features: Text Search + OCR + CNN Image Classification")
    print("🔧 Tech Stack: Flask + TensorFlow + OpenCV + Tesseract")
    print("=" * 60)

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'tensorflow', 'opencv-python', 'pandas', 
        'numpy', 'scikit-learn', 'pillow'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def initialize_database():
    """Initialize database with sample data"""
    print("\n📊 Initializing database...")
    
    try:
        # Run the sample data creation script
        result = subprocess.run([sys.executable, 'create_sample_data.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Database initialized with sample products")
            return True
        else:
            print(f"❌ Database initialization failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Database initialization timed out, but continuing...")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def test_services():
    """Quick test of core services"""
    print("\n🧪 Testing core services...")
    
    try:
        # Test enhanced CNN service
        from services.enhanced_cnn_model import cnn_service
        result = cnn_service.predict_category("test_laptop.jpg")
        print(f"  ✅ CNN Model: {result['category']} ({result['confidence']:.2f})")
        
        # Test database
        from services.database import DatabaseService
        db = DatabaseService()
        products = db.get_products(limit=1)
        print(f"  ✅ Database: {len(products)} products available")
        
        # Test recommendation engine
        from services.recommendation import RecommendationEngine
        rec_engine = RecommendationEngine(db)
        recs, response = rec_engine.get_recommendations("laptop", top_k=1)
        print(f"  ✅ Recommendations: {len(recs)} results")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Service test warning: {e}")
        print("  🔄 Continuing with demo (services will use fallbacks)")
        return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Flask application...")
    print("📱 Demo interfaces will be available at:")
    print("   🏠 Main: http://localhost:5000")
    print("   📝 Text Query: http://localhost:5000/text-query")
    print("   🖼️  OCR Query: http://localhost:5000/image-query")
    print("   📸 Image Search: http://localhost:5000/product-upload")
    print("\n⏰ Starting in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🎉 DEMO IS READY! Opening application...")
    print("=" * 60)
    
    # Start the Flask app
    try:
        os.system(f"{sys.executable} app.py")
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Thank you!")

def main():
    """Main demo startup function"""
    print_banner()
    
    # Check working directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run from project root directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Initialize database
    initialize_database()
    
    # Test services
    test_services()
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()
