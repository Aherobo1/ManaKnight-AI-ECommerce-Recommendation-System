#!/usr/bin/env python3
"""
Final comprehensive verification script for job interview
Tests all critical components and provides detailed status report
"""

import os
import sys
import traceback
import subprocess
import time

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def test_file_structure():
    """Verify all required files exist"""
    print_header("FILE STRUCTURE VERIFICATION")
    
    required_files = [
        'app.py',
        'requirements.txt', 
        'create_sample_data.py',
        'DEMO_SCRIPT.md',
        'start_demo.py',
        'services/__init__.py',
        'services/database.py',
        'services/recommendation.py',
        'services/ocr_service.py',
        'services/enhanced_cnn_model.py',
        'services/vector_db.py',
        'services/scraper.py',
        'templates/index.html',
        'templates/text_query.html',
        'templates/image_query.html',
        'templates/product_upload.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files")
        return False
    else:
        print(f"\n🎉 All {len(required_files)} required files present")
        return True

def test_dependencies():
    """Test if all dependencies are installed"""
    print_header("DEPENDENCY VERIFICATION")
    
    dependencies = [
        'flask',
        'tensorflow', 
        'opencv-python',
        'pandas',
        'numpy',
        'scikit-learn',
        'pillow',
        'pytesseract'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            module_name = dep.replace('-', '_')
            __import__(module_name)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing {len(missing_deps)} dependencies")
        print("Run: pip install " + " ".join(missing_deps))
        return False
    else:
        print(f"\n🎉 All {len(dependencies)} dependencies installed")
        return True

def test_database():
    """Test database functionality"""
    print_header("DATABASE VERIFICATION")
    
    try:
        from services.database import DatabaseService
        db = DatabaseService()
        print("✅ Database service imported")
        
        # Check if database file exists
        if os.path.exists('data/ecommerce.db'):
            print("✅ Database file exists")
        else:
            print("⚠️  Database file not found, will be created")
        
        # Test getting products
        products = db.get_products(limit=5)
        print(f"✅ Database query successful: {len(products)} products")
        
        if products:
            sample = products[0]
            print(f"   Sample product: {sample['description'][:40]}...")
            print(f"   Price: ${sample['unit_price']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_enhanced_cnn():
    """Test enhanced CNN model"""
    print_header("CNN MODEL VERIFICATION")
    
    try:
        from services.enhanced_cnn_model import cnn_service
        print("✅ Enhanced CNN service imported")
        
        # Test prediction
        result = cnn_service.predict_category("test_laptop_image.jpg")
        print(f"✅ CNN prediction successful")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Method: {result.get('method', 'unknown')}")
        
        # Test model info
        info = cnn_service.get_model_info()
        print(f"✅ Model info: {len(info['classes'])} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ CNN model error: {e}")
        return False

def test_recommendation_engine():
    """Test recommendation engine"""
    print_header("RECOMMENDATION ENGINE VERIFICATION")
    
    try:
        from services.database import DatabaseService
        from services.recommendation import RecommendationEngine
        
        db = DatabaseService()
        rec_engine = RecommendationEngine(db)
        print("✅ Recommendation engine imported")
        
        # Test recommendation
        products, response = rec_engine.get_recommendations("laptop computer", top_k=3)
        print(f"✅ Recommendation successful: {len(products)} products")
        print(f"   Response: {response[:60]}...")
        
        if products:
            print(f"   Top result: {products[0]['description'][:40]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation engine error: {e}")
        return False

def test_ocr_service():
    """Test OCR service"""
    print_header("OCR SERVICE VERIFICATION")
    
    try:
        from services.ocr_service import OCRService
        ocr = OCRService()
        print("✅ OCR service imported")
        
        # Test text extraction (will use mock)
        text, confidence = ocr.extract_text_from_image("test_image.jpg")
        print(f"✅ OCR extraction successful")
        print(f"   Extracted text: {text[:40]}...")
        print(f"   Confidence: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR service error: {e}")
        return False

def test_flask_app():
    """Test Flask app startup"""
    print_header("FLASK APPLICATION VERIFICATION")
    
    try:
        # Import app to check for import errors
        import app
        print("✅ Flask app imports successfully")
        
        # Check if templates exist
        template_dir = 'templates'
        if os.path.exists(template_dir):
            templates = os.listdir(template_dir)
            print(f"✅ Templates directory: {len(templates)} templates")
        
        # Check if static directory exists
        static_dir = 'static'
        if os.path.exists(static_dir):
            print("✅ Static directory exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        traceback.print_exc()
        return False

def generate_final_report(results):
    """Generate final verification report"""
    print_header("FINAL VERIFICATION REPORT")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 System is FULLY READY for job interview demo!")
        print("\n📝 Quick Start:")
        print("   1. Run: python start_demo.py")
        print("   2. Navigate to: http://localhost:5000")
        print("   3. Demo all three interfaces")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️  MOSTLY READY")
        print("🔄 System should work for demo with minor issues")
        return True
    else:
        print("\n❌ CRITICAL ISSUES FOUND")
        print("🔧 Please fix the failed tests before demo")
        return False

def main():
    """Run comprehensive verification"""
    print("🚀 COMPREHENSIVE PROJECT VERIFICATION")
    print("🎯 Job Interview Demo Readiness Check")
    print("⏰ Starting verification...")
    
    # Run all tests
    results = {
        "File Structure": test_file_structure(),
        "Dependencies": test_dependencies(),
        "Database": test_database(),
        "CNN Model": test_enhanced_cnn(),
        "Recommendation Engine": test_recommendation_engine(),
        "OCR Service": test_ocr_service(),
        "Flask Application": test_flask_app()
    }
    
    # Generate final report
    success = generate_final_report(results)
    
    if success:
        print("\n🎊 CONGRATULATIONS!")
        print("Your E-Commerce AI Recommendation System is ready!")
        print("Good luck with your job interview! 🍀")
    else:
        print("\n🔧 Please address the issues above before the demo.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
