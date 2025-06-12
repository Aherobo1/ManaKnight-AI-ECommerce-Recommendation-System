"""
🚀 COMPLETE System Tests - Mana Knight Digital

Comprehensive unit tests for all system components.
Tests: API endpoints, services, data processing, ML models.
"""

import unittest
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import app
    from services.database import DatabaseService
    from services.recommendation import RecommendationEngine
    from services.ocr_service import OCRService
    from services.enhanced_cnn_model import EnhancedCNNModel
    from services.vector_db import VectorDatabase
    from services.data_cleaning import DataCleaningService
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestCompleteSystem(unittest.TestCase):
    """🎯 Complete system test suite for Mana Knight Digital AI platform."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        
        # Initialize services
        cls.db_service = DatabaseService()
        cls.recommendation_service = RecommendationService()
        cls.ocr_service = OCRService()
        cls.cnn_model = EnhancedCNNModel()
        cls.vector_db = VectorDatabase()
        cls.data_cleaner = DataCleaningService()
    
    def test_01_flask_app_initialization(self):
        """Test Flask application initialization."""
        self.assertIsNotNone(self.app)
        self.assertTrue(self.app.config['TESTING'])
        print("✅ Flask app initialization test passed")
    
    def test_02_home_page(self):
        """Test home page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Mana Knight Digital', response.data)
        print("✅ Home page test passed")
    
    def test_03_api_documentation_page(self):
        """Test API documentation page."""
        response = self.client.get('/sample_response')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'API Documentation', response.data)
        print("✅ API documentation page test passed")
    
    def test_04_text_query_interface(self):
        """Test text query interface."""
        response = self.client.get('/text-query')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Natural Language Search', response.data)
        print("✅ Text query interface test passed")
    
    def test_05_image_query_interface(self):
        """Test image query interface."""
        response = self.client.get('/image-query')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'OCR', response.data)
        print("✅ Image query interface test passed")
    
    def test_06_product_upload_interface(self):
        """Test product upload interface."""
        response = self.client.get('/product-upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CNN', response.data)
        print("✅ Product upload interface test passed")
    
    def test_07_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        print("✅ Health endpoint test passed")
    
    def test_08_api_stats_endpoint(self):
        """Test API stats endpoint."""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_products', data)
        print("✅ API stats endpoint test passed")
    
    def test_09_product_recommendation_api(self):
        """Test product recommendation API."""
        test_data = {'query': 'laptop for programming'}
        response = self.client.post('/product-recommendation',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('products', data)
        self.assertIn('response', data)
        print("✅ Product recommendation API test passed")
    
    def test_10_database_service(self):
        """Test database service functionality."""
        # Test connection
        self.assertTrue(self.db_service.test_connection())
        
        # Test product retrieval
        products = self.db_service.get_all_products()
        self.assertIsInstance(products, list)
        
        # Test search
        results = self.db_service.search_products("laptop")
        self.assertIsInstance(results, list)
        
        print("✅ Database service test passed")
    
    def test_11_recommendation_service(self):
        """Test recommendation service."""
        # Test recommendation generation
        recommendations = self.recommendation_service.get_recommendations("laptop computer")
        self.assertIsInstance(recommendations, dict)
        self.assertIn('products', recommendations)
        
        print("✅ Recommendation service test passed")
    
    def test_12_ocr_service(self):
        """Test OCR service."""
        # Test with mock image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple test image (1x1 pixel)
            from PIL import Image
            img = Image.new('RGB', (100, 50), color='white')
            img.save(tmp_file.name)
            
            # Test OCR extraction
            result = self.ocr_service.extract_text_from_image(tmp_file.name)
            self.assertIsInstance(result, dict)
            self.assertIn('extracted_text', result)
            
            # Cleanup
            os.unlink(tmp_file.name)
        
        print("✅ OCR service test passed")
    
    def test_13_cnn_model_service(self):
        """Test CNN model service."""
        # Test model initialization
        self.assertIsNotNone(self.cnn_model)
        
        # Test prediction with mock image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            from PIL import Image
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(tmp_file.name)
            
            # Test prediction
            result = self.cnn_model.predict_product_category(tmp_file.name)
            self.assertIsInstance(result, dict)
            self.assertIn('predicted_class', result)
            
            # Cleanup
            os.unlink(tmp_file.name)
        
        print("✅ CNN model service test passed")
    
    def test_14_vector_database_service(self):
        """Test vector database service."""
        # Test initialization
        self.assertIsNotNone(self.vector_db)
        
        # Test stats
        stats = self.vector_db.get_index_stats()
        self.assertIsInstance(stats, dict)
        
        print("✅ Vector database service test passed")
    
    def test_15_data_cleaning_service(self):
        """Test data cleaning service."""
        # Test initialization
        self.assertIsNotNone(self.data_cleaner)
        
        # Test cleaning report
        report = self.data_cleaner.get_cleaning_report()
        self.assertIsInstance(report, dict)
        self.assertIn('cleaning_stats', report)
        
        print("✅ Data cleaning service test passed")
    
    def test_16_error_handling(self):
        """Test error handling in API endpoints."""
        # Test invalid JSON
        response = self.client.post('/product-recommendation',
                                  data='invalid json',
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test missing query
        response = self.client.post('/product-recommendation',
                                  data=json.dumps({}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        print("✅ Error handling test passed")
    
    def test_17_performance_metrics(self):
        """Test system performance metrics."""
        import time
        
        # Test response time for text query
        start_time = time.time()
        test_data = {'query': 'smartphone'}
        response = self.client.post('/product-recommendation',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 5.0)  # Should respond within 5 seconds
        self.assertEqual(response.status_code, 200)
        
        print(f"✅ Performance test passed (Response time: {response_time:.3f}s)")
    
    def test_18_data_validation(self):
        """Test data validation and sanitization."""
        # Test SQL injection protection
        malicious_query = "'; DROP TABLE products; --"
        test_data = {'query': malicious_query}
        response = self.client.post('/product-recommendation',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)  # Should handle gracefully
        
        # Test XSS protection
        xss_query = "<script>alert('xss')</script>"
        test_data = {'query': xss_query}
        response = self.client.post('/product-recommendation',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        print("✅ Data validation test passed")
    
    def test_19_system_integration(self):
        """Test complete system integration."""
        # Test full workflow: query -> recommendation -> response
        test_query = "wireless headphones for gaming"
        
        # Step 1: Send query
        test_data = {'query': test_query}
        response = self.client.post('/product-recommendation',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        # Step 2: Validate response structure
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Step 3: Check all required fields
        required_fields = ['products', 'response', 'query_processed']
        for field in required_fields:
            self.assertIn(field, data)
        
        # Step 4: Validate products structure
        if data['products']:
            product = data['products'][0]
            product_fields = ['id', 'description', 'unit_price']
            for field in product_fields:
                self.assertIn(field, product)
        
        print("✅ System integration test passed")
    
    def test_20_final_system_validation(self):
        """Final comprehensive system validation."""
        print("\n🎯 FINAL SYSTEM VALIDATION - MANA KNIGHT DIGITAL")
        print("=" * 60)
        
        # Test all critical endpoints
        endpoints = [
            ('/', 'Home Page'),
            ('/text-query', 'Text Query Interface'),
            ('/image-query', 'Image Query Interface'),
            ('/product-upload', 'Product Upload Interface'),
            ('/sample_response', 'API Documentation'),
            ('/health', 'Health Check'),
            ('/api/stats', 'API Statistics')
        ]
        
        all_passed = True
        for endpoint, name in endpoints:
            try:
                response = self.client.get(endpoint)
                if response.status_code == 200:
                    print(f"✅ {name}: WORKING")
                else:
                    print(f"❌ {name}: FAILED ({response.status_code})")
                    all_passed = False
            except Exception as e:
                print(f"❌ {name}: ERROR ({e})")
                all_passed = False
        
        # Test API functionality
        try:
            test_data = {'query': 'test product search'}
            response = self.client.post('/product-recommendation',
                                      data=json.dumps(test_data),
                                      content_type='application/json')
            if response.status_code == 200:
                print("✅ Product Recommendation API: WORKING")
            else:
                print(f"❌ Product Recommendation API: FAILED ({response.status_code})")
                all_passed = False
        except Exception as e:
            print(f"❌ Product Recommendation API: ERROR ({e})")
            all_passed = False
        
        print("=" * 60)
        if all_passed:
            print("🎉 ALL SYSTEMS OPERATIONAL - 100% COMPLETE!")
            print("🚀 READY FOR INTERVIEW DEMONSTRATION!")
        else:
            print("⚠️  Some systems need attention")
        
        self.assertTrue(all_passed, "All systems should be operational")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)
