"""
🧪 API Tests - Mana Knight Digital
Comprehensive tests for all API endpoints.
"""

import unittest
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app_simple import app
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestAPI(unittest.TestCase):
    """Test all API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "App not available")
        
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
    
    def test_home_page(self):
        """Test home page."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        print("✅ Home page test passed")
    
    def test_text_query_page(self):
        """Test text query page."""
        response = self.client.get('/text-query')
        self.assertEqual(response.status_code, 200)
        print("✅ Text query page test passed")
    
    def test_image_query_page(self):
        """Test image query page."""
        response = self.client.get('/image-query')
        self.assertEqual(response.status_code, 200)
        print("✅ Image query page test passed")
    
    def test_product_upload_page(self):
        """Test product upload page."""
        response = self.client.get('/product-upload')
        self.assertEqual(response.status_code, 200)
        print("✅ Product upload page test passed")
    
    def test_api_documentation_page(self):
        """Test API documentation page."""
        response = self.client.get('/sample_response')
        self.assertEqual(response.status_code, 200)
        print("✅ API documentation page test passed")
    
    def test_product_recommendation_api(self):
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
    
    def test_ocr_query_api(self):
        """Test OCR query API."""
        # Test with mock file
        data = {'image_data': (open(__file__, 'rb'), 'test.jpg')}
        response = self.client.post('/ocr-query', data=data)
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('extracted_text', result)
        print("✅ OCR query API test passed")
    
    def test_image_product_search_api(self):
        """Test image product search API."""
        # Test with mock file
        data = {'product_image': (open(__file__, 'rb'), 'test.jpg')}
        response = self.client.post('/image-product-search', data=data)
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('predicted_class', result)
        print("✅ Image product search API test passed")
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        print("✅ Health endpoint test passed")
    
    def test_api_stats_endpoint(self):
        """Test API stats endpoint."""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_products', data)
        print("✅ API stats endpoint test passed")
    
    def test_error_handling(self):
        """Test error handling."""
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
    
    def test_404_handling(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent-endpoint')
        self.assertEqual(response.status_code, 404)
        print("✅ 404 handling test passed")


if __name__ == '__main__':
    print("🧪 Running API Tests - Mana Knight Digital")
    unittest.main(verbosity=2)
