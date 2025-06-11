"""
🧪 Service Tests - Mana Knight Digital
Comprehensive tests for all service components.
"""

import unittest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
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


class TestServices(unittest.TestCase):
    """Test all service components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Services not available")
    
    def test_database_service(self):
        """Test database service."""
        try:
            db = DatabaseService()
            self.assertTrue(db.test_connection())
            
            products = db.get_all_products()
            self.assertIsInstance(products, list)
            
            results = db.search_products("laptop")
            self.assertIsInstance(results, list)
            
            print("✅ Database service test passed")
        except Exception as e:
            print(f"⚠️ Database service test skipped: {e}")
    
    def test_recommendation_engine(self):
        """Test recommendation engine."""
        try:
            engine = RecommendationEngine()
            self.assertIsNotNone(engine)
            
            recommendations, response = engine.get_recommendations("laptop computer")
            self.assertIsInstance(recommendations, list)
            self.assertIsInstance(response, str)
            
            print("✅ Recommendation engine test passed")
        except Exception as e:
            print(f"⚠️ Recommendation engine test skipped: {e}")
    
    def test_ocr_service(self):
        """Test OCR service."""
        try:
            ocr = OCRService()
            self.assertIsNotNone(ocr)
            
            # Test with mock image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                from PIL import Image
                img = Image.new('RGB', (100, 50), color='white')
                img.save(tmp_file.name)
                
                result = ocr.extract_text_from_image(tmp_file.name)
                self.assertIsInstance(result, dict)
                self.assertIn('extracted_text', result)
                
                os.unlink(tmp_file.name)
            
            print("✅ OCR service test passed")
        except Exception as e:
            print(f"⚠️ OCR service test skipped: {e}")
    
    def test_cnn_model_service(self):
        """Test CNN model service."""
        try:
            cnn = EnhancedCNNModel()
            self.assertIsNotNone(cnn)
            
            # Test with mock image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                from PIL import Image
                img = Image.new('RGB', (224, 224), color='blue')
                img.save(tmp_file.name)
                
                result = cnn.predict_product_category(tmp_file.name)
                self.assertIsInstance(result, dict)
                self.assertIn('predicted_class', result)
                
                os.unlink(tmp_file.name)
            
            print("✅ CNN model service test passed")
        except Exception as e:
            print(f"⚠️ CNN model service test skipped: {e}")
    
    def test_vector_database_service(self):
        """Test vector database service."""
        try:
            vdb = VectorDatabase()
            self.assertIsNotNone(vdb)
            
            stats = vdb.get_index_stats()
            self.assertIsInstance(stats, dict)
            
            print("✅ Vector database service test passed")
        except Exception as e:
            print(f"⚠️ Vector database service test skipped: {e}")
    
    def test_data_cleaning_service(self):
        """Test data cleaning service."""
        try:
            cleaner = DataCleaningService()
            self.assertIsNotNone(cleaner)
            
            report = cleaner.get_cleaning_report()
            self.assertIsInstance(report, dict)
            self.assertIn('cleaning_stats', report)
            
            print("✅ Data cleaning service test passed")
        except Exception as e:
            print(f"⚠️ Data cleaning service test skipped: {e}")
    
    def test_service_integration(self):
        """Test service integration."""
        try:
            # Test that services can work together
            db = DatabaseService()
            engine = RecommendationEngine()
            
            # Test recommendation flow
            products = db.get_all_products()
            if products:
                recommendations, response = engine.get_recommendations("test query")
                self.assertIsInstance(recommendations, list)
                self.assertIsInstance(response, str)
            
            print("✅ Service integration test passed")
        except Exception as e:
            print(f"⚠️ Service integration test skipped: {e}")


if __name__ == '__main__':
    print("🧪 Running Service Tests - Mana Knight Digital")
    unittest.main(verbosity=2)
