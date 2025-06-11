"""
Services Package

This package contains all the backend services for the E-Commerce Product Recommendation System.

Modules:
    - database: Database operations and connections
    - recommendation: Product recommendation engine
    - ocr_service: OCR functionality for text extraction
    - cnn_model: CNN model for image-based product detection
    - scraper: Web scraping utilities for data collection
    - vector_db: Vector database operations with Pinecone
"""

__version__ = "1.0.0"
__author__ = "E-Commerce Recommendation Team"

# Import main service classes for easy access
try:
    from .database import DatabaseService
except ImportError:
    DatabaseService = None

try:
    from .recommendation import RecommendationEngine
except ImportError:
    RecommendationEngine = None

try:
    from .ocr_service import OCRService
except ImportError:
    OCRService = None

try:
    from .cnn_model import CNNModel
except ImportError:
    CNNModel = None

try:
    from .scraper import WebScraper
except ImportError:
    WebScraper = None

try:
    from .vector_db import VectorDatabase
except ImportError:
    VectorDatabase = None

# List of available services
__all__ = [
    'DatabaseService',
    'RecommendationEngine', 
    'OCRService',
    'CNNModel',
    'WebScraper',
    'VectorDatabase'
]

# Service registry for dependency injection
SERVICE_REGISTRY = {
    'database': DatabaseService,
    'recommendation': RecommendationEngine,
    'ocr': OCRService,
    'cnn': CNNModel,
    'scraper': WebScraper,
    'vector_db': VectorDatabase
}

def get_service(service_name):
    """
    Get a service class by name.
    
    Args:
        service_name (str): Name of the service
        
    Returns:
        class: Service class or None if not found
    """
    return SERVICE_REGISTRY.get(service_name)

def list_services():
    """
    List all available services.
    
    Returns:
        list: List of available service names
    """
    return [name for name, cls in SERVICE_REGISTRY.items() if cls is not None]
