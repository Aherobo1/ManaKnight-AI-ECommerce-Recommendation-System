#!/usr/bin/env python3
"""Simple test of all services without Flask"""

import os
import sys

def test_services():
    """Test all services directly"""
    print("🚀 Testing E-Commerce Recommendation System")
    print("=" * 50)
    
    # Test 1: Database Service
    print("\n1️⃣ Testing Database Service...")
    try:
        from services.database import DatabaseService
        db = DatabaseService()
        
        # Add sample data
        import sqlite3
        conn = sqlite3.connect('data/ecommerce.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                unit_price REAL NOT NULL,
                country TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        sample_products = [
            ('WH001', 'Wireless Bluetooth Headphones', 89.99, 'United Kingdom'),
            ('LP002', 'Gaming Laptop 15 inch', 1299.99, 'United States'),
            ('SP003', 'Smartphone Case Protective', 19.99, 'Germany'),
        ]
        
        for stock_code, description, price, country in sample_products:
            cursor.execute('''
                INSERT OR REPLACE INTO products 
                (stock_code, description, unit_price, country)
                VALUES (?, ?, ?, ?)
            ''', (stock_code, description, price, country))
        
        conn.commit()
        conn.close()
        
        products = db.get_products(limit=3)
        print(f"✅ Database: Loaded {len(products)} products")
        for p in products:
            print(f"   - {p['description'][:30]}... ${p['unit_price']}")
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    # Test 2: Recommendation Engine
    print("\n2️⃣ Testing Recommendation Engine...")
    try:
        from services.recommendation import RecommendationEngine
        engine = RecommendationEngine()
        
        recommendations, response = engine.get_recommendations("wireless headphones", top_k=3)
        print(f"✅ Recommendations: Found {len(recommendations)} products")
        print(f"   Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
    
    # Test 3: OCR Service
    print("\n3️⃣ Testing OCR Service...")
    try:
        from services.ocr_service import OCRService
        ocr = OCRService()
        
        text, confidence = ocr.extract_text_from_image()
        print(f"✅ OCR: Extracted '{text}' (confidence: {confidence:.2f})")
    except Exception as e:
        print(f"❌ OCR error: {e}")
    
    # Test 4: CNN Model
    print("\n4️⃣ Testing CNN Model...")
    try:
        from services.cnn_model import CNNModel
        cnn = CNNModel()
        
        detected_class, confidence, predictions = cnn.predict_image()
        print(f"✅ CNN: Detected '{detected_class}' (confidence: {confidence:.2f})")
        print(f"   Top predictions: {[p['class'] for p in predictions[:3]]}")
    except Exception as e:
        print(f"❌ CNN error: {e}")
    
    # Test 5: Vector Database
    print("\n5️⃣ Testing Vector Database...")
    try:
        from services.vector_db import VectorDatabase
        vdb = VectorDatabase()
        
        stats = vdb.get_index_stats()
        print(f"✅ Vector DB: {stats}")
    except Exception as e:
        print(f"❌ Vector DB error: {e}")
    
    print("\n🎉 All service tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    test_services()
