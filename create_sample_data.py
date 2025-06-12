#!/usr/bin/env python3
"""Create sample data for the application"""

import os
import sqlite3

def create_sample_database():
    """Create sample database with products"""
    print("🔄 Creating sample database...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample products with categories (matching our updated examples)
    products = [
        # T-shirts
        ('TS001', 'Cotton T-Shirt Classic Fit Comfortable Casual Wear', 'Clothing', 19.99, 'United States'),
        ('TS002', 'Premium T-Shirt Organic Cotton Soft Breathable', 'Clothing', 29.99, 'Canada'),
        ('TS003', 'Designer T-Shirt Trendy Fashion Statement Piece', 'Clothing', 45.99, 'Italy'),

        # Antiques
        ('AN001', 'Vintage Wooden Clock Antique Handcrafted Timepiece', 'Antiques', 299.99, 'United Kingdom'),
        ('AN002', 'Antique Brass Compass Vintage Navigation Instrument', 'Antiques', 149.99, 'France'),
        ('AN003', 'Victorian Era Jewelry Box Ornate Antique Storage', 'Antiques', 189.99, 'United Kingdom'),

        # Teapots
        ('TP001', 'Ceramic Teapot Traditional Design Tea Brewing Set', 'Home & Kitchen', 39.99, 'China'),
        ('TP002', 'Glass Teapot Heat Resistant Transparent Tea Maker', 'Home & Kitchen', 24.99, 'Germany'),
        ('TP003', 'Stainless Steel Teapot Modern Durable Tea Server', 'Home & Kitchen', 49.99, 'Japan'),

        # Computer Accessories
        ('CA001', 'Wireless Mouse Ergonomic Computer Accessory Office', 'Electronics', 29.99, 'United States'),
        ('CA002', 'Mechanical Keyboard RGB Backlit Computer Gaming', 'Electronics', 89.99, 'Taiwan'),
        ('CA003', 'USB Hub Multi-Port Computer Accessory Expansion', 'Electronics', 19.99, 'South Korea'),
        ('CA004', 'Laptop Stand Adjustable Computer Accessory Ergonomic', 'Electronics', 39.99, 'Denmark'),

        # Additional products
        ('WH001', 'Wireless Bluetooth Headphones Premium Quality', 'Electronics', 89.99, 'United Kingdom'),
        ('LP002', 'Gaming Laptop High Performance 16GB RAM RTX 4060', 'Electronics', 1299.99, 'United States'),
        ('BK006', 'Programming Book Python Complete Guide', 'Books', 49.99, 'United Kingdom'),
        ('SP007', 'Sports Water Bottle Stainless Steel', 'Sports', 24.99, 'United States'),
        ('BT008', 'Beauty Face Cream Anti-Aging Moisturizer', 'Beauty', 39.99, 'France'),
    ]
    
    # Create database
    conn = sqlite3.connect('data/ecommerce.db')
    cursor = conn.cursor()
    
    # Drop existing table and create new one with category column
    cursor.execute('DROP TABLE IF EXISTS products')
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            unit_price REAL NOT NULL,
            country TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert products
    for stock_code, description, category, price, country in products:
        cursor.execute('''
            INSERT OR REPLACE INTO products
            (stock_code, description, category, unit_price, country)
            VALUES (?, ?, ?, ?, ?)
        ''', (stock_code, description, category, price, country))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created database with {len(products)} products")

def create_sample_directories():
    """Create sample directories"""
    directories = [
        'data/sample_images',
        'models',
        'static/uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created sample directories")

if __name__ == "__main__":
    create_sample_database()
    create_sample_directories()
    print("🎉 Sample data created successfully!")
