#!/usr/bin/env python3
"""Create sample data for the application"""

import os
import csv
import sqlite3

def create_sample_database():
    """Create sample database with products"""
    print("🔄 Creating sample database...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample products
    products = [
        ('WH001', 'Wireless Bluetooth Headphones', 89.99, 'United Kingdom'),
        ('LP002', 'Gaming Laptop 15 inch', 1299.99, 'United States'),
        ('SP003', 'Smartphone Case Protective', 19.99, 'Germany'),
        ('KT004', 'Kitchen Knife Set Stainless Steel', 79.99, 'France'),
        ('HD005', 'Home Decoration Vase Ceramic', 34.99, 'Italy'),
        ('BK006', 'Programming Book Python Guide', 49.99, 'United Kingdom'),
        ('SP007', 'Sports Water Bottle', 24.99, 'United States'),
        ('BT008', 'Beauty Face Cream Moisturizer', 39.99, 'France'),
        ('AU009', 'Car Phone Mount Holder', 29.99, 'Germany'),
        ('JW010', 'Silver Necklace Chain', 159.99, 'Italy'),
    ]
    
    # Create database
    conn = sqlite3.connect('data/ecommerce.db')
    cursor = conn.cursor()
    
    # Create table
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
    
    # Insert products
    for stock_code, description, price, country in products:
        cursor.execute('''
            INSERT OR REPLACE INTO products 
            (stock_code, description, unit_price, country)
            VALUES (?, ?, ?, ?)
        ''', (stock_code, description, price, country))
    
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
