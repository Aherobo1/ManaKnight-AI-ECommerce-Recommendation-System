"""
Database Service Module

Handles all database operations for the E-Commerce Product Recommendation System.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd
from contextlib import contextmanager


class DatabaseService:
    """
    Database service for managing product data and user interactions.
    """
    
    def __init__(self, db_path: str = "data/ecommerce.db"):
        """
        Initialize database service.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Products table
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
            
            # User queries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    results_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Product vectors table (for caching)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS product_vectors (
                    product_id INTEGER,
                    vector_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            conn.commit()

        # Add enhanced sample data on first initialization
        self.add_enhanced_sample_data()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def load_products_from_csv(self, csv_path: str) -> int:
        """
        Load products from CSV file into database.

        Args:
            csv_path (str): Path to CSV file

        Returns:
            int: Number of products loaded
        """
        try:
            df = pd.read_csv(csv_path)

            # Clean the data
            import re
            def clean_text(text):
                if pd.isna(text):
                    return ''
                # Remove special characters and emojis
                text = re.sub(r'[^\w\s\-\.]', ' ', str(text))
                return ' '.join(text.split())

            def clean_price(price):
                if pd.isna(price):
                    return 0.0
                price_str = str(price)
                price_clean = re.sub(r'[^\d\.]', '', price_str)
                try:
                    return float(price_clean) if price_clean else 0.0
                except:
                    return 0.0

            df['Description'] = df['Description'].apply(clean_text)
            df['UnitPrice'] = df['UnitPrice'].apply(clean_price)
            df['Country'] = df['Country'].apply(clean_text)
            df['StockCode'] = df['StockCode'].apply(clean_text)

            # Filter valid data
            df = df[df['Description'].str.len() > 3]
            df = df[df['UnitPrice'] > 0]

            with self.get_connection() as conn:
                cursor = conn.cursor()

                products_loaded = 0
                for _, row in df.iterrows():
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO products
                            (stock_code, description, unit_price, country)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            row['StockCode'],
                            row['Description'],
                            row['UnitPrice'],
                            row['Country']
                        ))
                        products_loaded += 1
                    except Exception as e:
                        continue

                conn.commit()
                return products_loaded

        except Exception as e:
            print(f"Error loading CSV: {e}")
            return 0

    def add_enhanced_sample_data(self):
        """Add enhanced sample data including items commonly requested in handwritten queries."""
        enhanced_products = [
            # T-shirts and clothing
            ('TSHIRT001', 'Cotton T-Shirt Classic Fit Comfortable', 19.99, 'USA'),
            ('TSHIRT002', 'Graphic T-Shirt Vintage Design Fashion', 24.99, 'USA'),
            ('TSHIRT003', 'Premium T-Shirt Organic Cotton Sustainable', 29.99, 'USA'),
            ('SHIRT001', 'Dress Shirt Business Professional', 39.99, 'Italy'),
            ('SHIRT002', 'Casual Shirt Cotton Comfortable', 34.99, 'Portugal'),

            # Antiques and collectibles
            ('ANTIQUE001', 'Vintage Pocket Watch Collectible Antique', 299.99, 'United Kingdom'),
            ('ANTIQUE002', 'Antique Vase Ceramic Decorative Collectible', 149.99, 'France'),
            ('ANTIQUE003', 'Vintage Jewelry Box Wooden Antique', 89.99, 'Germany'),
            ('ANTIQUE004', 'Collectible Coins Set Vintage Antique', 199.99, 'USA'),
            ('ANTIQUE005', 'Antique Mirror Ornate Frame Vintage', 179.99, 'Italy'),

            # Teapots and kitchen items
            ('TEAPOT001', 'Ceramic Teapot Traditional Design Kitchen', 45.99, 'China'),
            ('TEAPOT002', 'Glass Teapot Heat Resistant Modern', 39.99, 'Germany'),
            ('TEAPOT003', 'Cast Iron Teapot Japanese Style Traditional', 89.99, 'Japan'),
            ('TEAPOT004', 'Electric Teapot Stainless Steel Kitchen', 69.99, 'Germany'),
            ('TEAPOT005', 'Porcelain Teapot Elegant Design Fine China', 79.99, 'China'),

            # Computer accessories
            ('COMP001', 'Wireless Mouse Ergonomic Computer Accessory', 29.99, 'China'),
            ('COMP002', 'Mechanical Keyboard Gaming Computer Accessory', 89.99, 'Taiwan'),
            ('COMP003', 'USB Hub Multi-Port Computer Accessory', 24.99, 'China'),
            ('COMP004', 'Laptop Stand Adjustable Computer Accessory', 49.99, 'USA'),
            ('COMP005', 'Webcam HD 1080p Computer Accessory', 59.99, 'China'),
            ('COMP006', 'Computer Monitor Stand Desk Accessory', 39.99, 'China'),
            ('COMP007', 'Cable Management Kit Computer Accessory', 19.99, 'China'),
            ('COMP008', 'External Hard Drive Portable Computer Accessory', 79.99, 'Singapore'),

            # Additional popular items
            ('HEADSET001', 'Gaming Headset RGB Computer Accessory', 79.99, 'China'),
            ('MOUSE001', 'Gaming Mouse High DPI Computer Accessory', 49.99, 'China'),
            ('PAD001', 'Mouse Pad Large Gaming Computer Accessory', 19.99, 'China'),
        ]

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                added_count = 0
                for stock_code, description, price, country in enhanced_products:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO products
                            (stock_code, description, unit_price, country)
                            VALUES (?, ?, ?, ?)
                        ''', (stock_code, description, price, country))
                        added_count += 1
                    except Exception as e:
                        print(f"Error adding product {stock_code}: {e}")
                        continue

                conn.commit()
                print(f"Added {added_count} enhanced sample products")
                return added_count

        except Exception as e:
            print(f"Error adding enhanced sample data: {e}")
            return 0

    def get_products(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get products from database.
        
        Args:
            limit (int): Maximum number of products to return
            offset (int): Number of products to skip
            
        Returns:
            List[Dict]: List of product dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM products 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_products(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search products by description.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of matching products
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM products 
                WHERE description LIKE ? 
                ORDER BY description 
                LIMIT ?
            ''', (f'%{query}%', limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_product_by_stock_code(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        Get product by stock code.
        
        Args:
            stock_code (str): Product stock code
            
        Returns:
            Dict or None: Product data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM products WHERE stock_code = ?', (stock_code,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def log_user_query(self, query_text: str, query_type: str, results_count: int):
        """
        Log user query for analytics.
        
        Args:
            query_text (str): The user's query
            query_type (str): Type of query (text, ocr, image)
            results_count (int): Number of results returned
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_queries (query_text, query_type, results_count)
                VALUES (?, ?, ?)
            ''', (query_text, query_type, results_count))
            conn.commit()
    
    def get_query_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get query analytics for the last N days.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            Dict: Analytics data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total queries
            cursor.execute('''
                SELECT COUNT(*) as total_queries,
                       AVG(results_count) as avg_results,
                       query_type,
                       COUNT(*) as type_count
                FROM user_queries 
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY query_type
            '''.format(days))
            
            analytics = {
                'total_queries': 0,
                'avg_results': 0,
                'query_types': {}
            }
            
            for row in cursor.fetchall():
                analytics['total_queries'] += row['type_count']
                analytics['query_types'][row['query_type']] = {
                    'count': row['type_count'],
                    'avg_results': row['avg_results']
                }
            
            return analytics


def init_db():
    """Initialize database - convenience function."""
    db_service = DatabaseService()
    print("Database initialized successfully!")
    return db_service


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
