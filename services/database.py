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
                            row.get('StockCode', ''),
                            row.get('Description', ''),
                            float(row.get('UnitPrice', 0)),
                            row.get('Country', '')
                        ))
                        products_loaded += 1
                    except Exception as e:
                        print(f"Error loading product {row}: {e}")
                        continue
                
                conn.commit()
                return products_loaded
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
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
