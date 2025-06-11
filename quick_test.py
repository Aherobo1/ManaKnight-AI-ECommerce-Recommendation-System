#!/usr/bin/env python3
"""Quick test script for database initialization"""

import os
import sys
sys.path.append('.')

def test_database():
    try:
        from services.database import DatabaseService
        print('🔄 Initializing database...')
        db = DatabaseService()
        print('✅ Database initialized')

        print('🔄 Loading products...')
        loaded = db.load_products_from_csv('data/dataset/dataset.csv')
        print(f'✅ Loaded {loaded} products')

        products = db.get_products(limit=3)
        print('📦 Sample products:')
        for p in products:
            desc = p['description'][:40]
            price = p['unit_price']
            print(f'  - {desc}... ${price}')
        
        return True
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    test_database()
