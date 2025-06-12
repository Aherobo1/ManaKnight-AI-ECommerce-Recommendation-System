#!/usr/bin/env python3
"""
Test script for new text query examples
"""

import requests
import json

def test_examples():
    print('🧪 Testing Updated Text Query Examples')
    print('=' * 50)

    examples = [
        'Looking for a T-shirt',
        'Suggest some Antiques', 
        'Is there any teapot available?',
        'I want to buy some computer accessories.'
    ]

    for i, query in enumerate(examples, 1):
        try:
            print(f'\n{i}. Testing: "{query}"')
            
            response = requests.post('http://localhost:5000/product-recommendation',
                                   json={'query': query},
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                products = data.get('products', [])
                print(f'   ✅ Found {len(products)} products')
                
                if products:
                    for j, product in enumerate(products[:2], 1):
                        print(f'   {j}. {product.get("description", "N/A")} - ${product.get("unit_price", 0)}')
                
                print(f'   Response: {data.get("response", "N/A")[:60]}...')
            else:
                print(f'   ❌ Error: {response.status_code}')
                
        except requests.exceptions.ConnectionError:
            print(f'   ❌ Server not running')
            break
        except Exception as e:
            print(f'   ❌ Error: {e}')

    print('\n🎯 Test completed!')

if __name__ == "__main__":
    test_examples()
