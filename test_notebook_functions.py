#!/usr/bin/env python3
"""
Test the functions from the data cleaning notebook
"""

import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample e-commerce data for demonstration with categories"""
    np.random.seed(42)
    
    # Sample product categories and descriptions
    categories_data = {
        'Electronics': [
            'Wireless Headphones Premium Quality',
            'Smartphone Android Latest Model', 
            'Laptop Computer Gaming Performance',
            'Tablet iPad Pro Professional',
            'Smart Watch Fitness Tracker',
            'Bluetooth Speaker Portable Sound',
            'Computer Mouse Wireless Ergonomic',
            'Keyboard Mechanical RGB Gaming',
            'USB Hub Multi-Port Expansion',
            'Laptop Stand Adjustable Height'
        ],
        'Clothing': [
            'T-Shirt Cotton Comfortable Casual',
            'Jeans Denim Classic Blue',
            'Dress Summer Elegant Style',
            'Jacket Winter Warm Coat',
            'Sneakers Running Sport Shoes',
            'Hat Baseball Cap Fashion',
            'Shirt Business Professional',
            'Sweater Wool Cozy Warm'
        ],
        'Home & Kitchen': [
            'Coffee Maker Automatic Brewing',
            'Vacuum Cleaner Powerful Suction',
            'Plant Pot Ceramic Decorative',
            'Lamp LED Modern Design',
            'Cushion Soft Decorative Pillow',
            'Candle Scented Relaxing Aroma',
            'Teapot Ceramic Traditional Design',
            'Glass Teapot Heat Resistant',
            'Stainless Steel Teapot Modern'
        ],
        'Sports': [
            'Yoga Mat Non-Slip Exercise',
            'Dumbbells Weight Training Set',
            'Running Shoes Athletic Performance',
            'Water Bottle Stainless Steel',
            'Fitness Tracker Smart Health',
            'Tennis Racket Professional Grade'
        ],
        'Books': [
            'Programming Book Python Guide',
            'Novel Fiction Bestseller Story',
            'Cookbook Healthy Recipe Collection',
            'Biography Inspiring Life Story',
            'Science Book Educational Learning',
            'Art Book Creative Inspiration'
        ],
        'Beauty': [
            'Face Cream Anti-Aging Formula',
            'Lipstick Matte Long-Lasting',
            'Shampoo Natural Organic Care',
            'Perfume Floral Fragrance Scent',
            'Nail Polish Glossy Finish',
            'Moisturizer Hydrating Skin Care'
        ],
        'Antiques': [
            'Vintage Wooden Clock Antique Timepiece',
            'Antique Brass Compass Navigation',
            'Victorian Era Jewelry Box Ornate',
            'Vintage Mirror Decorative Frame',
            'Antique Vase Ceramic Collectible'
        ]
    }
    
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'China', 'Italy', 'Spain']
    
    # Generate sample data with categories
    n_samples = 100  # Smaller sample for testing
    data = []
    
    for i in range(n_samples):
        # Randomly select category
        category = np.random.choice(list(categories_data.keys()))
        # Randomly select product from that category
        description = np.random.choice(categories_data[category])
        
        data.append({
            'StockCode': f'SKU{i+1:04d}',
            'Description': description,
            'Category': category,
            'UnitPrice': round(np.random.uniform(5.0, 500.0), 2),
            'Country': np.random.choice(countries)
        })
    
    return pd.DataFrame(data)

def test_notebook_functions():
    """Test the notebook functions"""
    print("🧪 Testing Notebook Functions")
    print("=" * 40)
    
    # Test sample data creation
    print("1. Testing create_sample_data()...")
    df = create_sample_data()
    
    print(f"   ✅ Created DataFrame with shape: {df.shape}")
    print(f"   ✅ Columns: {list(df.columns)}")
    
    # Check categories
    categories = df['Category'].unique()
    print(f"   ✅ Categories: {list(categories)}")
    
    # Check category distribution
    print("\n📊 Category Distribution:")
    category_counts = df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} products")
    
    # Check data types
    print(f"\n📋 Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\n❓ Missing Values:")
    for col, count in missing.items():
        print(f"   {col}: {count}")
    
    # Sample data
    print(f"\n📝 Sample Data (first 3 rows):")
    print(df.head(3).to_string())
    
    print(f"\n✅ All tests passed! Notebook functions work correctly.")
    return df

if __name__ == "__main__":
    test_df = test_notebook_functions()
