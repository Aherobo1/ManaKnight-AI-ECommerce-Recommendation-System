"""
🚀 COMPLETE Data Cleaning & Preprocessing Service - Mana Knight Digital

Professional data cleaning and preprocessing for e-commerce product data.
Features: Duplicate removal, missing value handling, text normalization, data validation.
"""

import pandas as pd
import numpy as np
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DataCleaningService:
    """
    🎯 Professional data cleaning and preprocessing service for e-commerce data.
    """
    
    def __init__(self, database_path: str = "data/ecommerce.db"):
        """
        Initialize data cleaning service.
        
        Args:
            database_path (str): Path to SQLite database
        """
        self.database_path = database_path
        self.cleaning_stats = {
            'duplicates_removed': 0,
            'missing_values_handled': 0,
            'invalid_records_removed': 0,
            'text_normalized': 0,
            'prices_standardized': 0
        }
        
    def clean_ecommerce_dataset(self, input_file: str = None) -> Dict[str, Any]:
        """
        🧹 COMPLETE data cleaning pipeline for e-commerce dataset.
        
        Args:
            input_file (str): Path to input CSV file (optional)
            
        Returns:
            Dict[str, Any]: Cleaning results and statistics
        """
        try:
            print("🧹 Starting comprehensive data cleaning - Mana Knight Digital")
            
            # Load data
            if input_file and os.path.exists(input_file):
                df = pd.read_csv(input_file)
                print(f"📊 Loaded {len(df)} records from {input_file}")
            else:
                df = self._load_from_database()
                print(f"📊 Loaded {len(df)} records from database")
            
            original_count = len(df)
            
            # Step 1: Remove duplicates
            df = self._remove_duplicates(df)
            
            # Step 2: Handle missing values
            df = self._handle_missing_values(df)
            
            # Step 3: Standardize text fields
            df = self._normalize_text_fields(df)
            
            # Step 4: Clean and standardize prices
            df = self._standardize_prices(df)
            
            # Step 5: Validate and clean country data
            df = self._clean_country_data(df)
            
            # Step 6: Remove invalid records
            df = self._remove_invalid_records(df)
            
            # Step 7: Add data quality scores
            df = self._add_quality_scores(df)
            
            # Save cleaned data
            cleaned_count = len(df)
            self._save_cleaned_data(df)
            
            # Generate cleaning report
            results = {
                'original_records': original_count,
                'cleaned_records': cleaned_count,
                'records_removed': original_count - cleaned_count,
                'cleaning_stats': self.cleaning_stats,
                'data_quality_score': self._calculate_overall_quality_score(df),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Data cleaning complete!")
            print(f"   Original records: {original_count}")
            print(f"   Cleaned records: {cleaned_count}")
            print(f"   Records removed: {original_count - cleaned_count}")
            print(f"   Data quality score: {results['data_quality_score']:.2f}/100")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            return {'error': str(e)}
    
    def _load_from_database(self) -> pd.DataFrame:
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            df = pd.read_sql_query("SELECT * FROM products", conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records based on multiple criteria."""
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on description similarity
        if 'description' in df.columns:
            df = df.drop_duplicates(subset=['description'], keep='first')
        
        # Remove duplicates based on stock_code
        if 'stock_code' in df.columns:
            df = df.drop_duplicates(subset=['stock_code'], keep='first')
        
        removed = initial_count - len(df)
        self.cleaning_stats['duplicates_removed'] = removed
        
        if removed > 0:
            print(f"🗑️  Removed {removed} duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently."""
        initial_missing = df.isnull().sum().sum()
        
        # Fill missing descriptions with placeholder
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('Product Description Not Available')
        
        # Fill missing prices with median price
        if 'unit_price' in df.columns:
            median_price = df['unit_price'].median()
            df['unit_price'] = df['unit_price'].fillna(median_price)
        
        # Fill missing countries with 'Unknown'
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('Unknown')
        
        # Fill missing stock codes with generated codes
        if 'stock_code' in df.columns:
            missing_codes = df['stock_code'].isnull()
            df.loc[missing_codes, 'stock_code'] = [f'GEN{i:06d}' for i in range(missing_codes.sum())]
        
        final_missing = df.isnull().sum().sum()
        handled = initial_missing - final_missing
        self.cleaning_stats['missing_values_handled'] = handled
        
        if handled > 0:
            print(f"🔧 Handled {handled} missing values")
        
        return df
    
    def _normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and clean text fields."""
        normalized_count = 0
        
        if 'description' in df.columns:
            # Clean descriptions
            df['description'] = df['description'].apply(self._clean_text)
            normalized_count += len(df)
        
        if 'country' in df.columns:
            # Standardize country names
            df['country'] = df['country'].apply(self._standardize_country)
        
        self.cleaning_stats['text_normalized'] = normalized_count
        
        if normalized_count > 0:
            print(f"📝 Normalized {normalized_count} text fields")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-\.,\(\)]', '', text)
        
        # Capitalize first letter of each word
        text = text.title()
        
        return text
    
    def _standardize_country(self, country: str) -> str:
        """Standardize country names."""
        if pd.isna(country):
            return "Unknown"
        
        country = str(country).strip().title()
        
        # Common country name mappings
        country_mappings = {
            'Usa': 'United States',
            'Us': 'United States',
            'Uk': 'United Kingdom',
            'Uae': 'United Arab Emirates'
        }
        
        return country_mappings.get(country, country)
    
    def _standardize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data."""
        if 'unit_price' not in df.columns:
            return df
        
        standardized_count = 0
        
        # Convert to numeric, handling any string prices
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        
        # Remove negative prices
        df = df[df['unit_price'] >= 0]
        
        # Remove extremely high prices (outliers)
        price_99th = df['unit_price'].quantile(0.99)
        df = df[df['unit_price'] <= price_99th * 2]  # Allow some flexibility
        
        # Round to 2 decimal places
        df['unit_price'] = df['unit_price'].round(2)
        
        standardized_count = len(df)
        self.cleaning_stats['prices_standardized'] = standardized_count
        
        print(f"💰 Standardized {standardized_count} price records")
        
        return df
    
    def _clean_country_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate country data."""
        if 'country' not in df.columns:
            return df
        
        # List of valid countries (simplified)
        valid_countries = {
            'United States', 'United Kingdom', 'Germany', 'France', 'Italy',
            'Spain', 'Canada', 'Australia', 'Japan', 'China', 'India',
            'Brazil', 'Mexico', 'Netherlands', 'Belgium', 'Switzerland',
            'Sweden', 'Norway', 'Denmark', 'Finland', 'Unknown'
        }
        
        # Mark invalid countries as 'Unknown'
        invalid_mask = ~df['country'].isin(valid_countries)
        df.loc[invalid_mask, 'country'] = 'Unknown'
        
        return df
    
    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records that don't meet quality standards."""
        initial_count = len(df)
        
        # Remove records with empty descriptions
        if 'description' in df.columns:
            df = df[df['description'].str.len() > 5]
        
        # Remove records with zero or negative prices
        if 'unit_price' in df.columns:
            df = df[df['unit_price'] > 0]
        
        removed = initial_count - len(df)
        self.cleaning_stats['invalid_records_removed'] = removed
        
        if removed > 0:
            print(f"❌ Removed {removed} invalid records")
        
        return df
    
    def _add_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality scores to records."""
        df['quality_score'] = 100  # Start with perfect score
        
        # Deduct points for missing or poor quality data
        if 'description' in df.columns:
            short_desc_mask = df['description'].str.len() < 20
            df.loc[short_desc_mask, 'quality_score'] -= 20
        
        if 'country' in df.columns:
            unknown_country_mask = df['country'] == 'Unknown'
            df.loc[unknown_country_mask, 'quality_score'] -= 10
        
        # Ensure quality score is between 0 and 100
        df['quality_score'] = df['quality_score'].clip(0, 100)
        
        return df
    
    def _calculate_overall_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        if 'quality_score' in df.columns:
            return df['quality_score'].mean()
        return 85.0  # Default score
    
    def _save_cleaned_data(self, df: pd.DataFrame):
        """Save cleaned data back to database."""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Backup original table
            conn.execute("DROP TABLE IF EXISTS products_backup")
            conn.execute("CREATE TABLE products_backup AS SELECT * FROM products")
            
            # Replace with cleaned data
            df.to_sql('products', conn, if_exists='replace', index=False)
            
            conn.commit()
            conn.close()
            
            print("💾 Cleaned data saved to database")
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get detailed cleaning report."""
        return {
            'cleaning_stats': self.cleaning_stats,
            'timestamp': datetime.now().isoformat(),
            'service': 'Mana Knight Digital Data Cleaning Service'
        }


if __name__ == "__main__":
    # Test data cleaning service
    cleaner = DataCleaningService()
    results = cleaner.clean_ecommerce_dataset()
    print(f"Cleaning results: {results}")
