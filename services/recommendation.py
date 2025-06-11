"""
Recommendation Engine Module

Handles product recommendations using vector similarity search and NLP.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from .database import DatabaseService
from .vector_db import VectorDatabase


class RecommendationEngine:
    """
    Product recommendation engine using vector similarity and NLP.
    """
    
    def __init__(self, db_service: DatabaseService = None, vector_db: VectorDatabase = None):
        """
        Initialize recommendation engine.
        
        Args:
            db_service (DatabaseService): Database service instance
            vector_db (VectorDatabase): Vector database service instance
        """
        self.db_service = db_service or DatabaseService()
        self.vector_db = vector_db or VectorDatabase()
        self.vectorizer = None
        self.product_vectors = None
        self.products_data = None
        self.load_or_create_vectors()
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query for better matching.
        
        Args:
            query (str): Raw user query
            
        Returns:
            str: Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Expand common abbreviations
        abbreviations = {
            'laptop': 'laptop computer',
            'phone': 'smartphone mobile phone',
            'headphones': 'headphones earphones audio',
            'mouse': 'computer mouse wireless',
            'keyboard': 'computer keyboard wireless'
        }
        
        for abbrev, expansion in abbreviations.items():
            if abbrev in query:
                query = query.replace(abbrev, expansion)
        
        return query
    
    def load_or_create_vectors(self):
        """Load existing vectors or create new ones from database."""
        vector_file = 'models/product_vectors.pkl'
        
        if os.path.exists(vector_file):
            try:
                with open(vector_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.product_vectors = data['vectors']
                    self.products_data = data['products']
                print("Loaded existing product vectors")
                return
            except Exception as e:
                print(f"Error loading vectors: {e}")
        
        # Create new vectors
        self.create_product_vectors()
    
    def create_product_vectors(self):
        """Create TF-IDF vectors for all products."""
        print("Creating product vectors...")
        
        # Get all products from database
        products = self.db_service.get_products(limit=10000)
        
        if not products:
            print("No products found in database")
            return
        
        # Prepare text data for vectorization
        product_texts = []
        for product in products:
            text = f"{product['description']} {product.get('country', '')}"
            product_texts.append(text)
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.product_vectors = self.vectorizer.fit_transform(product_texts)
        self.products_data = products
        
        # Save vectors for future use
        os.makedirs('models', exist_ok=True)
        with open('models/product_vectors.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vectors': self.product_vectors,
                'products': self.products_data
            }, f)
        
        print(f"Created vectors for {len(products)} products")
    
    def get_recommendations(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get product recommendations based on query.
        
        Args:
            query (str): User query
            top_k (int): Number of recommendations to return
            
        Returns:
            Tuple[List[Dict], str]: (recommendations, natural language response)
        """
        if not self.vectorizer or self.product_vectors is None:
            return [], "Sorry, the recommendation system is not ready yet."
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                product = self.products_data[idx].copy()
                product['similarity_score'] = float(similarities[idx])
                recommendations.append(product)
        
        # Generate natural language response
        response = self.generate_response(query, recommendations)
        
        # Log query
        self.db_service.log_user_query(query, 'text', len(recommendations))
        
        return recommendations, response
    
    def generate_response(self, query: str, recommendations: List[Dict[str, Any]]) -> str:
        """
        Generate natural language response for recommendations.
        
        Args:
            query (str): Original user query
            recommendations (List[Dict]): Product recommendations
            
        Returns:
            str: Natural language response
        """
        if not recommendations:
            return f"I couldn't find any products matching '{query}'. Please try a different search term or browse our categories."
        
        # Start response
        response_parts = []
        
        if len(recommendations) == 1:
            product = recommendations[0]
            response_parts.append(f"I found a great match for '{query}': {product['description']}")
            response_parts.append(f"priced at ${product['unit_price']:.2f}")
        else:
            response_parts.append(f"I found {len(recommendations)} excellent options for '{query}':")
            
            for i, product in enumerate(recommendations[:3], 1):
                response_parts.append(f"{i}. {product['description']} (${product['unit_price']:.2f})")
        
        # Add helpful context
        if any('wireless' in query.lower() for query in [query]):
            response_parts.append("All recommended products offer great wireless connectivity.")
        
        if any(word in query.lower() for word in ['cheap', 'affordable', 'budget']):
            avg_price = np.mean([p['unit_price'] for p in recommendations])
            response_parts.append(f"These options are budget-friendly with an average price of ${avg_price:.2f}.")
        
        return " ".join(response_parts)
    
    def get_similar_products(self, product_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get products similar to a given product.
        
        Args:
            product_id (int): ID of the reference product
            top_k (int): Number of similar products to return
            
        Returns:
            List[Dict]: Similar products
        """
        if not self.product_vectors or product_id >= len(self.products_data):
            return []
        
        # Get similarities to the reference product
        product_vector = self.product_vectors[product_id]
        similarities = cosine_similarity(product_vector, self.product_vectors).flatten()
        
        # Get top matches (excluding the product itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        similar_products = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Minimum similarity threshold
                product = self.products_data[idx].copy()
                product['similarity_score'] = float(similarities[idx])
                similar_products.append(product)
        
        return similar_products
    
    def update_vectors(self):
        """Update product vectors with new data from database."""
        print("Updating product vectors...")
        self.create_product_vectors()
    
    def get_trending_products(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending products based on query frequency.
        
        Args:
            days (int): Number of days to consider
            limit (int): Maximum number of products to return
            
        Returns:
            List[Dict]: Trending products
        """
        # This is a simplified implementation
        # In a real system, you'd track product views/searches
        products = self.db_service.get_products(limit=limit)
        
        # Add mock trending scores
        for product in products:
            product['trending_score'] = np.random.random()
        
        # Sort by trending score
        products.sort(key=lambda x: x['trending_score'], reverse=True)
        
        return products
    
    def get_category_recommendations(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific category.
        
        Args:
            category (str): Product category
            limit (int): Maximum number of products to return
            
        Returns:
            List[Dict]: Category-based recommendations
        """
        return self.get_recommendations(category, top_k=limit)[0]


if __name__ == "__main__":
    # Test the recommendation engine
    engine = RecommendationEngine()
    recommendations, response = engine.get_recommendations("wireless headphones", top_k=3)
    print(f"Response: {response}")
    print(f"Found {len(recommendations)} recommendations")
