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

        # Remove special characters but keep spaces and hyphens
        query = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', query)

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Expand common abbreviations and synonyms
        abbreviations = {
            'laptop': 'laptop computer notebook',
            'phone': 'smartphone mobile phone cellphone',
            'headphones': 'headphones earphones audio headset',
            'mouse': 'computer mouse wireless optical',
            'keyboard': 'computer keyboard wireless mechanical',
            't-shirt': 'tshirt shirt clothing apparel fashion',
            'tshirt': 'tshirt shirt clothing apparel fashion',
            'shirt': 'shirt clothing apparel fashion',
            'antiques': 'antiques vintage collectibles old items',
            'teapot': 'teapot tea pot kitchen utensils cookware',
            'computer accessories': 'computer accessories peripherals hardware tech',
            'accessories': 'accessories peripherals add-ons extras'
        }

        for abbrev, expansion in abbreviations.items():
            if abbrev in query:
                query = query.replace(abbrev, expansion)

        return query

    def extract_multiple_queries(self, text: str) -> List[str]:
        """
        Extract multiple product queries from a single text (like handwritten lists).

        Args:
            text (str): Input text that may contain multiple queries

        Returns:
            List[str]: List of individual product queries
        """
        # Split by common separators
        separators = ['\n', '-', '•', '*', '1.', '2.', '3.', '4.', '5.']

        # Start with the full text
        queries = [text]

        # Split by each separator
        for separator in separators:
            new_queries = []
            for query in queries:
                if separator in query:
                    parts = query.split(separator)
                    new_queries.extend([part.strip() for part in parts if part.strip()])
                else:
                    new_queries.append(query)
            queries = new_queries

        # Clean up queries
        cleaned_queries = []
        for query in queries:
            # Remove common prefixes
            query = re.sub(r'^(looking for|suggest|is there|i want to buy|find me)', '', query, flags=re.IGNORECASE)
            query = query.strip()

            # Only keep queries with actual product terms
            if len(query) > 3 and any(word in query.lower() for word in [
                'shirt', 'antique', 'teapot', 'computer', 'accessory', 'laptop', 'phone',
                'headphone', 'mouse', 'keyboard', 'clothing', 'kitchen', 'tech', 'electronic'
            ]):
                cleaned_queries.append(query)

        return cleaned_queries if cleaned_queries else [text]

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
            query (str): User query (may contain multiple items)
            top_k (int): Number of recommendations to return

        Returns:
            Tuple[List[Dict], str]: (recommendations, natural language response)
        """
        if not self.vectorizer or self.product_vectors is None:
            return [], "Sorry, the recommendation system is not ready yet."

        # Extract multiple queries if present
        individual_queries = self.extract_multiple_queries(query)

        all_recommendations = []
        query_results = {}

        # Process each individual query
        for individual_query in individual_queries:
            # Preprocess query
            processed_query = self.preprocess_query(individual_query)

            # Vectorize query
            query_vector = self.vectorizer.transform([processed_query])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.product_vectors).flatten()

            # Get top matches for this specific query
            top_indices = np.argsort(similarities)[::-1][:max(2, top_k//len(individual_queries))]

            query_recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Lower threshold for better matching
                    product = self.products_data[idx].copy()
                    product['similarity_score'] = float(similarities[idx])
                    product['matched_query'] = individual_query
                    query_recommendations.append(product)

            if query_recommendations:
                query_results[individual_query] = query_recommendations
                all_recommendations.extend(query_recommendations)

        # If no individual queries worked, try the full query
        if not all_recommendations:
            processed_query = self.preprocess_query(query)
            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vector, self.product_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]

            for idx in top_indices:
                if similarities[idx] > 0.05:
                    product = self.products_data[idx].copy()
                    product['similarity_score'] = float(similarities[idx])
                    product['matched_query'] = query
                    all_recommendations.append(product)

        # Remove duplicates and sort by similarity
        seen_products = set()
        unique_recommendations = []
        for product in sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True):
            product_key = (product['stock_code'], product['description'])
            if product_key not in seen_products:
                seen_products.add(product_key)
                unique_recommendations.append(product)

        # Limit to top_k
        final_recommendations = unique_recommendations[:top_k]

        # Generate natural language response
        response = self.generate_multi_query_response(query, query_results, final_recommendations)

        # Log query
        self.db_service.log_user_query(query, 'text', len(final_recommendations))

        return final_recommendations, response
    
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

    def generate_multi_query_response(self, original_query: str, query_results: Dict[str, List], recommendations: List[Dict[str, Any]]) -> str:
        """
        Generate response for multiple queries extracted from handwritten text.

        Args:
            original_query (str): Original query text
            query_results (Dict): Results for each individual query
            recommendations (List[Dict]): Final recommendations

        Returns:
            str: Natural language response
        """
        if not recommendations:
            return f"I couldn't find specific products matching your requests. Please try more specific search terms."

        response_parts = []

        if len(query_results) > 1:
            response_parts.append(f"I found products for your multiple requests:")

            # Group recommendations by matched query
            for query, products in query_results.items():
                if products:
                    response_parts.append(f"\nFor '{query}':")
                    for i, product in enumerate(products[:2], 1):  # Show top 2 per query
                        response_parts.append(f"  {i}. {product['description']} (${product['unit_price']:.2f})")
        else:
            # Single query response
            response_parts.append(f"I found {len(recommendations)} great options for your request:")
            for i, product in enumerate(recommendations[:3], 1):
                response_parts.append(f"{i}. {product['description']} (${product['unit_price']:.2f})")

        # Add helpful context based on the types of products found
        product_types = set()
        for product in recommendations:
            desc_lower = product['description'].lower()
            if any(word in desc_lower for word in ['shirt', 'clothing', 'apparel']):
                product_types.add('clothing')
            elif any(word in desc_lower for word in ['antique', 'vintage', 'collectible']):
                product_types.add('antiques')
            elif any(word in desc_lower for word in ['teapot', 'tea', 'kitchen']):
                product_types.add('kitchen')
            elif any(word in desc_lower for word in ['computer', 'tech', 'electronic']):
                product_types.add('electronics')

        if 'clothing' in product_types:
            response_parts.append("\n💡 Tip: Check size charts before ordering clothing items.")
        if 'antiques' in product_types:
            response_parts.append("\n💡 Tip: Antique items may have unique characteristics and limited availability.")
        if 'kitchen' in product_types:
            response_parts.append("\n💡 Tip: Kitchen items come with care instructions for best results.")
        if 'electronics' in product_types:
            response_parts.append("\n💡 Tip: Electronics include warranty and technical support.")

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
