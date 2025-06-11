"""
Vector Database Service Module

Handles Pinecone vector database operations for product similarity search.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone not available. Install with: pip install pinecone-client")


class VectorDatabase:
    """
    Vector database service using Pinecone for similarity search.
    """
    
    def __init__(self, api_key: str = None, environment: str = None, index_name: str = "ecommerce-products"):
        """
        Initialize vector database connection.
        
        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            index_name (str): Name of the Pinecone index
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = index_name
        self.index = None
        self.dimension = 512  # Default vector dimension
        
        if PINECONE_AVAILABLE and self.api_key:
            self.initialize_pinecone()
        else:
            print("Pinecone not configured. Using fallback local storage.")
            self.use_local_storage = True
            self.local_vectors = {}
    
    def initialize_pinecone(self):
        """Initialize Pinecone connection and index."""
        try:
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine'
                )
                print(f"Created Pinecone index: {self.index_name}")
            
            self.index = pinecone.Index(self.index_name)
            self.use_local_storage = False
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.use_local_storage = True
            self.local_vectors = {}
    
    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """
        Insert or update vectors in the database.
        
        Args:
            vectors (List[Tuple]): List of (id, vector, metadata) tuples
            
        Returns:
            bool: Success status
        """
        try:
            if self.use_local_storage:
                return self._upsert_local(vectors)
            
            # Prepare vectors for Pinecone
            pinecone_vectors = []
            for vector_id, vector, metadata in vectors:
                pinecone_vectors.append({
                    'id': str(vector_id),
                    'values': vector,
                    'metadata': metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
    
    def _upsert_local(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to local storage."""
        try:
            for vector_id, vector, metadata in vectors:
                self.local_vectors[str(vector_id)] = {
                    'vector': vector,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Save to file
            os.makedirs('data', exist_ok=True)
            with open('data/local_vectors.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_vectors = {}
                for k, v in self.local_vectors.items():
                    serializable_vectors[k] = {
                        'vector': v['vector'] if isinstance(v['vector'], list) else v['vector'].tolist(),
                        'metadata': v['metadata'],
                        'timestamp': v['timestamp']
                    }
                json.dump(serializable_vectors, f)
            
            print(f"Saved {len(vectors)} vectors locally")
            return True
            
        except Exception as e:
            print(f"Error saving vectors locally: {e}")
            return False
    
    def query_vectors(self, query_vector: List[float], top_k: int = 10, 
                     filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query similar vectors from the database.
        
        Args:
            query_vector (List[float]): Query vector
            top_k (int): Number of results to return
            filter_dict (Dict): Metadata filters
            
        Returns:
            List[Dict]: Similar vectors with metadata and scores
        """
        try:
            if self.use_local_storage:
                return self._query_local(query_vector, top_k, filter_dict)
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in query_response['matches']:
                results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                })
            
            return results
            
        except Exception as e:
            print(f"Error querying vectors: {e}")
            return []
    
    def _query_local(self, query_vector: List[float], top_k: int = 10, 
                    filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query vectors from local storage."""
        try:
            # Load local vectors if not in memory
            if not self.local_vectors:
                self._load_local_vectors()
            
            if not self.local_vectors:
                return []
            
            # Calculate similarities
            similarities = []
            query_vector = np.array(query_vector)
            
            for vector_id, data in self.local_vectors.items():
                stored_vector = np.array(data['vector'])
                
                # Apply filters if provided
                if filter_dict:
                    metadata = data['metadata']
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, stored_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
                )
                
                similarities.append({
                    'id': vector_id,
                    'score': float(similarity),
                    'metadata': data['metadata']
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error querying local vectors: {e}")
            return []
    
    def _load_local_vectors(self):
        """Load vectors from local storage."""
        try:
            if os.path.exists('data/local_vectors.json'):
                with open('data/local_vectors.json', 'r') as f:
                    self.local_vectors = json.load(f)
                print(f"Loaded {len(self.local_vectors)} vectors from local storage")
        except Exception as e:
            print(f"Error loading local vectors: {e}")
            self.local_vectors = {}
    
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from the database.
        
        Args:
            vector_ids (List[str]): List of vector IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            if self.use_local_storage:
                for vector_id in vector_ids:
                    self.local_vectors.pop(str(vector_id), None)
                return self._save_local_vectors()
            
            self.index.delete(ids=vector_ids)
            print(f"Deleted {len(vector_ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def _save_local_vectors(self) -> bool:
        """Save local vectors to file."""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/local_vectors.json', 'w') as f:
                json.dump(self.local_vectors, f)
            return True
        except Exception as e:
            print(f"Error saving local vectors: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dict: Index statistics
        """
        try:
            if self.use_local_storage:
                return {
                    'total_vectors': len(self.local_vectors),
                    'dimension': self.dimension,
                    'storage_type': 'local'
                }
            
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', self.dimension),
                'storage_type': 'pinecone'
            }
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {'error': str(e)}
    
    def create_product_vectors(self, products: List[Dict[str, Any]], 
                             vectorizer_func: callable) -> bool:
        """
        Create and store vectors for products.
        
        Args:
            products (List[Dict]): Product data
            vectorizer_func (callable): Function to convert product to vector
            
        Returns:
            bool: Success status
        """
        try:
            vectors = []
            for product in products:
                vector = vectorizer_func(product)
                if vector is not None:
                    vectors.append((
                        product['id'],
                        vector.tolist() if hasattr(vector, 'tolist') else vector,
                        {
                            'stock_code': product.get('stock_code', ''),
                            'description': product.get('description', ''),
                            'unit_price': product.get('unit_price', 0),
                            'country': product.get('country', '')
                        }
                    ))
            
            return self.upsert_vectors(vectors)
            
        except Exception as e:
            print(f"Error creating product vectors: {e}")
            return False


if __name__ == "__main__":
    # Test vector database
    vdb = VectorDatabase()
    stats = vdb.get_index_stats()
    print(f"Vector database stats: {stats}")
