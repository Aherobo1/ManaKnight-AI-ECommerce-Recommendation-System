"""
🚀 Caching Service - Mana Knight Digital

High-performance caching layer with Redis fallback to in-memory cache.
Features: TTL support, cache invalidation, performance metrics.
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheService:
    """
    🎯 Professional caching service with Redis and in-memory fallback.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        """
        Initialize cache service.
        
        Args:
            redis_url (str): Redis connection URL
            default_ttl (int): Default TTL in seconds
        """
        self.default_ttl = default_ttl
        self.redis_client = None
        self.use_redis = False
        
        # In-memory cache fallback
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        
        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.use_redis = True
                logger.info("✅ Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory cache: {e}")
        else:
            logger.info("Redis not installed, using in-memory cache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None
        """
        try:
            if self.use_redis:
                value = self.redis_client.get(key)
                if value is not None:
                    self.hits += 1
                    return json.loads(value)
            else:
                # Check in-memory cache
                if key in self.memory_cache:
                    # Check if expired
                    if self._is_expired(key):
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                    else:
                        self.hits += 1
                        return self.memory_cache[key]
            
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (Optional[int]): TTL in seconds
            
        Returns:
            bool: Success status
        """
        try:
            ttl = ttl or self.default_ttl
            
            if self.use_redis:
                serialized_value = json.dumps(value, default=str)
                result = self.redis_client.setex(key, ttl, serialized_value)
                if result:
                    self.sets += 1
                return result
            else:
                # Store in memory cache
                self.memory_cache[key] = value
                self.cache_timestamps[key] = {
                    'created': datetime.now(),
                    'ttl': ttl
                }
                self.sets += 1
                
                # Clean up expired entries periodically
                self._cleanup_expired()
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: Success status
        """
        try:
            if self.use_redis:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key exists
        """
        try:
            if self.use_redis:
                return bool(self.redis_client.exists(key))
            else:
                if key in self.memory_cache:
                    if self._is_expired(key):
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                        return False
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: Success status
        """
        try:
            if self.use_redis:
                return bool(self.redis_client.flushdb())
            else:
                self.memory_cache.clear()
                self.cache_timestamps.clear()
                return True
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'cache_type': 'redis' if self.use_redis else 'memory',
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
        
        if self.use_redis:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'Unknown'),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_uptime': info.get('uptime_in_seconds', 0)
                })
            except:
                pass
        else:
            stats.update({
                'memory_entries': len(self.memory_cache),
                'memory_size_mb': self._get_memory_size()
            })
        
        return stats
    
    def cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Generated cache key
        """
        # Create a string representation of all arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Generate hash
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator for caching function results.
        
        Args:
            ttl (Optional[int]): TTL in seconds
            key_prefix (str): Prefix for cache key
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:{self.cache_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _is_expired(self, key: str) -> bool:
        """Check if memory cache entry is expired."""
        if key not in self.cache_timestamps:
            return True
        
        timestamp_info = self.cache_timestamps[key]
        created = timestamp_info['created']
        ttl = timestamp_info['ttl']
        
        return datetime.now() > created + timedelta(seconds=ttl)
    
    def _cleanup_expired(self):
        """Clean up expired entries from memory cache."""
        if len(self.memory_cache) > 1000:  # Only cleanup when cache is large
            expired_keys = [
                key for key in self.memory_cache.keys()
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
                del self.cache_timestamps[key]
    
    def _get_memory_size(self) -> float:
        """Get approximate memory size of cache in MB."""
        try:
            import sys
            total_size = 0
            for key, value in self.memory_cache.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0


# Global cache service instance
cache_service = CacheService()


def cached(ttl: Optional[int] = None, key_prefix: str = "api"):
    """
    Convenient decorator for caching API responses.
    
    Args:
        ttl (Optional[int]): TTL in seconds
        key_prefix (str): Prefix for cache key
    """
    return cache_service.cached(ttl, key_prefix)


if __name__ == "__main__":
    # Test cache service
    cache = CacheService()
    
    # Test basic operations
    cache.set("test_key", {"data": "test_value"}, ttl=60)
    result = cache.get("test_key")
    print(f"Cached result: {result}")
    
    # Test decorator
    @cache.cached(ttl=30, key_prefix="test")
    def expensive_function(x, y):
        time.sleep(1)  # Simulate expensive operation
        return x + y
    
    # First call (slow)
    start = time.time()
    result1 = expensive_function(1, 2)
    time1 = time.time() - start
    
    # Second call (fast, from cache)
    start = time.time()
    result2 = expensive_function(1, 2)
    time2 = time.time() - start
    
    print(f"First call: {result1} in {time1:.3f}s")
    print(f"Second call: {result2} in {time2:.3f}s")
    
    # Get stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
