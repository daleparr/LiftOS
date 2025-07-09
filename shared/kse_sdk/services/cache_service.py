"""
KSE Memory SDK Cache Service
"""

from typing import Dict, Any, Optional, List
import asyncio
import time
import logging
from ..core.interfaces import CacheInterface

logger = logging.getLogger(__name__)


class CacheService(CacheInterface):
    """
    In-memory cache service with TTL support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = config.get('default_ttl', 3600)  # 1 hour default
        self.max_size = config.get('max_size', 10000)
        self._cleanup_interval = config.get('cleanup_interval', 300)  # 5 minutes
        self._cleanup_task = None
        
    async def connect(self) -> bool:
        """Connect and initialize the cache service."""
        return await self.initialize()
        
    async def initialize(self) -> bool:
        """Initialize cache service."""
        try:
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
            logger.info("Cache service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return None
        
        # Update access time
        entry['accessed_at'] = time.time()
        return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            # Check cache size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k]['accessed_at'])
                del self.cache[oldest_key]
            
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'accessed_at': time.time(),
                'created_at': time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache value: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            if key in self.cache:
                del self.cache[key]
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache value: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and not expired
        """
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return False
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total_entries = len(self.cache)
        expired_count = 0
        
        current_time = time.time()
        for entry in self.cache.values():
            if current_time > entry['expires_at']:
                expired_count += 1
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_count,
            'active_entries': total_entries - expired_count,
            'max_size': self.max_size,
            'default_ttl': self.default_ttl
        }
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs for found keys
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            items: Dictionary of key-value pairs to cache
            ttl: Time to live in seconds
            
        Returns:
            True if all items were set successfully
        """
        try:
            for key, value in items.items():
                success = await self.set(key, value, ttl)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to batch set cache values: {e}")
            return False
    
    async def _cleanup_expired(self):
        """Background task to cleanup expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.cache.items():
                    if current_time > entry['expires_at']:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def shutdown(self):
        """Shutdown cache service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.cache.clear()
        logger.info("Cache service shutdown")