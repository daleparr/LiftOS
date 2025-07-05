"""
Enhanced KSE Memory Integration - Adapted from MMM Spine
Provides universal memory substrate with advanced embedding management
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import json
import uuid
from datetime import datetime

from ..kse_sdk.client import kse_client
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for neural embeddings with metadata."""
    embedding: np.ndarray
    text: str
    model: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class ConceptualSpace:
    """Represents a conceptual space for knowledge organization."""
    space_id: str
    name: str
    dimensions: List[str]
    concepts: Dict[str, Any]
    relationships: Dict[str, List[str]]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class MemoryOperation:
    """Represents a memory operation for tracking and optimization."""
    operation_id: str
    operation_type: str
    org_id: str
    duration_ns: int
    success: bool
    cache_hit: bool = False
    embedding_model: Optional[str] = None
    conceptual_space: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class EmbeddingManager:
    """
    Advanced embedding management with intelligent caching and optimization.
    Adapted from MMM Spine embedding_manager.py
    """
    
    def __init__(self, cache_size: int = 10000, cache_ttl: int = 3600):
        """
        Initialize embedding manager with caching capabilities.
        
        Args:
            cache_size: Maximum number of embeddings to cache
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Embedding cache with LRU eviction
        self.embedding_cache: Dict[str, EmbeddingCacheEntry] = {}
        self.cache_access_order: deque = deque(maxlen=cache_size)
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_operations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized EmbeddingManager with cache_size={cache_size}, ttl={cache_ttl}s")
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        return f"{model}:{hash(text)}"
    
    def _is_cache_entry_valid(self, entry: EmbeddingCacheEntry) -> bool:
        """Check if cache entry is still valid based on TTL."""
        return (time.time() - entry.timestamp) < self.cache_ttl
    
    def _evict_lru_entry(self):
        """Evict least recently used cache entry."""
        if self.cache_access_order:
            lru_key = self.cache_access_order.popleft()
            if lru_key in self.embedding_cache:
                del self.embedding_cache[lru_key]
    
    async def get_embedding(self, text: str, model: str = "default") -> np.ndarray:
        """
        Get embedding for text with intelligent caching.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector as numpy array
        """
        start_time = time.time_ns()
        cache_key = self._generate_cache_key(text, model)
        
        with self.lock:
            self.total_operations += 1
            
            # Check cache first
            if cache_key in self.embedding_cache:
                entry = self.embedding_cache[cache_key]
                if self._is_cache_entry_valid(entry):
                    # Cache hit
                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    
                    # Update access order
                    if cache_key in self.cache_access_order:
                        self.cache_access_order.remove(cache_key)
                    self.cache_access_order.append(cache_key)
                    
                    self.cache_hits += 1
                    duration = time.time_ns() - start_time
                    
                    logger.debug(f"Embedding cache hit for model={model}, duration={duration/1e6:.2f}ms")
                    return entry.embedding.copy()
                else:
                    # Cache entry expired
                    del self.embedding_cache[cache_key]
            
            # Cache miss - generate embedding
            self.cache_misses += 1
        
        try:
            # Generate embedding using KSE client
            embedding = await kse_client.generate_embedding(text, model)
            embedding_array = np.array(embedding)
            
            with self.lock:
                # Store in cache
                if len(self.embedding_cache) >= self.cache_size:
                    self._evict_lru_entry()
                
                cache_entry = EmbeddingCacheEntry(
                    embedding=embedding_array.copy(),
                    text=text,
                    model=model,
                    timestamp=time.time(),
                    access_count=1
                )
                
                self.embedding_cache[cache_key] = cache_entry
                self.cache_access_order.append(cache_key)
            
            duration = time.time_ns() - start_time
            logger.debug(f"Generated embedding for model={model}, duration={duration/1e6:.2f}ms")
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    async def batch_embeddings(self, texts: List[str], model: str = "default") -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        start_time = time.time_ns()
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        
        with self.lock:
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text, model)
                if cache_key in self.embedding_cache:
                    entry = self.embedding_cache[cache_key]
                    if self._is_cache_entry_valid(entry):
                        cached_embeddings[i] = entry.embedding.copy()
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        self.cache_hits += 1
                        continue
                
                uncached_texts.append((i, text))
                self.cache_misses += 1
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                batch_texts = [text for _, text in uncached_texts]
                batch_embeddings = await kse_client.batch_embeddings(batch_texts, model)
                
                # Cache new embeddings
                with self.lock:
                    for (i, text), embedding in zip(uncached_texts, batch_embeddings):
                        embedding_array = np.array(embedding)
                        
                        if len(self.embedding_cache) >= self.cache_size:
                            self._evict_lru_entry()
                        
                        cache_key = self._generate_cache_key(text, model)
                        cache_entry = EmbeddingCacheEntry(
                            embedding=embedding_array.copy(),
                            text=text,
                            model=model,
                            timestamp=time.time(),
                            access_count=1
                        )
                        
                        self.embedding_cache[cache_key] = cache_entry
                        self.cache_access_order.append(cache_key)
                        cached_embeddings[i] = embedding_array
                        
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {str(e)}")
                raise
        
        # Reconstruct results in original order
        results = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                results.append(cached_embeddings[i])
            else:
                # Fallback for any missing embeddings
                results.append(await self.get_embedding(texts[i], model))
        
        duration = time.time_ns() - start_time
        cache_hit_rate = len(cached_embeddings) / len(texts) if texts else 0
        
        logger.info(f"Batch embeddings completed: {len(texts)} texts, "
                   f"cache_hit_rate={cache_hit_rate:.2%}, duration={duration/1e6:.2f}ms")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        with self.lock:
            hit_rate = self.cache_hits / self.total_operations if self.total_operations > 0 else 0
            
            return {
                "cache_size": len(self.embedding_cache),
                "max_cache_size": self.cache_size,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_operations": self.total_operations,
                "hit_rate": hit_rate,
                "cache_utilization": len(self.embedding_cache) / self.cache_size
            }
    
    def clear_cache(self):
        """Clear embedding cache."""
        with self.lock:
            self.embedding_cache.clear()
            self.cache_access_order.clear()
            logger.info("Embedding cache cleared")


class EnhancedKSEIntegration:
    """
    Enhanced KSE integration with conceptual spaces and advanced memory operations.
    Adapted from MMM Spine kse_integration.py
    """
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize enhanced KSE integration.
        
        Args:
            embedding_manager: Optional embedding manager instance
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.conceptual_spaces: Dict[str, ConceptualSpace] = {}
        self.operation_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.total_operations = 0
        self.successful_operations = 0
        self.average_operation_time = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized EnhancedKSEIntegration")
    
    async def create_conceptual_space(self, org_id: str, name: str, 
                                    dimensions: List[str]) -> str:
        """
        Create a new conceptual space for knowledge organization.
        
        Args:
            org_id: Organization ID
            name: Name of the conceptual space
            dimensions: List of dimensional concepts
            
        Returns:
            Conceptual space ID
        """
        space_id = f"space_{org_id}_{uuid.uuid4().hex[:8]}"
        
        conceptual_space = ConceptualSpace(
            space_id=space_id,
            name=name,
            dimensions=dimensions,
            concepts={},
            relationships={}
        )
        
        with self.lock:
            self.conceptual_spaces[space_id] = conceptual_space
        
        logger.info(f"Created conceptual space: {space_id} for org {org_id}")
        return space_id
    
    async def add_concept_to_space(self, space_id: str, concept_id: str, 
                                 concept_data: Dict[str, Any]) -> bool:
        """
        Add a concept to a conceptual space.
        
        Args:
            space_id: Conceptual space ID
            concept_id: Unique concept identifier
            concept_data: Concept data and metadata
            
        Returns:
            Success status
        """
        if space_id not in self.conceptual_spaces:
            logger.error(f"Conceptual space not found: {space_id}")
            return False
        
        with self.lock:
            space = self.conceptual_spaces[space_id]
            space.concepts[concept_id] = concept_data
            space.updated_at = time.time()
        
        logger.debug(f"Added concept {concept_id} to space {space_id}")
        return True
    
    async def enhanced_memory_search(self, org_id: str, query: str, 
                                   search_type: str = "hybrid",
                                   conceptual_space: Optional[str] = None,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Enhanced memory search with conceptual space filtering.
        
        Args:
            org_id: Organization ID
            query: Search query
            search_type: Type of search (hybrid, neural, conceptual, knowledge)
            conceptual_space: Optional conceptual space to search within
            limit: Maximum number of results
            
        Returns:
            List of search results with enhanced metadata
        """
        start_time = time.time_ns()
        operation_id = str(uuid.uuid4())
        
        try:
            # Generate query embedding for neural search
            if search_type in ["hybrid", "neural"]:
                query_embedding = await self.embedding_manager.get_embedding(query)
            
            # Perform base KSE search
            base_results = await kse_client.hybrid_search(
                query=query,
                org_id=org_id,
                limit=limit * 2,  # Get more results for filtering
                search_type=search_type
            )
            
            # Enhanced filtering and ranking
            enhanced_results = []
            for result in base_results:
                enhanced_result = result.dict() if hasattr(result, 'dict') else result
                
                # Add conceptual space information if available
                if conceptual_space and conceptual_space in self.conceptual_spaces:
                    space = self.conceptual_spaces[conceptual_space]
                    enhanced_result["conceptual_space"] = {
                        "space_id": space.space_id,
                        "name": space.name,
                        "dimensions": space.dimensions
                    }
                
                # Add embedding similarity if neural search
                if search_type in ["hybrid", "neural"] and "embedding" in enhanced_result:
                    try:
                        result_embedding = np.array(enhanced_result["embedding"])
                        similarity = np.dot(query_embedding, result_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                        )
                        enhanced_result["embedding_similarity"] = float(similarity)
                    except Exception as e:
                        logger.warning(f"Failed to calculate embedding similarity: {e}")
                
                enhanced_results.append(enhanced_result)
            
            # Sort by relevance and limit results
            if search_type in ["hybrid", "neural"]:
                enhanced_results.sort(
                    key=lambda x: x.get("embedding_similarity", 0), 
                    reverse=True
                )
            
            final_results = enhanced_results[:limit]
            
            # Record operation
            duration = time.time_ns() - start_time
            operation = MemoryOperation(
                operation_id=operation_id,
                operation_type="enhanced_search",
                org_id=org_id,
                duration_ns=duration,
                success=True,
                conceptual_space=conceptual_space
            )
            
            self._record_operation(operation)
            
            logger.info(f"Enhanced memory search completed: {len(final_results)} results, "
                       f"duration={duration/1e6:.2f}ms")
            
            return final_results
            
        except Exception as e:
            duration = time.time_ns() - start_time
            operation = MemoryOperation(
                operation_id=operation_id,
                operation_type="enhanced_search",
                org_id=org_id,
                duration_ns=duration,
                success=False,
                conceptual_space=conceptual_space
            )
            
            self._record_operation(operation)
            logger.error(f"Enhanced memory search failed: {str(e)}")
            raise
    
    async def store_with_conceptual_mapping(self, org_id: str, content: str,
                                          metadata: Dict[str, Any],
                                          conceptual_space: Optional[str] = None,
                                          memory_type: str = "general") -> str:
        """
        Store memory with conceptual space mapping.
        
        Args:
            org_id: Organization ID
            content: Content to store
            metadata: Memory metadata
            conceptual_space: Optional conceptual space ID
            memory_type: Type of memory
            
        Returns:
            Memory ID
        """
        start_time = time.time_ns()
        operation_id = str(uuid.uuid4())
        
        try:
            # Generate embedding for content
            content_embedding = await self.embedding_manager.get_embedding(content)
            
            # Enhanced metadata with conceptual mapping
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                "has_embedding": True,
                "embedding_model": "default",
                "operation_id": operation_id
            })
            
            if conceptual_space and conceptual_space in self.conceptual_spaces:
                space = self.conceptual_spaces[conceptual_space]
                enhanced_metadata.update({
                    "conceptual_space_id": space.space_id,
                    "conceptual_space_name": space.name,
                    "conceptual_dimensions": space.dimensions
                })
            
            # Store in KSE with enhanced metadata
            memory_id = await kse_client.store_memory(
                org_id=org_id,
                content=content,
                metadata=enhanced_metadata,
                memory_type=memory_type
            )
            
            # Record operation
            duration = time.time_ns() - start_time
            operation = MemoryOperation(
                operation_id=operation_id,
                operation_type="enhanced_store",
                org_id=org_id,
                duration_ns=duration,
                success=True,
                conceptual_space=conceptual_space
            )
            
            self._record_operation(operation)
            
            logger.info(f"Enhanced memory storage completed: memory_id={memory_id}, "
                       f"duration={duration/1e6:.2f}ms")
            
            return memory_id
            
        except Exception as e:
            duration = time.time_ns() - start_time
            operation = MemoryOperation(
                operation_id=operation_id,
                operation_type="enhanced_store",
                org_id=org_id,
                duration_ns=duration,
                success=False,
                conceptual_space=conceptual_space
            )
            
            self._record_operation(operation)
            logger.error(f"Enhanced memory storage failed: {str(e)}")
            raise
    
    def _record_operation(self, operation: MemoryOperation):
        """Record memory operation for performance tracking."""
        with self.lock:
            self.operation_history.append(operation)
            self.total_operations += 1
            
            if operation.success:
                self.successful_operations += 1
            
            # Update average operation time
            total_time = sum(op.duration_ns for op in self.operation_history)
            self.average_operation_time = total_time / len(self.operation_history)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for enhanced KSE operations."""
        with self.lock:
            success_rate = (self.successful_operations / self.total_operations 
                          if self.total_operations > 0 else 0)
            
            recent_operations = list(self.operation_history)[-100:]  # Last 100 operations
            recent_avg_time = (sum(op.duration_ns for op in recent_operations) / 
                             len(recent_operations) if recent_operations else 0)
            
            return {
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "success_rate": success_rate,
                "average_operation_time_ms": self.average_operation_time / 1e6,
                "recent_average_time_ms": recent_avg_time / 1e6,
                "conceptual_spaces_count": len(self.conceptual_spaces),
                "embedding_cache_stats": self.embedding_manager.get_cache_stats()
            }
    
    def get_conceptual_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all conceptual spaces."""
        with self.lock:
            return {
                space_id: {
                    "name": space.name,
                    "dimensions": space.dimensions,
                    "concepts_count": len(space.concepts),
                    "relationships_count": len(space.relationships),
                    "created_at": space.created_at,
                    "updated_at": space.updated_at
                }
                for space_id, space in self.conceptual_spaces.items()
            }


class MemoryInterface:
    """
    Abstract interface for memory operations with enhanced capabilities.
    Provides consistent API for different memory backends.
    """
    
    def __init__(self, enhanced_kse: Optional[EnhancedKSEIntegration] = None):
        """
        Initialize memory interface.
        
        Args:
            enhanced_kse: Optional enhanced KSE integration instance
        """
        self.enhanced_kse = enhanced_kse or EnhancedKSEIntegration()
        logger.info("Initialized MemoryInterface")
    
    async def search(self, org_id: str, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Unified search interface with enhanced capabilities.
        
        Args:
            org_id: Organization ID
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        return await self.enhanced_kse.enhanced_memory_search(
            org_id=org_id,
            query=query,
            **kwargs
        )
    
    async def store(self, org_id: str, content: str, metadata: Dict[str, Any], 
                   **kwargs) -> str:
        """
        Unified storage interface with enhanced capabilities.
        
        Args:
            org_id: Organization ID
            content: Content to store
            metadata: Memory metadata
            **kwargs: Additional storage parameters
            
        Returns:
            Memory ID
        """
        return await self.enhanced_kse.store_with_conceptual_mapping(
            org_id=org_id,
            content=content,
            metadata=metadata,
            **kwargs
        )
    
    async def create_conceptual_space(self, org_id: str, name: str, 
                                    dimensions: List[str]) -> str:
        """
        Create conceptual space for knowledge organization.
        
        Args:
            org_id: Organization ID
            name: Space name
            dimensions: Dimensional concepts
            
        Returns:
            Conceptual space ID
        """
        return await self.enhanced_kse.create_conceptual_space(
            org_id=org_id,
            name=name,
            dimensions=dimensions
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return self.enhanced_kse.get_performance_stats()
    
    def get_conceptual_spaces_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about conceptual spaces."""
        return self.enhanced_kse.get_conceptual_spaces()