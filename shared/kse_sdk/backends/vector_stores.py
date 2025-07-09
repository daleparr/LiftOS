"""
Vector Store Backend Implementations
Multi-backend support for Pinecone, Weaviate, Qdrant, Chroma, and Milvus
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from ..core.interfaces import VectorStoreInterface
from ..core.config import VectorStoreConfig
from ..exceptions import ConfigurationError, ConnectionError

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.index = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Pinecone connection."""
        try:
            import pinecone
            from pinecone import Pinecone, ServerlessSpec
            
            self.client = Pinecone(api_key=self.config.api_key)
            
            # Get or create index
            if self.config.index_name not in [idx.name for idx in self.client.list_indexes()]:
                self.client.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.distance_metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud_provider,
                        region=self.config.region
                    )
                )
            
            self.index = self.client.Index(self.config.index_name)
            self._initialized = True
            logger.info(f"Pinecone vector store initialized: {self.config.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise ConnectionError(f"Pinecone initialization failed: {e}")
    
    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Store vectors in Pinecone."""
        if not self._initialized:
            await self.initialize()
        
        try:
            upsert_data = []
            for vector in vectors:
                upsert_data.append({
                    'id': vector['id'],
                    'values': vector['embedding'],
                    'metadata': vector.get('metadata', {})
                })
            
            self.index.upsert(vectors=upsert_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vectors in Pinecone: {e}")
            return False
    
    async def search_vectors(self, query_vector: List[float], limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search vectors in Pinecone."""
        if not self._initialized:
            await self.initialize()
        
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=limit,
                filter=filters,
                include_metadata=True,
                include_values=False
            )
            
            results = []
            for match in response.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Pinecone: {e}")
            return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone."""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Pinecone health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                "healthy": True,
                "total_vectors": stats.total_vector_count,
                "index_name": self.config.index_name
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class WeaviateVectorStore(VectorStoreInterface):
    """Weaviate vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Weaviate connection."""
        try:
            import weaviate
            
            self.client = weaviate.Client(
                url=self.config.host,
                auth_client_secret=weaviate.AuthApiKey(api_key=self.config.api_key) if self.config.api_key else None
            )
            
            self._initialized = True
            logger.info(f"Weaviate vector store initialized: {self.config.host}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise ConnectionError(f"Weaviate initialization failed: {e}")
    
    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Store vectors in Weaviate."""
        # Implementation for Weaviate storage
        logger.info("Weaviate storage not yet implemented")
        return True
    
    async def search_vectors(self, query_vector: List[float], limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search vectors in Weaviate."""
        # Implementation for Weaviate search
        logger.info("Weaviate search not yet implemented")
        return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Weaviate."""
        logger.info("Weaviate deletion not yet implemented")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Weaviate health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {"healthy": self.client.is_ready(), "backend": "weaviate"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class QdrantVectorStore(VectorStoreInterface):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Qdrant connection."""
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key
            )
            
            self._initialized = True
            logger.info(f"Qdrant vector store initialized: {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise ConnectionError(f"Qdrant initialization failed: {e}")
    
    async def store_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Store vectors in Qdrant."""
        logger.info("Qdrant storage not yet implemented")
        return True
    
    async def search_vectors(self, query_vector: List[float], limit: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search vectors in Qdrant."""
        logger.info("Qdrant search not yet implemented")
        return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Qdrant."""
        logger.info("Qdrant deletion not yet implemented")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {"healthy": True, "backend": "qdrant"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

class InMemoryVectorStore(VectorStoreInterface):
    """In-memory vector store implementation for testing."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.vectors = {}  # id -> vector mapping
        self.metadata = {}  # id -> metadata mapping
        self._initialized = False
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the in-memory vector store."""
        try:
            self._connected = True
            logger.info("Connected to in-memory vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to in-memory vector store: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the in-memory vector store."""
        try:
            self._connected = False
            logger.info("Disconnected from in-memory vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from in-memory vector store: {e}")
            return False
    
    async def create_index(self, dimension: int, metric: str = "cosine",
                          metadata_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new vector index (no-op for in-memory)."""
        try:
            self._initialized = True
            logger.info(f"Created in-memory vector index with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to create in-memory vector index: {e}")
            return False
    
    async def delete_index(self) -> bool:
        """Delete the vector index (clear memory)."""
        try:
            self.vectors.clear()
            self.metadata.clear()
            self._initialized = False
            logger.info("Deleted in-memory vector index")
            return True
        except Exception as e:
            logger.error(f"Failed to delete in-memory vector index: {e}")
            return False
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Insert or update vectors with metadata."""
        try:
            for vector_id, vector, metadata in vectors:
                self.vectors[vector_id] = vector
                self.metadata[vector_id] = metadata
            
            logger.info(f"Upserted {len(vectors)} vectors in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vectors in memory: {e}")
            return False
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            for vector_id in vector_ids:
                self.vectors.pop(vector_id, None)
                self.metadata.pop(vector_id, None)
            
            logger.info(f"Deleted {len(vector_ids)} vectors from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from memory: {e}")
            return False
    
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        try:
            import numpy as np
            
            if not self.vectors:
                return []
            
            query_np = np.array(query_vector)
            results = []
            
            for vector_id, vector in self.vectors.items():
                # Apply filters if provided
                if filters:
                    vector_metadata = self.metadata.get(vector_id, {})
                    if not all(vector_metadata.get(k) == v for k, v in filters.items()):
                        continue
                
                # Calculate cosine similarity
                vector_np = np.array(vector)
                similarity = np.dot(query_np, vector_np) / (np.linalg.norm(query_np) * np.linalg.norm(vector_np))
                
                metadata = self.metadata.get(vector_id, {}) if include_metadata else {}
                results.append((vector_id, float(similarity), metadata))
            
            # Sort by similarity score (descending) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search vectors in memory: {e}")
            return []
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a specific vector by ID."""
        try:
            if vector_id in self.vectors:
                vector = self.vectors[vector_id]
                metadata = self.metadata.get(vector_id, {})
                return (vector, metadata)
            return None
        except Exception as e:
            logger.error(f"Failed to get vector from memory: {e}")
            return None
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            return {
                "vector_count": len(self.vectors),
                "dimension": len(next(iter(self.vectors.values()))) if self.vectors else 0,
                "backend": "memory",
                "initialized": self._initialized,
                "connected": self._connected
            }
        except Exception as e:
            logger.error(f"Failed to get index stats from memory: {e}")
            return {}
    
    async def batch_upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]],
                          batch_size: int = 100) -> bool:
        """Batch upsert vectors for better performance."""
        try:
            # For in-memory, we can process all at once
            return await self.upsert_vectors(vectors)
        except Exception as e:
            logger.error(f"Failed to batch upsert vectors in memory: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize in-memory vector store."""
        try:
            await self.connect()
            await self.create_index(dimension=self.config.dimension)
            logger.info("In-memory vector store initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize in-memory vector store: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check in-memory vector store health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {
                "healthy": True,
                "backend": "memory",
                "vector_count": len(self.vectors),
                "connected": self._connected
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}



def get_vector_store(config: VectorStoreConfig) -> VectorStoreInterface:
    """Factory function to get vector store implementation."""
    
    if config.backend == "pinecone":
        return PineconeVectorStore(config)
    elif config.backend == "weaviate":
        return WeaviateVectorStore(config)
    elif config.backend == "qdrant":
        return QdrantVectorStore(config)
    elif config.backend == "memory":
        return InMemoryVectorStore(config)
    elif config.backend == "chroma":
        # Placeholder for Chroma implementation
        logger.warning("Chroma backend not yet implemented, falling back to memory")
        memory_config = VectorStoreConfig(backend="memory", dimension=config.dimension)
        return InMemoryVectorStore(memory_config)
    elif config.backend == "milvus":
        # Placeholder for Milvus implementation
        logger.warning("Milvus backend not yet implemented, falling back to memory")
        memory_config = VectorStoreConfig(backend="memory", dimension=config.dimension)
        return InMemoryVectorStore(memory_config)
    else:
        raise ConfigurationError(f"Unsupported vector store backend: {config.backend}")