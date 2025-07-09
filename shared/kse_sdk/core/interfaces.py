"""
Core interfaces for KSE Memory SDK components.
Universal interfaces supporting all domains and backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from .models import Entity, SearchQuery, SearchResult, ConceptualSpace, EmbeddingVector, KnowledgeGraph


class AdapterInterface(ABC):
    """Interface for platform adapters (Shopify, WooCommerce, Healthcare systems, etc.)."""
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to the platform."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the platform."""
        pass
    
    @abstractmethod
    async def get_entities(self, limit: int = 100, offset: int = 0, 
                          entity_type: Optional[str] = None) -> List[Entity]:
        """Retrieve entities from the platform."""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve a specific entity by ID."""
        pass
    
    @abstractmethod
    async def sync_entities(self, entity_type: Optional[str] = None) -> int:
        """Sync all entities from the platform."""
        pass
    
    @abstractmethod
    async def webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Handle webhook events from the platform."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the platform."""
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """Get list of supported entity types for this adapter."""
        pass


class VectorStoreInterface(ABC):
    """Interface for vector storage backends (Pinecone, Weaviate, Qdrant, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the vector store."""
        pass
    
    @abstractmethod
    async def create_index(self, dimension: int, metric: str = "cosine", 
                          metadata_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new vector index."""
        pass
    
    @abstractmethod
    async def delete_index(self) -> bool:
        """Delete the vector index."""
        pass
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Insert or update vectors with metadata."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a specific vector by ID."""
        pass
    
    @abstractmethod
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass
    
    @abstractmethod
    async def batch_upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]], 
                          batch_size: int = 100) -> bool:
        """Batch upsert vectors for better performance."""
        pass


class GraphStoreInterface(ABC):
    """Interface for graph storage backends (Neo4j, ArangoDB, Neptune, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the graph store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the graph store."""
        pass
    
    @abstractmethod
    async def create_node(self, node_id: str, labels: List[str], 
                         properties: Dict[str, Any]) -> bool:
        """Create a node in the graph."""
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph."""
        pass
    
    @abstractmethod
    async def create_relationship(self, source_id: str, target_id: str, 
                                 relationship_type: str, 
                                 properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, source_id: str, target_id: str, 
                                 relationship_type: str) -> bool:
        """Delete a relationship."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, node_id: str, relationship_types: Optional[List[str]] = None,
                           direction: str = "both", limit: int = 100) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        pass
    
    @abstractmethod
    async def find_path(self, source_id: str, target_id: str, 
                       max_depth: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Find path between two nodes."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom graph query."""
        pass
    
    @abstractmethod
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        pass


class ConceptStoreInterface(ABC):
    """Interface for concept storage backends (PostgreSQL, MongoDB, Elasticsearch, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the concept store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the concept store."""
        pass
    
    @abstractmethod
    async def store_conceptual_space(self, entity_id: str, 
                                   conceptual_space: ConceptualSpace) -> bool:
        """Store conceptual space for an entity."""
        pass
    
    @abstractmethod
    async def get_conceptual_space(self, entity_id: str) -> Optional[ConceptualSpace]:
        """Get conceptual space for an entity."""
        pass
    
    @abstractmethod
    async def update_conceptual_space(self, entity_id: str, 
                                    conceptual_space: ConceptualSpace) -> bool:
        """Update conceptual space for an entity."""
        pass
    
    @abstractmethod
    async def delete_conceptual_space(self, entity_id: str) -> bool:
        """Delete conceptual space for an entity."""
        pass
    
    @abstractmethod
    async def search_by_dimensions(self, dimension_filters: Dict[str, Tuple[float, float]], 
                                  domain: Optional[str] = None, 
                                  limit: int = 100) -> List[Tuple[str, ConceptualSpace, float]]:
        """Search entities by conceptual dimensions."""
        pass
    
    @abstractmethod
    async def get_dimension_statistics(self, domain: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for conceptual dimensions."""
        pass
    
    @abstractmethod
    async def batch_store_conceptual_spaces(self, 
                                          spaces: List[Tuple[str, ConceptualSpace]]) -> bool:
        """Batch store conceptual spaces."""
        pass


class EmbeddingServiceInterface(ABC):
    """Interface for embedding generation services."""
    
    @abstractmethod
    async def generate_text_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingVector:
        """Generate text embedding."""
        pass
    
    @abstractmethod
    async def generate_image_embedding(self, image_url: str, model: Optional[str] = None) -> EmbeddingVector:
        """Generate image embedding."""
        pass
    
    @abstractmethod
    async def batch_text_embeddings(self, texts: List[str], 
                                   model: Optional[str] = None) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def batch_image_embeddings(self, image_urls: List[str], 
                                    model: Optional[str] = None) -> List[EmbeddingVector]:
        """Generate embeddings for multiple images."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported embedding models."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        pass


class ConceptualServiceInterface(ABC):
    """Interface for conceptual analysis services."""
    
    @abstractmethod
    async def compute_dimensions(self, entity: Entity, 
                               custom_dimensions: Optional[Dict[str, str]] = None) -> ConceptualSpace:
        """Compute conceptual dimensions for an entity."""
        pass
    
    @abstractmethod
    async def batch_compute_dimensions(self, entities: List[Entity], 
                                     custom_dimensions: Optional[Dict[str, str]] = None) -> List[ConceptualSpace]:
        """Compute conceptual dimensions for multiple entities."""
        pass
    
    @abstractmethod
    async def analyze_dimension_relationships(self, domain: str) -> Dict[str, Dict[str, float]]:
        """Analyze relationships between conceptual dimensions."""
        pass
    
    @abstractmethod
    async def suggest_dimensions(self, entities: List[Entity], 
                               domain: str) -> Dict[str, str]:
        """Suggest relevant dimensions for a domain based on entities."""
        pass
    
    @abstractmethod
    async def validate_dimensions(self, conceptual_space: ConceptualSpace) -> Dict[str, Any]:
        """Validate conceptual space dimensions."""
        pass


class SearchServiceInterface(ABC):
    """Interface for search services."""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search based on query."""
        pass
    
    @abstractmethod
    async def semantic_search(self, query_text: str, domain: Optional[str] = None, 
                            limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic search."""
        pass
    
    @abstractmethod
    async def conceptual_search(self, dimension_filters: Dict[str, float], 
                              domain: Optional[str] = None, 
                              limit: int = 10) -> List[SearchResult]:
        """Perform conceptual search."""
        pass
    
    @abstractmethod
    async def graph_search(self, entity_id: str, relationship_types: Optional[List[str]] = None,
                          max_depth: int = 2, limit: int = 10) -> List[SearchResult]:
        """Perform graph-based search."""
        pass
    
    @abstractmethod
    async def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining multiple approaches."""
        pass
    
    @abstractmethod
    async def rerank_results(self, results: List[SearchResult], 
                           query: SearchQuery) -> List[SearchResult]:
        """Rerank search results."""
        pass


class CacheInterface(ABC):
    """Interface for caching services."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
    
    @abstractmethod
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        pass


class AnalyticsInterface(ABC):
    """Interface for analytics and monitoring services."""
    
    @abstractmethod
    async def track_search(self, query: SearchQuery, results: List[SearchResult], 
                          duration_ms: float) -> bool:
        """Track search operation."""
        pass
    
    @abstractmethod
    async def track_entity_operation(self, operation: str, entity_id: str, 
                                   duration_ms: float, success: bool) -> bool:
        """Track entity operation."""
        pass
    
    @abstractmethod
    async def get_search_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get search analytics."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get performance metrics."""
        pass
    
    @abstractmethod
    async def get_usage_statistics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get usage statistics."""
        pass


class SecurityInterface(ABC):
    """Interface for security and access control services."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user."""
        pass
    
    @abstractmethod
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize user action."""
        pass
    
    @abstractmethod
    async def encrypt_data(self, data: Any) -> str:
        """Encrypt sensitive data."""
        pass
    
    @abstractmethod
    async def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt sensitive data."""
        pass
    
    @abstractmethod
    async def audit_log(self, user_id: str, action: str, resource: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log audit event."""
        pass


class WorkflowInterface(ABC):
    """Interface for workflow orchestration services."""
    
    @abstractmethod
    async def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow."""
        pass
    
    @abstractmethod
    async def execute_workflow(self, workflow_id: str, 
                             inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        pass
    
    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution."""
        pass
    
    @abstractmethod
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        pass


class NotificationInterface(ABC):
    """Interface for notification services."""
    
    @abstractmethod
    async def send_notification(self, recipient: str, message: str, 
                              channel: str = "email", 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    async def subscribe(self, user_id: str, event_type: str, 
                       channel: str = "email") -> bool:
        """Subscribe to notifications."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, user_id: str, event_type: str, 
                         channel: str = "email") -> bool:
        """Unsubscribe from notifications."""
        pass
    
    @abstractmethod
    async def get_notification_history(self, user_id: str, 
                                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get notification history."""
        pass


# Factory interfaces for creating implementations
class BackendFactory(ABC):
    """Factory interface for creating backend implementations."""
    
    @abstractmethod
    def create_vector_store(self, config: Dict[str, Any]) -> VectorStoreInterface:
        """Create vector store implementation."""
        pass
    
    @abstractmethod
    def create_graph_store(self, config: Dict[str, Any]) -> GraphStoreInterface:
        """Create graph store implementation."""
        pass
    
    @abstractmethod
    def create_concept_store(self, config: Dict[str, Any]) -> ConceptStoreInterface:
        """Create concept store implementation."""
        pass
    
    @abstractmethod
    def create_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """Create cache implementation."""
        pass


class ServiceFactory(ABC):
    """Factory interface for creating service implementations."""
    
    @abstractmethod
    def create_embedding_service(self, config: Dict[str, Any]) -> EmbeddingServiceInterface:
        """Create embedding service implementation."""
        pass
    
    @abstractmethod
    def create_conceptual_service(self, config: Dict[str, Any]) -> ConceptualServiceInterface:
        """Create conceptual service implementation."""
        pass
    
    @abstractmethod
    def create_search_service(self, config: Dict[str, Any], 
                            vector_store: VectorStoreInterface,
                            graph_store: GraphStoreInterface,
                            concept_store: ConceptStoreInterface) -> SearchServiceInterface:
        """Create search service implementation."""
        pass
    
    @abstractmethod
    def create_analytics_service(self, config: Dict[str, Any]) -> AnalyticsInterface:
        """Create analytics service implementation."""
        pass