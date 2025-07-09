"""
Main KSE Memory class - the primary interface for the universal AI memory system.
Supports all domains: healthcare, finance, real estate, enterprise, research, retail, marketing, and more.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from datetime import datetime
import uuid

from .config import KSEConfig
from .models import Entity, SearchQuery, SearchResult, ConceptualSpace, EmbeddingVector, SearchType
from .interfaces import (
    AdapterInterface,
    VectorStoreInterface,
    GraphStoreInterface,
    ConceptStoreInterface,
    EmbeddingServiceInterface,
    ConceptualServiceInterface,
    SearchServiceInterface,
    CacheInterface,
    AnalyticsInterface,
    SecurityInterface,
    WorkflowInterface,
    NotificationInterface
)
from ..exceptions import KSEError, ConfigurationError, ConnectionError, AuthenticationError
from ..backends import get_vector_store, get_graph_store, get_concept_store
from ..services import (
    EmbeddingService,
    ConceptualService,
    SearchService,
    CacheService,
    AnalyticsService,
    SecurityService,
    WorkflowService,
    NotificationService
)
from ..adapters import get_adapter


logger = logging.getLogger(__name__)


class KSEMemory:
    """
    Main KSE Memory class providing universal hybrid AI memory capabilities.
    
    This class combines Knowledge Graphs, Conceptual Spaces, and Neural Embeddings
    to create an intelligent memory system for any domain - healthcare, finance,
    real estate, enterprise, research, retail, marketing, and more.
    """
    
    def __init__(self, config: Optional[KSEConfig] = None):
        """
        Initialize KSE Memory system.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or KSEConfig()
        self._validate_config()
        
        # Core components
        self.adapter: Optional[AdapterInterface] = None
        self.vector_store: Optional[VectorStoreInterface] = None
        self.graph_store: Optional[GraphStoreInterface] = None
        self.concept_store: Optional[ConceptStoreInterface] = None
        
        # Services
        self.embedding_service: Optional[EmbeddingServiceInterface] = None
        self.conceptual_service: Optional[ConceptualServiceInterface] = None
        self.search_service: Optional[SearchServiceInterface] = None
        self.cache_service: Optional[CacheInterface] = None
        self.analytics_service: Optional[AnalyticsInterface] = None
        self.security_service: Optional[SecurityInterface] = None
        self.workflow_service: Optional[WorkflowInterface] = None
        self.notification_service: Optional[NotificationInterface] = None
        
        # State tracking
        self._initialized = False
        self._connected = False
        self._session_id = str(uuid.uuid4())
        
        # Performance tracking
        self._operation_count = 0
        self._total_operation_time = 0.0
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level.value))
        logger.info(f"KSE Memory initialized with config: {self.config.app_name} v{self.config.version}")
        logger.info(f"Session ID: {self._session_id}")
        logger.info(f"Default domain: {self.config.default_domain}")
        logger.info(f"Supported domains: {', '.join(self.config.supported_domains)}")
    
    def _validate_config(self):
        """Validate configuration and raise errors if invalid."""
        errors = self.config.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {', '.join(errors)}")
    
    async def initialize(self, adapter_type: Optional[str] = None, 
                        adapter_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the KSE Memory system with optional adapter.
        
        Args:
            adapter_type: Type of adapter (e.g., 'shopify', 'woocommerce', 'healthcare', 'finance')
            adapter_config: Configuration for the adapter
            
        Returns:
            True if initialization successful
            
        Raises:
            KSEError: If initialization fails
        """
        try:
            logger.info(f"Initializing KSE Memory system")
            start_time = datetime.utcnow()
            
            # Initialize adapter if specified
            if adapter_type and adapter_config:
                logger.info(f"Initializing {adapter_type} adapter")
                self.adapter = get_adapter(adapter_type)
                await self.adapter.connect(adapter_config)
                logger.info(f"Adapter {adapter_type} connected successfully")
            
            # Initialize storage backends
            logger.info("Initializing storage backends")
            
            # Vector store
            self.vector_store = get_vector_store(self.config.vector_store_config)
            await self.vector_store.connect()
            logger.info(f"Vector store ({self.config.vector_store_config.backend}) connected")
            
            # Graph store
            self.graph_store = get_graph_store(self.config.graph_store_config)
            await self.graph_store.connect()
            logger.info(f"Graph store ({self.config.graph_store_config.backend}) connected")
            
            # Concept store
            self.concept_store = get_concept_store(self.config.concept_store_config)
            await self.concept_store.connect()
            logger.info(f"Concept store ({self.config.concept_store_config.backend}) connected")
            
            # Initialize services
            logger.info("Initializing services")
            
            # Cache service (initialize first as other services may use it)
            self.cache_service = CacheService(self.config.cache)
            await self.cache_service.connect()
            logger.info("Cache service initialized")
            
            # Embedding service
            self.embedding_service = EmbeddingService(
                config=self.config.embedding_config,
                cache_service=self.cache_service
            )
            logger.info("Embedding service initialized")
            
            # Conceptual service
            self.conceptual_service = ConceptualService(
                config=self.config.conceptual_config,
                cache_service=self.cache_service
            )
            logger.info("Conceptual service initialized")
            
            # Search service (requires all storage backends)
            self.search_service = SearchService(
                config=self.config.search,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                concept_store=self.concept_store,
                embedding_service=self.embedding_service,
                cache_service=self.cache_service,
            )
            logger.info("Search service initialized")
            
            # Optional services
            if self.config.enable_metrics:
                self.analytics_service = AnalyticsService(self.config)
                logger.info("Analytics service initialized")
            
            if self.config.enable_auth:
                self.security_service = SecurityService(self.config)
                logger.info("Security service initialized")
            
            # Workflow and notification services
            self.workflow_service = WorkflowService(self.config)
            self.notification_service = NotificationService(self.config)
            logger.info("Workflow and notification services initialized")
            
            # Mark as initialized
            self._initialized = True
            self._connected = True
            
            initialization_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"KSE Memory initialization completed successfully in {initialization_time:.2f}s")
            
            # Track initialization
            if self.analytics_service:
                await self.analytics_service.track_entity_operation(
                    operation="initialize",
                    entity_id=self._session_id,
                    duration_ms=initialization_time * 1000,
                    success=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize KSE Memory: {str(e)}")
            self._initialized = False
            self._connected = False
            raise KSEError(f"Initialization failed: {str(e)}") from e
    
    async def disconnect(self) -> bool:
        """
        Disconnect from all services and clean up resources.
        
        Returns:
            True if disconnection successful
        """
        try:
            logger.info("Disconnecting KSE Memory")
            
            # Disconnect from all services in reverse order
            services = [
                ("notification_service", self.notification_service),
                ("workflow_service", self.workflow_service),
                ("security_service", self.security_service),
                ("analytics_service", self.analytics_service),
                ("search_service", self.search_service),
                ("conceptual_service", self.conceptual_service),
                ("embedding_service", self.embedding_service),
                ("cache_service", self.cache_service)
            ]
            
            for service_name, service in services:
                if service and hasattr(service, 'disconnect'):
                    try:
                        await service.disconnect()
                        logger.debug(f"{service_name} disconnected")
                    except Exception as e:
                        logger.warning(f"Error disconnecting {service_name}: {e}")
            
            # Disconnect from storage backends
            backends = [
                ("concept_store", self.concept_store),
                ("graph_store", self.graph_store),
                ("vector_store", self.vector_store)
            ]
            
            for backend_name, backend in backends:
                if backend:
                    try:
                        await backend.disconnect()
                        logger.debug(f"{backend_name} disconnected")
                    except Exception as e:
                        logger.warning(f"Error disconnecting {backend_name}: {e}")
            
            # Disconnect adapter
            if self.adapter:
                try:
                    await self.adapter.disconnect()
                    logger.debug("Adapter disconnected")
                except Exception as e:
                    logger.warning(f"Error disconnecting adapter: {e}")
            
            self._connected = False
            logger.info("KSE Memory disconnected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")
            return False
    
    def _ensure_initialized(self):
        """Ensure the system is initialized."""
        if not self._initialized:
            raise KSEError("KSE Memory system not initialized. Call initialize() first.")
    
    def _ensure_connected(self):
        """Ensure the system is connected."""
        if not self._connected:
            raise ConnectionError("KSE Memory system not connected.")
    
    async def _track_operation(self, operation: str, duration_ms: float, success: bool, 
                              entity_id: Optional[str] = None):
        """Track operation for analytics."""
        self._operation_count += 1
        self._total_operation_time += duration_ms
        
        if self.analytics_service:
            await self.analytics_service.track_entity_operation(
                operation=operation,
                entity_id=entity_id or "unknown",
                duration_ms=duration_ms,
                success=success
            )
    
    # Entity Management Methods
    
    async def add_entity(self, entity: Entity, compute_embeddings: bool = True, 
                        compute_concepts: bool = True, 
                        create_graph_nodes: bool = True) -> bool:
        """
        Add an entity to the KSE Memory system.
        
        Args:
            entity: Entity to add
            compute_embeddings: Whether to compute embeddings
            compute_concepts: Whether to compute conceptual dimensions
            create_graph_nodes: Whether to create graph nodes
            
        Returns:
            True if entity added successfully
            
        Raises:
            KSEError: If adding entity fails
        """
        self._ensure_initialized()
        self._ensure_connected()
        
        start_time = datetime.utcnow()
        
        try:
            logger.debug(f"Adding entity: {entity.id} (domain: {entity.domain}, type: {entity.entity_type})")
            
            # Compute embeddings if requested
            if compute_embeddings and self.embedding_service:
                # Generate text embedding
                text_content = f"{entity.title} {entity.description} {' '.join(entity.tags)}"
                entity.text_embedding = await self.embedding_service.generate_text_embedding(text_content)
                logger.debug(f"Generated text embedding for entity {entity.id}")
                
                # Generate image embedding if applicable
                image_url = entity.get_property("image_url") or entity.get_property("primary_image")
                if image_url:
                    try:
                        entity.image_embedding = await self.embedding_service.generate_image_embedding(image_url)
                        logger.debug(f"Generated image embedding for entity {entity.id}")
                    except Exception as e:
                        logger.warning(f"Failed to generate image embedding for {entity.id}: {e}")
            
            # Compute conceptual dimensions if requested
            if compute_concepts and self.conceptual_service:
                if not entity.conceptual_space:
                    entity.conceptual_space = await self.conceptual_service.compute_dimensions(entity)
                    logger.debug(f"Computed conceptual dimensions for entity {entity.id}")
            
            # Store in vector store
            if entity.text_embedding and self.vector_store:
                metadata = {
                    "entity_id": entity.id,
                    "title": entity.title,
                    "domain": entity.domain,
                    "entity_type": entity.entity_type,
                    "tags": entity.tags,
                    "categories": entity.categories,
                    "created_at": entity.created_at.isoformat(),
                    **entity.properties
                }
                
                await self.vector_store.upsert_vectors([
                    (entity.id, entity.text_embedding.vector, metadata)
                ])
                logger.debug(f"Stored vector for entity {entity.id}")
            
            # Store in concept store
            if entity.conceptual_space and self.concept_store:
                await self.concept_store.store_conceptual_space(entity.id, entity.conceptual_space)
                logger.debug(f"Stored conceptual space for entity {entity.id}")
            
            # Create graph nodes if requested
            if create_graph_nodes and self.graph_store:
                labels = [entity.domain, entity.entity_type, "Entity"]
                properties = {
                    "title": entity.title,
                    "description": entity.description,
                    "domain": entity.domain,
                    "entity_type": entity.entity_type,
                    "created_at": entity.created_at.isoformat(),
                    "version": entity.version,
                    **entity.properties
                }
                
                await self.graph_store.create_node(entity.id, labels, properties)
                logger.debug(f"Created graph node for entity {entity.id}")
                
                # Create relationships based on categories and tags
                for category in entity.categories:
                    category_id = f"category_{category.lower().replace(' ', '_')}"
                    await self.graph_store.create_node(
                        category_id, 
                        ["Category"], 
                        {"name": category, "type": "category"}
                    )
                    await self.graph_store.create_relationship(
                        entity.id, category_id, "BELONGS_TO_CATEGORY"
                    )
                
                for tag in entity.tags:
                    tag_id = f"tag_{tag.lower().replace(' ', '_')}"
                    await self.graph_store.create_node(
                        tag_id, 
                        ["Tag"], 
                        {"name": tag, "type": "tag"}
                    )
                    await self.graph_store.create_relationship(
                        entity.id, tag_id, "HAS_TAG"
                    )
            
            # Update entity metadata
            entity.updated_at = datetime.utcnow()
            entity.version += 1
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("add_entity", duration_ms, True, entity.id)
            
            logger.info(f"Successfully added entity {entity.id} in {duration_ms:.2f}ms")
            return True
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("add_entity", duration_ms, False, entity.id)
            logger.error(f"Failed to add entity {entity.id}: {str(e)}")
            raise KSEError(f"Failed to add entity: {str(e)}") from e
    
    async def get_entity(self, entity_id: str, include_embeddings: bool = False,
                        include_conceptual: bool = True, 
                        include_graph: bool = True) -> Optional[Entity]:
        """
        Retrieve an entity by ID.
        
        Args:
            entity_id: Entity ID
            include_embeddings: Whether to include embeddings
            include_conceptual: Whether to include conceptual space
            include_graph: Whether to include graph relationships
            
        Returns:
            Entity if found, None otherwise
        """
        self._ensure_initialized()
        self._ensure_connected()
        
        start_time = datetime.utcnow()
        
        try:
            # Try cache first
            cache_key = f"entity:{entity_id}"
            if self.cache_service:
                cached_entity = await self.cache_service.get(cache_key)
                if cached_entity:
                    logger.debug(f"Retrieved entity {entity_id} from cache")
                    return Entity.from_dict(cached_entity)
            
            # Get from vector store (primary source)
            if self.vector_store:
                vector_data = await self.vector_store.get_vector(entity_id)
                if vector_data:
                    vector, metadata = vector_data
                    
                    # Reconstruct entity from metadata
                    entity = Entity(
                        id=entity_id,
                        title=metadata.get("title", ""),
                        description=metadata.get("description", ""),
                        domain=metadata.get("domain", self.config.default_domain),
                        entity_type=metadata.get("entity_type", "unknown"),
                        tags=metadata.get("tags", []),
                        categories=metadata.get("categories", []),
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                        properties={k: v for k, v in metadata.items() 
                                  if k not in ["entity_id", "title", "description", "domain", 
                                             "entity_type", "tags", "categories", "created_at"]}
                    )
                    
                    # Add embeddings if requested
                    if include_embeddings:
                        entity.text_embedding = EmbeddingVector(
                            vector=vector,
                            model=self.config.embedding.text_model,
                            dimension=len(vector)
                        )
                    
                    # Add conceptual space if requested
                    if include_conceptual and self.concept_store:
                        entity.conceptual_space = await self.concept_store.get_conceptual_space(entity_id)
                    
                    # Add graph relationships if requested
                    if include_graph and self.graph_store:
                        # This would be expanded to include actual relationship data
                        pass
                    
                    # Cache the entity
                    if self.cache_service:
                        await self.cache_service.set(cache_key, entity.to_dict(), ttl=300)
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    await self._track_operation("get_entity", duration_ms, True, entity_id)
                    
                    logger.debug(f"Retrieved entity {entity_id} in {duration_ms:.2f}ms")
                    return entity
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("get_entity", duration_ms, True, entity_id)
            
            logger.debug(f"Entity {entity_id} not found")
            return None
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("get_entity", duration_ms, False, entity_id)
            logger.error(f"Failed to get entity {entity_id}: {str(e)}")
            raise KSEError(f"Failed to get entity: {str(e)}") from e
    
    async def update_entity(self, entity: Entity, recompute_embeddings: bool = False,
                           recompute_concepts: bool = False) -> bool:
        """
        Update an existing entity.
        
        Args:
            entity: Updated entity
            recompute_embeddings: Whether to recompute embeddings
            recompute_concepts: Whether to recompute conceptual dimensions
            
        Returns:
            True if update successful
        """
        self._ensure_initialized()
        self._ensure_connected()
        
        # Update version and timestamp
        entity.updated_at = datetime.utcnow()
        entity.version += 1
        
        # Recompute if requested
        if recompute_embeddings or recompute_concepts:
            return await self.add_entity(
                entity, 
                compute_embeddings=recompute_embeddings,
                compute_concepts=recompute_concepts
            )
        else:
            return await self.add_entity(entity, compute_embeddings=False, compute_concepts=False)
    
    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from all stores.
        
        Args:
            entity_id: Entity ID to delete
            
        Returns:
            True if deletion successful
        """
        self._ensure_initialized()
        self._ensure_connected()
        
        start_time = datetime.utcnow()
        
        try:
            # Delete from all stores
            tasks = []
            
            if self.vector_store:
                tasks.append(self.vector_store.delete_vectors([entity_id]))
            
            if self.concept_store:
                tasks.append(self.concept_store.delete_conceptual_space(entity_id))
            
            if self.graph_store:
                tasks.append(self.graph_store.delete_node(entity_id))
            
            if self.cache_service:
                tasks.append(self.cache_service.delete(f"entity:{entity_id}"))
            
            # Execute all deletions
            await asyncio.gather(*tasks, return_exceptions=True)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("delete_entity", duration_ms, True, entity_id)
            
            logger.info(f"Deleted entity {entity_id} in {duration_ms:.2f}ms")
            return True
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("delete_entity", duration_ms, False, entity_id)
            logger.error(f"Failed to delete entity {entity_id}: {str(e)}")
            raise KSEError(f"Failed to delete entity: {str(e)}") from e
    
    # Search Methods
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform universal search across all domains.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        self._ensure_initialized()
        self._ensure_connected()
        
        if not self.search_service:
            raise KSEError("Search service not available")
        
        start_time = datetime.utcnow()
        
        try:
            results = await self.search_service.search(query)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Track search
            if self.analytics_service:
                await self.analytics_service.track_search(query, results, duration_ms)
            
            logger.info(f"Search completed: {len(results)} results in {duration_ms:.2f}ms")
            return results
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_operation("search", duration_ms, False)
            logger.error(f"Search failed: {str(e)}")
            raise KSEError(f"Search failed: {str(e)}") from e
    
    async def semantic_search(self, query_text: str, domain: Optional[str] = None,
                             limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic search."""
        if not self.search_service:
            raise KSEError("Search service not available")
        
        return await self.search_service.semantic_search(
            query_text=query_text,
            domain=domain or self.config.default_domain,
            limit=limit,
            threshold=threshold
        )
    
    async def conceptual_search(self, dimension_filters: Dict[str, float],
                               domain: Optional[str] = None, 
                               limit: int = 10) -> List[SearchResult]:
        """Perform conceptual search."""
        if not self.search_service:
            raise KSEError("Search service not available")
        
        return await self.search_service.conceptual_search(
            dimension_filters=dimension_filters,
            domain=domain or self.config.default_domain,
            limit=limit
        )
    
    async def graph_search(self, entity_id: str, relationship_types: Optional[List[str]] = None,
                          max_depth: int = 2, limit: int = 10) -> List[SearchResult]:
        """Perform graph-based search."""
        if not self.search_service:
            raise KSEError("Search service not available")
        
        return await self.search_service.graph_search(
            entity_id=entity_id,
            relationship_types=relationship_types,
            max_depth=max_depth,
            limit=limit
        )
    
    # Batch Operations
    
    async def batch_add_entities(self, entities: List[Entity], 
                                batch_size: int = 100) -> Dict[str, bool]:
        """
        Add multiple entities in batches.
        
        Args:
            entities: List of entities to add
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping entity IDs to success status
        """
        results = {}
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_tasks = [self.add_entity(entity) for entity in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for entity, result in zip(batch, batch_results):
                results[entity.id] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    logger.error(f"Failed to add entity {entity.id}: {result}")
        
        return results
    
    # Analytics and Monitoring
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "session_id": self._session_id,
            "initialized": self._initialized,
            "connected": self._connected,
            "operation_count": self._operation_count,
            "average_operation_time_ms": (
                self._total_operation_time / self._operation_count 
                if self._operation_count > 0 else 0
            ),
            "config": {
                "app_name": self.config.app_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "default_domain": self.config.default_domain,
                "supported_domains": self.config.supported_domains
            }
        }
        
        # Add component stats
        if self.vector_store:
            try:
                stats["vector_store"] = await self.vector_store.get_index_stats()
            except Exception as e:
                stats["vector_store"] = {"error": str(e)}
        
        if self.graph_store:
            try:
                stats["graph_store"] = await self.graph_store.get_graph_stats()
            except Exception as e:
                stats["graph_store"] = {"error": str(e)}
        
        if self.cache_service:
            try:
                stats["cache"] = await self.cache_service.get_stats()
            except Exception as e:
                stats["cache"] = {"error": str(e)}
        
        if self.analytics_service:
            try:
                stats["analytics"] = await self.analytics_service.get_performance_metrics()
            except Exception as e:
                stats["analytics"] = {"error": str(e)}
        
        return stats
    
    # Context managers
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    # Domain-specific convenience methods
    
    def create_healthcare_entity(self, id: str, title: str, description: str, 
                                entity_type: str = "medical_record", **kwargs) -> Entity:
        """Create a healthcare entity."""
        return Entity.create_healthcare_entity(id, title, description, entity_type, **kwargs)
    
    def create_finance_entity(self, id: str, title: str, description: str,
                             entity_type: str = "financial_instrument", **kwargs) -> Entity:
        """Create a finance entity."""
        return Entity.create_finance_entity(id, title, description, entity_type, **kwargs)
    
    def create_real_estate_entity(self, id: str, title: str, description: str,
                                 entity_type: str = "property", **kwargs) -> Entity:
        """Create a real estate entity."""
        return Entity.create_real_estate_entity(id, title, description, entity_type, **kwargs)
    
    def create_enterprise_entity(self, id: str, title: str, description: str,
                                entity_type: str = "resource", **kwargs) -> Entity:
        """Create an enterprise entity."""
        return Entity.create_enterprise_entity(id, title, description, entity_type, **kwargs)
    
    def create_research_entity(self, id: str, title: str, description: str,
                              entity_type: str = "paper", **kwargs) -> Entity:
        """Create a research entity."""
        return Entity.create_research_entity(id, title, description, entity_type, **kwargs)
    
    def create_retail_entity(self, id: str, title: str, description: str,
                            entity_type: str = "product", **kwargs) -> Entity:
        """Create a retail entity."""
        return Entity.create_retail_entity(id, title, description, entity_type, **kwargs)
    
    def create_marketing_entity(self, id: str, title: str, description: str,
                               entity_type: str = "campaign", **kwargs) -> Entity:
        """Create a marketing entity."""
        return Entity.create_marketing_entity(id, title, description, entity_type, **kwargs)


# Convenience functions

async def create_kse_memory(config: Optional[KSEConfig] = None,
                           adapter_type: Optional[str] = None,
                           adapter_config: Optional[Dict[str, Any]] = None) -> KSEMemory:
    """
    Create and initialize a KSE Memory instance.
    
    Args:
        config: Configuration object
        adapter_type: Type of adapter to use
        adapter_config: Adapter configuration
        
    Returns:
        Initialized KSE Memory instance
    """
    kse = KSEMemory(config)
    await kse.initialize(adapter_type, adapter_config)
    return kse


def create_domain_config(domain: str, **kwargs) -> KSEConfig:
    """
    Create a domain-specific configuration.
    
    Args:
        domain: Domain name
        **kwargs: Additional configuration options
        
    Returns:
        Domain-specific configuration
    """
    config = KSEConfig()
    config.default_domain = domain
    
    # Apply domain-specific defaults
    if domain == "healthcare":
        config.vector_store.index_name = "kse-healthcare"
        config.graph_store.database = "healthcare_graph"
        config.concept_store.database = "healthcare_concepts"
    elif domain == "finance":
        config.vector_store.index_name = "kse-finance"
        config.graph_store.database = "finance_graph"
        config.concept_store.database = "finance_concepts"
    elif domain == "real_estate":
        config.vector_store.index_name = "kse-realestate"
        config.graph_store.database = "realestate_graph"
        config.concept_store.database = "realestate_concepts"
    elif domain == "enterprise":
        config.vector_store.index_name = "kse-enterprise"
        config.graph_store.database = "enterprise_graph"
        config.concept_store.database = "enterprise_concepts"
    elif domain == "research":
        config.vector_store.index_name = "kse-research"
        config.graph_store.database = "research_graph"
        config.concept_store.database = "research_concepts"
    elif domain == "retail":
        config.vector_store.index_name = "kse-retail"
        config.graph_store.database = "retail_graph"
        config.concept_store.database = "retail_concepts"
    elif domain == "marketing":
        config.vector_store.index_name = "kse-marketing"
        config.graph_store.database = "marketing_graph"
        config.concept_store.database = "marketing_concepts"
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config