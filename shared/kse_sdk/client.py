"""
KSE Memory SDK client wrapper for Lift OS Core
Universal KSE-SDK Integration Layer
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from shared.utils.config import get_config
from shared.utils.logging import get_logger
from shared.models.base import MemorySearchResult, MemoryInsights
from .core import KSEMemory, Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
from .causal_models import (
    CausalSearchQuery, CausalSearchResult, CausalInsights,
    CausalMemoryEntry, CausalKnowledgeGraph
)


class LiftKSEClient:
    """Lift OS Core wrapper for Universal KSE Memory SDK"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('kse_client', 'memory-service')
        
        # Initialize universal KSE Memory with configuration
        kse_config = KSEConfig(
            vector_store_type="memory",  # Use in-memory for testing
            graph_store_type="memory",   # Use in-memory for testing
            concept_store_type="memory", # Use in-memory for testing
            embedding_model="mock",      # Use mock for testing
            cache_enabled=True,
            analytics_enabled=True,
            security_enabled=True
        )
        
        self.kse_memory = KSEMemory(config=kse_config)
        self.org_contexts: Dict[str, Any] = {}
        self._initialized = False
        
        # Causal-aware extensions
        self.causal_enabled = True
        self.causal_graphs: Dict[str, CausalKnowledgeGraph] = {}
    
    async def initialize(self):
        """Initialize Universal KSE Memory SDK"""
        if self._initialized:
            return
        
        try:
            # Initialize universal KSE Memory
            await self.kse_memory.initialize()
            
            self._initialized = True
            self.logger.info("Universal KSE Memory SDK initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Universal KSE Memory SDK: {str(e)}")
            raise
    
    async def initialize_org_memory(self, org_id: str) -> str:
        """Initialize memory context for organization"""
        if not self._initialized:
            await self.initialize()
        
        context_id = f"org_{org_id}"
        
        if context_id not in self.org_contexts:
            self.org_contexts[context_id] = {
                'org_id': org_id,
                'created_at': datetime.utcnow(),
                'active': True,
                'memory_count': 0
            }
            self.logger.info(f"Memory context initialized for org {org_id}")
        
        return context_id
    
    async def hybrid_search(
        self, 
        org_id: str, 
        query: str, 
        search_type: str = "hybrid", 
        limit: int = 10, 
        filters: Dict = None
    ) -> List[MemorySearchResult]:
        """Perform memory search using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Create universal search query
            search_query = SearchQuery(
                query=query,
                search_type=search_type,
                limit=limit,
                filters=filters or {},
                organization_id=org_id
            )
            
            # Perform search using universal KSE Memory
            results = await self.kse_memory.search(search_query)
            
            # Convert to legacy format for backward compatibility
            memory_results = []
            for result in results:
                memory_result = MemorySearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    memory_type=result.metadata.get("memory_type", "general"),
                    timestamp=result.metadata.get("timestamp", datetime.utcnow())
                )
                memory_results.append(memory_result)
            
            return memory_results
                
        except Exception as e:
            self.logger.error(f"Search failed for org {org_id}: {str(e)}")
            raise
    
    async def store_memory(
        self, 
        org_id: str, 
        content: str, 
        memory_type: str = "general", 
        metadata: Dict = None
    ) -> str:
        """Store memory content using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Create universal entity
            entity_metadata = {
                "memory_type": memory_type,
                "source": "lift_os_core",
                "organization_id": org_id,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            entity = Entity(
                id=f"memory_{org_id}_{datetime.utcnow().timestamp()}",
                content=content,
                entity_type="memory",
                domain="general",
                metadata=entity_metadata
            )
            
            # Store using universal KSE Memory
            memory_id = await self.kse_memory.store(entity)
            
            # Update context stats
            if f"org_{org_id}" in self.org_contexts:
                self.org_contexts[f"org_{org_id}"]["memory_count"] += 1
            
            self.logger.info(f"Memory stored successfully: {memory_id} for org {org_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory for org {org_id}: {str(e)}")
            raise
    
    async def get_memory_insights(self, org_id: str) -> MemoryInsights:
        """Get memory analytics and insights using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Get insights using universal analytics service
            insights_data = await self.kse_memory.get_analytics(
                filters={"organization_id": org_id}
            )
            
            insights = MemoryInsights(
                total_memories=insights_data.get("total_entities", 0),
                dominant_concepts=insights_data.get("dominant_concepts", []),
                knowledge_density=insights_data.get("knowledge_density", 0.0),
                temporal_patterns=insights_data.get("temporal_patterns", {}),
                semantic_clusters=insights_data.get("semantic_clusters", []),
                memory_types=insights_data.get("entity_types", {})
            )
            
            self.logger.info(f"Memory insights generated for org {org_id}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get memory insights for org {org_id}: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Universal KSE SDK health"""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Check universal KSE Memory health
            health_status = await self.kse_memory.health_check()
            
            return {
                "status": "healthy" if health_status.get("healthy") else "unhealthy",
                "healthy": health_status.get("healthy", False),
                "active_contexts": len([ctx for ctx in self.org_contexts.values() if ctx.get('active', False)]),
                "total_contexts": len(self.org_contexts),
                "backend": "universal_kse",
                "kse_status": health_status
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    async def cleanup_org_context(self, org_id: str):
        """Cleanup memory context for organization"""
        context_id = f"org_{org_id}"
        
        if context_id in self.org_contexts:
            self.org_contexts[context_id]['active'] = False
            self.logger.info(f"Memory context deactivated for org {org_id}")
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about active contexts"""
        active_contexts = sum(1 for ctx in self.org_contexts.values() if ctx.get('active', False))
        
        return {
            "total_contexts": len(self.org_contexts),
            "active_contexts": active_contexts,
            "backend": "universal_kse"
        }
    
    async def causal_search(
        self, 
        org_id: str, 
        query: str, 
        causal_filters: Dict = None,
        relationship_types: List[str] = None,
        minimum_strength: float = 0.5,
        limit: int = 10
    ) -> List[CausalSearchResult]:
        """Perform causal-aware search using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Create enhanced search query with causal filters
            enhanced_filters = {
                "organization_id": org_id,
                "causal_enabled": True,
                "minimum_strength": minimum_strength,
                **(causal_filters or {})
            }
            
            if relationship_types:
                enhanced_filters["relationship_types"] = relationship_types
            
            search_query = SearchQuery(
                query=query,
                search_type="causal_hybrid",
                limit=limit,
                filters=enhanced_filters,
                organization_id=org_id
            )
            
            # Perform causal search using universal KSE Memory
            results = await self.kse_memory.search(search_query)
            
            # Convert to causal search results
            causal_results = []
            for result in results:
                causal_result = CausalSearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score,
                    causal_relationships=result.metadata.get("causal_relationships", []),
                    temporal_context=result.metadata.get("temporal_context", {}),
                    causal_strength=result.metadata.get("causal_strength", 0.0),
                    metadata=result.metadata
                )
                causal_results.append(causal_result)
            
            self.logger.info(f"Causal search completed for org {org_id}: {len(causal_results)} results")
            return causal_results
            
        except Exception as e:
            self.logger.error(f"Causal search failed for org {org_id}: {str(e)}")
            raise
    
    async def store_causal_memory(
        self,
        org_id: str,
        causal_entry: CausalMemoryEntry
    ) -> str:
        """Store causal memory using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Create universal entity with causal metadata
            causal_metadata = {
                "memory_type": "causal",
                "source": "lift_os_causal",
                "organization_id": org_id,
                "causal_relationships": [rel.dict() for rel in causal_entry.causal_relationships],
                "temporal_context": causal_entry.temporal_context,
                "causal_metadata": causal_entry.causal_metadata,
                "platform_context": causal_entry.platform_context,
                "experiment_id": causal_entry.experiment_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            entity = Entity(
                id=f"causal_{org_id}_{datetime.utcnow().timestamp()}",
                content=causal_entry.content,
                entity_type="causal_memory",
                domain="causal",
                metadata=causal_metadata
            )
            
            # Store using universal KSE Memory
            memory_id = await self.kse_memory.store(entity)
            
            # Update context stats
            if f"org_{org_id}" in self.org_contexts:
                self.org_contexts[f"org_{org_id}"]["memory_count"] += 1
            
            self.logger.info(f"Causal memory stored successfully: {memory_id} for org {org_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store causal memory for org {org_id}: {str(e)}")
            raise
    
    async def get_causal_insights(self, org_id: str) -> CausalInsights:
        """Get causal insights using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Get causal insights using universal analytics service
            insights_data = await self.kse_memory.get_analytics(
                filters={
                    "organization_id": org_id,
                    "entity_type": "causal_memory"
                }
            )
            
            insights = CausalInsights(
                organization_id=org_id,
                total_causal_memories=insights_data.get("total_entities", 0),
                dominant_causal_patterns=insights_data.get("causal_patterns", []),
                temporal_causal_trends=insights_data.get("temporal_trends", {}),
                relationship_strength_distribution=insights_data.get("strength_distribution", {}),
                causal_clusters=insights_data.get("causal_clusters", []),
                knowledge_graph_stats=insights_data.get("graph_stats", {}),
                causal_anomalies=insights_data.get("anomalies", []),
                generated_at=datetime.utcnow()
            )
            
            self.logger.info(f"Causal insights generated for org {org_id}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get causal insights for org {org_id}: {str(e)}")
            raise
    
    async def build_causal_knowledge_graph(self, org_id: str) -> CausalKnowledgeGraph:
        """Build causal knowledge graph using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if graph already exists
            if org_id in self.causal_graphs:
                return self.causal_graphs[org_id]
            
            # Build graph using universal KSE Memory graph service
            graph_data = await self.kse_memory.build_knowledge_graph(
                filters={"organization_id": org_id, "causal_enabled": True}
            )
            
            # Create knowledge graph object
            graph = CausalKnowledgeGraph(
                id=f"causal_graph_{org_id}_{datetime.utcnow().isoformat()}",
                organization_id=org_id,
                nodes=graph_data.get("nodes", []),
                edges=graph_data.get("edges", []),
                metadata=graph_data.get("metadata", {}),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Cache the graph
            self.causal_graphs[org_id] = graph
            
            self.logger.info(f"Causal knowledge graph built for org {org_id}")
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to build causal knowledge graph for org {org_id}: {str(e)}")
            raise
    
    async def temporal_causal_search(
        self,
        org_id: str,
        query: str,
        time_range: Dict[str, datetime],
        lag_analysis: bool = True,
        limit: int = 10
    ) -> List[CausalSearchResult]:
        """Perform temporal causal search using universal KSE architecture"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create temporal causal search query
            temporal_filters = {
                "organization_id": org_id,
                "causal_enabled": True,
                "temporal_range": time_range,
                "lag_analysis": lag_analysis
            }
            
            search_query = SearchQuery(
                query=query,
                search_type="causal_temporal",
                limit=limit,
                filters=temporal_filters,
                organization_id=org_id
            )
            
            # Perform temporal search using universal KSE Memory
            results = await self.kse_memory.search(search_query)
            
            # Convert to causal search results
            causal_results = []
            for result in results:
                causal_result = CausalSearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score,
                    causal_relationships=result.metadata.get("causal_relationships", []),
                    temporal_context=result.metadata.get("temporal_context", {}),
                    causal_strength=result.metadata.get("causal_strength", 0.0),
                    metadata=result.metadata
                )
                causal_results.append(causal_result)
            
            self.logger.info(f"Temporal causal search completed for org {org_id}: {len(causal_results)} results")
            return causal_results
            
        except Exception as e:
            self.logger.error(f"Temporal causal search failed for org {org_id}: {str(e)}")
            raise


# Global KSE client instance using universal architecture
kse_client = LiftKSEClient()