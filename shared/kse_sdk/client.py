"""
KSE Memory SDK client wrapper for Lift OS Core
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from shared.utils.config import get_config
from shared.utils.logging import get_logger
from shared.models.base import MemorySearchResult, MemoryInsights
from .pinecone_client import PineconeKSEClient
from .causal_models import (
    CausalSearchQuery, CausalSearchResult, CausalInsights,
    CausalMemoryEntry, CausalKnowledgeGraph
)


class LiftKSEClient:
    """Lift OS Core wrapper for KSE Memory SDK using Pinecone"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('kse_client', 'memory-service')
        self.pinecone_client = PineconeKSEClient()
        self.org_contexts: Dict[str, Any] = {}
        self._initialized = False
        # Causal-aware extensions
        self.causal_enabled = True
        self.causal_graphs: Dict[str, CausalKnowledgeGraph] = {}
    
    async def initialize(self):
        """Initialize KSE Memory SDK with Pinecone backend"""
        if self._initialized:
            return
        
        try:
            # Initialize Pinecone client
            success = await self.pinecone_client.initialize()
            if not success:
                raise Exception("Failed to initialize Pinecone KSE client")
            
            self._initialized = True
            self.logger.info("KSE Memory SDK initialized successfully with Pinecone backend")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize KSE Memory SDK: {str(e)}")
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
        """Perform memory search using specified type"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            if search_type == "neural":
                return await self.pinecone_client.neural_search(query, org_id, limit, filters)
            elif search_type == "conceptual":
                return await self.pinecone_client.conceptual_search(query, org_id, limit, filters)
            elif search_type == "knowledge":
                return await self.pinecone_client.knowledge_search(query, org_id, limit, filters)
            elif search_type == "hybrid":
                return await self.pinecone_client.hybrid_search(query, org_id, limit, filters)
            else:
                # Default to neural search
                return await self.pinecone_client.neural_search(query, org_id, limit, filters)
                
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
        """Store memory content"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Prepare metadata
            full_metadata = {
                "memory_type": memory_type,
                "source": "lift_os_core",
                **(metadata or {})
            }
            
            # Store using Pinecone client
            memory_id = await self.pinecone_client.store_memory(content, full_metadata, org_id)
            
            # Update context stats
            if f"org_{org_id}" in self.org_contexts:
                self.org_contexts[f"org_{org_id}"]["memory_count"] += 1
            
            self.logger.info(f"Memory stored successfully: {memory_id} for org {org_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory for org {org_id}: {str(e)}")
            raise
    
    async def get_memory_insights(self, org_id: str) -> MemoryInsights:
        """Get memory analytics and insights"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Get insights from Pinecone client
            insights_data = await self.pinecone_client.analyze_context(org_id)
            
            insights = MemoryInsights(
                total_memories=insights_data.get("count", 0),
                dominant_concepts=insights_data.get("concepts", []),
                knowledge_density=insights_data.get("density", 0.0),
                temporal_patterns=insights_data.get("temporal", {}),
                semantic_clusters=insights_data.get("clusters", []),
                memory_types=insights_data.get("memory_types", {})
            )
            
            self.logger.info(f"Memory insights generated for org {org_id}")
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get memory insights for org {org_id}: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check KSE SDK health"""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Check Pinecone client health
            pinecone_health = await self.pinecone_client.health_check()
            
            return {
                "status": "healthy" if pinecone_health.get("healthy") else "unhealthy",
                "healthy": pinecone_health.get("healthy", False),
                "active_contexts": len([ctx for ctx in self.org_contexts.values() if ctx.get('active', False)]),
                "total_contexts": len(self.org_contexts),
                "backend": "pinecone",
                "pinecone_status": pinecone_health
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
    
    async def causal_search(
        self, 
        org_id: str, 
        query: str, 
        causal_filters: Dict = None,
        relationship_types: List[str] = None,
        minimum_strength: float = 0.5,
        limit: int = 10
    ) -> List[CausalSearchResult]:
        """Perform causal-aware search"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Create causal search query
            causal_query = CausalSearchQuery(
                query=query,
                organization_id=org_id,
                search_type="causal_hybrid",
                causal_filters=causal_filters,
                relationship_types=relationship_types,
                minimum_strength=minimum_strength,
                limit=limit
            )
            
            # Perform causal search using Pinecone client
            results = await self.pinecone_client.causal_search(causal_query)
            
            self.logger.info(f"Causal search completed for org {org_id}: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Causal search failed for org {org_id}: {str(e)}")
            raise
    
    async def store_causal_memory(
        self,
        org_id: str,
        causal_entry: CausalMemoryEntry
    ) -> str:
        """Store causal memory with relationship embeddings"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Prepare causal metadata
            causal_metadata = {
                "memory_type": "causal",
                "source": "lift_os_causal",
                "causal_relationships": [rel.dict() for rel in causal_entry.causal_relationships],
                "temporal_context": causal_entry.temporal_context,
                "causal_metadata": causal_entry.causal_metadata,
                "platform_context": causal_entry.platform_context,
                "experiment_id": causal_entry.experiment_id
            }
            
            # Store using Pinecone client
            memory_id = await self.pinecone_client.store_causal_memory(
                content=causal_entry.content,
                metadata=causal_metadata,
                org_id=org_id
            )
            
            # Update context stats
            if f"org_{org_id}" in self.org_contexts:
                self.org_contexts[f"org_{org_id}"]["memory_count"] += 1
            
            self.logger.info(f"Causal memory stored successfully: {memory_id} for org {org_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store causal memory for org {org_id}: {str(e)}")
            raise
    
    async def get_causal_insights(self, org_id: str) -> CausalInsights:
        """Get causal insights and analytics"""
        if not self._initialized:
            await self.initialize()
        
        await self.initialize_org_memory(org_id)
        
        try:
            # Get causal insights from Pinecone client
            insights_data = await self.pinecone_client.analyze_causal_context(org_id)
            
            insights = CausalInsights(
                organization_id=org_id,
                total_causal_memories=insights_data.get("total_causal_memories", 0),
                dominant_causal_patterns=insights_data.get("patterns", []),
                temporal_causal_trends=insights_data.get("temporal_trends", {}),
                relationship_strength_distribution=insights_data.get("strength_distribution", {}),
                causal_clusters=insights_data.get("clusters", []),
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
        """Build causal knowledge graph for organization"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if graph already exists
            if org_id in self.causal_graphs:
                return self.causal_graphs[org_id]
            
            # Build graph using Pinecone client
            graph_data = await self.pinecone_client.build_causal_knowledge_graph(org_id)
            
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
        """Perform temporal causal search with lag analysis"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create temporal causal search query
            causal_query = CausalSearchQuery(
                query=query,
                organization_id=org_id,
                search_type="causal_temporal",
                temporal_range=time_range,
                limit=limit
            )
            
            # Perform temporal search
            results = await self.pinecone_client.temporal_causal_search(causal_query, lag_analysis)
            
            self.logger.info(f"Temporal causal search completed for org {org_id}: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal causal search failed for org {org_id}: {str(e)}")
            raise
        
        return {
            "total_contexts": len(self.org_contexts),
            "active_contexts": active_contexts,
            "backend": "pinecone"
        }


# Global KSE client instance
kse_client = LiftKSEClient()