"""
Memory Service Integration for Agentic Microservice

This module integrates the Agentic microservice with LiftOS's memory service
to provide persistent storage and retrieval of agent knowledge, experiences,
and learned patterns.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid

# Import LiftOS shared components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.base import MemorySearchRequest, MemoryStoreRequest, MemoryInsights
from shared.kse_sdk.client import kse_client
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class AgentMemoryManager:
    """
    Manager for agent memory operations including storing experiences,
    retrieving relevant knowledge, and managing agent learning patterns.
    """
    
    def __init__(self):
        self.memory_namespace = "agentic"
        self.agent_memories: Dict[str, List[Dict[str, Any]]] = {}
        
    async def store_agent_experience(
        self,
        agent_id: str,
        experience_type: str,
        experience_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5
    ) -> str:
        """
        Store an agent's experience in memory for future retrieval and learning.
        
        Args:
            agent_id: ID of the agent
            experience_type: Type of experience (e.g., "decision", "outcome", "error")
            experience_data: The actual experience data
            context: Additional context information
            importance_score: Importance score (0.0 to 1.0)
            
        Returns:
            Memory ID of the stored experience
        """
        memory_id = str(uuid.uuid4())
        
        # Prepare memory entry
        memory_entry = {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "experience_type": experience_type,
            "timestamp": datetime.now().isoformat(),
            "experience_data": experience_data,
            "context": context or {},
            "importance_score": importance_score,
            "access_count": 0,
            "last_accessed": None
        }
        
        # Store in local cache
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = []
        self.agent_memories[agent_id].append(memory_entry)
        
        # Create memory store request for KSE
        store_request = MemoryStoreRequest(
            content=json.dumps(experience_data),
            metadata={
                "agent_id": agent_id,
                "experience_type": experience_type,
                "importance_score": importance_score,
                "timestamp": memory_entry["timestamp"],
                "context": json.dumps(context or {})
            },
            namespace=f"{self.memory_namespace}.{agent_id}",
            tags=[experience_type, f"agent_{agent_id}", "experience"]
        )
        
        try:
            # Store in KSE memory service
            await kse_client.store_memory(store_request)
            logger.info(f"Stored experience for agent {agent_id}: {experience_type}")
            
        except Exception as e:
            logger.error(f"Failed to store experience in KSE: {e}")
            # Continue with local storage even if KSE fails
        
        return memory_id
    
    async def retrieve_relevant_experiences(
        self,
        agent_id: str,
        query: str,
        experience_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant experiences for an agent based on a query.
        
        Args:
            agent_id: ID of the agent
            query: Search query for relevant experiences
            experience_types: Filter by specific experience types
            limit: Maximum number of results
            min_importance: Minimum importance score
            
        Returns:
            List of relevant experiences
        """
        relevant_experiences = []
        
        try:
            # Search in KSE memory service
            search_request = MemorySearchRequest(
                query=query,
                namespace=f"{self.memory_namespace}.{agent_id}",
                limit=limit,
                filters={
                    "agent_id": agent_id,
                    "importance_score": {"$gte": min_importance}
                }
            )
            
            if experience_types:
                search_request.filters["experience_type"] = {"$in": experience_types}
            
            search_results = await kse_client.search_memories(search_request)
            
            # Convert search results to experience format
            for result in search_results.results:
                experience = {
                    "memory_id": result.id,
                    "agent_id": agent_id,
                    "experience_type": result.metadata.get("experience_type"),
                    "timestamp": result.metadata.get("timestamp"),
                    "experience_data": json.loads(result.content),
                    "context": json.loads(result.metadata.get("context", "{}")),
                    "importance_score": result.metadata.get("importance_score", 0.0),
                    "relevance_score": result.score
                }
                relevant_experiences.append(experience)
                
        except Exception as e:
            logger.error(f"Failed to search KSE memories: {e}")
            
            # Fallback to local search
            local_experiences = self.agent_memories.get(agent_id, [])
            for exp in local_experiences:
                if exp["importance_score"] >= min_importance:
                    if not experience_types or exp["experience_type"] in experience_types:
                        # Simple text matching for local search
                        if query.lower() in json.dumps(exp["experience_data"]).lower():
                            relevant_experiences.append(exp)
        
        # Update access counts
        for exp in relevant_experiences:
            await self._update_memory_access(exp["memory_id"], agent_id)
        
        # Sort by relevance/importance
        relevant_experiences.sort(
            key=lambda x: (x.get("relevance_score", 0) + x["importance_score"]) / 2,
            reverse=True
        )
        
        logger.info(f"Retrieved {len(relevant_experiences)} relevant experiences for agent {agent_id}")
        return relevant_experiences[:limit]
    
    async def _update_memory_access(self, memory_id: str, agent_id: str) -> None:
        """Update access count and timestamp for a memory."""
        # Update local cache
        agent_memories = self.agent_memories.get(agent_id, [])
        for memory in agent_memories:
            if memory["memory_id"] == memory_id:
                memory["access_count"] += 1
                memory["last_accessed"] = datetime.now().isoformat()
                break
    
    async def store_agent_learning_pattern(
        self,
        agent_id: str,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        confidence_score: float = 0.5
    ) -> str:
        """
        Store a learned pattern discovered by an agent.
        
        Args:
            agent_id: ID of the agent
            pattern_type: Type of pattern (e.g., "correlation", "causal", "temporal")
            pattern_data: The pattern data
            confidence_score: Confidence in the pattern (0.0 to 1.0)
            
        Returns:
            Pattern ID
        """
        pattern_id = str(uuid.uuid4())
        
        pattern_entry = {
            "pattern_id": pattern_id,
            "agent_id": agent_id,
            "pattern_type": pattern_type,
            "pattern_data": pattern_data,
            "confidence_score": confidence_score,
            "discovered_at": datetime.now().isoformat(),
            "validation_count": 0,
            "success_rate": 0.0
        }
        
        # Store as experience with special type
        await self.store_agent_experience(
            agent_id=agent_id,
            experience_type="learned_pattern",
            experience_data=pattern_entry,
            context={"pattern_type": pattern_type},
            importance_score=confidence_score
        )
        
        logger.info(f"Stored learning pattern for agent {agent_id}: {pattern_type}")
        return pattern_id
    
    async def get_agent_knowledge_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a summary of an agent's accumulated knowledge and experiences.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Knowledge summary
        """
        agent_memories = self.agent_memories.get(agent_id, [])
        
        # Analyze experience types
        experience_types = {}
        total_importance = 0
        recent_experiences = 0
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for memory in agent_memories:
            exp_type = memory["experience_type"]
            experience_types[exp_type] = experience_types.get(exp_type, 0) + 1
            total_importance += memory["importance_score"]
            
            memory_date = datetime.fromisoformat(memory["timestamp"])
            if memory_date > cutoff_date:
                recent_experiences += 1
        
        # Calculate knowledge metrics
        total_experiences = len(agent_memories)
        avg_importance = total_importance / total_experiences if total_experiences > 0 else 0
        
        summary = {
            "agent_id": agent_id,
            "total_experiences": total_experiences,
            "recent_experiences_7d": recent_experiences,
            "average_importance": avg_importance,
            "experience_type_distribution": experience_types,
            "knowledge_depth_score": min(total_experiences / 100, 1.0),  # Normalized to 0-1
            "learning_velocity": recent_experiences / 7,  # Experiences per day
            "last_updated": datetime.now().isoformat()
        }
        
        return summary
    
    async def consolidate_agent_memories(
        self,
        agent_id: str,
        consolidation_threshold: int = 1000
    ) -> Dict[str, Any]:
        """
        Consolidate agent memories by merging similar experiences and
        removing low-importance memories when threshold is exceeded.
        
        Args:
            agent_id: ID of the agent
            consolidation_threshold: Maximum number of memories to keep
            
        Returns:
            Consolidation results
        """
        agent_memories = self.agent_memories.get(agent_id, [])
        
        if len(agent_memories) <= consolidation_threshold:
            return {
                "consolidation_needed": False,
                "current_count": len(agent_memories),
                "threshold": consolidation_threshold
            }
        
        # Sort by importance and recency
        sorted_memories = sorted(
            agent_memories,
            key=lambda x: (x["importance_score"], x["timestamp"]),
            reverse=True
        )
        
        # Keep top memories
        consolidated_memories = sorted_memories[:consolidation_threshold]
        removed_count = len(agent_memories) - len(consolidated_memories)
        
        # Update agent memories
        self.agent_memories[agent_id] = consolidated_memories
        
        logger.info(f"Consolidated memories for agent {agent_id}: removed {removed_count} low-importance memories")
        
        return {
            "consolidation_needed": True,
            "original_count": len(agent_memories),
            "consolidated_count": len(consolidated_memories),
            "removed_count": removed_count,
            "threshold": consolidation_threshold
        }


class AgentKnowledgeGraph:
    """
    Manages knowledge graphs for agents, tracking relationships between
    concepts, experiences, and learned patterns.
    """
    
    def __init__(self, memory_manager: AgentMemoryManager):
        self.memory_manager = memory_manager
        self.knowledge_graphs: Dict[str, Dict[str, Any]] = {}
    
    async def build_knowledge_graph(self, agent_id: str) -> Dict[str, Any]:
        """
        Build a knowledge graph for an agent based on their experiences.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Knowledge graph structure
        """
        # Get all agent experiences
        all_experiences = self.memory_manager.agent_memories.get(agent_id, [])
        
        # Initialize graph structure
        graph = {
            "nodes": {},  # concept_id -> node_data
            "edges": [],  # relationships between concepts
            "clusters": {},  # grouped related concepts
            "metrics": {}
        }
        
        # Extract concepts from experiences
        concepts = {}
        for exp in all_experiences:
            exp_data = exp["experience_data"]
            
            # Extract key concepts (simplified)
            for key, value in exp_data.items():
                if isinstance(value, (str, int, float)):
                    concept_id = f"{key}_{value}"
                    if concept_id not in concepts:
                        concepts[concept_id] = {
                            "concept_id": concept_id,
                            "type": key,
                            "value": value,
                            "frequency": 0,
                            "importance": 0,
                            "experiences": []
                        }
                    
                    concepts[concept_id]["frequency"] += 1
                    concepts[concept_id]["importance"] += exp["importance_score"]
                    concepts[concept_id]["experiences"].append(exp["memory_id"])
        
        # Add concepts as nodes
        graph["nodes"] = concepts
        
        # Find relationships between concepts
        for concept1_id, concept1 in concepts.items():
            for concept2_id, concept2 in concepts.items():
                if concept1_id != concept2_id:
                    # Check for co-occurrence in experiences
                    common_experiences = set(concept1["experiences"]) & set(concept2["experiences"])
                    if len(common_experiences) > 1:  # Threshold for relationship
                        edge = {
                            "source": concept1_id,
                            "target": concept2_id,
                            "weight": len(common_experiences),
                            "type": "co_occurrence"
                        }
                        graph["edges"].append(edge)
        
        # Calculate graph metrics
        graph["metrics"] = {
            "total_nodes": len(concepts),
            "total_edges": len(graph["edges"]),
            "density": len(graph["edges"]) / (len(concepts) * (len(concepts) - 1)) if len(concepts) > 1 else 0,
            "most_connected_concept": max(concepts.keys(), key=lambda x: concepts[x]["frequency"]) if concepts else None
        }
        
        # Store the graph
        self.knowledge_graphs[agent_id] = graph
        
        logger.info(f"Built knowledge graph for agent {agent_id}: {graph['metrics']['total_nodes']} nodes, {graph['metrics']['total_edges']} edges")
        return graph
    
    async def get_related_concepts(
        self,
        agent_id: str,
        concept: str,
        max_distance: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept within the knowledge graph.
        
        Args:
            agent_id: ID of the agent
            concept: The concept to find relations for
            max_distance: Maximum distance in the graph
            
        Returns:
            List of related concepts
        """
        graph = self.knowledge_graphs.get(agent_id)
        if not graph:
            await self.build_knowledge_graph(agent_id)
            graph = self.knowledge_graphs.get(agent_id, {})
        
        related_concepts = []
        
        # Find direct connections
        for edge in graph.get("edges", []):
            if edge["source"] == concept:
                related_concepts.append({
                    "concept": edge["target"],
                    "distance": 1,
                    "relationship_strength": edge["weight"]
                })
            elif edge["target"] == concept:
                related_concepts.append({
                    "concept": edge["source"],
                    "distance": 1,
                    "relationship_strength": edge["weight"]
                })
        
        # Sort by relationship strength
        related_concepts.sort(key=lambda x: x["relationship_strength"], reverse=True)
        
        return related_concepts


# Global memory manager instance
_global_memory_manager: Optional[AgentMemoryManager] = None


async def get_memory_manager() -> AgentMemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = AgentMemoryManager()
    
    return _global_memory_manager


async def initialize_memory_integration() -> None:
    """Initialize the memory integration."""
    await get_memory_manager()
    logger.info("Agentic memory integration initialized")