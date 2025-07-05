"""
Pinecone-based implementation of KSE Memory SDK
"""
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib

import numpy as np
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI

from .models import MemorySearchResult, MemoryInsights
from .causal_models import (
    CausalSearchQuery, CausalSearchResult, CausalKnowledgeGraph,
    CausalRelationship, CausalKnowledgeNode, CausalKnowledgeEdge
)


class PineconeKSEClient:
    """Pinecone-based implementation of KSE Memory functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_client = None
        self.openai_client = None
        self.index = None
        self.initialized = False
        
        # Configuration
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_host = os.getenv('PINECONE_INDEX_HOST')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'liftos-core')
        self.region = os.getenv('PINECONE_REGION', 'us-east-1')
        self.dimension = int(os.getenv('PINECONE_DIMENSION', '1536'))
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.embedding_model = os.getenv('LLM_MODEL', 'text-embedding-ada-002')
        
    async def initialize(self) -> bool:
        """Initialize Pinecone and OpenAI clients"""
        try:
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY environment variable is required")
            
            if not self.openai_api_key:
                raise ValueError("LLM_API_KEY or OPENAI_API_KEY environment variable is required")
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=self.api_key)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            # Connect to index
            self.index = self.pinecone_client.Index(
                name=self.index_name,
                host=self.index_host
            )
            
            # Test connectivity
            await self._test_connectivity()
            
            self.initialized = True
            self.logger.info("Pinecone KSE client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone KSE client: {str(e)}")
            return False
    
    async def _test_connectivity(self):
        """Test Pinecone and OpenAI connectivity"""
        # Test Pinecone
        stats = self.index.describe_index_stats()
        self.logger.info(f"Pinecone index stats: {stats}")
        
        # Test OpenAI
        test_embedding = await self._get_embedding("test connectivity")
        if len(test_embedding) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(test_embedding)}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            
            # Truncate or pad embedding to match index dimension
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            elif len(embedding) < self.dimension:
                # Pad with zeros if needed
                embedding.extend([0.0] * (self.dimension - len(embedding)))
            
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {str(e)}")
            raise
    
    def _generate_memory_id(self, content: str, org_id: str) -> str:
        """Generate unique memory ID"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{org_id}_{timestamp}_{content_hash}"
    
    async def neural_search(self, query: str, org_id: str, limit: int = 10, filters: Dict = None) -> List[MemorySearchResult]:
        """Perform neural/semantic search using embeddings"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare filter for organization
            pinecone_filter = {"org_id": org_id}
            if filters:
                pinecone_filter.update(filters)
            
            # Search in Pinecone
            search_response = self.index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Format results
            results = []
            for match in search_response.matches:
                result = MemorySearchResult(
                    id=match.id,
                    content=match.metadata.get('content', ''),
                    score=float(match.score),
                    metadata=match.metadata,
                    memory_type=match.metadata.get('memory_type', 'general'),
                    timestamp=datetime.fromisoformat(match.metadata.get('timestamp', datetime.utcnow().isoformat()))
                )
                results.append(result)
            
            self.logger.info(f"Neural search completed: {len(results)} results for org {org_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Neural search failed for org {org_id}: {str(e)}")
            raise
    
    async def conceptual_search(self, query: str, org_id: str, limit: int = 10, filters: Dict = None) -> List[MemorySearchResult]:
        """Perform conceptual search with expanded query concepts"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Expand query with conceptual variations
            expanded_query = await self._expand_query_concepts(query)
            
            # Use neural search with expanded query
            return await self.neural_search(expanded_query, org_id, limit, filters)
            
        except Exception as e:
            self.logger.error(f"Conceptual search failed for org {org_id}: {str(e)}")
            raise
    
    async def _expand_query_concepts(self, query: str) -> str:
        """Expand query with related concepts using LLM"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Expand the given query with related concepts and synonyms. Return only the expanded query text."
                    },
                    {
                        "role": "user",
                        "content": f"Expand this query with related concepts: {query}"
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            expanded = response.choices[0].message.content.strip()
            return f"{query} {expanded}"
            
        except Exception as e:
            self.logger.warning(f"Failed to expand query concepts: {str(e)}")
            return query  # Fallback to original query
    
    async def knowledge_search(self, query: str, org_id: str, limit: int = 10, filters: Dict = None) -> List[MemorySearchResult]:
        """Perform knowledge graph-style search"""
        # For now, implement as enhanced neural search with entity extraction
        if not self.initialized:
            await self.initialize()
        
        try:
            # Extract entities from query
            entities = await self._extract_entities(query)
            
            # Search for each entity and combine results
            all_results = []
            
            # Search for original query
            results = await self.neural_search(query, org_id, limit // 2, filters)
            all_results.extend(results)
            
            # Search for entities
            for entity in entities[:3]:  # Limit to top 3 entities
                entity_results = await self.neural_search(entity, org_id, limit // 4, filters)
                all_results.extend(entity_results)
            
            # Remove duplicates and sort by score
            seen_ids = set()
            unique_results = []
            for result in sorted(all_results, key=lambda x: x.score, reverse=True):
                if result.id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result.id)
            
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Knowledge search failed for org {org_id}: {str(e)}")
            raise
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using LLM"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract key entities (people, places, concepts, organizations) from the text. Return as a comma-separated list."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            entities_text = response.choices[0].message.content.strip()
            entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            return entities
            
        except Exception as e:
            self.logger.warning(f"Failed to extract entities: {str(e)}")
            return []
    
    async def hybrid_search(self, query: str, org_id: str, limit: int = 10, filters: Dict = None, search_type: str = "hybrid") -> List[MemorySearchResult]:
        """Perform hybrid search combining neural, conceptual, and knowledge approaches"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Run different search types in parallel
            neural_task = self.neural_search(query, org_id, limit // 2, filters)
            conceptual_task = self.conceptual_search(query, org_id, limit // 3, filters)
            knowledge_task = self.knowledge_search(query, org_id, limit // 3, filters)
            
            neural_results, conceptual_results, knowledge_results = await asyncio.gather(
                neural_task, conceptual_task, knowledge_task, return_exceptions=True
            )
            
            # Combine results
            all_results = []
            
            if isinstance(neural_results, list):
                all_results.extend(neural_results)
            
            if isinstance(conceptual_results, list):
                all_results.extend(conceptual_results)
            
            if isinstance(knowledge_results, list):
                all_results.extend(knowledge_results)
            
            # Remove duplicates and re-score
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result.id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result.id)
            
            # Sort by score and return top results
            unique_results.sort(key=lambda x: x.score, reverse=True)
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed for org {org_id}: {str(e)}")
            raise
    
    async def store_memory(self, content: str, metadata: Dict, org_id: str) -> str:
        """Store memory in Pinecone"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Generate memory ID
            memory_id = self._generate_memory_id(content, org_id)
            
            # Get embedding for content
            embedding = await self._get_embedding(content)
            
            # Prepare metadata
            full_metadata = {
                "content": content,
                "org_id": org_id,
                "timestamp": datetime.utcnow().isoformat(),
                "memory_type": metadata.get("memory_type", "general"),
                **metadata
            }
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[{
                    "id": memory_id,
                    "values": embedding,
                    "metadata": full_metadata
                }]
            )
            
            self.logger.info(f"Memory stored successfully: {memory_id} for org {org_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory for org {org_id}: {str(e)}")
            raise
    
    async def analyze_context(self, org_id: str) -> Dict[str, Any]:
        """Analyze memory context for organization"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get index stats for the organization
            stats = self.index.describe_index_stats(filter={"org_id": org_id})
            
            # Get sample memories for analysis
            sample_results = await self.neural_search("", org_id, limit=50)
            
            # Analyze patterns
            memory_types = {}
            concepts = []
            
            for result in sample_results:
                memory_type = result.metadata.get("memory_type", "general")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                
                # Extract concepts from content
                if result.content:
                    content_concepts = await self._extract_entities(result.content)
                    concepts.extend(content_concepts)
            
            # Count concept frequency
            concept_counts = {}
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Get top concepts
            top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "count": len(sample_results),
                "concepts": [concept for concept, count in top_concepts],
                "density": len(sample_results) / max(stats.total_vector_count, 1),
                "temporal": self._analyze_temporal_patterns(sample_results),
                "clusters": [],  # Could implement clustering analysis
                "memory_types": memory_types
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze context for org {org_id}: {str(e)}")
            return {
                "count": 0,
                "concepts": [],
                "density": 0.0,
                "temporal": {},
                "clusters": [],
                "memory_types": {}
            }
    
    def _analyze_temporal_patterns(self, results: List[MemorySearchResult]) -> Dict[str, Any]:
        """Analyze temporal patterns in memories"""
        if not results:
            return {}
        
        # Group by hour of day
        hour_counts = {}
        for result in results:
            hour = result.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
        
        return {
            "peak_hour": peak_hour,
            "hourly_distribution": hour_counts,
            "total_timespan_days": (datetime.utcnow() - min(r.timestamp for r in results)).days if results else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of Pinecone and OpenAI services"""
        try:
            if not self.initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # Test Pinecone connectivity
            stats = self.index.describe_index_stats()
            
            # Test OpenAI connectivity
            await self._get_embedding("health check")
            
            return {
                "status": "healthy",
                "healthy": True,
                "pinecone_vectors": stats.total_vector_count,
                "pinecone_dimension": self.dimension,
                "openai_model": self.embedding_model
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }