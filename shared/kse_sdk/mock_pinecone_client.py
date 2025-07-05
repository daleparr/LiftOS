"""
Mock Pinecone implementation for testing KSE Memory SDK
"""
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib
import uuid

import numpy as np
import openai
from openai import OpenAI

from .models import MemorySearchResult, MemoryInsights


class MockPinecone:
    """Mock Pinecone client for testing"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._indexes = {}
    
    def Index(self, name: str):
        if name not in self._indexes:
            self._indexes[name] = MockIndex(name)
        return self._indexes[name]


class MockIndex:
    """Mock Pinecone index for testing"""
    def __init__(self, name: str):
        self.name = name
        self._vectors = {}
    
    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = ""):
        for vector in vectors:
            key = f"{namespace}:{vector['id']}" if namespace else vector['id']
            self._vectors[key] = vector
        return {"upserted_count": len(vectors)}
    
    def query(self, vector: List[float], top_k: int = 10, namespace: str = "", 
              filter: Dict[str, Any] = None, include_metadata: bool = True):
        # Simple mock similarity search
        results = []
        for key, stored_vector in self._vectors.items():
            if namespace and not key.startswith(f"{namespace}:"):
                continue
            
            # Mock similarity score (random for testing)
            score = np.random.random()
            
            if filter:
                # Simple filter check
                metadata = stored_vector.get('metadata', {})
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            results.append({
                'id': stored_vector['id'],
                'score': score,
                'metadata': stored_vector.get('metadata', {}) if include_metadata else None
            })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return {'matches': results[:top_k]}
    
    def delete(self, ids: List[str], namespace: str = ""):
        deleted = 0
        for id in ids:
            key = f"{namespace}:{id}" if namespace else id
            if key in self._vectors:
                del self._vectors[key]
                deleted += 1
        return {"deleted_count": deleted}
    
    def describe_index_stats(self):
        return {
            "namespaces": {
                "": {"vector_count": len(self._vectors)}
            },
            "dimension": 1536,
            "index_fullness": 0.1,
            "total_vector_count": len(self._vectors)
        }


class MockServerlessSpec:
    """Mock ServerlessSpec for testing"""
    def __init__(self, cloud: str, region: str):
        self.cloud = cloud
        self.region = region


class PineconeKSEClient:
    """
    Mock Pinecone-based implementation of KSE Memory SDK for testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        openai_api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("LLM_API_KEY or OPENAI_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize mock Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY', 'mock-key')
        self.pc = MockPinecone(api_key=pinecone_api_key)
        
        # Get configuration
        self.index_host = os.getenv('PINECONE_INDEX_HOST', 'mock-host')
        self.index_name = self.index_host.split('.')[0] if '.' in self.index_host else 'liftos-core'
        self.region = os.getenv('PINECONE_REGION', 'us-east-1')
        self.dimension = int(os.getenv('PINECONE_DIMENSION', '1536'))
        
        # Initialize index
        self.index = self.pc.Index(self.index_name)
        
        self.logger.info(f"Mock PineconeKSEClient initialized with index: {self.index_name}")
    
    async def initialize(self):
        """Initialize the KSE client - mock implementation"""
        try:
            self.logger.info("Mock PineconeKSEClient initialization completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            return False
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return mock embedding for testing
            return [0.1] * self.dimension
    
    async def store_memory(self, content: str, memory_type: str, 
                          organization_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory in the vector database"""
        try:
            # Generate embedding
            embedding = await self.get_embedding(content)
            
            # Create memory ID
            memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            full_metadata = {
                'content': content,
                'memory_type': memory_type,
                'organization_id': organization_id,
                'created_at': datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[{
                    'id': memory_id,
                    'values': embedding,
                    'metadata': full_metadata
                }],
                namespace=organization_id
            )
            
            self.logger.info(f"Stored memory {memory_id} for organization {organization_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            raise
    
    async def neural_search(self, query: str, organization_id: str, 
                           memory_type: Optional[str] = None, limit: int = 10,
                           threshold: float = 0.7) -> List[MemorySearchResult]:
        """Perform neural similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
            
            # Prepare filter
            filter_dict = {'organization_id': organization_id}
            if memory_type:
                filter_dict['memory_type'] = memory_type
            
            # Search
            results = self.index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=organization_id,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Convert to MemorySearchResult objects
            search_results = []
            for match in results['matches']:
                if match['score'] >= threshold:
                    metadata = match['metadata']
                    result = MemorySearchResult(
                        id=match['id'],
                        content=metadata['content'],
                        metadata=metadata,
                        score=match['score'],
                        memory_type=metadata['memory_type'],
                        organization_id=metadata['organization_id'],
                        created_at=datetime.fromisoformat(metadata['created_at'])
                    )
                    search_results.append(result)
            
            self.logger.info(f"Neural search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in neural search: {e}")
            return []
    
    async def conceptual_search(self, concept: str, organization_id: str,
                               limit: int = 10) -> List[MemorySearchResult]:
        """Search for memories related to a concept"""
        # For mock implementation, use neural search
        return await self.neural_search(concept, organization_id, limit=limit)
    
    async def knowledge_search(self, domain: str, organization_id: str,
                              limit: int = 10) -> List[MemorySearchResult]:
        """Search for knowledge in a specific domain"""
        # For mock implementation, use neural search with domain filter
        return await self.neural_search(domain, organization_id, 
                                       memory_type='knowledge', limit=limit)
    
    async def hybrid_search(self, query: str, organization_id: str,
                           limit: int = 10) -> List[MemorySearchResult]:
        """Perform hybrid search combining multiple methods"""
        # For mock implementation, just use neural search
        return await self.neural_search(query, organization_id, limit=limit)
    
    async def get_memory_insights(self, organization_id: str) -> MemoryInsights:
        """Get analytics and insights about memory usage"""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            # Mock insights for testing
            insights = MemoryInsights(
                total_memories=stats.get('total_vector_count', 0),
                memory_types={'general': 10, 'knowledge': 5, 'context': 3},
                organizations={organization_id: stats.get('total_vector_count', 0)},
                recent_activity=[],
                storage_stats=stats
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting memory insights: {e}")
            return MemoryInsights(
                total_memories=0,
                memory_types={},
                organizations={},
                recent_activity=[],
                storage_stats={}
            )
    
    async def delete_memory(self, memory_id: str, organization_id: str) -> bool:
        """Delete a specific memory"""
        try:
            result = self.index.delete(
                ids=[memory_id],
                namespace=organization_id
            )
            
            success = result.get('deleted_count', 0) > 0
            if success:
                self.logger.info(f"Deleted memory {memory_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the KSE system"""
        try:
            # Test OpenAI connection
            openai_status = "healthy"
            try:
                await self.get_embedding("test")
            except Exception:
                openai_status = "unhealthy"
            
            # Test Pinecone connection (mock always healthy)
            pinecone_status = "healthy"
            
            is_healthy = openai_status == 'healthy' and pinecone_status == 'healthy'
            return {
                'healthy': is_healthy,
                'status': 'healthy' if is_healthy else 'unhealthy',
                'openai': openai_status,
                'pinecone': pinecone_status,
                'index_name': self.index_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }