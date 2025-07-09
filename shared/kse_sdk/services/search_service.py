"""
KSE Memory SDK Search Service
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import logging
from datetime import datetime
from ..core.interfaces import SearchServiceInterface
from ..core.models import Entity, SearchResult, SearchQuery, SearchType
from ..exceptions import SearchError, ValidationError

logger = logging.getLogger(__name__)


class SearchService(SearchServiceInterface):
    """
    Search service for orchestrating hybrid search across neural embeddings,
    conceptual spaces, and knowledge graphs.
    """
    
    def __init__(self, config: Dict[str, Any], vector_store=None, graph_store=None,
                 concept_store=None, embedding_service=None, cache_service=None):
        """
        Initialize search service.
        
        Args:
            config: Configuration dictionary
            vector_store: Vector store instance
            graph_store: Graph store instance
            concept_store: Concept store instance
            embedding_service: Embedding service instance
            cache_service: Cache service instance
        """
        self.config = config
        
        # Store references to services
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.concept_store = concept_store
        self.embedding_service = embedding_service
        self.cache_service = cache_service
        
        # Search configuration
        self.default_search_type = config.get('default_search_type', 'hybrid')
        self.max_results = config.get('max_results', 100)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # Weight configuration for hybrid search
        self.search_weights = config.get('search_weights', {
            'neural': 0.4,
            'conceptual': 0.3,
            'graph': 0.3
        })
        
        # Search history for analytics
        self.search_history: List[Dict[str, Any]] = []
        self.max_history_size = config.get('max_history_size', 1000)
        
    async def initialize(self) -> bool:
        """Initialize search service."""
        try:
            logger.info("Search service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            return False
    
    async def search(self, query: str, search_type: str = "hybrid",
                    filters: Optional[Dict[str, Any]] = None,
                    limit: int = 10, **kwargs) -> List[SearchResult]:
        """
        Perform search across KSE Memory.
        
        Args:
            query: Search query
            search_type: Type of search (neural, conceptual, graph, hybrid)
            filters: Search filters
            limit: Maximum results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        try:
            search_start = datetime.now()
            
            # Validate inputs
            if not query.strip():
                raise ValidationError("Search query cannot be empty")
            
            limit = min(limit, self.max_results)
            
            # Perform search based on type
            if search_type == "neural":
                results = await self._neural_search(query, filters, limit, **kwargs)
            elif search_type == "conceptual":
                results = await self._conceptual_search(query, filters, limit, **kwargs)
            elif search_type == "graph":
                results = await self._graph_search(query, filters, limit, **kwargs)
            elif search_type == "hybrid":
                results = await self._hybrid_search(query, filters, limit, **kwargs)
            else:
                raise SearchError(f"Unsupported search type: {search_type}")
            
            # Log search for analytics
            search_duration = (datetime.now() - search_start).total_seconds()
            await self._log_search(query, search_type, len(results), search_duration)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search operation failed: {e}", search_type=search_type)
    
    async def semantic_search(self, query: str, domain: Optional[str] = None,
                            limit: int = 10) -> List[SearchResult]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            domain: Domain to search within
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            filters = {'domain': domain} if domain else None
            return await self.search(query, "neural", filters, limit)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise SearchError(f"Semantic search failed: {e}", search_type="semantic")
    
    async def conceptual_search(self, query: str, conceptual_filters: Optional[Dict[str, Any]] = None,
                              limit: int = 10) -> List[SearchResult]:
        """
        Perform conceptual space search.
        
        Args:
            query: Search query
            conceptual_filters: Conceptual dimension filters
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            filters = conceptual_filters or {}
            return await self.search(query, "conceptual", filters, limit)
            
        except Exception as e:
            logger.error(f"Conceptual search failed: {e}")
            raise SearchError(f"Conceptual search failed: {e}", search_type="conceptual")
    
    async def graph_search(self, query: str, relationship_types: Optional[List[str]] = None,
                          max_depth: int = 3, limit: int = 10) -> List[SearchResult]:
        """
        Perform knowledge graph search.
        
        Args:
            query: Search query
            relationship_types: Types of relationships to traverse
            max_depth: Maximum traversal depth
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            filters = {
                'relationship_types': relationship_types,
                'max_depth': max_depth
            }
            return await self.search(query, "graph", filters, limit)
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise SearchError(f"Graph search failed: {e}", search_type="graph")
    
    async def multi_modal_search(self, queries: Dict[str, str],
                               search_weights: Optional[Dict[str, float]] = None,
                               limit: int = 10) -> List[SearchResult]:
        """
        Perform multi-modal search with different query types.
        
        Args:
            queries: Dictionary of query_type -> query_text
            search_weights: Weights for different search modes
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            weights = search_weights or self.search_weights
            all_results = []
            
            # Perform searches for each query type
            for query_type, query_text in queries.items():
                if query_type in ['neural', 'conceptual', 'graph']:
                    results = await self.search(query_text, query_type, limit=limit)
                    
                    # Weight the results
                    weight = weights.get(query_type, 1.0)
                    for result in results:
                        result.score *= weight
                        result.metadata['search_type'] = query_type
                    
                    all_results.extend(results)
            
            # Merge and rank results
            merged_results = await self._merge_search_results(all_results)
            return merged_results[:limit]
            
        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            raise SearchError(f"Multi-modal search failed: {e}", search_type="multi_modal")
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum suggestions
            
        Returns:
            List of search suggestions
        """
        try:
            # Simple suggestion based on search history
            suggestions = []
            
            for search_record in self.search_history:
                query = search_record.get('query', '')
                if (partial_query.lower() in query.lower() and 
                    query not in suggestions and 
                    len(suggestions) < limit):
                    suggestions.append(query)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get search analytics and statistics.
        
        Returns:
            Search analytics data
        """
        try:
            if not self.search_history:
                return {
                    'total_searches': 0,
                    'avg_results': 0,
                    'avg_duration': 0,
                    'popular_queries': [],
                    'search_types': {}
                }
            
            total_searches = len(self.search_history)
            total_results = sum(record.get('result_count', 0) for record in self.search_history)
            total_duration = sum(record.get('duration', 0) for record in self.search_history)
            
            # Count search types
            search_types = {}
            query_counts = {}
            
            for record in self.search_history:
                search_type = record.get('search_type', 'unknown')
                search_types[search_type] = search_types.get(search_type, 0) + 1
                
                query = record.get('query', '')
                query_counts[query] = query_counts.get(query, 0) + 1
            
            # Get popular queries
            popular_queries = sorted(
                query_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                'total_searches': total_searches,
                'avg_results': total_results / total_searches,
                'avg_duration': total_duration / total_searches,
                'popular_queries': [{'query': q, 'count': c} for q, c in popular_queries],
                'search_types': search_types
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {}
    
    async def _neural_search(self, query: str, filters: Optional[Dict[str, Any]],
                           limit: int, **kwargs) -> List[SearchResult]:
        """Perform neural embedding search."""
        try:
            # This would integrate with the vector store backend
            # For now, return mock results
            results = []
            
            # Mock neural search results
            for i in range(min(limit, 5)):
                result = SearchResult(
                    entity=Entity(
                        id=f"neural_entity_{i}",
                        type="document",
                        data={
                            'title': f"Neural Result {i+1}",
                            'content': f"Content matching query: {query}",
                            'source': 'neural_search'
                        }
                    ),
                    score=0.9 - (i * 0.1),
                    metadata={
                        'search_type': 'neural',
                        'embedding_model': 'text-embedding-ada-002'
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Neural search failed: {e}")
            return []
    
    async def _conceptual_search(self, query: str, filters: Optional[Dict[str, Any]],
                               limit: int, **kwargs) -> List[SearchResult]:
        """Perform conceptual space search."""
        try:
            # This would integrate with the conceptual service
            # For now, return mock results
            results = []
            
            # Mock conceptual search results
            for i in range(min(limit, 5)):
                result = SearchResult(
                    entity=Entity(
                        id=f"conceptual_entity_{i}",
                        type="concept",
                        data={
                            'title': f"Conceptual Result {i+1}",
                            'description': f"Conceptually related to: {query}",
                            'domain': filters.get('domain', 'enterprise') if filters else 'enterprise'
                        }
                    ),
                    score=0.85 - (i * 0.1),
                    metadata={
                        'search_type': 'conceptual',
                        'conceptual_dimensions': ['quality', 'relevance', 'importance']
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Conceptual search failed: {e}")
            return []
    
    async def _graph_search(self, query: str, filters: Optional[Dict[str, Any]],
                          limit: int, **kwargs) -> List[SearchResult]:
        """Perform knowledge graph search."""
        try:
            # This would integrate with the graph store backend
            # For now, return mock results
            results = []
            
            # Mock graph search results
            for i in range(min(limit, 5)):
                result = SearchResult(
                    entity=Entity(
                        id=f"graph_entity_{i}",
                        type="node",
                        data={
                            'title': f"Graph Result {i+1}",
                            'description': f"Connected to query: {query}",
                            'relationships': [f"related_to_{j}" for j in range(3)]
                        }
                    ),
                    score=0.8 - (i * 0.1),
                    metadata={
                        'search_type': 'graph',
                        'path_length': i + 1,
                        'relationship_types': ['related_to', 'similar_to']
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _hybrid_search(self, query: str, filters: Optional[Dict[str, Any]],
                           limit: int, **kwargs) -> List[SearchResult]:
        """Perform hybrid search combining all search types."""
        try:
            # Perform searches with different types
            neural_results = await self._neural_search(query, filters, limit)
            conceptual_results = await self._conceptual_search(query, filters, limit)
            graph_results = await self._graph_search(query, filters, limit)
            
            # Apply weights
            for result in neural_results:
                result.score *= self.search_weights['neural']
                result.metadata['search_component'] = 'neural'
            
            for result in conceptual_results:
                result.score *= self.search_weights['conceptual']
                result.metadata['search_component'] = 'conceptual'
            
            for result in graph_results:
                result.score *= self.search_weights['graph']
                result.metadata['search_component'] = 'graph'
            
            # Merge all results
            all_results = neural_results + conceptual_results + graph_results
            merged_results = await self._merge_search_results(all_results)
            
            return merged_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _merge_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Merge and deduplicate search results."""
        try:
            # Group by entity ID
            entity_groups = {}
            
            for result in results:
                entity_id = result.entity.id
                if entity_id not in entity_groups:
                    entity_groups[entity_id] = []
                entity_groups[entity_id].append(result)
            
            # Merge grouped results
            merged_results = []
            
            for entity_id, group in entity_groups.items():
                if len(group) == 1:
                    merged_results.append(group[0])
                else:
                    # Merge multiple results for same entity
                    best_result = max(group, key=lambda x: x.score)
                    
                    # Combine metadata
                    combined_metadata = {}
                    for result in group:
                        combined_metadata.update(result.metadata)
                    
                    # Add fusion information
                    combined_metadata['fusion_count'] = len(group)
                    combined_metadata['fusion_scores'] = [r.score for r in group]
                    
                    best_result.metadata = combined_metadata
                    merged_results.append(best_result)
            
            # Sort by score
            merged_results.sort(key=lambda x: x.score, reverse=True)
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to merge search results: {e}")
            return results
    
    async def _log_search(self, query: str, search_type: str,
                         result_count: int, duration: float) -> None:
        """Log search for analytics."""
        try:
            search_record = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'search_type': search_type,
                'result_count': result_count,
                'duration': duration
            }
            
            self.search_history.append(search_record)
            
            # Trim history if too large
            if len(self.search_history) > self.max_history_size:
                self.search_history = self.search_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"Failed to log search: {e}")

    # Abstract method implementations required by SearchServiceInterface
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search based on SearchQuery object."""
        try:
            # Call the string-based search method with proper parameters
            if hasattr(self, '_search_by_string'):
                return await self._search_by_string(
                    query=query.query,
                    search_type=query.search_type.value,
                    filters=query.filters,
                    limit=query.limit
                )
            else:
                # Fallback to hybrid search for SearchQuery objects
                return await self._hybrid_search(
                    query=query.query,
                    filters=query.filters,
                    limit=query.limit
                )
        except Exception as e:
            logger.error(f"SearchQuery-based search failed: {e}")
            raise SearchError(f"Search failed: {e}")

    async def semantic_search(self, query_text: str, domain: Optional[str] = None,
                            limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic search (interface implementation)."""
        try:
            filters = {'domain': domain} if domain else None
            results = await self.search(query_text, "neural", filters, limit)
            # Filter by threshold
            return [r for r in results if r.score >= threshold]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise SearchError(f"Semantic search failed: {e}")

    async def conceptual_search(self, dimension_filters: Dict[str, float],
                              domain: Optional[str] = None,
                              limit: int = 10) -> List[SearchResult]:
        """Perform conceptual search (interface implementation)."""
        try:
            filters = {'domain': domain, 'dimensions': dimension_filters}
            # Use first dimension as query for mock implementation
            query = list(dimension_filters.keys())[0] if dimension_filters else "conceptual_query"
            return await self.search(query, "conceptual", filters, limit)
        except Exception as e:
            logger.error(f"Conceptual search failed: {e}")
            raise SearchError(f"Conceptual search failed: {e}")

    async def graph_search(self, entity_id: str, relationship_types: Optional[List[str]] = None,
                          max_depth: int = 2, limit: int = 10) -> List[SearchResult]:
        """Perform graph-based search (interface implementation)."""
        try:
            filters = {
                'entity_id': entity_id,
                'relationship_types': relationship_types,
                'max_depth': max_depth
            }
            return await self.search(entity_id, "graph", filters, limit)
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise SearchError(f"Graph search failed: {e}")

    async def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining multiple approaches."""
        try:
            return await self._hybrid_search(
                query=query.query,
                filters=query.filters,
                limit=query.limit
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")

    async def rerank_results(self, results: List[SearchResult],
                           query: SearchQuery) -> List[SearchResult]:
        """Rerank search results based on query context."""
        try:
            # Simple reranking based on query relevance
            # In a real implementation, this would use ML models
            
            # Apply query-specific weights
            reranked_results = []
            for result in results:
                new_score = result.score
                
                # Boost score if entity type matches query preferences
                if query.entity_types and result.entity.type in query.entity_types:
                    new_score *= 1.2
                
                # Apply domain relevance
                if query.domain and result.entity.data.get('domain') == query.domain:
                    new_score *= 1.1
                
                # Apply conceptual filters
                if query.conceptual_filters:
                    # Mock conceptual relevance calculation
                    conceptual_boost = sum(query.conceptual_filters.values()) / len(query.conceptual_filters)
                    new_score *= (1.0 + conceptual_boost * 0.1)
                
                # Create new result with updated score
                reranked_result = SearchResult(
                    entity=result.entity,
                    score=min(new_score, 1.0),  # Cap at 1.0
                    search_type=result.search_type,
                    explanation=f"Reranked: {result.explanation or 'N/A'}",
                    conceptual_similarity=result.conceptual_similarity,
                    semantic_similarity=result.semantic_similarity,
                    graph_relevance=result.graph_relevance
                )
                reranked_results.append(reranked_result)
            
            # Sort by new scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Result reranking failed: {e}")
            return results  # Return original results if reranking fails