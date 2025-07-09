"""
KSE Memory SDK Conceptual Service
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import numpy as np
import logging
from datetime import datetime
from ..core.interfaces import ConceptualServiceInterface
from ..core.models import ConceptualSpace, DOMAIN_DIMENSIONS
from ..exceptions import ConceptualError, ValidationError

logger = logging.getLogger(__name__)


class ConceptualService(ConceptualServiceInterface):
    """
    Conceptual service for managing conceptual spaces and dimensional reasoning.
    Handles domain-specific conceptual dimensions and semantic relationships.
    """
    
    def __init__(self, config: Dict[str, Any], cache_service=None):
        """
        Initialize conceptual service.
        
        Args:
            config: Configuration dictionary or dataclass
            cache_service: Optional cache service
        """
        self.config = config
        self.cache_service = cache_service
        self.conceptual_spaces: Dict[str, ConceptualSpace] = {}
        self.domain_mappings: Dict[str, str] = {}  # entity_type -> domain
        
        # Handle both dict and dataclass config
        if hasattr(config, '__dict__'):
            # Dataclass config
            self.default_domain = getattr(config, 'default_domain', 'enterprise')
            self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
            self.max_concepts_per_space = getattr(config, 'max_concepts_per_space', 1000)
        else:
            # Dict config
            self.default_domain = config.get('default_domain', 'enterprise')
            self.similarity_threshold = config.get('similarity_threshold', 0.7)
            self.max_concepts_per_space = config.get('max_concepts_per_space', 1000)
        
    async def initialize(self) -> bool:
        """Initialize conceptual service."""
        try:
            # Initialize default conceptual spaces for all domains
            await self._initialize_domain_spaces()
            
            logger.info("Conceptual service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize conceptual service: {e}")
            return False
    
    async def create_conceptual_space(self, domain: str, 
                                    custom_dimensions: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conceptual space for a domain.
        
        Args:
            domain: Domain name
            custom_dimensions: Custom dimensions to add
            
        Returns:
            Conceptual space ID
        """
        try:
            # Get base dimensions for domain
            base_dimensions = DOMAIN_DIMENSIONS.get(domain, DOMAIN_DIMENSIONS['enterprise'])
            
            # Merge with custom dimensions
            if custom_dimensions:
                dimensions = {**base_dimensions, **custom_dimensions}
            else:
                dimensions = base_dimensions
            
            # Create conceptual space
            space = ConceptualSpace(
                domain=domain,
                dimensions=dimensions,
                metadata={'created_at': datetime.now().isoformat()}
            )
            
            space_id = f"{domain}_space"
            self.conceptual_spaces[space_id] = space
            
            logger.info(f"Created conceptual space for domain: {domain}")
            return space_id
            
        except Exception as e:
            logger.error(f"Failed to create conceptual space: {e}")
            raise ConceptualError(f"Failed to create conceptual space: {e}", concept=domain)
    
    async def map_entity_to_conceptual_space(self, entity_data: Dict[str, Any], 
                                           domain: Optional[str] = None) -> Dict[str, float]:
        """
        Map an entity to its position in conceptual space.
        
        Args:
            entity_data: Entity data to map
            domain: Target domain (auto-detected if not provided)
            
        Returns:
            Conceptual coordinates
        """
        try:
            # Determine domain
            if not domain:
                domain = await self._detect_domain(entity_data)
            
            # Get conceptual space
            space_id = f"{domain}_space"
            if space_id not in self.conceptual_spaces:
                await self.create_conceptual_space(domain)
            
            space = self.conceptual_spaces[space_id]
            
            # Map entity to conceptual coordinates
            coordinates = {}
            
            for dimension_name, dimension_config in space.dimensions.items():
                coordinate = await self._calculate_dimension_coordinate(
                    entity_data, dimension_name, dimension_config
                )
                coordinates[dimension_name] = coordinate
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Failed to map entity to conceptual space: {e}")
            raise ConceptualError(f"Entity mapping failed: {e}")
    
    async def calculate_conceptual_similarity(self, entity1_coords: Dict[str, float],
                                            entity2_coords: Dict[str, float],
                                            weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate conceptual similarity between two entities.
        
        Args:
            entity1_coords: First entity's conceptual coordinates
            entity2_coords: Second entity's conceptual coordinates
            weights: Dimension weights for similarity calculation
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Get common dimensions
            common_dims = set(entity1_coords.keys()) & set(entity2_coords.keys())
            
            if not common_dims:
                return 0.0
            
            # Calculate weighted distance
            total_distance = 0.0
            total_weight = 0.0
            
            for dim in common_dims:
                weight = weights.get(dim, 1.0) if weights else 1.0
                distance = abs(entity1_coords[dim] - entity2_coords[dim])
                
                total_distance += weight * distance
                total_weight += weight
            
            # Normalize distance to similarity
            if total_weight == 0:
                return 0.0
            
            avg_distance = total_distance / total_weight
            similarity = max(0.0, 1.0 - avg_distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to calculate conceptual similarity: {e}")
            return 0.0
    
    async def find_conceptually_similar_entities(self, target_coords: Dict[str, float],
                                               candidate_coords: List[Dict[str, float]],
                                               threshold: Optional[float] = None,
                                               max_results: int = 10) -> List[Tuple[int, float]]:
        """
        Find conceptually similar entities.
        
        Args:
            target_coords: Target entity coordinates
            candidate_coords: List of candidate entity coordinates
            threshold: Similarity threshold
            max_results: Maximum results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            threshold = threshold or self.similarity_threshold
            similarities = []
            
            for i, candidate_coords_item in enumerate(candidate_coords):
                similarity = await self.calculate_conceptual_similarity(
                    target_coords, candidate_coords_item
                )
                
                if similarity >= threshold:
                    similarities.append((i, similarity))
            
            # Sort by similarity (descending) and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar entities: {e}")
            return []
    
    async def get_conceptual_space_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a conceptual space.
        
        Args:
            domain: Domain name
            
        Returns:
            Conceptual space information
        """
        try:
            space_id = f"{domain}_space"
            if space_id not in self.conceptual_spaces:
                return None
            
            space = self.conceptual_spaces[space_id]
            
            return {
                'domain': space.domain,
                'dimensions': list(space.dimensions.keys()),
                'dimension_count': len(space.dimensions),
                'metadata': space.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get conceptual space info: {e}")
            return None
    
    async def update_conceptual_space(self, domain: str, 
                                    new_dimensions: Dict[str, Any]) -> bool:
        """
        Update a conceptual space with new dimensions.
        
        Args:
            domain: Domain name
            new_dimensions: New dimensions to add
            
        Returns:
            True if updated successfully
        """
        try:
            space_id = f"{domain}_space"
            if space_id not in self.conceptual_spaces:
                return False
            
            space = self.conceptual_spaces[space_id]
            space.dimensions.update(new_dimensions)
            space.metadata['updated_at'] = datetime.now().isoformat()
            
            logger.info(f"Updated conceptual space for domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update conceptual space: {e}")
            return False
    
    async def analyze_conceptual_clusters(self, entity_coords_list: List[Dict[str, float]],
                                        num_clusters: int = 5) -> Dict[str, Any]:
        """
        Analyze conceptual clusters in entity coordinates.
        
        Args:
            entity_coords_list: List of entity coordinates
            num_clusters: Number of clusters to identify
            
        Returns:
            Cluster analysis results
        """
        try:
            if not entity_coords_list:
                return {'clusters': [], 'centroids': [], 'analysis': {}}
            
            # Simple k-means clustering in conceptual space
            clusters = await self._perform_conceptual_clustering(
                entity_coords_list, num_clusters
            )
            
            # Calculate cluster statistics
            analysis = {
                'total_entities': len(entity_coords_list),
                'num_clusters': len(clusters),
                'cluster_sizes': [len(cluster) for cluster in clusters],
                'avg_cluster_size': sum(len(cluster) for cluster in clusters) / len(clusters) if clusters else 0
            }
            
            return {
                'clusters': clusters,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze conceptual clusters: {e}")
            return {'clusters': [], 'analysis': {}}
    
    async def _initialize_domain_spaces(self) -> None:
        """Initialize conceptual spaces for all domains."""
        try:
            for domain in DOMAIN_DIMENSIONS.keys():
                await self.create_conceptual_space(domain)
                
        except Exception as e:
            logger.error(f"Failed to initialize domain spaces: {e}")
            raise
    
    async def _detect_domain(self, entity_data: Dict[str, Any]) -> str:
        """Detect the most appropriate domain for an entity."""
        try:
            # Simple domain detection based on entity fields
            entity_type = entity_data.get('type', '').lower()
            
            # Healthcare indicators
            if any(keyword in str(entity_data).lower() for keyword in 
                   ['patient', 'medical', 'diagnosis', 'treatment', 'hospital', 'doctor']):
                return 'healthcare'
            
            # Finance indicators
            elif any(keyword in str(entity_data).lower() for keyword in 
                     ['price', 'cost', 'revenue', 'profit', 'investment', 'financial']):
                return 'finance'
            
            # Real estate indicators
            elif any(keyword in str(entity_data).lower() for keyword in 
                     ['property', 'location', 'address', 'square_feet', 'bedrooms']):
                return 'real_estate'
            
            # Research indicators
            elif any(keyword in str(entity_data).lower() for keyword in 
                     ['research', 'study', 'experiment', 'hypothesis', 'publication']):
                return 'research'
            
            # Retail indicators
            elif any(keyword in str(entity_data).lower() for keyword in 
                     ['product', 'brand', 'category', 'inventory', 'sales']):
                return 'retail'
            
            # Marketing indicators
            elif any(keyword in str(entity_data).lower() for keyword in 
                     ['campaign', 'audience', 'engagement', 'conversion', 'channel']):
                return 'marketing'
            
            # Default to enterprise
            else:
                return self.default_domain
                
        except Exception as e:
            logger.error(f"Failed to detect domain: {e}")
            return self.default_domain
    
    async def _calculate_dimension_coordinate(self, entity_data: Dict[str, Any],
                                            dimension_name: str,
                                            dimension_config: Dict[str, Any]) -> float:
        """Calculate coordinate for a specific dimension."""
        try:
            dimension_type = dimension_config.get('type', 'numeric')
            
            if dimension_type == 'numeric':
                return await self._calculate_numeric_coordinate(
                    entity_data, dimension_name, dimension_config
                )
            elif dimension_type == 'categorical':
                return await self._calculate_categorical_coordinate(
                    entity_data, dimension_name, dimension_config
                )
            elif dimension_type == 'semantic':
                return await self._calculate_semantic_coordinate(
                    entity_data, dimension_name, dimension_config
                )
            else:
                return 0.5  # Default middle value
                
        except Exception as e:
            logger.error(f"Failed to calculate dimension coordinate: {e}")
            return 0.5
    
    async def _calculate_numeric_coordinate(self, entity_data: Dict[str, Any],
                                          dimension_name: str,
                                          dimension_config: Dict[str, Any]) -> float:
        """Calculate coordinate for numeric dimension."""
        try:
            # Get value from entity data
            value = entity_data.get(dimension_name, 0)
            
            # Normalize to 0-1 range
            min_val = dimension_config.get('min', 0)
            max_val = dimension_config.get('max', 100)
            
            if max_val == min_val:
                return 0.5
            
            normalized = (float(value) - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Failed to calculate numeric coordinate: {e}")
            return 0.5
    
    async def _calculate_categorical_coordinate(self, entity_data: Dict[str, Any],
                                              dimension_name: str,
                                              dimension_config: Dict[str, Any]) -> float:
        """Calculate coordinate for categorical dimension."""
        try:
            # Get value from entity data
            value = str(entity_data.get(dimension_name, '')).lower()
            
            # Map categories to coordinates
            categories = dimension_config.get('categories', [])
            if not categories:
                return 0.5
            
            if value in categories:
                index = categories.index(value)
                return index / (len(categories) - 1) if len(categories) > 1 else 0.5
            else:
                return 0.5  # Unknown category
                
        except Exception as e:
            logger.error(f"Failed to calculate categorical coordinate: {e}")
            return 0.5
    
    async def _calculate_semantic_coordinate(self, entity_data: Dict[str, Any],
                                           dimension_name: str,
                                           dimension_config: Dict[str, Any]) -> float:
        """Calculate coordinate for semantic dimension."""
        try:
            # Get text value from entity data
            text_fields = dimension_config.get('text_fields', [dimension_name])
            text_content = ' '.join(
                str(entity_data.get(field, '')) for field in text_fields
            )
            
            if not text_content.strip():
                return 0.5
            
            # Simple semantic scoring based on keywords
            positive_keywords = dimension_config.get('positive_keywords', [])
            negative_keywords = dimension_config.get('negative_keywords', [])
            
            text_lower = text_content.lower()
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
            negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)
            
            total_keywords = len(positive_keywords) + len(negative_keywords)
            if total_keywords == 0:
                return 0.5
            
            # Calculate semantic coordinate
            net_score = positive_score - negative_score
            normalized_score = (net_score + total_keywords) / (2 * total_keywords)
            
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate semantic coordinate: {e}")
            return 0.5
    
    async def _perform_conceptual_clustering(self, entity_coords_list: List[Dict[str, float]],
                                           num_clusters: int) -> List[List[int]]:
        """Perform clustering in conceptual space."""
        try:
            if len(entity_coords_list) <= num_clusters:
                # Each entity is its own cluster
                return [[i] for i in range(len(entity_coords_list))]
            
            # Simple k-means clustering
            # Initialize centroids randomly
            import random
            centroids = random.sample(range(len(entity_coords_list)), num_clusters)
            
            clusters = [[] for _ in range(num_clusters)]
            
            # Assign entities to nearest centroid
            for i, coords in enumerate(entity_coords_list):
                best_cluster = 0
                best_similarity = -1
                
                for j, centroid_idx in enumerate(centroids):
                    centroid_coords = entity_coords_list[centroid_idx]
                    similarity = await self.calculate_conceptual_similarity(
                        coords, centroid_coords
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = j
                
                clusters[best_cluster].append(i)
            
            # Remove empty clusters
            clusters = [cluster for cluster in clusters if cluster]
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to perform clustering: {e}")
            return [[i] for i in range(len(entity_coords_list))]
    
    async def compute_dimensions(self, entity: 'Entity',
                               custom_dimensions: Optional[Dict[str, str]] = None) -> 'ConceptualSpace':
        """Compute conceptual dimensions for an entity."""
        try:
            # Mock implementation - create basic conceptual space
            from ..core.models import ConceptualSpace
            
            # Basic dimensions based on entity type and domain
            dimensions = {
                'relevance': 0.8,
                'complexity': 0.6,
                'importance': 0.7,
                'novelty': 0.5
            }
            
            # Add custom dimensions if provided
            if custom_dimensions:
                for dim_name, dim_value in custom_dimensions.items():
                    try:
                        dimensions[dim_name] = float(dim_value)
                    except (ValueError, TypeError):
                        dimensions[dim_name] = 0.5
            
            conceptual_space = ConceptualSpace(
                entity_id=entity.id,
                domain=entity.domain,
                dimensions=dimensions,
                metadata={'computed_at': 'mock_timestamp'}
            )
            
            return conceptual_space
        except Exception as e:
            logger.error(f"Failed to compute dimensions: {e}")
            raise
    
    async def batch_compute_dimensions(self, entities: List['Entity'],
                                     custom_dimensions: Optional[Dict[str, str]] = None) -> List['ConceptualSpace']:
        """Compute conceptual dimensions for multiple entities."""
        try:
            spaces = []
            for entity in entities:
                space = await self.compute_dimensions(entity, custom_dimensions)
                spaces.append(space)
            return spaces
        except Exception as e:
            logger.error(f"Failed to batch compute dimensions: {e}")
            raise
    
    async def analyze_dimension_relationships(self, domain: str) -> Dict[str, Dict[str, float]]:
        """Analyze relationships between conceptual dimensions."""
        try:
            # Mock implementation - return basic dimension correlations
            return {
                'relevance': {'complexity': 0.3, 'importance': 0.7, 'novelty': 0.2},
                'complexity': {'relevance': 0.3, 'importance': 0.4, 'novelty': 0.6},
                'importance': {'relevance': 0.7, 'complexity': 0.4, 'novelty': 0.1},
                'novelty': {'relevance': 0.2, 'complexity': 0.6, 'importance': 0.1}
            }
        except Exception as e:
            logger.error(f"Failed to analyze dimension relationships: {e}")
            raise
    
    async def suggest_dimensions(self, entities: List['Entity'],
                               domain: str) -> Dict[str, str]:
        """Suggest dimensions for entities in a domain."""
        try:
            # Mock implementation - return basic dimension suggestions
            suggestions = {
                'relevance': 'How relevant is this entity to the domain context',
                'complexity': 'How complex or sophisticated is this entity',
                'importance': 'How important is this entity in the domain',
                'novelty': 'How novel or unique is this entity'
            }
            
            # Add domain-specific suggestions
            if domain == 'healthcare':
                suggestions.update({
                    'clinical_impact': 'Impact on clinical outcomes',
                    'safety_profile': 'Safety and risk assessment'
                })
            elif domain == 'finance':
                suggestions.update({
                    'risk_level': 'Financial risk assessment',
                    'liquidity': 'Asset liquidity measure'
                })
            elif domain == 'marketing':
                suggestions.update({
                    'engagement_potential': 'Potential for customer engagement',
                    'conversion_likelihood': 'Likelihood of conversion'
                })
            
            return suggestions
        except Exception as e:
            logger.error(f"Failed to suggest dimensions: {e}")
            raise
    
    async def validate_dimensions(self, dimensions: Dict[str, float],
                                domain: str) -> Dict[str, bool]:
        """Validate conceptual dimensions for a domain."""
        try:
            # Mock implementation - basic validation
            validation_results = {}
            
            for dim_name, dim_value in dimensions.items():
                # Check if value is in valid range [0, 1]
                is_valid = isinstance(dim_value, (int, float)) and 0 <= dim_value <= 1
                validation_results[dim_name] = is_valid
            
            return validation_results
        except Exception as e:
            logger.error(f"Failed to validate dimensions: {e}")
            raise