"""
Concept Store Backend Implementations
Multi-backend support for PostgreSQL, MongoDB, and Elasticsearch
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from ..core.interfaces import ConceptStoreInterface
from ..core.config import ConceptStoreConfig
from ..exceptions import ConfigurationError, ConnectionError

logger = logging.getLogger(__name__)


class PostgreSQLConceptStore(ConceptStoreInterface):
    """PostgreSQL concept store implementation."""
    
    def __init__(self, config: ConceptStoreConfig):
        self.config = config
        self.pool = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize PostgreSQL connection."""
        try:
            import asyncpg
            
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=1,
                max_size=10
            )
            
            # Create tables if they don't exist
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS concepts (
                        id VARCHAR PRIMARY KEY,
                        entity_id VARCHAR NOT NULL,
                        domain VARCHAR NOT NULL,
                        coordinates JSONB NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_concepts_entity_id ON concepts(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
                    CREATE INDEX IF NOT EXISTS idx_concepts_coordinates ON concepts USING GIN(coordinates);
                """)
            
            self._initialized = True
            logger.info(f"PostgreSQL concept store initialized: {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL initialization failed: {e}")
    
    async def store_concepts(self, concepts: List[Dict[str, Any]]) -> bool:
        """Store concepts in PostgreSQL."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                for concept in concepts:
                    await conn.execute("""
                        INSERT INTO concepts (id, entity_id, domain, coordinates, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            coordinates = EXCLUDED.coordinates,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """, 
                    concept['id'],
                    concept['entity_id'],
                    concept['domain'],
                    concept['coordinates'],
                    concept.get('metadata', {})
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store concepts in PostgreSQL: {e}")
            return False
    
    async def search_concepts(self, domain: str, coordinates: Dict[str, float], 
                            threshold: float = 0.1, limit: int = 10) -> List[Dict[str, Any]]:
        """Search concepts by similarity in PostgreSQL."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build similarity query based on coordinates
            coord_conditions = []
            for dim, value in coordinates.items():
                coord_conditions.append(f"ABS((coordinates->>'{dim}')::float - {value}) < {threshold}")
            
            where_clause = " AND ".join(coord_conditions)
            
            query = f"""
                SELECT id, entity_id, domain, coordinates, metadata
                FROM concepts
                WHERE domain = $1 AND {where_clause}
                ORDER BY (
                    SELECT AVG(ABS((coordinates->>key)::float - value::float))
                    FROM jsonb_each_text($2::jsonb) AS kv(key, value)
                    WHERE coordinates ? key
                )
                LIMIT $3
            """
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, domain, coordinates, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'entity_id': row['entity_id'],
                        'domain': row['domain'],
                        'coordinates': row['coordinates'],
                        'metadata': row['metadata']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search concepts in PostgreSQL: {e}")
            return []
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept by ID from PostgreSQL."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, entity_id, domain, coordinates, metadata FROM concepts WHERE id = $1",
                    concept_id
                )
                
                if row:
                    return {
                        'id': row['id'],
                        'entity_id': row['entity_id'],
                        'domain': row['domain'],
                        'coordinates': row['coordinates'],
                        'metadata': row['metadata']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get concept from PostgreSQL: {e}")
            return None
    
    async def delete_concepts(self, concept_ids: List[str]) -> bool:
        """Delete concepts from PostgreSQL."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM concepts WHERE id = ANY($1)",
                    concept_ids
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete concepts from PostgreSQL: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check PostgreSQL health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM concepts")
                
                return {
                    "healthy": True,
                    "concept_count": result,
                    "backend": "postgresql"
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class MongoDBConceptStore(ConceptStoreInterface):
    """MongoDB concept store implementation."""
    
    def __init__(self, config: ConceptStoreConfig):
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize MongoDB connection."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            connection_string = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self.client = AsyncIOMotorClient(connection_string)
            self.db = self.client[self.config.database]
            self.collection = self.db.concepts
            
            # Create indexes
            await self.collection.create_index("entity_id")
            await self.collection.create_index("domain")
            await self.collection.create_index("coordinates")
            
            self._initialized = True
            logger.info(f"MongoDB concept store initialized: {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise ConnectionError(f"MongoDB initialization failed: {e}")
    
    async def store_concepts(self, concepts: List[Dict[str, Any]]) -> bool:
        """Store concepts in MongoDB."""
        if not self._initialized:
            await self.initialize()
        
        try:
            operations = []
            for concept in concepts:
                operations.append({
                    'replaceOne': {
                        'filter': {'id': concept['id']},
                        'replacement': concept,
                        'upsert': True
                    }
                })
            
            if operations:
                await self.collection.bulk_write(operations)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store concepts in MongoDB: {e}")
            return False
    
    async def search_concepts(self, domain: str, coordinates: Dict[str, float], 
                            threshold: float = 0.1, limit: int = 10) -> List[Dict[str, Any]]:
        """Search concepts by similarity in MongoDB."""
        logger.info("MongoDB concept search not yet implemented")
        return []
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept by ID from MongoDB."""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.collection.find_one({'id': concept_id})
            return result
            
        except Exception as e:
            logger.error(f"Failed to get concept from MongoDB: {e}")
            return None
    
    async def delete_concepts(self, concept_ids: List[str]) -> bool:
        """Delete concepts from MongoDB."""
        if not self._initialized:
            await self.initialize()
        
        try:
            await self.collection.delete_many({'id': {'$in': concept_ids}})
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete concepts from MongoDB: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            count = await self.collection.count_documents({})
            
            return {
                "healthy": True,
                "concept_count": count,
                "backend": "mongodb"
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class ElasticsearchConceptStore(ConceptStoreInterface):
    """Elasticsearch concept store implementation."""
    
    def __init__(self, config: ConceptStoreConfig):
        self.config = config
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Elasticsearch connection."""
        try:
            from elasticsearch import AsyncElasticsearch
            
            self.client = AsyncElasticsearch(
                hosts=[f"{self.config.host}:{self.config.port}"],
                http_auth=(self.config.username, self.config.password) if self.config.username else None
            )
            
            # Create index if it doesn't exist
            index_name = "concepts"
            if not await self.client.indices.exists(index=index_name):
                await self.client.indices.create(
                    index=index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "entity_id": {"type": "keyword"},
                                "domain": {"type": "keyword"},
                                "coordinates": {"type": "object"},
                                "metadata": {"type": "object"}
                            }
                        }
                    }
                )
            
            self._initialized = True
            logger.info(f"Elasticsearch concept store initialized: {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise ConnectionError(f"Elasticsearch initialization failed: {e}")
    
    async def store_concepts(self, concepts: List[Dict[str, Any]]) -> bool:
        """Store concepts in Elasticsearch."""
        logger.info("Elasticsearch concept storage not yet implemented")
        return True
    
    async def search_concepts(self, domain: str, coordinates: Dict[str, float], 
                            threshold: float = 0.1, limit: int = 10) -> List[Dict[str, Any]]:
        """Search concepts by similarity in Elasticsearch."""
        logger.info("Elasticsearch concept search not yet implemented")
        return []
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept by ID from Elasticsearch."""
        logger.info("Elasticsearch concept retrieval not yet implemented")
        return None
    
    async def delete_concepts(self, concept_ids: List[str]) -> bool:
        """Delete concepts from Elasticsearch."""
        logger.info("Elasticsearch concept deletion not yet implemented")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {"healthy": True, "backend": "elasticsearch"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

class InMemoryConceptStore(ConceptStoreInterface):
    """In-memory concept store implementation for testing."""
    
    def __init__(self, config: ConceptStoreConfig):
        self.config = config
        self.conceptual_spaces = {}  # entity_id -> ConceptualSpace
        self.dimension_stats = {}  # domain -> dimension -> stats
        self._initialized = False
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the in-memory concept store."""
        try:
            self._connected = True
            logger.info("Connected to in-memory concept store")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to in-memory concept store: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the in-memory concept store."""
        try:
            self._connected = False
            logger.info("Disconnected from in-memory concept store")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from in-memory concept store: {e}")
            return False
    
    async def store_conceptual_space(self, entity_id: str,
                                   conceptual_space) -> bool:
        """Store conceptual space for an entity."""
        try:
            # For testing, we'll store as dict if it's not already
            if hasattr(conceptual_space, '__dict__'):
                space_data = conceptual_space.__dict__
            else:
                space_data = conceptual_space
            
            self.conceptual_spaces[entity_id] = space_data
            
            # Update dimension statistics
            domain = space_data.get('domain', 'general')
            if domain not in self.dimension_stats:
                self.dimension_stats[domain] = {}
            
            dimensions = space_data.get('dimensions', {})
            for dim_name, dim_value in dimensions.items():
                if dim_name not in self.dimension_stats[domain]:
                    self.dimension_stats[domain][dim_name] = {
                        'min': dim_value, 'max': dim_value, 'sum': 0, 'count': 0
                    }
                
                stats = self.dimension_stats[domain][dim_name]
                stats['min'] = min(stats['min'], dim_value)
                stats['max'] = max(stats['max'], dim_value)
                stats['sum'] += dim_value
                stats['count'] += 1
            
            logger.info(f"Stored conceptual space for entity {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store conceptual space in memory: {e}")
            return False
    
    async def get_conceptual_space(self, entity_id: str):
        """Get conceptual space for an entity."""
        try:
            return self.conceptual_spaces.get(entity_id)
        except Exception as e:
            logger.error(f"Failed to get conceptual space from memory: {e}")
            return None
    
    async def update_conceptual_space(self, entity_id: str,
                                    conceptual_space) -> bool:
        """Update conceptual space for an entity."""
        try:
            # Same as store for in-memory implementation
            return await self.store_conceptual_space(entity_id, conceptual_space)
        except Exception as e:
            logger.error(f"Failed to update conceptual space in memory: {e}")
            return False
    
    async def delete_conceptual_space(self, entity_id: str) -> bool:
        """Delete conceptual space for an entity."""
        try:
            if entity_id in self.conceptual_spaces:
                del self.conceptual_spaces[entity_id]
                logger.info(f"Deleted conceptual space for entity {entity_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete conceptual space from memory: {e}")
            return False
    
    async def search_by_dimensions(self, dimension_filters: Dict[str, Tuple[float, float]],
                                  domain: Optional[str] = None,
                                  limit: int = 100) -> List[Tuple[str, Any, float]]:
        """Search entities by conceptual dimensions."""
        try:
            results = []
            
            for entity_id, space_data in self.conceptual_spaces.items():
                # Filter by domain if specified
                if domain and space_data.get('domain') != domain:
                    continue
                
                dimensions = space_data.get('dimensions', {})
                score = 0
                matches = 0
                
                # Check if entity matches dimension filters
                for dim_name, (min_val, max_val) in dimension_filters.items():
                    if dim_name in dimensions:
                        dim_value = dimensions[dim_name]
                        if min_val <= dim_value <= max_val:
                            matches += 1
                            # Simple scoring based on how close to center of range
                            center = (min_val + max_val) / 2
                            distance = abs(dim_value - center)
                            range_size = max_val - min_val
                            score += 1 - (distance / (range_size / 2)) if range_size > 0 else 1
                
                # Only include if matches all filters
                if matches == len(dimension_filters):
                    results.append((entity_id, space_data, score))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search by dimensions in memory: {e}")
            return []
    
    async def get_dimension_statistics(self, domain: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for conceptual dimensions."""
        try:
            if domain:
                domain_stats = self.dimension_stats.get(domain, {})
                result = {}
                for dim_name, stats in domain_stats.items():
                    if stats['count'] > 0:
                        result[dim_name] = {
                            'min': stats['min'],
                            'max': stats['max'],
                            'avg': stats['sum'] / stats['count'],
                            'count': stats['count']
                        }
                return result
            else:
                # Return stats for all domains
                result = {}
                for domain_name, domain_stats in self.dimension_stats.items():
                    result[domain_name] = {}
                    for dim_name, stats in domain_stats.items():
                        if stats['count'] > 0:
                            result[domain_name][dim_name] = {
                                'min': stats['min'],
                                'max': stats['max'],
                                'avg': stats['sum'] / stats['count'],
                                'count': stats['count']
                            }
                return result
        except Exception as e:
            logger.error(f"Failed to get dimension statistics from memory: {e}")
            return {}
    
    async def batch_store_conceptual_spaces(self,
                                          spaces: List[Tuple[str, Any]]) -> bool:
        """Batch store conceptual spaces."""
        try:
            for entity_id, conceptual_space in spaces:
                await self.store_conceptual_space(entity_id, conceptual_space)
            
            logger.info(f"Batch stored {len(spaces)} conceptual spaces")
            return True
        except Exception as e:
            logger.error(f"Failed to batch store conceptual spaces in memory: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize in-memory concept store."""
        try:
            await self.connect()
            self._initialized = True
            logger.info("In-memory concept store initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize in-memory concept store: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check in-memory concept store health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {
                "healthy": True,
                "backend": "memory",
                "conceptual_space_count": len(self.conceptual_spaces),
                "connected": self._connected
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}



def get_concept_store(config: ConceptStoreConfig) -> ConceptStoreInterface:
    """Factory function to get concept store implementation."""
    
    if config.backend == "postgresql":
        return PostgreSQLConceptStore(config)
    elif config.backend == "mongodb":
        return MongoDBConceptStore(config)
    elif config.backend == "elasticsearch":
        return ElasticsearchConceptStore(config)
    elif config.backend == "memory":
        return InMemoryConceptStore(config)
    else:
        raise ConfigurationError(f"Unsupported concept store backend: {config.backend}")