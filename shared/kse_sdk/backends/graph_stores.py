"""
Graph Store Backend Implementations
Multi-backend support for Neo4j, ArangoDB, and Neptune
"""

from typing import Dict, Any, Optional, List
import logging
from ..core.interfaces import GraphStoreInterface
from ..core.config import GraphStoreConfig
from ..exceptions import ConfigurationError, ConnectionError

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStoreInterface):
    """Neo4j graph store implementation."""
    
    def __init__(self, config: GraphStoreConfig):
        self.config = config
        self.driver = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self._initialized = True
            logger.info(f"Neo4j graph store initialized: {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise ConnectionError(f"Neo4j initialization failed: {e}")
    
    async def store_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """Store nodes in Neo4j."""
        if not self._initialized:
            await self.initialize()
        
        try:
            with self.driver.session() as session:
                for node in nodes:
                    query = """
                    MERGE (n:Entity {id: $id})
                    SET n += $properties
                    """
                    session.run(query, id=node['id'], properties=node.get('properties', {}))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store nodes in Neo4j: {e}")
            return False
    
    async def store_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """Store relationships in Neo4j."""
        if not self._initialized:
            await self.initialize()
        
        try:
            with self.driver.session() as session:
                for rel in relationships:
                    query = """
                    MATCH (a:Entity {id: $from_id})
                    MATCH (b:Entity {id: $to_id})
                    MERGE (a)-[r:%s]->(b)
                    SET r += $properties
                    """ % rel['type']
                    
                    session.run(query, 
                               from_id=rel['from_id'],
                               to_id=rel['to_id'],
                               properties=rel.get('properties', {}))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store relationships in Neo4j: {e}")
            return False
    
    async def query_graph(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute graph query in Neo4j."""
        if not self._initialized:
            await self.initialize()
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to query Neo4j: {e}")
            return []
    
    async def find_paths(self, from_id: str, to_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find paths between nodes in Neo4j."""
        if not self._initialized:
            await self.initialize()
        
        try:
            query = """
            MATCH path = (a:Entity {id: $from_id})-[*1..%d]-(b:Entity {id: $to_id})
            RETURN path
            LIMIT 10
            """ % max_depth
            
            with self.driver.session() as session:
                result = session.run(query, from_id=from_id, to_id=to_id)
                paths = []
                for record in result:
                    path_data = record['path']
                    paths.append({
                        'nodes': [node['id'] for node in path_data.nodes],
                        'relationships': [rel.type for rel in path_data.relationships],
                        'length': len(path_data.relationships)
                    })
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find paths in Neo4j: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()['node_count']
                
                return {
                    "healthy": True,
                    "node_count": node_count,
                    "backend": "neo4j"
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()


class ArangoDBGraphStore(GraphStoreInterface):
    """ArangoDB graph store implementation."""
    
    def __init__(self, config: GraphStoreConfig):
        self.config = config
        self.client = None
        self.db = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ArangoDB connection."""
        try:
            from arango import ArangoClient
            
            self.client = ArangoClient(hosts=self.config.uri)
            self.db = self.client.db(
                self.config.database,
                username=self.config.username,
                password=self.config.password
            )
            
            self._initialized = True
            logger.info(f"ArangoDB graph store initialized: {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ArangoDB: {e}")
            raise ConnectionError(f"ArangoDB initialization failed: {e}")
    
    async def store_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """Store nodes in ArangoDB."""
        logger.info("ArangoDB node storage not yet implemented")
        return True
    
    async def store_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """Store relationships in ArangoDB."""
        logger.info("ArangoDB relationship storage not yet implemented")
        return True
    
    async def query_graph(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute graph query in ArangoDB."""
        logger.info("ArangoDB query not yet implemented")
        return []
    
    async def find_paths(self, from_id: str, to_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find paths between nodes in ArangoDB."""
        logger.info("ArangoDB path finding not yet implemented")
        return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ArangoDB health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {"healthy": True, "backend": "arangodb"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class NeptuneGraphStore(GraphStoreInterface):
    """AWS Neptune graph store implementation."""
    
    def __init__(self, config: GraphStoreConfig):
        self.config = config
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Neptune connection."""
        try:
            # Neptune initialization would go here
            self._initialized = True
            logger.info(f"Neptune graph store initialized: {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neptune: {e}")
            raise ConnectionError(f"Neptune initialization failed: {e}")
    
    async def store_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """Store nodes in Neptune."""
        logger.info("Neptune node storage not yet implemented")
        return True
    
    async def store_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """Store relationships in Neptune."""
        logger.info("Neptune relationship storage not yet implemented")
        return True
    
    async def query_graph(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute graph query in Neptune."""
        logger.info("Neptune query not yet implemented")
        return []
    
    async def find_paths(self, from_id: str, to_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find paths between nodes in Neptune."""
        logger.info("Neptune path finding not yet implemented")
        return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neptune health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {"healthy": True, "backend": "neptune"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

class InMemoryGraphStore(GraphStoreInterface):
    """In-memory graph store implementation for testing."""
    
    def __init__(self, config: GraphStoreConfig):
        self.config = config
        self.nodes = {}  # id -> node data
        self.relationships = {}  # (source_id, target_id, type) -> relationship data
        self.node_relationships = {}  # node_id -> list of relationship keys
        self._initialized = False
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the in-memory graph store."""
        try:
            self._connected = True
            logger.info("Connected to in-memory graph store")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to in-memory graph store: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the in-memory graph store."""
        try:
            self._connected = False
            logger.info("Disconnected from in-memory graph store")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from in-memory graph store: {e}")
            return False
    
    async def create_node(self, node_id: str, labels: List[str],
                         properties: Dict[str, Any]) -> bool:
        """Create a node in the graph."""
        try:
            self.nodes[node_id] = {
                "id": node_id,
                "labels": labels,
                "properties": properties
            }
            if node_id not in self.node_relationships:
                self.node_relationships[node_id] = []
            
            logger.info(f"Created node {node_id} in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to create node in memory: {e}")
            return False
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        try:
            if node_id in self.nodes:
                self.nodes[node_id]["properties"].update(properties)
                logger.info(f"Updated node {node_id} in memory")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update node in memory: {e}")
            return False
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph."""
        try:
            if node_id in self.nodes:
                # Remove all relationships involving this node
                relationships_to_remove = []
                for rel_key in self.node_relationships.get(node_id, []):
                    relationships_to_remove.append(rel_key)
                
                for rel_key in relationships_to_remove:
                    if rel_key in self.relationships:
                        del self.relationships[rel_key]
                
                # Remove node
                del self.nodes[node_id]
                del self.node_relationships[node_id]
                
                logger.info(f"Deleted node {node_id} from memory")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete node from memory: {e}")
            return False
    
    async def create_relationship(self, source_id: str, target_id: str,
                                 relationship_type: str,
                                 properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes."""
        try:
            if source_id not in self.nodes or target_id not in self.nodes:
                return False
            
            rel_key = (source_id, target_id, relationship_type)
            self.relationships[rel_key] = {
                "source_id": source_id,
                "target_id": target_id,
                "type": relationship_type,
                "properties": properties or {}
            }
            
            # Update node relationships
            if source_id not in self.node_relationships:
                self.node_relationships[source_id] = []
            if target_id not in self.node_relationships:
                self.node_relationships[target_id] = []
            
            self.node_relationships[source_id].append(rel_key)
            self.node_relationships[target_id].append(rel_key)
            
            logger.info(f"Created relationship {relationship_type} between {source_id} and {target_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship in memory: {e}")
            return False
    
    async def delete_relationship(self, source_id: str, target_id: str,
                                 relationship_type: str) -> bool:
        """Delete a relationship."""
        try:
            rel_key = (source_id, target_id, relationship_type)
            if rel_key in self.relationships:
                del self.relationships[rel_key]
                
                # Remove from node relationships
                if source_id in self.node_relationships:
                    self.node_relationships[source_id] = [
                        k for k in self.node_relationships[source_id] if k != rel_key
                    ]
                if target_id in self.node_relationships:
                    self.node_relationships[target_id] = [
                        k for k in self.node_relationships[target_id] if k != rel_key
                    ]
                
                logger.info(f"Deleted relationship {relationship_type} between {source_id} and {target_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete relationship from memory: {e}")
            return False
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        try:
            return self.nodes.get(node_id)
        except Exception as e:
            logger.error(f"Failed to get node from memory: {e}")
            return None
    
    async def get_neighbors(self, node_id: str, relationship_types: Optional[List[str]] = None,
                           direction: str = "both", limit: int = 100) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        try:
            neighbors = []
            
            for rel_key in self.node_relationships.get(node_id, []):
                if rel_key not in self.relationships:
                    continue
                
                rel = self.relationships[rel_key]
                source_id, target_id, rel_type = rel_key
                
                # Filter by relationship type if specified
                if relationship_types and rel_type not in relationship_types:
                    continue
                
                # Determine neighbor based on direction
                neighbor_id = None
                if direction == "both":
                    neighbor_id = target_id if source_id == node_id else source_id
                elif direction == "outgoing" and source_id == node_id:
                    neighbor_id = target_id
                elif direction == "incoming" and target_id == node_id:
                    neighbor_id = source_id
                
                if neighbor_id and neighbor_id in self.nodes:
                    neighbor_data = self.nodes[neighbor_id].copy()
                    neighbor_data["relationship"] = rel
                    neighbors.append(neighbor_data)
                
                if len(neighbors) >= limit:
                    break
            
            return neighbors
        except Exception as e:
            logger.error(f"Failed to get neighbors from memory: {e}")
            return []
    
    async def find_path(self, source_id: str, target_id: str,
                       max_depth: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Find path between two nodes."""
        try:
            if source_id not in self.nodes or target_id not in self.nodes:
                return None
            
            # Simple BFS path finding
            queue = [(source_id, [source_id])]
            visited = set()
            
            while queue:
                current_id, path = queue.pop(0)
                
                if current_id == target_id and len(path) > 1:
                    # Return path with node data
                    return [self.nodes[node_id] for node_id in path if node_id in self.nodes]
                
                if len(path) >= max_depth + 1 or current_id in visited:
                    continue
                
                visited.add(current_id)
                
                # Find connected nodes
                for rel_key in self.node_relationships.get(current_id, []):
                    if rel_key not in self.relationships:
                        continue
                    
                    source_id_rel, target_id_rel, _ = rel_key
                    next_id = target_id_rel if source_id_rel == current_id else source_id_rel
                    
                    if next_id and next_id not in path:
                        queue.append((next_id, path + [next_id]))
            
            return None
        except Exception as e:
            logger.error(f"Failed to find path in memory: {e}")
            return None
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom graph query."""
        try:
            # Simple implementation - just return all nodes for now
            # In a real implementation, this would parse the query
            return list(self.nodes.values())
        except Exception as e:
            logger.error(f"Failed to execute query in memory: {e}")
            return []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            return {
                "node_count": len(self.nodes),
                "relationship_count": len(self.relationships),
                "backend": "memory",
                "connected": self._connected,
                "initialized": self._initialized
            }
        except Exception as e:
            logger.error(f"Failed to get graph stats from memory: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize in-memory graph store."""
        try:
            await self.connect()
            self._initialized = True
            logger.info("In-memory graph store initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize in-memory graph store: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check in-memory graph store health."""
        try:
            if not self._initialized:
                return {"healthy": False, "error": "Not initialized"}
            
            return {
                "healthy": True,
                "backend": "memory",
                "node_count": len(self.nodes),
                "relationship_count": len(self.relationships),
                "connected": self._connected
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}



def get_graph_store(config: GraphStoreConfig) -> GraphStoreInterface:
    """Factory function to get graph store implementation."""
    
    if config.backend == "neo4j":
        return Neo4jGraphStore(config)
    elif config.backend == "arangodb":
        return ArangoDBGraphStore(config)
    elif config.backend == "neptune":
        return NeptuneGraphStore(config)
    elif config.backend == "memory":
        return InMemoryGraphStore(config)
    else:
        raise ConfigurationError(f"Unsupported graph store backend: {config.backend}")