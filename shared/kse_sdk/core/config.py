"""
KSE Memory SDK Configuration
"""

from typing import Dict, Any, Optional, Literal, List
from dataclasses import dataclass, field
from enum import Enum
import os


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingModel(Enum):
    """Supported embedding models"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    backend: str = "pinecone"
    dimension: int = 1536
    metric: str = "cosine"
    index_name: str = "kse-memory"
    api_key: Optional[str] = None
    environment: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: Optional[str] = None


@dataclass
class GraphStoreConfig:
    """Graph store configuration"""
    backend: str = "neo4j"
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: Optional[str] = None
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50


@dataclass
class ConceptStoreConfig:
    """Concept store configuration"""
    backend: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "kse_concepts"
    username: str = "postgres"
    password: Optional[str] = None
    table_name: str = "concepts"
    max_connections: int = 20


@dataclass
class EmbeddingConfig:
    """Embedding service configuration"""
    default_model: EmbeddingModel = EmbeddingModel.OPENAI
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    huggingface_api_key: Optional[str] = field(default_factory=lambda: os.getenv('HUGGINGFACE_API_KEY'))
    cohere_api_key: Optional[str] = field(default_factory=lambda: os.getenv('COHERE_API_KEY'))
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30


@dataclass
class ConceptualConfig:
    """Conceptual service configuration"""
    default_domain: str = "enterprise"
    similarity_threshold: float = 0.7
    max_concepts_per_space: int = 1000
    enable_dimensional_reasoning: bool = True
    cache_conceptual_spaces: bool = True


@dataclass
class SearchConfig:
    """Search service configuration"""
    default_search_type: str = "hybrid"
    max_results: int = 100
    similarity_threshold: float = 0.7
    max_history_size: int = 1000
    search_weights: Dict[str, float] = field(default_factory=lambda: {
        'neural': 0.4,
        'conceptual': 0.3,
        'graph': 0.3
    })


@dataclass
class KSEConfig:
    """
    Main configuration class for KSE Memory SDK.
    """
    
    # Application settings
    app_name: str = "liftos"
    environment: str = "development"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    version: str = "2.1.0"
    default_domain: str = "general"
    supported_domains: List[str] = field(default_factory=lambda: [
        "healthcare", "finance", "real_estate", "enterprise",
        "research", "retail", "marketing", "general"
    ])
    
    # Store type selections
    vector_store_type: str = "pinecone"
    graph_store_type: str = "neo4j"
    concept_store_type: str = "postgresql"
    embedding_model: str = "openai"
    
    # Store configurations
    vector_store_config: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph_store_config: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    concept_store_config: ConceptStoreConfig = field(default_factory=ConceptStoreConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    conceptual_config: ConceptualConfig = field(default_factory=ConceptualConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    
    # Service settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    analytics_enabled: bool = True
    security_enabled: bool = True
    workflow_enabled: bool = True
    notification_enabled: bool = True
    enable_metrics: bool = True
    enable_auth: bool = False
    
    # Performance settings
    max_concurrent_operations: int = 10
    operation_timeout: int = 30
    retry_attempts: int = 3
    batch_size: int = 100
    
    # Legacy compatibility - keeping old structure for backward compatibility
    vector_store: Dict[str, Any] = field(default_factory=lambda: {
        'backend': 'pinecone',
        'dimension': 1536,
        'metric': 'cosine'
    })
    
    graph_store: Dict[str, Any] = field(default_factory=lambda: {
        'backend': 'neo4j',
        'uri': 'bolt://localhost:7687'
    })
    
    concept_store: Dict[str, Any] = field(default_factory=lambda: {
        'backend': 'postgresql',
        'host': 'localhost',
        'port': 5432
    })
    
    embedding_service: Dict[str, Any] = field(default_factory=lambda: {
        'default_model': 'openai',
        'models': {
            'openai': {
                'model_name': 'text-embedding-ada-002',
                'api_key': os.getenv('OPENAI_API_KEY')
            }
        }
    })
    
    # Search configuration
    search: Dict[str, Any] = field(default_factory=lambda: {
        'default_search_type': 'hybrid',
        'max_results': 100,
        'similarity_threshold': 0.7,
        'max_history_size': 1000,
        'search_weights': {
            'neural': 0.4,
            'conceptual': 0.3,
            'graph': 0.3
        }
    })
    
    # Cache configuration
    cache: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'ttl': 3600,
        'max_size': 10000
    })
    
    # Analytics configuration
    analytics: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'track_operations': True,
        'track_performance': True
    })
    
    # Security configuration
    security: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'require_auth': False,
        'audit_logging': True
    })

    def __post_init__(self):
        """Post-initialization to sync legacy and new configurations"""
        # Sync vector store config
        self.vector_store_config.backend = self.vector_store_type
        self.vector_store_config.dimension = self.vector_store.get('dimension', 1536)
        self.vector_store_config.metric = self.vector_store.get('metric', 'cosine')
        
        # Sync graph store config
        self.graph_store_config.backend = self.graph_store_type
        self.graph_store_config.uri = self.graph_store.get('uri', 'bolt://localhost:7687')
        
        # Sync concept store config
        self.concept_store_config.backend = self.concept_store_type
        self.concept_store_config.host = self.concept_store.get('host', 'localhost')
        self.concept_store_config.port = self.concept_store.get('port', 5432)
        
        # Sync embedding config
        if self.embedding_model == "openai":
            self.embedding_config.default_model = EmbeddingModel.OPENAI
        elif self.embedding_model == "huggingface":
            self.embedding_config.default_model = EmbeddingModel.HUGGINGFACE
        elif self.embedding_model == "sentence_transformers":
            self.embedding_config.default_model = EmbeddingModel.SENTENCE_TRANSFORMERS
        elif self.embedding_model == "cohere":
            self.embedding_config.default_model = EmbeddingModel.COHERE
        
        # Sync search config
        self.search_config.default_search_type = self.search.get('default_search_type', 'hybrid')
        self.search_config.max_results = self.search.get('max_results', 100)
        self.search_config.similarity_threshold = self.search.get('similarity_threshold', 0.7)
        self.search_config.max_history_size = self.search.get('max_history_size', 1000)
        self.search_config.search_weights = self.search.get('search_weights', {
            'neural': 0.4,
            'conceptual': 0.3,
            'graph': 0.3
        })

    def validate(self) -> list:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate vector store type
        valid_vector_stores = ["pinecone", "weaviate", "qdrant", "chroma", "milvus", "memory"]
        if self.vector_store_type not in valid_vector_stores:
            errors.append(f"Invalid vector_store_type: {self.vector_store_type}. Must be one of {valid_vector_stores}")
        
        # Validate graph store type
        valid_graph_stores = ["neo4j", "arangodb", "neptune", "networkx", "memory"]
        if self.graph_store_type not in valid_graph_stores:
            errors.append(f"Invalid graph_store_type: {self.graph_store_type}. Must be one of {valid_graph_stores}")
        
        # Validate concept store type
        valid_concept_stores = ["postgresql", "mongodb", "elasticsearch", "sqlite", "memory"]
        if self.concept_store_type not in valid_concept_stores:
            errors.append(f"Invalid concept_store_type: {self.concept_store_type}. Must be one of {valid_concept_stores}")
        
        # Validate embedding model
        valid_embedding_models = ["openai", "huggingface", "sentence_transformers", "cohere", "mock"]
        if self.embedding_model not in valid_embedding_models:
            errors.append(f"Invalid embedding_model: {self.embedding_model}. Must be one of {valid_embedding_models}")
        
        # Validate performance settings
        if self.max_concurrent_operations <= 0:
            errors.append("max_concurrent_operations must be greater than 0")
        
        if self.operation_timeout <= 0:
            errors.append("operation_timeout must be greater than 0")
        
        if self.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be greater than 0")
        
        if self.cache_ttl <= 0:
            errors.append("cache_ttl must be greater than 0")
        
        return errors


def get_config() -> KSEConfig:
    """Get default KSE configuration"""
    return KSEConfig()


def get_config_from_env() -> KSEConfig:
    """Get KSE configuration from environment variables"""
    config = KSEConfig()
    
    # Override with environment variables if present
    config.app_name = os.getenv('KSE_APP_NAME', config.app_name)
    config.environment = os.getenv('KSE_ENVIRONMENT', config.environment)
    config.debug = os.getenv('KSE_DEBUG', 'false').lower() == 'true'
    
    config.vector_store_type = os.getenv('KSE_VECTOR_STORE', config.vector_store_type)
    config.graph_store_type = os.getenv('KSE_GRAPH_STORE', config.graph_store_type)
    config.concept_store_type = os.getenv('KSE_CONCEPT_STORE', config.concept_store_type)
    config.embedding_model = os.getenv('KSE_EMBEDDING_MODEL', config.embedding_model)
    
    config.cache_enabled = os.getenv('KSE_CACHE_ENABLED', 'true').lower() == 'true'
    config.analytics_enabled = os.getenv('KSE_ANALYTICS_ENABLED', 'true').lower() == 'true'
    config.security_enabled = os.getenv('KSE_SECURITY_ENABLED', 'true').lower() == 'true'
    
    return config