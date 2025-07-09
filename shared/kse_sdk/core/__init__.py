"""
KSE-SDK Core Module
Universal AI Memory System for LiftOS
"""

from .memory import KSEMemory
from .models import (
    Entity, ConceptualSpace, SearchQuery, SearchResult, 
    EmbeddingVector, KnowledgeGraph, SearchType, DOMAIN_DIMENSIONS
)
from .config import (
    KSEConfig, VectorStoreConfig, GraphStoreConfig, 
    ConceptStoreConfig, EmbeddingConfig, LogLevel, EmbeddingModel
)
from .interfaces import (
    AdapterInterface, VectorStoreInterface, GraphStoreInterface,
    ConceptStoreInterface, EmbeddingServiceInterface, 
    ConceptualServiceInterface, SearchServiceInterface, CacheInterface
)

__all__ = [
    # Core classes
    'KSEMemory',
    
    # Models
    'Entity', 'ConceptualSpace', 'SearchQuery', 'SearchResult',
    'EmbeddingVector', 'KnowledgeGraph', 'SearchType', 'DOMAIN_DIMENSIONS',
    
    # Configuration
    'KSEConfig', 'VectorStoreConfig', 'GraphStoreConfig', 
    'ConceptStoreConfig', 'EmbeddingConfig', 'LogLevel', 'EmbeddingModel',
    
    # Interfaces
    'AdapterInterface', 'VectorStoreInterface', 'GraphStoreInterface',
    'ConceptStoreInterface', 'EmbeddingServiceInterface',
    'ConceptualServiceInterface', 'SearchServiceInterface', 'CacheInterface'
]

__version__ = "2.1.0"