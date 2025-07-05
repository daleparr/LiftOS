"""
Causal-aware KSE Memory SDK Models
Enhanced models for causal relationship embeddings and temporal causal analysis
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel
from enum import Enum


class CausalRelationType(str, Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    MODERATOR = "moderator"
    COLLIDER = "collider"
    SPURIOUS = "spurious"


class TemporalDirection(str, Enum):
    """Temporal direction of causal relationships"""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"
    SIMULTANEOUS = "simultaneous"


class CausalRelationship(BaseModel):
    """Represents a causal relationship between variables"""
    cause_variable: str
    effect_variable: str
    relationship_type: CausalRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    temporal_direction: TemporalDirection
    time_lag: Optional[int] = None  # in days
    context: Dict[str, Any] = {}
    evidence: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalEmbedding(BaseModel):
    """Causal-aware embedding with temporal and relationship context"""
    id: str
    content: str
    embedding_vector: List[float]
    causal_relationships: List[CausalRelationship]
    temporal_context: Dict[str, Any]
    causal_metadata: Dict[str, Any]
    organization_id: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TemporalCausalEmbedding(BaseModel):
    """Time-aware causal embedding for temporal analysis"""
    id: str
    content: str
    embedding_vector: List[float]
    timestamp: datetime
    time_window: Dict[str, datetime]  # start_time, end_time
    causal_context: Dict[str, Any]
    temporal_features: Dict[str, float]
    lag_relationships: List[Dict[str, Any]]
    organization_id: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalKnowledgeNode(BaseModel):
    """Node in causal knowledge graph"""
    id: str
    variable_name: str
    variable_type: str  # campaign, ad_set, creative, audience, etc.
    platform: str  # meta, google, klaviyo
    description: str
    properties: Dict[str, Any]
    embedding_vector: List[float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalKnowledgeEdge(BaseModel):
    """Edge in causal knowledge graph"""
    id: str
    source_node_id: str
    target_node_id: str
    relationship: CausalRelationship
    weight: float
    evidence_strength: float
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalKnowledgeGraph(BaseModel):
    """Complete causal knowledge graph"""
    id: str
    organization_id: str
    nodes: List[CausalKnowledgeNode]
    edges: List[CausalKnowledgeEdge]
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalSearchQuery(BaseModel):
    """Search query for causal relationships"""
    query: str
    organization_id: str
    search_type: str = "causal_hybrid"  # causal_neural, causal_temporal, causal_graph, causal_hybrid
    causal_filters: Optional[Dict[str, Any]] = None
    temporal_range: Optional[Dict[str, datetime]] = None
    relationship_types: Optional[List[CausalRelationType]] = None
    minimum_strength: float = 0.5
    minimum_confidence: float = 0.7
    limit: int = 10
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalSearchResult(BaseModel):
    """Result from causal-aware search"""
    id: str
    content: str
    causal_score: float
    temporal_score: float
    relationship_score: float
    combined_score: float
    causal_relationships: List[CausalRelationship]
    temporal_context: Dict[str, Any]
    metadata: Dict[str, Any]
    organization_id: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalInsights(BaseModel):
    """Causal insights from memory analysis"""
    organization_id: str
    total_causal_memories: int
    dominant_causal_patterns: List[Dict[str, Any]]
    temporal_causal_trends: Dict[str, Any]
    relationship_strength_distribution: Dict[str, int]
    causal_clusters: List[Dict[str, Any]]
    knowledge_graph_stats: Dict[str, Any]
    causal_anomalies: List[Dict[str, Any]]
    generated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalMemoryEntry(BaseModel):
    """Causal memory entry for storage"""
    content: str
    memory_type: str = "causal"
    organization_id: str
    causal_relationships: List[CausalRelationship]
    temporal_context: Dict[str, Any]
    causal_metadata: Dict[str, Any]
    platform_context: Optional[str] = None
    experiment_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalConceptSpace(BaseModel):
    """Causal concept space for semantic understanding"""
    id: str
    organization_id: str
    concept_name: str
    causal_dimensions: List[str]
    embedding_space: List[List[float]]
    relationship_mappings: Dict[str, List[str]]
    temporal_evolution: Dict[str, Any]
    confidence_scores: Dict[str, float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CausalSemanticCluster(BaseModel):
    """Cluster of causally related concepts"""
    id: str
    organization_id: str
    cluster_name: str
    member_concepts: List[str]
    causal_coherence_score: float
    temporal_stability: float
    dominant_relationships: List[CausalRelationType]
    cluster_centroid: List[float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }