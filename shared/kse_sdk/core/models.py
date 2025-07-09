"""
Core Universal Models for KSE Memory SDK
Universal AI memory system for any domain - healthcare, finance, enterprise, research, retail, and more.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import warnings


class SearchType(Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    HYBRID = "hybrid"
    KNOWLEDGE_GRAPH = "knowledge_graph"


# Domain-specific conceptual dimension templates
DOMAIN_DIMENSIONS = {
    "healthcare": {
        "urgency": "How urgent or time-sensitive the entity is",
        "complexity": "Technical or procedural complexity level",
        "invasiveness": "How invasive or disruptive the entity is",
        "cost_effectiveness": "Economic efficiency and value",
        "safety": "Safety profile and risk level",
        "accessibility": "How accessible or available the entity is"
    },
    "finance": {
        "risk": "Risk level and volatility",
        "liquidity": "How easily convertible to cash",
        "growth_potential": "Expected growth and returns",
        "stability": "Consistency and reliability",
        "complexity": "Structural and operational complexity",
        "regulatory_compliance": "Adherence to regulations"
    },
    "real_estate": {
        "location_quality": "Desirability and convenience of location",
        "condition": "Physical state and maintenance level",
        "investment_potential": "Expected appreciation and returns",
        "size_efficiency": "Space utilization and layout quality",
        "amenities": "Available features and facilities",
        "market_demand": "Current and projected demand"
    },
    "enterprise": {
        "importance": "Strategic importance and priority",
        "complexity": "Technical or operational complexity",
        "urgency": "Time sensitivity and deadlines",
        "impact": "Potential business impact",
        "resource_intensity": "Required resources and effort",
        "stakeholder_value": "Value to stakeholders"
    },
    "research": {
        "novelty": "Originality and innovation level",
        "impact": "Potential scientific or practical impact",
        "rigor": "Methodological quality and reliability",
        "accessibility": "Ease of understanding and application",
        "reproducibility": "Ability to replicate results",
        "interdisciplinary": "Cross-domain applicability"
    },
    "retail": {
        "elegance": "Aesthetic appeal and sophistication",
        "comfort": "User comfort and ease of use",
        "boldness": "Distinctive and attention-grabbing qualities",
        "modernity": "Contemporary style and innovation",
        "minimalism": "Simplicity and clean design",
        "luxury": "Premium quality and exclusivity",
        "functionality": "Practical utility and performance",
        "versatility": "Adaptability to different uses",
        "seasonality": "Time-specific relevance",
        "innovation": "Technological or design innovation"
    },
    "marketing": {
        "engagement": "Ability to capture and hold attention",
        "conversion_potential": "Likelihood to drive desired actions",
        "brand_alignment": "Consistency with brand identity",
        "emotional_impact": "Emotional resonance and connection",
        "viral_potential": "Shareability and word-of-mouth appeal",
        "targeting_precision": "Accuracy in reaching intended audience",
        "cost_efficiency": "Return on investment and cost-effectiveness",
        "timing_relevance": "Appropriateness for current market timing",
        "competitive_advantage": "Differentiation from competitors",
        "measurability": "Ease of tracking and measuring success"
    }
}


@dataclass
class ConceptualSpace:
    """
    Flexible conceptual space for any domain.
    
    Replaces hardcoded retail-specific ConceptualDimensions with a dynamic system
    that can adapt to any industry or use case.
    """
    dimensions: Dict[str, float] = field(default_factory=dict)
    domain: Optional[str] = None
    dimension_descriptions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with domain-specific dimensions if domain is specified."""
        if self.domain and self.domain in DOMAIN_DIMENSIONS and not self.dimensions:
            domain_dims = DOMAIN_DIMENSIONS[self.domain]
            self.dimensions = {dim: 0.0 for dim in domain_dims.keys()}
            self.dimension_descriptions = domain_dims.copy()
    
    @classmethod
    def create_for_domain(cls, domain: str, custom_dimensions: Optional[Dict[str, str]] = None) -> "ConceptualSpace":
        """
        Create a conceptual space for a specific domain.
        
        Args:
            domain: Domain name (healthcare, finance, retail, etc.)
            custom_dimensions: Optional custom dimensions to override defaults
            
        Returns:
            ConceptualSpace configured for the domain
        """
        if domain in DOMAIN_DIMENSIONS:
            dimensions_def = DOMAIN_DIMENSIONS[domain].copy()
            if custom_dimensions:
                dimensions_def.update(custom_dimensions)
        else:
            dimensions_def = custom_dimensions or {}
        
        return cls(
            dimensions={dim: 0.0 for dim in dimensions_def.keys()},
            domain=domain,
            dimension_descriptions=dimensions_def
        )
    
    @classmethod
    def create_custom(cls, dimensions: Dict[str, str], domain: Optional[str] = None) -> "ConceptualSpace":
        """
        Create a custom conceptual space with arbitrary dimensions.
        
        Args:
            dimensions: Dict mapping dimension names to descriptions
            domain: Optional domain identifier
            
        Returns:
            ConceptualSpace with custom dimensions
        """
        return cls(
            dimensions={dim: 0.0 for dim in dimensions.keys()},
            domain=domain,
            dimension_descriptions=dimensions
        )
    
    @classmethod
    def create_healthcare_space(cls) -> "ConceptualSpace":
        """Create a healthcare conceptual space."""
        return cls.create_for_domain("healthcare")
    
    @classmethod
    def create_finance_space(cls) -> "ConceptualSpace":
        """Create a finance conceptual space."""
        return cls.create_for_domain("finance")
    
    @classmethod
    def create_real_estate_space(cls) -> "ConceptualSpace":
        """Create a real estate conceptual space."""
        return cls.create_for_domain("real_estate")
    
    @classmethod
    def create_enterprise_space(cls) -> "ConceptualSpace":
        """Create an enterprise conceptual space."""
        return cls.create_for_domain("enterprise")
    
    @classmethod
    def create_research_space(cls) -> "ConceptualSpace":
        """Create a research conceptual space."""
        return cls.create_for_domain("research")
    
    @classmethod
    def create_retail_space(cls) -> "ConceptualSpace":
        """Create a retail conceptual space."""
        return cls.create_for_domain("retail")
    
    @classmethod
    def create_marketing_space(cls) -> "ConceptualSpace":
        """Create a marketing conceptual space."""
        return cls.create_for_domain("marketing")
    
    def set_dimension(self, dimension: str, value: float, description: Optional[str] = None):
        """Set a dimension value and optionally its description."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Dimension value must be between 0.0 and 1.0, got {value}")
        
        self.dimensions[dimension] = value
        if description:
            self.dimension_descriptions[dimension] = description
    
    def get_dimension(self, dimension: str) -> Optional[float]:
        """Get a dimension value."""
        return self.dimensions.get(dimension)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimensions": self.dimensions,
            "domain": self.domain,
            "dimension_descriptions": self.dimension_descriptions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptualSpace":
        """Create from dictionary representation."""
        return cls(
            dimensions=data.get("dimensions", {}),
            domain=data.get("domain"),
            dimension_descriptions=data.get("dimension_descriptions", {})
        )


@dataclass
class EmbeddingVector:
    """Represents an embedding vector with metadata."""
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate vector dimensions."""
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimension {self.dimension}")


@dataclass
class KnowledgeGraph:
    """Represents knowledge graph relationships."""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_node(self, node_id: str, properties: Dict[str, Any]):
        """Add a node to the knowledge graph."""
        self.nodes[node_id] = properties
    
    def add_edge(self, source: str, target: str, relationship: str, properties: Optional[Dict[str, Any]] = None):
        """Add an edge to the knowledge graph."""
        edge = {
            "source": source,
            "target": target,
            "relationship": relationship,
            "properties": properties or {}
        }
        self.edges.append(edge)


@dataclass
class Entity:
    """
    Universal entity class that can represent any domain object.
    
    Replaces the hardcoded Product class with a flexible system that supports
    healthcare records, financial instruments, real estate properties, 
    enterprise resources, research papers, retail products, and more.
    """
    id: str
    title: str
    description: str
    domain: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # AI-powered components
    text_embedding: Optional[EmbeddingVector] = None
    image_embedding: Optional[EmbeddingVector] = None
    conceptual_space: Optional[ConceptualSpace] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    source: Optional[str] = None
    external_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize domain-specific conceptual space if not provided."""
        if not self.conceptual_space and self.domain:
            self.conceptual_space = ConceptualSpace.create_for_domain(self.domain)
    
    @classmethod
    def create_healthcare_entity(cls, id: str, title: str, description: str,
                                entity_type: str = "medical_record", **kwargs) -> "Entity":
        """Create a healthcare entity (patient record, treatment, medication, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="healthcare",
            entity_type=entity_type,
            properties=kwargs
        )
    
    @classmethod
    def create_finance_entity(cls, id: str, title: str, description: str,
                             entity_type: str = "financial_instrument", **kwargs) -> "Entity":
        """Create a finance entity (stock, bond, portfolio, transaction, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="finance",
            entity_type=entity_type,
            properties=kwargs
        )
    
    @classmethod
    def create_real_estate_entity(cls, id: str, title: str, description: str,
                                 entity_type: str = "property", **kwargs) -> "Entity":
        """Create a real estate entity (property, listing, market data, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="real_estate",
            entity_type=entity_type,
            **kwargs
        )
    
    @classmethod
    def create_enterprise_entity(cls, id: str, title: str, description: str,
                                entity_type: str = "resource", **kwargs) -> "Entity":
        """Create an enterprise entity (project, task, document, employee, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="enterprise",
            entity_type=entity_type,
            **kwargs
        )
    
    @classmethod
    def create_research_entity(cls, id: str, title: str, description: str,
                              entity_type: str = "paper", **kwargs) -> "Entity":
        """Create a research entity (paper, dataset, experiment, finding, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="research",
            entity_type=entity_type,
            **kwargs
        )
    
    @classmethod
    def create_retail_entity(cls, id: str, title: str, description: str,
                            entity_type: str = "product", **kwargs) -> "Entity":
        """Create a retail entity (product, inventory, customer, order, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="retail",
            entity_type=entity_type,
            **kwargs
        )
    
    @classmethod
    def create_marketing_entity(cls, id: str, title: str, description: str,
                               entity_type: str = "campaign", **kwargs) -> "Entity":
        """Create a marketing entity (campaign, ad, audience, creative, etc.)."""
        return cls(
            id=id,
            title=title,
            description=description,
            domain="marketing",
            entity_type=entity_type,
            **kwargs
        )
    
    def set_conceptual_dimension(self, dimension: str, value: float):
        """Set a conceptual dimension value."""
        if not self.conceptual_space:
            self.conceptual_space = ConceptualSpace.create_for_domain(self.domain)
        self.conceptual_space.set_dimension(dimension, value)
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def get_conceptual_dimension(self, dimension: str) -> Optional[float]:
        """Get a conceptual dimension value."""
        if self.conceptual_space:
            return self.conceptual_space.get_dimension(dimension)
        return None
    
    def add_property(self, key: str, value: Any):
        """Add a custom property to the entity."""
        self.properties[key] = value
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a custom property value."""
        return self.properties.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "tags": self.tags,
            "categories": self.categories,
            "text_embedding": self.text_embedding.__dict__ if self.text_embedding else None,
            "image_embedding": self.image_embedding.__dict__ if self.image_embedding else None,
            "conceptual_space": self.conceptual_space.to_dict() if self.conceptual_space else None,
            "knowledge_graph": self.knowledge_graph.__dict__ if self.knowledge_graph else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "source": self.source,
            "external_id": self.external_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary representation."""
        # Handle datetime fields
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow()
        updated_at = datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow()
        
        # Handle embedding fields
        text_embedding = None
        if data.get("text_embedding"):
            text_embedding = EmbeddingVector(**data["text_embedding"])
        
        image_embedding = None
        if data.get("image_embedding"):
            image_embedding = EmbeddingVector(**data["image_embedding"])
        
        # Handle conceptual space
        conceptual_space = None
        if data.get("conceptual_space"):
            conceptual_space = ConceptualSpace.from_dict(data["conceptual_space"])
        
        # Handle knowledge graph
        knowledge_graph = None
        if data.get("knowledge_graph"):
            knowledge_graph = KnowledgeGraph(**data["knowledge_graph"])
        
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            domain=data["domain"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            conceptual_space=conceptual_space,
            knowledge_graph=knowledge_graph,
            created_at=created_at,
            updated_at=updated_at,
            version=data.get("version", 1),
            source=data.get("source"),
            external_id=data.get("external_id")
        )


@dataclass
class SearchQuery:
    """Universal search query for any domain."""
    query: str
    domain: Optional[str] = None
    entity_types: List[str] = field(default_factory=list)
    search_type: SearchType = SearchType.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    conceptual_filters: Dict[str, float] = field(default_factory=dict)
    limit: int = 10
    threshold: float = 0.7
    include_embeddings: bool = False
    include_conceptual: bool = True
    include_knowledge_graph: bool = True


@dataclass
class SearchResult:
    """Universal search result for any domain."""
    entity: Entity
    score: float
    search_type: SearchType
    explanation: Optional[str] = None
    conceptual_similarity: Optional[float] = None
    semantic_similarity: Optional[float] = None
    graph_relevance: Optional[float] = None


# Backward compatibility - deprecated retail-specific class
@dataclass
class ConceptualDimensions:
    """
    DEPRECATED: Use ConceptualSpace instead.
    
    Legacy retail-specific conceptual dimensions. This class is maintained for
    backward compatibility but will be removed in v3.0.0.
    """
    elegance: float = 0.0
    comfort: float = 0.0
    boldness: float = 0.0
    modernity: float = 0.0
    minimalism: float = 0.0
    luxury: float = 0.0
    functionality: float = 0.0
    versatility: float = 0.0
    seasonality: float = 0.0
    innovation: float = 0.0
    
    def __post_init__(self):
        """Issue deprecation warning."""
        warnings.warn(
            "ConceptualDimensions is deprecated and will be removed in v3.0.0. "
            "Use ConceptualSpace.create_for_domain('retail') instead.",
            DeprecationWarning,
            stacklevel=2
        )


# Backward compatibility - deprecated Product class
@dataclass
class Product:
    """
    DEPRECATED: Use Entity.create_retail_entity() instead.
    
    Legacy product class maintained for backward compatibility.
    """
    id: str
    title: str
    description: str
    price: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Issue deprecation warning."""
        warnings.warn(
            "Product class is deprecated and will be removed in v3.0.0. "
            "Use Entity.create_retail_entity() instead.",
            DeprecationWarning,
            stacklevel=2
        )