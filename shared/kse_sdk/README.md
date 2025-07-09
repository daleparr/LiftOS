# KSE Memory SDK - Universal AI Memory System

## Overview

The KSE Memory SDK is a universal artificial intelligence memory system that serves as the foundational substrate for LiftOS. It provides a comprehensive, domain-agnostic framework for storing, analyzing, and retrieving information across multiple domains including healthcare, finance, real estate, enterprise, research, retail, and marketing.

## Architecture

### Three-Pillar Hybrid Search System

The KSE-SDK implements a sophisticated hybrid search architecture combining:

1. **Neural Embeddings** - Semantic similarity through vector embeddings
2. **Conceptual Spaces** - Domain-specific dimensional analysis
3. **Knowledge Graphs** - Relationship-based traversal and reasoning

### Universal Entity System

Unlike traditional domain-specific systems, the KSE-SDK uses a universal `Entity` model that can represent any type of information:

```python
from shared.kse_sdk.core.models import Entity

# Healthcare entity
patient = Entity(
    id="patient_001",
    name="John Doe - Diabetes Case",
    description="Type 2 diabetes with hypertension",
    domain="healthcare",
    entity_type="patient_case",
    attributes={
        "age": 45,
        "diagnosis": "Type 2 Diabetes",
        "severity": "moderate"
    },
    tags=["diabetes", "chronic", "metabolic"]
)

# Financial entity
investment = Entity(
    id="investment_001", 
    name="Tesla Inc. Analysis",
    description="Growth stock analysis",
    domain="finance",
    entity_type="investment_opportunity",
    attributes={
        "symbol": "TSLA",
        "risk_level": "high",
        "market_cap": 800000000000
    },
    tags=["growth", "technology", "automotive"]
)
```

## Supported Domains

### 1. Healthcare
- **Entities**: Patients, treatments, diagnoses, medical devices
- **Dimensions**: Severity, urgency, treatment complexity, patient impact
- **Use Cases**: Clinical decision support, patient matching, treatment recommendations

### 2. Finance
- **Entities**: Investments, portfolios, transactions, market analysis
- **Dimensions**: Risk level, liquidity, return potential, market volatility
- **Use Cases**: Investment analysis, risk assessment, portfolio optimization

### 3. Real Estate
- **Entities**: Properties, market analysis, valuations, transactions
- **Dimensions**: Price range, location desirability, property condition, investment potential
- **Use Cases**: Property valuation, market analysis, investment opportunities

### 4. Enterprise
- **Entities**: Projects, processes, resources, strategies
- **Dimensions**: Business impact, strategic importance, resource requirements, timeline urgency
- **Use Cases**: Project management, resource allocation, strategic planning

### 5. Research
- **Entities**: Papers, datasets, experiments, methodologies
- **Dimensions**: Novelty, methodological rigor, reproducibility, impact potential
- **Use Cases**: Literature review, research discovery, collaboration matching

### 6. Retail
- **Entities**: Products, customers, transactions, inventory
- **Dimensions**: Price range, quality tier, brand prestige, market demand
- **Use Cases**: Product recommendations, inventory optimization, customer analysis

### 7. Marketing
- **Entities**: Campaigns, audiences, content, channels
- **Dimensions**: Engagement potential, reach scope, conversion likelihood, brand alignment
- **Use Cases**: Campaign optimization, audience targeting, content strategy

## Quick Start

### Basic Usage

```python
import asyncio
from shared.kse_sdk.core.config import KSEConfig, VectorStoreConfig
from shared.kse_sdk.core.memory import KSEMemory
from shared.kse_sdk.core.models import Entity, SearchQuery

async def main():
    # Configure KSE Memory
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='pinecone',
            api_key='your-api-key',
            environment='us-west1-gcp',
            index_name='universal-entities'
        ),
        embedding_config={
            'provider': 'openai',
            'model': 'text-embedding-ada-002',
            'api_key': 'your-openai-key'
        }
    )
    
    # Initialize KSE Memory
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create and store an entity
    entity = Entity(
        id="example_001",
        name="Example Entity",
        description="This is an example entity",
        domain="enterprise",
        entity_type="example"
    )
    
    await kse_memory.store_entity(entity)
    
    # Search for entities
    query = SearchQuery(
        text="example entity",
        domain="enterprise",
        limit=10
    )
    
    results = await kse_memory.search(query)
    print(f"Found {len(results.entities)} entities")

asyncio.run(main())
```

### Domain-Specific Usage

```python
# Healthcare example
patient = await kse_memory.create_healthcare_entity(
    name="Patient Case Study",
    description="Complex diabetes case",
    entity_type="patient_case",
    severity="high",
    urgency="medium"
)

# Finance example  
investment = await kse_memory.create_finance_entity(
    name="Investment Analysis",
    description="Tech stock evaluation",
    entity_type="investment_opportunity",
    risk_level="medium",
    liquidity="high"
)

# Real estate example
property_entity = await kse_memory.create_real_estate_entity(
    name="Downtown Condo",
    description="Luxury 2BR condo",
    entity_type="residential_property",
    price_range="high",
    location_desirability="excellent"
)
```

## Configuration

### Vector Store Backends

The KSE-SDK supports multiple vector store backends:

```python
# Pinecone
vector_config = VectorStoreConfig(
    provider='pinecone',
    api_key='your-pinecone-key',
    environment='us-west1-gcp',
    index_name='entities'
)

# Weaviate
vector_config = VectorStoreConfig(
    provider='weaviate',
    url='http://localhost:8080',
    class_name='Entity'
)

# Qdrant
vector_config = VectorStoreConfig(
    provider='qdrant',
    url='localhost:6333',
    collection_name='entities'
)

# Chroma
vector_config = VectorStoreConfig(
    provider='chroma',
    host='localhost',
    port=8000,
    collection_name='entities'
)

# Milvus
vector_config = VectorStoreConfig(
    provider='milvus',
    host='localhost',
    port=19530,
    collection_name='entities'
)
```

### Graph Store Backends

```python
# Neo4j
graph_config = GraphStoreConfig(
    provider='neo4j',
    uri='bolt://localhost:7687',
    username='neo4j',
    password='password'
)

# ArangoDB
graph_config = GraphStoreConfig(
    provider='arangodb',
    url='http://localhost:8529',
    database='kse_graph',
    username='root',
    password='password'
)

# Amazon Neptune
graph_config = GraphStoreConfig(
    provider='neptune',
    endpoint='your-neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com',
    port=8182
)
```

### Concept Store Backends

```python
# PostgreSQL
concept_config = ConceptStoreConfig(
    provider='postgresql',
    host='localhost',
    port=5432,
    database='kse_concepts',
    username='postgres',
    password='password'
)

# MongoDB
concept_config = ConceptStoreConfig(
    provider='mongodb',
    host='localhost',
    port=27017,
    database='kse_concepts'
)

# Elasticsearch
concept_config = ConceptStoreConfig(
    provider='elasticsearch',
    host='localhost',
    port=9200,
    index='kse_concepts'
)
```

## Migration from Legacy LiftOS

The KSE-SDK includes comprehensive migration utilities to transition from legacy LiftOS data:

```python
from shared.kse_sdk.migration import KSEMigrator, create_migration_plan

# Create migration plan
legacy_data = {
    'products': [...],  # Legacy product data
    'conceptual_dimensions': {...},  # Legacy dimensional data
    'memory_data': {...}  # Legacy memory service data
}

migration_plan = await create_migration_plan(legacy_data, target_config)
print(f"Migration will take approximately {migration_plan['estimated_duration']} seconds")

# Execute migration
migrator = KSEMigrator(target_config)
await migrator.initialize()

# Migrate products to universal entities
migrated_entities = await migrator.migrate_products_to_entities(legacy_data['products'])

# Migrate conceptual dimensions
migrated_spaces = await migrator.migrate_conceptual_dimensions(legacy_data['conceptual_dimensions'])

# Validate migration
validation_report = await migrator.validate_migration()
print(f"Migration success rate: {validation_report['success_rate']:.1f}%")
```

## Advanced Features

### Hybrid Search

```python
# Neural + Conceptual + Graph search
query = SearchQuery(
    text="innovative AI solutions",
    search_types=['neural', 'conceptual', 'graph'],
    domain="healthcare",
    limit=20
)

results = await kse_memory.search(query)
```

### Cross-Domain Analysis

```python
# Search across all domains
query = SearchQuery(
    text="artificial intelligence innovation",
    # No domain filter - searches all domains
    limit=50
)

results = await kse_memory.search(query)

# Analyze results by domain
domain_distribution = {}
for entity in results.entities:
    domain = entity.domain
    domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
```

### Conceptual Analysis

```python
from shared.kse_sdk.services.conceptual_service import ConceptualService

conceptual_service = ConceptualService(config)

# Analyze entity conceptually
conceptual_space = await conceptual_service.analyze_entity(entity)
print(f"Conceptual coordinates: {conceptual_space.coordinates}")

# Compare entities
similarities = await conceptual_service.compare_entities(entity1, entity2)
print(f"Dimensional similarities: {similarities}")

# Find conceptual neighbors
neighbors = await conceptual_service.find_conceptual_neighbors(
    entity, candidate_entities, threshold=0.8
)
```

### Relationship Analysis

```python
# Find related entities through graph relationships
related_entities = await kse_memory.search_by_relationships(
    entity_id="patient_001",
    relationship_types=["treated_with", "diagnosed_with"],
    max_depth=2,
    limit=10
)
```

## Backward Compatibility

The KSE-SDK maintains full backward compatibility with existing LiftOS code:

```python
# Legacy Product creation still works
product = await kse_memory.create_product(
    name="Legacy Product",
    description="This uses the old API",
    price=99.99,
    category="Electronics"
)

# Legacy ConceptualDimensions still work
from shared.kse_sdk.core.models import ConceptualDimensions
legacy_dims = ConceptualDimensions(
    price_range=0.8,
    quality_tier=0.9,
    brand_prestige=0.7,
    market_demand=0.6
)
```

## Performance Optimization

### Batch Operations

```python
# Store multiple entities efficiently
entities = [entity1, entity2, entity3, ...]
await kse_memory.store_entities_batch(entities)

# Batch search
queries = [query1, query2, query3, ...]
results = await kse_memory.search_batch(queries)
```

### Caching

```python
# Enable caching for better performance
config = KSEConfig(
    cache_config={
        'enabled': True,
        'ttl_seconds': 3600,
        'max_size': 10000
    }
)
```

### Async Operations

All KSE-SDK operations are fully asynchronous for optimal performance:

```python
# Concurrent operations
tasks = [
    kse_memory.store_entity(entity1),
    kse_memory.store_entity(entity2),
    kse_memory.search(query1),
    kse_memory.search(query2)
]

results = await asyncio.gather(*tasks)
```

## Error Handling

```python
from shared.kse_sdk.exceptions import (
    KSEError, ConfigurationError, ConnectionError, 
    SearchError, ValidationError
)

try:
    await kse_memory.store_entity(entity)
except ValidationError as e:
    print(f"Entity validation failed: {e}")
except ConnectionError as e:
    print(f"Storage backend connection failed: {e}")
except KSEError as e:
    print(f"General KSE error: {e}")
```

## Testing

```python
# Run the comprehensive example
python shared/kse_sdk/examples/universal_usage_example.py

# Test specific domains
from shared.kse_sdk.examples.universal_usage_example import healthcare_example
await healthcare_example()
```

## Architecture Benefits

### 1. Universal Substrate
- Single API for all domains
- Consistent data model across use cases
- Unified search and analysis capabilities

### 2. Scalable Design
- Pluggable backend adapters
- Horizontal scaling support
- Efficient batch operations

### 3. Domain Expertise
- Domain-specific conceptual dimensions
- Specialized entity factories
- Contextual embedding strategies

### 4. Hybrid Intelligence
- Neural semantic understanding
- Conceptual dimensional analysis
- Graph relationship reasoning

### 5. Migration Support
- Seamless transition from legacy systems
- Backward compatibility preservation
- Comprehensive validation tools

## Integration with LiftOS

The KSE-SDK serves as the universal intelligence substrate for all LiftOS services:

- **Memory Service**: Uses KSE-SDK as the core memory system
- **Search Service**: Leverages hybrid search capabilities
- **Analytics Service**: Builds on conceptual analysis
- **Data Ingestion**: Stores all ingested data as universal entities
- **Streamlit Interface**: Provides domain-specific views of universal data

## Future Roadmap

- [ ] Graph store adapter implementations
- [ ] Concept store adapter implementations
- [ ] Advanced analytics service
- [ ] Security and access control service
- [ ] Workflow orchestration service
- [ ] Real-time notification service
- [ ] Multi-modal entity support (images, audio, video)
- [ ] Federated learning capabilities
- [ ] Edge deployment support

## Contributing

The KSE-SDK is designed for extensibility. To add new domains:

1. Add domain dimensions to `DOMAIN_DIMENSIONS` in `core/models.py`
2. Implement domain-specific entity factory methods in `core/memory.py`
3. Add domain-specific coordinate generation in `services/conceptual_service.py`
4. Create domain-specific examples in `examples/`

## License

This KSE Memory SDK is part of the LiftOS project and follows the same licensing terms.