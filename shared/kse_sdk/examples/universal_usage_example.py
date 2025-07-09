"""
KSE Memory SDK Universal Usage Example

This example demonstrates how to use the universal KSE-SDK across multiple domains
including healthcare, finance, real estate, enterprise, research, retail, and marketing.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any

# Import the universal KSE-SDK components
from ..core.config import KSEConfig, VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig
from ..core.memory import KSEMemory
from ..core.models import Entity, SearchQuery, DOMAIN_DIMENSIONS
from ..services.embedding_service import EmbeddingService
from ..services.conceptual_service import ConceptualService
from ..services.search_service import SearchService
from ..adapters.vector_stores import PineconeAdapter
from ..migration import KSEMigrator, create_migration_plan


async def healthcare_example():
    """Demonstrate healthcare domain usage."""
    print("\n=== Healthcare Domain Example ===")
    
    # Create healthcare-specific configuration
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='pinecone',
            api_key='your-pinecone-key',
            environment='us-west1-gcp',
            index_name='healthcare-entities'
        ),
        embedding_config={
            'provider': 'openai',
            'model': 'text-embedding-ada-002',
            'api_key': 'your-openai-key'
        }
    )
    
    # Initialize KSE Memory for healthcare
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create healthcare entities
    patient_entity = Entity(
        id="patient_001",
        name="John Doe - Diabetes Management",
        description="Type 2 diabetes patient with hypertension comorbidity",
        domain="healthcare",
        entity_type="patient_case",
        attributes={
            "age": 45,
            "diagnosis": "Type 2 Diabetes Mellitus",
            "comorbidities": ["Hypertension", "Obesity"],
            "severity": "moderate",
            "treatment_plan": "Metformin + lifestyle modification"
        },
        tags=["diabetes", "chronic", "metabolic"]
    )
    
    treatment_entity = Entity(
        id="treatment_001",
        name="Metformin Extended Release",
        description="First-line treatment for type 2 diabetes",
        domain="healthcare",
        entity_type="treatment",
        attributes={
            "drug_class": "Biguanide",
            "dosage": "500mg twice daily",
            "side_effects": ["GI upset", "Lactic acidosis (rare)"],
            "contraindications": ["Kidney disease", "Heart failure"]
        },
        tags=["medication", "diabetes", "oral"]
    )
    
    # Store healthcare entities
    await kse_memory.store_entity(patient_entity)
    await kse_memory.store_entity(treatment_entity)
    
    # Search for similar cases
    search_query = SearchQuery(
        text="diabetes patient with complications",
        domain="healthcare",
        entity_filters={"entity_type": "patient_case"},
        limit=5
    )
    
    results = await kse_memory.search(search_query)
    print(f"Found {len(results.entities)} similar healthcare cases")
    
    # Demonstrate healthcare-specific conceptual analysis
    print(f"Healthcare dimensions: {DOMAIN_DIMENSIONS['healthcare']}")
    
    return kse_memory


async def finance_example():
    """Demonstrate finance domain usage."""
    print("\n=== Finance Domain Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='weaviate',
            url='http://localhost:8080',
            class_name='FinancialEntity'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create financial entities
    investment_entity = Entity(
        id="investment_001",
        name="Tesla Inc. (TSLA) Investment Analysis",
        description="Growth stock analysis for electric vehicle manufacturer",
        domain="finance",
        entity_type="investment_opportunity",
        attributes={
            "symbol": "TSLA",
            "sector": "Automotive/Technology",
            "market_cap": 800000000000,  # $800B
            "pe_ratio": 45.2,
            "risk_level": "high",
            "analyst_rating": "Buy"
        },
        tags=["growth", "technology", "automotive", "esg"]
    )
    
    portfolio_entity = Entity(
        id="portfolio_001",
        name="Aggressive Growth Portfolio",
        description="High-risk, high-reward investment portfolio",
        domain="finance",
        entity_type="portfolio",
        attributes={
            "total_value": 1000000,
            "risk_tolerance": "aggressive",
            "time_horizon": "long_term",
            "allocation": {
                "stocks": 0.8,
                "bonds": 0.1,
                "alternatives": 0.1
            }
        },
        tags=["growth", "aggressive", "long_term"]
    )
    
    await kse_memory.store_entity(investment_entity)
    await kse_memory.store_entity(portfolio_entity)
    
    # Search for similar investments
    results = await kse_memory.search_by_domain(
        "finance", 
        "high growth technology stocks", 
        limit=10
    )
    print(f"Found {len(results)} similar financial instruments")
    
    print(f"Finance dimensions: {DOMAIN_DIMENSIONS['finance']}")
    
    return kse_memory


async def real_estate_example():
    """Demonstrate real estate domain usage."""
    print("\n=== Real Estate Domain Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='qdrant',
            url='localhost:6333',
            collection_name='real_estate_entities'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create real estate entities
    property_entity = Entity(
        id="property_001",
        name="Downtown Luxury Condo - 123 Main St",
        description="Modern 2BR/2BA condo in prime downtown location",
        domain="real_estate",
        entity_type="residential_property",
        attributes={
            "price": 850000,
            "bedrooms": 2,
            "bathrooms": 2,
            "square_feet": 1200,
            "location": "downtown",
            "property_type": "condominium",
            "year_built": 2020,
            "amenities": ["gym", "rooftop", "concierge"]
        },
        tags=["luxury", "downtown", "modern", "investment"]
    )
    
    market_entity = Entity(
        id="market_001",
        name="Downtown Real Estate Market Analysis",
        description="Market trends and analysis for downtown residential properties",
        domain="real_estate",
        entity_type="market_analysis",
        attributes={
            "average_price_per_sqft": 700,
            "market_trend": "appreciating",
            "inventory_level": "low",
            "days_on_market": 25,
            "price_growth_yoy": 0.08
        },
        tags=["market", "downtown", "residential", "trends"]
    )
    
    await kse_memory.store_entity(property_entity)
    await kse_memory.store_entity(market_entity)
    
    # Search for similar properties
    results = await kse_memory.search_by_domain(
        "real_estate",
        "luxury downtown condominium",
        limit=5
    )
    print(f"Found {len(results)} similar properties")
    
    print(f"Real Estate dimensions: {DOMAIN_DIMENSIONS['real_estate']}")
    
    return kse_memory


async def enterprise_example():
    """Demonstrate enterprise domain usage."""
    print("\n=== Enterprise Domain Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='chroma',
            host='localhost',
            port=8000,
            collection_name='enterprise_entities'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create enterprise entities
    project_entity = Entity(
        id="project_001",
        name="Digital Transformation Initiative",
        description="Company-wide digital transformation project",
        domain="enterprise",
        entity_type="project",
        attributes={
            "budget": 5000000,
            "duration_months": 18,
            "team_size": 25,
            "business_impact": "high",
            "risk_level": "medium",
            "stakeholders": ["CEO", "CTO", "Department Heads"]
        },
        tags=["transformation", "digital", "strategic", "high-priority"]
    )
    
    process_entity = Entity(
        id="process_001",
        name="Customer Onboarding Process",
        description="Streamlined process for new customer acquisition",
        domain="enterprise",
        entity_type="business_process",
        attributes={
            "process_owner": "Sales Operations",
            "cycle_time_days": 7,
            "automation_level": "partial",
            "customer_satisfaction": 4.2,
            "efficiency_rating": "good"
        },
        tags=["onboarding", "customer", "sales", "process"]
    )
    
    await kse_memory.store_entity(project_entity)
    await kse_memory.store_entity(process_entity)
    
    # Search for similar projects
    results = await kse_memory.search_by_domain(
        "enterprise",
        "digital transformation high impact project",
        limit=5
    )
    print(f"Found {len(results)} similar enterprise initiatives")
    
    print(f"Enterprise dimensions: {DOMAIN_DIMENSIONS['enterprise']}")
    
    return kse_memory


async def research_example():
    """Demonstrate research domain usage."""
    print("\n=== Research Domain Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='milvus',
            host='localhost',
            port=19530,
            collection_name='research_entities'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create research entities
    paper_entity = Entity(
        id="paper_001",
        name="Attention Is All You Need",
        description="Seminal paper introducing the Transformer architecture",
        domain="research",
        entity_type="research_paper",
        attributes={
            "authors": ["Vaswani", "Shazeer", "Parmar", "et al."],
            "venue": "NIPS 2017",
            "citations": 50000,
            "field": "Natural Language Processing",
            "novelty": "breakthrough",
            "impact_factor": "very_high"
        },
        tags=["transformer", "attention", "nlp", "deep-learning"]
    )
    
    dataset_entity = Entity(
        id="dataset_001",
        name="ImageNet Large Scale Visual Recognition Challenge",
        description="Large-scale dataset for visual object recognition research",
        domain="research",
        entity_type="dataset",
        attributes={
            "size_gb": 150,
            "num_images": 14000000,
            "num_classes": 1000,
            "data_type": "image",
            "license": "research_only",
            "quality": "high"
        },
        tags=["computer-vision", "classification", "benchmark", "large-scale"]
    )
    
    await kse_memory.store_entity(paper_entity)
    await kse_memory.store_entity(dataset_entity)
    
    # Search for similar research
    results = await kse_memory.search_by_domain(
        "research",
        "transformer architecture deep learning",
        limit=5
    )
    print(f"Found {len(results)} similar research papers")
    
    print(f"Research dimensions: {DOMAIN_DIMENSIONS['research']}")
    
    return kse_memory


async def retail_example():
    """Demonstrate retail domain usage (legacy compatibility)."""
    print("\n=== Retail Domain Example (Legacy Compatible) ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='pinecone',
            api_key='your-pinecone-key',
            environment='us-west1-gcp',
            index_name='retail-entities'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create retail entities (backward compatible with Product)
    product_entity = Entity(
        id="product_001",
        name="iPhone 15 Pro",
        description="Latest flagship smartphone from Apple",
        domain="retail",
        entity_type="product",
        attributes={
            "price": 999.99,
            "brand": "Apple",
            "category": "Electronics",
            "sku": "IPHONE15PRO-128GB",
            "stock_quantity": 50,
            "rating": 4.8,
            "reviews_count": 1250
        },
        tags=["smartphone", "apple", "premium", "electronics"]
    )
    
    # Demonstrate backward compatibility
    legacy_product = await kse_memory.create_product(
        name="Samsung Galaxy S24",
        description="Android flagship smartphone",
        price=899.99,
        category="Electronics",
        brand="Samsung"
    )
    
    await kse_memory.store_entity(product_entity)
    await kse_memory.store_entity(legacy_product)
    
    # Search for similar products
    results = await kse_memory.search_by_domain(
        "retail",
        "premium smartphone flagship",
        limit=5
    )
    print(f"Found {len(results)} similar products")
    
    print(f"Retail dimensions: {DOMAIN_DIMENSIONS['retail']}")
    
    return kse_memory


async def marketing_example():
    """Demonstrate marketing domain usage."""
    print("\n=== Marketing Domain Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='weaviate',
            url='http://localhost:8080',
            class_name='MarketingEntity'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create marketing entities
    campaign_entity = Entity(
        id="campaign_001",
        name="Summer Sale 2024 Campaign",
        description="Multi-channel summer promotion campaign",
        domain="marketing",
        entity_type="campaign",
        attributes={
            "budget": 100000,
            "duration_days": 30,
            "channels": ["email", "social", "display", "search"],
            "target_audience": "millennials_gen_z",
            "engagement_potential": "high",
            "expected_roi": 3.5
        },
        tags=["summer", "sale", "multi-channel", "seasonal"]
    )
    
    audience_entity = Entity(
        id="audience_001",
        name="Tech-Savvy Millennials",
        description="Technology-oriented millennial demographic segment",
        domain="marketing",
        entity_type="audience_segment",
        attributes={
            "age_range": "25-40",
            "income_level": "middle_to_high",
            "interests": ["technology", "sustainability", "experiences"],
            "preferred_channels": ["social", "mobile", "video"],
            "engagement_rate": 0.08
        },
        tags=["millennials", "technology", "engaged", "digital-native"]
    )
    
    await kse_memory.store_entity(campaign_entity)
    await kse_memory.store_entity(audience_entity)
    
    # Search for similar campaigns
    results = await kse_memory.search_by_domain(
        "marketing",
        "multi-channel promotional campaign",
        limit=5
    )
    print(f"Found {len(results)} similar marketing campaigns")
    
    print(f"Marketing dimensions: {DOMAIN_DIMENSIONS['marketing']}")
    
    return kse_memory


async def cross_domain_search_example():
    """Demonstrate cross-domain search capabilities."""
    print("\n=== Cross-Domain Search Example ===")
    
    config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='pinecone',
            api_key='your-pinecone-key',
            environment='us-west1-gcp',
            index_name='universal-entities'
        )
    )
    
    kse_memory = KSEMemory(config)
    await kse_memory.initialize()
    
    # Create entities across multiple domains
    entities = [
        Entity(
            id="health_tech_001",
            name="AI-Powered Diagnostic Tool",
            description="Machine learning system for medical image analysis",
            domain="healthcare",
            entity_type="medical_device",
            tags=["ai", "diagnostics", "innovation"]
        ),
        Entity(
            id="fintech_001",
            name="Blockchain Payment Platform",
            description="Cryptocurrency-based payment processing system",
            domain="finance",
            entity_type="fintech_product",
            tags=["blockchain", "payments", "innovation"]
        ),
        Entity(
            id="proptech_001",
            name="Smart Building Management System",
            description="IoT-based system for building automation and energy efficiency",
            domain="real_estate",
            entity_type="proptech_solution",
            tags=["iot", "automation", "innovation"]
        )
    ]
    
    # Store all entities
    for entity in entities:
        await kse_memory.store_entity(entity)
    
    # Cross-domain search for innovation
    search_query = SearchQuery(
        text="innovative AI technology solutions",
        # No domain filter - search across all domains
        limit=10
    )
    
    results = await kse_memory.search(search_query)
    print(f"Cross-domain search found {len(results.entities)} innovative solutions")
    
    # Display results by domain
    domain_counts = {}
    for entity in results.entities:
        domain = entity.domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("Results by domain:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} entities")
    
    return kse_memory


async def migration_example():
    """Demonstrate migration from legacy LiftOS data."""
    print("\n=== Migration Example ===")
    
    # Simulate legacy LiftOS data
    legacy_data = {
        'products': [
            {
                'id': 'prod_001',
                'name': 'Wireless Headphones',
                'description': 'High-quality wireless headphones',
                'price': 199.99,
                'category': 'Electronics',
                'brand': 'AudioTech'
            },
            {
                'id': 'prod_002',
                'name': 'Smart Watch',
                'description': 'Fitness tracking smart watch',
                'price': 299.99,
                'category': 'Wearables',
                'brand': 'TechCorp'
            }
        ],
        'conceptual_dimensions': {
            'prod_001': {
                'price_range': 0.4,
                'quality_tier': 0.8,
                'brand_prestige': 0.6,
                'market_demand': 0.7
            },
            'prod_002': {
                'price_range': 0.6,
                'quality_tier': 0.9,
                'brand_prestige': 0.8,
                'market_demand': 0.8
            }
        }
    }
    
    # Create migration plan
    target_config = KSEConfig(
        vector_store=VectorStoreConfig(
            provider='pinecone',
            api_key='your-pinecone-key',
            environment='us-west1-gcp',
            index_name='migrated-entities'
        )
    )
    
    migration_plan = await create_migration_plan(legacy_data, target_config)
    print(f"Migration plan created with {len(migration_plan['steps'])} steps")
    print(f"Estimated duration: {migration_plan['estimated_duration']:.1f} seconds")
    
    # Execute migration
    migrator = KSEMigrator(target_config)
    await migrator.initialize()
    
    # Migrate products
    migrated_entities = await migrator.migrate_products_to_entities(legacy_data['products'])
    print(f"Migrated {len(migrated_entities)} products to universal entities")
    
    # Migrate conceptual dimensions
    migrated_spaces = await migrator.migrate_conceptual_dimensions(legacy_data['conceptual_dimensions'])
    print(f"Migrated {len(migrated_spaces)} conceptual spaces")
    
    # Validate migration
    validation_report = await migrator.validate_migration()
    print(f"Migration validation: {validation_report['success_rate']:.1f}% success rate")
    
    return migrator


async def main():
    """Run all examples demonstrating universal KSE-SDK capabilities."""
    print("KSE Memory SDK Universal Usage Examples")
    print("=" * 50)
    
    try:
        # Run domain-specific examples
        await healthcare_example()
        await finance_example()
        await real_estate_example()
        await enterprise_example()
        await research_example()
        await retail_example()
        await marketing_example()
        
        # Run cross-domain examples
        await cross_domain_search_example()
        
        # Run migration example
        await migration_example()
        
        print("\n=== All Examples Completed Successfully ===")
        print("The universal KSE-SDK supports:")
        print("✓ Healthcare domain with medical entities and clinical dimensions")
        print("✓ Finance domain with investment entities and risk dimensions")
        print("✓ Real estate domain with property entities and market dimensions")
        print("✓ Enterprise domain with business entities and impact dimensions")
        print("✓ Research domain with academic entities and novelty dimensions")
        print("✓ Retail domain with product entities (backward compatible)")
        print("✓ Marketing domain with campaign entities and engagement dimensions")
        print("✓ Cross-domain search and analysis")
        print("✓ Migration from legacy LiftOS data")
        print("✓ Hybrid search across neural, conceptual, and graph pillars")
        
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())