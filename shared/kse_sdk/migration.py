"""
KSE Memory SDK Migration Utilities

Utilities to migrate existing LiftOS data to the universal KSE-SDK format.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from .core.models import Entity, ConceptualSpace, DOMAIN_DIMENSIONS
from .core.config import KSEConfig
from .core.memory import KSEMemory
from .exceptions import ValidationError, MigrationError

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Migration-specific errors."""
    pass


class KSEMigrator:
    """
    Migrates existing LiftOS data to universal KSE-SDK format.
    """
    
    def __init__(self, config: KSEConfig):
        """
        Initialize migrator with KSE configuration.
        
        Args:
            config: KSE configuration
        """
        self.config = config
        self.kse_memory = None
        self.migration_log = []
    
    async def initialize(self):
        """Initialize KSE Memory system for migration."""
        self.kse_memory = KSEMemory(self.config)
        await self.kse_memory.initialize()
    
    async def migrate_products_to_entities(self, products: List[Dict[str, Any]]) -> List[Entity]:
        """
        Migrate legacy Product objects to universal Entity format.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of migrated Entity objects
        """
        migrated_entities = []
        
        for product in products:
            try:
                entity = await self._product_to_entity(product)
                migrated_entities.append(entity)
                
                self.migration_log.append({
                    'type': 'product_migration',
                    'source_id': product.get('id'),
                    'target_id': entity.id,
                    'status': 'success',
                    'timestamp': datetime.utcnow()
                })
                
            except Exception as e:
                logger.error(f"Failed to migrate product {product.get('id')}: {str(e)}")
                self.migration_log.append({
                    'type': 'product_migration',
                    'source_id': product.get('id'),
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                })
        
        return migrated_entities
    
    async def migrate_conceptual_dimensions(
        self, 
        legacy_dimensions: Dict[str, Any]
    ) -> Dict[str, ConceptualSpace]:
        """
        Migrate legacy ConceptualDimensions to universal ConceptualSpace format.
        
        Args:
            legacy_dimensions: Legacy dimensional data
            
        Returns:
            Dictionary of migrated ConceptualSpace objects
        """
        migrated_spaces = {}
        
        for entity_id, dimensions in legacy_dimensions.items():
            try:
                # Determine domain from legacy data
                domain = self._infer_domain_from_legacy(dimensions)
                
                # Create universal conceptual space
                conceptual_space = ConceptualSpace(
                    entity_id=entity_id,
                    domain=domain,
                    dimensions=DOMAIN_DIMENSIONS[domain],
                    coordinates=await self._migrate_coordinates(dimensions, domain),
                    confidence=dimensions.get('confidence', 0.7),
                    metadata={
                        'migrated_from': 'legacy_conceptual_dimensions',
                        'original_dimensions': list(dimensions.keys()),
                        'migration_timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                migrated_spaces[entity_id] = conceptual_space
                
                self.migration_log.append({
                    'type': 'conceptual_migration',
                    'entity_id': entity_id,
                    'domain': domain,
                    'status': 'success',
                    'timestamp': datetime.utcnow()
                })
                
            except Exception as e:
                logger.error(f"Failed to migrate conceptual dimensions for {entity_id}: {str(e)}")
                self.migration_log.append({
                    'type': 'conceptual_migration',
                    'entity_id': entity_id,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                })
        
        return migrated_spaces
    
    async def migrate_memory_service_data(self, memory_data: Dict[str, Any]) -> bool:
        """
        Migrate existing memory service data to new KSE format.
        
        Args:
            memory_data: Legacy memory service data
            
        Returns:
            True if migration successful
        """
        try:
            # Extract entities from memory data
            entities = memory_data.get('entities', [])
            embeddings = memory_data.get('embeddings', {})
            relationships = memory_data.get('relationships', [])
            
            # Migrate entities
            migrated_entities = []
            for entity_data in entities:
                entity = await self._legacy_entity_to_universal(entity_data)
                migrated_entities.append(entity)
            
            # Store migrated entities in KSE Memory
            for entity in migrated_entities:
                await self.kse_memory.store_entity(entity)
            
            # Migrate embeddings if present
            if embeddings:
                await self._migrate_embeddings(embeddings, migrated_entities)
            
            # Migrate relationships if present
            if relationships:
                await self._migrate_relationships(relationships)
            
            self.migration_log.append({
                'type': 'memory_service_migration',
                'entities_migrated': len(migrated_entities),
                'embeddings_migrated': len(embeddings),
                'relationships_migrated': len(relationships),
                'status': 'success',
                'timestamp': datetime.utcnow()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Memory service migration failed: {str(e)}")
            self.migration_log.append({
                'type': 'memory_service_migration',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow()
            })
            return False
    
    async def validate_migration(self) -> Dict[str, Any]:
        """
        Validate the migration results.
        
        Returns:
            Validation report
        """
        validation_report = {
            'total_migrations': len(self.migration_log),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'migration_types': {},
            'errors': [],
            'warnings': []
        }
        
        for log_entry in self.migration_log:
            migration_type = log_entry['type']
            status = log_entry['status']
            
            if migration_type not in validation_report['migration_types']:
                validation_report['migration_types'][migration_type] = {
                    'success': 0, 'failed': 0
                }
            
            if status == 'success':
                validation_report['successful_migrations'] += 1
                validation_report['migration_types'][migration_type]['success'] += 1
            else:
                validation_report['failed_migrations'] += 1
                validation_report['migration_types'][migration_type]['failed'] += 1
                validation_report['errors'].append({
                    'type': migration_type,
                    'error': log_entry.get('error'),
                    'timestamp': log_entry['timestamp']
                })
        
        # Check for potential issues
        if validation_report['failed_migrations'] > 0:
            validation_report['warnings'].append(
                f"{validation_report['failed_migrations']} migrations failed"
            )
        
        success_rate = (validation_report['successful_migrations'] / 
                       validation_report['total_migrations']) * 100
        
        if success_rate < 95:
            validation_report['warnings'].append(
                f"Migration success rate is {success_rate:.1f}%, below recommended 95%"
            )
        
        validation_report['success_rate'] = success_rate
        return validation_report
    
    async def _product_to_entity(self, product: Dict[str, Any]) -> Entity:
        """
        Convert legacy Product to universal Entity.
        
        Args:
            product: Product dictionary
            
        Returns:
            Universal Entity object
        """
        # Extract product attributes
        product_id = product.get('id') or product.get('product_id')
        name = product.get('name') or product.get('title')
        description = product.get('description')
        
        if not product_id or not name:
            raise ValidationError("Product must have id and name")
        
        # Create entity attributes from product data
        attributes = {}
        
        # Standard product attributes
        if 'price' in product:
            attributes['price'] = product['price']
        if 'category' in product:
            attributes['category'] = product['category']
        if 'brand' in product:
            attributes['brand'] = product['brand']
        if 'sku' in product:
            attributes['sku'] = product['sku']
        if 'stock_quantity' in product:
            attributes['stock_quantity'] = product['stock_quantity']
        
        # Additional product-specific attributes
        for key, value in product.items():
            if key not in ['id', 'product_id', 'name', 'title', 'description'] and value is not None:
                attributes[key] = value
        
        # Extract tags
        tags = product.get('tags', [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',')]
        
        return Entity(
            id=f"migrated_product_{product_id}",
            name=name,
            description=description,
            domain='retail',  # Products default to retail domain
            entity_type='product',
            attributes=attributes,
            tags=tags,
            metadata={
                'migrated_from': 'legacy_product',
                'original_id': product_id,
                'migration_timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def _infer_domain_from_legacy(self, dimensions: Dict[str, Any]) -> str:
        """
        Infer domain from legacy dimensional data.
        
        Args:
            dimensions: Legacy dimensional data
            
        Returns:
            Inferred domain
        """
        # Check for domain-specific dimension names
        dimension_keys = set(dimensions.keys())
        
        # Healthcare indicators
        healthcare_dims = {'severity', 'urgency', 'treatment_complexity', 'patient_impact'}
        if dimension_keys.intersection(healthcare_dims):
            return 'healthcare'
        
        # Finance indicators
        finance_dims = {'risk_level', 'liquidity', 'return_potential', 'market_volatility'}
        if dimension_keys.intersection(finance_dims):
            return 'finance'
        
        # Real estate indicators
        real_estate_dims = {'price_range', 'location_desirability', 'property_condition', 'investment_potential'}
        if dimension_keys.intersection(real_estate_dims):
            return 'real_estate'
        
        # Retail indicators (legacy default)
        retail_dims = {'price_range', 'quality_tier', 'brand_prestige', 'market_demand'}
        if dimension_keys.intersection(retail_dims):
            return 'retail'
        
        # Default to enterprise for unknown patterns
        return 'enterprise'
    
    async def _migrate_coordinates(self, legacy_dimensions: Dict[str, Any], domain: str) -> Dict[str, float]:
        """
        Migrate legacy coordinates to new domain-specific format.
        
        Args:
            legacy_dimensions: Legacy dimensional coordinates
            domain: Target domain
            
        Returns:
            Migrated coordinates
        """
        target_dimensions = DOMAIN_DIMENSIONS[domain]
        migrated_coords = {}
        
        # Map legacy dimensions to new domain dimensions
        for target_dim in target_dimensions:
            # Try direct mapping first
            if target_dim in legacy_dimensions:
                migrated_coords[target_dim] = float(legacy_dimensions[target_dim])
            else:
                # Try semantic mapping
                mapped_value = await self._semantic_dimension_mapping(
                    target_dim, legacy_dimensions, domain
                )
                migrated_coords[target_dim] = mapped_value
        
        return migrated_coords
    
    async def _semantic_dimension_mapping(
        self, 
        target_dim: str, 
        legacy_dimensions: Dict[str, Any], 
        domain: str
    ) -> float:
        """
        Perform semantic mapping between legacy and target dimensions.
        
        Args:
            target_dim: Target dimension name
            legacy_dimensions: Legacy dimensional data
            domain: Target domain
            
        Returns:
            Mapped coordinate value
        """
        # Semantic mapping rules
        mapping_rules = {
            'retail': {
                'price_range': ['price', 'cost', 'value'],
                'quality_tier': ['quality', 'grade', 'tier'],
                'brand_prestige': ['brand', 'prestige', 'reputation'],
                'market_demand': ['demand', 'popularity', 'sales']
            },
            'healthcare': {
                'severity': ['severity', 'critical', 'serious'],
                'urgency': ['urgency', 'priority', 'immediate'],
                'treatment_complexity': ['complexity', 'difficulty', 'advanced'],
                'patient_impact': ['impact', 'outcome', 'effect']
            }
            # Add more domain mappings as needed
        }
        
        domain_rules = mapping_rules.get(domain, {})
        target_synonyms = domain_rules.get(target_dim, [])
        
        # Look for semantic matches
        for synonym in target_synonyms:
            for legacy_key, legacy_value in legacy_dimensions.items():
                if synonym.lower() in legacy_key.lower():
                    return float(legacy_value)
        
        # Default to neutral position if no mapping found
        return 0.5
    
    async def _legacy_entity_to_universal(self, entity_data: Dict[str, Any]) -> Entity:
        """
        Convert legacy entity data to universal Entity format.
        
        Args:
            entity_data: Legacy entity data
            
        Returns:
            Universal Entity object
        """
        # Determine entity type and domain
        entity_type = entity_data.get('type', 'unknown')
        domain = entity_data.get('domain', 'enterprise')
        
        # Ensure domain is valid
        if domain not in DOMAIN_DIMENSIONS:
            domain = 'enterprise'
        
        return Entity(
            id=entity_data.get('id', f"migrated_{datetime.utcnow().timestamp()}"),
            name=entity_data.get('name', 'Unknown Entity'),
            description=entity_data.get('description'),
            domain=domain,
            entity_type=entity_type,
            attributes=entity_data.get('attributes', {}),
            tags=entity_data.get('tags', []),
            metadata={
                'migrated_from': 'legacy_entity',
                'migration_timestamp': datetime.utcnow().isoformat(),
                **entity_data.get('metadata', {})
            }
        )
    
    async def _migrate_embeddings(self, embeddings: Dict[str, Any], entities: List[Entity]):
        """
        Migrate legacy embeddings to new format.
        
        Args:
            embeddings: Legacy embedding data
            entities: Migrated entities
        """
        # This would integrate with the embedding service to store migrated embeddings
        # Implementation depends on the legacy embedding format
        pass
    
    async def _migrate_relationships(self, relationships: List[Dict[str, Any]]):
        """
        Migrate legacy relationships to new graph format.
        
        Args:
            relationships: Legacy relationship data
        """
        # This would integrate with the graph store to migrate relationships
        # Implementation depends on the legacy relationship format
        pass
    
    def get_migration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive migration report.
        
        Returns:
            Migration report with statistics and logs
        """
        return {
            'migration_log': self.migration_log,
            'total_operations': len(self.migration_log),
            'successful_operations': len([log for log in self.migration_log if log['status'] == 'success']),
            'failed_operations': len([log for log in self.migration_log if log['status'] == 'failed']),
            'migration_types': list(set(log['type'] for log in self.migration_log)),
            'start_time': min(log['timestamp'] for log in self.migration_log) if self.migration_log else None,
            'end_time': max(log['timestamp'] for log in self.migration_log) if self.migration_log else None
        }


async def create_migration_plan(
    legacy_data: Dict[str, Any], 
    target_config: KSEConfig
) -> Dict[str, Any]:
    """
    Create a migration plan for legacy data.
    
    Args:
        legacy_data: Legacy data to migrate
        target_config: Target KSE configuration
        
    Returns:
        Migration plan with steps and estimates
    """
    plan = {
        'steps': [],
        'estimated_duration': 0,
        'data_analysis': {},
        'recommendations': []
    }
    
    # Analyze legacy data
    if 'products' in legacy_data:
        product_count = len(legacy_data['products'])
        plan['steps'].append({
            'step': 'migrate_products',
            'description': f'Migrate {product_count} products to universal entities',
            'estimated_time': product_count * 0.1  # seconds
        })
        plan['data_analysis']['products'] = product_count
    
    if 'conceptual_dimensions' in legacy_data:
        dimension_count = len(legacy_data['conceptual_dimensions'])
        plan['steps'].append({
            'step': 'migrate_dimensions',
            'description': f'Migrate {dimension_count} conceptual dimensions',
            'estimated_time': dimension_count * 0.05
        })
        plan['data_analysis']['conceptual_dimensions'] = dimension_count
    
    if 'memory_data' in legacy_data:
        plan['steps'].append({
            'step': 'migrate_memory',
            'description': 'Migrate memory service data',
            'estimated_time': 30  # seconds
        })
    
    # Calculate total estimated duration
    plan['estimated_duration'] = sum(step['estimated_time'] for step in plan['steps'])
    
    # Add recommendations
    if plan['estimated_duration'] > 300:  # 5 minutes
        plan['recommendations'].append(
            'Consider running migration in batches for large datasets'
        )
    
    if 'products' in legacy_data and len(legacy_data['products']) > 1000:
        plan['recommendations'].append(
            'Enable parallel processing for product migration'
        )
    
    return plan