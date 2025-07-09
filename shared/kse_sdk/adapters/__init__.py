"""
KSE Memory SDK Adapters Module

This module provides adapter functions for integrating with various backend systems
and external services. Adapters handle the translation between KSE SDK interfaces
and specific backend implementations.
"""

from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_adapter(adapter_type: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    Get an adapter instance for the specified type and configuration.
    
    This function serves as a factory for creating adapter instances that
    integrate KSE SDK with various backend systems and external services.
    
    Args:
        adapter_type: Type of adapter to create (e.g., 'vector', 'graph', 'concept')
        config: Configuration dictionary for the adapter
        
    Returns:
        Adapter instance or None if not found
        
    Example:
        >>> config = {'backend': 'pinecone', 'api_key': 'xxx'}
        >>> adapter = get_adapter('vector', config)
    """
    try:
        if adapter_type == 'vector':
            return _get_vector_adapter(config)
        elif adapter_type == 'graph':
            return _get_graph_adapter(config)
        elif adapter_type == 'concept':
            return _get_concept_adapter(config)
        elif adapter_type == 'cache':
            return _get_cache_adapter(config)
        elif adapter_type == 'analytics':
            return _get_analytics_adapter(config)
        elif adapter_type == 'security':
            return _get_security_adapter(config)
        elif adapter_type == 'workflow':
            return _get_workflow_adapter(config)
        elif adapter_type == 'notification':
            return _get_notification_adapter(config)
        else:
            logger.warning(f"Unknown adapter type: {adapter_type}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create {adapter_type} adapter: {e}")
        return None


def _get_vector_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get vector store adapter."""
    try:
        from ..backends.vector_stores import get_vector_store
        backend = config.get('backend', 'pinecone')
        return get_vector_store(backend, config)
    except Exception as e:
        logger.error(f"Failed to create vector adapter: {e}")
        return None


def _get_graph_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get graph store adapter."""
    try:
        from ..backends.graph_stores import get_graph_store
        backend = config.get('backend', 'neo4j')
        return get_graph_store(backend, config)
    except Exception as e:
        logger.error(f"Failed to create graph adapter: {e}")
        return None


def _get_concept_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get concept store adapter."""
    try:
        from ..backends.concept_stores import get_concept_store
        backend = config.get('backend', 'postgresql')
        return get_concept_store(backend, config)
    except Exception as e:
        logger.error(f"Failed to create concept adapter: {e}")
        return None


def _get_cache_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get cache service adapter."""
    try:
        from ..services.cache_service import CacheService
        return CacheService(config)
    except Exception as e:
        logger.error(f"Failed to create cache adapter: {e}")
        return None


def _get_analytics_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get analytics service adapter."""
    try:
        from ..services.analytics_service import AnalyticsService
        return AnalyticsService(config)
    except Exception as e:
        logger.error(f"Failed to create analytics adapter: {e}")
        return None


def _get_security_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get security service adapter."""
    try:
        from ..services.security_service import SecurityService
        return SecurityService(config)
    except Exception as e:
        logger.error(f"Failed to create security adapter: {e}")
        return None


def _get_workflow_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get workflow service adapter."""
    try:
        from ..services.workflow_service import WorkflowService
        return WorkflowService(config)
    except Exception as e:
        logger.error(f"Failed to create workflow adapter: {e}")
        return None


def _get_notification_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Get notification service adapter."""
    try:
        from ..services.notification_service import NotificationService
        return NotificationService(config)
    except Exception as e:
        logger.error(f"Failed to create notification adapter: {e}")
        return None


# Legacy compatibility functions
def create_vector_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Legacy function for creating vector adapters."""
    return get_adapter('vector', config)


def create_graph_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Legacy function for creating graph adapters."""
    return get_adapter('graph', config)


def create_concept_adapter(config: Dict[str, Any]) -> Optional[Any]:
    """Legacy function for creating concept adapters."""
    return get_adapter('concept', config)


__all__ = [
    'get_adapter',
    'create_vector_adapter',
    'create_graph_adapter', 
    'create_concept_adapter'
]