"""
KSE Memory SDK Exceptions

This module defines all custom exceptions used throughout the KSE Memory SDK.
"""


class KSEError(Exception):
    """Base exception for all KSE Memory SDK errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(KSEError):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class ConnectionError(KSEError):
    """Raised when there's a connection error to backend services."""
    
    def __init__(self, message: str, service: str = None, details: dict = None):
        super().__init__(message, "CONNECTION_ERROR", details)
        self.service = service


class AuthenticationError(KSEError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, auth_type: str = None, details: dict = None):
        super().__init__(message, "AUTH_ERROR", details)
        self.auth_type = auth_type


class AuthorizationError(KSEError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        super().__init__(message, "AUTHZ_ERROR", details)
        self.operation = operation


class ValidationError(KSEError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None, details: dict = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field


class SearchError(KSEError):
    """Raised when search operations fail."""
    
    def __init__(self, message: str, search_type: str = None, details: dict = None):
        super().__init__(message, "SEARCH_ERROR", details)
        self.search_type = search_type


class StorageError(KSEError):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str, storage_type: str = None, details: dict = None):
        super().__init__(message, "STORAGE_ERROR", details)
        self.storage_type = storage_type


class EmbeddingError(KSEError):
    """Raised when embedding operations fail."""
    
    def __init__(self, message: str, model: str = None, details: dict = None):
        super().__init__(message, "EMBEDDING_ERROR", details)
        self.model = model


class ConceptualError(KSEError):
    """Raised when conceptual space operations fail."""
    
    def __init__(self, message: str, concept: str = None, details: dict = None):
        super().__init__(message, "CONCEPTUAL_ERROR", details)
        self.concept = concept


class GraphError(KSEError):
    """Raised when knowledge graph operations fail."""
    
    def __init__(self, message: str, graph_type: str = None, details: dict = None):
        super().__init__(message, "GRAPH_ERROR", details)
        self.graph_type = graph_type


class CacheError(KSEError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_type: str = None, details: dict = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.cache_type = cache_type


class WorkflowError(KSEError):
    """Raised when workflow operations fail."""
    
    def __init__(self, message: str, workflow_id: str = None, details: dict = None):
        super().__init__(message, "WORKFLOW_ERROR", details)
        self.workflow_id = workflow_id


class NotificationError(KSEError):
    """Raised when notification operations fail."""
    
    def __init__(self, message: str, channel: str = None, details: dict = None):
        super().__init__(message, "NOTIFICATION_ERROR", details)
        self.channel = channel


class SecurityError(KSEError):
    """Raised when security operations fail."""
    
    def __init__(self, message: str, security_type: str = None, details: dict = None):
        super().__init__(message, "SECURITY_ERROR", details)
        self.security_type = security_type


class AnalyticsError(KSEError):
    """Raised when analytics operations fail."""
    
    def __init__(self, message: str, metric: str = None, details: dict = None):
        super().__init__(message, "ANALYTICS_ERROR", details)
        self.metric = metric


class RateLimitError(KSEError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str = None, details: dict = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.limit_type = limit_type


class TimeoutError(KSEError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.operation = operation


class ResourceNotFoundError(KSEError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        super().__init__(message, "RESOURCE_NOT_FOUND", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateResourceError(KSEError):
    """Raised when attempting to create a duplicate resource."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        super().__init__(message, "DUPLICATE_RESOURCE", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class IncompatibleVersionError(KSEError):
    """Raised when there's a version compatibility issue."""
    
    def __init__(self, message: str, required_version: str = None, current_version: str = None, details: dict = None):
        super().__init__(message, "VERSION_INCOMPATIBLE", details)
        self.required_version = required_version
        self.current_version = current_version


class BackendError(KSEError):
    """Raised when backend operations fail."""
    
    def __init__(self, message: str, backend_type: str = None, details: dict = None):
        super().__init__(message, "BACKEND_ERROR", details)
        self.backend_type = backend_type


# Exception hierarchy for easy catching
BACKEND_ERRORS = (ConnectionError, StorageError, EmbeddingError, GraphError, CacheError, BackendError)
AUTH_ERRORS = (AuthenticationError, AuthorizationError, SecurityError)
VALIDATION_ERRORS = (ValidationError, ConfigurationError)
OPERATION_ERRORS = (SearchError, ConceptualError, WorkflowError, NotificationError, AnalyticsError)
RESOURCE_ERRORS = (ResourceNotFoundError, DuplicateResourceError)
SYSTEM_ERRORS = (RateLimitError, TimeoutError, IncompatibleVersionError)

ALL_KSE_ERRORS = (
    KSEError,
    *BACKEND_ERRORS,
    *AUTH_ERRORS, 
    *VALIDATION_ERRORS,
    *OPERATION_ERRORS,
    *RESOURCE_ERRORS,
    *SYSTEM_ERRORS
)