"""
Logging utilities for Lift OS Core
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger
from shared.utils.config import get_config


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records"""
    
    def __init__(self, correlation_id: str = None):
        super().__init__()
        self.correlation_id = correlation_id
    
    def filter(self, record):
        record.correlation_id = getattr(record, 'correlation_id', self.correlation_id or 'unknown')
        return True


class LiftOSFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for Lift OS Core"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = getattr(record, 'service', 'unknown')
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add user context if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'org_id'):
            log_record['org_id'] = record.org_id
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'endpoint'):
            log_record['endpoint'] = record.endpoint
        if hasattr(record, 'method'):
            log_record['method'] = record.method
        
        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_record['duration'] = record.duration
        if hasattr(record, 'status_code'):
            log_record['status_code'] = record.status_code


def setup_logging(
    service_name: str,
    log_level: str = None,
    correlation_id: str = None
) -> logging.Logger:
    """Setup logging for a service"""
    
    config = get_config()
    level = log_level or config.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if config.is_development():
        # Use simple format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Use JSON format for production
        formatter = LiftOSFormatter(
            '%(timestamp)s %(level)s %(service)s %(logger)s %(message)s'
        )
    
    handler.setFormatter(formatter)
    
    # Add correlation ID filter
    if correlation_id:
        handler.addFilter(CorrelationIdFilter(correlation_id))
    
    logger.addHandler(handler)
    
    # Set service name
    logger = logging.LoggerAdapter(logger, {'service': service_name})
    
    return logger


def get_logger(name: str, service_name: str = None) -> logging.Logger:
    """Get a logger instance"""
    service = service_name or 'lift-os-core'
    return setup_logging(f"{service}.{name}")


class RequestLogger:
    """Logger for HTTP requests"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        user_id: str = None,
        org_id: str = None,
        correlation_id: str = None,
        request_id: str = None,
        **kwargs
    ):
        """Log incoming request"""
        extra = {
            'method': method,
            'endpoint': endpoint,
            'user_id': user_id,
            'org_id': org_id,
            'correlation_id': correlation_id,
            'request_id': request_id,
            **kwargs
        }
        
        self.logger.info(
            f"Request: {method} {endpoint}",
            extra=extra
        )
    
    def log_response(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: str = None,
        org_id: str = None,
        correlation_id: str = None,
        request_id: str = None,
        **kwargs
    ):
        """Log response"""
        extra = {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration': duration,
            'user_id': user_id,
            'org_id': org_id,
            'correlation_id': correlation_id,
            'request_id': request_id,
            **kwargs
        }
        
        level = logging.ERROR if status_code >= 500 else logging.WARNING if status_code >= 400 else logging.INFO
        
        self.logger.log(
            level,
            f"Response: {method} {endpoint} - {status_code} ({duration:.3f}s)",
            extra=extra
        )


class MemoryLogger:
    """Logger for memory operations"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_search(
        self,
        org_id: str,
        query: str,
        search_type: str,
        results_count: int,
        duration: float,
        user_id: str = None,
        correlation_id: str = None,
        **kwargs
    ):
        """Log memory search operation"""
        extra = {
            'operation': 'memory_search',
            'org_id': org_id,
            'user_id': user_id,
            'search_type': search_type,
            'results_count': results_count,
            'duration': duration,
            'correlation_id': correlation_id,
            'query_length': len(query),
            **kwargs
        }
        
        self.logger.info(
            f"Memory search: {search_type} query returned {results_count} results ({duration:.3f}s)",
            extra=extra
        )
    
    def log_storage(
        self,
        org_id: str,
        memory_type: str,
        content_length: int,
        memory_id: str,
        user_id: str = None,
        correlation_id: str = None,
        **kwargs
    ):
        """Log memory storage operation"""
        extra = {
            'operation': 'memory_storage',
            'org_id': org_id,
            'user_id': user_id,
            'memory_type': memory_type,
            'content_length': content_length,
            'memory_id': memory_id,
            'correlation_id': correlation_id,
            **kwargs
        }
        
        self.logger.info(
            f"Memory stored: {memory_type} ({content_length} chars) -> {memory_id}",
            extra=extra
        )
    
    def log_context_init(
        self,
        org_id: str,
        context_id: str,
        domain: str,
        user_id: str = None,
        correlation_id: str = None,
        **kwargs
    ):
        """Log memory context initialization"""
        extra = {
            'operation': 'memory_context_init',
            'org_id': org_id,
            'user_id': user_id,
            'context_id': context_id,
            'domain': domain,
            'correlation_id': correlation_id,
            **kwargs
        }
        
        self.logger.info(
            f"Memory context initialized: {context_id} (domain: {domain})",
            extra=extra
        )


class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_auth_success(
        self,
        user_id: str,
        org_id: str,
        method: str,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs
    ):
        """Log successful authentication"""
        extra = {
            'event': 'auth_success',
            'user_id': user_id,
            'org_id': org_id,
            'auth_method': method,
            'ip_address': ip_address,
            'user_agent': user_agent,
            **kwargs
        }
        
        self.logger.info(
            f"Authentication successful: {user_id} via {method}",
            extra=extra
        )
    
    def log_auth_failure(
        self,
        email: str,
        method: str,
        reason: str,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs
    ):
        """Log failed authentication"""
        extra = {
            'event': 'auth_failure',
            'email': email,
            'auth_method': method,
            'failure_reason': reason,
            'ip_address': ip_address,
            'user_agent': user_agent,
            **kwargs
        }
        
        self.logger.warning(
            f"Authentication failed: {email} via {method} - {reason}",
            extra=extra
        )
    
    def log_permission_denied(
        self,
        user_id: str,
        org_id: str,
        resource: str,
        required_permission: str,
        **kwargs
    ):
        """Log permission denied"""
        extra = {
            'event': 'permission_denied',
            'user_id': user_id,
            'org_id': org_id,
            'resource': resource,
            'required_permission': required_permission,
            **kwargs
        }
        
        self.logger.warning(
            f"Permission denied: {user_id} accessing {resource} (needs {required_permission})",
            extra=extra
        )


# Global logger instances
def get_request_logger(service_name: str) -> RequestLogger:
    """Get request logger for a service"""
    logger = get_logger('requests', service_name)
    return RequestLogger(logger)


def get_memory_logger(service_name: str) -> MemoryLogger:
    """Get memory logger for a service"""
    logger = get_logger('memory', service_name)
    return MemoryLogger(logger)


def get_security_logger(service_name: str) -> SecurityLogger:
    """Get security logger for a service"""
    logger = get_logger('security', service_name)
    return SecurityLogger(logger)