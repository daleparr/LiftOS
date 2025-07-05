"""
Structured logging utilities for Lift OS Core services
Provides JSON-formatted logging with service context
"""

import structlog
import logging
import logging.config
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

class LiftOSLogger:
    """Centralized logging configuration for Lift OS services"""
    
    def __init__(self, service_name: str, log_level: str = "INFO", enable_json: bool = True):
        self.service_name = service_name
        self.log_level = log_level.upper()
        self.enable_json = enable_json
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure structured logging"""
        
        # Configure standard library logging
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": self._get_processor(),
                },
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(colors=True),
                },
            },
            "handlers": {
                "default": {
                    "level": self.log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.enable_json else "console",
                    "stream": sys.stdout,
                },
                "error": {
                    "level": "ERROR",
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.enable_json else "console",
                    "stream": sys.stderr,
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": self.log_level,
                    "propagate": True,
                },
                "error": {
                    "handlers": ["error"],
                    "level": "ERROR",
                    "propagate": False,
                },
            }
        })
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_service_context,
                self._get_processor(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _get_processor(self):
        """Get the appropriate processor based on configuration"""
        if self.enable_json:
            return structlog.processors.JSONRenderer()
        else:
            return structlog.dev.ConsoleRenderer(colors=True)
    
    def _add_service_context(self, logger, method_name, event_dict):
        """Add service context to all log entries"""
        event_dict["service"] = self.service_name
        event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
        event_dict["version"] = os.getenv("SERVICE_VERSION", "1.0.0")
        return event_dict
    
    def get_logger(self, name: Optional[str] = None) -> structlog.BoundLogger:
        """Get a configured logger instance"""
        logger_name = name or self.service_name
        logger = structlog.get_logger(logger_name)
        return logger.bind(service=self.service_name)

class RequestLogger:
    """HTTP request logging middleware"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    async def log_request(self, request, call_next):
        """Log HTTP requests and responses"""
        start_time = datetime.utcnow()
        
        # Log incoming request
        self.logger.info(
            "HTTP request started",
            method=request.method,
            url=str(request.url),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            request_id=self._generate_request_id()
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Log response
            self.logger.info(
                "HTTP request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration,
                response_size=response.headers.get("content-length")
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.error(
                "HTTP request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                duration_seconds=duration,
                exc_info=True
            )
            raise
    
    def _get_client_ip(self, request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())

class DatabaseLogger:
    """Database operation logging"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_query(self, query: str, params: Dict = None, duration: float = None):
        """Log database queries"""
        self.logger.debug(
            "Database query executed",
            query=query,
            params=params,
            duration_seconds=duration
        )
    
    def log_transaction(self, operation: str, table: str, affected_rows: int = None):
        """Log database transactions"""
        self.logger.info(
            "Database transaction",
            operation=operation,
            table=table,
            affected_rows=affected_rows
        )
    
    def log_error(self, error: Exception, query: str = None):
        """Log database errors"""
        self.logger.error(
            "Database error",
            error=str(error),
            query=query,
            exc_info=True
        )

class SecurityLogger:
    """Security event logging"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempts"""
        self.logger.info(
            "Authentication attempt",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            event_type="authentication"
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, granted: bool):
        """Log authorization decisions"""
        self.logger.info(
            "Authorization check",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            event_type="authorization"
        )
    
    def log_rate_limit(self, ip_address: str, endpoint: str, limit_exceeded: bool):
        """Log rate limiting events"""
        self.logger.warning(
            "Rate limit check",
            ip_address=ip_address,
            endpoint=endpoint,
            limit_exceeded=limit_exceeded,
            event_type="rate_limit"
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Log security violations"""
        self.logger.error(
            "Security violation detected",
            violation_type=violation_type,
            details=details,
            event_type="security_violation"
        )

# Global logger instances
_loggers: Dict[str, LiftOSLogger] = {}

def setup_service_logging(service_name: str, log_level: str = "INFO", enable_json: bool = True) -> structlog.BoundLogger:
    """Setup logging for a service and return configured logger"""
    global _loggers
    
    if service_name not in _loggers:
        _loggers[service_name] = LiftOSLogger(
            service_name=service_name,
            log_level=log_level,
            enable_json=enable_json
        )
    
    return _loggers[service_name].get_logger()

def get_service_logger(service_name: str) -> structlog.BoundLogger:
    """Get existing logger for service"""
    global _loggers
    
    if service_name not in _loggers:
        return setup_service_logging(service_name)
    
    return _loggers[service_name].get_logger()

# Convenience functions for different log types
def get_request_logger(service_name: str) -> RequestLogger:
    """Get request logger for service"""
    logger = get_service_logger(service_name)
    return RequestLogger(logger)

def get_database_logger(service_name: str) -> DatabaseLogger:
    """Get database logger for service"""
    logger = get_service_logger(service_name)
    return DatabaseLogger(logger)

def get_security_logger(service_name: str) -> SecurityLogger:
    """Get security logger for service"""
    logger = get_service_logger(service_name)
    return SecurityLogger(logger)