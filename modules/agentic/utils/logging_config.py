"""
Logging Configuration for Agentic Module

Provides centralized logging configuration for the Agentic microservice.
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path


def setup_logging(
    name: str,
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration for the Agentic module.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Optional log file path
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_structured_logging(
    name: str,
    level: str = "INFO",
    service_name: str = "agentic",
    service_version: str = "1.0.0"
) -> logging.Logger:
    """
    Setup structured logging with JSON format for production environments.
    
    Args:
        name: Logger name
        level: Logging level
        service_name: Name of the service
        service_version: Version of the service
        
    Returns:
        Configured logger with structured output
    """
    
    import json
    from datetime import datetime
    
    class StructuredFormatter(logging.Formatter):
        """Custom formatter for structured JSON logging."""
        
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "service": service_name,
                "version": service_version,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in [
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                ]:
                    log_entry[key] = value
            
            return json.dumps(log_entry)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def configure_third_party_loggers(level: str = "WARNING") -> None:
    """
    Configure logging levels for third-party libraries.
    
    Args:
        level: Logging level for third-party libraries
    """
    
    # Common third-party loggers to configure
    third_party_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "asyncio",
        "uvicorn",
        "fastapi",
        "pydantic"
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


def add_correlation_id_filter(logger: logging.Logger) -> None:
    """
    Add a filter to include correlation IDs in log messages.
    
    Args:
        logger: Logger to add the filter to
    """
    
    import contextvars
    
    # Context variable for correlation ID
    correlation_id_var = contextvars.ContextVar('correlation_id', default=None)
    
    class CorrelationIdFilter(logging.Filter):
        """Filter to add correlation ID to log records."""
        
        def filter(self, record):
            correlation_id = correlation_id_var.get()
            if correlation_id:
                record.correlation_id = correlation_id
            return True
    
    logger.addFilter(CorrelationIdFilter())


def setup_performance_logging(
    name: str,
    level: str = "INFO"
) -> logging.Logger:
    """
    Setup performance-focused logging for monitoring and debugging.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Performance logger instance
    """
    
    logger = logging.getLogger(f"{name}.performance")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Performance-specific format
    perf_format = (
        "%(asctime)s - PERF - %(name)s - %(levelname)s - "
        "%(message)s"
    )
    
    formatter = logging.Formatter(perf_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation
    logger.propagate = False
    
    return logger


def setup_audit_logging(
    name: str,
    log_file: str = "logs/audit.log"
) -> logging.Logger:
    """
    Setup audit logging for security and compliance.
    
    Args:
        name: Logger name
        log_file: Audit log file path
        
    Returns:
        Audit logger instance
    """
    
    logger = logging.getLogger(f"{name}.audit")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Audit-specific format
    audit_format = (
        "%(asctime)s - AUDIT - %(message)s"
    )
    
    formatter = logging.Formatter(audit_format)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation
    logger.propagate = False
    
    return logger


# Convenience function for common setup
def setup_agentic_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for the Agentic microservice with common defaults.
    
    Args:
        level: Logging level
        structured: Whether to use structured JSON logging
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    
    if structured:
        logger = setup_structured_logging("agentic", level)
    else:
        logger = setup_logging("agentic", level, log_file=log_file)
    
    # Configure third-party loggers
    configure_third_party_loggers("WARNING")
    
    # Add correlation ID filter
    add_correlation_id_filter(logger)
    
    return logger