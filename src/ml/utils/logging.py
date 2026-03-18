"""
Logging utilities for BONGAS-ML

Provides centralized logging configuration and utilities for the entire package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = '10 MB',
    retention: str = '30 days',
    compression: str = 'zip',
    format: Optional[str] = None
) -> None:
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        rotation: Log rotation size
        retention: Log retention period
        compression: Log compression format
        format: Custom log format
    """
    
    # Remove default loguru handler
    logger.remove()
    
    # Set log format
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=format,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True  # Thread-safe
        )
    
    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str):
    """Get a logger instance with the specified name"""
    return logger


def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = logger.opt(depth=1).info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = logger.opt(depth=1).info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.opt(depth=1).error(f"Failed {func.__name__}: {e}")
            raise
    return wrapper


class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(
        self,
        level: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        format: Optional[str] = None
    ):
        self.level = level
        self.log_file = log_file
        self.format = format
        self.old_config = None
    
    def __enter__(self):
        # Save current configuration
        self.old_config = logger._core.handlers.copy()
        
        # Apply new configuration
        setup_logging(
            level=self.level or 'INFO',
            log_file=self.log_file,
            format=self.format
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old configuration
        logger._core.handlers = self.old_config


def configure_package_logging():
    """Configure logging for the entire bongas_ml package"""
    
    # Configure root logger
    setup_logging(
        level='INFO',
        log_file=None,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )
    
    # Configure specific loggers
    logger.level("TRAINING", no=25, color="<yellow>", icon="🚀")
    logger.level("VALIDATION", no=26, color="<green>", icon="✅")
    logger.level("EXPORT", no=27, color="<blue>", icon="📦")
    logger.level("REGISTRY", no=28, color="<magenta>", icon="🗄️")
    
    logger.info("BONGAS-ML package logging configured")


# Initialize package logging when module is imported
configure_package_logging()