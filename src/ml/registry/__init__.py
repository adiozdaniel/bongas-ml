"""
Model Registry for BONGAS-ML

This package provides:
- ModelRegistryClient: Client for bongas-server model registry
- Model upload/download with metadata management
- Version control and staging workflows
- Customer-specific model handling
"""

from .client import ModelRegistryClient

__all__ = [
    "ModelRegistryClient",
]