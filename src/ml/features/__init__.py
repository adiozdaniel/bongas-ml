"""
Feature Engineering for BONGAS-ML

This package provides:
- FeatureExtractor: Automatic feature extraction from raw data
- FeatureTransformer: Feature scaling, normalization, and encoding
- EmbeddingManager: Embedding layer management and optimization
"""

from .extractors import FeatureExtractor
from .transformers import FeatureTransformer
from .embeddings import EmbeddingManager

__all__ = [
    "FeatureExtractor",
    "FeatureTransformer",
    "EmbeddingManager",
]