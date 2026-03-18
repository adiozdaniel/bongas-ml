"""
Data loading and preprocessing for BONGAS-ML

This package provides:
- TrainingDataLoader: PostgreSQL data loader with feature engineering
- Data validation and preprocessing utilities
- Customer-specific data handling
"""

from .loader import TrainingDataLoader

__all__ = [
    "TrainingDataLoader",
]