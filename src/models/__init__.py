"""
ML Model implementations for BONGAS-ML

This package contains various recommendation model architectures:
- TwoTowerModel: Dual encoder architecture for user-item matching
- BERT4RecModel: Sequential recommendation with transformer
- NCFModel: Neural Collaborative Filtering
- DINModel: Deep Interest Network
- WideAndDeepModel: Wide & Deep learning
- AutoIntModel: Automatic feature interaction
"""

from .base import BaseModel
from .two_tower import TwoTowerModel

__all__ = [
    "BaseModel",
    "TwoTowerModel",
]
