"""
BONGAS-ML: Machine Learning Package for BONGAS-AI

This package provides:
- Model training and evaluation
- Feature engineering and preprocessing
- Model export and optimization
- Model registry and deployment
- Validation and quality assurance
"""

# Version
__version__ = "1.0.0"

# Core modules
from .models import BaseModel, TwoTowerModel
from .training import Trainer, TrainingDataset, ValidationCallback
from .features import FeatureExtractor, FeatureTransformer
from .export import ONNXExporter, optimize_onnx_model, validate_onnx_model
from .registry import ModelRegistryClient
from .validation import AccuracyValidator, BaselineComparator, ShadowTester
from .utils import setup_logging, get_logger

# Re-export commonly used items
__all__ = [
    # Models
    "BaseModel",
    "TwoTowerModel",
    
    # Training
    "Trainer",
    "TrainingDataset", 
    "ValidationCallback",
    
    # Features
    "FeatureExtractor",
    "FeatureTransformer",
    
    # Export
    "ONNXExporter",
    "optimize_onnx_model",
    "validate_onnx_model",
    
    # Registry
    "ModelRegistryClient",
    
    # Validation
    "AccuracyValidator",
    "BaselineComparator", 
    "ShadowTester",
    
    # Utils
    "setup_logging",
    "get_logger",
]