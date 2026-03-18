"""
Training pipeline for BONGAS-ML models

This package provides:
- CustomerModelTrainer: Main training orchestrator
- TrainingDataLoader: Data loading and preprocessing
- TrainingCallbacks: Training hooks and callbacks
- LossFunctions: Custom loss functions for recommendation
"""

from .trainer import CustomerModelTrainer
from .datasets import (
    TrainingDataset, 
    NegativeSamplingDataset,
    SequentialDataset,
    PairwiseDataset,
    DataProcessor,
    create_data_loader
)
from .callbacks import (
    TrainingCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LRLoggingCallback,
    MetricsLoggingCallback,
    ProgressBarCallback,
    TensorBoardCallback,
    GradientClippingCallback,
    ReduceLROnPlateauCallback,
    TrainingHistoryCallback
)
from .losses import (
    BCEWithLogitsLoss,
    PairwiseHingeLoss, 
    ListMLELoss,
    WeightedMSELoss,
    FocalLoss,
    ContrastiveLoss,
    TripletLoss,
    CustomLoss,
    get_loss_function
)

__all__ = [
    # Trainer
    "CustomerModelTrainer",
    
    # Datasets
    "TrainingDataset",
    "NegativeSamplingDataset", 
    "SequentialDataset",
    "PairwiseDataset",
    "DataProcessor",
    "create_data_loader",
    
    # Callbacks
    "TrainingCallback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LRLoggingCallback",
    "MetricsLoggingCallback",
    "ProgressBarCallback",
    "TensorBoardCallback",
    "GradientClippingCallback",
    "ReduceLROnPlateauCallback",
    "TrainingHistoryCallback",
    
    # Losses
    "BCEWithLogitsLoss",
    "PairwiseHingeLoss",
    "ListMLELoss",
    "WeightedMSELoss",
    "FocalLoss",
    "ContrastiveLoss",
    "TripletLoss",
    "CustomLoss",
    "get_loss_function",
]
