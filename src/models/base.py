"""
Base model class for all BONGAS-ML models

Provides common functionality for model training, evaluation, and serialization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger


class BaseModel(nn.Module, ABC):
    """Abstract base class for all recommendation models"""
    
    def __init__(self):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_trained = False
        self._training_history = []
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on"""
        return self._device
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained"""
        return self._is_trained
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseModel':
        """Move model to specified device"""
        if device is None:
            device = self._device
        self._device = device
        self.to(device)
        return self
    
    @abstractmethod
    def forward(
        self, 
        user_features: torch.Tensor, 
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def get_user_embeddings(self, user_features: torch.Tensor) -> torch.Tensor:
        """Get user embeddings from the model"""
        pass
    
    @abstractmethod
    def get_item_embeddings(self, item_features: torch.Tensor) -> torch.Tensor:
        """Get item embeddings from the model"""
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model state dict"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'device': self._device.type,
            'is_trained': self._is_trained,
            'training_history': self._training_history
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> 'BaseModel':
        """Load model from state dict"""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device)
        
        # Verify model class compatibility
        if checkpoint['model_class'] != self.__class__.__name__:
            logger.warning(f"Model class mismatch: expected {self.__class__.__name__}, "
                          f"got {checkpoint['model_class']}")
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint.get('is_trained', False)
        self._training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Model loaded from {path}")
        return self
    
    @classmethod
    def load_from_path(cls, path: Union[str, Path]) -> 'BaseModel':
        """Load model from path, creating appropriate instance"""
        checkpoint = torch.load(path, map_location='cpu')
        model_class_name = checkpoint['model_class']
        
        # Create instance of appropriate class
        if model_class_name == 'TwoTowerModel':
            from .two_tower import TwoTowerModel
            model = TwoTowerModel()
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")
        
        return model.load(path)
    
    def get_model_size(self) -> int:
        """Get model size in bytes"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_class': self.__class__.__name__,
            'device': self._device.type,
            'is_trained': self._is_trained,
            'model_size_bytes': self.get_model_size(),
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'training_history': self._training_history
        }
    
    def set_training_mode(self, mode: bool = True) -> None:
        """Set training mode for the model"""
        self.train(mode)
        self._is_trained = not mode
    
    def evaluate_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        k: int = 10
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Convert to numpy for metric calculation
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy
        if predictions.ndim == 1:
            # Binary classification
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = (predictions_binary == targets).mean()
            metrics['accuracy'] = float(accuracy)
        else:
            # Ranking metrics
            # Hit Rate at K
            hit_rate = self._calculate_hit_rate(predictions, targets, k)
            metrics[f'hit_rate_at_{k}'] = hit_rate
            
            # NDCG at K
            ndcg = self._calculate_ndcg(predictions, targets, k)
            metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    def _calculate_hit_rate(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Calculate Hit Rate at K"""
        # Get top-K predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Check if target is in top-K
        hit_mask = torch.zeros_like(predictions, dtype=torch.bool)
        for i, target in enumerate(targets):
            hit_mask[i, top_k_indices[i]] = True
        
        hit_rate = hit_mask[torch.arange(len(targets)), targets].float().mean().item()
        return hit_rate
    
    def _calculate_ndcg(self, predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Calculate NDCG at K"""
        # Get top-K predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Calculate DCG
        dcg = torch.zeros(len(targets))
        for i in range(len(targets)):
            target_item = targets[i]
            if target_item in top_k_indices[i]:
                rank = (top_k_indices[i] == target_item).nonzero(as_tuple=True)[0].item() + 1
                dcg[i] = 1.0 / torch.log2(torch.tensor(rank + 1.0))
        
        # Calculate IDCG (ideal DCG)
        idcg = torch.ones(len(targets))  # Perfect ranking would have DCG = 1
        
        ndcg = (dcg / idcg).mean().item()
        return ndcg
    
    def get_recommendations(
        self, 
        user_features: torch.Tensor, 
        item_features: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """Get top-K recommendations for users"""
        self.eval()
        
        with torch.no_grad():
            scores = self.forward(user_features, item_features)
            _, top_k_indices = torch.topk(scores, k, dim=1)
        
        return top_k_indices
    
    def warm_start(self, pretrained_path: Union[str, Path]) -> None:
        """Initialize model with pretrained weights"""
        logger.info(f"Warm starting model from {pretrained_path}")
        self.load(pretrained_path)
    
    def reset_parameters(self) -> None:
        """Reset model parameters to random initialization"""
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        self._is_trained = False
        self._training_history = []
        logger.info("Model parameters reset")