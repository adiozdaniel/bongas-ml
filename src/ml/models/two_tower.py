"""
Two-Tower Model for BONGAS-ML

Dual encoder architecture for user-item matching.
Each tower encodes user/item features into embeddings,
then computes similarity scores for recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .base import BaseModel


class TwoTowerModel(BaseModel):
    """Two-Tower Neural Network for recommendation systems"""
    
    def __init__(
        self,
        user_feature_dim: int = 128,
        item_feature_dim: int = 128,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        activation: str = 'relu',
        normalize_embeddings: bool = True
    ):
        """
        Initialize Two-Tower Model
        
        Args:
            user_feature_dim: Dimension of user features
            item_feature_dim: Dimension of item features
            embedding_dim: Dimension of output embeddings
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'gelu')
            normalize_embeddings: Whether to normalize embeddings
        """
        super().__init__()
        
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # User tower
        self.user_tower = self._build_tower(
            input_dim=user_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Item tower
        self.item_tower = self._build_tower(
            input_dim=item_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"TwoTowerModel initialized with:")
        logger.info(f"  User features: {user_feature_dim}")
        logger.info(f"  Item features: {item_feature_dim}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Normalize embeddings: {normalize_embeddings}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module"""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _build_tower(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Sequential:
        """Build a tower network"""
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        user_features: torch.Tensor, 
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            user_features: User feature tensor [batch_size, user_feature_dim]
            item_features: Item feature tensor [batch_size, item_feature_dim]
        
        Returns:
            Similarity scores [batch_size, 1]
        """
        # Get embeddings
        user_embeddings = self.get_user_embeddings(user_features)
        item_embeddings = self.get_item_embeddings(item_features)
        
        # Compute similarity scores
        if self.normalize_embeddings:
            # Cosine similarity
            scores = F.cosine_similarity(user_embeddings, item_embeddings, dim=1, eps=1e-8)
        else:
            # Dot product similarity
            scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        return scores.unsqueeze(1)  # [batch_size, 1]
    
    def get_user_embeddings(self, user_features: torch.Tensor) -> torch.Tensor:
        """Get user embeddings"""
        embeddings = self.user_tower(user_features)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_item_embeddings(self, item_features: torch.Tensor) -> torch.Tensor:
        """Get item embeddings"""
        embeddings = self.item_tower(item_features)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_similarity_matrix(
        self, 
        user_features: torch.Tensor, 
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between all users and items
        
        Args:
            user_features: User features [num_users, user_feature_dim]
            item_features: Item features [num_items, item_feature_dim]
        
        Returns:
            Similarity matrix [num_users, num_items]
        """
        user_embeddings = self.get_user_embeddings(user_features)
        item_embeddings = self.get_item_embeddings(item_features)
        
        if self.normalize_embeddings:
            # Cosine similarity matrix
            similarity_matrix = torch.mm(user_embeddings, item_embeddings.t())
        else:
            # Dot product similarity matrix
            similarity_matrix = torch.mm(user_embeddings, item_embeddings.t())
        
        return similarity_matrix
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'user_feature_dim': self.user_feature_dim,
            'item_feature_dim': self.item_feature_dim,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': [layer.out_features for layer in self.user_tower if hasattr(layer, 'out_features')],
            'dropout': [layer.p for layer in self.user_tower if hasattr(layer, 'p')][0] if any(hasattr(layer, 'p') for layer in self.user_tower) else 0.2,
            'normalize_embeddings': self.normalize_embeddings,
            'activation': self.activation.__class__.__name__
        }
    
    def save_config(self, path: Union[str, Path]) -> None:
        """Save model configuration"""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.get_model_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model config saved to {path}")
    
    @classmethod
    def load_config(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model configuration"""
        import json
        from pathlib import Path
        
        path = Path(path)
        with open(path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def freeze_user_tower(self) -> None:
        """Freeze user tower parameters"""
        for param in self.user_tower.parameters():
            param.requires_grad = False
        logger.info("User tower frozen")
    
    def freeze_item_tower(self) -> None:
        """Freeze item tower parameters"""
        for param in self.item_tower.parameters():
            param.requires_grad = False
        logger.info("Item tower frozen")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")
    
    def get_tower_parameters(self, tower: str = 'user') -> List[nn.Parameter]:
        """Get parameters from specific tower"""
        if tower == 'user':
            return list(self.user_tower.parameters())
        elif tower == 'item':
            return list(self.item_tower.parameters())
        else:
            raise ValueError(f"Unknown tower: {tower}")
    
    def get_parameter_stats(self) -> Dict[str, float]:
        """Get parameter statistics"""
        stats = {}
        
        # User tower stats
        user_params = self.get_tower_parameters('user')
        stats['user_tower_params'] = sum(p.numel() for p in user_params)
        stats['user_tower_trainable'] = sum(p.numel() for p in user_params if p.requires_grad)
        
        # Item tower stats
        item_params = self.get_tower_parameters('item')
        stats['item_tower_params'] = sum(p.numel() for p in item_params)
        stats['item_tower_trainable'] = sum(p.numel() for p in item_params if p.requires_grad)
        
        # Overall stats
        stats['total_params'] = stats['user_tower_params'] + stats['item_tower_params']
        stats['total_trainable'] = stats['user_tower_trainable'] + stats['item_tower_trainable']
        
        return stats