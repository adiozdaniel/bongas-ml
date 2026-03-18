"""
Custom loss functions for BONGAS-ML

Provides various loss functions optimized for recommendation systems:
- BCEWithLogitsLoss: Binary cross-entropy for implicit feedback
- PairwiseHingeLoss: Hinge loss for pairwise ranking
- ListMLELoss: List-wise ranking loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy with logits loss for implicit feedback"""
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize BCE loss
        
        Args:
            pos_weight: Weight for positive samples (for imbalanced data)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss
        
        Args:
            predictions: Model predictions [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size, 1] or [batch_size]
        
        Returns:
            Loss value
        """
        # Ensure proper shape
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        return self.bce_loss(predictions, targets)


class PairwiseHingeLoss(nn.Module):
    """Pairwise hinge loss for learning to rank"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize pairwise hinge loss
        
        Args:
            margin: Margin for hinge loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise hinge loss
        
        Args:
            positive_scores: Scores for positive items [batch_size]
            negative_scores: Scores for negative items [batch_size]
        
        Returns:
            Loss value
        """
        # Compute margin-based loss
        loss = torch.clamp(self.margin - positive_scores + negative_scores, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ListMLELoss(nn.Module):
    """List-wise ranking loss (ListMLE)"""
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize ListMLE loss
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ListMLE loss
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Ground truth relevance [batch_size, num_items]
        
        Returns:
            Loss value
        """
        # Sort predictions and targets by relevance
        _, sorted_indices = torch.sort(targets, dim=1, descending=True)
        
        sorted_predictions = torch.gather(predictions, 1, sorted_indices)
        
        # Compute ListMLE loss
        log_likelihoods = []
        for i in range(predictions.size(0)):  # For each sample in batch
            pred = sorted_predictions[i]  # [num_items]
            
            # Compute log-likelihood for this permutation
            log_likelihood = 0.0
            for j in range(len(pred)):
                # Log-softmax of remaining items
                remaining_scores = pred[j:]
                log_softmax = F.log_softmax(remaining_scores, dim=0)
                log_likelihood += log_softmax[0]  # First item has highest relevance
            
            log_likelihoods.append(log_likelihood)
        
        log_likelihoods = torch.stack(log_likelihoods)
        loss = -log_likelihoods
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error loss"""
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted MSE loss
        
        Args:
            weights: Sample weights [batch_size]
        """
        super().__init__()
        self.weights = weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss
        
        Args:
            predictions: Model predictions [batch_size]
            targets: Ground truth values [batch_size]
        
        Returns:
            Loss value
        """
        mse = (predictions - targets) ** 2
        
        if self.weights is not None:
            mse = mse * self.weights
        
        return mse.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
            labels: Similarity labels (1 for similar, 0 for dissimilar)
        
        Returns:
            Loss value
        """
        # Compute Euclidean distance
        distances = torch.norm(embeddings1 - embeddings2, dim=1)
        
        # Compute contrastive loss
        loss = 0.5 * (
            labels * distances ** 2 +
            (1 - labels) * torch.clamp(self.margin - distances, min=0.0) ** 2
        )
        
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive embeddings [batch_size, embedding_dim]
            negative_embeddings: Negative embeddings [batch_size, embedding_dim]
        
        Returns:
            Loss value
        """
        # Compute distances
        pos_distances = torch.norm(anchor_embeddings - positive_embeddings, dim=1)
        neg_distances = torch.norm(anchor_embeddings - negative_embeddings, dim=1)
        
        # Compute triplet loss
        loss = torch.clamp(pos_distances - neg_distances + self.margin, min=0.0)
        
        return loss.mean()


class CustomLoss(nn.Module):
    """Custom composite loss function"""
    
    def __init__(
        self,
        loss_type: str = 'bce',
        **kwargs
    ):
        """
        Initialize custom loss
        
        Args:
            loss_type: Type of loss ('bce', 'hinge', 'listmle', etc.)
            **kwargs: Additional loss parameters
        """
        super().__init__()
        
        if loss_type == 'bce':
            self.loss_fn = BCEWithLogitsLoss(**kwargs)
        elif loss_type == 'hinge':
            self.loss_fn = PairwiseHingeLoss(**kwargs)
        elif loss_type == 'listmle':
            self.loss_fn = ListMLELoss(**kwargs)
        elif loss_type == 'mse':
            self.loss_fn = WeightedMSELoss(**kwargs)
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(**kwargs)
        elif loss_type == 'contrastive':
            self.loss_fn = ContrastiveLoss(**kwargs)
        elif loss_type == 'triplet':
            self.loss_fn = TripletLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""
        return self.loss_fn(*args, **kwargs)


# Convenience functions for common loss combinations
def get_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """Get loss function by name"""
    
    if loss_type == 'bce':
        return BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'hinge':
        return PairwiseHingeLoss(**kwargs)
    elif loss_type == 'listmle':
        return ListMLELoss(**kwargs)
    elif loss_type == 'mse':
        return WeightedMSELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif loss_type == 'triplet':
        return TripletLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")