"""
Accuracy validation for BONGAS-ML

Provides accuracy validation gates with configurable thresholds
for model quality assurance.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from ..models.base import BaseModel


class AccuracyValidator:
    """Validates model accuracy against configurable thresholds"""
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize accuracy validator
        
        Args:
            thresholds: Dictionary of metric thresholds
        """
        self.thresholds = thresholds or {
            'min_accuracy': 0.70,
            'min_ndcg_at_10': 0.50,
            'min_hit_rate_at_10': 0.60,
            'max_loss': 1.0
        }
        
        logger.info(f"Accuracy validator initialized with thresholds: {self.thresholds}")
    
    def validate(
        self,
        model: BaseModel,
        test_data: Dict[str, torch.Tensor],
        batch_size: int = 256
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate model accuracy against thresholds
        
        Args:
            model: Trained model to validate
            test_data: Test dataset
            batch_size: Batch size for evaluation
        
        Returns:
            Tuple of (validation_passed, metrics)
        """
        
        try:
            model.eval()
            device = model.device
            
            # Create dataset and dataloader
            from ..training.datasets import TrainingDataset
            dataset = TrainingDataset(test_data)
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Evaluate model
            total_loss = 0.0
            all_predictions = []
            all_targets = []
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    user_features = batch['user_features'].to(device)
                    item_features = batch['item_features'].to(device)
                    targets = batch['targets'].to(device)
                    
                    predictions = model(user_features, item_features)
                    
                    # Calculate loss
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(predictions, targets.unsqueeze(1))
                    total_loss += loss.item()
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    num_batches += 1
            
            # Calculate metrics
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            
            metrics = self._calculate_metrics(all_predictions, all_targets)
            avg_loss = total_loss / num_batches
            metrics['loss'] = avg_loss
            
            # Validate against thresholds
            passed = self._validate_thresholds(metrics)
            
            logger.info(f"Accuracy validation {'PASSED' if passed else 'FAILED'}")
            for metric, value in metrics.items():
                threshold = self.thresholds.get(f'min_{metric}', self.thresholds.get(f'max_{metric}'))
                if threshold:
                    status = "✓" if self._check_metric(metric, value, threshold) else "✗"
                    logger.info(f"  {metric}: {value:.4f} {status} (threshold: {threshold})")
            
            return passed, metrics
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            return False, {}
    
    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate validation metrics"""
        
        metrics = {}
        
        # Convert to numpy for metric calculation
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        
        # Binary classification metrics
        if predictions_np.ndim == 1:
            # Convert to binary predictions
            binary_predictions = (predictions_np > 0.5).astype(int)
            
            # Accuracy
            accuracy = (binary_predictions == targets_np).mean()
            metrics['accuracy'] = float(accuracy)
            
            # Precision, Recall, F1
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets_np, binary_predictions, average='binary', zero_division=0
            )
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
        
        else:
            # Ranking metrics
            # Hit Rate at K
            for k in [5, 10, 20]:
                hit_rate = self._calculate_hit_rate(predictions_np, targets_np, k)
                metrics[f'hit_rate_at_{k}'] = hit_rate
            
            # NDCG at K
            for k in [5, 10, 20]:
                ndcg = self._calculate_ndcg(predictions_np, targets_np, k)
                metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    def _calculate_hit_rate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """Calculate Hit Rate at K"""
        # Get top-K predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Check if target is in top-K
        hit_mask = torch.zeros_like(predictions, dtype=torch.bool)
        for i, target in enumerate(targets):
            hit_mask[i, top_k_indices[i]] = True
        
        hit_rate = hit_mask[torch.arange(len(targets)), targets].float().mean().item()
        return hit_rate
    
    def _calculate_ndcg(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
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
    
    def _validate_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics against thresholds"""
        
        passed = True
        
        for metric, value in metrics.items():
            # Check minimum thresholds
            min_key = f'min_{metric}'
            if min_key in self.thresholds:
                threshold = self.thresholds[min_key]
                if value < threshold:
                    logger.warning(f"Metric {metric} ({value:.4f}) below threshold ({threshold})")
                    passed = False
            
            # Check maximum thresholds
            max_key = f'max_{metric}'
            if max_key in self.thresholds:
                threshold = self.thresholds[max_key]
                if value > threshold:
                    logger.warning(f"Metric {metric} ({value:.4f}) above threshold ({threshold})")
                    passed = False
        
        return passed
    
    def _check_metric(
        self,
        metric: str,
        value: float,
        threshold: float
    ) -> bool:
        """Check if metric meets threshold"""
        
        if metric.startswith('min_'):
            return value >= threshold
        elif metric.startswith('max_'):
            return value <= threshold
        else:
            return True
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update validation thresholds"""
        
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated validation thresholds: {self.thresholds}")
    
    def get_threshold_report(self) -> Dict[str, Any]:
        """Get current threshold configuration"""
        
        return {
            'thresholds': self.thresholds,
            'description': 'Validation thresholds for model accuracy gates',
            'updated_at': str(time.time())
        }