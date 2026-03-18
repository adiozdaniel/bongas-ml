"""
Baseline comparison validation for BONGAS-ML

Compares new models against baseline models to ensure improvements
and prevent regression in model performance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from ..models.base import BaseModel
from ..registry.client import ModelRegistryClient


class BaselineComparator:
    """Compares new models against baseline models"""
    
    def __init__(
        self,
        registry_client: Optional[ModelRegistryClient] = None,
        improvement_threshold: float = 0.02,
        metrics_to_compare: Optional[List[str]] = None
    ):
        """
        Initialize baseline comparator
        
        Args:
            registry_client: Model registry client for baseline retrieval
            improvement_threshold: Minimum improvement required (e.g., 0.02 = 2%)
            metrics_to_compare: List of metrics to compare
        """
        self.registry_client = registry_client
        self.improvement_threshold = improvement_threshold
        self.metrics_to_compare = metrics_to_compare or [
            'accuracy', 'ndcg_at_10', 'hit_rate_at_10'
        ]
        
        logger.info(f"Baseline comparator initialized:")
        logger.info(f"  Improvement threshold: {improvement_threshold}")
        logger.info(f"  Metrics: {self.metrics_to_compare}")
    
    def compare(
        self,
        new_model: BaseModel,
        baseline_path: Optional[Union[str, Path]] = None,
        customer_id: Optional[str] = None,
        test_data: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 256
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare new model against baseline
        
        Args:
            new_model: New model to evaluate
            baseline_path: Path to baseline model file
            customer_id: Customer ID for registry lookup
            test_data: Test dataset
            batch_size: Batch size for evaluation
        
        Returns:
            Tuple of (comparison_passed, comparison_results)
        """
        
        try:
            # Get baseline model
            baseline_model = self._get_baseline_model(baseline_path, customer_id)
            if baseline_model is None:
                logger.warning("No baseline model found, skipping comparison")
                return True, {'skipped': True, 'reason': 'no_baseline'}
            
            # Evaluate both models
            logger.info("Evaluating new model...")
            new_metrics = self._evaluate_model(new_model, test_data, batch_size)
            
            logger.info("Evaluating baseline model...")
            baseline_metrics = self._evaluate_model(baseline_model, test_data, batch_size)
            
            # Compare metrics
            comparison_results = self._compare_metrics(new_metrics, baseline_metrics)
            
            # Determine if comparison passes
            passed = self._check_improvement(comparison_results)
            
            logger.info(f"Baseline comparison {'PASSED' if passed else 'FAILED'}")
            
            return passed, comparison_results
            
        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            return False, {'error': str(e)}
    
    def _get_baseline_model(
        self,
        baseline_path: Optional[Union[str, Path]],
        customer_id: Optional[str]
    ) -> Optional[BaseModel]:
        """Get baseline model from file or registry"""
        
        if baseline_path:
            # Load from local file
            baseline_path = Path(baseline_path)
            if baseline_path.exists():
                logger.info(f"Loading baseline model from {baseline_path}")
                return BaseModel.load_from_path(baseline_path)
            else:
                logger.warning(f"Baseline model file not found: {baseline_path}")
                return None
        
        elif customer_id and self.registry_client:
            # Load from registry
            logger.info(f"Loading baseline model for customer {customer_id} from registry")
            baseline_path = self.registry_client.download_latest_model(
                customer_id=customer_id,
                stage='production'
            )
            if baseline_path:
                return BaseModel.load_from_path(baseline_path)
        
        return None
    
    def _evaluate_model(
        self,
        model: BaseModel,
        test_data: Optional[Dict[str, torch.Tensor]],
        batch_size: int
    ) -> Dict[str, float]:
        """Evaluate model on test data"""
        
        if test_data is None:
            logger.warning("No test data provided for baseline comparison")
            return {}
        
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
        
        return metrics
    
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
    
    def _compare_metrics(
        self,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare metrics between new and baseline models"""
        
        comparison = {
            'new_metrics': new_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': {},
            'regressions': {},
            'unchanged': {}
        }
        
        for metric in self.metrics_to_compare:
            if metric in new_metrics and metric in baseline_metrics:
                new_value = new_metrics[metric]
                baseline_value = baseline_metrics[metric]
                
                if baseline_value == 0:
                    # Avoid division by zero
                    improvement_pct = float('inf') if new_value > 0 else 0.0
                else:
                    improvement_pct = (new_value - baseline_value) / baseline_value
                
                result = {
                    'new_value': new_value,
                    'baseline_value': baseline_value,
                    'absolute_difference': new_value - baseline_value,
                    'relative_difference': improvement_pct
                }
                
                # Categorize result
                if improvement_pct >= self.improvement_threshold:
                    comparison['improvements'][metric] = result
                elif improvement_pct <= -self.improvement_threshold:
                    comparison['regressions'][metric] = result
                else:
                    comparison['unchanged'][metric] = result
        
        return comparison
    
    def _check_improvement(self, comparison_results: Dict[str, Any]) -> bool:
        """Check if improvements meet requirements"""
        
        # Check for critical regressions
        critical_metrics = ['accuracy', 'ndcg_at_10']
        for metric in critical_metrics:
            if metric in comparison_results['regressions']:
                regression = comparison_results['regressions'][metric]
                if abs(regression['relative_difference']) > 0.05:  # 5% regression
                    logger.error(f"Critical regression in {metric}: {regression['relative_difference']:.2%}")
                    return False
        
        # Check for required improvements
        required_improvements = len(comparison_results['improvements'])
        total_comparable_metrics = len(comparison_results['improvements']) + len(comparison_results['regressions'])
        
        if total_comparable_metrics > 0:
            improvement_rate = required_improvements / total_comparable_metrics
            
            # Require at least some improvements or no significant regressions
            if improvement_rate < 0.3 and len(comparison_results['regressions']) > 0:
                logger.warning(f"Low improvement rate: {improvement_rate:.2%}")
                return False
        
        return True
    
    def set_improvement_threshold(self, threshold: float) -> None:
        """Set improvement threshold"""
        self.improvement_threshold = threshold
        logger.info(f"Set improvement threshold to {threshold}")
    
    def set_metrics_to_compare(self, metrics: List[str]) -> None:
        """Set metrics to compare"""
        self.metrics_to_compare = metrics
        logger.info(f"Set metrics to compare: {metrics}")
    
    def get_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate comparison report"""
        
        report_lines = [
            "# Baseline Comparison Report",
            "",
            "## Summary",
        ]
        
        if comparison_results.get('skipped'):
            report_lines.append(f"- **Status**: SKIPPED ({comparison_results['reason']})")
            return "\n".join(report_lines)
        
        # Count results
        num_improvements = len(comparison_results['improvements'])
        num_regressions = len(comparison_results['regressions'])
        num_unchanged = len(comparison_results['unchanged'])
        
        report_lines.extend([
            f"- **Improvements**: {num_improvements}",
            f"- **Regressions**: {num_regressions}",
            f"- **Unchanged**: {num_unchanged}",
            "",
            "## Improvements",
        ])
        
        for metric, result in comparison_results['improvements'].items():
            report_lines.append(
                f"- **{metric}**: {result['new_value']:.4f} "
                f"(+{result['relative_difference']:.2%})"
            )
        
        if comparison_results['regressions']:
            report_lines.extend([
                "",
                "## Regressions",
            ])
            for metric, result in comparison_results['regressions'].items():
                report_lines.append(
                    f"- **{metric}**: {result['new_value']:.4f} "
                    f"({result['relative_difference']:.2%})"
                )
        
        if comparison_results['unchanged']:
            report_lines.extend([
                "",
                "## Unchanged",
            ])
            for metric, result in comparison_results['unchanged'].items():
                report_lines.append(
                    f"- **{metric}**: {result['new_value']:.4f} "
                    f"({result['relative_difference']:.2%})"
                )
        
        return "\n".join(report_lines)