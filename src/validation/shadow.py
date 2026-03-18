"""
Shadow testing validation for BONGAS-ML

Implements shadow testing to validate new models against production models
in a live environment without affecting user experience.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from ..models.base import BaseModel
from ..registry.client import ModelRegistryClient


class ShadowTester:
    """Implements shadow testing for model validation"""
    
    def __init__(
        self,
        registry_client: Optional[ModelRegistryClient] = None,
        shadow_duration: int = 3600,  # 1 hour
        sample_rate: float = 0.1,     # 10% of traffic
        metrics_window: int = 1000    # Track metrics over N requests
    ):
        """
        Initialize shadow tester
        
        Args:
            registry_client: Model registry client
            shadow_duration: Duration of shadow testing in seconds
            sample_rate: Fraction of traffic to shadow (0.0 to 1.0)
            metrics_window: Number of requests to track for metrics
        """
        self.registry_client = registry_client
        self.shadow_duration = shadow_duration
        self.sample_rate = sample_rate
        self.metrics_window = metrics_window
        
        # Shadow testing state
        self.shadow_start_time = None
        self.shadow_active = False
        self.metrics_buffer = []
        
        logger.info(f"Shadow tester initialized:")
        logger.info(f"  Duration: {shadow_duration}s")
        logger.info(f"  Sample rate: {sample_rate}")
        logger.info(f"  Metrics window: {metrics_window}")
    
    def start_shadow_testing(
        self,
        new_model: BaseModel,
        customer_id: str,
        baseline_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Start shadow testing
        
        Args:
            new_model: New model to shadow
            customer_id: Customer ID
            baseline_path: Path to baseline model
        
        Returns:
            True if shadow testing started successfully
        """
        
        try:
            # Get baseline model
            baseline_model = self._get_baseline_model(baseline_path, customer_id)
            if baseline_model is None:
                logger.error("No baseline model available for shadow testing")
                return False
            
            self.new_model = new_model
            self.baseline_model = baseline_model
            self.customer_id = customer_id
            
            # Start shadow testing
            self.shadow_start_time = time.time()
            self.shadow_active = True
            self.metrics_buffer = []
            
            logger.info(f"Shadow testing started for customer {customer_id}")
            logger.info(f"  New model: {type(new_model).__name__}")
            logger.info(f"  Baseline model: {type(baseline_model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start shadow testing: {e}")
            return False
    
    def stop_shadow_testing(self) -> Dict[str, Any]:
        """
        Stop shadow testing and return results
        
        Returns:
            Shadow testing results
        """
        
        if not self.shadow_active:
            logger.warning("Shadow testing not active")
            return {'error': 'not_active'}
        
        duration = time.time() - self.shadow_start_time
        results = {
            'duration': duration,
            'total_requests': len(self.metrics_buffer),
            'metrics': self._calculate_shadow_metrics(),
            'recommendation': self._get_recommendation()
        }
        
        self.shadow_active = False
        self.shadow_start_time = None
        self.metrics_buffer = []
        
        logger.info(f"Shadow testing stopped after {duration:.2f}s")
        logger.info(f"  Total requests: {results['total_requests']}")
        logger.info(f"  Recommendation: {results['recommendation']}")
        
        return results
    
    def shadow_request(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Shadow a single request through both models
        
        Args:
            user_features: User features tensor
            item_features: Item features tensor
            targets: Optional target values for comparison
        
        Returns:
            Shadow testing results for this request
        """
        
        if not self.shadow_active:
            return None
        
        try:
            # Check if we should sample this request
            import random
            if random.random() > self.sample_rate:
                return None
            
            # Run both models
            with torch.no_grad():
                new_prediction = self.new_model(user_features, item_features)
                baseline_prediction = self.baseline_model(user_features, item_features)
            
            # Calculate differences
            prediction_diff = torch.abs(new_prediction - baseline_prediction).mean().item()
            
            result = {
                'timestamp': time.time(),
                'prediction_diff': prediction_diff,
                'new_prediction': new_prediction.item(),
                'baseline_prediction': baseline_prediction.item(),
                'target': targets.item() if targets is not None else None
            }
            
            # Add accuracy metrics if targets available
            if targets is not None:
                new_accuracy = self._calculate_accuracy(new_prediction, targets)
                baseline_accuracy = self._calculate_accuracy(baseline_prediction, targets)
                result.update({
                    'new_accuracy': new_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'accuracy_diff': abs(new_accuracy - baseline_accuracy)
                })
            
            # Store in buffer
            self.metrics_buffer.append(result)
            
            # Keep buffer size manageable
            if len(self.metrics_buffer) > self.metrics_window:
                self.metrics_buffer.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Shadow request failed: {e}")
            return None
    
    def is_shadow_testing_complete(self) -> bool:
        """Check if shadow testing duration has completed"""
        
        if not self.shadow_active:
            return False
        
        elapsed = time.time() - self.shadow_start_time
        return elapsed >= self.shadow_duration
    
    def get_shadow_status(self) -> Dict[str, Any]:
        """Get current shadow testing status"""
        
        if not self.shadow_active:
            return {
                'active': False,
                'message': 'Shadow testing not active'
            }
        
        elapsed = time.time() - self.shadow_start_time
        progress = min(elapsed / self.shadow_duration, 1.0)
        
        return {
            'active': True,
            'elapsed_time': elapsed,
            'duration': self.shadow_duration,
            'progress': progress,
            'sampled_requests': len(self.metrics_buffer),
            'customer_id': self.customer_id
        }
    
    def _get_baseline_model(
        self,
        baseline_path: Optional[Union[str, Path]],
        customer_id: str
    ) -> Optional[BaseModel]:
        """Get baseline model from file or registry"""
        
        if baseline_path:
            baseline_path = Path(baseline_path)
            if baseline_path.exists():
                logger.info(f"Loading baseline model from {baseline_path}")
                return BaseModel.load_from_path(baseline_path)
        
        elif self.registry_client:
            logger.info(f"Loading baseline model for customer {customer_id} from registry")
            baseline_path = self.registry_client.download_latest_model(
                customer_id=customer_id,
                stage='production'
            )
            if baseline_path:
                return BaseModel.load_from_path(baseline_path)
        
        return None
    
    def _calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Calculate accuracy for predictions"""
        
        # Convert to binary predictions
        binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
        accuracy = (binary_predictions == targets).float().mean().item()
        return accuracy
    
    def _calculate_shadow_metrics(self) -> Dict[str, float]:
        """Calculate shadow testing metrics"""
        
        if not self.metrics_buffer:
            return {}
        
        # Extract metrics
        prediction_diffs = [r['prediction_diff'] for r in self.metrics_buffer]
        
        metrics = {
            'avg_prediction_diff': sum(prediction_diffs) / len(prediction_diffs),
            'max_prediction_diff': max(prediction_diffs),
            'min_prediction_diff': min(prediction_diffs),
            'std_prediction_diff': torch.tensor(prediction_diffs).std().item()
        }
        
        # Add accuracy metrics if available
        accuracy_diffs = [r.get('accuracy_diff', 0) for r in self.metrics_buffer if 'accuracy_diff' in r]
        if accuracy_diffs:
            metrics.update({
                'avg_accuracy_diff': sum(accuracy_diffs) / len(accuracy_diffs),
                'max_accuracy_diff': max(accuracy_diffs)
            })
        
        return metrics
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on shadow testing results"""
        
        if not self.metrics_buffer:
            return "insufficient_data"
        
        metrics = self._calculate_shadow_metrics()
        
        # Check prediction stability
        avg_diff = metrics.get('avg_prediction_diff', 0)
        max_diff = metrics.get('max_prediction_diff', 0)
        
        # Thresholds for decision making
        stable_threshold = 0.1    # Average prediction difference
        acceptable_threshold = 0.3  # Maximum acceptable difference
        
        if avg_diff < stable_threshold:
            return "promote"  # Stable predictions, safe to promote
        elif avg_diff < acceptable_threshold:
            return "caution"  # Some differences, proceed with caution
        else:
            return "reject"   # Too different, don't promote
    
    def set_sample_rate(self, rate: float) -> None:
        """Set shadow testing sample rate"""
        if 0.0 <= rate <= 1.0:
            self.sample_rate = rate
            logger.info(f"Set shadow sample rate to {rate}")
        else:
            logger.error("Sample rate must be between 0.0 and 1.0")
    
    def set_shadow_duration(self, duration: int) -> None:
        """Set shadow testing duration"""
        self.shadow_duration = duration
        logger.info(f"Set shadow duration to {duration}s")
    
    def set_metrics_window(self, window: int) -> None:
        """Set metrics window size"""
        self.metrics_window = window
        logger.info(f"Set metrics window to {window}")
    
    def get_shadow_report(self, results: Dict[str, Any]) -> str:
        """Generate shadow testing report"""
        
        report_lines = [
            "# Shadow Testing Report",
            "",
            "## Summary",
            f"- **Duration**: {results['duration']:.2f}s",
            f"- **Total Requests**: {results['total_requests']}",
            f"- **Recommendation**: {results['recommendation']}",
            "",
            "## Metrics",
        ]
        
        for metric, value in results['metrics'].items():
            report_lines.append(f"- **{metric}**: {value:.4f}")
        
        return "\n".join(report_lines)