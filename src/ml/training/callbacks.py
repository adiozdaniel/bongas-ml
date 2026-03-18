"""
Training callbacks for BONGAS-ML

Provides hooks and callbacks for training monitoring, early stopping,
model checkpointing, and other training lifecycle events.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger


class TrainingCallback(ABC):
    """Abstract base class for training callbacks"""
    
    @abstractmethod
    def on_epoch_start(self, **kwargs) -> None:
        """Called at the start of each epoch"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, **kwargs) -> None:
        """Called at the end of each epoch"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback based on validation loss"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping callback
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        metrics = kwargs.get('metrics', {})
        val_loss = metrics.get('val_loss', 0.0)
        
        score = val_loss if self.mode == 'min' else -val_loss
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered"""
        return self.early_stop


class ModelCheckpointCallback(TrainingCallback):
    """Model checkpointing callback"""
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        model_name: Optional[str] = None,
        save_best_only: bool = True,
        save_frequency: int = 1
    ):
        """
        Initialize model checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name for the model
            save_best_only: Only save if validation loss improves
            save_frequency: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name or 'model'
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        
        self.best_loss = float('inf')
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        epoch = kwargs.get('epoch', 0)
        metrics = kwargs.get('metrics', {})
        val_loss = metrics.get('val_loss', 0.0)
        
        # Check if we should save
        should_save = False
        
        if self.save_best_only:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                should_save = True
        else:
            if epoch % self.save_frequency == 0:
                should_save = True
        
        if should_save:
            # Save model
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}_loss_{val_loss:.4f}.pt"
            
            # Get model from kwargs or assume it's available
            model = kwargs.get('model')
            if model:
                model.save(checkpoint_path)
            else:
                logger.warning("Model not available for checkpointing")
            
            logger.info(f"Model checkpoint saved: {checkpoint_path}")


class LRLoggingCallback(TrainingCallback):
    """Learning rate logging callback"""
    
    def __init__(self, optimizer: Optional[torch.optim.Optimizer] = None):
        self.optimizer = optimizer
    
    def on_epoch_start(self, **kwargs) -> None:
        if self.optimizer:
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {lr:.6f}")
    
    def on_epoch_end(self, **kwargs) -> None:
        pass


class MetricsLoggingCallback(TrainingCallback):
    """Metrics logging callback"""
    
    def __init__(self, log_frequency: int = 1):
        self.log_frequency = log_frequency
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        epoch = kwargs.get('epoch', 0)
        metrics = kwargs.get('metrics', {})
        
        if epoch % self.log_frequency == 0:
            logger.info(f"Epoch {epoch} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")


class ProgressBarCallback(TrainingCallback):
    """Progress bar callback for training"""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def on_epoch_start(self, **kwargs) -> None:
        self.current_epoch = kwargs.get('epoch', 0)
        logger.info(f"Epoch {self.current_epoch + 1}/{self.total_epochs}")
    
    def on_epoch_end(self, **kwargs) -> None:
        pass


class TensorBoardCallback(TrainingCallback):
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = './logs', model_name: Optional[str] = None):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.model_name = model_name
        except ImportError:
            logger.warning("TensorBoard not available, skipping TensorBoard callback")
            self.writer = None
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        if self.writer is None:
            return
        
        epoch = kwargs.get('epoch', 0)
        metrics = kwargs.get('metrics', {})
        
        # Log metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'{key}', value, epoch)
        
        # Log learning rate
        optimizer = kwargs.get('optimizer')
        if optimizer:
            lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch)
        
        self.writer.flush()
    
    def close(self) -> None:
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class GradientClippingCallback(TrainingCallback):
    """Gradient clipping callback"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        pass
    
    def on_batch_end(self, model: torch.nn.Module, **kwargs) -> None:
        """Clip gradients after backward pass"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)


class ReduceLROnPlateauCallback(TrainingCallback):
    """Reduce learning rate on plateau callback"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose
        )
    
    def on_epoch_start(self, **kwargs) -> None:
        pass
    
    def on_epoch_end(self, **kwargs) -> None:
        metrics = kwargs.get('metrics', {})
        val_loss = metrics.get('val_loss', 0.0)
        
        self.scheduler.step(val_loss)


class TrainingHistoryCallback(TrainingCallback):
    """Training history tracking callback"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
    
    def on_epoch_start(self, **kwargs) -> None:
        self.epoch_start_time = kwargs.get('epoch_start_time')
    
    def on_epoch_end(self, **kwargs) -> None:
        epoch = kwargs.get('epoch', 0)
        metrics = kwargs.get('metrics', {})
        
        # Store metrics
        self.history['train_loss'].append(metrics.get('train_loss', 0.0))
        self.history['val_loss'].append(metrics.get('val_loss', 0.0))
        
        # Store learning rate
        optimizer = kwargs.get('optimizer')
        if optimizer:
            lr = optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(lr)
        
        # Store epoch time
        if hasattr(self, 'epoch_start_time'):
            epoch_time = kwargs.get('epoch_end_time', 0) - self.epoch_start_time
            self.history['epoch_times'].append(epoch_time)
    
    def get_history(self) -> Dict[str, Any]:
        """Get training history"""
        return self.history