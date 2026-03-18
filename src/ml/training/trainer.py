"""
Customer Model Trainer for BONGAS-ML

Main orchestrator for training models per customer with validation gates.
Handles the complete training pipeline from data loading to model export.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..models.base import BaseModel
from ..models.two_tower import TwoTowerModel
from .callbacks import TrainingCallback, EarlyStoppingCallback, ModelCheckpointCallback
from .datasets import TrainingDataset
from .losses import BCEWithLogitsLoss, PairwiseHingeLoss, ListMLELoss


class CustomerModelTrainer:
    """Main trainer for customer-specific models"""
    
    def __init__(
        self,
        model: Optional[BaseModel] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train (if None, will use TwoTowerModel)
            device: Device to train on (auto-detect if None)
            callbacks: List of training callbacks
        """
        self.model = model or TwoTowerModel()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to_device(self.device)
        
        self.callbacks = callbacks or []
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_training(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adam',
        loss_type: str = 'bce',
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Setup training components"""
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup loss function
        if loss_type.lower() == 'bce':
            self.criterion = BCEWithLogitsLoss()
        elif loss_type.lower() == 'hinge':
            self.criterion = PairwiseHingeLoss()
        elif loss_type.lower() == 'listmle':
            self.criterion = ListMLELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
        
        # Setup scheduler
        if scheduler_type:
            if scheduler_type.lower() == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=3,
                    verbose=True,
                    **(scheduler_params or {})
                )
            elif scheduler_type.lower() == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.1,
                    **(scheduler_params or {})
                )
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Optimizer: {optimizer_type} (lr={learning_rate})")
        logger.info(f"  Loss: {loss_type}")
        logger.info(f"  Scheduler: {scheduler_type}")
    
    def train(
        self,
        training_data: Dict[str, torch.Tensor],
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adam',
        loss_type: str = 'bce',
        scheduler_type: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train model with validation gates
        
        Args:
            training_data: Dictionary containing training tensors
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            optimizer_type: Type of optimizer
            loss_type: Type of loss function
            scheduler_type: Type of learning rate scheduler
            model_name: Name for the model
            **kwargs: Additional training parameters
        
        Returns:
            Tuple of (trained_model, metrics)
        """
        
        # Setup training
        self.setup_training(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            loss_type=loss_type,
            scheduler_type=scheduler_type
        )
        
        # Create dataset and dataloaders
        dataset = TrainingDataset(training_data)
        train_loader, val_loader = self._create_data_loaders(
            dataset, batch_size, validation_split
        )
        
        # Setup callbacks
        self._setup_callbacks(model_name, **kwargs)
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # Run callbacks before epoch
                self._run_callbacks('on_epoch_start', epoch=epoch)
                
                # Train epoch
                train_metrics = self._train_epoch(train_loader)
                
                # Validate epoch
                val_metrics = self._validate_epoch(val_loader)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.append(epoch_metrics)
                
                # Run callbacks after epoch
                self._run_callbacks('on_epoch_end', epoch=epoch, metrics=epoch_metrics)
                
                # Check early stopping
                if self._check_early_stopping(val_metrics['val_loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final validation and metrics
        final_metrics = self._get_final_metrics()
        
        return self.model, final_metrics
    
    def evaluate(
        self, 
        model: BaseModel, 
        test_data: Dict[str, torch.Tensor], 
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Evaluate model on test data"""
        
        model.eval()
        dataset = TrainingDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions = model(user_features, item_features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = model.evaluate_metrics(all_predictions, all_targets)
        metrics['test_loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def save_model(self, model: BaseModel, path: Union[str, Path]) -> None:
        """Save trained model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> BaseModel:
        """Load trained model"""
        path = Path(path)
        return self.model.load(path)
    
    def _create_data_loaders(
        self, 
        dataset: TrainingDataset, 
        batch_size: int, 
        validation_split: float
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created dataloaders:")
        logger.info(f"  Train: {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            user_features = batch['user_features'].to(self.device)
            item_features = batch['item_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(user_features, item_features)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions = self.model(user_features, item_features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        avg_loss = total_loss / num_batches
        metrics = self.model.evaluate_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def _setup_callbacks(self, model_name: Optional[str] = None, **kwargs) -> None:
        """Setup training callbacks"""
        
        # Add default callbacks if not already present
        callback_types = [type(cb) for cb in self.callbacks]
        
        if EarlyStoppingCallback not in callback_types:
            self.callbacks.append(EarlyStoppingCallback(patience=5))
        
        if ModelCheckpointCallback not in callback_types:
            checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
            self.callbacks.append(ModelCheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                model_name=model_name
            ))
    
    def _run_callbacks(self, event: str, **kwargs) -> None:
        """Run callbacks for specific event"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered"""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                return callback.check_early_stopping(val_loss)
        return False
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics"""
        
        if not self.training_history:
            return {}
        
        final_metrics = self.training_history[-1].copy()
        final_metrics['best_val_loss'] = min(
            [epoch['val_loss'] for epoch in self.training_history]
        )
        final_metrics['final_val_loss'] = self.training_history[-1]['val_loss']
        final_metrics['epochs_trained'] = len(self.training_history)
        
        return final_metrics