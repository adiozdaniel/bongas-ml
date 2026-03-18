"""
Training datasets for BONGAS-ML

Provides PyTorch datasets and data loaders for recommendation training.
Handles data preprocessing, batching, and sampling strategies.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from loguru import logger


class TrainingDataset(Dataset):
    """PyTorch dataset for training recommendation models"""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Initialize training dataset
        
        Args:
            data: Dictionary containing tensors:
                - user_features: [num_samples, user_feature_dim]
                - item_features: [num_samples, item_feature_dim] 
                - targets: [num_samples] (binary or continuous labels)
        """
        self.user_features = data['user_features']
        self.item_features = data['item_features']
        self.targets = data['targets']
        
        # Validate data shapes
        assert len(self.user_features) == len(self.item_features) == len(self.targets), \
            "All tensors must have the same length"
        
        self.num_samples = len(self.targets)
        
        logger.info(f"Training dataset created with {self.num_samples} samples")
        logger.info(f"User features shape: {self.user_features.shape}")
        logger.info(f"Item features shape: {self.item_features.shape}")
        logger.info(f"Targets shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        """Return dataset length"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        return {
            'user_features': self.user_features[idx],
            'item_features': self.item_features[idx],
            'targets': self.targets[idx]
        }


class NegativeSamplingDataset(TrainingDataset):
    """Dataset with negative sampling for implicit feedback"""
    
    def __init__(
        self,
        positive_data: Dict[str, torch.Tensor],
        num_items: int,
        negative_ratio: int = 4,
        sampling_strategy: str = 'uniform'
    ):
        """
        Initialize negative sampling dataset
        
        Args:
            positive_data: Dictionary with positive interactions
            num_items: Total number of items
            negative_ratio: Ratio of negative to positive samples
            sampling_strategy: 'uniform' or 'popularity'
        """
        super().__init__(positive_data)
        
        self.num_items = num_items
        self.negative_ratio = negative_ratio
        self.sampling_strategy = sampling_strategy
        
        # Generate negative samples
        self._generate_negative_samples()
    
    def _generate_negative_samples(self) -> None:
        """Generate negative samples"""
        num_positives = len(self.targets)
        num_negatives = num_positives * self.negative_ratio
        
        # Get positive item indices
        positive_item_indices = self.item_features[:, 0].long()  # Assuming item ID is first feature
        
        # Generate negative item indices
        if self.sampling_strategy == 'uniform':
            negative_item_indices = torch.randint(
                0, self.num_items, (num_negatives,)
            )
        elif self.sampling_strategy == 'popularity':
            # TODO: Implement popularity-based sampling
            negative_item_indices = torch.randint(0, self.num_items, (num_negatives,))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Ensure negatives are different from positives
        for i in range(len(negative_item_indices)):
            while negative_item_indices[i] in positive_item_indices:
                negative_item_indices[i] = torch.randint(0, self.num_items, (1,))
        
        # Create negative features (copy user features, replace item)
        negative_user_features = self.user_features.repeat(self.negative_ratio, 1)
        negative_item_features = self.item_features.repeat(self.negative_ratio, 1)
        negative_item_features[:, 0] = negative_item_indices.float()  # Replace item ID
        
        # Combine positive and negative samples
        self.user_features = torch.cat([self.user_features, negative_user_features], dim=0)
        self.item_features = torch.cat([self.item_features, negative_item_features], dim=0)
        
        # Create targets (1 for positive, 0 for negative)
        positive_targets = torch.ones(num_positives)
        negative_targets = torch.zeros(num_negatives)
        self.targets = torch.cat([positive_targets, negative_targets], dim=0)
        
        # Shuffle the combined dataset
        indices = torch.randperm(len(self.targets))
        self.user_features = self.user_features[indices]
        self.item_features = self.item_features[indices]
        self.targets = self.targets[indices]
        
        self.num_samples = len(self.targets)
        
        logger.info(f"Negative sampling dataset created:")
        logger.info(f"  Original positives: {num_positives}")
        logger.info(f"  Generated negatives: {num_negatives}")
        logger.info(f"  Total samples: {self.num_samples}")


class SequentialDataset(Dataset):
    """Dataset for sequential recommendation models"""
    
    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        max_seq_length: int = 50
    ):
        """
        Initialize sequential dataset
        
        Args:
            sequences: [num_sequences, seq_length] - user interaction sequences
            targets: [num_sequences] - target items for next-item prediction
            max_seq_length: Maximum sequence length
        """
        self.sequences = sequences
        self.targets = targets
        self.max_seq_length = max_seq_length
        
        # Pad sequences to max length
        if sequences.shape[1] < max_seq_length:
            padding = torch.zeros(
                (sequences.shape[0], max_seq_length - sequences.shape[1]),
                dtype=sequences.dtype
            )
            self.sequences = torch.cat([sequences, padding], dim=1)
        
        logger.info(f"Sequential dataset created:")
        logger.info(f"  Sequences shape: {self.sequences.shape}")
        logger.info(f"  Targets shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx]
        }


class PairwiseDataset(Dataset):
    """Dataset for pairwise learning to rank"""
    
    def __init__(
        self,
        user_features: torch.Tensor,
        positive_item_features: torch.Tensor,
        negative_item_features: torch.Tensor
    ):
        """
        Initialize pairwise dataset
        
        Args:
            user_features: [num_pairs, user_feature_dim]
            positive_item_features: [num_pairs, item_feature_dim]
            negative_item_features: [num_pairs, item_feature_dim]
        """
        assert len(user_features) == len(positive_item_features) == len(negative_item_features)
        
        self.user_features = user_features
        self.positive_item_features = positive_item_features
        self.negative_item_features = negative_item_features
        self.num_pairs = len(user_features)
        
        logger.info(f"Pairwise dataset created with {self.num_pairs} pairs")
    
    def __len__(self) -> int:
        return self.num_pairs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'user_features': self.user_features[idx],
            'positive_item_features': self.positive_item_features[idx],
            'negative_item_features': self.negative_item_features[idx]
        }


class DataProcessor:
    """Utility class for data preprocessing and feature engineering"""
    
    @staticmethod
    def normalize_features(features: torch.Tensor) -> torch.Tensor:
        """Normalize features to unit norm"""
        norms = torch.norm(features, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
        return features / norms
    
    @staticmethod
    def standardize_features(features: torch.Tensor) -> torch.Tensor:
        """Standardize features (zero mean, unit variance)"""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)  # Avoid division by zero
        return (features - mean) / std
    
    @staticmethod
    def create_interaction_matrix(
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        ratings: Optional[torch.Tensor] = None,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None
    ) -> torch.Tensor:
        """Create interaction matrix from user-item interactions"""
        
        if num_users is None:
            num_users = user_ids.max().item() + 1
        if num_items is None:
            num_items = item_ids.max().item() + 1
        
        if ratings is None:
            ratings = torch.ones_like(user_ids, dtype=torch.float32)
        
        # Create sparse matrix indices
        indices = torch.stack([user_ids, item_ids], dim=0)
        
        # Create sparse tensor
        interaction_matrix = torch.sparse_coo_tensor(
            indices,
            ratings,
            size=(num_users, num_items)
        )
        
        return interaction_matrix.to_dense()
    
    @staticmethod
    def split_data(
        data: Dict[str, torch.Tensor],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Split data into train, validation, and test sets"""
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        torch.manual_seed(random_seed)
        
        num_samples = len(next(iter(data.values())))
        indices = torch.randperm(num_samples)
        
        train_end = int(train_ratio * num_samples)
        val_end = int((train_ratio + val_ratio) * num_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        def split_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                tensor[train_indices],
                tensor[val_indices],
                tensor[test_indices]
            )
        
        train_data = {key: split_tensor(value)[0] for key, value in data.items()}
        val_data = {key: split_tensor(value)[1] for key, value in data.items()}
        test_data = {key: split_tensor(value)[2] for key, value in data.items()}
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Validation: {len(val_indices)} samples")
        logger.info(f"  Test: {len(test_indices)} samples")
        
        return train_data, val_data, test_data


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    """Create PyTorch data loader with optimized settings"""
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )