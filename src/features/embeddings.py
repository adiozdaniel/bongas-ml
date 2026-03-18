"""
Embedding utilities for BONGAS-ML

Provides embedding layer management and optimization including:
- Embedding layer creation and management
- Pre-trained embedding loading
- Embedding optimization and compression
- Embedding visualization and analysis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger


class EmbeddingManager:
    """Manages embedding layers for categorical features"""
    
    def __init__(self):
        self.embeddings = {}
        self.vocab_sizes = {}
        self.embedding_dims = {}
        self.pretrained_embeddings = {}
        self.embedding_stats = {}
        
        logger.info("Embedding manager initialized")
    
    def create_embedding_layer(
        self,
        feature_name: str,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        pretrained_weights: Optional[torch.Tensor] = None,
        freeze: bool = False
    ) -> nn.Embedding:
        """
        Create and register an embedding layer
        
        Args:
            feature_name: Name of the feature
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            padding_idx: Index for padding token
            pretrained_weights: Pre-trained embedding weights
            freeze: Whether to freeze embeddings during training
        
        Returns:
            PyTorch embedding layer
        """
        
        try:
            # Create embedding layer
            embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
            
            # Initialize with pretrained weights if provided
            if pretrained_weights is not None:
                if pretrained_weights.shape != (vocab_size, embedding_dim):
                    raise ValueError(f"Pretrained weights shape {pretrained_weights.shape} "
                                   f"doesn't match embedding shape ({vocab_size}, {embedding_dim})")
                
                embedding.weight.data.copy_(pretrained_weights)
                self.pretrained_embeddings[feature_name] = pretrained_weights.clone()
                logger.info(f"Initialized {feature_name} embeddings with pretrained weights")
            else:
                # Initialize with Xavier uniform
                nn.init.xavier_uniform_(embedding.weight.data)
                logger.info(f"Initialized {feature_name} embeddings with Xavier uniform")
            
            # Freeze if requested
            if freeze:
                embedding.weight.requires_grad = False
                logger.info(f"Froze {feature_name} embeddings")
            
            # Store metadata
            self.embeddings[feature_name] = embedding
            self.vocab_sizes[feature_name] = vocab_size
            self.embedding_dims[feature_name] = embedding_dim
            
            # Calculate statistics
            self._calculate_embedding_stats(feature_name, embedding.weight.data)
            
            logger.info(f"Created embedding layer for {feature_name}: {vocab_size} x {embedding_dim}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding layer for {feature_name}: {e}")
            raise
    
    def load_pretrained_embeddings(
        self,
        feature_name: str,
        file_path: str,
        format: str = 'word2vec',
        delimiter: str = ' '
    ) -> torch.Tensor:
        """
        Load pre-trained embeddings from file
        
        Args:
            feature_name: Name of the feature
            file_path: Path to embedding file
            format: Format of embedding file ('word2vec', 'glove', 'fasttext')
            delimiter: Delimiter for text-based formats
        
        Returns:
            Pre-trained embedding weights
        """
        
        try:
            if format == 'word2vec':
                weights = self._load_word2vec_format(file_path)
            elif format == 'glove':
                weights = self._load_glove_format(file_path, delimiter)
            elif format == 'fasttext':
                weights = self._load_fasttext_format(file_path)
            else:
                raise ValueError(f"Unknown embedding format: {format}")
            
            self.pretrained_embeddings[feature_name] = weights
            logger.info(f"Loaded pretrained embeddings for {feature_name}: {weights.shape}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load pretrained embeddings for {feature_name}: {e}")
            raise
    
    def _load_word2vec_format(self, file_path: str) -> torch.Tensor:
        """Load embeddings in word2vec format"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split()
            vocab_size, embedding_dim = int(header[0]), int(header[1])
            
            weights = torch.zeros(vocab_size, embedding_dim)
            
            for i, line in enumerate(f):
                parts = line.strip().split()
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                weights[i] = vector
        
        return weights
    
    def _load_glove_format(self, file_path: str, delimiter: str) -> torch.Tensor:
        """Load embeddings in GloVe format"""
        
        vectors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                vectors.append(vector)
        
        weights = torch.stack(vectors)
        return weights
    
    def _load_fasttext_format(self, file_path: str) -> torch.Tensor:
        """Load embeddings in FastText format"""
        
        # FastText format is similar to word2vec
        return self._load_word2vec_format(file_path)
    
    def get_embedding_layer(self, feature_name: str) -> Optional[nn.Embedding]:
        """Get embedding layer by feature name"""
        return self.embeddings.get(feature_name)
    
    def get_embedding_weights(self, feature_name: str) -> Optional[torch.Tensor]:
        """Get embedding weights by feature name"""
        
        embedding = self.embeddings.get(feature_name)
        if embedding is not None:
            return embedding.weight.data
        return None
    
    def get_pretrained_weights(self, feature_name: str) -> Optional[torch.Tensor]:
        """Get pretrained weights by feature name"""
        return self.pretrained_embeddings.get(feature_name)
    
    def optimize_embeddings(
        self,
        feature_name: str,
        method: str = 'pruning',
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Optimize embeddings using various methods
        
        Args:
            feature_name: Name of the feature
            method: Optimization method ('pruning', 'quantization', 'compression')
            threshold: Threshold for optimization
        
        Returns:
            Optimized embedding weights
        """
        
        try:
            weights = self.get_embedding_weights(feature_name)
            if weights is None:
                logger.warning(f"No embeddings found for {feature_name}")
                return None
            
            if method == 'pruning':
                optimized_weights = self._prune_embeddings(weights, threshold)
            elif method == 'quantization':
                optimized_weights = self._quantize_embeddings(weights, threshold)
            elif method == 'compression':
                optimized_weights = self._compress_embeddings(weights, threshold)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Update embedding layer
            embedding = self.embeddings[feature_name]
            embedding.weight.data.copy_(optimized_weights)
            
            # Update statistics
            self._calculate_embedding_stats(feature_name, optimized_weights)
            
            logger.info(f"Optimized embeddings for {feature_name} using {method}")
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Failed to optimize embeddings for {feature_name}: {e}")
            raise
    
    def _prune_embeddings(self, weights: torch.Tensor, threshold: float) -> torch.Tensor:
        """Prune embeddings by removing low-magnitude vectors"""
        
        # Calculate L2 norm for each embedding vector
        norms = torch.norm(weights, dim=1, keepdim=True)
        
        # Create mask for vectors above threshold
        mask = norms.squeeze() > threshold
        
        # Keep only vectors above threshold
        pruned_weights = weights[mask]
        
        logger.info(f"Pruned embeddings from {weights.shape[0]} to {pruned_weights.shape[0]} vectors")
        return pruned_weights
    
    def _quantize_embeddings(self, weights: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
        """Quantize embeddings to reduce precision"""
        
        # Calculate min and max values
        min_val = weights.min()
        max_val = weights.max()
        
        # Quantize to specified number of bits
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        quantized = torch.round((weights - min_val) / scale)
        dequantized = quantized * scale + min_val
        
        logger.info(f"Quantized embeddings to {num_bits} bits")
        return dequantized
    
    def _compress_embeddings(self, weights: torch.Tensor, compression_ratio: float) -> torch.Tensor:
        """Compress embeddings using PCA"""
        
        # Convert to numpy for PCA
        weights_np = weights.detach().numpy()
        
        # Apply PCA
        from sklearn.decomposition import PCA
        n_components = max(1, int(weights.shape[1] * compression_ratio))
        
        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(weights_np)
        
        # Reconstruct
        reconstructed = pca.inverse_transform(compressed)
        
        logger.info(f"Compressed embeddings from {weights.shape[1]} to {n_components} dimensions")
        return torch.tensor(reconstructed, dtype=torch.float32)
    
    def _calculate_embedding_stats(self, feature_name: str, weights: torch.Tensor) -> None:
        """Calculate and store embedding statistics"""
        
        stats = {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'norm_mean': float(torch.norm(weights, dim=1).mean()),
            'norm_std': float(torch.norm(weights, dim=1).std()),
            'sparsity': float((weights == 0).sum() / weights.numel()),
            'memory_usage_mb': float(weights.numel() * weights.element_size() / (1024 * 1024))
        }
        
        self.embedding_stats[feature_name] = stats
    
    def get_embedding_statistics(self, feature_name: str) -> Optional[Dict[str, float]]:
        """Get embedding statistics by feature name"""
        return self.embedding_stats.get(feature_name)
    
    def get_all_embedding_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all embeddings"""
        return self.embedding_stats.copy()
    
    def visualize_embeddings(
        self,
        feature_name: str,
        method: str = 'pca',
        n_components: int = 2,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize embeddings using dimensionality reduction
        
        Args:
            feature_name: Name of the feature
            method: Visualization method ('pca', 'tsne', 'umap')
            n_components: Number of components for visualization
            output_path: Optional path to save visualization
        
        Returns:
            Reduced dimension embeddings
        """
        
        try:
            weights = self.get_embedding_weights(feature_name)
            if weights is None:
                logger.warning(f"No embeddings found for {feature_name}")
                return None
            
            # Convert to numpy
            weights_np = weights.detach().numpy()
            
            # Apply dimensionality reduction
            if method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
            elif method == 'umap':
                try:
                    from umap import UMAP
                    reducer = UMAP(n_components=n_components, random_state=42)
                except ImportError:
                    logger.error("UMAP not available. Install with: pip install umap-learn")
                    return None
            else:
                raise ValueError(f"Unknown visualization method: {method}")
            
            reduced = reducer.fit_transform(weights_np)
            
            if output_path:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
                plt.title(f'{feature_name} Embeddings - {method.upper()}')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved embedding visualization to {output_path}")
            
            logger.info(f"Generated {method.upper()} visualization for {feature_name} embeddings")
            return reduced
            
        except Exception as e:
            logger.error(f"Failed to visualize embeddings for {feature_name}: {e}")
            raise
    
    def save_embeddings(
        self,
        feature_name: str,
        file_path: str,
        format: str = 'torch'
    ) -> None:
        """
        Save embeddings to file
        
        Args:
            feature_name: Name of the feature
            file_path: Path to save embeddings
            format: Format to save ('torch', 'numpy', 'txt')
        """
        
        try:
            weights = self.get_embedding_weights(feature_name)
            if weights is None:
                logger.warning(f"No embeddings found for {feature_name}")
                return
            
            if format == 'torch':
                torch.save(weights, file_path)
            elif format == 'numpy':
                np.save(file_path, weights.numpy())
            elif format == 'txt':
                np.savetxt(file_path, weights.numpy())
            else:
                raise ValueError(f"Unknown save format: {format}")
            
            logger.info(f"Saved embeddings for {feature_name} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings for {feature_name}: {e}")
            raise
    
    def load_embeddings(
        self,
        feature_name: str,
        file_path: str,
        vocab_size: int,
        embedding_dim: int,
        format: str = 'torch'
    ) -> nn.Embedding:
        """
        Load embeddings from file
        
        Args:
            feature_name: Name of the feature
            file_path: Path to load embeddings from
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            format: Format to load ('torch', 'numpy', 'txt')
        
        Returns:
            PyTorch embedding layer
        """
        
        try:
            if format == 'torch':
                weights = torch.load(file_path)
            elif format == 'numpy':
                weights = torch.tensor(np.load(file_path))
            elif format == 'txt':
                weights = torch.tensor(np.loadtxt(file_path))
            else:
                raise ValueError(f"Unknown load format: {format}")
            
            # Create embedding layer
            embedding = self.create_embedding_layer(
                feature_name=feature_name,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                pretrained_weights=weights
            )
            
            logger.info(f"Loaded embeddings for {feature_name} from {file_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {feature_name}: {e}")
            raise
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for all embeddings"""
        
        memory_usage = {}
        total_memory = 0
        
        for feature_name, stats in self.embedding_stats.items():
            memory_usage[feature_name] = stats['memory_usage_mb']
            total_memory += stats['memory_usage_mb']
        
        memory_usage['total'] = total_memory
        return memory_usage
    
    def reset(self) -> None:
        """Reset all embeddings"""
        
        self.embeddings = {}
        self.vocab_sizes = {}
        self.embedding_dims = {}
        self.pretrained_embeddings = {}
        self.embedding_stats = {}
        
        logger.info("Embedding manager reset")