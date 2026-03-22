"""
Small Language Model (SLM) definition for The Swahili Brain.
Supports LoRA adapters for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .base import BaseModel

class SLMHead(nn.Module):
    """
    A lightweight Transformer head for language generation.
    Designed to be used as a 'Student Head' in the Sovereign Training Pillar.
    """
    def __init__(
        self, 
        vocab_size: int = 32000, 
        hidden_dim: int = 768, 
        num_layers: int = 6, 
        num_heads: int = 12
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        # Simplified attention mask handling for TransformerEncoder
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

class LanguageModel(BaseModel):
    """
    Sovereign Language Model (The Swahili Brain).
    Wraps the SLMHead and provides training utilities.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 768):
        super().__init__()
        self.model = SLMHead(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    def get_user_embeddings(self, user_features: torch.Tensor) -> torch.Tensor:
        # Not applicable for LM, but required by BaseModel
        return torch.zeros((user_features.size(0), 1))

    def get_item_embeddings(self, item_features: torch.Tensor) -> torch.Tensor:
        # Not applicable for LM, but required by BaseModel
        return torch.zeros((item_features.size(0), 1))

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim
        })
        return info
