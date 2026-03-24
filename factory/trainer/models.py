import torch
import torch.nn as nn
from typing import Tuple

class VisionAuditorHead(nn.Module):
    """
    The Local Student Head for Visual Compliance and Semantics.
    
    This lightweight Multi-Layer Perceptron (MLP) takes the raw, massive 
    vector output (Visual DNA) from the frozen 1.2B 'sight-core' model 
    and translates it into specific, client-defined business metrics.
    
    Inputs:
        visual_dna (Tensor): [batch_size, 1024]
        
    Outputs:
        safety_logits (Tensor): [batch_size, 3] e.g., GE, PG, 18+
        vibe_logits (Tensor): [batch_size, 10] e.g., 'High-Energy', 'Cinematic'
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, 
                 num_safety_classes: int = 3, num_vibe_classes: int = 10):
        super().__init__()
        
        # Shared feature extraction layer
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        
        # Task-specific branches
        self.safety_classifier = nn.Linear(hidden_dim, num_safety_classes)
        self.vibe_classifier = nn.Linear(hidden_dim, num_vibe_classes)

    def forward(self, visual_dna: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_network(visual_dna)
        
        safety_logits = self.safety_classifier(features)
        vibe_logits = self.vibe_classifier(features)
        
        return safety_logits, vibe_logits


class TribeConductorHead(nn.Module):
    """
    The Local Student Head for Persona-Based Ranking.
    
    Learns the probability of a specific Behavioral Tribe interacting 
    with a specific piece of Content DNA based entirely on local, 
    on-premise ClickHouse telemetry.
    
    Inputs:
        tribe_embedding (Tensor): [batch_size, 64] 
        item_dna (Tensor): [batch_size, 1024]
        
    Outputs:
        interaction_probability (Tensor): [batch_size, 1] (Sigmoid confidence)
    """
    def __init__(self, tribe_dim: int = 64, item_dim: int = 1024, hidden_dim: int = 128):
        super().__init__()
        
        # Early Fusion Architecture
        self.mlp = nn.Sequential(
            nn.Linear(tribe_dim + item_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, tribe_embedding: torch.Tensor, item_dna: torch.Tensor) -> torch.Tensor:
        # Concatenate the Tribe Profile with the Content DNA
        combined_features = torch.cat([tribe_embedding, item_dna], dim=-1)
        probability = self.mlp(combined_features)
        return probability


class SovereignAudioHead(nn.Module):
    """
    The Local Student Head for Audio Intelligence (The Swahili Brain).
    
    Implements 'Distillation over Memorization' by taking the 1024-dimensional 
    dense vector (Audio DNA) from the frozen foundation backbone and 
    mapping it to linguistic tokens and semantic metadata.
    
    Architecture:
        - Input: Audio DNA [batch_size, 1024]
        - Output: Linguistic Tokens [batch_size, vocab_size]
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, vocab_size: int = 32000):
        super().__init__()
        
        # The Linguistic Bridge: Mapping latent signals to tokens
        self.bridge = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def forward(self, audio_dna: torch.Tensor) -> torch.Tensor:
        return self.bridge(audio_dna)
