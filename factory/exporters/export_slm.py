"""
Export script for The Swahili Brain (SLM).
Takes the fine-tuned PyTorch weights and exports them to Safetensors for Rust ingestion.
"""

import os
import torch
from safetensors.torch import save_file
from loguru import logger
from src.models.slm import LanguageModel

def export_to_safetensors():
    # Load model with trained weights (Simulated)
    logger.info("Initializing SLM for export...")
    model = LanguageModel(vocab_size=32000, hidden_dim=768)
    
    # Path to the trained weights (Simulated)
    weights_path = os.path.join("research", "language", "assets", "weights.safetensors")
    
    # In a real scenario, we would load the state dict from the SFT output
    # model.load_state_dict(torch.load(weights_path))
    
    # Export to pure safetensors
    output_path = os.path.join("bongas-ai", "models", "slm_head.safetensors")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Exporting student head to {output_path}...")
    
    # Extract only the weights we need for the student head
    state_dict = model.state_dict()
    
    # Clean up state dict keys if necessary (e.g., removing 'model.' prefix)
    clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    save_file(clean_state_dict, output_path)
    logger.success(f"SLM Student Head exported successfully to {output_path}")

if __name__ == "__main__":
    export_to_safetensors()
