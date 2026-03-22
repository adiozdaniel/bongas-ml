"""
BONGAS-ML: Sovereign Base Exporter (Candle Edition)

This script exports the massive 1.2B 'sight-core' foundation model 
into pure-Rust compatible .safetensors.

By shipping .safetensors, we achieve a 100% Rust-native 
ML stack with ZERO external C++ dependencies.
"""

import torch
import shutil
from pathlib import Path
from loguru import logger
from transformers import AutoModel
from safetensors.torch import save_file

def export_sight_core_to_sovereign():
    logger.info("=== BONGAS-ML: Sovereign Base Exporter (Candle) ===")
    
    source_dir = Path("src/models/sight-core")
    release_dir = Path("../bongas-ai/release/models/frozen/sight-core")
    release_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = release_dir / "sight_core.safetensors"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    logger.info(f"Loading 1.2B Foundation Model from: {source_dir}")
    
    try:
        # Load the Model
        model = AutoModel.from_pretrained(source_dir)
        model.eval()
        
        logger.info(f"Exporting state dict to .safetensors...")
        
        # Get state dict and ensure CPU/float32
        state_dict = model.state_dict()
        clean_state_dict = {k: v.cpu().to(torch.float32) for k, v in state_dict.items()}
        
        save_file(clean_state_dict, output_path)
        
        logger.success(f"Successfully exported pure-Rust weights -> {output_path}")
        
        # Copy config files needed for Candle initialization
        for config_file in ["config.json", "vision_preprocessor.json"]:
            src = source_dir / config_file
            if src.exists():
                shutil.copy(src, release_dir / config_file)
                logger.info(f"Copied config -> {config_file}")
            
        logger.info("=== Sovereign Base Export Complete ===")
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")

if __name__ == "__main__":
    export_sight_core_to_sovereign()
