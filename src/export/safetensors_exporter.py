"""
Pure-Rust ML Exporter for BONGAS-ML.
Utilizes .safetensors for true single-binary Rust deployments.
"""

import os
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from safetensors.torch import save_file
from loguru import logger
from ..models.base import BaseModel

class SafetensorsExporter:
    """Exporter for .safetensors format (Candle-compatible)."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export(
        self, 
        model: BaseModel, 
        model_name: str,
        version: str = "1.0.0"
    ) -> Path:
        """Export model weights to .safetensors."""
        output_path = self.output_dir / f"{model_name}_{version}.safetensors"
        
        logger.info(f"Exporting {model_name} to {output_path}...")
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Ensure all tensors are on CPU and in a consistent format
        clean_state_dict = {k: v.cpu().to(torch.float32) for k, v in state_dict.items()}
        
        save_file(clean_state_dict, output_path)
        
        logger.success(f"Successfully exported {model_name} to pure-Rust format")
        return output_path

def export_model_to_sovereign(model: BaseModel, model_name: str):
    """Convenience function for factory exports."""
    exporter = SafetensorsExporter(output_dir="bongas-ai/models")
    return exporter.export(model, model_name)
