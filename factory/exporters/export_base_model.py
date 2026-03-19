"""
BONGAS-ML: Sovereign Base Exporter

This script converts the massive 1.2B 'sight-core' foundation model 
(currently stored as PyTorch .safetensors) into a highly optimized 
ONNX computational graph.

By shipping the .onnx file to the client instead of .safetensors, we:
1. Eliminate the need for a massive 2GB+ PyTorch installation on the client server.
2. Enable sub-millisecond, hardware-accelerated inference via ONNX Runtime in Rust.
3. Bake the architecture directly into the graph, obfuscating the original model structure.
"""

import torch
import shutil
from pathlib import Path
from loguru import logger
from transformers import AutoModel, AutoVideoProcessor

def export_sight_core_to_onnx():
    logger.info("=== BONGAS-ML: Sovereign Base Exporter ===")
    
    # 1. Define Paths
    # The "Factory" location where we keep the raw PyTorch weights
    source_dir = Path("src/models/sight-core")
    
    # The "Release" location where we ship the obfuscated ONNX model
    release_dir = Path("../bongas-ai/release/models/frozen/sight-core")
    release_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = release_dir / "sight_core.onnx"
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        logger.error("Please ensure the sight-core weights are in bongas-ml/models/sight-core")
        return

    logger.info(f"Loading 1.2B Foundation Model from: {source_dir}")
    
    try:
        # 2. Load the Model and Processor
        # We load it from our local renamed directory
        model = AutoModel.from_pretrained(source_dir)
        processor = AutoVideoProcessor.from_pretrained(source_dir)
        
        # 3. Prepare Dummy Input for Tracing
        # Foundation Vision Model 2 expects a specific 5D tensor shape for video frames:
        # [batch_size, num_channels, num_frames, height, width]
        # We use the processor config to get the exact dimensions
        crop_size = model.config.crop_size
        frames_per_clip = model.config.frames_per_clip
        in_chans = model.config.in_chans
        
        logger.info(f"Model configured for: {frames_per_clip} frames, {crop_size}x{crop_size} resolution, {in_chans} channels.")
        
        # Create a dummy video tensor matching the exact shape
        dummy_video = torch.randn(1, in_chans, frames_per_clip, crop_size, crop_size)
        
        # Move model to eval mode (critical for ONNX export)
        model.eval()
        
        # 4. Export to ONNX
        logger.info(f"Tracing PyTorch graph and exporting to ONNX... This may take a few minutes for a 1.2B parameter model.")
        
        # We use a custom wrapper to extract just the vision features, 
        # bypassing any HuggingFace specific output classes that might not translate well to ONNX
        class VisionFeatureExtractor(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, pixel_values):
                # We only want the raw embedding vectors
                return self.base_model.get_vision_features(pixel_values)
                
        extractor = VisionFeatureExtractor(model)
        
        torch.onnx.export(
            extractor,
            (dummy_video,),
            str(onnx_path),
            export_params=True,
            opset_version=16, # Use a recent opset for complex models
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["visual_dna"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"}, # Allow variable batch sizes for inference
                "visual_dna": {0: "batch_size"}
            }
        )
        
        logger.success(f"Successfully exported ONNX Base Model -> {onnx_path}")
        
        # 5. Copy the critical preprocessor config
        # The Rust sidecar needs this to know how to format the raw video before feeding it to ONNX
        preprocessor_src = source_dir / "vision_preprocessor.json"
        preprocessor_dest = release_dir / "vision_preprocessor.json"
        
        if preprocessor_src.exists():
            shutil.copy(preprocessor_src, preprocessor_dest)
            logger.info(f"Copied critical preprocessor config -> {preprocessor_dest}")
        else:
            logger.warning("vision_preprocessor.json not found! The Rust sidecar will need this to format video frames.")
            
        logger.info("=== Sovereign Base Export Complete ===")
        logger.info("The frozen release bundle is ready for on-premise deployment.")
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")

if __name__ == "__main__":
    export_sight_core_to_onnx()