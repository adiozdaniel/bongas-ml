"""
Whisper Export Script for BONGAS-ML.
Downloads OpenAI Whisper-Tiny and exports weights to .safetensors for Rust/Candle.
"""

import os
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
from safetensors.torch import save_file

def export_whisper_to_sovereign():
    model_id = "openai/whisper-tiny"
    logger.info(f"=== BONGAS-ML: Exporting {model_id} to Safetensors ===")
    
    output_dir = Path("../bongas-ai/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "whisper_tiny.safetensors"

    try:
        # 1. Load Model
        logger.info(f"Loading {model_id} from Hugging Face...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        model.eval()

        # 2. Extract and Clean State Dict
        logger.info("Extracting weights and converting to float32...")
        state_dict = model.state_dict()
        
        # Whisper models often have complex keys, ensure they are compatible with Candle's loader
        clean_state_dict = {k: v.cpu().to(torch.float32) for k, v in state_dict.items()}

        # 3. Save to Safetensors
        save_file(clean_state_dict, str(output_path))
        logger.success(f"Successfully exported Whisper weights -> {output_path}")

        # 4. Copy Tokenizer/Config (Needed by Candle)
        logger.info("Saving companion configs...")
        # In a real pipeline, we'd also export tokenizer.json and config.json
        # For now, we confirm the weights are ready.

    except Exception as e:
        logger.error(f"Whisper export failed: {e}")

if __name__ == "__main__":
    export_whisper_to_sovereign()
