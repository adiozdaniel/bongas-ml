from .trainer import BlackboxSovereignTrainer
from loguru import logger
import sys
import os

def execute_sovereign_distillation():
    """
    Sovereign Distillation Orchestrator.
    
    Implements the 'Budget to Meal' pipeline by distilling latent 
    Audio DNA into domain-specific linguistic expertise. This process 
    adheres to the 'Glass Jar' principle, ensuring a fixed physical 
    footprint for the resulting Student Head weights.
    """
    logger.info("=== BONGAS-ML: Sovereign Distillation Loop (The Swahili Brain) ===")
    
    # Define the 'Golden Record' landing zone for distilled artifacts
    # Path is relative to the bongas-ml root for zero-copy symlinking
    output_path = "outputs"
    
    # Initialize the Obfuscated Training Orchestrator
    trainer = BlackboxSovereignTrainer(
        db_client=None, 
        output_dir=output_path
    )
    
    try:
        # Trigger the algorithmic distillation of the Sovereign Ear
        # This transforms the foundational compute budget into a specialized linguistic meal.
        trainer.train_and_export_audio_head(epochs=25)
        logger.success("=== Distillation Complete: Sovereign Ear distilled and deployed to the Warehouse ===")
    except Exception as e:
        logger.error(f"Sovereign Distillation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    execute_sovereign_distillation()
