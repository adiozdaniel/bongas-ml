#!/usr/bin/env python3
"""
BONGAS-ML Training Script

Command-line interface for training ML models per customer.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from data.loader import TrainingDataLoader
from models.two_tower import TwoTowerModel
from training.trainer import CustomerModelTrainer
from utils.logging import setup_logging


@click.group()
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
@click.option('--log-file', type=click.Path())
def cli(log_level: str, log_file: Optional[str]):
    """BONGAS-ML Training CLI"""
    setup_logging(log_level, log_file)


@cli.command()
@click.option('--customer-id', required=True, help='Customer ID to train for')
@click.option('--db-url', required=True, help='Database connection URL')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=256, help='Training batch size')
@click.option('--learning-rate', default=1e-3, help='Learning rate')
@click.option('--validation-split', default=0.1, help='Validation split ratio')
@click.option('--output-dir', default='./outputs', help='Output directory for models')
@click.option('--model-name', help='Custom model name')
@click.option('--days-back', default=30, help='Days of data to use for training')
@click.option('--force', is_flag=True, help='Force training even if model exists')
def train(
    customer_id: str,
    db_url: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    output_dir: str,
    model_name: Optional[str],
    days_back: int,
    force: bool
):
    """Train a model for a specific customer"""
    
    logger.info(f"Starting training for customer {customer_id}")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    try:
        # Load training data
        logger.info("Loading training data...")
        data_loader = TrainingDataLoader(db_url)
        training_data = data_loader.load_customer_data(customer_id, days_back)
        
        if not training_data:
            logger.error(f"No training data found for customer {customer_id}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(training_data['interactions'])} interactions")
        
        # Initialize trainer
        trainer = CustomerModelTrainer()
        
        # Train model
        logger.info("Starting model training...")
        model, metrics = trainer.train(
            training_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            model_name=model_name or f"model_{customer_id}_{epochs}e"
        )
        
        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / f"{model_name or f'model_{customer_id}'}.pt"
        trainer.save_model(model, model_path)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final validation loss: {metrics['final_val_loss']:.6f}")
        logger.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--customer-id', required=True, help='Customer ID to validate')
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--db-url', required=True, help='Database connection URL')
@click.option('--test-split', default=0.2, help='Test split ratio')
@click.option('--batch-size', default=256, help='Evaluation batch size')
def validate(
    customer_id: str,
    model_path: str,
    db_url: str,
    test_split: float,
    batch_size: int
):
    """Validate a trained model"""
    
    logger.info(f"Validating model for customer {customer_id}")
    
    try:
        # Load model
        trainer = CustomerModelTrainer()
        model = trainer.load_model(Path(model_path))
        
        # Load test data
        data_loader = TrainingDataLoader(db_url)
        test_data = data_loader.load_customer_data(customer_id, days_back=30)
        
        # Evaluate model
        metrics = trainer.evaluate(model, test_data, batch_size=batch_size)
        
        logger.info("Validation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--customer-id', required=True, help='Customer ID to export')
@click.option('--model-path', required=True, help='Path to trained model')
@click.option('--output-dir', default='./outputs', help='Output directory for ONNX model')
@click.option('--model-name', help='Custom ONNX model name')
@click.option('--user-feature-dim', default=128, help='User feature dimension')
@click.option('--item-feature-dim', default=128, help='Item feature dimension')
@click.option('--optimize', is_flag=True, help='Optimize ONNX model')
def export(
    customer_id: str,
    model_path: str,
    output_dir: str,
    model_name: Optional[str],
    user_feature_dim: int,
    item_feature_dim: int,
    optimize: bool
):
    """Export model to ONNX format"""
    
    logger.info(f"Exporting model for customer {customer_id}")
    
    try:
        from export.onnx_exporter import ONNXExporter
        
        # Load model
        trainer = CustomerModelTrainer()
        model = trainer.load_model(Path(model_path))
        
        # Export to ONNX
        exporter = ONNXExporter(output_dir)
        metadata = exporter.export_two_tower(
            model,
            model_name=model_name or f"onnx_model_{customer_id}",
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            optimize=optimize
        )
        
        logger.info(f"Model exported successfully!")
        logger.info(f"ONNX path: {metadata['onnx_path']}")
        logger.info(f"Accuracy match: {metadata['accuracy_match']:.6f}")
        logger.info(f"Model size: {metadata['model_size_mb']:.2f} MB")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()