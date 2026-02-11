"""
BONGAS-ML Command Line Interface

Provides CLI commands for training, exporting, and managing models.
"""

import argparse
import sys
from pathlib import Path

from .utils import setup_logging, get_logger
from .training import Trainer
from .export import ONNXExporter
from .registry import ModelRegistryClient
from .validation import AccuracyValidator


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="BONGAS-ML: Machine Learning for BONGAS-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m bongas_ml train --data data.csv --model-type two_tower --epochs 100
  
  # Export model to ONNX
  python -m bongas_ml export --model model.pth --output model.onnx
  
  # Validate model
  python -m bongas_ml validate --model model.onnx --test-data test.csv
  
  # Upload model to registry
  python -m bongas_ml registry upload --model model.onnx --customer customer123
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', required=True, help='Training data file')
    train_parser.add_argument('--model-type', default='two_tower', help='Model type')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--output', required=True, help='Output model file')
    train_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('--model', required=True, help='Input model file')
    export_parser.add_argument('--output', required=True, help='Output ONNX file')
    export_parser.add_argument('--optimize', action='store_true', help='Optimize model')
    export_parser.add_argument('--quantize', action='store_true', help='Quantize model')
    export_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model')
    validate_parser.add_argument('--model', required=True, help='Model file')
    validate_parser.add_argument('--test-data', required=True, help='Test data file')
    validate_parser.add_argument('--thresholds', help='Validation thresholds file')
    validate_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Registry commands
    registry_parser = subparsers.add_parser('registry', help='Model registry operations')
    registry_subparsers = registry_parser.add_subparsers(dest='registry_command')
    
    # Registry upload
    upload_parser = registry_subparsers.add_parser('upload', help='Upload model to registry')
    upload_parser.add_argument('--model', required=True, help='Model file')
    upload_parser.add_argument('--customer', required=True, help='Customer ID')
    upload_parser.add_argument('--registry-url', required=True, help='Registry URL')
    upload_parser.add_argument('--api-key', help='API key')
    upload_parser.add_argument('--model-type', default='two_tower', help='Model type')
    upload_parser.add_argument('--stage', default='staging', help='Deployment stage')
    upload_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Registry download
    download_parser = registry_subparsers.add_parser('download', help='Download model from registry')
    download_parser.add_argument('--customer', required=True, help='Customer ID')
    download_parser.add_argument('--output', required=True, help='Output file')
    download_parser.add_argument('--registry-url', required=True, help='Registry URL')
    download_parser.add_argument('--api-key', help='API key')
    download_parser.add_argument('--model-id', help='Specific model ID')
    download_parser.add_argument('--stage', default='production', help='Deployment stage')
    download_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Registry list
    list_parser = registry_subparsers.add_parser('list', help='List models in registry')
    list_parser.add_argument('--customer', help='Customer ID')
    list_parser.add_argument('--registry-url', required=True, help='Registry URL')
    list_parser.add_argument('--api-key', help='API key')
    list_parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)
    
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'export':
            export_model(args)
        elif args.command == 'validate':
            validate_model(args)
        elif args.command == 'registry':
            handle_registry_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def train_model(args):
    """Train a model"""
    from .models import TwoTowerModel
    
    logger = get_logger(__name__)
    logger.info(f"Starting model training: {args.model_type}")
    
    # Load data
    logger.info(f"Loading data from: {args.data}")
    # TODO: Implement data loading
    
    # Create model
    model = TwoTowerModel(
        user_feature_dim=128,  # TODO: Infer from data
        item_feature_dim=64,
        embedding_dim=32
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Train model
    logger.info("Training model...")
    # TODO: Implement training
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    
    logger.info(f"Model training completed: {output_path}")


def export_model(args):
    """Export model to ONNX"""
    logger = get_logger(__name__)
    logger.info(f"Exporting model: {args.model}")
    
    # Load model
    # TODO: Implement model loading
    
    # Create exporter
    exporter = ONNXExporter()
    
    # Export model
    result = exporter.export(
        model_path=args.model,
        output_path=args.output,
        optimize=args.optimize,
        quantize=args.quantize
    )
    
    if result['success']:
        logger.info(f"Model exported successfully: {args.output}")
    else:
        logger.error(f"Model export failed: {result['error']}")
        sys.exit(1)


def validate_model(args):
    """Validate model"""
    logger = get_logger(__name__)
    logger.info(f"Validating model: {args.model}")
    
    # Load model
    # TODO: Implement model loading
    
    # Load test data
    # TODO: Implement data loading
    
    # Create validator
    validator = AccuracyValidator()
    
    # Validate model
    passed, metrics = validator.validate(
        model=None,  # TODO: Load model
        test_data=None  # TODO: Load test data
    )
    
    if passed:
        logger.info("Model validation PASSED")
    else:
        logger.error("Model validation FAILED")
        sys.exit(1)


def handle_registry_command(args):
    """Handle registry commands"""
    logger = get_logger(__name__)
    
    # Create registry client
    client = ModelRegistryClient(
        registry_url=args.registry_url,
        api_key=args.api_key
    )
    
    if args.registry_command == 'upload':
        # Upload model
        logger.info(f"Uploading model for customer: {args.customer}")
        
        metadata = {
            'description': f'Model uploaded via CLI for customer {args.customer}',
            'source': 'CLI',
            'version': '1.0.0'
        }
        
        result = client.upload_model(
            customer_id=args.customer,
            model_path=args.model,
            metadata=metadata,
            model_type=args.model_type,
            stage=args.stage
        )
        
        if result:
            logger.info(f"Model uploaded successfully: {result.get('model_id')}")
        else:
            logger.error("Model upload failed")
            sys.exit(1)
    
    elif args.registry_command == 'download':
        # Download model
        logger.info(f"Downloading model for customer: {args.customer}")
        
        result = client.download_model(
            customer_id=args.customer,
            model_id=args.model_id,
            stage=args.stage,
            output_path=args.output
        )
        
        if result:
            logger.info(f"Model downloaded successfully: {result}")
        else:
            logger.error("Model download failed")
            sys.exit(1)
    
    elif args.registry_command == 'list':
        # List models
        logger.info("Listing models...")
        
        models = client.list_models(
            customer_id=args.customer
        )
        
        logger.info(f"Found {len(models)} models:")
        for model in models:
            logger.info(f"  - {model['model_id']} ({model['stage']})")
    
    else:
        logger.error(f"Unknown registry command: {args.registry_command}")
        sys.exit(1)


if __name__ == '__main__':
    main()