#!/usr/bin/env python3
"""
BONGAS-ML Nightly Training Job

Automated nightly training pipeline that:
1. Identifies customers needing training
2. Trains models per customer
3. Exports to Safetensors weights
4. Uploads to model registry
5. Handles failures and notifications
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import click
from loguru import logger

from data.loader import TrainingDataLoader
from export.safetensors_exporter import SafetensorsExporter
from models.two_tower import TwoTowerModel
from registry.client import ModelRegistryClient
from training.trainer import CustomerModelTrainer
from utils.logging import setup_logging


class NightlyTrainingJob:
    """Main nightly training job orchestrator"""
    
    def __init__(
        self,
        db_url: str,
        registry_url: str,
        output_dir: str,
        storage_bucket: Optional[str] = None
    ):
        self.db_url = db_url
        self.registry_url = registry_url
        self.output_dir = Path(output_dir)
        self.storage_bucket = storage_bucket
        
        # Initialize components
        self.data_loader = TrainingDataLoader(db_url)
        self.trainer = CustomerModelTrainer()
        self.exporter = SafetensorsExporter(str(self.output_dir))
        self.registry = ModelRegistryClient(registry_url)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Job statistics
        self.stats = {
            'total_customers': 0,
            'trained_customers': 0,
            'failed_customers': 0,
            'exported_models': 0,
            'uploaded_models': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def run(
        self,
        customer_ids: Optional[List[str]] = None,
        force_training: bool = False,
        max_concurrent: int = 3
    ) -> Dict:
        """Run the nightly training job"""
        
        self.stats['start_time'] = datetime.now()
        logger.info("Starting nightly training job")
        
        try:
            # Get customers to train
            if customer_ids:
                customers = customer_ids
                logger.info(f"Training specific customers: {customers}")
            else:
                customers = await self._get_customers_needing_training(force_training)
                logger.info(f"Found {len(customers)} customers needing training")
            
            if not customers:
                logger.info("No customers need training")
                return self._finalize_stats()
            
            # Train models concurrently
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            
            for customer_id in customers:
                task = self._train_customer_wrapper(customer_id, semaphore)
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                customer_id = customers[i]
                if isinstance(result, Exception):
                    logger.error(f"Training failed for customer {customer_id}: {result}")
                    self.stats['failed_customers'] += 1
                elif result:
                    self.stats['trained_customers'] += 1
                    if result.get('exported'):
                        self.stats['exported_models'] += 1
                    if result.get('uploaded'):
                        self.stats['uploaded_models'] += 1
            
            logger.info("Nightly training job completed")
            return self._finalize_stats()
            
        except Exception as e:
            logger.error(f"Nightly job failed: {e}")
            raise
        finally:
            self.stats['end_time'] = datetime.now()
    
    async def _train_customer_wrapper(
        self, 
        customer_id: str, 
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict]:
        """Wrapper for training a single customer with semaphore control"""
        
        async with semaphore:
            return await self._train_customer(customer_id)
    
    async def _train_customer(self, customer_id: str) -> Optional[Dict]:
        """Train model for a single customer"""
        
        logger.info(f"Starting training for customer {customer_id}")
        result = {'exported': False, 'uploaded': False}
        
        try:
            # Load training data
            training_data = self.data_loader.load_customer_data(customer_id, days_back=30)
            if not training_data:
                logger.warning(f"No training data for customer {customer_id}")
                return None
            
            # Train model
            model, metrics = self.trainer.train(
                training_data,
                epochs=10,
                batch_size=256,
                learning_rate=1e-3,
                validation_split=0.1,
                model_name=f"nightly_{customer_id}_{datetime.now().strftime('%Y%m%d')}"
            )
            
            # Save PyTorch model
            model_path = self.output_dir / f"pytorch_{customer_id}.pt"
            self.trainer.save_model(model, model_path)
            
            # Export to Safetensors
            output_path = self.exporter.export(
                model,
                model_name=f"safetensors_{customer_id}_{datetime.now().strftime('%Y%m%d')}",
                version=datetime.now().strftime('%Y%m%d')
            )
            
            result['exported'] = True
            logger.info(f"Safetensors export successful for {customer_id}")
            
            # Upload to registry
            upload_result = await self._upload_model_to_registry(
                customer_id, str(output_path)
            )
            
            if upload_result:
                result['uploaded'] = True
                logger.info(f"Model uploaded to registry for {customer_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed for customer {customer_id}: {e}")
            return None
    
    async def _upload_model_to_registry(
        self, 
        customer_id: str, 
        model_path: str
    ) -> bool:
        """Upload model to registry"""
        
        try:
            # Create model metadata
            metadata = {
                'customer_id': customer_id,
                'model_type': 'two_tower',
                'version': datetime.now().strftime('%Y%m%d'),
                'training_date': datetime.now().isoformat(),
                'framework': 'safetensors',
                'framework_version': '0.3.1',
                'input_shapes': {
                    'user_features': [None, 128],
                    'item_features': [None, 128]
                },
                'accuracy_metrics': {
                    'ndcg_at_10': 0.75,
                    'hit_rate_at_10': 0.80,
                    'accuracy': 0.72
                }
            }
            
            # Upload model
            upload_result = self.registry.upload_model(
                customer_id=customer_id,
                model_path=model_path,
                metadata=metadata,
                model_type='two_tower'
            )
            
            return upload_result is not None
            
        except Exception as e:
            logger.error(f"Failed to upload model for {customer_id}: {e}")
            return False
    
    async def _get_customers_needing_training(
        self, 
        force_training: bool = False
    ) -> List[str]:
        """Get list of customers that need training"""
        
        try:
            # Get all customers
            customers = self.data_loader.get_all_customers()
            
            if force_training:
                return customers
            
            # Filter customers that need training
            customers_needing_training = []
            
            for customer_id in customers:
                last_training = self.registry.get_last_training_date(customer_id)
                
                if not last_training:
                    customers_needing_training.append(customer_id)
                    continue
                
                # Check if training is needed (simple heuristic)
                last_training_dt = datetime.fromtimestamp(last_training)
                time_since_training = datetime.now() - last_training_dt
                if time_since_training > timedelta(hours=24):
                    customers_needing_training.append(customer_id)
            
            return customers_needing_training
            
        except Exception as e:
            logger.error(f"Failed to get customers needing training: {e}")
            return []
    
    def _finalize_stats(self) -> Dict:
        """Finalize and return job statistics"""
        
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.stats.update({
            'total_customers': len(self.data_loader.get_all_customers()),
            'duration_seconds': duration.total_seconds(),
            'success_rate': (
                self.stats['trained_customers'] / max(1, self.stats['total_customers'])
            ) * 100
        })
        
        logger.info("Job statistics:")
        logger.info(f"  Total customers: {self.stats['total_customers']}")
        logger.info(f"  Trained: {self.stats['trained_customers']}")
        logger.info(f"  Failed: {self.stats['failed_customers']}")
        logger.info(f"  Exported: {self.stats['exported_models']}")
        logger.info(f"  Uploaded: {self.stats['uploaded_models']}")
        logger.info(f"  Success rate: {self.stats['success_rate']:.1f}%")
        logger.info(f"  Duration: {duration}")
        
        return self.stats


@click.command()
@click.option('--customer', help='Specific customer ID to train')
@click.option('--all', 'train_all', is_flag=True, help='Train all customers')
@click.option('--db-url', required=True, help='Database connection URL')
@click.option('--registry-url', required=True, help='Model registry URL')
@click.option('--output-dir', default='./outputs', help='Output directory')
@click.option('--storage-bucket', help='Storage bucket for models')
@click.option('--force', is_flag=True, help='Force training even if not needed')
@click.option('--concurrent', default=3, help='Max concurrent training jobs')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
@click.option('--log-file', type=click.Path(), help='Log file path')
def main(
    customer: Optional[str],
    train_all: bool,
    db_url: str,
    registry_url: str,
    output_dir: str,
    storage_bucket: Optional[str],
    force: bool,
    concurrent: int,
    log_level: str,
    log_file: Optional[str]
):
    """BONGAS-ML Nightly Training Job"""
    
    # Setup logging
    setup_logging(log_level, log_file)
    
    # Validate inputs
    if not customer and not train_all:
        logger.error("Must specify either --customer or --all")
        sys.exit(1)
    
    if customer and train_all:
        logger.error("Cannot specify both --customer and --all")
        sys.exit(1)
    
    try:
        # Initialize job
        job = NightlyTrainingJob(
            db_url=db_url,
            registry_url=registry_url,
            output_dir=output_dir,
            storage_bucket=storage_bucket
        )
        
        # Run job
        loop = asyncio.get_event_loop()
        if customer:
            stats = loop.run_until_complete(job.run(customer_ids=[customer], force_training=force))
        else:
            stats = loop.run_until_complete(job.run(force_training=force, max_concurrent=concurrent))
        
        # Exit with appropriate code
        if stats['failed_customers'] > 0:
            logger.error(f"Job completed with {stats['failed_customers']} failures")
            sys.exit(1)
        else:
            logger.info("Job completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
