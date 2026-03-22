"""
Model Registry Client for BONGAS-ML

Handles communication with bongas-server for model management:
- Model upload and download
- Model metadata management
- Version control and staging
- Customer-specific model handling
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
from loguru import logger

from ..models.base import BaseModel


class ModelRegistryClient:
    """Client for interacting with bongas-server model registry"""
    
    def __init__(
        self,
        registry_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize model registry client
        
        Args:
            registry_url: URL of the model registry API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.registry_url = registry_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # API endpoints
        self.endpoints = {
            'upload': f"{self.registry_url}/api/v1/models/upload",
            'download': f"{self.registry_url}/api/v1/models/download",
            'list': f"{self.registry_url}/api/v1/models/list",
            'metadata': f"{self.registry_url}/api/v1/models/metadata",
            'delete': f"{self.registry_url}/api/v1/models/delete",
            'staging': f"{self.registry_url}/api/v1/models/staging",
            'promote': f"{self.registry_url}/api/v1/models/promote",
            'health': f"{self.registry_url}/api/v1/health"
        }
        
        logger.info(f"Model registry client initialized: {self.registry_url}")
    
    def upload_model(
        self,
        customer_id: str,
        model_path: Union[str, Path],
        metadata: Dict[str, Any],
        model_type: str = 'two_tower',
        version: Optional[str] = None,
        stage: str = 'staging'
    ) -> Optional[Dict[str, Any]]:
        """
        Upload model to registry
        
        Args:
            customer_id: Customer identifier
            model_path: Path to model file
            metadata: Model metadata
            model_type: Type of model
            version: Model version (auto-generated if None)
            stage: Deployment stage ('staging', 'production')
        
        Returns:
            Upload response or None if failed
        """
        
        try:
            model_path = Path(model_path)
            
            # Prepare metadata
            upload_metadata = {
                'customer_id': customer_id,
                'model_type': model_type,
                'version': version or self._generate_version(),
                'stage': stage,
                'framework': 'safetensors',
                'framework_version': '0.3.1',
                'upload_timestamp': time.time(),
                'file_size': model_path.stat().st_size,
                **metadata
            }
            
            # Prepare files for upload
            files = {
                'model_file': (model_path.name, open(model_path, 'rb'), 'application/octet-stream')
            }
            
            # Prepare form data
            data = {
                'metadata': str(upload_metadata)  # Convert to string for form data
            }
            
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            logger.info(f"Uploading model for customer {customer_id}...")
            
            response = requests.post(
                self.endpoints['upload'],
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Model uploaded successfully: {result.get('model_id')}")
                return result
            else:
                logger.error(f"Model upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            return None
        finally:
            # Close file handles
            if 'files' in locals():
                for file_tuple in files.values():
                    file_tuple[1].close()
    
    def download_model(
        self,
        customer_id: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Download model from registry
        
        Args:
            customer_id: Customer identifier
            model_id: Specific model ID (if None, uses latest)
            version: Model version
            stage: Deployment stage
            output_path: Output path for downloaded model
        
        Returns:
            Path to downloaded model or None if failed
        """
        
        try:
            # Build query parameters
            params = {'customer_id': customer_id}
            if model_id:
                params['model_id'] = model_id
            if version:
                params['version'] = version
            if stage:
                params['stage'] = stage
            
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            logger.info(f"Downloading model for customer {customer_id}...")
            
            response = requests.get(
                self.endpoints['download'],
                params=params,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 200:
                # Determine output path
                if output_path is None:
                    filename = response.headers.get('Content-Disposition', '').split('filename=')[-1]
                    if filename:
                        output_path = Path(filename.strip('"'))
                    else:
                        output_path = Path(f"model_{customer_id}_{int(time.time())}.safetensors")
                
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Model downloaded to: {output_path}")
                return output_path
            else:
                logger.error(f"Model download failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return None
    
    def list_models(
        self,
        customer_id: Optional[str] = None,
        model_type: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List models in registry
        
        Args:
            customer_id: Filter by customer
            model_type: Filter by model type
            stage: Filter by deployment stage
        
        Returns:
            List of model metadata
        """
        
        try:
            # Build query parameters
            params = {}
            if customer_id:
                params['customer_id'] = customer_id
            if model_type:
                params['model_type'] = model_type
            if stage:
                params['stage'] = stage
            
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            logger.info("Listing models...")
            
            response = requests.get(
                self.endpoints['list'],
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Found {len(models)} models")
                return models
            else:
                logger.error(f"List models failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"List models failed: {e}")
            return []
    
    def get_model_metadata(
        self,
        customer_id: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata
        
        Args:
            customer_id: Customer identifier
            model_id: Specific model ID
            version: Model version
            stage: Deployment stage
        
        Returns:
            Model metadata or None if not found
        """
        
        try:
            # Build query parameters
            params = {'customer_id': customer_id}
            if model_id:
                params['model_id'] = model_id
            if version:
                params['version'] = version
            if stage:
                params['stage'] = stage
            
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            logger.info(f"Getting metadata for customer {customer_id}...")
            
            response = requests.get(
                self.endpoints['metadata'],
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                metadata = response.json()
                logger.info(f"Retrieved metadata for model: {metadata.get('model_id')}")
                return metadata
            else:
                logger.error(f"Get metadata failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Get metadata failed: {e}")
            return None
    
    def delete_model(
        self,
        customer_id: str,
        model_id: str
    ) -> bool:
        """
        Delete model from registry
        
        Args:
            customer_id: Customer identifier
            model_id: Model ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            data = {
                'customer_id': customer_id,
                'model_id': model_id
            }
            
            logger.info(f"Deleting model {model_id} for customer {customer_id}...")
            
            response = requests.delete(
                self.endpoints['delete'],
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_id} deleted successfully")
                return True
            else:
                logger.error(f"Delete model failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Delete model failed: {e}")
            return False
    
    def promote_model(
        self,
        customer_id: str,
        model_id: str,
        from_stage: str,
        to_stage: str
    ) -> bool:
        """
        Promote model between stages
        
        Args:
            customer_id: Customer identifier
            model_id: Model ID to promote
            from_stage: Current stage
            to_stage: Target stage
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Add authentication headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            data = {
                'customer_id': customer_id,
                'model_id': model_id,
                'from_stage': from_stage,
                'to_stage': to_stage
            }
            
            logger.info(f"Promoting model {model_id} from {from_stage} to {to_stage}...")
            
            response = requests.post(
                self.endpoints['promote'],
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_id} promoted successfully")
                return True
            else:
                logger.error(f"Promote model failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Promote model failed: {e}")
            return False
    
    def get_last_training_date(
        self,
        customer_id: str,
        model_type: str = 'two_tower'
    ) -> Optional[float]:
        """
        Get last training date for customer
        
        Args:
            customer_id: Customer identifier
            model_type: Type of model
        
        Returns:
            Unix timestamp of last training or None if not found
        """
        
        try:
            models = self.list_models(
                customer_id=customer_id,
                model_type=model_type,
                stage='production'
            )
            
            if not models:
                return None
            
            # Get the most recent model
            latest_model = max(models, key=lambda x: x.get('upload_timestamp', 0))
            return latest_model.get('upload_timestamp')
            
        except Exception as e:
            logger.error(f"Get last training date failed: {e}")
            return None
    
    def check_health(self) -> bool:
        """Check registry health"""
        
        try:
            response = requests.get(self.endpoints['health'], timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _generate_version(self) -> str:
        """Generate version string based on timestamp"""
        return f"v{int(time.time())}"
    
    def get_customer_models(
        self,
        customer_id: str,
        include_staging: bool = True,
        include_production: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all models for a customer organized by stage
        
        Args:
            customer_id: Customer identifier
            include_staging: Include staging models
            include_production: Include production models
        
        Returns:
            Dictionary with models organized by stage
        """
        
        stages = []
        if include_staging:
            stages.append('staging')
        if include_production:
            stages.append('production')
        
        result = {}
        for stage in stages:
            models = self.list_models(
                customer_id=customer_id,
                stage=stage
            )
            result[stage] = models
        
        return result
    
    def download_latest_model(
        self,
        customer_id: str,
        stage: str = 'production',
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Download the latest model for a customer
        
        Args:
            customer_id: Customer identifier
            stage: Deployment stage
            output_path: Output path for downloaded model
        
        Returns:
            Path to downloaded model or None if failed
        """
        
        try:
            models = self.list_models(
                customer_id=customer_id,
                stage=stage
            )
            
            if not models:
                logger.warning(f"No models found for customer {customer_id} in stage {stage}")
                return None
            
            # Get the most recent model
            latest_model = max(models, key=lambda x: x.get('upload_timestamp', 0))
            model_id = latest_model['model_id']
            
            return self.download_model(
                customer_id=customer_id,
                model_id=model_id,
                output_path=output_path
            )
            
        except Exception as e:
            logger.error(f"Download latest model failed: {e}")
            return None