"""
ONNX Exporter for BONGAS-ML

Handles PyTorch to ONNX conversion with optimization, validation,
and model size reduction for production deployment.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger

from ..models.base import BaseModel
from ..models.two_tower import TwoTowerModel
from .optimize import optimize_onnx_model
from .validate import validate_onnx_model


class ONNXExporter:
    """ONNX model exporter with optimization and validation"""
    
    def __init__(self, output_dir: str):
        """
        Initialize ONNX exporter
        
        Args:
            output_dir: Directory to save exported models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ONNX exporter initialized with output directory: {self.output_dir}")
    
    def export_two_tower(
        self,
        model: Union[BaseModel, TwoTowerModel],
        model_name: str,
        user_feature_dim: int = 128,
        item_feature_dim: int = 128,
        optimize: bool = True,
        validate: bool = True,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> Dict[str, Any]:
        """
        Export Two-Tower model to ONNX format
        
        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            user_feature_dim: User feature dimension
            item_feature_dim: Item feature dimension
            optimize: Whether to optimize the ONNX model
            validate: Whether to validate the ONNX model
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes configuration
        
        Returns:
            Dictionary containing export metadata
        """
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy inputs
        dummy_user_features = torch.randn(1, user_feature_dim)
        dummy_item_features = torch.randn(1, item_feature_dim)
        
        # Set default names if not provided
        if input_names is None:
            input_names = ['user_features', 'item_features']
        if output_names is None:
            output_names = ['similarity_scores']
        if dynamic_axes is None:
            dynamic_axes = {
                'user_features': {0: 'batch_size'},
                'item_features': {0: 'batch_size'},
                'similarity_scores': {0: 'batch_size'}
            }
        
        # Create ONNX path
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        try:
            # Export to ONNX
            logger.info(f"Exporting model to ONNX: {onnx_path}")
            
            torch.onnx.export(
                model,
                (dummy_user_features, dummy_item_features),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info("ONNX export completed successfully")
            
            # Get model metadata
            metadata = self._get_model_metadata(onnx_path, model)
            
            # Optimize model if requested
            if optimize:
                logger.info("Optimizing ONNX model...")
                optimized_path = self.output_dir / f"{model_name}_optimized.onnx"
                optimize_onnx_model(str(onnx_path), str(optimized_path))
                onnx_path = optimized_path
                metadata['optimized'] = True
                metadata['optimized_path'] = str(optimized_path)
            
            # Validate model if requested
            if validate:
                logger.info("Validating ONNX model...")
                validation_result = validate_onnx_model(
                    str(onnx_path),
                    dummy_user_features,
                    dummy_item_features,
                    model
                )
                
                metadata['validation'] = validation_result
                
                if not validation_result['passed']:
                    logger.warning("ONNX validation failed, but continuing...")
                else:
                    logger.info("ONNX validation passed")
            
            # Update metadata with final path
            metadata['onnx_path'] = str(onnx_path)
            
            logger.info(f"Model export completed: {onnx_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def export_model(
        self,
        model: BaseModel,
        model_name: str,
        dummy_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        optimize: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Generic model export to ONNX format
        
        Args:
            model: Trained PyTorch model
            model_name: Name for the exported model
            dummy_inputs: Dummy inputs for tracing
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes configuration
            optimize: Whether to optimize the ONNX model
            validate: Whether to validate the ONNX model
        
        Returns:
            Dictionary containing export metadata
        """
        
        # Set model to evaluation mode
        model.eval()
        
        # Set default names if not provided
        if input_names is None:
            input_names = [f'input_{i}' for i in range(len(dummy_inputs) if isinstance(dummy_inputs, tuple) else 1)]
        if output_names is None:
            output_names = ['output']
        if dynamic_axes is None:
            dynamic_axes = {}
        
        # Create ONNX path
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        try:
            # Export to ONNX
            logger.info(f"Exporting model to ONNX: {onnx_path}")
            
            torch.onnx.export(
                model,
                dummy_inputs,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info("ONNX export completed successfully")
            
            # Get model metadata
            metadata = self._get_model_metadata(onnx_path, model)
            
            # Optimize model if requested
            if optimize:
                logger.info("Optimizing ONNX model...")
                optimized_path = self.output_dir / f"{model_name}_optimized.onnx"
                optimize_onnx_model(str(onnx_path), str(optimized_path))
                onnx_path = optimized_path
                metadata['optimized'] = True
                metadata['optimized_path'] = str(optimized_path)
            
            # Validate model if requested
            if validate:
                logger.info("Validating ONNX model...")
                validation_result = self._validate_generic_model(
                    str(onnx_path),
                    dummy_inputs,
                    model
                )
                
                metadata['validation'] = validation_result
                
                if not validation_result['passed']:
                    logger.warning("ONNX validation failed, but continuing...")
                else:
                    logger.info("ONNX validation passed")
            
            # Update metadata with final path
            metadata['onnx_path'] = str(onnx_path)
            
            logger.info(f"Model export completed: {onnx_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _get_model_metadata(self, onnx_path: Path, model: BaseModel) -> Dict[str, Any]:
        """Get model metadata"""
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Calculate model size
        model_size_bytes = onnx_path.stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Get model info
        model_info = model.get_model_info()
        
        metadata = {
            'model_name': onnx_path.stem,
            'onnx_path': str(onnx_path),
            'model_size_bytes': model_size_bytes,
            'model_size_mb': model_size_mb,
            'pytorch_model_info': model_info,
            'onnx_opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None,
            'optimized': False,
            'validation': None,
            'export_timestamp': str(onnx_path.stat().st_mtime)
        }
        
        return metadata
    
    def _validate_generic_model(
        self,
        onnx_path: str,
        dummy_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        pytorch_model: BaseModel
    ) -> Dict[str, Any]:
        """Validate generic ONNX model against PyTorch model"""
        
        try:
            # Prepare inputs
            if isinstance(dummy_inputs, torch.Tensor):
                dummy_inputs = (dummy_inputs,)
            
            # Convert to numpy
            numpy_inputs = [inp.detach().cpu().numpy() for inp in dummy_inputs]
            
            # Run PyTorch model
            with torch.no_grad():
                pytorch_outputs = pytorch_model(*dummy_inputs)
                if isinstance(pytorch_outputs, torch.Tensor):
                    pytorch_outputs = [pytorch_outputs]
                pytorch_outputs = [out.detach().cpu().numpy() for out in pytorch_outputs]
            
            # Run ONNX model
            ort_session = ort.InferenceSession(onnx_path)
            onnx_inputs = {name: inp for name, inp in zip(ort_session.get_inputs(), numpy_inputs)}
            onnx_outputs = ort_session.run(None, onnx_inputs)
            
            # Compare outputs
            validation_results = []
            for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
                # Calculate metrics
                mse = np.mean((pytorch_out - onnx_out) ** 2)
                max_diff = np.max(np.abs(pytorch_out - onnx_out))
                mean_diff = np.mean(np.abs(pytorch_out - onnx_out))
                
                # Check if outputs match within tolerance
                tolerance = 1e-4
                passed = np.allclose(pytorch_out, onnx_out, atol=tolerance, rtol=tolerance)
                
                validation_results.append({
                    'output_index': i,
                    'passed': passed,
                    'mse': float(mse),
                    'max_difference': float(max_diff),
                    'mean_difference': float(mean_diff),
                    'tolerance': tolerance
                })
            
            # Overall validation result
            all_passed = all(result['passed'] for result in validation_results)
            
            return {
                'passed': all_passed,
                'outputs': validation_results,
                'num_outputs': len(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'outputs': []
            }
    
    def batch_export(
        self,
        models: Dict[str, BaseModel],
        export_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch export multiple models
        
        Args:
            models: Dictionary of model_name -> model
            export_config: Configuration for export
        
        Returns:
            Dictionary of export results
        """
        
        results = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"Exporting model: {model_name}")
                
                # Get model-specific config
                model_config = export_config.get(model_name, export_config)
                
                # Export model
                result = self.export_model(
                    model=model,
                    model_name=model_name,
                    **model_config
                )
                
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to export model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def cleanup_old_models(
        self,
        keep_latest: int = 5,
        model_pattern: str = "*.onnx"
    ) -> None:
        """
        Clean up old model files
        
        Args:
            keep_latest: Number of latest models to keep
            model_pattern: Pattern to match model files
        """
        
        model_files = list(self.output_dir.glob(model_pattern))
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(model_files) <= keep_latest:
            logger.info("No old models to clean up")
            return
        
        files_to_delete = model_files[keep_latest:]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                logger.info(f"Deleted old model: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {len(files_to_delete)} old model files")