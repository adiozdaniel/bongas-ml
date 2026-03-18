"""
ONNX Model Validation for BONGAS-ML

Provides comprehensive validation of ONNX models including:
- Accuracy validation against PyTorch models
- Performance benchmarking
- Model integrity checks
- Compatibility validation
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger


def validate_onnx_model(
    onnx_path: Union[str, Path],
    dummy_user_features: torch.Tensor,
    dummy_item_features: torch.Tensor,
    pytorch_model: torch.nn.Module,
    num_test_samples: int = 100,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Validate ONNX model against PyTorch model
    
    Args:
        onnx_path: Path to ONNX model
        dummy_user_features: Sample user features
        dummy_item_features: Sample item features
        pytorch_model: Reference PyTorch model
        num_test_samples: Number of test samples to generate
        tolerance: Tolerance for output comparison
    
    Returns:
        Dictionary containing validation results
    """
    
    onnx_path = Path(onnx_path)
    
    try:
        # Load ONNX model
        logger.info(f"Loading ONNX model: {onnx_path}")
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Set PyTorch model to evaluation mode
        pytorch_model.eval()
        
        # Generate test data
        logger.info(f"Generating {num_test_samples} test samples...")
        test_data = _generate_test_data(
            dummy_user_features, dummy_item_features, num_test_samples
        )
        
        # Run validation
        logger.info("Running model validation...")
        validation_results = _validate_model_outputs(
            ort_session, pytorch_model, test_data, tolerance
        )
        
        # Run performance benchmark
        logger.info("Running performance benchmark...")
        performance_results = _benchmark_model_performance(ort_session, test_data)
        
        # Run integrity checks
        logger.info("Running model integrity checks...")
        integrity_results = _check_model_integrity(onnx_path, ort_session)
        
        # Combine results
        results = {
            'passed': validation_results['passed'],
            'validation': validation_results,
            'performance': performance_results,
            'integrity': integrity_results,
            'validation_timestamp': str(onnx_path.stat().st_mtime),
            'num_test_samples': num_test_samples,
            'tolerance': tolerance
        }
        
        if results['passed']:
            logger.info("ONNX model validation PASSED")
        else:
            logger.warning("ONNX model validation FAILED")
        
        return results
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return {
            'passed': False,
            'error': str(e),
            'validation': {},
            'performance': {},
            'integrity': {}
        }


def _generate_test_data(
    dummy_user_features: torch.Tensor,
    dummy_item_features: torch.Tensor,
    num_samples: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate test data for validation"""
    
    user_dim = dummy_user_features.shape[1]
    item_dim = dummy_item_features.shape[1]
    
    test_data = []
    for _ in range(num_samples):
        # Generate random features
        user_features = np.random.randn(user_dim).astype(np.float32)
        item_features = np.random.randn(item_dim).astype(np.float32)
        test_data.append((user_features, item_features))
    
    return test_data


def _validate_model_outputs(
    ort_session: ort.InferenceSession,
    pytorch_model: torch.nn.Module,
    test_data: List[Tuple[np.ndarray, np.ndarray]],
    tolerance: float
) -> Dict[str, Any]:
    """Validate model outputs match between ONNX and PyTorch"""
    
    onnx_inputs = [inp.name for inp in ort_session.get_inputs()]
    
    metrics = {
        'mse': [],
        'max_difference': [],
        'mean_difference': [],
        'relative_error': [],
        'passed_samples': 0,
        'total_samples': len(test_data)
    }
    
    for user_features, item_features in test_data:
        try:
            # Run PyTorch model
            with torch.no_grad():
                pytorch_output = pytorch_model(
                    torch.tensor(user_features).unsqueeze(0),
                    torch.tensor(item_features).unsqueeze(0)
                )
                pytorch_output = pytorch_output.detach().cpu().numpy().flatten()
            
            # Run ONNX model
            onnx_input_dict = {
                onnx_inputs[0]: user_features.reshape(1, -1),
                onnx_inputs[1]: item_features.reshape(1, -1)
            }
            onnx_output = ort_session.run(None, onnx_input_dict)[0].flatten()
            
            # Calculate metrics
            mse = np.mean((pytorch_output - onnx_output) ** 2)
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            rel_error = np.mean(np.abs((pytorch_output - onnx_output) / (pytorch_output + 1e-8)))
            
            metrics['mse'].append(mse)
            metrics['max_difference'].append(max_diff)
            metrics['mean_difference'].append(mean_diff)
            metrics['relative_error'].append(rel_error)
            
            # Check if sample passed
            if np.allclose(pytorch_output, onnx_output, atol=tolerance, rtol=tolerance):
                metrics['passed_samples'] += 1
                
        except Exception as e:
            logger.warning(f"Sample validation failed: {e}")
    
    # Calculate aggregate metrics
    avg_mse = np.mean(metrics['mse']) if metrics['mse'] else float('inf')
    avg_max_diff = np.mean(metrics['max_difference']) if metrics['max_difference'] else float('inf')
    avg_mean_diff = np.mean(metrics['mean_difference']) if metrics['mean_difference'] else float('inf')
    avg_rel_error = np.mean(metrics['relative_error']) if metrics['relative_error'] else float('inf')
    
    passed = metrics['passed_samples'] / metrics['total_samples'] >= 0.95
    
    return {
        'passed': passed,
        'passed_samples': metrics['passed_samples'],
        'total_samples': metrics['total_samples'],
        'pass_rate': metrics['passed_samples'] / metrics['total_samples'],
        'avg_mse': float(avg_mse),
        'avg_max_difference': float(avg_max_diff),
        'avg_mean_difference': float(avg_mean_diff),
        'avg_relative_error': float(avg_rel_error),
        'tolerance': tolerance
    }


def _benchmark_model_performance(
    ort_session: ort.InferenceSession,
    test_data: List[Tuple[np.ndarray, np.ndarray]],
    num_warmup_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark model performance"""
    
    onnx_inputs = [inp.name for inp in ort_session.get_inputs()]
    
    # Warmup runs
    logger.info(f"Running {num_warmup_runs} warmup iterations...")
    for _ in range(num_warmup_runs):
        sample = test_data[0]
        onnx_input_dict = {
            onnx_inputs[0]: sample[0].reshape(1, -1),
            onnx_inputs[1]: sample[1].reshape(1, -1)
        }
        ort_session.run(None, onnx_input_dict)
    
    # Performance benchmark
    logger.info("Running performance benchmark...")
    inference_times = []
    
    for user_features, item_features in test_data:
        onnx_input_dict = {
            onnx_inputs[0]: user_features.reshape(1, -1),
            onnx_inputs[1]: item_features.reshape(1, -1)
        }
        
        # Measure inference time
        start_time = time.perf_counter()
        ort_session.run(None, onnx_input_dict)
        end_time = time.perf_counter()
        
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    
    performance = {
        'avg_inference_time_ms': float(np.mean(inference_times)),
        'min_inference_time_ms': float(np.min(inference_times)),
        'max_inference_time_ms': float(np.max(inference_times)),
        'std_inference_time_ms': float(np.std(inference_times)),
        'p50_inference_time_ms': float(np.percentile(inference_times, 50)),
        'p95_inference_time_ms': float(np.percentile(inference_times, 95)),
        'p99_inference_time_ms': float(np.percentile(inference_times, 99)),
        'total_samples': len(test_data),
        'throughput_samples_per_sec': float(len(test_data) / (np.sum(inference_times) / 1000))
    }
    
    return performance


def _check_model_integrity(
    onnx_path: Path,
    ort_session: ort.InferenceSession
) -> Dict[str, Any]:
    """Check model integrity and compatibility"""
    
    try:
        # Load ONNX model for inspection
        model = onnx.load(str(onnx_path))
        
        # Basic integrity checks
        integrity = {
            'model_loaded': True,
            'session_created': True,
            'model_size_mb': onnx_path.stat().st_size / (1024 * 1024),
            'opset_version': model.opset_import[0].version if model.opset_import else None,
            'ir_version': model.ir_version,
            'num_nodes': len(model.graph.node),
            'num_inputs': len(model.graph.input),
            'num_outputs': len(model.graph.output),
            'num_initializers': len(model.graph.initializer),
        }
        
        # Check for potential issues
        issues = []
        
        # Check for unsupported operations
        supported_ops = ort.get_available_providers()
        if 'CUDAExecutionProvider' in supported_ops:
            logger.info("CUDA execution provider available")
        else:
            logger.info("CUDA execution provider not available")
        
        # Check model structure
        if integrity['num_nodes'] == 0:
            issues.append("Model has no nodes")
        
        if integrity['num_inputs'] == 0:
            issues.append("Model has no inputs")
        
        if integrity['num_outputs'] == 0:
            issues.append("Model has no outputs")
        
        integrity['issues'] = issues
        integrity['has_issues'] = len(issues) > 0
        
        return integrity
        
    except Exception as e:
        logger.error(f"Model integrity check failed: {e}")
        return {
            'model_loaded': False,
            'session_created': False,
            'error': str(e),
            'issues': [str(e)],
            'has_issues': True
        }


def validate_model_compatibility(
    onnx_path: Union[str, Path],
    target_devices: List[str] = ['CPU', 'CUDA']
) -> Dict[str, Any]:
    """Validate model compatibility with different execution providers"""
    
    onnx_path = Path(onnx_path)
    compatibility = {}
    
    for device in target_devices:
        try:
            if device == 'CUDA' and not ort.get_device() == 'GPU':
                compatibility[device] = {
                    'available': False,
                    'error': 'CUDA not available on this system'
                }
                continue
            
            # Try to create session with specific provider
            providers = [f'{device}ExecutionProvider']
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Test inference
            input_info = session.get_inputs()[0]
            test_input = np.random.randn(*input_info.shape).astype(np.float32)
            
            if len(session.get_inputs()) > 1:
                # Multi-input model
                test_inputs = {inp.name: np.random.randn(*inp.shape).astype(np.float32) 
                             for inp in session.get_inputs()}
            else:
                # Single input model
                test_inputs = {input_info.name: test_input}
            
            session.run(None, test_inputs)
            
            compatibility[device] = {
                'available': True,
                'test_passed': True,
                'providers': session.get_providers()
            }
            
        except Exception as e:
            compatibility[device] = {
                'available': False,
                'error': str(e),
                'test_passed': False
            }
    
    return compatibility


def generate_validation_report(
    validation_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Generate a human-readable validation report"""
    
    report_lines = [
        "# ONNX Model Validation Report",
        "",
        "## Summary",
        f"- **Validation Status**: {'PASSED' if validation_results['passed'] else 'FAILED'}",
        f"- **Pass Rate**: {validation_results['validation']['pass_rate']:.1%}",
        f"- **Test Samples**: {validation_results['num_test_samples']}",
        "",
        "## Validation Metrics",
        f"- **Average MSE**: {validation_results['validation']['avg_mse']:.2e}",
        f"- **Average Max Difference**: {validation_results['validation']['avg_max_difference']:.2e}",
        f"- **Average Mean Difference**: {validation_results['validation']['avg_mean_difference']:.2e}",
        f"- **Average Relative Error**: {validation_results['validation']['avg_relative_error']:.2%}",
        "",
        "## Performance Metrics",
        f"- **Average Inference Time**: {validation_results['performance']['avg_inference_time_ms']:.2f} ms",
        f"- **P95 Inference Time**: {validation_results['performance']['p95_inference_time_ms']:.2f} ms",
        f"- **Throughput**: {validation_results['performance']['throughput_samples_per_sec']:.2f} samples/sec",
        "",
        "## Model Integrity",
        f"- **Model Size**: {validation_results['integrity']['model_size_mb']:.2f} MB",
        f"- **Opset Version**: {validation_results['integrity']['opset_version']}",
        f"- **Number of Nodes**: {validation_results['integrity']['num_nodes']}",
        f"- **Issues Found**: {len(validation_results['integrity']['issues'])}",
    ]
    
    if validation_results['integrity']['issues']:
        report_lines.extend([
            "",
            "### Issues",
        ])
        for issue in validation_results['integrity']['issues']:
            report_lines.append(f"- {issue}")
    
    report = "\n".join(report_lines)
    
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to: {output_path}")
    
    return report