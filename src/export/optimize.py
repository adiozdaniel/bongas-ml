"""
ONNX Model Optimization for BONGAS-ML

Provides model optimization techniques including:
- Graph optimization
- Quantization
- Pruning
- Model size reduction
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import onnx
from onnx import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from loguru import logger


def optimize_onnx_model(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    optimization_level: str = 'basic',
    quantize: bool = True,
    quantization_type: str = 'int8'
) -> Dict[str, Any]:
    """
    Optimize ONNX model with various techniques
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
        optimization_level: Level of optimization ('basic', 'extended', 'all')
        quantize: Whether to apply quantization
        quantization_type: Type of quantization ('int8', 'uint8')
    
    Returns:
        Dictionary containing optimization metadata
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load model
    logger.info(f"Loading ONNX model: {input_path}")
    model = onnx.load(str(input_path))
    
    # Get original model info
    original_size = input_path.stat().st_size
    original_size_mb = original_size / (1024 * 1024)
    
    # Apply graph optimizations
    optimized_model = _apply_graph_optimizations(model, optimization_level)
    
    # Apply quantization if requested
    if quantize:
        logger.info(f"Applying {quantization_type} quantization...")
        optimized_model = _apply_quantization(optimized_model, quantization_type)
    
    # Save optimized model
    logger.info(f"Saving optimized model: {output_path}")
    onnx.save(optimized_model, str(output_path))
    
    # Get optimized model info
    optimized_size = output_path.stat().st_size
    optimized_size_mb = optimized_size / (1024 * 1024)
    size_reduction = (original_size - optimized_size) / original_size * 100
    
    metadata = {
        'original_size_bytes': original_size,
        'original_size_mb': original_size_mb,
        'optimized_size_bytes': optimized_size,
        'optimized_size_mb': optimized_size_mb,
        'size_reduction_percent': size_reduction,
        'optimization_level': optimization_level,
        'quantized': quantize,
        'quantization_type': quantization_type,
        'optimization_timestamp': str(output_path.stat().st_mtime)
    }
    
    logger.info(f"Model optimization completed:")
    logger.info(f"  Original size: {original_size_mb:.2f} MB")
    logger.info(f"  Optimized size: {optimized_size_mb:.2f} MB")
    logger.info(f"  Size reduction: {size_reduction:.1f}%")
    
    return metadata


def _apply_graph_optimizations(
    model: onnx.ModelProto,
    optimization_level: str
) -> onnx.ModelProto:
    """Apply ONNX graph optimizations"""
    
    logger.info(f"Applying {optimization_level} graph optimizations...")
    
    # Get available optimization passes
    all_passes = optimizer.get_available_passes()
    
    # Select passes based on optimization level
    if optimization_level == 'basic':
        passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'extract_constant_to_initializer',
            'eliminate_unused_initializer'
        ]
    elif optimization_level == 'extended':
        passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'extract_constant_to_initializer',
            'eliminate_unused_initializer',
            'eliminate_deadend',
            'eliminate_nop_transpose',
            'fuse_add_bias_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_log_softmax',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm'
        ]
    elif optimization_level == 'all':
        passes = all_passes
    else:
        raise ValueError(f"Unknown optimization level: {optimization_level}")
    
    # Filter passes that are available
    available_passes = [p for p in passes if p in all_passes]
    
    if not available_passes:
        logger.warning("No optimization passes available")
        return model
    
    # Apply optimizations
    try:
        optimized_model = optimizer.optimize(model, passes=available_passes)
        logger.info(f"Applied {len(available_passes)} optimization passes")
        return optimized_model
    except Exception as e:
        logger.error(f"Graph optimization failed: {e}")
        return model


def _apply_quantization(
    model: onnx.ModelProto,
    quantization_type: str
) -> onnx.ModelProto:
    """Apply quantization to ONNX model"""
    
    try:
        # Convert quantization type string to enum
        if quantization_type == 'int8':
            qtype = QuantType.QInt8
        elif quantization_type == 'uint8':
            qtype = QuantType.QUInt8
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Create temporary files for quantization
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save model to temporary file
        onnx.save(model, temp_path)
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model_input=temp_path,
            model_output=temp_path,
            weight_type=qtype,
            optimize_model=True
        )
        
        # Clean up temporary file
        import os
        os.unlink(temp_path)
        
        logger.info(f"Applied {quantization_type} quantization")
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return model


def get_model_statistics(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Get detailed statistics about ONNX model"""
    
    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    
    # Basic info
    stats = {
        'model_path': str(model_path),
        'model_size_bytes': model_path.stat().st_size,
        'opset_version': model.opset_import[0].version if model.opset_import else None,
        'ir_version': model.ir_version,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'domain': model.domain,
        'model_version': model.model_version,
    }
    
    # Graph info
    graph = model.graph
    stats.update({
        'num_nodes': len(graph.node),
        'num_inputs': len(graph.input),
        'num_outputs': len(graph.output),
        'num_initializers': len(graph.initializer),
        'num_value_info': len(graph.value_info),
    })
    
    # Node types
    node_types = {}
    for node in graph.node:
        node_type = node.op_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    stats['node_types'] = node_types
    
    # Input/output info
    input_info = []
    for inp in graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in inp.type.tensor_type.shape.dim]
        input_info.append({
            'name': inp.name,
            'type': inp.type.tensor_type.elem_type,
            'shape': shape
        })
    
    output_info = []
    for out in graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in out.type.tensor_type.shape.dim]
        output_info.append({
            'name': out.name,
            'type': out.type.tensor_type.elem_type,
            'shape': shape
        })
    
    stats['inputs'] = input_info
    stats['outputs'] = output_info
    
    return stats


def compare_models(
    model1_path: Union[str, Path],
    model2_path: Union[str, Path]
) -> Dict[str, Any]:
    """Compare two ONNX models"""
    
    stats1 = get_model_statistics(model1_path)
    stats2 = get_model_statistics(model2_path)
    
    comparison = {
        'model1': stats1,
        'model2': stats2,
        'size_difference_bytes': stats2['model_size_bytes'] - stats1['model_size_bytes'],
        'size_difference_percent': (
            (stats2['model_size_bytes'] - stats1['model_size_bytes']) / stats1['model_size_bytes'] * 100
        ),
        'node_difference': stats2['num_nodes'] - stats1['num_nodes'],
        'input_difference': stats2['num_inputs'] - stats1['num_inputs'],
        'output_difference': stats2['num_outputs'] - stats1['num_outputs'],
    }
    
    return comparison


def validate_optimization(
    original_model_path: Union[str, Path],
    optimized_model_path: Union[str, Path]
) -> Dict[str, Any]:
    """Validate that optimization preserved model behavior"""
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Load models
        original_session = ort.InferenceSession(str(original_model_path))
        optimized_session = ort.InferenceSession(str(optimized_model_path))
        
        # Get input info
        input_info = original_session.get_inputs()[0]
        input_shape = input_info.shape
        input_type = input_info.type
        
        # Generate random test input
        if 'float' in input_type:
            test_input = np.random.randn(*input_shape).astype(np.float32)
        else:
            test_input = np.random.randint(0, 100, size=input_shape).astype(np.int64)
        
        # Run inference
        original_output = original_session.run(None, {input_info.name: test_input})
        optimized_output = optimized_session.run(None, {input_info.name: test_input})
        
        # Compare outputs
        validation_results = []
        for i, (orig_out, opt_out) in enumerate(zip(original_output, optimized_output)):
            mse = np.mean((orig_out - opt_out) ** 2)
            max_diff = np.max(np.abs(orig_out - opt_out))
            mean_diff = np.mean(np.abs(orig_out - opt_out))
            
            # Check if outputs are close enough
            tolerance = 1e-3
            passed = np.allclose(orig_out, opt_out, atol=tolerance, rtol=tolerance)
            
            validation_results.append({
                'output_index': i,
                'passed': passed,
                'mse': float(mse),
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'tolerance': tolerance
            })
        
        # Overall validation
        all_passed = all(result['passed'] for result in validation_results)
        
        return {
            'passed': all_passed,
            'outputs': validation_results,
            'num_outputs': len(validation_results),
            'validation_timestamp': str(Path(optimized_model_path).stat().st_mtime)
        }
        
    except Exception as e:
        logger.error(f"Optimization validation failed: {e}")
        return {
            'passed': False,
            'error': str(e),
            'outputs': []
        }