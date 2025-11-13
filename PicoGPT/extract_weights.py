#!/usr/bin/env python3
"""
TensorFlow Lite Weight Extractor

This script extracts weights from a TFLite model and generates C header files
for use in custom C implementations of neural network operations.
"""

import numpy as np
import tensorflow as tf
import os
import re
from typing import Dict, List, Tuple, Any

def load_tflite_model(model_path: str):
    """Load and return TFLite interpreter and model details."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def analyze_model_structure(interpreter):
    """Analyze the TFLite model structure and print detailed information."""
    print("=== TensorFlow Lite Model Analysis ===")
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"\nModel Overview:")
    print(f"  Input tensors: {len(input_details)}")
    print(f"  Output tensors: {len(output_details)}")
    print(f"  Total tensors: {len(tensor_details)}")
    
    # Print input details
    print(f"\nInput Details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Type: {detail['dtype']}")
        print(f"    Index: {detail['index']}")
    
    # Print output details
    print(f"\nOutput Details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Type: {detail['dtype']}")
        print(f"    Index: {detail['index']}")
    
    return tensor_details

def categorize_tensors(tensor_details):
    """Categorize tensors by operation type and identify weights."""
    weights = []
    activations = []
    constants = []
    
    print(f"\n=== Tensor Analysis ===")
    
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        
        print(f"\nTensor {i}: {name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {dtype}")
        print(f"  Quantization: {tensor.get('quantization_parameters', {})}")
        
        # Categorize based on name patterns and shape
        if len(shape) > 1 and np.prod(shape) > 1:
            # Likely a weight tensor if it's multi-dimensional and not too small
            if any(keyword in name.lower() for keyword in ['weight', 'kernel', 'bias', 'dense', 'conv', 'attention']):
                weights.append((i, tensor))
            elif 'const' in name.lower() or tensor.get('quantization_parameters', {}).get('scales') is not None:
                constants.append((i, tensor))
            else:
                activations.append((i, tensor))
        else:
            activations.append((i, tensor))
    
    print(f"\nCategorization Summary:")
    print(f"  Weights/Parameters: {len(weights)}")
    print(f"  Constants: {len(constants)}")
    print(f"  Activations/Intermediate: {len(activations)}")
    
    return weights, constants, activations

def extract_tensor_data(interpreter, tensor_index):
    """Extract the actual data from a tensor."""
    try:
        tensor_data = interpreter.get_tensor(tensor_index)
        return tensor_data
    except Exception as e:
        print(f"Could not extract tensor {tensor_index}: {e}")
        return None

def identify_operation_types(tensor_details, weights, constants):
    """Try to identify which operations each weight belongs to."""
    operations = {
        'attention': [],
        'ffn': [],
        'embedding': [],
        'layer_norm': [],
        'output': [],
        'unknown': []
    }
    
    for idx, tensor in weights + constants:
        name = tensor['name'].lower()
        shape = tensor['shape']
        
        # Identify operation type based on name patterns and shapes
        if any(keyword in name for keyword in ['attention', 'attn', 'query', 'key', 'value', 'qkv']):
            operations['attention'].append((idx, tensor))
        elif any(keyword in name for keyword in ['ffn', 'feed_forward', 'mlp', 'fc']):
            operations['ffn'].append((idx, tensor))
        elif any(keyword in name for keyword in ['embedding', 'embed', 'token']):
            operations['embedding'].append((idx, tensor))
        elif any(keyword in name for keyword in ['layer_norm', 'layernorm', 'norm']):
            operations['layer_norm'].append((idx, tensor))
        elif any(keyword in name for keyword in ['output', 'logits', 'projection']):
            operations['output'].append((idx, tensor))
        else:
            operations['unknown'].append((idx, tensor))
    
    return operations

def main():
    model_path = "gpt.tflite"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    print(f"Loading TFLite model: {model_path}")
    interpreter = load_tflite_model(model_path)
    
    # Analyze model structure
    tensor_details = analyze_model_structure(interpreter)
    
    # Categorize tensors
    weights, constants, activations = categorize_tensors(tensor_details)
    
    # Identify operation types
    operations = identify_operation_types(tensor_details, weights, constants)
    
    print(f"\n=== Operation Analysis ===")
    for op_type, tensors in operations.items():
        if tensors:
            print(f"\n{op_type.upper()} Operations ({len(tensors)} tensors):")
            for idx, tensor in tensors:
                print(f"  Tensor {idx}: {tensor['name']}")
                print(f"    Shape: {tensor['shape']}")
                
                # Try to extract actual data
                data = extract_tensor_data(interpreter, idx)
                if data is not None:
                    print(f"    Data type: {data.dtype}")
                    print(f"    Data shape: {data.shape}")
                    print(f"    Value range: [{np.min(data):.6f}, {np.max(data):.6f}]")

if __name__ == "__main__":
    main()