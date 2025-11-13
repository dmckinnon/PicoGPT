#!/usr/bin/env python3
"""
TensorFlow Lite Weight Extractor - Enhanced Version

This script extracts weights from a TFLite model and generates C header files
for use in custom C implementations of neural network operations.

Based on the model analysis, this appears to be a 2-layer GPT model with:
- Embedding layer (47 vocab x 32 dims)
- Position embeddings (31 positions x 32 dims)
- Layer norm parameters
- 2 transformer blocks, each with:
  - Multi-head attention (QKV projection: 96x32, output projection: 32x32)
  - Feed-forward network (32->128->32)
"""

import numpy as np
import tensorflow as tf
import os
import re
from typing import Dict, List, Tuple, Any

def load_tflite_model(model_path: str):
    """Load and return TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def analyze_and_extract_weights(interpreter):
    """Extract and categorize all weight tensors from the model."""
    tensor_details = interpreter.get_tensor_details()
    
    # Based on the tensor analysis, identify the weight tensors
    weight_tensors = {
        # Embedding layers
        'token_embedding': None,      # [47, 32] - vocab embedding
        'position_embedding': None,   # [31, 32] - position embedding
        
        # Layer 1 weights
        'layer1_ln_gamma': None,      # [32] - layer norm scale
        'layer1_ln_beta': None,       # [32] - layer norm bias  
        'layer1_attn_qkv': None,      # [96, 32] - attention QKV projection
        'layer1_attn_out': None,      # [32, 32] - attention output projection
        'layer1_ffn_up': None,        # [32, 128] - FFN up projection
        'layer1_ffn_down': None,      # [128, 32] - FFN down projection
        
        # Layer 2 weights
        'layer2_ln_gamma': None,      # [32] - layer norm scale
        'layer2_ln_beta': None,       # [32] - layer norm bias
        'layer2_attn_qkv': None,      # [96, 32] - attention QKV projection  
        'layer2_attn_out': None,      # [32, 32] - attention output projection
        'layer2_ffn_up': None,        # [32, 128] - FFN up projection
        'layer2_ffn_down': None,      # [128, 32] - FFN down projection
        
        # Final layer norm
        'final_ln_gamma': None,       # [32] - final layer norm scale
        'final_ln_beta': None,        # [32] - final layer norm bias
        
        # Attention mask
        'attention_mask': None,       # [31, 31] - causal attention mask
    }
    
    # Extract tensors based on their shapes and positions
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        
        try:
            data = interpreter.get_tensor(i)
            if data is None:
                continue
                
            # Match tensors based on shapes
            if np.array_equal(shape, [47, 32]):
                weight_tensors['token_embedding'] = (i, tensor, data)
                print(f"Found token embedding: {name}, shape: {shape}")
                
            elif np.array_equal(shape, [31, 32]):
                weight_tensors['position_embedding'] = (i, tensor, data)
                print(f"Found position embedding: {name}, shape: {shape}")
                
            elif np.array_equal(shape, [31, 31]):
                weight_tensors['attention_mask'] = (i, tensor, data)
                print(f"Found attention mask: {name}, shape: {shape}")
                
            elif np.array_equal(shape, [32]) and 'multiply' in name:
                # Layer norm gamma parameters (scales)
                if 'multiply_97' in name:
                    weight_tensors['layer1_ln_gamma'] = (i, tensor, data)
                    print(f"Found layer 1 LN gamma: {name}, shape: {shape}")
                elif 'multiply_102' in name:
                    weight_tensors['layer1_ln_beta'] = (i, tensor, data)  # This might be beta
                    print(f"Found layer 1 LN beta: {name}, shape: {shape}")
                elif 'multiply_105' in name:
                    weight_tensors['layer2_ln_gamma'] = (i, tensor, data)
                    print(f"Found layer 2 LN gamma: {name}, shape: {shape}")
                elif 'multiply_110' in name:
                    weight_tensors['layer2_ln_beta'] = (i, tensor, data)
                    print(f"Found layer 2 LN beta: {name}, shape: {shape}")
                elif 'multiply_113' in name:
                    weight_tensors['final_ln_gamma'] = (i, tensor, data)
                    print(f"Found final LN gamma: {name}, shape: {shape}")
                    
            elif np.array_equal(shape, [96, 32]):
                # Attention QKV projections
                if i == 5:  # First QKV matrix
                    weight_tensors['layer1_attn_qkv'] = (i, tensor, data)
                    print(f"Found layer 1 attention QKV: {name}, shape: {shape}")
                elif i == 9:  # Second QKV matrix
                    weight_tensors['layer2_attn_qkv'] = (i, tensor, data)
                    print(f"Found layer 2 attention QKV: {name}, shape: {shape}")
                    
            elif np.array_equal(shape, [32, 32]):
                # Attention output projections
                if i == 4:  # First attention output
                    weight_tensors['layer1_attn_out'] = (i, tensor, data)
                    print(f"Found layer 1 attention output: {name}, shape: {shape}")
                elif i == 8:  # Second attention output
                    weight_tensors['layer2_attn_out'] = (i, tensor, data)
                    print(f"Found layer 2 attention output: {name}, shape: {shape}")
                    
            elif np.array_equal(shape, [32, 128]):
                # FFN up projections
                if i == 2:  # First FFN up
                    weight_tensors['layer1_ffn_up'] = (i, tensor, data)
                    print(f"Found layer 1 FFN up: {name}, shape: {shape}")
                elif i == 6:  # Second FFN up  
                    weight_tensors['layer2_ffn_up'] = (i, tensor, data)
                    print(f"Found layer 2 FFN up: {name}, shape: {shape}")
                    
            elif np.array_equal(shape, [128, 32]):
                # FFN down projections
                if i == 3:  # First FFN down
                    weight_tensors['layer1_ffn_down'] = (i, tensor, data)
                    print(f"Found layer 1 FFN down: {name}, shape: {shape}")
                elif i == 7:  # Second FFN down
                    weight_tensors['layer2_ffn_down'] = (i, tensor, data)
                    print(f"Found layer 2 FFN down: {name}, shape: {shape}")
                    
        except Exception as e:
            print(f"Could not extract tensor {i} ({name}): {e}")
            continue
    
    return weight_tensors

def generate_c_header(weight_tensors, output_dir="./"):
    """Generate C header files for the extracted weights."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate individual header files for different components
    
    # 1. Embeddings
    generate_embedding_header(weight_tensors, output_dir)
    
    # 2. Layer 1
    generate_layer_header(weight_tensors, 1, output_dir)
    
    # 3. Layer 2
    generate_layer_header(weight_tensors, 2, output_dir)
    
    # 4. Constants and utilities
    generate_constants_header(weight_tensors, output_dir)
    
    # 5. Master header
    generate_master_header(output_dir)

def generate_embedding_header(weight_tensors, output_dir):
    """Generate header for embedding layers."""
    header_content = """#ifndef GPT_EMBEDDINGS_H
#define GPT_EMBEDDINGS_H

#include <stdint.h>

// Model dimensions
#define VOCAB_SIZE 47
#define MAX_SEQ_LEN 31
#define EMBED_DIM 32

"""
    
    # Token embedding
    if weight_tensors['token_embedding']:
        _, tensor, data = weight_tensors['token_embedding']
        header_content += f"// Token embedding matrix [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += "static const float token_embedding[VOCAB_SIZE][EMBED_DIM] = {\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += f"  // Token {i}\n"
        header_content += "};\n\n"
    
    # Position embedding
    if weight_tensors['position_embedding']:
        _, tensor, data = weight_tensors['position_embedding']
        header_content += f"// Position embedding matrix [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += "static const float position_embedding[MAX_SEQ_LEN][EMBED_DIM] = {\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += f"  // Position {i}\n"
        header_content += "};\n\n"
    
    header_content += "#endif // GPT_EMBEDDINGS_H\n"
    
    with open(os.path.join(output_dir, "gpt_embeddings.h"), "w") as f:
        f.write(header_content)
    
    print("Generated gpt_embeddings.h")

def generate_layer_header(weight_tensors, layer_num, output_dir):
    """Generate header for a transformer layer."""
    header_content = f"""#ifndef GPT_LAYER{layer_num}_H
#define GPT_LAYER{layer_num}_H

#include <stdint.h>

// Layer {layer_num} dimensions
#define LAYER{layer_num}_EMBED_DIM 32
#define LAYER{layer_num}_FFN_DIM 128
#define LAYER{layer_num}_NUM_HEADS 1
#define LAYER{layer_num}_HEAD_DIM 32

"""
    
    layer_prefix = f"layer{layer_num}"
    
    # Layer norm parameters
    if weight_tensors[f'{layer_prefix}_ln_gamma']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_ln_gamma']
        header_content += f"// Layer {layer_num} LayerNorm gamma (scale) parameters\n"
        header_content += f"static const float layer{layer_num}_ln_gamma[LAYER{layer_num}_EMBED_DIM] = {{\n    "
        for i, val in enumerate(data):
            header_content += f"{val:.8f}f"
            if i < len(data) - 1:
                header_content += ", "
            if (i + 1) % 8 == 0 and i < len(data) - 1:
                header_content += "\n    "
        header_content += "\n};\n\n"
    
    if weight_tensors[f'{layer_prefix}_ln_beta']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_ln_beta']
        header_content += f"// Layer {layer_num} LayerNorm beta (bias) parameters\n"
        header_content += f"static const float layer{layer_num}_ln_beta[LAYER{layer_num}_EMBED_DIM] = {{\n    "
        for i, val in enumerate(data):
            header_content += f"{val:.8f}f"
            if i < len(data) - 1:
                header_content += ", "
            if (i + 1) % 8 == 0 and i < len(data) - 1:
                header_content += "\n    "
        header_content += "\n};\n\n"
    
    # Attention QKV weights
    if weight_tensors[f'{layer_prefix}_attn_qkv']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_attn_qkv']
        header_content += f"// Layer {layer_num} Attention QKV projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"// First 32 rows: Query, Next 32 rows: Key, Last 32 rows: Value\n"
        header_content += f"static const float layer{layer_num}_attn_qkv[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            
            # Add comments for QKV sections
            if i == 0:
                header_content += "  // Query weights start"
            elif i == 31:
                header_content += "  // Key weights start"  
            elif i == 63:
                header_content += "  // Value weights start"
            header_content += "\n"
        header_content += "};\n\n"
    
    # Attention output weights
    if weight_tensors[f'{layer_prefix}_attn_out']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_attn_out']
        header_content += f"// Layer {layer_num} Attention output projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"static const float layer{layer_num}_attn_out[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += "\n"
        header_content += "};\n\n"
    
    # FFN up projection
    if weight_tensors[f'{layer_prefix}_ffn_up']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_ffn_up']
        header_content += f"// Layer {layer_num} FFN up projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"static const float layer{layer_num}_ffn_up[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += "\n"
        header_content += "};\n\n"
    
    # FFN down projection  
    if weight_tensors[f'{layer_prefix}_ffn_down']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_ffn_down']
        header_content += f"// Layer {layer_num} FFN down projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"static const float layer{layer_num}_ffn_down[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.8f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += "\n"
        header_content += "};\n\n"
    
    header_content += f"#endif // GPT_LAYER{layer_num}_H\n"
    
    with open(os.path.join(output_dir, f"gpt_layer{layer_num}.h"), "w") as f:
        f.write(header_content)
    
    print(f"Generated gpt_layer{layer_num}.h")

def generate_constants_header(weight_tensors, output_dir):
    """Generate header for constants and utilities."""
    header_content = """#ifndef GPT_CONSTANTS_H
#define GPT_CONSTANTS_H

#include <stdint.h>
#include <math.h>

// Model configuration
#define VOCAB_SIZE 47
#define MAX_SEQ_LEN 31
#define EMBED_DIM 32
#define FFN_DIM 128
#define NUM_LAYERS 2
#define NUM_HEADS 1
#define HEAD_DIM 32

// Mathematical constants
#define LAYER_NORM_EPS 1e-5f
#define SQRT_HEAD_DIM 5.656854249f  // sqrt(32)
#define INV_SQRT_HEAD_DIM 0.176776695f  // 1/sqrt(32)

// Attention mask (lower triangular for causal attention)
"""
    
    if weight_tensors['attention_mask']:
        _, tensor, data = weight_tensors['attention_mask']
        header_content += f"static const float attention_mask[MAX_SEQ_LEN][MAX_SEQ_LEN] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isinf(val) and val < 0:
                    header_content += "-INFINITY"
                else:
                    header_content += f"{val:.1f}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += "\n"
        header_content += "};\n\n"
    
    # Final layer norm
    if weight_tensors['final_ln_gamma']:
        _, tensor, data = weight_tensors['final_ln_gamma']
        header_content += "// Final LayerNorm gamma (scale) parameters\n"
        header_content += "static const float final_ln_gamma[EMBED_DIM] = {\n    "
        for i, val in enumerate(data):
            header_content += f"{val:.8f}f"
            if i < len(data) - 1:
                header_content += ", "
            if (i + 1) % 8 == 0 and i < len(data) - 1:
                header_content += "\n    "
        header_content += "\n};\n\n"
    
    header_content += """// Utility functions
static inline float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float softmax_stable(float x, float max_val) {
    return expf(x - max_val);
}

#endif // GPT_CONSTANTS_H
"""
    
    with open(os.path.join(output_dir, "gpt_constants.h"), "w") as f:
        f.write(header_content)
    
    print("Generated gpt_constants.h")

def generate_master_header(output_dir):
    """Generate master header that includes all others."""
    header_content = """#ifndef GPT_WEIGHTS_H
#define GPT_WEIGHTS_H

/*
 * GPT Model Weights
 * Extracted from TensorFlow Lite model
 * 
 * Model Architecture:
 * - 2-layer transformer 
 * - 47 token vocabulary
 * - 31 maximum sequence length
 * - 32 embedding dimensions
 * - 128 FFN hidden dimensions
 * - Single attention head
 * 
 * Usage:
 * Include this header in your C implementation to access all model weights.
 */

#include "gpt_constants.h"
#include "gpt_embeddings.h"
#include "gpt_layer1.h"
#include "gpt_layer2.h"

// Weight access macros for convenience
#define TOKEN_EMBED(vocab_id, dim) token_embedding[vocab_id][dim]
#define POS_EMBED(pos, dim) position_embedding[pos][dim]

#define LAYER1_ATTN_QKV(row, col) layer1_attn_qkv[row][col] 
#define LAYER1_ATTN_OUT(row, col) layer1_attn_out[row][col]
#define LAYER1_FFN_UP(row, col) layer1_ffn_up[row][col]
#define LAYER1_FFN_DOWN(row, col) layer1_ffn_down[row][col]

#define LAYER2_ATTN_QKV(row, col) layer2_attn_qkv[row][col]
#define LAYER2_ATTN_OUT(row, col) layer2_attn_out[row][col]  
#define LAYER2_FFN_UP(row, col) layer2_ffn_up[row][col]
#define LAYER2_FFN_DOWN(row, col) layer2_ffn_down[row][col]

#endif // GPT_WEIGHTS_H
"""
    
    with open(os.path.join(output_dir, "gpt_weights.h"), "w") as f:
        f.write(header_content)
    
    print("Generated gpt_weights.h (master header)")

def main():
    model_path = "gpt.tflite"
    output_dir = "./gpt_weights"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    print(f"Loading TFLite model: {model_path}")
    interpreter = load_tflite_model(model_path)
    
    print("\nExtracting and categorizing weights...")
    weight_tensors = analyze_and_extract_weights(interpreter)
    
    print(f"\nGenerating C header files in {output_dir}...")
    generate_c_header(weight_tensors, output_dir)
    
    print("\nWeight extraction complete!")
    print("\nGenerated files:")
    print("- gpt_weights.h (master header)")
    print("- gpt_constants.h (model constants and utilities)")
    print("- gpt_embeddings.h (token and position embeddings)")
    print("- gpt_layer1.h (first transformer layer weights)")
    print("- gpt_layer2.h (second transformer layer weights)")
    
    print("\nTo use in your C code:")
    print('#include "gpt_weights.h"')

if __name__ == "__main__":
    main()