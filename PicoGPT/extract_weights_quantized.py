#!/usr/bin/env python3
"""
TensorFlow Lite Quantized Weight Extractor

This script extracts weights from a TFLite model and generates C header files
with int8 quantized weights and per-tensor quantization parameters.

The weights are named following the convention from gpt_weights folder:
- layer1_attn_qkv, layer1_attn_out, layer1_ffn_up, layer1_ffn_down
- layer2_attn_qkv, layer2_attn_out, layer2_ffn_up, layer2_ffn_down
- token_embedding, position_embedding
- layer1_ln_gamma, layer1_ln_beta, etc.

Each weight tensor includes:
- int8 quantized weights
- scale (float)
- zero_point (int32)
"""

import numpy as np
import tensorflow as tf
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class QuantParams:
    scale: float
    zero_point: int
    qmin: int = -128
    qmax: int = 127

def load_tflite_model(model_path: str):
    """Load and return TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def choose_symmetric_int8_params(max_abs: float) -> QuantParams:
    """Symmetric int8 quantization, zero_point = 0, scale = max_abs / 127."""
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        return QuantParams(scale=1.0, zero_point=0)
    return QuantParams(scale=float(max_abs) / 127.0, zero_point=0)

def quantize_to_int8(x_f32: np.ndarray, qp: QuantParams) -> np.ndarray:
    """Quantize float32 array to int8."""
    q = np.round(x_f32 / qp.scale + qp.zero_point)
    q = np.clip(q, qp.qmin, qp.qmax).astype(np.int8)
    return q

def analyze_and_extract_weights(interpreter):
    """Extract and categorize all weight tensors from the model."""
    tensor_details = interpreter.get_tensor_details()
    
    # Mapping from tensor index/name to friendly name based on shape analysis
    # From the model analysis:
    # arith.constant1  [32, 128]  -> layer1_ffn_up
    # arith.constant2  [128, 32]  -> layer1_ffn_down
    # arith.constant3  [32, 32]   -> layer1_attn_out
    # arith.constant4  [96, 32]   -> layer1_attn_qkv
    # arith.constant5  [32, 128]  -> layer2_ffn_up
    # arith.constant6  [128, 32]  -> layer2_ffn_down
    # arith.constant7  [32, 32]   -> layer2_attn_out
    # arith.constant8  [96, 32]   -> layer2_attn_qkv
    # arith.constant9  [47, 32]   -> token_embedding
    # model_1/tf.math.add_107/Add/y [31, 32] -> position_embedding
    # arith.constant13 [31, 31]   -> attention_mask
    # model_1/tf.math.multiply_97/Mul/y  [32] -> layer1_ln_gamma
    # model_1/tf.math.multiply_102/Mul/y [32] -> layer1_ln_beta
    # model_1/tf.math.multiply_105/Mul/y [32] -> layer2_ln_gamma
    # model_1/tf.math.multiply_110/Mul/y [32] -> layer2_ln_beta
    # model_1/tf.math.multiply_113/Mul/y [32] -> final_ln_gamma
    
    weight_tensors = {}
    
    # Tensor name to friendly name mapping
    name_mapping = {
        'arith.constant1': 'layer1_ffn_up',
        'arith.constant2': 'layer1_ffn_down', 
        'arith.constant3': 'layer1_attn_out',
        'arith.constant4': 'layer1_attn_qkv',
        'arith.constant5': 'layer2_ffn_up',
        'arith.constant6': 'layer2_ffn_down',
        'arith.constant7': 'layer2_attn_out',
        'arith.constant8': 'layer2_attn_qkv',
        'arith.constant9': 'token_embedding',
        'model_1/tf.math.add_107/Add/y': 'position_embedding',
        'arith.constant13': 'attention_mask',
        'model_1/tf.math.multiply_97/Mul/y': 'layer1_ln_gamma',
        'model_1/tf.math.multiply_102/Mul/y': 'layer1_ln_beta',
        'model_1/tf.math.multiply_105/Mul/y': 'layer2_ln_gamma',
        'model_1/tf.math.multiply_110/Mul/y': 'layer2_ln_beta',
        'model_1/tf.math.multiply_113/Mul/y': 'final_ln_gamma',
    }
    
    for i, tensor in enumerate(tensor_details):
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        
        # Only extract weight tensors (float32 with meaningful shapes)
        if dtype != np.float32:
            continue
            
        # Skip activation tensors (3D or 4D intermediate tensors)
        if len(shape) > 2:
            continue
            
        # Skip scalar or very small tensors that aren't weights
        if len(shape) == 0 or (len(shape) == 1 and shape[0] <= 1):
            continue
            
        try:
            data = interpreter.get_tensor(i)
            if data is None:
                continue
            
            # Map to friendly name if available
            friendly_name = name_mapping.get(name, None)
            
            if friendly_name:
                weight_tensors[friendly_name] = {
                    'index': i,
                    'original_name': name,
                    'shape': list(shape),
                    'data': data.copy()
                }
                print(f"Found {friendly_name}: {name}, shape: {shape}")
            else:
                # Check if this is a shape we want but didn't map
                if len(shape) == 2 or (len(shape) == 1 and shape[0] == 32):
                    print(f"Unmapped tensor: {name}, shape: {shape}")
                    
        except Exception as e:
            print(f"Could not extract tensor {i} ({name}): {e}")
            continue
    
    return weight_tensors

def quantize_weights(weight_tensors: Dict) -> Tuple[Dict[str, QuantParams], Dict[str, np.ndarray]]:
    """Apply per-tensor symmetric int8 quantization to all weights."""
    qparams = {}
    int8_weights = {}
    
    for name, tensor_info in weight_tensors.items():
        data = tensor_info['data']
        max_abs = float(np.max(np.abs(data)))
        
        qp = choose_symmetric_int8_params(max_abs)
        q_data = quantize_to_int8(data, qp)
        
        qparams[name] = qp
        int8_weights[name] = q_data
        
        print(f"Quantized {name}: max_abs={max_abs:.6f}, scale={qp.scale:.10f}")
    
    return qparams, int8_weights

def format_int8_array_1d(data: np.ndarray, values_per_line: int = 16) -> str:
    """Format a 1D int8 array as C code."""
    flat = data.flatten()
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i+values_per_line]
        line = ", ".join(f"{v:4d}" for v in chunk)
        lines.append(f"    {line}")
    return ",\n".join(lines)

def format_int8_array_2d(data: np.ndarray) -> str:
    """Format a 2D int8 array as C code."""
    lines = []
    for i in range(data.shape[0]):
        row = data[i]
        row_str = ", ".join(f"{v:4d}" for v in row)
        lines.append(f"    {{{row_str}}}")
    return ",\n".join(lines)

def generate_quantized_embedding_header(weight_tensors: Dict, qparams: Dict, 
                                         int8_weights: Dict, output_dir: str):
    """Generate header for quantized embedding layers."""
    
    header = """#ifndef GPT_EMBEDDINGS_Q_H
#define GPT_EMBEDDINGS_Q_H

#include <stdint.h>

// Model dimensions
#define VOCAB_SIZE 47
#define MAX_SEQ_LEN 31
#define EMBED_DIM 32

"""
    
    # Token embedding
    if 'token_embedding' in int8_weights:
        name = 'token_embedding'
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        
        header += f"""// Token embedding matrix [{shape[0]}x{shape[1]}] - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
// To dequantize: float_val = (int8_val - zero_point) * scale
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[VOCAB_SIZE][EMBED_DIM] = {{
{format_int8_array_2d(q_data)}
}};

"""

    # Position embedding
    if 'position_embedding' in int8_weights:
        name = 'position_embedding'
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        
        header += f"""// Position embedding matrix [{shape[0]}x{shape[1]}] - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[MAX_SEQ_LEN][EMBED_DIM] = {{
{format_int8_array_2d(q_data)}
}};

"""
    
    header += "#endif // GPT_EMBEDDINGS_Q_H\n"
    
    filepath = os.path.join(output_dir, "gpt_embeddings_q.h")
    with open(filepath, "w") as f:
        f.write(header)
    print(f"Generated {filepath}")

def generate_quantized_layer_header(layer_num: int, weight_tensors: Dict, 
                                     qparams: Dict, int8_weights: Dict, output_dir: str):
    """Generate header for a quantized transformer layer."""
    
    prefix = f"layer{layer_num}"
    
    header = f"""#ifndef GPT_LAYER{layer_num}_Q_H
#define GPT_LAYER{layer_num}_Q_H

#include <stdint.h>

// Layer {layer_num} dimensions
#define LAYER{layer_num}_EMBED_DIM 32
#define LAYER{layer_num}_FFN_DIM 128
#define LAYER{layer_num}_NUM_HEADS 1
#define LAYER{layer_num}_HEAD_DIM 32

"""
    
    # Layer norm gamma (scale)
    name = f'{prefix}_ln_gamma'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        header += f"""// Layer {layer_num} LayerNorm gamma (scale) - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[LAYER{layer_num}_EMBED_DIM] = {{
{format_int8_array_1d(q_data)}
}};

"""

    # Layer norm beta (bias)
    name = f'{prefix}_ln_beta'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        header += f"""// Layer {layer_num} LayerNorm beta (bias) - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[LAYER{layer_num}_EMBED_DIM] = {{
{format_int8_array_1d(q_data)}
}};

"""

    # Attention QKV weights
    name = f'{prefix}_attn_qkv'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        header += f"""// Layer {layer_num} Attention QKV projection [{shape[0]}x{shape[1]}] - Quantized INT8
// First 32 rows: Query, Next 32 rows: Key, Last 32 rows: Value
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[{shape[0]}][{shape[1]}] = {{
{format_int8_array_2d(q_data)}
}};

"""

    # Attention output weights
    name = f'{prefix}_attn_out'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        header += f"""// Layer {layer_num} Attention output projection [{shape[0]}x{shape[1]}] - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[{shape[0]}][{shape[1]}] = {{
{format_int8_array_2d(q_data)}
}};

"""

    # FFN up projection
    name = f'{prefix}_ffn_up'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        header += f"""// Layer {layer_num} FFN up projection [{shape[0]}x{shape[1]}] - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[{shape[0]}][{shape[1]}] = {{
{format_int8_array_2d(q_data)}
}};

"""

    # FFN down projection
    name = f'{prefix}_ffn_down'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        shape = weight_tensors[name]['shape']
        header += f"""// Layer {layer_num} FFN down projection [{shape[0]}x{shape[1]}] - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[{shape[0]}][{shape[1]}] = {{
{format_int8_array_2d(q_data)}
}};

"""
    
    header += f"#endif // GPT_LAYER{layer_num}_Q_H\n"
    
    filepath = os.path.join(output_dir, f"gpt_layer{layer_num}_q.h")
    with open(filepath, "w") as f:
        f.write(header)
    print(f"Generated {filepath}")

def generate_quantized_constants_header(weight_tensors: Dict, qparams: Dict,
                                         int8_weights: Dict, output_dir: str):
    """Generate header for quantized constants and utilities."""
    
    header = """#ifndef GPT_CONSTANTS_Q_H
#define GPT_CONSTANTS_Q_H

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

"""
    
    # Helper function to format float values, handling inf properly
    def format_float(v):
        if np.isinf(v):
            return "-INFINITY" if v < 0 else "INFINITY"
        else:
            return f"{v:.1f}f"
    
    # Attention mask (kept as float since it's used for masking, not computation)
    if 'attention_mask' in weight_tensors:
        data = weight_tensors['attention_mask']['data']
        shape = weight_tensors['attention_mask']['shape']
        header += f"""// Attention mask (lower triangular for causal attention)
// Note: kept as float32 since this is used for masking, not matrix ops
static const float attention_mask[MAX_SEQ_LEN][MAX_SEQ_LEN] = {{
"""
        for i in range(data.shape[0]):
            row_str = ", ".join(format_float(v) for v in data[i])
            header += f"    {{{row_str}}}"
            if i < data.shape[0] - 1:
                header += ","
            header += "\n"
        header += "};\n\n"
    
    # Final layer norm gamma
    name = 'final_ln_gamma'
    if name in int8_weights:
        q_data = int8_weights[name]
        qp = qparams[name]
        header += f"""// Final LayerNorm gamma (scale) - Quantized INT8
// Quantization: scale={qp.scale:.10f}, zero_point={qp.zero_point}
static const float {name}_scale = {qp.scale:.10e}f;
static const int32_t {name}_zero_point = {qp.zero_point};

static const int8_t {name}_q[EMBED_DIM] = {{
{format_int8_array_1d(q_data)}
}};

"""
    
    header += """// Utility functions
static inline float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float softmax_stable(float x, float max_val) {
    return expf(x - max_val);
}

// Dequantization helper: converts int8 to float
static inline float dequantize_int8(int8_t q_val, float scale, int32_t zero_point) {
    return ((float)q_val - (float)zero_point) * scale;
}

// Quantized matrix-vector multiply with dequantization
// output[M] = dequant(weights_q[M][K]) @ input[K]
// where weights are int8 and input/output are float
static inline void matmul_q_dequant(
    float* output,           // [M]
    const int8_t* weights_q, // [M x K] row-major
    const float* input,      // [K]
    int M, int K,
    float weight_scale,
    int32_t weight_zp
) {
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float w = dequantize_int8(weights_q[m * K + k], weight_scale, weight_zp);
            sum += w * input[k];
        }
        output[m] = sum;
    }
}

#endif // GPT_CONSTANTS_Q_H
"""
    
    filepath = os.path.join(output_dir, "gpt_constants_q.h")
    with open(filepath, "w") as f:
        f.write(header)
    print(f"Generated {filepath}")

def generate_master_header(output_dir: str):
    """Generate master header that includes all quantized headers."""
    
    header = """#ifndef GPT_WEIGHTS_Q_H
#define GPT_WEIGHTS_Q_H

/*
 * GPT Model Weights - Quantized INT8
 * Extracted from TensorFlow Lite model and quantized per-tensor
 * 
 * Model Architecture:
 * - 2-layer transformer 
 * - 47 token vocabulary
 * - 31 maximum sequence length
 * - 32 embedding dimensions
 * - 128 FFN hidden dimensions
 * - Single attention head
 * 
 * Quantization:
 * - Per-tensor symmetric int8 quantization
 * - Each tensor has its own scale and zero_point
 * - Use dequantize_int8() or matmul_q_dequant() for inference
 * 
 * Usage:
 * Include this header in your C implementation to access all model weights.
 * Use the *_scale and *_zero_point constants with each *_q array.
 */

#include "gpt_constants_q.h"
#include "gpt_embeddings_q.h"
#include "gpt_layer1_q.h"
#include "gpt_layer2_q.h"

// Convenience macros for accessing quantized weights
#define TOKEN_EMBED_Q(vocab_id, dim) token_embedding_q[vocab_id][dim]
#define POS_EMBED_Q(pos, dim) position_embedding_q[pos][dim]

#define LAYER1_ATTN_QKV_Q(row, col) layer1_attn_qkv_q[row][col]
#define LAYER1_ATTN_OUT_Q(row, col) layer1_attn_out_q[row][col]
#define LAYER1_FFN_UP_Q(row, col) layer1_ffn_up_q[row][col]
#define LAYER1_FFN_DOWN_Q(row, col) layer1_ffn_down_q[row][col]

#define LAYER2_ATTN_QKV_Q(row, col) layer2_attn_qkv_q[row][col]
#define LAYER2_ATTN_OUT_Q(row, col) layer2_attn_out_q[row][col]
#define LAYER2_FFN_UP_Q(row, col) layer2_ffn_up_q[row][col]
#define LAYER2_FFN_DOWN_Q(row, col) layer2_ffn_down_q[row][col]

#endif // GPT_WEIGHTS_Q_H
"""
    
    filepath = os.path.join(output_dir, "gpt_weights_q.h")
    with open(filepath, "w") as f:
        f.write(header)
    print(f"Generated {filepath}")

def save_quant_report(qparams: Dict, weight_tensors: Dict, output_path: str):
    """Save quantization parameters report as JSON."""
    report = {
        "mode": "per_tensor",
        "description": "Per-tensor symmetric int8 quantization",
        "tensors": {}
    }
    
    for name, qp in qparams.items():
        tensor_info = weight_tensors.get(name, {})
        shape = tensor_info.get('shape', [])
        # Convert numpy types to Python native types for JSON serialization
        shape = [int(x) for x in shape]
        report["tensors"][name] = {
            "scale": float(qp.scale),
            "zero_point": int(qp.zero_point),
            "qmin": int(qp.qmin),
            "qmax": int(qp.qmax),
            "shape": shape,
            "original_name": tensor_info.get('original_name', '')
        }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved quantization report to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and quantize TFLite weights to C headers')
    parser.add_argument('--model', default='gpt.tflite', help='Path to TFLite model')
    parser.add_argument('--output_dir', default='./gpt_weights_q', help='Output directory for headers')
    parser.add_argument('--report', default='quant_report_named.json', help='Quantization report output')
    args = parser.parse_args()
    
    model_path = args.model
    output_dir = args.output_dir
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading TFLite model: {model_path}")
    interpreter = load_tflite_model(model_path)
    
    print("\nExtracting and categorizing weights...")
    weight_tensors = analyze_and_extract_weights(interpreter)
    
    print(f"\nFound {len(weight_tensors)} weight tensors")
    
    print("\nApplying per-tensor quantization...")
    qparams, int8_weights = quantize_weights(weight_tensors)
    
    print(f"\nGenerating quantized C header files in {output_dir}...")
    
    # Generate all headers
    generate_quantized_embedding_header(weight_tensors, qparams, int8_weights, output_dir)
    generate_quantized_layer_header(1, weight_tensors, qparams, int8_weights, output_dir)
    generate_quantized_layer_header(2, weight_tensors, qparams, int8_weights, output_dir)
    generate_quantized_constants_header(weight_tensors, qparams, int8_weights, output_dir)
    generate_master_header(output_dir)
    
    # Save quantization report
    report_path = os.path.join(output_dir, args.report)
    save_quant_report(qparams, weight_tensors, report_path)
    
    print("\n" + "="*60)
    print("Weight extraction and quantization complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {output_dir}/gpt_weights_q.h (master header)")
    print(f"  - {output_dir}/gpt_constants_q.h (constants and utilities)")
    print(f"  - {output_dir}/gpt_embeddings_q.h (token and position embeddings)")
    print(f"  - {output_dir}/gpt_layer1_q.h (first transformer layer)")
    print(f"  - {output_dir}/gpt_layer2_q.h (second transformer layer)")
    print(f"  - {output_dir}/{args.report} (quantization parameters)")
    print("\nTo use in your C code:")
    print('  #include "gpt_weights_q.h"')
    print("\nEach weight tensor has:")
    print("  - <name>_q: int8 quantized weights array")
    print("  - <name>_scale: float scale factor")
    print("  - <name>_zero_point: int32 zero point")
    print("\nTo dequantize: float_val = (int8_val - zero_point) * scale")

if __name__ == "__main__":
    main()
