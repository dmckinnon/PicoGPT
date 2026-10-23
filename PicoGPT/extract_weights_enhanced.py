#!/usr/bin/env python3
"""
TensorFlow Lite Weight Extractor - Enhanced Version

This script extracts weights from a TFLite model and generates C header files
for use in custom C implementations of neural network operations.

The supported graph is a 2-layer GPT model with:
- Embedding layer (47 vocab x 32 dims)
- Position embeddings (31 positions x 32 dims)
- Layer norm parameters
- 2 transformer blocks, each with:
  - Multi-head attention (QKV projection: 96x32, output projection: 32x32)
  - Feed-forward network (32->128->32)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

def load_tflite_model(model_path: str):
    """Read constants and graph edges without executing the model."""
    return schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(Path(model_path).read_bytes(), 0)
    )

def analyze_and_extract_weights(model):
    """Follow the bundled model's dataflow; reject unsupported architectures."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"Unsupported TFLite graph: {message}")

    require(len(model.subgraphs) == 1, "expected one subgraph")
    graph = model.subgraphs[0]
    require(len(graph.inputs) == len(graph.outputs) == 1, "expected one input/output")
    require(graph.tensors[graph.inputs[0]].shape.tolist() == [1, 31], "input shape")
    require(graph.tensors[graph.inputs[0]].type == schema.TensorType.INT32, "input dtype")
    producers = {int(out): op for op in graph.operators for out in op.outputs}
    weights = {}

    def producer(index, kind):
        require(index in producers, f"tensor {index} has no producer ({kind})")
        op = producers[index]
        code = model.operatorCodes[op.opcodeIndex].builtinCode
        require(code == getattr(schema.BuiltinOperator, kind), f"tensor {index}: expected {kind}")
        require(getattr(op.builtinOptions, "fusedActivationFunction", 0) == 0,
                f"tensor {index}: fused activation")
        return op

    def constant(index, shape, dtype=schema.TensorType.FLOAT32):
        require(index >= 0, "missing constant")
        tensor = graph.tensors[index]
        require(tensor.type == dtype and tuple(tensor.shape) == shape,
                f"tensor {index}: expected shape {shape} and dtype {dtype}")
        data = model.buffers[tensor.buffer].data
        require(data is not None and len(data) > 0, f"tensor {index} is not constant")
        return np.frombuffer(data, dtype="<f4" if dtype == schema.TensorType.FLOAT32 else "<i4").reshape(shape)

    def save(name, index, shape, layout="unchanged"):
        data = constant(index, shape)
        if layout == "linear":
            data = data.T
        elif layout == "qkv":
            # C consumes three consecutive [input, output] matrices.
            data = data.reshape(3, 32, 32).transpose(0, 2, 1).reshape(96, 32)
        weights[name] = (int(index), {"name": graph.tensors[index].name.decode()}, data)

    def linear(output, name, shape, layout="linear"):
        op = producer(output, "FULLY_CONNECTED")
        require(len(op.inputs) == 3 and op.inputs[2] == -1, f"{name}: unexpected bias")
        require(op.builtinOptions.weightsFormat == 0, f"{name}: packed weights")
        save(name, op.inputs[1], shape, layout)
        return op.inputs[0]

    def normalization(output, prefix):
        add = producer(output, "ADD")
        scaled = producer(add.inputs[0], "MUL")
        offset = producer(add.inputs[1], "SUB")
        mean_scaled = producer(offset.inputs[1], "MUL")
        x, scale = scaled.inputs
        require(mean_scaled.inputs[1] == scale, f"{prefix}: normalization scale")
        mean = producer(mean_scaled.inputs[0], "MEAN")
        require(mean.inputs[0] == x, f"{prefix}: normalization mean")
        scale_op = producer(scale, "MUL")
        inv_std = producer(scale_op.inputs[0], "RSQRT")
        variance_eps = producer(inv_std.inputs[0], "ADD")
        variance = producer(variance_eps.inputs[0], "MEAN")
        difference = producer(variance.inputs[0], "SQUARED_DIFFERENCE")
        require(difference.inputs.tolist() == [x, mean_scaled.inputs[0]],
                f"{prefix}: normalization variance")
        for reduction in (mean, variance):
            require(reduction.builtinOptions.keepDims, f"{prefix}: reduction dimensions")
            require(constant(reduction.inputs[1], (1,), schema.TensorType.INT32).tolist() == [2],
                    f"{prefix}: reduction axis")
        require(constant(variance_eps.inputs[1], ()).item() == np.float32(1e-5),
                f"{prefix}: epsilon")
        save(f"{prefix}_gamma", scale_op.inputs[1], (32,))
        save(f"{prefix}_beta", offset.inputs[0], (32,))
        return x

    def reshape_input(output, shape):
        op = producer(output, "RESHAPE")
        require(tuple(graph.tensors[output].shape) == shape, "attention reshape output")
        require(constant(op.inputs[1], (len(shape),), schema.TensorType.INT32).tolist() == list(shape),
                "attention reshape dimensions")
        return op.inputs[0]

    def attention_input(output):
        context = producer(reshape_input(output, (1, 31, 32)), "BATCH_MATMUL")
        softmax = producer(context.inputs[0], "SOFTMAX")
        require(softmax.builtinOptions.beta == 1.0, "softmax temperature")
        masked = producer(softmax.inputs[0], "ADD")
        scores = producer(masked.inputs[0], "BATCH_MATMUL")
        for op in (context, scores):
            require(not op.builtinOptions.adjX and not op.builtinOptions.adjY,
                    "attention matrix transpose")
        save("attention_mask", masked.inputs[1], (31, 31))
        expected_mask = np.where(np.triu(np.ones((31, 31), dtype=bool), 1), -np.inf, 0)
        require(np.array_equal(weights["attention_mask"][2], expected_mask), "causal mask")
        query = producer(scores.inputs[0], "MUL")
        key = producer(scores.inputs[1], "MUL")
        for op in (query, key):
            require(constant(op.inputs[1], (1,)).item() == np.float32(32 ** -0.25),
                    "attention scaling")
        transpose = producer(key.inputs[0], "TRANSPOSE")
        require(constant(transpose.inputs[1], (4,), schema.TensorType.INT32).tolist() == [0, 2, 3, 1],
                "key transpose axes")
        slices = [
            producer(reshape_input(query.inputs[0], (1, 1, 31, 32)), "STRIDED_SLICE"),
            producer(reshape_input(transpose.inputs[0], (1, 31, 1, 32)), "STRIDED_SLICE"),
            producer(reshape_input(context.inputs[1], (1, 1, 31, 32)), "STRIDED_SLICE"),
        ]
        qkv = slices[0].inputs[0]
        for part, op in enumerate(slices):
            require(op.inputs[0] == qkv, "Q/K/V must share a projection")
            require(constant(op.inputs[1], (3,), schema.TensorType.INT32).tolist() == [0, 0, part * 32],
                    "Q/K/V slice start")
            require(constant(op.inputs[3], (3,), schema.TensorType.INT32).tolist() == [1, 1, 1],
                    "Q/K/V slice stride")
            options = op.builtinOptions
            require(options.beginMask == 3 and options.endMask == (7 if part == 2 else 3)
                    and not options.ellipsisMask and not options.newAxisMask
                    and not options.shrinkAxisMask and not options.offset, "Q/K/V slice masks")
            if part < 2:
                require(constant(op.inputs[2], (3,), schema.TensorType.INT32).tolist() == [0, 0, (part + 1) * 32],
                        "Q/K/V slice end")
        return qkv

    head = producer(graph.outputs[0], "FULLY_CONNECTED")
    last_hidden = linear(graph.outputs[0], "token_embedding", (47, 32), "unchanged")
    gather = producer(last_hidden, "GATHER")
    require(gather.builtinOptions.axis == 1 and gather.builtinOptions.batchDims == 0,
            "last-token gather axis")
    require(constant(gather.inputs[1], (1,), schema.TensorType.INT32).item() in (-1, 30),
            "last-token gather index")
    current = normalization(gather.inputs[0], "final_ln")

    for layer in (2, 1):
        prefix = f"layer{layer}"
        residual = producer(current, "ADD")
        hidden = linear(residual.inputs[1], f"{prefix}_ffn_down", (32, 128))
        gelu = producer(hidden, "GELU")
        require(not gelu.builtinOptions.approximate, "expected exact GELU")
        normalized = linear(gelu.inputs[0], f"{prefix}_ffn_up", (128, 32))
        require(normalization(normalized, f"{prefix}_ffn_ln") == residual.inputs[0],
                f"{prefix}: FFN residual")
        attention_residual = producer(residual.inputs[0], "ADD")
        context = linear(attention_residual.inputs[1], f"{prefix}_attn_out", (32, 32))
        qkv = attention_input(context)
        normalized = linear(qkv, f"{prefix}_attn_qkv", (96, 32), "qkv")
        current = normalization(normalized, f"{prefix}_ln")
        require(current == attention_residual.inputs[0], f"{prefix}: attention residual")

    embeddings = producer(current, "ADD")
    save("position_embedding", embeddings.inputs[1], (31, 32))
    lookup = producer(embeddings.inputs[0], "GATHER")
    require(lookup.inputs.tolist() == [head.inputs[1], graph.inputs[0]],
            "token embeddings must be tied to the output projection")
    require(lookup.builtinOptions.axis == 0 and lookup.builtinOptions.batchDims == 0,
            "token embedding axis")
    return weights

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
    header_content = """#ifndef GPT_WEIGHTS_EMBEDDINGS_H
#define GPT_WEIGHTS_EMBEDDINGS_H

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
                header_content += f"{data[i, j]:.9e}f"
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
                header_content += f"{data[i, j]:.9e}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            header_content += f"  // Position {i}\n"
        header_content += "};\n\n"
    
    header_content += "#endif // GPT_WEIGHTS_EMBEDDINGS_H\n"
    
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
    
    for norm, label in (("ln", "Attention"), ("ffn_ln", "FFN")):
        for parameter in ("gamma", "beta"):
            name = f"{layer_prefix}_{norm}_{parameter}"
            _, tensor, data = weight_tensors[name]
            header_content += f"// {label} LayerNorm {parameter}: {tensor['name']}\n"
            header_content += f"static const float {name}[LAYER{layer_num}_EMBED_DIM] = {{\n    "
            for i, val in enumerate(data):
                header_content += f"{val:.9e}f"
                if i < len(data) - 1:
                    header_content += ", "
                if (i + 1) % 8 == 0 and i < len(data) - 1:
                    header_content = header_content.rstrip() + "\n    "
            header_content += "\n};\n\n"
    
    # Attention QKV weights
    if weight_tensors[f'{layer_prefix}_attn_qkv']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_attn_qkv']
        header_content += f"// Layer {layer_num} Attention QKV projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"// Source: {tensor['name']}; each Q/K/V block transposed to [input, output]\n"
        header_content += "// First 32 rows: Query, Next 32 rows: Key, Last 32 rows: Value\n"
        header_content += f"static const float layer{layer_num}_attn_qkv[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.9e}f"
                if j < data.shape[1] - 1:
                    header_content += ", "
            header_content += "}"
            if i < data.shape[0] - 1:
                header_content += ","
            
            # Add comments for QKV sections
            if i == 0:
                header_content += "  // Query weights start"
            elif i == 32:
                header_content += "  // Key weights start"  
            elif i == 64:
                header_content += "  // Value weights start"
            header_content += "\n"
        header_content += "};\n\n"
    
    # Attention output weights
    if weight_tensors[f'{layer_prefix}_attn_out']:
        _, tensor, data = weight_tensors[f'{layer_prefix}_attn_out']
        header_content += f"// Layer {layer_num} Attention output projection weights [{data.shape[0]}x{data.shape[1]}]\n"
        header_content += f"// Source: {tensor['name']}; transposed to [input, output]\n"
        header_content += f"static const float layer{layer_num}_attn_out[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.9e}f"
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
        header_content += f"// Source: {tensor['name']}; transposed to [input, output]\n"
        header_content += f"static const float layer{layer_num}_ffn_up[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.9e}f"
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
        header_content += f"// Source: {tensor['name']}; transposed to [input, output]\n"
        header_content += f"static const float layer{layer_num}_ffn_down[{data.shape[0]}][{data.shape[1]}] = {{\n"
        for i in range(data.shape[0]):
            header_content += "    {"
            for j in range(data.shape[1]):
                header_content += f"{data[i, j]:.9e}f"
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
    for parameter in ("gamma", "beta"):
        _, tensor, data = weight_tensors[f'final_ln_{parameter}']
        header_content += f"// Final LayerNorm {parameter}: {tensor['name']}\n"
        header_content += f"static const float final_ln_{parameter}[EMBED_DIM] = {{\n    "
        for i, val in enumerate(data):
            header_content += f"{val:.9e}f"
            if i < len(data) - 1:
                header_content += ", "
            if (i + 1) % 8 == 0 and i < len(data) - 1:
                header_content = header_content.rstrip() + "\n    "
        header_content += "\n};\n\n"
    
    header_content += """// Utility functions
static inline float gelu_activation(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt.tflite")
    parser.add_argument("--output-dir", default="gpt_weights")
    args = parser.parse_args()
    model_path = args.model
    output_dir = args.output_dir
    
    print(f"Loading TFLite model: {model_path}")
    model = load_tflite_model(model_path)
    
    print("\nExtracting and categorizing weights...")
    weight_tensors = analyze_and_extract_weights(model)
    
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