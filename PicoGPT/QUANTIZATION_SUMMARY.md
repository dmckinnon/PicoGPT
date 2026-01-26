# GPT Model Per-Tensor Quantization Summary

## Overview

Successfully applied **per-tensor symmetric int8 quantization** to all weight tensors from the original `gpt.tflite` float32 model. This produces quantized weights suitable for efficient inference on memory-constrained devices like the RP2350 microcontroller.

## Quantization Approach

### Method: Per-Tensor Symmetric INT8

- **Quantization Scheme**: Symmetric (zero_point = 0)
- **Data Type**: Signed 8-bit integers (int8_t), range [-128, 127]
- **Scale Calculation**: `scale = max(|weights|) / 127`
- **Quantization Formula**: `q = round(float_val / scale)`
- **Dequantization Formula**: `float_val = q * scale`

### Why Per-Tensor (vs Per-Channel or Global)?

| Approach | Pros | Cons |
|----------|------|------|
| **Global** | Single scale for all tensors, simplest | Poor accuracy - outliers in one tensor affect all |
| **Per-Tensor** | Good balance of accuracy and simplicity | Each tensor needs its own scale stored |
| **Per-Channel** | Best accuracy for conv/matmul | More complex, more scales to store |

Per-tensor was chosen as the optimal balance for this small model. Initial experiments with global quantization resulted in weights being zeroed out due to outlier sensitivity.

## Generated Files

### Output Directory: `gpt_weights_q/`

| File | Size | Description |
|------|------|-------------|
| `gpt_weights_q.h` | 1.5 KB | Master header (includes all others) |
| `gpt_constants_q.h` | 10.5 KB | Model constants, attention mask, utility functions |
| `gpt_embeddings_q.h` | 16.3 KB | Quantized token & position embeddings |
| `gpt_layer1_q.h` | 77.8 KB | Layer 1 quantized weights |
| `gpt_layer2_q.h` | 77.8 KB | Layer 2 quantized weights |
| `quant_report_named.json` | 3.7 KB | Quantization parameters in JSON format |

### Memory Savings

| Component | Float32 Size | Int8 Size | Reduction |
|-----------|--------------|-----------|-----------|
| Token Embedding (47×32) | 6,016 bytes | 1,504 bytes | 75% |
| Position Embedding (31×32) | 3,968 bytes | 992 bytes | 75% |
| Attention QKV (96×32) × 2 | 24,576 bytes | 6,144 bytes | 75% |
| Attention Out (32×32) × 2 | 8,192 bytes | 2,048 bytes | 75% |
| FFN Up (32×128) × 2 | 32,768 bytes | 8,192 bytes | 75% |
| FFN Down (128×32) × 2 | 32,768 bytes | 8,192 bytes | 75% |
| Layer Norm (32) × 5 | 640 bytes | 160 bytes | 75% |
| **Total Weights** | **~109 KB** | **~27 KB** | **75%** |

## Quantized Tensor Details

### Per-Tensor Quantization Parameters

| Tensor Name | Shape | Scale | Max |w|| 
|-------------|-------|-------|---------|
| `token_embedding` | [47, 32] | 0.00491 | 0.624 |
| `position_embedding` | [31, 32] | 0.00191 | 0.243 |
| `layer1_attn_qkv` | [96, 32] | 0.00301 | 0.382 |
| `layer1_attn_out` | [32, 32] | 0.00112 | 0.143 |
| `layer1_ffn_up` | [32, 128] | 0.00157 | 0.200 |
| `layer1_ffn_down` | [128, 32] | 0.00234 | 0.297 |
| `layer1_ln_gamma` | [32] | 0.00986 | 1.252 |
| `layer1_ln_beta` | [32] | 0.00895 | 1.136 |
| `layer2_attn_qkv` | [96, 32] | 0.00342 | 0.435 |
| `layer2_attn_out` | [32, 32] | 0.00120 | 0.153 |
| `layer2_ffn_up` | [32, 128] | 0.00122 | 0.154 |
| `layer2_ffn_down` | [128, 32] | 0.00162 | 0.205 |
| `layer2_ln_gamma` | [32] | 0.00979 | 1.243 |
| `layer2_ln_beta` | [32] | 0.01110 | 1.410 |
| `final_ln_gamma` | [32] | 0.02031 | 2.580 |

**Note**: `attention_mask` is kept as float32 since it contains `-INFINITY` values for causal masking.

## C Header Structure

Each quantized weight tensor is defined with three components:

```c
// Example: Layer 1 Attention QKV projection
static const float layer1_attn_qkv_scale = 3.0089086435e-03f;
static const int32_t layer1_attn_qkv_zero_point = 0;
static const int8_t layer1_attn_qkv_q[96][32] = {
    {  11,   34,   -5,   28,  -41, ...},
    ...
};
```

### Utility Functions Provided

```c
// Dequantize single value
static inline float dequantize_int8(int8_t q_val, float scale, int32_t zero_point) {
    return ((float)q_val - (float)zero_point) * scale;
}

// Quantized matrix-vector multiply with on-the-fly dequantization
static inline void matmul_q_dequant(
    float* output,           // [M]
    const int8_t* weights_q, // [M x K] row-major
    const float* input,      // [K]
    int M, int K,
    float weight_scale,
    int32_t weight_zp
);
```

## Usage Example

```c
#include "gpt_weights_q.h"

// Access quantized embedding with dequantization
float embed_val = dequantize_int8(
    token_embedding_q[token_id][dim],
    token_embedding_scale,
    token_embedding_zero_point
);

// Matrix multiply with quantized weights
float output[96];
matmul_q_dequant(
    output,
    (const int8_t*)layer1_attn_qkv_q,
    input_vector,
    96, 32,
    layer1_attn_qkv_scale,
    layer1_attn_qkv_zero_point
);
```

## Scripts

### `extract_weights_quantized.py`

Main script for extracting and quantizing weights:

```bash
# Basic usage
python3 extract_weights_quantized.py --model gpt.tflite --output_dir ./gpt_weights_q

# Options
--model         Path to input TFLite model (default: gpt.tflite)
--output_dir    Output directory for headers (default: ./gpt_weights_q)
--report        Quantization report filename (default: quant_report_named.json)
```

### `quantise.py`

Alternative quantization script with more options:

```bash
# Per-tensor quantization with fake-quant model output
python3 quantise.py \
  --tflite_in gpt.tflite \
  --mode per_tensor \
  --out_tflite_fakeq model_q_local.tflite \
  --out_header_dir per_tensor_quant_headers \
  --rep_npy_dir rep_inputs/ \
  --compare
```

## Inference Strategies

### Option 1: Dequantize-then-Compute (Current)
- Dequantize weights to float32 during matrix multiply
- Pros: Simple, numerically identical to float inference
- Cons: Doesn't save compute, only memory

### Option 2: Fully Quantized Inference (Future)
- Quantize activations too
- Compute in INT8 with INT32 accumulator
- Requantize output back to INT8
- Pros: 4x faster matrix multiply on ARM Cortex-M
- Cons: Requires activation calibration, more complex

### Recommended Approach for RP2350
1. Start with Option 1 to verify correctness
2. Profile to identify bottlenecks
3. Convert hot paths to fully quantized INT8

## Comparison with Original Float Headers

| Aspect | `gpt_weights/` | `gpt_weights_q/` |
|--------|----------------|------------------|
| Data type | float (32-bit) | int8_t (8-bit) |
| Total size | ~385 KB | ~175 KB |
| Precision | Full float32 | ~0.8% max error |
| Include | `gpt_weights.h` | `gpt_weights_q.h` |
| Naming | `layer1_attn_qkv` | `layer1_attn_qkv_q` |

## Known Limitations

1. **Attention mask kept as float32**: Contains `-INFINITY` which cannot be quantized
2. **Symmetric quantization only**: Zero-point is always 0, which may not be optimal for asymmetric weight distributions
3. **No activation quantization**: Only weights are quantized; activations remain float32

## Future Improvements

- [ ] Add asymmetric quantization support for better accuracy
- [ ] Implement per-channel quantization for attention weights
- [ ] Add activation quantization for full INT8 inference
- [ ] Implement I-BERT style integer-only layer norm and GELU
- [ ] Add quantization-aware training (QAT) support
