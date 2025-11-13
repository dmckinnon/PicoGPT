# GPT TensorFlow Lite Weight Extraction Summary

## Overview
Successfully extracted all weights from `gpt.tflite` and generated C header files for use in custom C implementations.

## Model Architecture Discovered
- **Model Type**: 2-layer Transformer (GPT-style)
- **Vocabulary Size**: 47 tokens
- **Maximum Sequence Length**: 31 tokens  
- **Embedding Dimensions**: 32
- **FFN Hidden Dimensions**: 128
- **Number of Attention Heads**: 1 (single-head attention)
- **Total Layers**: 2 transformer blocks

## Generated Files

### 1. `gpt_weights.h` (Master Header)
- Main include file that includes all other headers
- Contains convenient macros for accessing weights
- Provides model architecture documentation

### 2. `gpt_constants.h`
- Model configuration constants
- Mathematical constants (layer norm epsilon, attention scaling)
- Causal attention mask (31x31 lower triangular matrix)
- Final layer norm parameters
- Utility functions (GELU activation, softmax)

### 3. `gpt_embeddings.h`
- **Token Embedding**: `token_embedding[47][32]` - vocabulary embeddings
- **Position Embedding**: `position_embedding[31][32]` - positional encodings

### 4. `gpt_layer1.h` (First Transformer Layer)
- **Layer Norm**: `layer1_ln_gamma[32]` and `layer1_ln_beta[32]`
- **Attention QKV**: `layer1_attn_qkv[96][32]` 
  - Rows 0-31: Query projection weights
  - Rows 32-63: Key projection weights  
  - Rows 64-95: Value projection weights
- **Attention Output**: `layer1_attn_out[32][32]`
- **FFN Up**: `layer1_ffn_up[32][128]` - first FFN linear layer
- **FFN Down**: `layer1_ffn_down[128][32]` - second FFN linear layer

### 5. `gpt_layer2.h` (Second Transformer Layer)
- Same structure as Layer 1 with different weights
- All weights follow the same naming pattern with `layer2_` prefix

## Weight Tensor Mapping

| TensorFlow Lite Tensor | Purpose | Shape | C Array Name |
|------------------------|---------|-------|--------------|
| `arith.constant9` | Token embeddings | [47, 32] | `token_embedding` |
| `model_1/tf.math.add_107/Add/y` | Position embeddings | [31, 32] | `position_embedding` |
| `arith.constant13` | Attention mask | [31, 31] | `attention_mask` |
| `arith.constant4` | Layer 1 Attention QKV | [96, 32] | `layer1_attn_qkv` |
| `arith.constant3` | Layer 1 Attention Output | [32, 32] | `layer1_attn_out` |
| `arith.constant1` | Layer 1 FFN Up | [32, 128] | `layer1_ffn_up` |
| `arith.constant2` | Layer 1 FFN Down | [128, 32] | `layer1_ffn_down` |
| `arith.constant8` | Layer 2 Attention QKV | [96, 32] | `layer2_attn_qkv` |
| `arith.constant7` | Layer 2 Attention Output | [32, 32] | `layer2_attn_out` |
| `arith.constant5` | Layer 2 FFN Up | [32, 128] | `layer2_ffn_up` |
| `arith.constant6` | Layer 2 FFN Down | [128, 32] | `layer2_ffn_down` |
| Various `multiply` tensors | Layer norm parameters | [32] | `layer*_ln_gamma/beta` |

## Usage in C Code

To use these weights in your C implementation:

```c
#include "gpt_weights.h"

// Access token embedding for token 5, dimension 10
float embedding_val = TOKEN_EMBED(5, 10);

// Access layer 1 attention QKV weights  
float qkv_weight = LAYER1_ATTN_QKV(0, 0);  // Query weight [0,0]

// Layer norm parameters
float ln_scale = layer1_ln_gamma[0];
float ln_bias = layer1_ln_beta[0];

// Attention mask for causal masking
float mask_val = attention_mask[i][j];
```

## Implementation Notes

1. **Attention QKV Layout**: The QKV weights are stored as a single [96, 32] matrix:
   - Rows 0-31: Query projection weights
   - Rows 32-63: Key projection weights  
   - Rows 64-95: Value projection weights

2. **Layer Norm**: Each layer has gamma (scale) and beta (bias) parameters

3. **Attention Mask**: Pre-computed causal mask with 0.0 for allowed positions and -INFINITY for masked positions

4. **GELU Activation**: Utility function provided in constants header

5. **Weight Precision**: All weights stored as 32-bit floats with 8 decimal precision

## Scripts Generated

- `extract_weights.py`: Initial analysis script
- `extract_weights_enhanced.py`: Complete extraction script that can be reused for other TFLite models

The extraction script can be easily modified to work with other TensorFlow Lite models by adjusting the tensor identification logic based on shapes and naming patterns.