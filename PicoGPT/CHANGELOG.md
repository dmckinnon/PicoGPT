# PicoGPT Project Context & Changelog

> **Purpose**: This document serves as a global context reference for any agent (human or AI) working on this repository. It describes the project state, architecture decisions, what works, what doesn't, and the history of changes.

---

## Project Overview

**PicoGPT** is a C implementation of a small GPT-style transformer model designed to run on the **Raspberry Pi Pico 2 (RP2350)** microcontroller. The model weights are extracted from a TensorFlow Lite file and embedded directly into C headers.

### Target Hardware
- **MCU**: RP2350 (Raspberry Pi Pico 2)
- **RAM**: 520 KB SRAM
- **Flash**: 4 MB (16 MB on some boards)
- **CPU**: Dual ARM Cortex-M33 @ 150 MHz

### Model Architecture
| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 47 tokens |
| Max Sequence Length | 31 tokens |
| Embedding Dimension | 32 |
| FFN Hidden Dimension | 128 |
| Number of Layers | 2 |
| Attention Heads | 1 (single-head) |
| Total Parameters | ~27K |

---

## Repository Structure

```
PicoGPT/
├── main.c                      # Main entry point with test harness
├── simple_test.c               # Simplified test for weight access
├── attention_test.c            # INT8 quantized attention implementation
├── Makefile                    # Build system
│
├── gpt_model.h                 # High-level model API
├── gpt_attention.h             # Attention layer implementation
├── gpt_ffn.h                   # Feed-forward network implementation
├── gpt_layer_norm.h            # Layer normalization
├── gpt_embeddings_ops.h        # Embedding lookup operations
├── gpt_utils.h                 # Utility functions
├── arena_allocator.h           # Memory arena allocator
│
├── gpt_weights/                # Float32 weight headers
│   ├── gpt_weights.h           # Master include
│   ├── gpt_constants.h         # Model constants
│   ├── gpt_embeddings.h        # Token & position embeddings
│   ├── gpt_layer1.h            # Layer 1 weights
│   └── gpt_layer2.h            # Layer 2 weights
│
├── gpt_weights_q/              # INT8 quantized weight headers
│   ├── gpt_weights_q.h         # Master include
│   ├── gpt_constants_q.h       # Constants + dequant utilities
│   ├── gpt_embeddings_q.h      # Quantized embeddings
│   ├── gpt_layer1_q.h          # Layer 1 quantized weights
│   ├── gpt_layer2_q.h          # Layer 2 quantized weights
│   └── quant_report_named.json # Quantization parameters
│
├── extract_weights.py          # Basic weight extraction
├── extract_weights_enhanced.py # Enhanced extraction with naming
├── extract_weights_quantized.py# Quantized header generation
├── quantise.py                 # TFLite quantization tool
├── generate_rep_data.py        # Representative data generator
│
├── gpt.tflite                  # Original float32 model
├── model_q_local.tflite        # Per-tensor quantized model
├── gpt_global_q.tflite         # Global quantized model (deprecated)
│
├── WEIGHT_EXTRACTION_SUMMARY.md  # Float extraction documentation
├── QUANTIZATION_SUMMARY.md       # Quantization documentation
└── CHANGELOG.md                  # This file
```

---

## Current State Summary

### ✅ What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Float32 weight extraction | ✅ Complete | `extract_weights_enhanced.py` |
| Float32 C headers | ✅ Complete | `gpt_weights/*.h` |
| Per-tensor INT8 quantization | ✅ Complete | `extract_weights_quantized.py` |
| INT8 C headers with scales | ✅ Complete | `gpt_weights_q/*.h` |
| Basic test (weight access) | ✅ Passes | `simple_test.c` |
| Arena allocator | ✅ Works on desktop, needs on-device testing | `arena_allocator.h` |
| Makefile build system | ✅ Works | Host builds succeed |

### ⚠️ Partially Working

| Component | Status | Notes |
|-----------|--------|-------|
| Main inference (`main.c`) | ⚠️ Untested | Compiles but not validated |
| Attention implementation | ⚠️ In progress | `attention_test.c` needs validation |
| RP2350 cross-compilation | ⚠️ Not tested | `pico_sdk_import.cmake` present |

### ❌ Not Yet Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Full INT8 inference | ❌ TODO | Activations still float32 |
| On-device testing | ❌ TODO | No hardware test yet |
| Tokenizer | ❌ TODO | `vocab.cpp` exists but not integrated |
| Accuracy validation | ❌ TODO | Need to compare with TFLite |

---

## Changelog

### 2026-01-22: INT8 Quantized Headers with Named Tensors

**Added:**
- `extract_weights_quantized.py` - New script combining weight extraction with per-tensor quantization
- `gpt_weights_q/` directory with properly named quantized headers:
  - `gpt_weights_q.h` - Master header
  - `gpt_constants_q.h` - Constants and utility functions
  - `gpt_embeddings_q.h` - Token and position embeddings
  - `gpt_layer1_q.h` - Layer 1 weights with scales
  - `gpt_layer2_q.h` - Layer 2 weights with scales
  - `quant_report_named.json` - Quantization parameters

**Changed:**
- Quantized tensors now use meaningful names (`layer1_attn_qkv_q` instead of `arith_constant4_q`)
- Each tensor includes `*_scale` and `*_zero_point` alongside the `*_q` array

**Technical Details:**
- Per-tensor symmetric INT8 quantization (zero_point = 0)
- Scales range from 0.00112 to 0.02031
- 75% memory reduction (float32 → int8)

---

### 2026-01-16: Quantization Infrastructure

**Added:**
- `quantise.py` - TFLite weight quantization script with multiple modes
- `generate_rep_data.py` - Representative data generator for calibration
- `rep_inputs/` - Directory with sample input data (sample_0000.npy, etc.)
- `per_tensor_quant_headers/` - Per-tensor quantized headers (raw tensor names)
- `global_quant_headers/` - Global quantization headers (deprecated)
- `model_q_local.tflite` - Per-tensor fake-quantized TFLite model
- `gpt_global_q.tflite` - Global quantized model (failed - too much precision loss)
- `quant_report.json` - Quantization parameters report

**Findings:**
- Global quantization failed: large weights in one tensor caused small weights to round to zero
- Per-tensor quantization successful: each tensor gets optimal scale
- Fake-quantization preserves float32 format while simulating quantization effects

---

### 2025-10-25: Initial Float32 Weight Extraction

**Added:**
- `extract_weights.py` - Basic analysis script
- `extract_weights_enhanced.py` - Complete extraction with tensor identification
- `gpt_weights/` directory:
  - `gpt_weights.h` - Master header
  - `gpt_constants.h` - Model constants, attention mask, GELU
  - `gpt_embeddings.h` - Token [47×32] and position [31×32] embeddings
  - `gpt_layer1.h` - Layer 1 weights (QKV, output, FFN up/down, layer norm)
  - `gpt_layer2.h` - Layer 2 weights
- `WEIGHT_EXTRACTION_SUMMARY.md` - Documentation

**Technical Details:**
- Discovered model architecture from tensor shapes
- Mapped TFLite tensor names (`arith.constant*`) to meaningful names
- Attention QKV stored as [96×32] matrix (Q, K, V concatenated)

---

### 2025-10-25: Initial C Implementation

**Added:**
- `arena_allocator.h` - Stack-based memory allocator for embedded
- `gpt_utils.h` - Math utilities (softmax, GELU, etc.)
- `gpt_embeddings_ops.h` - Embedding lookup
- `gpt_layer_norm.h` - Layer normalization
- `gpt_attention.h` - Multi-head attention (single head)
- `gpt_ffn.h` - Feed-forward network
- `gpt_model.h` - High-level model API
- `main.c` - Test harness
- `simple_test.c` - Weight access test
- `Makefile` - Build system

---

### Initial Commit

**Added:**
- `gpt.tflite` - Original trained model (130 KB)
- `vocab.cpp` - Vocabulary handling (unused)
- `pico_sdk_import.cmake` - Pico SDK integration

---

## Test Status

### Tests That Pass ✅

```bash
make simple_test && ./simple_test
```
- Weight access test: Verifies all headers can be included and weights accessed
- Memory analysis: Estimates memory usage for different sequence lengths
- Basic operations: Matrix multiply, layer norm on sample data

### Tests Not Yet Run ⚠️

| Test | Command | Status |
|------|---------|--------|
| Main inference | `make && ./gpt_model` | Not validated |
| INT8 attention | `gcc attention_test.c -o attention_test` | Compiles, not run |
| Cross-compile | `cmake -DPICO_SDK_PATH=...` | Not attempted |
| On-device | Flash to Pico 2 | No hardware test |

### Validation TODO

1. [ ] Compare `simple_test` output against TFLite interpreter
2. [ ] Validate attention scores match reference implementation
3. [ ] End-to-end inference comparison with Python
4. [ ] Memory usage profiling on actual RP2350

---

## Key Technical Decisions

### 1. Weights in Headers (not binary files)
**Why**: Simpler deployment to MCU, compiler can optimize access patterns
**Trade-off**: Larger compile times, but weights are in flash not RAM

### 2. Arena Allocator
**Why**: No malloc/free fragmentation, predictable memory usage
**Location**: `arena_allocator.h`

### 3. Per-Tensor Quantization
**Why**: Global quantization failed (outliers zeroed small weights)
**Result**: Each tensor has its own scale factor stored alongside

### 4. Float32 Activations with INT8 Weights
**Why**: Simplest correct implementation first
**Future**: Full INT8 inference for 4x speedup

### 5. Single Attention Head
**Why**: Model was trained this way (simplifies implementation)
**Note**: QKV matrix is [96×32] = 3 × [32×32] for Q, K, V

---

## Known Issues & Gotchas

### 1. Attention Mask Contains Infinity
The causal attention mask uses `-INFINITY` for masked positions. This is kept as float32 even in quantized headers.

### 2. Layer Norm Beta Misnaming
In `multiply_102` etc., the TFLite names don't clearly indicate gamma vs beta. Current mapping based on usage context.

### 3. Tensor Index Order
TFLite tensors are not in a predictable order. The extraction script uses shape + name pattern matching.

### 4. FFN Dimension Mismatch
`attention_test.c` uses `HIDDEN_DIM=96`, but actual FFN hidden is 128. This may be a test artifact.

---

## Environment Setup

### Python (for weight extraction)
```bash
cd PicoGPT
python3 -m venv gpt_venv
source gpt_venv/bin/activate
pip install tensorflow numpy flatbuffers
```

### Host Build (Linux/macOS)
```bash
make all        # Build main and simple_test
make test       # Run simple_test
./simple_test   # Direct run
```

### Cross-Compile for RP2350 (TODO)
```bash
mkdir build && cd build
cmake -DPICO_SDK_PATH=/path/to/pico-sdk ..
make
```

---

## Files to Regenerate

If the TFLite model changes, regenerate headers with:

```bash
# Activate Python environment
source gpt_venv/bin/activate

# Generate float32 headers
python3 extract_weights_enhanced.py

# Generate INT8 quantized headers
python3 extract_weights_quantized.py --model gpt.tflite --output_dir ./gpt_weights_q
```

---

## Contact & References

- **TensorFlow Lite Schema**: Used for parsing `.tflite` files
- **I-BERT Paper**: Reference for integer-only transformer ops: https://arxiv.org/pdf/2101.01321
- **RP2350 Datasheet**: Memory constraints and ARM Cortex-M33 capabilities

---

*Last updated: 2026-01-23*
