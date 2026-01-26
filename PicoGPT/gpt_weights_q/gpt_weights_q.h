#ifndef GPT_WEIGHTS_Q_H
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
