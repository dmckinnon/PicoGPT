#ifndef GPT_WEIGHTS_H
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
