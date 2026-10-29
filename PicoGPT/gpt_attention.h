#ifndef GPT_ATTENTION_H
#define GPT_ATTENTION_H

#include "arena_allocator.h"
#include "gpt_utils.h"
#include "gpt_weights/gpt_constants.h"
#include "gpt_profile.h"

#ifndef GPT_ATTENTION_UNROLL4
#define GPT_ATTENTION_UNROLL4 0
#endif

/**
 * QKV projection operation
 * Projects input to Query, Key, Value using combined QKV weight matrix
 * 
 * @param input: Input tensor [seq_len, embed_dim]
 * @param qkv_weights: QKV projection weights [3*embed_dim, embed_dim]
 * @param query: Output query tensor [seq_len, embed_dim]
 * @param key: Output key tensor [seq_len, embed_dim]  
 * @param value: Output value tensor [seq_len, embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 */
static inline void qkv_projection(
    const float* input,
    const float* qkv_weights,
    float* query,
    float* key,
    float* value,
    size_t seq_len,
    size_t embed_dim
) {
    // QKV weights are arranged as [Q_weights; K_weights; V_weights]
    // Each sub-matrix is [input_dim, output_dim], transposed during extraction.
    
    const float* q_weights = qkv_weights;                           // [0:embed_dim, :]
    const float* k_weights = qkv_weights + embed_dim * embed_dim;   // [embed_dim:2*embed_dim, :]
    const float* v_weights = qkv_weights + 2 * embed_dim * embed_dim; // [2*embed_dim:3*embed_dim, :]
    
    // Project to Query
    matmul_unroll4(input, seq_len, embed_dim, q_weights, embed_dim, query);
    
    // Project to Key
    matmul_unroll4(input, seq_len, embed_dim, k_weights, embed_dim, key);
    
    // Project to Value
    matmul_unroll4(input, seq_len, embed_dim, v_weights, embed_dim, value);
}

/* One query against all key rows, sharing each query element across four scores. */
static inline void attention_query_scores(
    const float* query,
    const float* key,
    float* scores,
    size_t seq_len,
    size_t embed_dim
) {
    const float scale = INV_SQRT_HEAD_DIM;
    size_t j = 0;
#if GPT_ATTENTION_UNROLL4
    for (; seq_len - j >= 4; j += 4) {
        const float* key0 = key + j * embed_dim;
        const float* key1 = key0 + embed_dim;
        const float* key2 = key1 + embed_dim;
        const float* key3 = key2 + embed_dim;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (size_t k = 0; k < embed_dim; k++) {
            float q = query[k];
            sum0 += q * key0[k];
            sum1 += q * key1[k];
            sum2 += q * key2[k];
            sum3 += q * key3[k];
        }
        scores[j] = sum0 * scale;
        scores[j + 1] = sum1 * scale;
        scores[j + 2] = sum2 * scale;
        scores[j + 3] = sum3 * scale;
    }
#endif
    for (; j < seq_len; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < embed_dim; k++) {
            sum += query[k] * key[j * embed_dim + k];
        }
        scores[j] = sum * scale;
    }
}

static inline void attention_weighted_values(
    const float* scores,
    const float* value,
    float* output,
    size_t query_count,
    size_t seq_len,
    size_t embed_dim
) {
#if GPT_ATTENTION_UNROLL4
    matmul_unroll4(scores, query_count, seq_len, value, embed_dim, output);
#else
    matmul_baseline(scores, query_count, seq_len, value, embed_dim, output);
#endif
}

/**
 * Scaled dot-product attention
 * Computes attention weights and applies to values
 * 
 * @param query: Query tensor [seq_len, embed_dim]
 * @param key: Key tensor [seq_len, embed_dim]
 * @param value: Value tensor [seq_len, embed_dim]
 * @param output: Output tensor [seq_len, embed_dim]
 * @param attention_scores: Temporary tensor for scores [seq_len, seq_len]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension (head dimension)
 */
static inline void scaled_dot_product_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* attention_scores,
    size_t seq_len,
    size_t embed_dim
) {
    // Compute Q * K^T / root(embed_dim)
    // query: [seq_len, embed_dim], key^T: [embed_dim, seq_len] -> scores: [seq_len, seq_len]
    GPT_PROFILE_BEGIN(GPT_STAGE_SCORES);
    for (size_t i = 0; i < seq_len; i++) {
        attention_query_scores(query + i * embed_dim, key,
                               attention_scores + i * seq_len, seq_len, embed_dim);
    }
    GPT_PROFILE_END(GPT_STAGE_SCORES);
    
    // Apply causal mask and softmax for each query position
    GPT_PROFILE_BEGIN(GPT_STAGE_MASK_SOFTMAX);
    for (size_t i = 0; i < seq_len; i++) {
        float* score_row = attention_scores + i * seq_len;
        
        // Apply causal mask (can only attend to previous and current positions)
        size_t j = i + 1;
#if GPT_ATTENTION_UNROLL4
        for (; seq_len - j >= 4; j += 4) {
            score_row[j] = -INFINITY;
            score_row[j + 1] = -INFINITY;
            score_row[j + 2] = -INFINITY;
            score_row[j + 3] = -INFINITY;
        }
#endif
        for (; j < seq_len; j++) {
            score_row[j] = -INFINITY;
        }
        
        // Apply softmax to this row
        softmax_inplace(score_row, seq_len);
    }
    GPT_PROFILE_END(GPT_STAGE_MASK_SOFTMAX);
    
    // Apply attention weights to values: output = attention_scores * value
    // attention_scores: [seq_len, seq_len], value: [seq_len, embed_dim] -> output: [seq_len, embed_dim]
    GPT_PROFILE_BEGIN(GPT_STAGE_WEIGHTED_VALUES);
    attention_weighted_values(attention_scores, value, output, seq_len, seq_len, embed_dim);
    GPT_PROFILE_END(GPT_STAGE_WEIGHTED_VALUES);
}

/**
 * Attention output projection
 * Projects attention output back to original dimension
 * 
 * @param attention_output: Input from attention [seq_len, embed_dim]
 * @param output_weights: Output projection weights [embed_dim, embed_dim]
 * @param output: Final output tensor [seq_len, embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 */
static inline void attention_output_projection(
    const float* attention_output,
    const float* output_weights,
    float* output,
    size_t seq_len,
    size_t embed_dim
) {
    matmul(attention_output, seq_len, embed_dim, output_weights, embed_dim, output);
}

/**
 * Complete multi-head self-attention operation
 * Performs the full attention computation: QKV projection + scaled attention + output projection
 * 
 * @param arena: Memory arena for temporary allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param qkv_weights: QKV projection weights [3*embed_dim, embed_dim]
 * @param output_weights: Output projection weights [embed_dim, embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to output tensor [seq_len, embed_dim]
 */
static inline float* multi_head_attention(
    Arena* arena,
    const float* input,
    const float* qkv_weights,
    const float* output_weights,
    size_t seq_len,
    size_t embed_dim
) {
    // Allocate temporary tensors
    float* query = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* key = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* value = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* attention_scores = arena_alloc_tensor_2d(arena, seq_len, seq_len);
    float* attention_output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* final_output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    
    if (!query || !key || !value || !attention_scores || !attention_output || !final_output) {
        return NULL; // Out of memory
    }
    
    // Step 1: QKV Projection
    GPT_PROFILE_BEGIN(GPT_STAGE_QKV);
    qkv_projection(input, qkv_weights, query, key, value, seq_len, embed_dim);
    GPT_PROFILE_END(GPT_STAGE_QKV);
    
    // Step 2: Scaled Dot-Product Attention
    scaled_dot_product_attention(query, key, value, attention_output, 
                                attention_scores, seq_len, embed_dim);
    
    // Step 3: Output Projection
    GPT_PROFILE_BEGIN(GPT_STAGE_ATTN_PROJECTION);
    attention_output_projection(attention_output, output_weights, final_output, 
                               seq_len, embed_dim);
    GPT_PROFILE_END(GPT_STAGE_ATTN_PROJECTION);
    
    return final_output;
}

/**
 * Attention with residual connection
 * output = input + MultiHeadAttention(LayerNorm(input))
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param qkv_weights: QKV projection weights [3*embed_dim, embed_dim]
 * @param output_weights: Output projection weights [embed_dim, embed_dim]
 * @param ln_gamma: Layer norm scale parameters [embed_dim]
 * @param ln_beta: Layer norm bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to output tensor [seq_len, embed_dim]
 */
static inline float* attention_with_residual(
    Arena* arena,
    const float* input,
    const float* qkv_weights,
    const float* output_weights,
    const float* ln_gamma,
    const float* ln_beta,
    size_t seq_len,
    size_t embed_dim
) {
    // Pre-layer norm
    float* normalized_input = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!normalized_input) {
        return NULL;
    }
    
    GPT_PROFILE_BEGIN(GPT_STAGE_PRE_NORM);
    layer_norm(input, normalized_input, ln_gamma, ln_beta, seq_len, embed_dim, LAYER_NORM_EPS);
    GPT_PROFILE_END(GPT_STAGE_PRE_NORM);
    
    // Attention
    float* attention_output = multi_head_attention(arena, normalized_input, qkv_weights, 
                                                  output_weights, seq_len, embed_dim);
    if (!attention_output) {
        return NULL;
    }
    
    // Residual connection
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL;
    }
    
    size_t total_elements = seq_len * embed_dim;
    GPT_PROFILE_BEGIN(GPT_STAGE_ATTN_RESIDUAL);
    add_tensors(input, attention_output, output, total_elements);
    GPT_PROFILE_END(GPT_STAGE_ATTN_RESIDUAL);
    
    return output;
}

/**
 * Debug function to print attention weights
 */
static inline void print_attention_debug(
    const char* name,
    const float* attention_scores,
    size_t seq_len,
    size_t max_print
) {
    printf("\n=== %s Attention Scores ===\n", name);
    
    size_t print_len = seq_len < max_print ? seq_len : max_print;
    
    for (size_t i = 0; i < print_len; i++) {
        printf("Query %zu -> [", i);
        for (size_t j = 0; j < print_len; j++) {
            printf("%.3f", attention_scores[i * seq_len + j]);
            if (j < print_len - 1) printf(", ");
        }
        if (seq_len > max_print) {
            printf(", ... (%zu total)", seq_len);
        }
        printf("]\n");
    }
    
    if (seq_len > max_print) {
        printf("... (%zu total queries)\n", seq_len);
    }
}

/**
 * Optimized attention for inference (single token generation)
 * Computes attention for only the last token position
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim] 
 * @param qkv_weights: QKV projection weights [3*embed_dim, embed_dim]
 * @param output_weights: Output projection weights [embed_dim, embed_dim]
 * @param seq_len: Current sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to output for last position [1, embed_dim]
 */
static inline float* attention_last_token_only(
    Arena* arena,
    const float* input,
    const float* qkv_weights, 
    const float* output_weights,
    size_t seq_len,
    size_t embed_dim
) {
    // For inference, we only need to compute attention for the last token
    // But we still need all previous tokens' K and V for the attention computation
    
    // Full QKV projection (needed for complete K,V matrices)
    float* query = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* key = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    float* value = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    
    if (!query || !key || !value) {
        return NULL;
    }
    
    qkv_projection(input, qkv_weights, query, key, value, seq_len, embed_dim);
    
    // Compute attention only for last query position
    float* last_query = query + (seq_len - 1) * embed_dim;
    float* attention_scores = arena_alloc_tensor_1d(arena, seq_len);
    float* output_vector = arena_alloc_tensor_1d(arena, embed_dim);
    
    if (!attention_scores || !output_vector) {
        return NULL;
    }
    
    // Compute attention scores for last query against all keys
    attention_query_scores(last_query, key, attention_scores, seq_len, embed_dim);
    
    // Apply softmax
    softmax_inplace(attention_scores, seq_len);
    
    // Apply attention to values
    attention_weighted_values(attention_scores, value, output_vector, 1, seq_len, embed_dim);
    
    // Output projection
    float* final_output = arena_alloc_tensor_1d(arena, embed_dim);
    if (!final_output) {
        return NULL;
    }
    
    matmul(output_vector, 1, embed_dim, output_weights, embed_dim, final_output);
    
    return final_output;
}

#endif // GPT_ATTENTION_H