#ifndef GPT_FFN_H
#define GPT_FFN_H

#include "arena_allocator.h"
#include "gpt_utils.h"

/**
 * Feed-Forward Network (FFN) operation
 * Implements: FFN(x) = GELU(x * W1) * W2
 * This is a two-layer MLP with GELU activation
 * 
 * @param input: Input tensor [seq_len, embed_dim]
 * @param up_weights: First layer weights [embed_dim, ffn_dim] 
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param output: Output tensor [seq_len, embed_dim]
 * @param hidden: Temporary tensor for intermediate results [seq_len, ffn_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 */
static inline void ffn_forward(
    const float* input,
    const float* up_weights,
    const float* down_weights,
    float* output,
    float* hidden,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    // First linear layer: hidden = input * up_weights
    // input: [seq_len, embed_dim], up_weights: [embed_dim, ffn_dim] -> hidden: [seq_len, ffn_dim]
    matmul(input, seq_len, embed_dim, up_weights, ffn_dim, hidden);
    
    // Apply GELU activation in-place

    // given thsi is the only gelu spot, we can write this into the matmul
    apply_gelu(hidden, seq_len * ffn_dim);
    
    // Second linear layer: output = hidden * down_weights  
    // hidden: [seq_len, ffn_dim], down_weights: [ffn_dim, embed_dim] -> output: [seq_len, embed_dim]
    matmul(hidden, seq_len, ffn_dim, down_weights, embed_dim, output);
}

/**
 * FFN with arena allocation
 * Allocates all necessary tensors and performs FFN computation
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param up_weights: First layer weights [embed_dim, ffn_dim]
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 * @return: Pointer to output tensor [seq_len, embed_dim]
 */
static inline float* ffn_op(
    Arena* arena,
    const float* input,
    const float* up_weights,
    const float* down_weights,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    // Allocate hidden layer tensor
    float* hidden = arena_alloc_tensor_2d(arena, seq_len, ffn_dim);
    if (!hidden) {
        return NULL; // Out of memory
    }
    
    // Allocate output tensor
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL; // Out of memory
    }
    
    // Perform FFN computation
    ffn_forward(input, up_weights, down_weights, output, hidden, 
                seq_len, embed_dim, ffn_dim);
    
    return output;
}

/**
 * FFN with residual connection and layer norm
 * output = input + FFN(LayerNorm(input))
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param up_weights: First layer weights [embed_dim, ffn_dim]
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param ln_gamma: Layer norm scale parameters [embed_dim]
 * @param ln_beta: Layer norm bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 * @return: Pointer to output tensor [seq_len, embed_dim]
 */
static inline float* ffn_with_residual(
    Arena* arena,
    const float* input,
    const float* up_weights,
    const float* down_weights,
    const float* ln_gamma,
    const float* ln_beta,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    // Pre-layer norm
    float* normalized_input = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!normalized_input) {
        return NULL;
    }
    
    layer_norm(input, normalized_input, ln_gamma, ln_beta, seq_len, embed_dim, LAYER_NORM_EPS);
    
    // FFN
    float* ffn_output = ffn_op(arena, normalized_input, up_weights, down_weights, 
                              seq_len, embed_dim, ffn_dim);
    if (!ffn_output) {
        return NULL;
    }

    // what's this extra arena allocation happening here?
    // we can skip this by rolling the add into the ffn
    
    // Residual connection
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL;
    }
    
    size_t total_elements = seq_len * embed_dim;
    add_tensors(input, ffn_output, output, total_elements);
    
    return output;
}

/**
 * Optimized FFN for single token (inference)
 * Processes only the last token for autoregressive generation
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input vector [embed_dim] (single token)
 * @param up_weights: First layer weights [embed_dim, ffn_dim]
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 * @return: Pointer to output vector [embed_dim]
 */
static inline float* ffn_single_token(
    Arena* arena,
    const float* input,
    const float* up_weights,
    const float* down_weights,
    size_t embed_dim,
    size_t ffn_dim
) {
    // Allocate hidden vector
    float* hidden = arena_alloc_tensor_1d(arena, ffn_dim);
    if (!hidden) {
        return NULL;
    }
    
    // Allocate output vector
    float* output = arena_alloc_tensor_1d(arena, embed_dim);
    if (!output) {
        return NULL;
    }
    
    // First linear layer
    matmul(input, 1, embed_dim, up_weights, ffn_dim, hidden);
    
    // GELU activation
    apply_gelu(hidden, ffn_dim);
    
    // Second linear layer
    matmul(hidden, 1, ffn_dim, down_weights, embed_dim, output);
    
    return output;
}

/**
 * Batch FFN operation
 * Processes multiple sequences in parallel (useful for training)
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [batch_size, seq_len, embed_dim]
 * @param up_weights: First layer weights [embed_dim, ffn_dim]
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param batch_size: Number of sequences
 * @param seq_len: Sequence length
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 * @return: Pointer to output tensor [batch_size, seq_len, embed_dim]
 */
static inline float* ffn_batch(
    Arena* arena,
    const float* input,
    const float* up_weights,
    const float* down_weights,
    size_t batch_size,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    size_t total_seq_len = batch_size * seq_len;
    
    // Allocate tensors for the entire batch
    float* hidden = arena_alloc_tensor_2d(arena, total_seq_len, ffn_dim);
    float* output = arena_alloc_tensor_2d(arena, total_seq_len, embed_dim);
    
    if (!hidden || !output) {
        return NULL;
    }
    
    // Process the entire batch as one large sequence
    ffn_forward(input, up_weights, down_weights, output, hidden,
                total_seq_len, embed_dim, ffn_dim);
    
    return output;
}

/**
 * FFN with bias terms (if the model includes bias)
 * Some models include bias terms in the linear layers
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param up_weights: First layer weights [embed_dim, ffn_dim]
 * @param up_bias: First layer bias [ffn_dim] (can be NULL)
 * @param down_weights: Second layer weights [ffn_dim, embed_dim]
 * @param down_bias: Second layer bias [embed_dim] (can be NULL)
 * @param seq_len: Sequence length
 * @param embed_dim: Input/output embedding dimension
 * @param ffn_dim: Hidden FFN dimension
 * @return: Pointer to output tensor [seq_len, embed_dim]
 */
static inline float* ffn_with_bias(
    Arena* arena,
    const float* input,
    const float* up_weights,
    const float* up_bias,
    const float* down_weights,
    const float* down_bias,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    // Allocate hidden layer tensor
    float* hidden = arena_alloc_tensor_2d(arena, seq_len, ffn_dim);
    if (!hidden) {
        return NULL;
    }
    
    // Allocate output tensor
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL;
    }
    
    // First linear layer with bias
    matmul_bias(input, seq_len, embed_dim, up_weights, ffn_dim, up_bias, hidden);
    
    // Apply GELU activation
    apply_gelu(hidden, seq_len * ffn_dim);
    
    // Second linear layer with bias
    matmul_bias(hidden, seq_len, ffn_dim, down_weights, embed_dim, down_bias, output);
    
    return output;
}

/**
 * Debug function to print FFN intermediate values
 */
static inline void print_ffn_debug(
    const char* name,
    const float* input,
    const float* hidden,
    const float* output,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    printf("\n=== %s FFN Debug ===\n", name);
    
    // Print input statistics
    printf("Input (first 8): ");
    for (size_t i = 0; i < 8 && i < seq_len * embed_dim; i++) {
        printf("%.4f ", input[i]);
    }
    printf("\n");
    
    // Print hidden layer statistics
    printf("Hidden (first 8): ");
    for (size_t i = 0; i < 8 && i < seq_len * ffn_dim; i++) {
        printf("%.4f ", hidden[i]);
    }
    printf("\n");
    
    // Print output statistics
    printf("Output (first 8): ");
    for (size_t i = 0; i < 8 && i < seq_len * embed_dim; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n");
    
    // Compute and print some statistics
    float max_hidden = hidden[0], min_hidden = hidden[0];
    for (size_t i = 1; i < seq_len * ffn_dim; i++) {
        if (hidden[i] > max_hidden) max_hidden = hidden[i];
        if (hidden[i] < min_hidden) min_hidden = hidden[i];
    }
    
    printf("Hidden range: [%.4f, %.4f]\n", min_hidden, max_hidden);
}

#endif // GPT_FFN_H