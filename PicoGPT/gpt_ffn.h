#ifndef GPT_FFN_H
#define GPT_FFN_H

#include "arena_allocator.h"
#include "gpt_utils.h"
#include "gpt_profile.h"

#ifndef GPT_FFN_FUSE_GELU
#define GPT_FFN_FUSE_GELU 1
#endif

#if GPT_MULTICORE && GPT_FFN_FUSE_GELU
#error "Multicore FFN requires GPT_FFN_FUSE_GELU=0 so GELU can be dispatched separately"
#endif

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
 * Feed-Forward Network (FFN) operation
 * WITH RESIDUAL
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
static inline void ffn_forward_add_residual(
    float* input,
    const float* residual_input,
    const float* up_weights,
    const float* down_weights,
    float* output,
    float* hidden,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim,
    size_t total_elements
) {
    (void)total_elements;
    GPT_PROFILE_BEGIN(GPT_STAGE_FFN_UP);
#if GPT_FFN_FUSE_GELU
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = input + i * embed_dim;
        float* hidden_row = hidden + i * ffn_dim;
        size_t j = 0;
        for (; ffn_dim - j >= 4; j += 4) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (size_t k = 0; k < embed_dim; k++) {
                float a = row[k];
                const float* weights = up_weights + k * ffn_dim + j;
                sum0 += a * weights[0];
                sum1 += a * weights[1];
                sum2 += a * weights[2];
                sum3 += a * weights[3];
            }
            hidden_row[j] = gelu_activation(sum0);
            hidden_row[j + 1] = gelu_activation(sum1);
            hidden_row[j + 2] = gelu_activation(sum2);
            hidden_row[j + 3] = gelu_activation(sum3);
        }
        for (; j < ffn_dim; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < embed_dim; k++) {
                sum += row[k] * up_weights[k * ffn_dim + j];
            }
            hidden_row[j] = gelu_activation(sum);
        }
    }
    GPT_PROFILE_END(GPT_STAGE_FFN_UP);
#else
    matmul(input, seq_len, embed_dim, up_weights, ffn_dim, hidden);
    GPT_PROFILE_END(GPT_STAGE_FFN_UP);
    GPT_PROFILE_BEGIN(GPT_STAGE_GELU);
    apply_gelu(hidden, seq_len * ffn_dim);
    GPT_PROFILE_END(GPT_STAGE_GELU);
#endif

    /* The up-projection has completed on both cores before input is reused. */
    GPT_PROFILE_BEGIN(GPT_STAGE_FFN_DOWN_RESIDUAL);
    matmul_with_residual(hidden, seq_len, ffn_dim, down_weights, embed_dim, residual_input, output);
    GPT_PROFILE_END(GPT_STAGE_FFN_DOWN_RESIDUAL);
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
    float* input,
    const float* up_weights,
    const float* down_weights,
    const float* ln_gamma,
    const float* ln_beta,
    size_t seq_len,
    size_t embed_dim,
    size_t ffn_dim
) {
    // Pre-layer norm
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL;
    }
    
    GPT_PROFILE_BEGIN(GPT_STAGE_PRE_NORM);
    layer_norm(input, output, ln_gamma, ln_beta, seq_len, embed_dim, LAYER_NORM_EPS);
    GPT_PROFILE_END(GPT_STAGE_PRE_NORM);
    
    // Allocate hidden layer tensor
    float* hidden = arena_alloc_tensor_2d(arena, seq_len, ffn_dim);
    if (!hidden) {
        return NULL; // Out of memory
    }
    
    // Perform FFN computation
    // "output" here is the normalised input, and therefore input to the ffn

    size_t total_elements = seq_len * embed_dim;
    ffn_forward_add_residual(output, input, up_weights, down_weights, output, hidden,
                seq_len, embed_dim, ffn_dim, total_elements);


    
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

    // Apply within matmul
    
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