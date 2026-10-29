#ifndef GPT_FFN_H
#define GPT_FFN_H

#include "arena_allocator.h"
#include "gpt_utils.h"
#include "gpt_profile.h"

#ifndef GPT_FFN_FUSE_GELU
#define GPT_FFN_FUSE_GELU 1
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
    // First linear layer: hidden = input * up_weights
    // input: [seq_len, embed_dim], up_weights: [embed_dim, ffn_dim] -> hidden: [seq_len, ffn_dim]
    // matmul(input, seq_len, embed_dim, up_weights, ffn_dim, hidden);

    // combine matmul with GeLU to streamline
    size_t rows_a = seq_len;
    size_t cols_a = embed_dim;
    size_t cols_b = ffn_dim;
    float* A = input;
    const float* B = up_weights;
    float* C = hidden;
    GPT_PROFILE_BEGIN(GPT_STAGE_FFN_UP);
#if GPT_FFN_FUSE_GELU
    for (size_t i = 0; i < rows_a; i++) {
        const float* row = A + i * cols_a;
        float* output = C + i * cols_b;
        size_t j = 0;
        for (; cols_b - j >= 4; j += 4) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                float a = row[k];
                const float* weights = B + k * cols_b + j;
                sum0 += a * weights[0];
                sum1 += a * weights[1];
                sum2 += a * weights[2];
                sum3 += a * weights[3];
            }
            // erff actually requires a lot of registers, and therefore we have to write out
            // values to stack which slows this down v. just having GELU as a separate function
            // not unrolled
            output[j] = 0.5f * sum0 * (1.0f + erff(sum0 * 0.7071067811865475f));
            output[j + 1] = 0.5f * sum1 * (1.0f + erff(sum1 * 0.7071067811865475f));
            output[j + 2] = 0.5f * sum2 * (1.0f + erff(sum2 * 0.7071067811865475f));
            output[j + 3] = 0.5f * sum3 * (1.0f + erff(sum3 * 0.7071067811865475f));
        }
        for (; j < cols_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                sum += row[k] * B[k * cols_b + j];
            }
            output[j] = 0.5f * sum * (1.0f + erff(sum * 0.7071067811865475f));
        }
    }
    GPT_PROFILE_END(GPT_STAGE_FFN_UP);
#else
    // Keep the same four-output projection; only move GELU into its own pass.
    matmul_unroll4(input, seq_len, embed_dim, up_weights, ffn_dim, hidden);
    GPT_PROFILE_END(GPT_STAGE_FFN_UP);
    GPT_PROFILE_BEGIN(GPT_STAGE_GELU);
    apply_gelu(hidden, seq_len * ffn_dim);
    GPT_PROFILE_END(GPT_STAGE_GELU);
#endif

    // Second linear layer: output = hidden * down_weights
    // hidden: [seq_len, ffn_dim], down_weights: [ffn_dim, embed_dim] -> output: [seq_len, embed_dim]
    // the tensor "input" has had its values used in the first matmul, so it's safe to overwrite now
    // knowing that "input" and "output" tensors are actually the same tensor
    // matmul(hidden, seq_len, ffn_dim, down_weights, embed_dim, output);

    // Combine matmul with the residual add to streamline
    rows_a = seq_len;
    cols_a = ffn_dim;
    cols_b = embed_dim;
    A = hidden;
    B = down_weights;
    C = output;
    GPT_PROFILE_BEGIN(GPT_STAGE_FFN_DOWN_RESIDUAL);
    for (size_t i = 0; i < rows_a; i++) {
        const float* row = A + i * cols_a;
        float* output_row = C + i * cols_b;
        size_t j = 0;
        size_t element_number = i * cols_b;
        for (; cols_b - j >= 4; j += 4) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                float a = row[k];
                const float* weights = B + k * cols_b + j;
                sum0 += a * weights[0];
                sum1 += a * weights[1];
                sum2 += a * weights[2];
                sum3 += a * weights[3];
            }
            output_row[j] = sum0 + residual_input[element_number + j];
            output_row[j + 1] = sum1 + residual_input[element_number + j + 1];
            output_row[j + 2] = sum2 + residual_input[element_number + j + 2];
            output_row[j + 3] = sum3 + residual_input[element_number + j + 3];
        }
        for (; j < cols_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                sum += row[k] * B[k * cols_b + j];
            }
            output_row[j] = sum + residual_input[element_number + j];
        }
    }
    GPT_PROFILE_END(GPT_STAGE_FFN_DOWN_RESIDUAL);
    // Add in residual, to save allocating extra tensor?
    // Does this really speed things up?
    // for (size_t i = 0; i < total_elements; i++) {
    //     output[i] += residual_input[i];
    // }
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