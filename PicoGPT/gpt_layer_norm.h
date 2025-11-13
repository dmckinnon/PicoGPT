#ifndef GPT_LAYER_NORM_H
#define GPT_LAYER_NORM_H

#include "arena_allocator.h"
#include "gpt_weights/gpt_constants.h"
#include "gpt_utils.h"

/**
 * Layer normalization operation
 * Normalizes input tensor across the last dimension (embed_dim)
 * output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
 * 
 * @param input: Input tensor [seq_len, embed_dim]
 * @param output: Output tensor [seq_len, embed_dim] 
 * @param gamma: Scale parameters [embed_dim]
 * @param beta: Bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 */
static inline void layer_norm_op(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    size_t seq_len,
    size_t embed_dim
) {
    layer_norm(input, output, gamma, beta, seq_len, embed_dim, LAYER_NORM_EPS);
}

/**
 * Layer normalization with arena allocation
 * Allocates output tensor and performs layer normalization
 * 
 * @param arena: Memory arena for allocation
 * @param input: Input tensor [seq_len, embed_dim]
 * @param gamma: Scale parameters [embed_dim]
 * @param beta: Bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to allocated output tensor [seq_len, embed_dim]
 */
static inline float* layer_norm_alloc(
    Arena* arena,
    const float* input,
    const float* gamma,
    const float* beta,
    size_t seq_len,
    size_t embed_dim
) {
    // Allocate output tensor
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL; // Out of memory
    }
    
    // Perform layer normalization
    layer_norm_op(input, output, gamma, beta, seq_len, embed_dim);
    
    return output;
}

/**
 * In-place layer normalization
 * Normalizes the input tensor in-place
 * 
 * @param data: Input/output tensor [seq_len, embed_dim]
 * @param gamma: Scale parameters [embed_dim]
 * @param beta: Bias parameters [embed_dim] 
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 */
static inline void layer_norm_inplace(
    float* data,
    const float* gamma,
    const float* beta,
    size_t seq_len,
    size_t embed_dim
) {
    layer_norm_op(data, data, gamma, beta, seq_len, embed_dim);
}

/**
 * Pre-layer normalization (common in modern transformers)
 * Applies layer norm before the main operation
 * 
 * @param arena: Memory arena for allocation
 * @param input: Input tensor [seq_len, embed_dim]
 * @param gamma: Scale parameters [embed_dim]
 * @param beta: Bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to normalized tensor [seq_len, embed_dim]
 */
static inline float* pre_layer_norm(
    Arena* arena,
    const float* input,
    const float* gamma,
    const float* beta,
    size_t seq_len,
    size_t embed_dim
) {
    return layer_norm_alloc(arena, input, gamma, beta, seq_len, embed_dim);
}

/**
 * Post-layer normalization with residual connection
 * output = LayerNorm(input + residual)
 * 
 * @param arena: Memory arena for allocation
 * @param input: Main input tensor [seq_len, embed_dim]
 * @param residual: Residual tensor to add [seq_len, embed_dim]
 * @param gamma: Scale parameters [embed_dim]
 * @param beta: Bias parameters [embed_dim]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 * @return: Pointer to normalized output tensor [seq_len, embed_dim]
 */
static inline float* post_layer_norm_with_residual(
    Arena* arena,
    const float* input,
    const float* residual,
    const float* gamma,
    const float* beta,
    size_t seq_len,
    size_t embed_dim
) {
    // Allocate temporary tensor for sum
    float* sum = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!sum) {
        return NULL; // Out of memory
    }
    
    // Add input + residual
    size_t total_elements = seq_len * embed_dim;
    add_tensors(input, residual, sum, total_elements);
    
    // Allocate output tensor
    float* output = arena_alloc_tensor_2d(arena, seq_len, embed_dim);
    if (!output) {
        return NULL; // Out of memory
    }
    
    // Apply layer normalization
    layer_norm_op(sum, output, gamma, beta, seq_len, embed_dim);
    
    return output;
}

/**
 * Layer normalization statistics computation
 * Computes mean and variance for debugging/analysis
 * 
 * @param input: Input tensor [seq_len, embed_dim]
 * @param means: Output means [seq_len]
 * @param variances: Output variances [seq_len]
 * @param seq_len: Sequence length
 * @param embed_dim: Embedding dimension
 */
static inline void layer_norm_stats(
    const float* input,
    float* means,
    float* variances,
    size_t seq_len,
    size_t embed_dim
) {
    for (size_t seq = 0; seq < seq_len; seq++) {
        const float* input_row = input + seq * embed_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < embed_dim; i++) {
            mean += input_row[i];
        }
        mean /= embed_dim;
        means[seq] = mean;
        
        // Compute variance
        float variance = 0.0f;
        for (size_t i = 0; i < embed_dim; i++) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= embed_dim;
        variances[seq] = variance;
    }
}

/**
 * Debug function to print layer norm statistics
 */
static inline void print_layer_norm_debug(
    const char* name,
    const float* input,
    const float* output,
    size_t seq_len,
    size_t embed_dim
) {
    printf("\n=== %s Layer Norm Debug ===\n", name);
    
    // Print first few values of input and output
    printf("Input (first 8): ");
    for (size_t i = 0; i < 8 && i < seq_len * embed_dim; i++) {
        printf("%.4f ", input[i]);
    }
    printf("\n");
    
    printf("Output (first 8): ");
    for (size_t i = 0; i < 8 && i < seq_len * embed_dim; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n");
    
    // Compute and print statistics for first sequence position
    if (seq_len > 0) {
        float mean = 0.0f, variance = 0.0f;
        for (size_t i = 0; i < embed_dim; i++) {
            mean += input[i];
        }
        mean /= embed_dim;
        
        for (size_t i = 0; i < embed_dim; i++) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= embed_dim;
        
        printf("First position - Mean: %.6f, Variance: %.6f, Std: %.6f\n", 
               mean, variance, sqrtf(variance));
    }
}

#endif // GPT_LAYER_NORM_H