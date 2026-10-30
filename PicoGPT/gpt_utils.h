#ifndef GPT_UTILS_H
#define GPT_UTILS_H

#include "gpt_multicore.h"
#include <math.h>
#include <stdint.h>
#include <stddef.h>

// Mathematical constants
#define GPT_PI 3.14159265358979323846f
#define GPT_E  2.71828182845904523536f

#ifndef GPT_MATMUL_UNROLL4
#define GPT_MATMUL_UNROLL4 0
#endif

/**
 * GELU activation function (moved to constants to avoid redefinition)
 * Using the version from gpt_constants.h
 */

/**
 * Fast approximation of GELU
 * GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
 */
static inline float gelu_fast(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/**
 * Softmax function for a vector
 * Numerically stable version using max subtraction
 */
static inline void softmax(float* input, float* output, size_t length) {
    // Find maximum value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (size_t i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

/**
 * In-place softmax
 */
static inline void softmax_inplace(float* data, size_t length) {
    softmax(data, data, length);
}

/**
 * Layer normalization
 * output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
 */
static inline void layer_norm(
    const float* input,           // Input tensor [seq_len, embed_dim]
    float* output,                // Output tensor [seq_len, embed_dim]
    const float* gamma,           // Scale parameters [embed_dim]
    const float* beta,            // Bias parameters [embed_dim]
    size_t seq_len,
    size_t embed_dim,
    float epsilon
) {
    for (size_t seq = 0; seq < seq_len; seq++) {
        const float* input_row = input + seq * embed_dim;
        float* output_row = output + seq * embed_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < embed_dim; i++) {
            mean += input_row[i];
        }
        mean /= embed_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (size_t i = 0; i < embed_dim; i++) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= embed_dim;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        for (size_t i = 0; i < embed_dim; i++) {
            output_row[i] = gamma[i] * (input_row[i] - mean) * inv_std + beta[i];
        }
    }
}

/**
 * Matrix multiplication: C = A * B
 * A: [rows_a, cols_a]
 * B: [cols_a, cols_b] 
 * C: [rows_a, cols_b]
 */
static inline void matmul_baseline(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b,
    float* C
) {
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}


// Need to multithread this. Split the matrix in half
// Still means we may be accessing the banks at the same time
// Even with writing too
// so, question: which is slower? Computation or memory IO?
static inline void matmul_unroll4(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b,
    float* C
) {
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
            output[j] = sum0;
            output[j + 1] = sum1;
            output[j + 2] = sum2;
            output[j + 3] = sum3;
        }
        for (; j < cols_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                sum += row[k] * B[k * cols_b + j];
            }
            output[j] = sum;
        }
    }
}

/* C must not overlap A or B; exact C == residual is supported. */
static inline void matmul_with_residual_single_core(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b,
    const float* residual,
    float* C
)
{
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
            output_row[j] = sum0 + residual[element_number + j];
            output_row[j + 1] = sum1 + residual[element_number + j + 1];
            output_row[j + 2] = sum2 + residual[element_number + j + 2];
            output_row[j + 3] = sum3 + residual[element_number + j + 3];
        }
        for (; j < cols_b; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                sum += row[k] * B[k * cols_b + j];
            }
            output_row[j] = sum + residual[element_number + j];
        }
    }
}

/* Neither kernel supports output overlapping either input. */
static inline void matmul_single_core(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b,
    float* C
) {
#if GPT_MATMUL_UNROLL4
    matmul_unroll4(A, rows_a, cols_a, B, cols_b, C);
#else
    matmul_baseline(A, rows_a, cols_a, B, cols_b, C);
#endif
}

static inline void matmul(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b, float* C
) {
#if GPT_MULTICORE
    gpt_multicore_matmul(A, rows_a, cols_a, B, cols_b, NULL, C);
#else
    matmul_single_core(A, rows_a, cols_a, B, cols_b, C);
#endif
}

static inline void matmul_with_residual(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b, const float* residual, float* C
) {
#if GPT_MULTICORE
    gpt_multicore_matmul(A, rows_a, cols_a, B, cols_b, residual, C);
#else
    matmul_with_residual_single_core(A, rows_a, cols_a, B, cols_b, residual, C);
#endif
}

/**
 * Matrix multiplication with bias addition: C = A * B + bias
 * bias is broadcasted across all rows
 */
static inline void matmul_bias(
    const float* A, size_t rows_a, size_t cols_a,
    const float* B, size_t cols_b,
    const float* bias,
    float* C
) {
    matmul(A, rows_a, cols_a, B, cols_b, C);
    
    // Add bias
    if (bias) {
        for (size_t i = 0; i < rows_a; i++) {
            for (size_t j = 0; j < cols_b; j++) {

                // unroll this
                C[i * cols_b + j] += bias[j];
            }
        }
    }
}

/**
 * Element-wise addition: C = A + B
 */
static inline void add_tensors(
    const float* A,
    const float* B,
    float* C,
    size_t length
) {
    for (size_t i = 0; i < length; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * Copy tensor data
 */
static inline void copy_tensor(
    const float* src,
    float* dst,
    size_t length
) {
    // why not use memcpy
    for (size_t i = 0; i < length; i++) {
        dst[i] = src[i];
    }
}

/**
 * Zero out tensor data
 */
static inline void zero_tensor(float* data, size_t length) {
    // why not use memset
    for (size_t i = 0; i < length; i++) {
        data[i] = 0.0f;
    }
}

/**
 * Apply GELU activation to entire tensor
 */
static inline void apply_gelu_single_core(float* data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = gelu_activation(data[i]);
    }
}

static inline void apply_gelu(float* data, size_t length) {
#if GPT_MULTICORE
    gpt_multicore_gelu(data, length);
#else
    apply_gelu_single_core(data, length);
#endif
}

/**
 * Transpose a 2D matrix in-place (only works for square matrices)
 * For non-square matrices, use transpose_matrix
 */
static inline void transpose_square_matrix(float* matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = i + 1; j < size; j++) {
            float temp = matrix[i * size + j];
            matrix[i * size + j] = matrix[j * size + i];
            matrix[j * size + i] = temp;
        }
    }
}

/**
 * Transpose a 2D matrix: output = input^T
 * input: [rows, cols]
 * output: [cols, rows]
 */
static inline void transpose_matrix(
    const float* input, size_t rows, size_t cols,
    float* output
) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/**
 * Print tensor for debugging (first few elements)
 */
static inline void print_tensor_debug(
    const char* name,
    const float* data,
    size_t length,
    size_t max_print
) {
    printf("%s: [", name);
    size_t print_len = length < max_print ? length : max_print;
    for (size_t i = 0; i < print_len; i++) {
        printf("%.4f", data[i]);
        if (i < print_len - 1) printf(", ");
    }
    if (length > max_print) {
        printf(", ... (%zu total)", length);
    }
    printf("]\n");
}

#endif // GPT_UTILS_H