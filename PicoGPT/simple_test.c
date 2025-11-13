#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Include all headers
#include "arena_allocator.h"
#include "gpt_weights/gpt_constants.h"
#include "gpt_weights/gpt_embeddings.h"
#include "gpt_weights/gpt_layer1.h"
#include "gpt_weights/gpt_layer2.h"
#include "gpt_utils.h"

// Memory arena size
#define ARENA_SIZE (1024 * 1024)  // 1MB

/**
 * Simple embedding lookup function
 */
float* simple_embedding_lookup(Arena* arena, const int32_t* token_ids, size_t seq_len) {
    // Allocate output tensor
    float* embeddings = arena_alloc_tensor_2d(arena, seq_len, EMBED_DIM);
    if (!embeddings) {
        return NULL;
    }
    
    // Token embedding lookup
    for (size_t i = 0; i < seq_len; i++) {
        int32_t token_id = token_ids[i];
        
        // Bounds check
        if (token_id < 0 || token_id >= VOCAB_SIZE) {
            // Zero out invalid tokens
            for (size_t j = 0; j < EMBED_DIM; j++) {
                embeddings[i * EMBED_DIM + j] = 0.0f;
            }
            continue;
        }
        
        // Copy token embedding and add position embedding
        for (size_t j = 0; j < EMBED_DIM; j++) {
            embeddings[i * EMBED_DIM + j] = token_embedding[token_id][j];
            if (i < MAX_SEQ_LEN) {
                embeddings[i * EMBED_DIM + j] += position_embedding[i][j];
            }
        }
    }
    
    return embeddings;
}

/**
 * Simple matrix multiplication
 */
void simple_matmul(const float* A, const float* B, float* C, 
                   size_t rows_a, size_t cols_a, size_t cols_b) {
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

/**
 * Simple layer norm
 */
void simple_layer_norm(const float* input, float* output, const float* gamma, 
                      size_t seq_len, size_t embed_dim) {
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
        float inv_std = 1.0f / sqrtf(variance + LAYER_NORM_EPS);
        for (size_t i = 0; i < embed_dim; i++) {
            output_row[i] = gamma[i] * (input_row[i] - mean) * inv_std;
        }
    }
}

/**
 * Test basic functionality
 */
int test_basic_operations(Arena* arena) {
    printf("=== Testing Basic Operations ===\n");
    
    // Test embedding lookup
    const int32_t test_tokens[] = {1, 5, 10, 15, 20};
    size_t seq_len = 5;
    
    printf("Testing embedding lookup...\n");
    float* embeddings = simple_embedding_lookup(arena, test_tokens, seq_len);
    if (!embeddings) {
        printf("Error: Embedding lookup failed!\n");
        return -1;
    }
    
    // Print first few embedding values
    printf("First token embedding (first 8 dims): ");
    for (size_t i = 0; i < 8; i++) {
        printf("%.4f ", embeddings[i]);
    }
    printf("\n");
    
    // Test layer normalization
    printf("Testing layer normalization...\n");
    float* normalized = arena_alloc_tensor_2d(arena, seq_len, EMBED_DIM);
    if (!normalized) {
        printf("Error: Failed to allocate normalized tensor!\n");
        return -1;
    }
    
    simple_layer_norm(embeddings, normalized, layer1_ln_gamma, seq_len, EMBED_DIM);
    
    printf("After layer norm (first 8 dims): ");
    for (size_t i = 0; i < 8; i++) {
        printf("%.4f ", normalized[i]);
    }
    printf("\n");
    
    // Test matrix multiplication (simple projection)
    printf("Testing matrix multiplication...\n");
    float* projected = arena_alloc_tensor_2d(arena, seq_len, FFN_DIM);
    if (!projected) {
        printf("Error: Failed to allocate projected tensor!\n");
        return -1;
    }
    
    simple_matmul(normalized, (const float*)layer1_ffn_up, projected, 
                  seq_len, EMBED_DIM, FFN_DIM);
    
    printf("After projection (first 8 dims): ");
    for (size_t i = 0; i < 8; i++) {
        printf("%.4f ", projected[i]);
    }
    printf("\n");
    
    // Apply GELU activation
    printf("Testing GELU activation...\n");
    for (size_t i = 0; i < seq_len * FFN_DIM; i++) {
        projected[i] = gelu_activation(projected[i]);
    }
    
    printf("After GELU (first 8 dims): ");
    for (size_t i = 0; i < 8; i++) {
        printf("%.4f ", projected[i]);
    }
    printf("\n");
    
    printf("Arena usage: %zu bytes\n", arena_get_usage(arena));
    printf("All basic operations completed successfully!\n\n");
    
    return 0;
}

/**
 * Test weight access
 */
int test_weight_access(void) {
    printf("=== Testing Weight Access ===\n");
    
    printf("Model dimensions:\n");
    printf("  Vocabulary size: %d\n", VOCAB_SIZE);
    printf("  Max sequence length: %d\n", MAX_SEQ_LEN);
    printf("  Embedding dimension: %d\n", EMBED_DIM);
    printf("  FFN dimension: %d\n", FFN_DIM);
    
    printf("\nWeight matrix shapes:\n");
    printf("  Token embedding: [%d, %d]\n", VOCAB_SIZE, EMBED_DIM);
    printf("  Position embedding: [%d, %d]\n", MAX_SEQ_LEN, EMBED_DIM);
    printf("  Layer 1 attention QKV: [96, %d]\n", EMBED_DIM);
    printf("  Layer 1 attention output: [%d, %d]\n", EMBED_DIM, EMBED_DIM);
    printf("  Layer 1 FFN up: [%d, %d]\n", EMBED_DIM, FFN_DIM);
    printf("  Layer 1 FFN down: [%d, %d]\n", FFN_DIM, EMBED_DIM);
    
    // Test some weight values
    printf("\nSample weight values:\n");
    printf("  Token embedding[0][0]: %.6f\n", token_embedding[0][0]);
    printf("  Position embedding[0][0]: %.6f\n", position_embedding[0][0]);
    printf("  Layer 1 LN gamma[0]: %.6f\n", layer1_ln_gamma[0]);
    printf("  Layer 1 attention QKV[0][0]: %.6f\n", layer1_attn_qkv[0][0]);
    printf("  Layer 1 FFN up[0][0]: %.6f\n", layer1_ffn_up[0][0]);
    
    printf("Weight access test completed successfully!\n\n");
    
    return 0;
}

/**
 * Memory usage analysis
 */
void analyze_memory_usage(void) {
    printf("=== Memory Usage Analysis ===\n");
    
    size_t seq_lengths[] = {1, 5, 10, 20, 31};
    
    for (int i = 0; i < 5; i++) {
        size_t seq_len = seq_lengths[i];
        
        // Estimate memory for embeddings
        size_t embedding_mem = seq_len * EMBED_DIM * sizeof(float);
        
        // Estimate memory for one layer (attention + FFN)
        size_t layer_mem = 0;
        layer_mem += seq_len * EMBED_DIM * sizeof(float) * 4; // Q, K, V, output
        layer_mem += seq_len * seq_len * sizeof(float);       // attention scores
        layer_mem += seq_len * FFN_DIM * sizeof(float);       // FFN hidden
        layer_mem += seq_len * EMBED_DIM * sizeof(float) * 2; // layer norm + residual
        
        size_t total_estimated = embedding_mem + layer_mem * NUM_LAYERS;
        
        printf("Sequence length %2zu: ~%6zu bytes (%.1f KB)\n", 
               seq_len, total_estimated, total_estimated / 1024.0f);
    }
    
    printf("\nArena size: %d bytes (%.1f KB)\n", 
           ARENA_SIZE, ARENA_SIZE / 1024.0f);
    printf("Memory analysis completed!\n\n");
}

int main(void) {
    printf("GPT Model Test - Simplified Version\n");
    printf("===================================\n\n");
    
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    // Allocate memory arena
    uint8_t* memory_buffer = (uint8_t*)malloc(ARENA_SIZE);
    if (!memory_buffer) {
        printf("Error: Failed to allocate memory arena!\n");
        return -1;
    }
    
    Arena arena;
    arena_init(&arena, memory_buffer, ARENA_SIZE, 8);
    
    printf("Arena initialized: %d bytes available\n\n", ARENA_SIZE);
    
    // Run tests
    if (test_weight_access() != 0) {
        printf("Weight access test failed!\n");
        goto cleanup;
    }
    
    analyze_memory_usage();
    
    if (test_basic_operations(&arena) != 0) {
        printf("Basic operations test failed!\n");
        goto cleanup;
    }
    
    printf("=== All Tests Passed! ===\n");
    printf("The weights have been successfully extracted and are accessible in C.\n");
    printf("You can now build upon this foundation to implement the complete model.\n");

cleanup:
    free(memory_buffer);
    return 0;
}