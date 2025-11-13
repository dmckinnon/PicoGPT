#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "gpt_model.h"

// Memory arena size (adjust based on your needs)
#define ARENA_SIZE (1024 * 1024)  // 1MB

// Example token sequences for testing
static const int32_t example_prompt1[] = {1, 2, 3, 4, 5};
static const int32_t example_prompt2[] = {10, 15, 20, 25, 30, 35, 40};
static const int32_t example_prompt3[] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 46};

/**
 * Print a sequence of tokens
 */
void print_tokens(const char* label, const int32_t* tokens, size_t length) {
    printf("%s: [", label);
    for (size_t i = 0; i < length; i++) {
        printf("%d", tokens[i]);
        if (i < length - 1) printf(", ");
    }
    printf("]\n");
}

/**
 * Print logits with probabilities
 */
void print_logits(const float* logits, size_t vocab_size, size_t top_k) {
    printf("Top %zu token probabilities:\n", top_k);
    
    // Create pairs of (logit, index) for sorting
    typedef struct {
        float logit;
        int32_t token_id;
    } LogitPair;
    
    LogitPair pairs[VOCAB_SIZE];
    for (size_t i = 0; i < vocab_size; i++) {
        pairs[i].logit = logits[i];
        pairs[i].token_id = (int32_t)i;
    }
    
    // Simple bubble sort to find top-k (good enough for small vocab)
    for (size_t i = 0; i < top_k && i < vocab_size; i++) {
        for (size_t j = i + 1; j < vocab_size; j++) {
            if (pairs[j].logit > pairs[i].logit) {
                LogitPair temp = pairs[i];
                pairs[i] = pairs[j];
                pairs[j] = temp;
            }
        }
    }
    
    // Convert to probabilities for display
    float max_logit = pairs[0].logit;
    float total_prob = 0.0f;
    float probs[VOCAB_SIZE];
    
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_logit);
        total_prob += probs[i];
    }
    
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] /= total_prob;
    }
    
    // Print top-k
    for (size_t i = 0; i < top_k && i < vocab_size; i++) {
        int32_t token_id = pairs[i].token_id;
        printf("  Token %2d: %.4f (%.2f%%)\n", 
               token_id, pairs[i].logit, probs[token_id] * 100.0f);
    }
}

/**
 * Test basic model functionality
 */
int test_basic_inference(Arena* arena, const GPTConfig* config) {
    printf("\n=== Testing Basic Inference ===\n");
    
    // Test with different prompts
    const int32_t* prompts[] = {example_prompt1, example_prompt2, example_prompt3};
    size_t prompt_lengths[] = {5, 7, 10};
    const char* prompt_names[] = {"Short prompt", "Medium prompt", "Long prompt"};
    
    for (int test = 0; test < 3; test++) {
        printf("\n--- %s ---\n", prompt_names[test]);
        
        const int32_t* prompt = prompts[test];
        size_t prompt_len = prompt_lengths[test];
        
        print_tokens("Input", prompt, prompt_len);
        
        // Reset arena for this test
        arena_reset(arena);
        
        // Run inference
        float* logits = gpt_inference(arena, prompt, prompt_len, config);
        if (!logits) {
            printf("Error: Inference failed!\n");
            return -1;
        }
        
        printf("Inference successful!\n");
        print_logits(logits, config->vocab_size, 5);
        
        // Sample a token
        int32_t next_token = sample_token(logits, config->vocab_size, 1.0f);
        printf("Sampled next token: %d\n", next_token);
        
        printf("Arena usage: %zu bytes\n", arena_get_usage(arena));
    }
    
    return 0;
}

/**
 * Test text generation
 */
int test_text_generation(Arena* arena, const GPTConfig* config) {
    printf("\n=== Testing Text Generation ===\n");
    
    // Test generation with different parameters
    const int32_t prompt[] = {1, 2, 3};
    size_t prompt_len = 3;
    
    print_tokens("Prompt", prompt, prompt_len);
    
    // Generate with different temperatures
    float temperatures[] = {0.1f, 0.7f, 1.5f};
    const char* temp_names[] = {"Low (0.1)", "Medium (0.7)", "High (1.5)"};
    
    for (int test = 0; test < 3; test++) {
        printf("\n--- Temperature: %s ---\n", temp_names[test]);
        
        // Reset arena
        arena_reset(arena);
        
        int32_t output_tokens[MAX_SEQ_LEN];
        size_t total_len = generate_text(
            arena, prompt, prompt_len, 10, temperatures[test], 
            output_tokens, config
        );
        
        print_tokens("Generated", output_tokens, total_len);
        printf("Generated %zu new tokens\n", total_len - prompt_len);
        printf("Arena peak usage: %zu bytes\n", arena_get_usage(arena));
    }
    
    return 0;
}

/**
 * Test individual components
 */
int test_components(Arena* arena, const GPTConfig* config) {
    printf("\n=== Testing Individual Components ===\n");
    
    const int32_t tokens[] = {1, 5, 10, 15, 20};
    size_t seq_len = 5;
    
    arena_reset(arena);
    
    // Test embeddings
    printf("\n--- Testing Embeddings ---\n");
    float* embeddings = embedding_op(arena, tokens, seq_len);
    if (!embeddings) {
        printf("Error: Embedding test failed!\n");
        return -1;
    }
    print_embedding_debug("Test Embeddings", embeddings, seq_len, 3, 8);
    
    // Test layer normalization
    printf("\n--- Testing Layer Normalization ---\n");
    float* normalized = layer_norm_alloc(
        arena, embeddings, layer1_ln_gamma, layer1_ln_beta, 
        seq_len, config->embed_dim
    );
    if (!normalized) {
        printf("Error: Layer norm test failed!\n");
        return -1;
    }
    print_layer_norm_debug("Test LayerNorm", embeddings, normalized, seq_len, config->embed_dim);
    
    // Test attention
    printf("\n--- Testing Attention ---\n");
    float* attention_out = multi_head_attention(
        arena, normalized, (const float*)layer1_attn_qkv, 
        (const float*)layer1_attn_out, seq_len, config->embed_dim
    );
    if (!attention_out) {
        printf("Error: Attention test failed!\n");
        return -1;
    }
    print_tensor_debug("Attention Output", attention_out, seq_len * config->embed_dim, 8);
    
    // Test FFN
    printf("\n--- Testing FFN ---\n");
    float* ffn_out = ffn_op(
        arena, attention_out, (const float*)layer1_ffn_up, 
        (const float*)layer1_ffn_down, seq_len, config->embed_dim, config->ffn_dim
    );
    if (!ffn_out) {
        printf("Error: FFN test failed!\n");
        return -1;
    }
    print_tensor_debug("FFN Output", ffn_out, seq_len * config->embed_dim, 8);
    
    printf("All component tests passed!\n");
    printf("Total arena usage: %zu bytes\n", arena_get_usage(arena));
    
    return 0;
}

/**
 * Memory usage analysis
 */
void analyze_memory_usage(const GPTConfig* config) {
    printf("\n=== Memory Usage Analysis ===\n");
    
    size_t seq_lengths[] = {1, 5, 10, 20, 31};
    
    for (int i = 0; i < 5; i++) {
        size_t seq_len = seq_lengths[i];
        size_t estimated = estimate_memory_usage(config, seq_len);
        
        printf("Sequence length %2zu: ~%6zu bytes (%.1f KB)\n", 
               seq_len, estimated, estimated / 1024.0f);
    }
    
    printf("\nArena size: %d bytes (%.1f KB)\n", 
           ARENA_SIZE, ARENA_SIZE / 1024.0f);
}

/**
 * Performance benchmarking
 */
void benchmark_performance(Arena* arena, const GPTConfig* config) {
    printf("\n=== Performance Benchmark ===\n");
    
    const int32_t test_tokens[] = {1, 5, 10, 15, 20, 25, 30};
    size_t test_len = 7;
    
    int num_runs = 10;
    clock_t total_time = 0;
    
    printf("Running %d inference passes...\n", num_runs);
    
    for (int run = 0; run < num_runs; run++) {
        arena_reset(arena);
        
        clock_t start = clock();
        float* logits = gpt_inference(arena, test_tokens, test_len, config);
        clock_t end = clock();
        
        if (!logits) {
            printf("Error: Benchmark run %d failed!\n", run);
            return;
        }
        
        total_time += (end - start);
    }
    
    double avg_time_ms = ((double)total_time / CLOCKS_PER_SEC) * 1000.0 / num_runs;
    
    printf("Average inference time: %.2f ms\n", avg_time_ms);
    printf("Tokens per second: %.0f\n", test_len / (avg_time_ms / 1000.0));
    printf("Peak memory usage: %zu bytes\n", arena_get_usage(arena));
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    printf("GPT Model Test - Extracted from TensorFlow Lite\n");
    printf("============================================\n");
    
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    // Initialize model configuration
    GPTConfig config = gpt_config_init();
    print_model_info(&config);
    
    // Allocate memory arena
    uint8_t* memory_buffer = (uint8_t*)malloc(ARENA_SIZE);
    if (!memory_buffer) {
        printf("Error: Failed to allocate memory arena!\n");
        return -1;
    }
    
    Arena arena;
    arena_init(&arena, memory_buffer, ARENA_SIZE, 8);
    
    printf("Arena initialized: %d bytes available\n", ARENA_SIZE);
    
    // Run tests
    analyze_memory_usage(&config);
    
    if (test_components(&arena, &config) != 0) {
        printf("Component tests failed!\n");
        goto cleanup;
    }
    
    if (test_basic_inference(&arena, &config) != 0) {
        printf("Basic inference tests failed!\n");
        goto cleanup;
    }
    
    if (test_text_generation(&arena, &config) != 0) {
        printf("Text generation tests failed!\n");
        goto cleanup;
    }
    
    benchmark_performance(&arena, &config);
    
    printf("\n=== All Tests Passed! ===\n");
    printf("The GPT model implementation is working correctly.\n");
    printf("You can now use this code for inference or further development.\n");

cleanup:
    free(memory_buffer);
    return 0;
}