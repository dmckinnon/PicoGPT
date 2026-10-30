#ifndef GPT_MODEL_H
#define GPT_MODEL_H

#include "arena_allocator.h"
#include "gpt_weights/gpt_weights.h"
#include "gpt_embeddings_ops.h"
#include "gpt_layer_norm.h"
#include "gpt_attention.h"
#include "gpt_ffn.h"
#include "gpt_utils.h"
#include "gpt_profile.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * GPT Model Configuration
 */
typedef struct {
    size_t vocab_size;      // 47
    size_t max_seq_len;     // 31
    size_t embed_dim;       // 32
    size_t ffn_dim;         // 128
    size_t num_layers;      // 2
    size_t num_heads;       // 1
    float layer_norm_eps;   // 1e-5
} GPTConfig;

/**
 * Initialize GPT configuration with model parameters
 */
static inline GPTConfig gpt_config_init(void) {
    GPTConfig config = {
        .vocab_size = VOCAB_SIZE,
        .max_seq_len = MAX_SEQ_LEN,
        .embed_dim = EMBED_DIM,
        .ffn_dim = FFN_DIM,
        .num_layers = NUM_LAYERS,
        .num_heads = NUM_HEADS,
        .layer_norm_eps = LAYER_NORM_EPS
    };
    return config;
}

/**
 * Single transformer layer computation
 * Applies attention + FFN with residual connections
 * 
 * @param arena: Memory arena for allocations
 * @param input: Input tensor [seq_len, embed_dim]
 * @param layer_idx: Layer index (0 or 1)
 * @param config: Model configuration
 * @param seq_len: Current sequence length
 * @return: Pointer to layer output [seq_len, embed_dim]
 */
static inline float* transformer_layer(
    Arena* arena,
    const float* input,
    size_t layer_idx,
    const GPTConfig* config,
    size_t seq_len
) {
    // Get layer-specific weights
    const float* ln_gamma_1, *ln_beta_1, *ln_gamma_2, *ln_beta_2;
    const float* attn_qkv, *attn_out, *ffn_up, *ffn_down;
    
    if (layer_idx == 0) {
        // Layer 1 weights
        ln_gamma_1 = layer1_ln_gamma;
        ln_beta_1 = layer1_ln_beta;
        ln_gamma_2 = layer1_ffn_ln_gamma;
        ln_beta_2 = layer1_ffn_ln_beta;
        attn_qkv = (const float*)layer1_attn_qkv;
        attn_out = (const float*)layer1_attn_out;
        ffn_up = (const float*)layer1_ffn_up;
        ffn_down = (const float*)layer1_ffn_down;
    } else {
        // Layer 2 weights
        ln_gamma_1 = layer2_ln_gamma;
        ln_beta_1 = layer2_ln_beta;
        ln_gamma_2 = layer2_ffn_ln_gamma;
        ln_beta_2 = layer2_ffn_ln_beta;
        attn_qkv = (const float*)layer2_attn_qkv;
        attn_out = (const float*)layer2_attn_out;
        ffn_up = (const float*)layer2_ffn_up;
        ffn_down = (const float*)layer2_ffn_down;
    }
    
    // Self-attention with residual connection
    float* attn_output = attention_with_residual(
        arena, input, attn_qkv, attn_out, ln_gamma_1, ln_beta_1,
        seq_len, config->embed_dim
    );
    
    if (!attn_output) {
        return NULL; // Out of memory
    }
    
    // FFN with residual connection
    float* layer_output = ffn_with_residual(
        arena, attn_output, ffn_up, ffn_down, ln_gamma_2, ln_beta_2,
        seq_len, config->embed_dim, config->ffn_dim
    );
    
    return layer_output;
}

/**
 * Embeddings and transformer blocks, before final layer normalization
 * 
 * @param arena: Memory arena for allocations
 * @param token_ids: Input token sequence [seq_len]
 * @param seq_len: Sequence length
 * @param config: Model configuration
 * @return: Pointer to unnormalized hidden states [seq_len, embed_dim]
 */
static inline float* gpt_transformer_forward(
    Arena* arena,
    const int32_t* token_ids,
    size_t seq_len,
    const GPTConfig* config
) {
    // Bounds check
    if (seq_len == 0) {
        printf("Error: Input sequence must contain at least one token\n");
        return NULL;
    }
    if (seq_len > config->max_seq_len) {
        printf("Error: Sequence length %zu exceeds maximum %zu\n", 
               seq_len, config->max_seq_len);
        return NULL;
    }
    
    // 1. Embedding lookup (token + position)
    GPT_PROFILE_BEGIN(GPT_STAGE_EMBEDDINGS);
    float* embeddings = embedding_op(arena, token_ids, seq_len);
    GPT_PROFILE_END(GPT_STAGE_EMBEDDINGS);
    if (!embeddings) {
        printf("Error: Failed to allocate embeddings\n");
        return NULL;
    }
    
    float* current_output = embeddings;
    
    // 2. Apply transformer layers
    for (size_t layer = 0; layer < config->num_layers; layer++) {
        current_output = transformer_layer(arena, current_output, layer, config, seq_len);
        if (!current_output) {
            printf("Error: Failed in transformer layer %zu\n", layer);
            return NULL;
        }
    }

    return current_output;
}

/**
 * Complete forward pass, preserving normalized outputs for every input position.
 * Returns [seq_len, embed_dim] in the arena.
 */
static inline float* gpt_forward(
    Arena* arena,
    const int32_t* token_ids,
    size_t seq_len,
    const GPTConfig* config
) {
    float* hidden_states = gpt_transformer_forward(arena, token_ids, seq_len, config);
    if (!hidden_states) {
        return NULL;
    }

    GPT_PROFILE_BEGIN(GPT_STAGE_FINAL_NORM);
    float* final_output = layer_norm_alloc(
        arena, hidden_states, final_ln_gamma, final_ln_beta,
        seq_len, config->embed_dim
    );
    GPT_PROFILE_END(GPT_STAGE_FINAL_NORM);
    
    // this output is in embedding space
    return final_output;
}

/**
 * Generate logits for next token prediction
 * Applies the language modeling head (embedding weights transposed)
 * 
 * @param arena: Memory arena for allocations
 * @param hidden_states: Final layer output [seq_len, embed_dim]
 * @param seq_len: Sequence length
 * @param config: Model configuration
 * @return: Pointer to logits [seq_len, vocab_size]
 */
static inline float* generate_logits(
    Arena* arena,
    const float* hidden_states,
    size_t seq_len,
    const GPTConfig* config
) {
    // Allocate logits tensor
    float* logits = arena_alloc_tensor_2d(arena, seq_len, config->vocab_size);
    if (!logits) {
        return NULL;
    }
    
    // Apply language modeling head: logits = hidden_states * embedding_weights^T
    // hidden_states: [seq_len, embed_dim]
    // token_embedding: [vocab_size, embed_dim] -> transposed: [embed_dim, vocab_size]
    // result: [seq_len, vocab_size]
    
    for (size_t seq = 0; seq < seq_len; seq++) {
        for (size_t vocab = 0; vocab < config->vocab_size; vocab++) {
            float sum = 0.0f;
            for (size_t dim = 0; dim < config->embed_dim; dim++) {
                sum += hidden_states[seq * config->embed_dim + dim] * 
                       token_embedding[vocab][dim];
            }
            logits[seq * config->vocab_size + vocab] = sum;
        }
    }
    
    return logits;
}

/**
 * Complete GPT inference for next token prediction
 * Returns logits for the next token given input sequence
 * 
 * @param arena: Memory arena for allocations
 * @param token_ids: Input token sequence [seq_len]
 * @param seq_len: Sequence length
 * @param config: Model configuration
 * @return: Arena-owned logits [vocab_size], valid until arena reset/restore
 */
static inline float* gpt_inference(
    Arena* arena,
    const int32_t* token_ids,
    size_t seq_len,
    const GPTConfig* config
) {
    float* hidden_states = gpt_transformer_forward(arena, token_ids, seq_len, config);
    if (!hidden_states) {
        return NULL;
    }
    
    /* Final normalization and the vocabulary head are independent for each row. */
    const float* last_hidden = hidden_states + (seq_len - 1) * config->embed_dim;
    GPT_PROFILE_BEGIN(GPT_STAGE_FINAL_NORM);
    float* normalized = layer_norm_alloc(
        arena, last_hidden, final_ln_gamma, final_ln_beta, 1, config->embed_dim
    );
    GPT_PROFILE_END(GPT_STAGE_FINAL_NORM);
    if (!normalized) {
        printf("Error: Failed to allocate final token normalization\n");
        return NULL;
    }

    GPT_PROFILE_BEGIN(GPT_STAGE_LOGITS);
    float* logits = generate_logits(arena, normalized, 1, config);
    GPT_PROFILE_END(GPT_STAGE_LOGITS);
    if (!logits) {
        printf("Error: Failed to allocate next-token logits\n");
        return NULL;
    }
    return logits;
}

/**
 * Sample next token from logits using temperature sampling
 * 
 * @param logits: Logits for all vocabulary tokens [vocab_size]
 * @param vocab_size: Vocabulary size
 * @param temperature: Sampling temperature (1.0 = no change, <1.0 = more deterministic)
 * @return: Sampled token ID
 */
static inline int32_t sample_token(
    const float* logits,
    size_t vocab_size,
    float temperature
) {
    // Apply temperature scaling
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute probabilities with temperature
    float total_prob = 0.0f;
    float probs[VOCAB_SIZE]; // Using static array since vocab size is small
    
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        total_prob += probs[i];
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] /= total_prob;
    }
    
    // Sample using cumulative distribution
    float random_val = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (random_val <= cumulative) {
            return (int32_t)i;
        }
    }
    
    // Fallback to last token
    return (int32_t)(vocab_size - 1);
}

/**
 * Generate text autoregressively
 * 
 * @param arena: Memory arena for allocations
 * @param prompt_tokens: Initial prompt tokens
 * @param prompt_len: Length of prompt
 * @param max_new_tokens: Maximum number of new tokens to generate
 * @param temperature: Sampling temperature
 * @param output_tokens: Output buffer for generated tokens
 * @param config: Model configuration
 * @return: Total number of tokens in output (prompt + generated)
 */
static inline size_t generate_text(
    Arena* arena,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    size_t max_new_tokens,
    float temperature,
    int32_t* output_tokens,
    const GPTConfig* config
) {
    // Copy prompt to output buffer
    for (size_t i = 0; i < prompt_len; i++) {
        output_tokens[i] = prompt_tokens[i];
    }
    
    size_t current_len = prompt_len;
    
    // Generate tokens autoregressively
    for (size_t gen_step = 0; gen_step < max_new_tokens; gen_step++) {
        // Check if we've reached maximum sequence length
        if (current_len >= config->max_seq_len) {
            break;
        }
        
        // Reset arena for this generation step
        size_t checkpoint = arena_checkpoint(arena);
        
        // Get logits for next token
        float* logits = gpt_inference(arena, output_tokens, current_len, config);
        if (!logits) {
            printf("Error: Failed to generate logits at step %zu\n", gen_step);
            break;
        }
        
        // Sample next token
        int32_t next_token = sample_token(logits, config->vocab_size, temperature);
        
        // Add to sequence
        output_tokens[current_len] = next_token;
        current_len++;
        
        // Restore arena checkpoint (free temporary allocations)
        arena_restore(arena, checkpoint);
        
        // Optional: Check for end-of-sequence token
        // if (next_token == EOS_TOKEN_ID) break;
    }
    
    return current_len;
}

/**
 * Debug function to print model information
 */
static inline void print_model_info(const GPTConfig* config) {
    printf("\n=== GPT Model Configuration ===\n");
    printf("Vocabulary Size: %zu\n", config->vocab_size);
    printf("Max Sequence Length: %zu\n", config->max_seq_len);
    printf("Embedding Dimension: %zu\n", config->embed_dim);
    printf("FFN Hidden Dimension: %zu\n", config->ffn_dim);
    printf("Number of Layers: %zu\n", config->num_layers);
    printf("Number of Attention Heads: %zu\n", config->num_heads);
    printf("Layer Norm Epsilon: %.2e\n", config->layer_norm_eps);
    printf("============================\n\n");
}

/**
 * Estimate memory requirements for a forward pass
 */
static inline size_t estimate_memory_usage(const GPTConfig* config, size_t seq_len) {
    size_t memory = 0;
    
    // Embeddings
    memory += seq_len * config->embed_dim * sizeof(float);
    
    // Per layer (worst case - all intermediate tensors)
    for (size_t layer = 0; layer < config->num_layers; layer++) {
        // Attention
        memory += seq_len * config->embed_dim * sizeof(float) * 4; // Q, K, V, output
        memory += seq_len * seq_len * sizeof(float); // attention scores
        
        // FFN
        memory += seq_len * config->ffn_dim * sizeof(float); // hidden layer
        memory += seq_len * config->embed_dim * sizeof(float); // output
        
        // Layer norm and residual
        memory += seq_len * config->embed_dim * sizeof(float) * 2;
    }
    
    // Final layer norm and logits
    memory += seq_len * config->embed_dim * sizeof(float);
    memory += seq_len * config->vocab_size * sizeof(float);
    
    return memory;
}

#endif // GPT_MODEL_H