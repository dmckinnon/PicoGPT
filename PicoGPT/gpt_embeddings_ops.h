#ifndef GPT_EMBEDDINGS_H
#define GPT_EMBEDDINGS_H

#include "gpt_weights/gpt_embeddings.h"
#include "gpt_weights/gpt_constants.h"
#include "arena_allocator.h"
#include "gpt_utils.h"

/**
 * Token embedding lookup
 * Looks up embeddings for a sequence of token IDs and copies to output
 * 
 * @param token_ids: Array of token IDs [seq_len]
 * @param output: Output embeddings [seq_len, embed_dim]
 * @param seq_len: Sequence length
 */
static inline void token_embedding_lookup(
    const int32_t* token_ids,
    float* output,
    size_t seq_len
) {
    for (size_t i = 0; i < seq_len; i++) {
        int32_t token_id = token_ids[i];
        
        // Bounds check
        if (token_id < 0 || token_id >= VOCAB_SIZE) {
            // Invalid token ID - zero out the embedding
            for (size_t j = 0; j < EMBED_DIM; j++) {
                output[i * EMBED_DIM + j] = 0.0f;
            }
            continue;
        }
        
        // Copy the embedding
        for (size_t j = 0; j < EMBED_DIM; j++) {
            output[i * EMBED_DIM + j] = token_embedding[token_id][j];
        }
    }
}

/**
 * Position embedding lookup
 * Adds positional embeddings to token embeddings
 * 
 * @param embeddings: Input/output embeddings [seq_len, embed_dim]
 * @param seq_len: Sequence length (must be <= MAX_SEQ_LEN)
 */
static inline void add_position_embeddings(
    float* embeddings,
    size_t seq_len
) {
    // Ensure we don't exceed maximum sequence length
    if (seq_len > MAX_SEQ_LEN) {
        seq_len = MAX_SEQ_LEN;
    }
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < EMBED_DIM; j++) {
            embeddings[i * EMBED_DIM + j] += position_embedding[i][j];
        }
    }
}

/**
 * Combined token and position embedding lookup
 * Performs both token lookup and position addition in one step
 * 
 * @param token_ids: Array of token IDs [seq_len]
 * @param output: Output embeddings [seq_len, embed_dim]
 * @param seq_len: Sequence length
 */
static inline void embedding_lookup(
    const int32_t* token_ids,
    float* output,
    size_t seq_len
) {
    // First, get token embeddings
    token_embedding_lookup(token_ids, output, seq_len);
    
    // Then add position embeddings
    add_position_embeddings(output, seq_len);
}

/**
 * Embedding operation with arena allocation
 * Allocates output tensor and performs embedding lookup
 * 
 * @param arena: Memory arena for allocation
 * @param token_ids: Array of token IDs [seq_len]
 * @param seq_len: Sequence length
 * @return: Pointer to allocated embedding tensor [seq_len, embed_dim]
 */
static inline float* embedding_op(
    Arena* arena,
    const int32_t* token_ids,
    size_t seq_len
) {
    // Allocate output tensor
    float* embeddings = arena_alloc_tensor_2d(arena, seq_len, EMBED_DIM);
    if (!embeddings) {
        return NULL; // Out of memory
    }
    
    // Perform embedding lookup
    // this feels like an unnecessary extra layer
    embedding_lookup(token_ids, embeddings, seq_len);
    
    return embeddings;
}

/**
 * Debug function to print embedding information
 */
static inline void print_embedding_debug(
    const char* name,
    const float* embeddings,
    size_t seq_len,
    size_t max_tokens_print,
    size_t max_dims_print
) {
    printf("\n=== %s ===\n", name);
    
    size_t tokens_to_print = seq_len < max_tokens_print ? seq_len : max_tokens_print;
    size_t dims_to_print = EMBED_DIM < max_dims_print ? EMBED_DIM : max_dims_print;
    
    for (size_t i = 0; i < tokens_to_print; i++) {
        printf("Token %zu: [", i);
        for (size_t j = 0; j < dims_to_print; j++) {
            printf("%.4f", embeddings[i * EMBED_DIM + j]);
            if (j < dims_to_print - 1) printf(", ");
        }
        if (EMBED_DIM > max_dims_print) {
            printf(", ... (%d total)", EMBED_DIM);
        }
        printf("]\n");
    }
    
    if (seq_len > max_tokens_print) {
        printf("... (%zu total tokens)\n", seq_len);
    }
}

#endif // GPT_EMBEDDINGS_H