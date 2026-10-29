#ifndef ARENA_ALLOCATOR_H
#define ARENA_ALLOCATOR_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

/**
 * Simple arena allocator for tensor memory management
 * All allocations are done from a single contiguous buffer
 * No individual deallocations - reset the entire arena to free all memory
 */

typedef struct {
    uint8_t* buffer;        // Pointer to the memory buffer
    size_t size;           // Total size of the buffer
    size_t offset;         // Current offset in the buffer
    size_t alignment;      // Memory alignment (typically 4 or 8 bytes)
} Arena;

/**
 * Initialize an arena with a pre-allocated buffer
 */
static inline void arena_init(Arena* arena, void* buffer, size_t size, size_t alignment) {
    arena->buffer = (uint8_t*)buffer;
    arena->size = size;
    arena->offset = 0;
    arena->alignment = alignment;
}

/**
 * Align a size to the specified alignment
 */
static inline size_t arena_align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * Allocate aligned memory from the arena
 */
static inline void* arena_alloc(Arena* arena, size_t size) {
    // Align the current offset
    size_t aligned_offset = arena_align_size(arena->offset, arena->alignment);
    
    // Align the requested size
    size_t aligned_size = arena_align_size(size, arena->alignment);
    
    // Check if we have enough space
    if (aligned_offset + aligned_size > arena->size) {
        return NULL; // Out of memory
    }
    
    void* ptr = arena->buffer + aligned_offset;
    arena->offset = aligned_offset + aligned_size;
    
    // Zero the allocated memory
    // How necessary is this???
    // memset(ptr, 0, size);
    
    return ptr;
}

/**
 * Allocate memory for a tensor with specific dimensions
 */
static inline float* arena_alloc_tensor(Arena* arena, size_t dim1, size_t dim2, size_t dim3) {
    size_t total_elements = dim1 * dim2 * dim3;
    size_t total_size = total_elements * sizeof(float);
    return (float*)arena_alloc(arena, total_size);
}

/**
 * Allocate memory for a 1D tensor
 */
static inline float* arena_alloc_tensor_1d(Arena* arena, size_t dim1) {
    return arena_alloc_tensor(arena, dim1, 1, 1);
}

/**
 * Allocate memory for a 2D tensor
 */
static inline float* arena_alloc_tensor_2d(Arena* arena, size_t dim1, size_t dim2) {
    return arena_alloc_tensor(arena, dim1, dim2, 1);
}

/**
 * Allocate memory for a 3D tensor
 */
static inline float* arena_alloc_tensor_3d(Arena* arena, size_t dim1, size_t dim2, size_t dim3) {
    return arena_alloc_tensor(arena, dim1, dim2, dim3);
}

/**
 * Reset the arena (free all allocations)
 */
static inline void arena_reset(Arena* arena) {
    arena->offset = 0;
}

/**
 * Get the current memory usage
 */
static inline size_t arena_get_usage(const Arena* arena) {
    return arena->offset;
}

/**
 * Get the remaining free space
 */
static inline size_t arena_get_free_space(const Arena* arena) {
    return arena->size - arena->offset;
}

/**
 * Create a checkpoint for nested allocations
 */
static inline size_t arena_checkpoint(const Arena* arena) {
    return arena->offset;
}

/**
 * Restore to a checkpoint (free allocations made after the checkpoint)
 */
static inline void arena_restore(Arena* arena, size_t checkpoint) {
    assert(checkpoint <= arena->offset);
    arena->offset = checkpoint;
}

#endif // ARENA_ALLOCATOR_H