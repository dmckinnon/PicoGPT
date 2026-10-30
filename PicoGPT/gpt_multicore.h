#ifndef GPT_MULTICORE_H
#define GPT_MULTICORE_H

#include <stddef.h>

#ifndef GPT_MULTICORE
#define GPT_MULTICORE 0
#endif

#if GPT_MULTICORE
/* Core 0 foreground only; init owns core 1. Calls complete before returning. */
void gpt_multicore_init(void);
void gpt_multicore_matmul(
    const float* a, size_t rows, size_t inner,
    const float* b, size_t columns, const float* residual, float* output
);
void gpt_multicore_gelu(float* data, size_t length);
#else
static inline void gpt_multicore_init(void) {}
#endif

#endif // GPT_MULTICORE_H
