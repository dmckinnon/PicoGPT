#ifndef GPT_VOCAB_H
#define GPT_VOCAB_H

#include <stddef.h>

/* Token-ID to character table, not a NUL-terminated string. */
extern const char gpt_vocab[];
extern const size_t gpt_vocab_size;

#endif // GPT_VOCAB_H
