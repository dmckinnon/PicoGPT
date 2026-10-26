#include "gpt_vocab.h"

/* Extracted from nanoGPT/data/shakespeare_char/meta.pkl, in token-ID order. */
const char gpt_vocab[] = {
    '\n', ' ', ',', '-', '.', '0', '1', '2', '3', '4',
    '6', '?', 'A', 'D', 'H', 'L', 'M', 'N', 'Q', 'S',
    'T', 'U', 'W', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z',
};

const size_t gpt_vocab_size = sizeof(gpt_vocab) / sizeof(gpt_vocab[0]);
