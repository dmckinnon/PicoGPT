#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/clocks.h"

#include "gpt_model.h"
#include "gpt_vocab.h"
#include "pico_memory.h"

#define ARENA_SIZE (144 * 1024)
#define NEW_TOKENS 100

static _Alignas(8) uint8_t arena_memory[ARENA_SIZE];

/* TFLite logits for IDs 0..30, with its final GATHER index corrected to 30. */
static const float reference_logits[VOCAB_SIZE] = {
    -3.870406151e+00f, 3.310939074e+00f, -2.111715078e+00f, -5.154929161e+00f, -3.595356703e+00f, -3.859104872e+00f,
    -1.469592214e+00f, -2.233737469e+00f, -5.618237019e+00f, 8.414369226e-01f, -1.083295465e+00f, -3.618210316e+00f,
    1.081460357e+00f, -5.246846199e+00f, -7.265247703e-01f, -4.197725296e+00f, -2.859339714e+00f, -7.772410870e+00f,
    -2.249195814e+00f, -1.098150134e+00f, -3.203865051e+00f, -5.350651264e+00f, -1.628239393e+00f, 7.354065418e+00f,
    -1.285065651e+00f, 5.030747056e-01f, -1.986923814e+00f, 1.081216049e+01f, -2.901870728e+00f, -7.732359767e-01f,
    -8.390982747e-01f, -2.497374058e+00f, -1.146560311e+00f, 3.155444622e+00f, -1.453517914e+00f, 4.958459854e+00f,
    -8.419454694e-01f, -3.256937265e+00f, 2.225280106e-01f, 2.220644236e+00f, -3.710196316e-01f, -1.906719446e+00f,
    -2.463572264e+00f, 2.330612898e+00f, -1.479536414e+00f, 2.650636435e-01f, -4.853814125e+00f,
};

static float* infer(Arena* arena, const GPTConfig* config, const int32_t* tokens,
                    size_t length, size_t* peak, uint64_t* elapsed_us) {
    arena_reset(arena);
    uint64_t start = time_us_64();
    float* logits = gpt_inference(arena, tokens, length, config);
    *elapsed_us += time_us_64() - start;
    pico_memory_sample();
    size_t used = arena_get_usage(arena);
    if (used > *peak) {
        *peak = used;
    }
    if (!logits) {
        printf("ERROR: inference failed at context length %zu (arena %zu/%zu bytes).\n",
               length, used, arena->size);
        return NULL;
    }
    for (size_t i = 0; i < config->vocab_size; i++) {
        if (!isfinite(logits[i])) {
            printf("ERROR: non-finite logit %zu at context length %zu.\n", i, length);
            return NULL;
        }
    }
    return logits;
}

static bool check_reference(Arena* arena, const GPTConfig* config) {
    int32_t tokens[MAX_SEQ_LEN];
    for (size_t i = 0; i < MAX_SEQ_LEN; i++) {
        tokens[i] = (int32_t)i;
    }
    size_t peak = 0;
    uint64_t elapsed = 0;
    float* logits = infer(arena, config, tokens, MAX_SEQ_LEN, &peak, &elapsed);
    if (!logits) {
        return false;
    }
    bool passed = true;
    float maximum_error = 0.0f;
    size_t top_token = 0;
    for (size_t i = 0; i < VOCAB_SIZE; i++) {
        float error = fabsf(logits[i] - reference_logits[i]);
        if (error > maximum_error) {
            maximum_error = error;
        }
        if (error > 1e-5f + 1e-5f * fabsf(reference_logits[i])) {
            printf("ERROR: token %zu logit %.9f, expected %.9f.\n",
                   i, (double)logits[i], (double)reference_logits[i]);
            passed = false;
        }
        if (logits[i] > logits[top_token]) {
            top_token = i;
        }
    }
    printf("Full-context reference: %s; max absolute error %.9f; top token %zu.\n",
           passed ? "PASS" : "FAIL", (double)maximum_error, top_token);
    printf("31-token inference: %" PRIu64 " us; arena peak %zu/%zu bytes.\n",
           elapsed, peak, arena->size);
    return passed;
}

static bool generate_sample(Arena* arena, const GPTConfig* config) {
    int32_t tokens[NEW_TOKENS + 1] = {0};
    size_t peak = 0;
    uint64_t elapsed = 0;
    srand(42);
    printf("Sample: start=newline, temperature=0.8, seed=42, %d new characters, rolling context=%d:\n",
           NEW_TOKENS, MAX_SEQ_LEN);
    uint64_t start = time_us_64();
    for (size_t generated = 0; generated < NEW_TOKENS; generated++) {
        size_t length = generated + 1;
        size_t context_length = length < config->max_seq_len ? length : config->max_seq_len;
        const int32_t* context = tokens + length - context_length;
        float* logits = infer(arena, config, context, context_length, &peak, &elapsed);
        if (!logits) {
            return false;
        }
        int32_t next = sample_token(logits, config->vocab_size, 0.8f);
        if (next < 0 || (size_t)next >= gpt_vocab_size) {
            printf("ERROR: invalid generated token %" PRId32 ".\n", next);
            return false;
        }
        tokens[length] = next;
        putchar(gpt_vocab[next]);
        pico_memory_sample();
    }
    uint64_t wall_elapsed = time_us_64() - start;
    printf("\nInference total: %" PRIu64 " us (%.3f ms) for %d generated tokens (excludes sampling/printing).\n",
           elapsed, (double)elapsed / 1000.0, NEW_TOKENS);
    printf("Generation wall time: %" PRIu64 " us (%.3f ms), including sampling, printing and diagnostics.\n",
           wall_elapsed, (double)wall_elapsed / 1000.0);
    printf("Generation arena peak: %zu/%zu bytes.\n", peak, arena->size);
    return true;
}

static void run_demo(Arena* arena, const GPTConfig* config) {
#if defined(PICO_COPY_TO_RAM) && PICO_COPY_TO_RAM
    printf("\nPicoGPT: RP2350 Arm, float32, SRAM code+weights (copied from flash at boot), core 0 only\n");
#else
    printf("\nPicoGPT: RP2350 Arm, float32, flash/XIP weights, core 0 only\n");
#endif
    printf("System clock: %" PRIu32 " Hz\n", clock_get_hz(clk_sys));
    printf("Static arena: %zu bytes at %p; token weights at %p\n",
           sizeof(arena_memory), (void*)arena_memory, (const void*)token_embedding);
    printf("Core 0 stack guard enabled; core 1 is idle. No KV cache.\n");
    pico_memory_report("before inference");
    if (gpt_vocab_size != config->vocab_size || gpt_vocab[0] != '\n') {
        printf("ERROR: vocabulary does not match the model/start token.\n");
        return;
    }
    if (!check_reference(arena, config) || !generate_sample(arena, config)) {
        printf("ERROR: demo stopped; see diagnostics above.\n");
        pico_memory_report("failed demo");
        return;
    }
    pico_memory_report("after generation");
    printf("Demo complete. Send 'r' to repeat.\n");
}

int main(void) {
    pico_memory_init();
    if (!stdio_init_all()) {
        panic("Could not initialize USB stdio");
    }
    Arena arena;
    arena_init(&arena, arena_memory, sizeof(arena_memory), 8);
    GPTConfig config = gpt_config_init();
    bool connected = false;
    while (true) {
        if (!stdio_usb_connected()) {
            connected = false;
            sleep_ms(20);
            continue;
        }
        if (!connected) {
            connected = true;
            run_demo(&arena, &config);
        }
        int command = getchar_timeout_us(20000);
        if (command == 'r') {
            run_demo(&arena, &config);
        }
    }
}
