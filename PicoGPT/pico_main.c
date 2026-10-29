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

#if GPT_PROFILE_ENABLED
#include "hardware/structs/busctrl.h"
#include "hardware/sync.h"
#endif

#define ARENA_SIZE (144 * 1024)
#define NEW_TOKENS 100
#define MATMUL_REPEATS 10
#define MATMUL_SAMPLES 5

static _Alignas(8) uint8_t arena_memory[ARENA_SIZE];
static volatile float matmul_benchmark_sink;

#if GPT_PROFILE_ENABLED
static bool profile_capture;
static struct {
    uint64_t time_us;
    uint32_t calls;
} profile_stages[GPT_STAGE_COUNT];

uint64_t gpt_profile_now(void) {
    return profile_capture ? time_us_64() : 0;
}

void gpt_profile_record(GPTProfileStage stage, uint64_t elapsed_us) {
    if (profile_capture) {
        profile_stages[stage].time_us += elapsed_us;
        profile_stages[stage].calls++;
    }
}
#endif

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

typedef void (*MatmulKernel)(const float*, size_t, size_t, const float*, size_t, float*);

/* Keep both benchmark kernels out of line, with the same call overhead. */
static __attribute__((noinline, noclone)) void benchmark_baseline(
    const float* a, size_t rows, size_t inner, const float* b, size_t columns, float* output
) {
    matmul_baseline(a, rows, inner, b, columns, output);
}

static __attribute__((noinline, noclone)) void benchmark_unroll4(
    const float* a, size_t rows, size_t inner, const float* b, size_t columns, float* output
) {
    matmul_unroll4(a, rows, inner, b, columns, output);
}

static __attribute__((noinline, noclone)) uint64_t time_matmul(
    MatmulKernel kernel, const float* input, size_t rows, size_t inner,
    const float* weights, size_t columns, float* output
) {
    uint64_t start = time_us_64();
    for (size_t repeat = 0; repeat < MATMUL_REPEATS; repeat++) {
        kernel(input, rows, inner, weights, columns, output);
        matmul_benchmark_sink = output[repeat % (rows * columns)];
    }
    return time_us_64() - start;
}

static uint64_t median_matmul_time(uint64_t samples[MATMUL_SAMPLES]) {
    for (size_t i = 1; i < MATMUL_SAMPLES; i++) {
        for (size_t j = i; j > 0 && samples[j] < samples[j - 1]; j--) {
            uint64_t swap = samples[j];
            samples[j] = samples[j - 1];
            samples[j - 1] = swap;
        }
    }
    return samples[MATMUL_SAMPLES / 2];
}

static bool benchmark_matmul(Arena* arena) {
    const struct {
        const char* name;
        size_t inner;
        size_t columns;
        const float* weights;
    } shapes[] = {
        {"attention", EMBED_DIM, EMBED_DIM, (const float*)layer1_attn_qkv},
        {"FFN up", EMBED_DIM, FFN_DIM, (const float*)layer1_ffn_up},
        {"FFN down", FFN_DIM, EMBED_DIM, (const float*)layer1_ffn_down},
    };
    const size_t lengths[] = {1, MAX_SEQ_LEN};
    printf("\nMatmul A/B: median of %d batches x %d calls; alternating order; interrupts enabled.\n",
           MATMUL_SAMPLES, MATMUL_REPEATS);
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        size_t inner = shapes[shape].inner, columns = shapes[shape].columns;
        for (size_t length = 0; length < sizeof(lengths) / sizeof(lengths[0]); length++) {
            size_t rows = lengths[length];
            arena_reset(arena);
            float* input = arena_alloc_tensor_2d(arena, rows, inner);
            float* baseline = arena_alloc_tensor_2d(arena, rows, columns);
            float* unrolled = arena_alloc_tensor_2d(arena, rows, columns);
            if (!input || !baseline || !unrolled) {
                printf("ERROR: insufficient arena for matmul benchmark.\n");
                return false;
            }
            for (size_t i = 0; i < rows * inner; i++) {
                input[i] = (float)((int)(i % 31) - 15) / 17.0f;
            }
            for (size_t i = 0; i < rows * columns; i++) {
                baseline[i] = unrolled[i] = NAN;
            }
            benchmark_baseline(input, rows, inner, shapes[shape].weights, columns, baseline);
            benchmark_unroll4(input, rows, inner, shapes[shape].weights, columns, unrolled);
            for (size_t i = 0; i < rows * columns; i++) {
                if (!isfinite(baseline[i]) || !isfinite(unrolled[i]) || baseline[i] != unrolled[i]) {
                    printf("ERROR: matmul mismatch in %s, M=%zu at output %zu.\n",
                           shapes[shape].name, rows, i);
                    return false;
                }
            }
            uint64_t baseline_times[MATMUL_SAMPLES], unrolled_times[MATMUL_SAMPLES];
            for (size_t sample = 0; sample < MATMUL_SAMPLES; sample++) {
                for (size_t turn = 0; turn < 2; turn++) {
                    bool use_unrolled = ((sample + turn) % 2) != 0;
                    uint64_t elapsed = time_matmul(
                        use_unrolled ? benchmark_unroll4 : benchmark_baseline,
                        input, rows, inner, shapes[shape].weights, columns, baseline
                    );
                    if (use_unrolled) unrolled_times[sample] = elapsed;
                    else baseline_times[sample] = elapsed;
                }
            }
            uint64_t baseline_us = median_matmul_time(baseline_times);
            uint64_t unrolled_us = median_matmul_time(unrolled_times);
            if (!baseline_us || !unrolled_us) {
                printf("ERROR: matmul timing below timer resolution; increase MATMUL_REPEATS.\n");
                return false;
            }
            printf("  %s M=%zu K=%zu N=%zu: baseline %.3f us/call, unroll4 %.3f us/call, speedup %.3fx; outputs MATCH.\n",
                   shapes[shape].name, rows, inner, columns,
                   (double)baseline_us / MATMUL_REPEATS, (double)unrolled_us / MATMUL_REPEATS,
                   (double)baseline_us / (double)unrolled_us);
        }
    }
    arena_reset(arena);
    pico_memory_sample();
    return true;
}

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

#if GPT_PROFILE_ENABLED
static bool profile_stages_at_length(Arena* arena, const GPTConfig* config, size_t length) {
    static const char* const names[GPT_STAGE_COUNT] = {
        "Embeddings (+allocation)",
        "Pre-layer normalization",
        "Q/K/V projections",
        "Attention QK scores",
        "Causal mask + softmax",
        "Attention probabilities * V",
        "Attention output projection",
        "Attention residual addition",
#if GPT_FFN_FUSE_GELU
        "FFN up + fused GELU",
#else
        "FFN up projection",
#endif
        "Separate GELU",
        "FFN down + residual",
        "Final normalization (+allocation)",
        "All-position logits (+allocation)",
    };
    const size_t runs = 5;
    int32_t tokens[MAX_SEQ_LEN];
    for (size_t i = 0; i < MAX_SEQ_LEN; i++) tokens[i] = (int32_t)i;
    size_t peak = 0;
    uint64_t total = 0;
    if (!infer(arena, config, tokens, length, &peak, &total)) return false;
    total = 0;
    memset(profile_stages, 0, sizeof(profile_stages));
    profile_capture = true;
    for (size_t run = 0; run < runs; run++) {
        if (!infer(arena, config, tokens, length, &peak, &total)) {
            profile_capture = false;
            return false;
        }
    }
    profile_capture = false;
    if (!total) {
        printf("ERROR: inference profiling interval is below timer resolution.\n");
        return false;
    }
    uint64_t accounted = 0;
    printf("\nStage profile: context=%zu, %zu runs, both blocks combined, interrupts enabled.\n",
           length, runs);
    printf("  Instrumented inference average: %.3f us; arena peak %zu bytes.\n",
           (double)total / runs, peak);
    printf("  Stage                              us/inference   %% total   calls over all runs\n");
    for (size_t stage = 0; stage < GPT_STAGE_COUNT; stage++) {
        accounted += profile_stages[stage].time_us;
        printf("  %-34s %10.3f    %6.2f    %" PRIu32 "\n", names[stage],
               (double)profile_stages[stage].time_us / runs,
               100.0 * (double)profile_stages[stage].time_us / (double)total,
               profile_stages[stage].calls);
    }
    if (accounted > total) {
        printf("ERROR: stage accounting exceeds measured inference time.\n");
        return false;
    }
    printf("  Other/allocator/instrumentation    %10.3f    %6.2f\n",
           (double)(total - accounted) / runs, 100.0 * (double)(total - accounted) / (double)total);
    return true;
}

static bool profile_sram_bus(Arena* arena, const GPTConfig* config) {
    if (busctrl_hw->perfctr_en) {
        printf("ERROR: bus counters already enabled; leaving their owner undisturbed.\n");
        return false;
    }
    uint32_t saved_selectors[4];
    for (size_t i = 0; i < 4; i++) saved_selectors[i] = busctrl_hw->counter[i].sel;
    int32_t tokens[MAX_SEQ_LEN];
    for (size_t i = 0; i < MAX_SEQ_LEN; i++) tokens[i] = (int32_t)i;
    struct {
        uint32_t counts[4];
        uint64_t elapsed;
    } measurements[10] = {0};
    bool passed = true;
    size_t peak = 0;
    profile_capture = false;
    for (size_t bank = 0; bank < 10; bank++) {
        /* SDK events are contiguous: upstream, downstream, contested, access. */
        uint32_t first_event = arbiter_sram0_perf_event_stall_upstream - 4 * bank;
        for (size_t event = 0; event < 4; event++) {
            busctrl_hw->counter[event].sel = first_event + event;
            busctrl_hw->counter[event].value = 0;
        }
        __dsb();
        busctrl_hw->perfctr_en = 1;
        __dsb();
        float* logits = infer(arena, config, tokens, MAX_SEQ_LEN, &peak,
                              &measurements[bank].elapsed);
        __dsb();
        busctrl_hw->perfctr_en = 0;
        __dsb();
        for (size_t event = 0; event < 4; event++) {
            measurements[bank].counts[event] = busctrl_hw->counter[event].value;
            if (measurements[bank].counts[event] == BUSCTRL_PERFCTR0_BITS) {
                printf("ERROR: SRAM%zu event %zu counter saturated; discard this report.\n",
                       bank, event);
                passed = false;
            }
        }
        if (!logits || !passed) {
            passed = false;
            break;
        }
    }
    for (size_t i = 0; i < 4; i++) busctrl_hw->counter[i].sel = saved_selectors[i];
    if (!passed) return false;
    printf("\nSRAM bus events: one full-context pass per bank; includes wrapper/validation and interrupts.\n");
    printf("  Bank   accesses   contested accesses   upstream stall cycles   downstream stall cycles   inference us\n");
    for (size_t bank = 0; bank < 10; bank++) {
        const uint32_t* counts = measurements[bank].counts;
        printf("  %4zu %10" PRIu32 " %20" PRIu32 " %23" PRIu32 " %25" PRIu32 " %14" PRIu64 "\n",
               bank, counts[3], counts[2], counts[0], counts[1], measurements[bank].elapsed);
    }
    printf("  Counters include instruction fetches and all bus masters, not just tensor data.\n");
    printf("  Banks are sampled on separate runs. Do not sum stall counts into a CPU stalled-time percentage.\n");
    printf("  Low stalls do not eliminate load/store instruction costs; high accesses alone do not prove a bottleneck.\n");
    return true;
}

static void run_profile(Arena* arena, const GPTConfig* config) {
    printf("\nProfiling changes code layout/register allocation; compare stage shares, not absolute speed against other UF2s.\n");
    if (!check_reference(arena, config) ||
        !profile_stages_at_length(arena, config, 1) ||
        !profile_stages_at_length(arena, config, MAX_SEQ_LEN) ||
        !profile_sram_bus(arena, config)) {
        printf("ERROR: profiling stopped; see diagnostics above.\n");
        return;
    }
    pico_memory_report("after profiling");
    printf("Profile complete. Send 'p' to repeat.\n");
}
#endif

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
#if GPT_PROFILE_ENABLED
    printf("Profiling build: send 'p' for stage timing and SRAM bus counters; ordinary timings are instrumented.\n");
#endif
#if GPT_MATMUL_UNROLL4
    printf("Model matmul: four outputs at once.\n");
#else
    printf("Model matmul: baseline dot products.\n");
#endif
#if GPT_FFN_FUSE_GELU
    printf("FFN: buffer reuse, fused residual, fused GELU.\n");
#else
    printf("FFN: buffer reuse, fused residual, separate GELU.\n");
#endif
#if GPT_ATTENTION_UNROLL4
    printf("Attention: four scores, four values, four mask writes at once.\n");
#else
    printf("Attention: scalar scores, values and mask writes.\n");
#endif
    pico_memory_report("before inference");
    if (gpt_vocab_size != config->vocab_size || gpt_vocab[0] != '\n') {
        printf("ERROR: vocabulary does not match the model/start token.\n");
        return;
    }
    if (!benchmark_matmul(arena) || !check_reference(arena, config) ||
        !generate_sample(arena, config)) {
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
#if GPT_PROFILE_ENABLED
        if (command == 'p') {
            run_profile(&arena, &config);
        }
#endif
    }
}
