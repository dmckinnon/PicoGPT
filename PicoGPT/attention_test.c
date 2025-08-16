/*
 * attention_head.c
 *
 * A reference implementation of a single‑head self‑attention block and a
 * simple two‑layer feed‑forward network (MLP) using 8‑bit quantized
 * inputs and weights with 32‑bit accumulators.  This file is designed to
 * support two build targets: an offline simulation on a host PC and an
 * embedded build for the RP2350 microcontroller.  When compiled for the
 * microcontroller, timing and I/O stubs can be provided via preprocessor
 * defines.  When run on a host PC the code will dump intermediate
 * results to a file so that a Python script can verify correctness.
 *
 * This code makes a number of simplifications in order to keep the
 * implementation manageable for a first prototype:
 *  • The quantization scheme is uniform, symmetric and per‑tensor.  Both
 *    activations and weights are stored as signed 8‑bit integers in the
 *    range [‑128, 127].  All matrix multiplications accumulate into
 *    32‑bit integers and are requantized back to 8‑bit by right shifting
 *    with rounding.  For numerical clarity the exact scale factors are
 *    not exposed here; instead a single shift constant controls the
 *    effective dynamic range.
 *  • Non‑linear operations such as Softmax and LayerNorm are computed
 *    using floating point internally.  Although integer‑only
 *    approximations exist (see I‑BERT for a discussion【244357888843366†L114-L118】), the goal of this
 *    exercise is to obtain a correct baseline and measure the cost of the
 *    integer matrix multiplications.  After computing the non‑linear
 *    operation in float, results are quantized back to 8‑bit for
 *    subsequent integer operations.  The I‑BERT paper notes that
 *    performing MatMul in INT8 with INT32 accumulation and then
 *    performing the non‑linear functions in INT32 precision before
 *    requantization is a practical design choice【244357888843366†L114-L118】【244357888843366†L1045-L1055】.
 *  • The feed‑forward network uses a ReLU activation between its two
 *    linear layers for simplicity.  GPT models often use GELU; however
 *    GELU is harder to implement efficiently with integer arithmetic and
 *    therefore ReLU suffices for this test.  Should a more accurate
 *    approximation be required, the I‑BERT paper describes how GELU can
 *    be approximated with a low‑order polynomial【244357888843366†L214-L313】.
 *
 * To switch between host and embedded builds, define RUN_ON_RP2350 when
 * compiling for the microcontroller.  On the host, the main() function
 * seeds the pseudo‑random number generator, constructs deterministic
 * dummy data, runs the attention head and writes the output tensor to
 * "attention_output.bin" for later inspection.  On the embedded target
 * the main() function should be replaced or extended by the user to
 * provide actual inputs and to record timing information with on‑chip
 * peripherals.  A simple timing stub using clock() is provided for the
 * host build.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "pico/stdlib.h"

/* Compile‑time constants for tensor dimensions.  Changing these values
 * allows you to experiment with different model sizes without touching
 * the rest of the code.  The values here reflect the user’s test case
 * of 32 tokens and an embedding dimension of 32.  The hidden dimension
 * in the MLP is 96 as described by the user.
 */
#define SEQ_LEN    32
#define DIM        32
#define HIDDEN_DIM 96

/* Right shift used when requantizing from INT32 accumulators back to
 * INT8.  A shift of 8 corresponds to dividing by 256.  Adjusting this
 * value will trade off dynamic range versus saturation.  In practice,
 * proper scaling factors are derived from calibration data; here we
 * choose 8 as a reasonable default.  See the discussion in I‑BERT where
 * MatMul is performed with INT8 inputs and INT32 accumulation【244357888843366†L1045-L1055】.
 */
#define REQ_SHIFT  8

/* A small epsilon to avoid divide‑by‑zero in LayerNorm. */
#define LN_EPS     1e-5f

/* Type aliases for clarity.  All quantized values are stored in
 * int8_t.  Note that C treats int8_t as signed; values range from
 * –128 to 127.  Activations are assumed to be symmetric around zero.
 */
typedef int8_t q8_t;

/* Clamp a 32‑bit integer to the range of int8_t.  This helper
 * saturates values outside [‑128, 127].  Without saturation the
 * behavior of casting a large int32_t directly to int8_t is
 * implementation defined.
 */
static inline q8_t clamp_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (q8_t)x;
}

/* Convert a floating point value in approximately [‑1.0, 1.0] to a
 * quantized int8_t.  Values outside the range are saturated.  In this
 * reference code we use 127 as the scaling factor (i.e. 1.0 maps to
 * 127).  When dequantizing we divide by the same 127.  This matches
 * uniform symmetric quantization described in many works【244357888843366†L214-L240】.
 */
static inline q8_t quantize_float_to_q8(float x) {
    float scaled = x * 127.0f;
    if (scaled > 127.0f) scaled = 127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    return (q8_t)lrintf(scaled);
}

/* Convert a quantized int8_t back to floating point.  The scale
 * factor 1/127 yields values roughly in [‑1.0, 1.0].
 */
static inline float dequantize_q8(q8_t x) {
    return ((float)x) / 127.0f;
}

/* Requantize an INT32 accumulator to INT8 by right shifting with
 * rounding.  The shift amount should be chosen based on the maximum
 * possible magnitude of the accumulator.  Rounding prevents systematic
 * bias when converting from high precision to low precision.  This
 * matches the requirement of requantization after performing
 * operations in INT32 precision【244357888843366†L1045-L1055】.
 */
static inline q8_t requantize_int32(int32_t x) {
    /* Add rounding offset and shift. */
    int32_t shifted = (x + (1 << (REQ_SHIFT - 1))) >> REQ_SHIFT;
    return clamp_int8(shifted);
}

/* Compute the Q, K and V projections.  The combined weight matrix has
 * shape (3*DIM, DIM) where the first DIM rows correspond to Q, the
 * next DIM rows correspond to K and the final DIM rows correspond to V.
 * Each output is a SEQ_LEN × DIM array of int8 values.  Inputs and
 * weights are int8 and accumulations are 32‑bit.  After the dot
 * product we requantize with a right shift.
 */
static void compute_qkv(const q8_t input[SEQ_LEN][DIM],
                        const q8_t weight_qkv[3*DIM][DIM],
                        q8_t Q[SEQ_LEN][DIM],
                        q8_t K[SEQ_LEN][DIM],
                        q8_t V[SEQ_LEN][DIM])
{
    for (int token = 0; token < SEQ_LEN; ++token) {
        for (int out = 0; out < 3 * DIM; ++out) {
            int32_t acc = 0;
            for (int d = 0; d < DIM; ++d) {
                /* Cast to int32_t to avoid overflow in the product. */
                int32_t a = (int32_t)input[token][d];
                int32_t b = (int32_t)weight_qkv[out][d];
                acc += a * b;
            }
            /* Requantize accumulator. */
            q8_t out_val = requantize_int32(acc);
            if (out < DIM) {
                Q[token][out] = out_val;
            } else if (out < 2 * DIM) {
                K[token][out - DIM] = out_val;
            } else {
                V[token][out - 2 * DIM] = out_val;
            }
        }
    }
}

/* Compute the attention scores and output.  This function takes
 * quantized Q, K and V matrices and produces a quantized context
 * output.  For each query token we compute dot products with all keys,
 * apply scaling by 1/sqrt(DIM), perform a softmax over the 32 token
 * positions and then take the weighted sum over V.  All intensive
 * operations are performed in floating point for simplicity.
 * 
 * 
 * Not happy with the amount of float here, I intend for this to all be int
 */
static void compute_attention(const q8_t Q[SEQ_LEN][DIM],
                              const q8_t K[SEQ_LEN][DIM],
                              const q8_t V[SEQ_LEN][DIM],
                              q8_t context[SEQ_LEN][DIM])
{
    const float scale = 1.0f / sqrtf((float)DIM);
    /* For each token i as query */
    for (int i = 0; i < SEQ_LEN; ++i) {
        /* Compute scores against all keys. */
        float scores[SEQ_LEN];
        float max_score = -INFINITY;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                float q_f = dequantize_q8(Q[i][d]);
                float k_f = dequantize_q8(K[j][d]);
                dot += q_f * k_f;
            }
            dot *= scale;
            scores[j] = dot;
            if (dot > max_score) max_score = dot;
        }
        /* Subtract max for numerical stability and exponentiate. */
        float exp_scores[SEQ_LEN];
        float sum_exp = 0.0f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float e = expf(scores[j] - max_score);
            exp_scores[j] = e;
            sum_exp += e;
        }
        /* Compute weighted sum of V. */
        for (int d = 0; d < DIM; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                float weight = exp_scores[j] / sum_exp;
                float v_f = dequantize_q8(V[j][d]);
                acc += weight * v_f;
            }
            /* Quantize result back to int8. */
            context[i][d] = quantize_float_to_q8(acc);
        }
    }
}

/* Add a residual connection: out[i][d] = clamp(x[i][d] + y[i][d]).  We
 * operate directly on int8 values and saturate the sum to avoid
 * overflow.  The two input tensors are assumed to have the same
 * quantization scale.  A wider type is used for the addition.
 */
static void add_residual(const q8_t x[SEQ_LEN][DIM],
                         const q8_t y[SEQ_LEN][DIM],
                         q8_t out[SEQ_LEN][DIM])
{
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < DIM; ++d) {
            int16_t sum = (int16_t)x[i][d] + (int16_t)y[i][d];
            /* Clamp to int8 range. */
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            out[i][d] = (q8_t)sum;
        }
    }
}

/* Apply Layer Normalization to each token vector.  We dequantize the
 * input, compute the mean and variance along the feature dimension,
 * normalize and then requantize.  This implementation uses float for
 * clarity.  I‑BERT notes that computing LayerNorm on INT32
 * accumulations before requantization yields good accuracy and has
 * little overhead【244357888843366†L1045-L1055】.  For a fully integer implementation one
 * could approximate 1/sqrt(var) with a polynomial and perform the
 * normalization in fixed point【244357888843366†L214-L313】.  An epsilon term prevents division by
 * zero when the variance is extremely small.
 */
static void apply_layernorm(const q8_t input[SEQ_LEN][DIM],
                            q8_t output[SEQ_LEN][DIM])
{
    for (int i = 0; i < SEQ_LEN; ++i) {
        /* Compute mean of dequantized values. */
        float mean = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            mean += dequantize_q8(input[i][d]);
        }
        mean /= (float)DIM;
        /* Compute variance. */
        float var = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            float val = dequantize_q8(input[i][d]);
            float diff = val - mean;
            var += diff * diff;
        }
        var /= (float)DIM;
        float inv_std = 1.0f / sqrtf(var + LN_EPS);
        /* Normalize and requantize. */
        for (int d = 0; d < DIM; ++d) {
            float val = dequantize_q8(input[i][d]);
            float norm = (val - mean) * inv_std; /* gamma=1, beta=0 */
            output[i][d] = quantize_float_to_q8(norm);
        }
    }
}

/* Compute one fully connected layer: out[i][h] = sum_j in[i][j] * W[h][j] */
static void fully_connected(const q8_t input[SEQ_LEN][DIM],
                            const q8_t weights[HIDDEN_DIM][DIM],
                            q8_t output[SEQ_LEN][HIDDEN_DIM])
{
    for (int token = 0; token < SEQ_LEN; ++token) {
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            int32_t acc = 0;
            for (int d = 0; d < DIM; ++d) {
                acc += (int32_t)input[token][d] * (int32_t)weights[h][d];
            }
            output[token][h] = requantize_int32(acc);
        }
    }
}

/* Compute second fully connected: out[i][d] = sum_h in[i][h] * W2[d][h] */
static void fully_connected2(const q8_t input[SEQ_LEN][HIDDEN_DIM],
                             const q8_t weights[DIM][HIDDEN_DIM],
                             q8_t output[SEQ_LEN][DIM])
{
    for (int token = 0; token < SEQ_LEN; ++token) {
        for (int d = 0; d < DIM; ++d) {
            int32_t acc = 0;
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                acc += (int32_t)input[token][h] * (int32_t)weights[d][h];
            }
            output[token][d] = requantize_int32(acc);
        }
    }
}

/* Apply an in‑place ReLU activation on HIDDEN_DIM features: max(0,x). */
static void relu_inplace(q8_t data[SEQ_LEN][HIDDEN_DIM])
{
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            if (data[i][h] < 0) data[i][h] = 0;
        }
    }
}

/* Top‑level function to run the entire attention head with MLP.  The
 * sequence is:
 *   1. QKV projection
 *   2. Scaled dot‑product attention and context computation
 *   3. Residual add (context + input)
 *   4. LayerNorm
 *   5. Feed‑forward (FC→ReLU→FC)
 *   6. Residual add (MLP output + LayerNorm output)
 *   7. LayerNorm (optional, can be commented out if not desired)
 *
 * The function writes the final output into the provided buffer.  It
 * takes pointers to weight matrices for QKV, the first MLP layer and
 * the second MLP layer.  All buffers must be preallocated with the
 * appropriate dimensions.
 */
static void run_attention_head(const q8_t input[SEQ_LEN][DIM],
                               const q8_t weight_qkv[3*DIM][DIM],
                               const q8_t weight_fc1[HIDDEN_DIM][DIM],
                               const q8_t weight_fc2[DIM][HIDDEN_DIM],
                               q8_t output[SEQ_LEN][DIM])
{
    /* Step 1: QKV projection. */
    q8_t Q[SEQ_LEN][DIM];
    q8_t K[SEQ_LEN][DIM];
    q8_t V[SEQ_LEN][DIM];
    compute_qkv(input, weight_qkv, Q, K, V);

    /* Step 2: Scaled dot‑product attention. */
    q8_t context[SEQ_LEN][DIM];
    compute_attention(Q, K, V, context);

    /* Step 3: Residual add.  Add context back to input. */
    q8_t resid1[SEQ_LEN][DIM];
    add_residual(input, context, resid1);

    /* Step 4: LayerNorm. */
    q8_t ln1[SEQ_LEN][DIM];
    apply_layernorm(resid1, ln1);

    /* Step 5: Feed‑forward network. */
    q8_t hidden[SEQ_LEN][HIDDEN_DIM];
    fully_connected(ln1, weight_fc1, hidden);
    relu_inplace(hidden);
    q8_t mlp_out[SEQ_LEN][DIM];
    fully_connected2(hidden, weight_fc2, mlp_out);

    /* Step 6: Second residual add. */
    q8_t resid2[SEQ_LEN][DIM];
    add_residual(mlp_out, ln1, resid2);

    /* Step 7: Optional LayerNorm.  Uncomment the following line
     * if a second layer normalization is required (as in many GPT
     * implementations).  For this prototype we leave the second
     * normalization disabled by default. */
    // apply_layernorm(resid2, output);
    /* If second LayerNorm is disabled, copy resid2 to output. */
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < DIM; ++d) {
            output[i][d] = resid2[i][d];
        }
    }
}

/* Dump the output tensor to a binary file for offline inspection.  Each
 * element is written as a single int8_t in row‑major order.  A
 * companion Python script can read this file and compare it with a
 * reference implementation.  Returns 0 on success. */
static int dump_output_to_file(const char *filename,
                               const q8_t data[SEQ_LEN][DIM])
{
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, sizeof(q8_t), SEQ_LEN * DIM, f);
    fclose(f);
    return (written == SEQ_LEN * DIM) ? 0 : -1;
}

/* Create deterministic dummy data for testing.  This helper fills the
 * input and weight tensors with pseudo‑random values in [‑128, 127].  A
 * fixed seed ensures reproducibility.  Real deployments should
 * initialize these tensors with trained model weights and actual
 * embeddings. */
static void fill_dummy_data(q8_t input[SEQ_LEN][DIM],
                            q8_t w_qkv[3*DIM][DIM],
                            q8_t w_fc1[HIDDEN_DIM][DIM],
                            q8_t w_fc2[DIM][HIDDEN_DIM])
{
    /* Fixed seed for reproducibility. */
    srand(12345);
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < DIM; ++d) {
            input[i][d] = (q8_t)(rand() % 256 - 128);
        }
    }
    for (int i = 0; i < 3 * DIM; ++i) {
        for (int d = 0; d < DIM; ++d) {
            w_qkv[i][d] = (q8_t)(rand() % 256 - 128);
        }
    }
    for (int h = 0; h < HIDDEN_DIM; ++h) {
        for (int d = 0; d < DIM; ++d) {
            w_fc1[h][d] = (q8_t)(rand() % 256 - 128);
        }
    }
    for (int d = 0; d < DIM; ++d) {
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            w_fc2[d][h] = (q8_t)(rand() % 256 - 128);
        }
    }
}

/* Entry point for host simulation.  When RUN_ON_RP2350 is defined the
 * user should replace or augment this function to acquire real data
 * from the microcontroller, to call run_attention_head() and to record
 * runtime with on‑chip timers.  On the host we use clock() to
 * measure elapsed processor time for the entire forward pass. */
int attention_test(void)
{
    /* On the microcontroller the user can provide their own input
     * tensors and call run_attention_head().  Timing can be
     * performed using DWT cycle counters or a hardware timer.  See
     * platform‑specific documentation for details.  This stub is
     * intentionally left blank. */
    //return 0;
    /* Host simulation. */
    static q8_t input[SEQ_LEN][DIM];
    static q8_t w_qkv[3*DIM][DIM];
    static q8_t w_fc1[HIDDEN_DIM][DIM];
    static q8_t w_fc2[DIM][HIDDEN_DIM];
    static q8_t output[SEQ_LEN][DIM];

    /* Populate dummy data. */
    fill_dummy_data(input, w_qkv, w_fc1, w_fc2);

    /* Measure runtime. */
    //clock_t start = clock();
    int64_t start = time_us_64(); 
    run_attention_head(input, w_qkv, w_fc1, w_fc2, output);
    int64_t elapsed = time_us_64() - start;
    //clock_t end = clock();
    //double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Attention head forward pass took %lld microseconds on host.\n", elapsed);

#ifndef ON_TARGET
    /* Dump output to file for offline verification. */
    if (dump_output_to_file("attention_output.bin", output) != 0) {
        fprintf(stderr, "Failed to write output file\n");
        return 1;
    }
    printf("Output tensor written to attention_output.bin\n");
    return 0;
#endif
}