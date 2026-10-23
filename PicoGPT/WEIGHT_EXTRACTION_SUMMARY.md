# GPT TensorFlow Lite Weight Extraction Summary

## Scope and architecture

The standalone C implementation reproduces the optimized, bundled
[`gpt.tflite`](gpt.tflite) graph in float32. It uses two pre-normalized transformer
blocks, one attention head per block, 32-dimensional embeddings, a 128-dimensional
FFN, a 47-token vocabulary, and a maximum context of 31 tokens. The output
projection shares the token embedding weights.

The C runtime needs only the C standard library and libm. TensorFlow is used
offline for extraction and regression tests, not during C inference. This does
not yet establish agreement with the original PyTorch model or performance on
the Pico.

## Regenerating the headers

From this directory, using a Python environment with TensorFlow and NumPy:

```sh
make extract-weights PYTHON=python3
# Or select an input/output location explicitly:
python3 extract_weights_enhanced.py --model gpt.tflite --output-dir gpt_weights
```

[`extract_weights_enhanced.py`](extract_weights_enhanced.py) reads FlatBuffer
constants and follows graph edges backward from the output through both blocks.
Layer order is determined by residual connections, not tensor numbers, names,
or shapes alone. It checks the supported normalization, attention, and activation
operations and fails on incompatible graphs instead of guessing. It is a
model-specific extractor, not a general TFLite-to-C compiler.

The generated files remain:

- [`gpt_weights.h`](gpt_weights/gpt_weights.h): master include and access macros.
- [`gpt_constants.h`](gpt_weights/gpt_constants.h): dimensions, mask, final
  normalization, and exact GELU.
- [`gpt_embeddings.h`](gpt_weights/gpt_embeddings.h): token and position tables.
- [`gpt_layer1.h`](gpt_weights/gpt_layer1.h) and
  [`gpt_layer2.h`](gpt_weights/gpt_layer2.h): each block's normalization and
  projection parameters.

Float literals have sufficient decimal precision to round-trip the original
float32 constants. Regeneration is tested byte-for-byte against the checked-in
headers.

## Matrix layouts and graph mapping

TFLite `FULLY_CONNECTED` stores weights as `[output, input]`. C `matmul` consumes
`[input, output]`. The extractor transposes the weights once, leaving the
readable C multiplication loops unchanged.

QKV is stored as three consecutive `[32, 32]` matrices in one `[96, 32]` array:
Q occupies rows 0-31, K rows 32-63, and V rows 64-95. **Each block is transposed
individually**, so its row is an input feature and its column an output feature.
The embedding tables are not transposed; the language-model head explicitly
multiplies by the token table's transpose.

| TFLite constant | Graph role | TFLite shape | C shape |
|-----------------|------------|--------------|---------|
| `arith.constant9` | Token embeddings / output head | [47, 32] | [47, 32] |
| `model_1/tf.math.add_107/Add/y` | Position embeddings | [31, 32] | [31, 32] |
| `arith.constant8` | Layer 1 QKV | [96, 32] | [96, 32], blockwise transpose |
| `arith.constant7` | Layer 1 attention output | [32, 32] | [32, 32], transposed |
| `arith.constant6` | Layer 1 FFN up | [128, 32] | [32, 128] |
| `arith.constant5` | Layer 1 FFN down | [32, 128] | [128, 32] |
| `arith.constant4` | Layer 2 QKV | [96, 32] | [96, 32], blockwise transpose |
| `arith.constant3` | Layer 2 attention output | [32, 32] | [32, 32], transposed |
| `arith.constant2` | Layer 2 FFN up | [128, 32] | [32, 128] |
| `arith.constant1` | Layer 2 FFN down | [32, 128] | [128, 32] |

Names in this table describe the bundled file; the extractor does not rely on
them. Generated headers also annotate each parameter's source tensor.

## Normalization and activation

Each block has two distinct normalization scales:

| C parameter | TFLite scale tensor suffix |
|-------------|-----------------------------|
| `layer1_ln_gamma` (before attention) | `tf.math.multiply_97/Mul/y` |
| `layer1_ffn_ln_gamma` | `tf.math.multiply_102/Mul/y` |
| `layer2_ln_gamma` (before attention) | `tf.math.multiply_105/Mul/y` |
| `layer2_ffn_ln_gamma` | `tf.math.multiply_110/Mul/y` |
| `final_ln_gamma` | `tf.math.multiply_113/Mul/y` |

All five corresponding beta arrays come from the graph's zero-valued
`arith.constant10`. Normalization uses epsilon `1e-5`.

The FFN uses **exact GELU**, `0.5*x*(1 + erf(x/sqrt(2)))`, matching the graph's
`approximate=false` option. Attention applies a causal mask and `1/sqrt(32)`
scaling; the TFLite graph distributes this as `32^(-1/4)` on both Q and K.

## Numerical regression tests

```sh
make parity-test PYTHON=python3
```

The tests require TensorFlow, NumPy, FlatBuffers, and a C compiler. They were
validated with TensorFlow 2.20.0. `PYTHON`, `CC`, and `CFLAGS` select the local
interpreter/compiler; for the existing local environment, use
`PYTHON=./gpt_venv/bin/python`. The probe is built in a temporary directory and
does not overwrite the host executables.

[`tests/test_parity.py`](tests/test_parity.py) checks:

- All 47 logits and the highest-scoring token on 104 full-length inputs:
  four structured cases and 100 seeded random cases.
- Every context length from 1 through 31, including independence from future
  tokens in the padded TFLite input.
- 26 intermediate tensors spanning embeddings, both blocks' normalization,
  Q/K/V, attention probabilities/context, FFNs, residuals, and final normalization.
- The existing single-token attention/FFN helpers.
- Reproducible headers, independence from tensor numbering, and explicit failures
  for unsupported graphs.

The tolerance for every numerical comparison is
`abs(C - TFLite) <= 1e-5 + 1e-5 * abs(TFLite)`, with matching shapes and finite
values required. All 104 full-length cases pass; the observed maximum absolute
logit error was approximately `1.48e-5`. Tests also pass with an unoptimized C
build under AddressSanitizer and UndefinedBehaviorSanitizer.

### TFLite last-token selection compatibility

The original file has a final `GATHER` index of `-1`, which the tested runtime
rejects. The test harness explicitly replaces **only that index in memory** with
`30` for a full context. For shorter C contexts, TFLite still receives 31 tokens
but selects position `length - 1`; causal masking prevents the padding from
affecting that position. Intermediate tests disable delegation to retain tensors.

Neither extraction nor testing changes the model file. These tests establish
parity with that explicitly corrected last-position selection, not that the
unmodified file can execute in the tested runtime. Comparing against the
original PyTorch model remains a separate check of conversion fidelity.