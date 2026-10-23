# PicoGPT
A GPT on a RP pico

The current host implementation runs a tiny, two-block GPT in readable C using
weights extracted from the bundled TFLite model. It uses float32 arithmetic and
arena-allocated intermediate tensors; no TensorFlow runtime is needed by C.

```sh
cd PicoGPT
make
./gpt_model
```

With TensorFlow, NumPy, and FlatBuffers available in your Python environment,
run `make parity-test PYTHON=python3` to compare C logits and intermediate tensors
against TFLite. See the [weight extraction and parity notes](PicoGPT/WEIGHT_EXTRACTION_SUMMARY.md)
for matrix layouts, regeneration, tolerances, and the in-memory correction needed
for the model's final TFLite `GATHER` index.

The original Pico hardware experiment remains separate from the trained-model
host implementation. Hardware deployment and quantization are not covered by
the numerical parity tests.
