#!/usr/bin/env python3
"""
Generate representative input data for TFLite model quantization.
Creates .npy files in rep_inputs/ directory matching the model's input shape/dtype.
"""

import os
import numpy as np
import tensorflow as tf

def main():
    model_path = "gpt.tflite"
    output_dir = "rep_inputs"
    num_samples = 128
    seed = 42

    # Load model to get input details
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    
    in_details = interp.get_input_details()
    
    print(f"Model has {len(in_details)} input(s):")
    for i, d in enumerate(in_details):
        print(f"  Input {i}: name='{d['name']}', shape={d['shape']}, dtype={d['dtype']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    for sample_idx in range(num_samples):
        if len(in_details) == 1:
            # Single input model - save array directly
            d = in_details[0]
            shape = tuple(int(s) if s > 0 else 1 for s in d["shape"])
            dtype = d["dtype"]
            
            if dtype == np.int32 or dtype == np.int64:
                # For token inputs, generate random token IDs
                # Vocab size is 47 for this model (detected from output shape)
                vocab_size = 47
                x = rng.integers(low=0, high=vocab_size, size=shape, dtype=dtype)
            elif dtype == np.float32:
                x = rng.standard_normal(size=shape).astype(np.float32)
            else:
                x = rng.standard_normal(size=shape).astype(dtype)
            
            data = x
        else:
            # Multi-input model - save as dict
            data = {}
            for d in in_details:
                shape = tuple(int(s) if s > 0 else 1 for s in d["shape"])
                dtype = d["dtype"]
                name = d["name"]
                
                if dtype == np.int32 or dtype == np.int64:
                    vocab_size = 50257
                    x = rng.integers(low=0, high=vocab_size, size=shape, dtype=dtype)
                elif dtype == np.float32:
                    x = rng.standard_normal(size=shape).astype(np.float32)
                else:
                    x = rng.standard_normal(size=shape).astype(dtype)
                
                data[name] = x
        
        # Save as .npy file
        filename = os.path.join(output_dir, f"sample_{sample_idx:04d}.npy")
        np.save(filename, data)
    
    print(f"\nGenerated {num_samples} representative samples in '{output_dir}/'")

if __name__ == "__main__":
    main()
