#!/usr/bin/env python3
"""
tflite_weight_ptq.py

Post-training *weight* quantization for an existing float32 .tflite model.

Key features:
- GLOBAL weight quant params: one scale/zero_point for ALL weight tensors.
- PER_TENSOR weight quant params: one scale/zero_point per weight tensor.
- Calibration loop over representative inputs (for optional activation histograms + output comparison).
- "Fake-quantized" TFLite output: quantize->dequantize weights back to float32 so the model stays runnable.
- Export C++ headers with int8 weights + quant params.

Dependencies:
- tensorflow (for tflite Interpreter and schema_py_generated)
- numpy
- flatbuffers (pulled in by tensorflow typically)

Usage examples:

# 1) Fake-quant weights with a single global scale, compare outputs, write new tflite and headers
python tflite_weight_ptq.py \
  --tflite_in model_fp32.tflite \
  --out_tflite_fakeq model_fakeq_global.tflite \
  --out_header_dir out_headers_global \
  --mode global \
  --rep_npy_dir rep_inputs/ \
  --num_samples 128 \
  --compare

  Tried this, resulted in confusingly named files, filled with zeroes
  Model file looks fine, weights are just buggered. Try per tensor

# 2) Per-tensor weight quantization, random representative inputs, write header only
python tflite_weight_ptq.py \
  --tflite_in model_fp32.tflite \
  --out_header_dir out_headers_pertensor \
  --mode per_tensor \
  --num_samples 64 \
  --seed 0

This appeared to work better, just need to pull out the quant data and associate with each tensor 
like extract weights, to take it into account
Next question: how can we transfer quantisations across ops? I'd like to avoid dequant and requant
This does not export the model with quanted weights

We need to associate the quantisation params with the sets of weights, and also
apply the quantisations to all the weights so they are default 8 bit

Representative data formats:
- --rep_npy_dir: directory containing .npy files. Each file can be:
    * a single input array (for single-input models), or
    * a Python object array / dict saved via np.save containing multiple inputs (advanced),
  OR you can provide --rep_npz where keys match input tensor names (optional extension you can add).
- If no rep data is provided, random inputs are generated from input shapes/dtypes.

"""

import argparse
import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

import flatbuffers


# ----------------------------
# Quantization helpers
# ----------------------------

@dataclass
class QuantParams:
    scale: float
    zero_point: int  # for int8
    qmin: int = -128
    qmax: int = 127

def choose_symmetric_int8_params(max_abs: float) -> QuantParams:
    # symmetric int8, zero_point = 0
    # scale = max_abs / 127
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        # Avoid division by zero; fall back to scale=1.0
        return QuantParams(scale=1.0, zero_point=0)
    return QuantParams(scale=float(max_abs) / 127.0, zero_point=0)

def quantize_dequantize_float_to_int8(x_f32: np.ndarray, qp: QuantParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_q_int8, x_dq_float32) using qp.
    """
    scale = qp.scale
    zp = qp.zero_point
    # Quantize
    q = np.round(x_f32 / scale + zp)
    q = np.clip(q, qp.qmin, qp.qmax).astype(np.int8)
    # Dequantize
    dq = (q.astype(np.float32) - zp) * scale
    return q, dq

def safe_c_identifier(name: str) -> str:
    # Convert tensor names to valid C identifiers
    s = name.strip()
    s = re.sub(r'[^0-9a-zA-Z_]', '_', s)
    s = re.sub(r'_+', '_', s)
    if re.match(r'^[0-9]', s):
        s = "_" + s
    return s

def format_c_array_int8(var_name: str, arr: np.ndarray, indent: str="    ") -> str:
    flat = arr.flatten()
    lines = []
    lines.append(f"static const int8_t {var_name}[{flat.size}] = {{")
    # 16 values per line
    for i in range(0, flat.size, 16):
        chunk = flat[i:i+16].tolist()
        lines.append(indent + ", ".join(str(int(v)) for v in chunk) + ("," if i+16 < flat.size else ""))
    lines.append("};")
    return "\n".join(lines)

def format_c_shape_macros(prefix: str, shape: List[int]) -> str:
    out = []
    out.append(f"// Shape: {shape}")
    for i, d in enumerate(shape):
        out.append(f"#define {prefix}_DIM{i} {int(d)}")
    out.append(f"#define {prefix}_NUMEL {int(np.prod(shape))}")
    return "\n".join(out)


# ----------------------------
# Activation histogram collector (optional / for reporting)
# ----------------------------

@dataclass
class RunningHistogram:
    bins: int
    min_val: float
    max_val: float
    counts: np.ndarray

    @staticmethod
    def create(bins: int, min_val: float, max_val: float) -> "RunningHistogram":
        return RunningHistogram(bins=bins, min_val=min_val, max_val=max_val, counts=np.zeros((bins,), dtype=np.int64))

    def update(self, x: np.ndarray):
        x = x.astype(np.float32).ravel()
        if x.size == 0:
            return
        # Clip to histogram range
        x = np.clip(x, self.min_val, self.max_val)
        hist, _ = np.histogram(x, bins=self.bins, range=(self.min_val, self.max_val))
        self.counts += hist.astype(np.int64)

    def percentile_absmax(self, pct: float = 99.9) -> float:
        """
        Estimate symmetric absmax at percentile from histogram of values (not absolute values).
        We'll approximate by looking at tails.
        """
        total = self.counts.sum()
        if total <= 0:
            return 0.0
        cdf = np.cumsum(self.counts) / total
        # Find lower and upper quantiles
        low_q = (100.0 - pct) / 200.0
        high_q = 1.0 - low_q
        # Bin centers
        edges = np.linspace(self.min_val, self.max_val, self.bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        lo_idx = int(np.searchsorted(cdf, low_q))
        hi_idx = int(np.searchsorted(cdf, high_q))
        lo = float(centers[np.clip(lo_idx, 0, self.bins - 1)])
        hi = float(centers[np.clip(hi_idx, 0, self.bins - 1)])
        return max(abs(lo), abs(hi))


# ----------------------------
# TFLite parsing / rebuilding
# ----------------------------

@dataclass
class WeightTensorInfo:
    subgraph_idx: int
    tensor_idx: int
    tensor_name: str
    shape: List[int]
    buffer_idx: int

def load_tflite_model_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def model_to_object(model_bytes: bytes) -> schema_fb.ModelT:
    model = schema_fb.Model.GetRootAsModel(model_bytes, 0)
    # Convert to mutable object tree
    model_obj = schema_fb.ModelT.InitFromObj(model)
    return model_obj

def object_to_model_bytes(model_obj: schema_fb.ModelT) -> bytes:
    builder = flatbuffers.Builder(1024 * 1024)  # Pre-allocate 1MB
    off = model_obj.Pack(builder)
    builder.Finish(off, b"TFL3")  # TFLite file identifier
    return bytes(builder.Output())

def find_float32_constant_weight_tensors(model_obj: schema_fb.ModelT) -> List[WeightTensorInfo]:
    """
    Heuristic: weight tensors are FLOAT32 tensors whose buffer contains data.
    """
    out: List[WeightTensorInfo] = []
    for sg_i, sg in enumerate(model_obj.subgraphs):
        for t_i, t in enumerate(sg.tensors):
            if t.type != schema_fb.TensorType.FLOAT32:
                continue
            buf_i = int(t.buffer)
            if buf_i <= 0 or buf_i >= len(model_obj.buffers):
                continue
            b = model_obj.buffers[buf_i]
            data = b.data
            if data is None or len(data) == 0:
                continue
            name = t.name.decode("utf-8") if isinstance(t.name, (bytes, bytearray)) else (t.name or f"tensor_{t_i}")
            shape = [int(d) for d in t.shape] if t.shape is not None else []
            out.append(WeightTensorInfo(subgraph_idx=sg_i, tensor_idx=t_i, tensor_name=name, shape=shape, buffer_idx=buf_i))
    return out

def read_buffer_as_float32(model_obj: schema_fb.ModelT, buffer_idx: int, shape: List[int]) -> np.ndarray:
    b = model_obj.buffers[buffer_idx]
    raw = bytes(b.data)
    arr = np.frombuffer(raw, dtype=np.float32)
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)
    return arr.copy()

def write_buffer_from_float32(model_obj: schema_fb.ModelT, buffer_idx: int, arr_f32: np.ndarray):
    arr_f32 = np.asarray(arr_f32, dtype=np.float32)
    model_obj.buffers[buffer_idx].data = arr_f32.tobytes()


# ----------------------------
# Representative data loading / generation
# ----------------------------

def get_input_details(interp: tf.lite.Interpreter):
    return interp.get_input_details()

def set_inputs(interp: tf.lite.Interpreter, inputs: List[np.ndarray]):
    in_details = interp.get_input_details()
    assert len(inputs) == len(in_details), f"Expected {len(in_details)} inputs, got {len(inputs)}"
    for i, (d, x) in enumerate(zip(in_details, inputs)):
        x = np.asarray(x)
        # Cast to expected dtype
        if x.dtype != d["dtype"]:
            x = x.astype(d["dtype"])
        # Resize if needed (rare; only if input is dynamic and provided shape differs)
        if tuple(x.shape) != tuple(d["shape"]):
            # If input has -1 dims, allow resizing
            try:
                interp.resize_tensor_input(d["index"], x.shape, strict=False)
                interp.allocate_tensors()
            except Exception:
                # If cannot resize, attempt reshape if same numel
                if np.prod(x.shape) == np.prod(d["shape"]):
                    x = x.reshape(d["shape"])
                else:
                    raise
        interp.set_tensor(d["index"], x)

def load_representative_samples(interp: tf.lite.Interpreter,
                                rep_npy_dir: Optional[str],
                                num_samples: int,
                                seed: int) -> List[List[np.ndarray]]:
    """
    Returns list of samples; each sample is a list of input arrays (one per model input).
    """
    rng = np.random.default_rng(seed)
    in_details = interp.get_input_details()

    samples: List[List[np.ndarray]] = []

    if rep_npy_dir:
        files = [os.path.join(rep_npy_dir, f) for f in os.listdir(rep_npy_dir) if f.lower().endswith(".npy")]
        files.sort()
        if len(files) == 0:
            raise RuntimeError(f"No .npy files found in {rep_npy_dir}")

        for fp in files[:num_samples]:
            obj = np.load(fp, allow_pickle=True)
            if len(in_details) == 1 and isinstance(obj, np.ndarray):
                samples.append([obj])
            elif isinstance(obj, dict):
                # dict mapping by input name or index-like keys
                one = []
                for d in in_details:
                    key = d["name"]
                    if key in obj:
                        one.append(obj[key])
                    else:
                        raise RuntimeError(f"{fp}: missing key '{key}' for multi-input model.")
                samples.append(one)
            else:
                # Try object array/list for multi-input
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    lst = list(obj.tolist())
                    if len(lst) != len(in_details):
                        raise RuntimeError(f"{fp}: expected {len(in_details)} inputs, got {len(lst)}")
                    samples.append([np.asarray(x) for x in lst])
                else:
                    raise RuntimeError(f"{fp}: unrecognized format for multi-input model.")
        return samples

    # Random generation
    for _ in range(num_samples):
        one = []
        for d in in_details:
            shape = tuple(int(s) for s in d["shape"])
            dtype = d["dtype"]
            # If dynamic (-1), choose 1 for that dim
            shape = tuple(1 if s < 0 else s for s in shape)
            if np.issubdtype(dtype, np.floating):
                x = rng.standard_normal(size=shape, dtype=np.float32).astype(dtype)
            elif dtype == np.int32:
                x = rng.integers(low=0, high=128, size=shape, dtype=np.int32)
            elif dtype == np.int64:
                x = rng.integers(low=0, high=128, size=shape, dtype=np.int64)
            elif dtype == np.uint8:
                x = rng.integers(low=0, high=255, size=shape, dtype=np.uint8)
            else:
                # fallback
                x = rng.standard_normal(size=shape, dtype=np.float32).astype(dtype)
            one.append(x)
        samples.append(one)
    return samples


# ----------------------------
# Calibration + comparison
# ----------------------------

def run_inference_collect(interp: tf.lite.Interpreter,
                          samples: List[List[np.ndarray]],
                          collect_activations: bool,
                          act_bins: int,
                          act_range: float) -> Tuple[List[np.ndarray], Dict[int, RunningHistogram], Dict[int, Tuple[float,float]]]:
    """
    Returns:
      outputs_list: list of model outputs (flattened list concatenated if multiple outputs)
      act_hists: histogram per tensor_index (only if collect_activations)
      act_minmax: per tensor_index (min, max) across runs
    """
    tensor_details = interp.get_tensor_details()
    float_tensor_indices = [d["index"] for d in tensor_details if d["dtype"] == np.float32]

    act_hists: Dict[int, RunningHistogram] = {}
    act_minmax: Dict[int, Tuple[float,float]] = {}

    outputs_list: List[np.ndarray] = []

    out_details = interp.get_output_details()

    if collect_activations:
        for tidx in float_tensor_indices:
            act_hists[tidx] = RunningHistogram.create(bins=act_bins, min_val=-act_range, max_val=act_range)
            act_minmax[tidx] = (np.inf, -np.inf)

    for one in samples:
        set_inputs(interp, one)
        interp.invoke()

        # Outputs
        outs = []
        for od in out_details:
            y = interp.get_tensor(od["index"])
            outs.append(y.astype(np.float32).ravel())
        outputs_list.append(np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0])

        if collect_activations:
            for tidx in float_tensor_indices:
                try:
                    x = interp.get_tensor(tidx)
                except ValueError:
                    # Some tensors may not be readable depending on delegate/internal state
                    continue
                x = x.astype(np.float32)
                mn = float(np.min(x)) if x.size else np.inf
                mx = float(np.max(x)) if x.size else -np.inf
                prev_mn, prev_mx = act_minmax.get(tidx, (np.inf, -np.inf))
                act_minmax[tidx] = (min(prev_mn, mn), max(prev_mx, mx))
                act_hists[tidx].update(x)

    return outputs_list, act_hists, act_minmax

def compare_outputs(ref: List[np.ndarray], test: List[np.ndarray]) -> Dict[str, float]:
    """
    Simple numeric comparison across a list of output vectors.
    """
    assert len(ref) == len(test)
    diffs = []
    ref_all = []
    test_all = []
    for r, t in zip(ref, test):
        r = r.astype(np.float32).ravel()
        t = t.astype(np.float32).ravel()
        n = min(r.size, t.size)
        r = r[:n]
        t = t[:n]
        ref_all.append(r)
        test_all.append(t)
        diffs.append(t - r)

    R = np.concatenate(ref_all, axis=0)
    T = np.concatenate(test_all, axis=0)
    D = np.concatenate(diffs, axis=0)

    mae = float(np.mean(np.abs(D)))
    rmse = float(np.sqrt(np.mean(D * D)))
    denom = float(np.linalg.norm(R) * np.linalg.norm(T) + 1e-12)
    cosine = float(np.dot(R, T) / denom)

    return {"mae": mae, "rmse": rmse, "cosine": cosine}


# ----------------------------
# Main quantization logic
# ----------------------------

def quantize_weights(model_obj: schema_fb.ModelT,
                     weight_infos: List[WeightTensorInfo],
                     mode: str,
                     percentile: Optional[float],
                     out_report_path: Optional[str]) -> Tuple[Dict[str, QuantParams], Dict[str, np.ndarray]]:
    """
    Returns:
      qparams_by_tensor_name
      int8_weights_by_tensor_name (same shapes as original tensors)
    """
    # Load all weights
    weights: Dict[str, np.ndarray] = {}
    maxabs_by_name: Dict[str, float] = {}

    for wi in weight_infos:
        w = read_buffer_as_float32(model_obj, wi.buffer_idx, wi.shape)
        weights[wi.tensor_name] = w
        maxabs = float(np.max(np.abs(w))) if w.size else 0.0
        maxabs_by_name[wi.tensor_name] = maxabs

    qparams_by_name: Dict[str, QuantParams] = {}
    int8_by_name: Dict[str, np.ndarray] = {}

    if mode == "global":
        # One global scale across ALL weight tensors.
        # Optionally use percentile trimming per tensor -> take max of per-tensor trimmed.
        if percentile is None:
            global_maxabs = max(maxabs_by_name.values()) if maxabs_by_name else 0.0
        else:
            # Percentile on weights: approximate by np.percentile(|w|, pct)
            per = []
            for name, w in weights.items():
                if w.size == 0:
                    per.append(0.0)
                    continue
                v = np.percentile(np.abs(w).ravel(), percentile)
                per.append(float(v))
            global_maxabs = max(per) if per else 0.0

        qp = choose_symmetric_int8_params(global_maxabs)

        for name, w in weights.items():
            q, _dq = quantize_dequantize_float_to_int8(w, qp)
            qparams_by_name[name] = qp
            int8_by_name[name] = q

    elif mode == "per_tensor":
        for name, w in weights.items():
            if w.size == 0:
                qp = QuantParams(scale=1.0, zero_point=0)
            else:
                if percentile is None:
                    maxabs = float(np.max(np.abs(w)))
                else:
                    maxabs = float(np.percentile(np.abs(w).ravel(), percentile))
                qp = choose_symmetric_int8_params(maxabs)
            q, _dq = quantize_dequantize_float_to_int8(w, qp)
            qparams_by_name[name] = qp
            int8_by_name[name] = q
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if out_report_path:
        report = {
            "mode": mode,
            "percentile_abs": percentile,
            "tensors": {
                name: {
                    "scale": qp.scale,
                    "zero_point": qp.zero_point,
                    "qmin": qp.qmin,
                    "qmax": qp.qmax,
                    "maxabs_weight": maxabs_by_name.get(name, None),
                    "shape": list(weights[name].shape),
                }
                for name, qp in qparams_by_name.items()
            }
        }
        with open(out_report_path, "w") as f:
            json.dump(report, f, indent=2)

    return qparams_by_name, int8_by_name

def apply_fake_quant_weights_inplace(model_obj: schema_fb.ModelT,
                                     weight_infos: List[WeightTensorInfo],
                                     qparams_by_name: Dict[str, QuantParams],
                                     int8_by_name: Dict[str, np.ndarray]):
    """
    Replace float32 weight buffers with dequantized float32 values (fake quantization).
    Model remains float32 and should still execute in TFLite.
    """
    for wi in weight_infos:
        name = wi.tensor_name
        qp = qparams_by_name[name]
        q = int8_by_name[name]
        dq = (q.astype(np.float32) - qp.zero_point) * qp.scale
        write_buffer_from_float32(model_obj, wi.buffer_idx, dq.reshape(wi.shape))

def write_cpp_headers(out_dir: str,
                      qparams_by_name: Dict[str, QuantParams],
                      int8_by_name: Dict[str, np.ndarray]):
    os.makedirs(out_dir, exist_ok=True)

    # One header per tensor (easy to include selectively)
    for name, q in int8_by_name.items():
        qp = qparams_by_name[name]
        ident = safe_c_identifier(name)

        shape = list(q.shape)
        prefix = ident.upper()

        header_path = os.path.join(out_dir, f"{ident}.h")
        with open(header_path, "w") as f:
            f.write("#pragma once\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"// Original tensor name: {name}\n\n")
            f.write(format_c_shape_macros(prefix, shape) + "\n\n")
            f.write(f"// Quantization params (symmetric int8)\n")
            f.write(f"static const float {ident}_scale = {qp.scale:.10g}f;\n")
            f.write(f"static const int32_t {ident}_zero_point = {qp.zero_point};\n\n")
            f.write(format_c_array_int8(f"{ident}_q", q) + "\n")

    # Also write an index header
    index_path = os.path.join(out_dir, "all_weights.h")
    with open(index_path, "w") as f:
        f.write("#pragma once\n\n")
        for name in int8_by_name.keys():
            ident = safe_c_identifier(name)
            f.write(f'#include "{ident}.h"\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite_in", required=True, help="Input float32 TFLite model")
    ap.add_argument("--mode", choices=["global", "per_tensor"], default="global",
                    help="Weight quantization mode")
    ap.add_argument("--percentile_abs", type=float, default=None,
                    help="If set (e.g. 99.9), use percentile(|w|) instead of max(|w|) to reduce outlier impact")
    ap.add_argument("--out_tflite_fakeq", default=None,
                    help="Write fake-quantized float32 TFLite here (quantize->dequantize weights)")
    ap.add_argument("--out_header_dir", default=None,
                    help="Directory to write C++ headers containing int8 weights + scales")
    ap.add_argument("--out_report_json", default="quant_report.json",
                    help="Write quantization params report JSON")
    ap.add_argument("--compare", action="store_true",
                    help="Run original vs fake-quant model comparison on representative samples")
    ap.add_argument("--collect_activation_hists", action="store_true",
                    help="Collect activation histograms (for analysis; not required for weight-only quant)")
    ap.add_argument("--act_bins", type=int, default=2048, help="Histogram bins for activations")
    ap.add_argument("--act_range", type=float, default=10.0,
                    help="Activation histogram clipping range [-act_range, act_range]")

    ap.add_argument("--rep_npy_dir", default=None,
                    help="Directory of .npy representative inputs. If omitted, random inputs are generated.")
    ap.add_argument("--num_samples", type=int, default=64, help="Number of representative samples")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    # Load and parse model
    model_bytes = load_tflite_model_bytes(args.tflite_in)
    model_obj = model_to_object(model_bytes)

    weight_infos = find_float32_constant_weight_tensors(model_obj)
    if len(weight_infos) == 0:
        print("No FLOAT32 constant weight tensors found (buffers with data). Nothing to do.")
        return

    print(f"Found {len(weight_infos)} FLOAT32 constant tensors (treated as weights).")
    for wi in weight_infos[:10]:
        print(f"  - {wi.tensor_name} shape={wi.shape} buffer={wi.buffer_idx}")
    if len(weight_infos) > 10:
        print(f"  ... and {len(weight_infos)-10} more")

    # Build interpreters
    interp_ref = tf.lite.Interpreter(model_path=args.tflite_in)
    interp_ref.allocate_tensors()

    samples = load_representative_samples(interp_ref, args.rep_npy_dir, args.num_samples, args.seed)
    print(f"Loaded/generated {len(samples)} representative samples.")

    # Optional: reference outputs / activation hists
    ref_outputs, ref_hists, ref_minmax = run_inference_collect(
        interp_ref,
        samples,
        collect_activations=args.collect_activation_hists,
        act_bins=args.act_bins,
        act_range=args.act_range
    )

    # Quantize weights (params + int8 arrays)
    qparams_by_name, int8_by_name = quantize_weights(
        model_obj=model_obj,
        weight_infos=weight_infos,
        mode=args.mode,
        percentile=args.percentile_abs,
        out_report_path=args.out_report_json
    )
    print(f"Wrote quant params report: {args.out_report_json}")

    # Export C++ headers
    if args.out_header_dir:
        write_cpp_headers(args.out_header_dir, qparams_by_name, int8_by_name)
        print(f"Wrote C++ headers to: {args.out_header_dir}")

    # Fake-quantize weights into a runnable float32 TFLite model
    fakeq_path = args.out_tflite_fakeq
    if fakeq_path or args.compare:
        # Make a fresh copy from original bytes to avoid compounding if re-run
        model_obj_fake = model_to_object(model_bytes)

        weight_infos_fake = find_float32_constant_weight_tensors(model_obj_fake)
        apply_fake_quant_weights_inplace(model_obj_fake, weight_infos_fake, qparams_by_name, int8_by_name)

        fake_bytes = object_to_model_bytes(model_obj_fake)

        if fakeq_path:
            with open(fakeq_path, "wb") as f:
                f.write(fake_bytes)
            print(f"Wrote fake-quantized TFLite: {fakeq_path}")

        if args.compare:
            # Compare via Interpreter - load from file if available, otherwise from bytes
            if fakeq_path:
                interp_test = tf.lite.Interpreter(model_path=fakeq_path)
            else:
                interp_test = tf.lite.Interpreter(model_content=fake_bytes)
            interp_test.allocate_tensors()

            test_outputs, test_hists, test_minmax = run_inference_collect(
                interp_test,
                samples,
                collect_activations=False,
                act_bins=args.act_bins,
                act_range=args.act_range
            )

            metrics = compare_outputs(ref_outputs, test_outputs)
            print("\n=== Output comparison: original vs fake-quant ===")
            print(json.dumps(metrics, indent=2))

    # Optional activation histogram summary
    if args.collect_activation_hists:
        # Just print a few tensors' min/max and a percentile absmax estimate
        print("\n=== Activation histogram summary (reference model) ===")
        # Sort by tensor index
        items = sorted(ref_minmax.items(), key=lambda kv: kv[0])
        for tidx, (mn, mx) in items[:20]:
            est = ref_hists[tidx].percentile_absmax(99.9) if tidx in ref_hists else None
            print(f"tensor {tidx:4d}: min={mn:.5g} max={mx:.5g} p99.9_absmax≈{est:.5g}")
        if len(items) > 20:
            print(f"... and {len(items)-20} more tensors")

    print("\nDone.")


if __name__ == "__main__":
    main()
   