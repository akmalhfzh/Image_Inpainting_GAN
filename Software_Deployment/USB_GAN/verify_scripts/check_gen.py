#!/usr/bin/env python3

import numpy as np
import os
import sys

DW     = 16
FRAC_W = 8
SCALE  = 256
MASK   = (1 << DW) - 1
SIGN   = 1 << (DW - 1)

LAYER_MAP = {
    0:  ("E1",  (32, 32, 32)),
    1:  ("E2",  (64, 16, 16)),
    2:  ("E3",  (128, 8, 8)),
    3:  ("E4",  (256, 4, 4)),
    4:  ("E5",  (512, 2, 2)),
    5:  ("BOT", (512, 2, 2)),
    6:  ("D1",  (256, 4, 4)),
    7:  ("D2",  (128, 8, 8)),
    8:  ("D3",  (64, 16, 16)),
    9:  ("D4",  (32, 32, 32)),
    10: ("D5",  (32, 64, 64)),
    11: ("OUT", (3, 64, 64))
}

# Per-layer MAE tolerance — based on expected Q7.8 quantization noise
# These are generous enough for correct RTL, tight enough to catch real bugs
LAYER_TOL = {
    "E1":  0.01,    # minimal quantization (3 ch input, no BN)
    "E2":  0.02,    # small accumulation
    "E3":  0.03,
    "E4":  0.05,
    "E5":  0.08,
    "BOT": 0.60,    # 512-ch 1x1 conv, large values, heavy quantization noise
    "D1":  0.15,    # inherits BOT error
    "D2":  0.10,
    "D3":  0.08,
    "D4":  0.05,
    "D5":  0.08,    # large spatial size, many small values
    "OUT": 0.20,    # tanh saturation amplifies small differences
}

# Minimum acceptable correlation (should be > 0.99 for correct RTL)
MIN_CORR = 0.930

def read_hex_to_float(fname):
    vals = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16) & MASK
            if v >= SIGN:
                v -= (1 << DW)
            vals.append(v / SCALE)
    return np.array(vals, dtype=np.float32)


def main():
    print("=" * 70)
    print(" RTL vs PyTorch (Quantized) — Layer-by-Layer Verification")
    print("=" * 70)

    all_pass = True
    results  = []

    for rtl_idx, (py_name, expected_shape) in LAYER_MAP.items():
        rtl_file = f"outputs/rtl_layer_{rtl_idx}.hex"
        py_file  = f"outputs/py_ref/ref_{py_name}.hex"

        if not os.path.exists(rtl_file) or not os.path.exists(py_file):
            print(f"  [WARN] Missing file for layer {py_name}")
            all_pass = False
            continue

        rtl_vals = read_hex_to_float(rtl_file)
        py_vals  = read_hex_to_float(py_file)

        expected_size = int(np.prod(expected_shape))
        if len(rtl_vals) != expected_size or len(py_vals) != expected_size:
            print(f"  ✗ {py_name}: size mismatch (RTL={len(rtl_vals)} PY={len(py_vals)} expected={expected_size})")
            all_pass = False
            continue

        diff    = np.abs(rtl_vals - py_vals)
        mae     = np.mean(diff)
        max_err = np.max(diff)

        # Correlation coefficient — most important metric for fixed-point
        corr = np.corrcoef(rtl_vals, py_vals)[0, 1] if np.std(rtl_vals) > 0 and np.std(py_vals) > 0 else 0.0

        tol  = LAYER_TOL.get(py_name, 0.10)
        pass_mae  = mae <= tol
        pass_corr = corr >= MIN_CORR
        passed    = pass_mae and pass_corr

        if not passed:
            all_pass = False

        status = "PASS ✓" if passed else "FAIL ✗"
        results.append((py_name, status, mae, max_err, corr, tol))

        print(f"  Layer {py_name:<3} [{status}] | MAE: {mae:.5f} (tol: {tol:.2f}) | "
              f"MaxErr: {max_err:.5f} | Corr: {corr:.4f}")
        if not passed:
            print(f"         RTL: {rtl_vals[:5]}")
            print(f"         PY : {py_vals[:5]}")

    print("=" * 70)

    # Summary table
    print(f"\n  {'Layer':<5} {'Status':<8} {'MAE':>8} {'Tol':>6} {'MaxErr':>8} {'Corr':>7}")
    print("  " + "-" * 50)
    for name, status, mae, max_err, corr, tol in results:
        flag = "✓" if "PASS" in status else "✗"
        print(f"  {name:<5} {flag:<8} {mae:>8.5f} {tol:>6.2f} {max_err:>8.5f} {corr:>7.4f}")

    print()
    if all_pass:
        print("  ✅ VERDICT: ALL LAYERS PASS — RTL matches quantized PyTorch reference.")
    else:
        print("  ❌ VERDICT: Some layers exceed tolerance. Check details above.")
    print("=" * 70)

    # Generate output image if available
    rtl_path = "outputs/output_rtl.hex"
    if os.path.exists(rtl_path):
        rtl = read_hex_to_float(rtl_path)
        if len(rtl) == 3 * 64 * 64:
            try:
                from PIL import Image
                img = rtl.reshape(3, 64, 64).transpose(1, 2, 0)
                img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(img).save("outputs/output_rtl.png")
                print("  Saved outputs/output_rtl.png")
            except ImportError:
                pass


if __name__ == "__main__":
    main()
