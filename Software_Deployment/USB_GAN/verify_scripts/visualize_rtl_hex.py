#!/usr/bin/env python3
"""
visualize_rtl_hex.py — Visualisasi output hex RTL per layer
============================================================
Solusi untuk masalah "loreng abu-abu":
  1. Signed Q8.8 fix     : interpret hex sebagai int16 (bukan uint16)
  2. CHW→HWC reorder     : RTL dump dalam format [C,H,W], PIL butuh [H,W,C]
  3. Normalisasi per-layer: tiap layer punya range berbeda, normalize sebelum display
  4. Output layer (tanh)  : map [-1,1] → [0,255]

Usage:
    python3 visualize_rtl_hex.py                    # semua layer
    python3 visualize_rtl_hex.py --layer 11         # layer tertentu
    python3 visualize_rtl_hex.py --compare          # RTL vs PyTorch ref
"""

import numpy as np
import os
import argparse
from PIL import Image

# ── Konfigurasi shape per layer (C, H, W) ─────────────────────────────────────
LAYER_SHAPES = {
    0:  (32,  32, 32),   # E1 output
    1:  (64,  16, 16),   # E2 output
    2:  (128,  8,  8),   # E3 output
    3:  (256,  4,  4),   # E4 output
    4:  (512,  2,  2),   # E5 output
    5:  (512,  2,  2),   # BOT output
    6:  (256,  4,  4),   # D1 output
    7:  (128,  8,  8),   # D2 output
    8:  (64,  16, 16),   # D3 output
    9:  (32,  32, 32),   # D4 output
    10: (32,  64, 64),   # D5 output
    11: (3,   64, 64),   # OUT output ← ini yang jadi gambar akhir
}

LAYER_NAMES = {
    0: "E1", 1: "E2", 2: "E3", 3: "E4", 4: "E5",
    5: "BOT",
    6: "D1", 7: "D2", 8: "D3", 9: "D4", 10: "D5",
    11: "OUT"
}

FRAC_W = 8  # Q8.8 fixed-point


def load_hex_layer(layer_idx: int, hex_dir: str = "outputs") -> np.ndarray | None:
    """
    Load file hex RTL untuk satu layer.
    Return: numpy array float32 dengan shape (C, H, W), atau None jika gagal.
    """
    fname = os.path.join(hex_dir, f"rtl_layer_{layer_idx}.hex")
    if not os.path.exists(fname):
        print(f"  [WARN] {fname} tidak ditemukan")
        return None

    raw = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # [FIX-1] Interpret sebagai int16 (signed), BUKAN uint16
            val_uint = int(line, 16) & 0xFFFF
            val_int  = val_uint if val_uint < 0x8000 else val_uint - 0x10000
            raw.append(val_int)

    c, h, w = LAYER_SHAPES[layer_idx]
    expected = c * h * w

    if len(raw) != expected:
        print(f"  [WARN] Layer {layer_idx}: expect {expected} values, got {len(raw)}")
        # Pad atau truncate
        if len(raw) < expected:
            raw.extend([0] * (expected - len(raw)))
        else:
            raw = raw[:expected]

    # [FIX-2] Reshape dalam format CHW
    arr = np.array(raw, dtype=np.float32)
    # Konversi dari Q8.8 fixed-point ke float
    arr = arr / (2 ** FRAC_W)
    arr = arr.reshape(c, h, w)  # shape: (C, H, W)

    return arr


def chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    """CHW → HWC untuk display"""
    return np.transpose(arr, (1, 2, 0))   # (C,H,W) → (H,W,C)


def normalize_to_uint8(arr: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    [FIX-3] Normalisasi nilai aktivasi ke range [0, 255].
    mode:
      'minmax' → stretch min..max ke 0..255 (cocok untuk feature map)
      'tanh'   → map [-1,1] ke [0,255]       (cocok untuk layer OUT)
      'center' → center pada 0, abs scale    (cocok untuk debug BN output)
    """
    if mode == "tanh":
        arr_clip = np.clip(arr, -1.0, 1.0)
        return ((arr_clip + 1.0) * 127.5).astype(np.uint8)
    elif mode == "minmax":
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    elif mode == "center":
        vabs = np.abs(arr).max()
        if vabs < 1e-6:
            return np.full_like(arr, 128, dtype=np.uint8)
        return ((arr / vabs + 1.0) * 127.5).astype(np.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def visualize_layer(layer_idx: int, save_dir: str = "outputs/viz",
                    hex_dir: str = "outputs"):
    """Visualisasi satu layer: tiap channel + composite RGB."""
    os.makedirs(save_dir, exist_ok=True)

    arr = load_hex_layer(layer_idx, hex_dir)
    if arr is None:
        return

    c, h, w   = arr.shape
    name      = LAYER_NAMES[layer_idx]

    # ── Pilih mode normalisasi ─────────────────────────────────────────────────
    if layer_idx == 11:
        norm_mode = "tanh"    # Output layer: tanh → [-1,1] → [0,255]
    else:
        norm_mode = "minmax"  # Feature map: stretch ke full range

    print(f"\n[Layer {layer_idx} | {name}]  shape=({c},{h},{w})"
          f"  min={arr.min():.4f}  max={arr.max():.4f}"
          f"  mean={arr.mean():.4f}  std={arr.std():.4f}")

    # ── Gambar akhir (layer 11 = RGB output) ──────────────────────────────────
    if layer_idx == 11 and c == 3:
        hwc       = chw_to_hwc(arr)          # [FIX-2]
        img_uint8 = normalize_to_uint8(hwc, mode="tanh")   # [FIX-3]
        img       = Image.fromarray(img_uint8, mode="RGB")
        out_path  = os.path.join(save_dir, f"layer_{layer_idx}_{name}_output.png")
        img.save(out_path)
        print(f"  ✓ Saved final output image: {out_path}")
        return

    # ── Feature map: tampilkan grid channel ───────────────────────────────────
    n_show = min(c, 16)      # Tampilkan max 16 channel pertama
    grid_cols = 4
    grid_rows = (n_show + grid_cols - 1) // grid_cols

    cell_h, cell_w = max(h, 32), max(w, 32)
    canvas_h = grid_rows * cell_h + (grid_rows - 1) * 2
    canvas_w = grid_cols * cell_w + (grid_cols - 1) * 2
    canvas   = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for i in range(n_show):
        row = i // grid_cols
        col = i %  grid_cols
        ch_data  = arr[i]                               # shape (H, W)
        ch_uint8 = normalize_to_uint8(ch_data, norm_mode)
        ch_resized = np.array(
            Image.fromarray(ch_uint8).resize((cell_w, cell_h), Image.NEAREST)
        )
        y0 = row * (cell_h + 2)
        x0 = col * (cell_w + 2)
        canvas[y0:y0+cell_h, x0:x0+cell_w] = ch_resized

    img      = Image.fromarray(canvas, mode="L")
    out_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}_channels.png")
    img.save(out_path)
    print(f"  ✓ Saved channel grid ({n_show}/{c} ch): {out_path}")


def compare_rtl_vs_ref(layer_idx: int,
                       rtl_dir: str = "outputs",
                       ref_dir: str = "outputs/py_ref",
                       save_dir: str = "outputs/viz"):
    """Side-by-side comparison RTL vs PyTorch reference."""
    os.makedirs(save_dir, exist_ok=True)

    arr_rtl = load_hex_layer(layer_idx, rtl_dir)
    arr_ref = load_hex_layer(layer_idx, ref_dir)

    if arr_rtl is None or arr_ref is None:
        print(f"  [SKIP] Layer {layer_idx}: file tidak lengkap untuk compare")
        return

    # Hitung error
    err    = np.abs(arr_rtl - arr_ref)
    mae    = err.mean()
    maxerr = err.max()
    name   = LAYER_NAMES[layer_idx]
    print(f"  Compare Layer {layer_idx} {name}: MAE={mae:.5f}  MaxErr={maxerr:.5f}")

    # Buat gambar error map (normalized)
    err_norm = normalize_to_uint8(err, "minmax")
    c, h, w  = err_norm.shape
    err_2d   = err_norm.mean(axis=0)   # Average across channels
    err_img  = Image.fromarray(err_2d.astype(np.uint8)).resize((256, 256), Image.NEAREST)

    out_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}_error.png")
    err_img.save(out_path)
    print(f"  ✓ Error map saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer index to visualize (-1 = semua)")
    parser.add_argument("--compare", action="store_true",
                        help="Bandingkan RTL vs PyTorch reference")
    parser.add_argument("--rtl-dir",  default="outputs")
    parser.add_argument("--ref-dir",  default="outputs/py_ref")
    parser.add_argument("--save-dir", default="outputs/viz")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    layers = [args.layer] if args.layer >= 0 else list(LAYER_SHAPES.keys())

    print("=" * 60)
    print("  RTL Hex → Image Visualizer")
    print("  Fix: signed Q8.8, CHW→HWC reorder, proper normalization")
    print("=" * 60)

    for l in layers:
        if args.compare:
            compare_rtl_vs_ref(l, args.rtl_dir, args.ref_dir, args.save_dir)
        else:
            visualize_layer(l, args.save_dir, args.rtl_dir)

    print(f"\nSelesai. Output disimpan di: {args.save_dir}/")

    # Quick sanity check untuk layer 11
    if 11 in layers and not args.compare:
        arr = load_hex_layer(11, args.rtl_dir)
        if arr is not None:
            print(f"\n[Sanity Check Layer 11]")
            print(f"  Range      : [{arr.min():.4f}, {arr.max():.4f}]")
            print(f"  Harusnya   : ~ [-1.0, 1.0] (output tanh)")
            if arr.max() <= 1.05 and arr.min() >= -1.05:
                print(f"  Status     : ✓ Range tanh OK")
            else:
                print(f"  Status     : ✗ Range out of tanh bounds! Cek aktivasi layer 11")


if __name__ == "__main__":
    main()
