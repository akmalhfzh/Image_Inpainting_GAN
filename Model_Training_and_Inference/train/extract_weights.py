#!/usr/bin/env python3
"""
extract_weights.py
──────────────────
Parse generator .bin weight file → hex files siap diload ke RTL Verilog.

Format .bin (dari save_weights_bin):
  Raw float32, tulis berurutan sesuai model.state_dict().items()
  Tidak ada header, tidak ada delimiter — pure bytes.

Output per layer:
  layer_XX_weight.hex   → untuk conv2d_core / convtranspose2d_core
  layer_XX_bn_scale.hex → untuk batchnorm2d  (precomputed gamma/sqrt(var+eps))
  layer_XX_bn_offset.hex → untuk batchnorm2d (precomputed beta - mean*scale)

Usage:
  python3 extract_weights.py gw_latest.bin
  python3 extract_weights.py generator_epoch662.bin
"""

import numpy as np
import struct
import os
import sys

# ── Fixed-point config ────────────────────────────────────────────────────────
DW     = 16
FRAC_W = 8
SCALE  = 1 << FRAC_W   # 256
EPS    = 0.8            # BatchNorm2d eps di kode lo: nn.BatchNorm2d(out_feat, 0.8)

# ── Generator state_dict layout ───────────────────────────────────────────────
# Diambil dari kode Generator persis, urutan Sequential:
#   downsample(3,  64,  normalize=False)  → Conv2d, LeakyReLU          (no BN)
#   downsample(64, 64,  normalize=True)   → Conv2d, BN, LeakyReLU
#   downsample(64, 128, normalize=True)   → Conv2d, BN, LeakyReLU
#   downsample(128,256, normalize=True)   → Conv2d, BN, LeakyReLU
#   downsample(256,512, normalize=True)   → Conv2d, BN, LeakyReLU
#   Conv2d(512, 4000, 1)                  → bottleneck (no BN)
#   upsample(4000,512)                    → ConvT, BN, ReLU
#   upsample(512, 256)                    → ConvT, BN, ReLU
#   upsample(256, 128)                    → ConvT, BN, ReLU
#   upsample(128, 64)                     → ConvT, BN, ReLU
#   Conv2d(64, 3, 3, 1, 1)               → output head (no BN)
#   Tanh()                                → no params
#
# state_dict key order (PyTorch nn.Sequential dengan index):
#   model.0.weight                    Conv2d E1 weight
#                                     (no BN karena normalize=False)
#   model.1.weight, .bias, .running_mean, .running_var, .num_batches_tracked
#                                     BN E2
#   model.2.weight                    Conv2d E2
#   model.3.weight,.bias,.running_mean,.running_var,.num_batches_tracked   BN E2
#   ...
#
# CATATAN: PyTorch Sequential index mengikuti urutan LAYER di list,
# bukan urutan blok. Kita perlu map index ke layer logic.

LAYOUT = [
    # (name,        type,      shape,              has_bias)
    # ── ENCODER ──────────────────────────────────────────────────────────────
    ("E1_conv",     "conv",    (64,  3,   4, 4),   True),
    # normalize=False → tidak ada BN di E1
    ("E2_conv",     "conv",    (64,  64,  4, 4),   True),
    ("E2_bn",       "bn",      (64,),               None),   # gamma,beta,mean,var,nb
    ("E3_conv",     "conv",    (128, 64,  4, 4),   True),
    ("E3_bn",       "bn",      (128,),              None),
    ("E4_conv",     "conv",    (256, 128, 4, 4),   True),
    ("E4_bn",       "bn",      (256,),              None),
    ("E5_conv",     "conv",    (512, 256, 4, 4),   True),
    ("E5_bn",       "bn",      (512,),              None),
    # ── BOTTLENECK ────────────────────────────────────────────────────────────
    ("BOT_conv",    "conv",    (4000,512, 1, 1),   True),
    # ── DECODER ───────────────────────────────────────────────────────────────
    ("D1_convt",    "convt",   (4000,512, 4, 4),   True),
    ("D1_bn",       "bn",      (512,),              None),
    ("D2_convt",    "convt",   (512, 256, 4, 4),   True),
    ("D2_bn",       "bn",      (256,),              None),
    ("D3_convt",    "convt",   (256, 128, 4, 4),   True),
    ("D3_bn",       "bn",      (128,),              None),
    ("D4_convt",    "convt",   (128, 64,  4, 4),   True),
    ("D4_bn",       "bn",      (64,),               None),
    # ── OUTPUT HEAD ──────────────────────────────────────────────────────────
    ("OUT_conv",    "conv",    (3,   64,  3, 3),   True),
    # Tanh: no params
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_fp(arr):
    """float32 → int16 Q7.8, clamp to [-32768, 32767]"""
    v = np.round(arr.astype(np.float64) * SCALE).astype(np.int64)
    return np.clip(v, -(1<<(DW-1)), (1<<(DW-1))-1).astype(np.int16)

def write_hex(fname, arr_int16):
    """Write int16 array as 4-char hex, one per line (2's complement)"""
    with open(fname, "w") as f:
        for v in np.array(arr_int16).flatten():
            f.write(f"{int(v) & 0xFFFF:04x}\n")

def write_hex_40(fname, arr_int32):
    """Write int32/int64 as 10-char hex (for ACC_W=40)"""
    with open(fname, "w") as f:
        for v in np.array(arr_int32).flatten():
            f.write(f"{int(v) & 0xFFFFFFFFFF:010x}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_weights.py <path/to/generator.bin>")
        print("\nAvailable files might be:")
        print("  gw_latest.bin")
        print("  generator_epoch662.bin")
        sys.exit(1)

    bin_path = sys.argv[1]
    print("=" * 60)
    print(f"  extract_weights.py")
    print(f"  Input : {bin_path}")
    print(f"  Format: Q{DW-FRAC_W-1}.{FRAC_W}  SCALE={SCALE}  EPS_BN={EPS}")
    print("=" * 60)

    # ── Load raw bytes ─────────────────────────────────────────────────────────
    with open(bin_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)

    total_floats = len(raw)
    print(f"\nFile size  : {os.path.getsize(bin_path) / 1e6:.2f} MB")
    print(f"Total float: {total_floats:,}")

    # ── Parse sequentially ─────────────────────────────────────────────────────
    out_dir = "weights_hex"
    os.makedirs(out_dir, exist_ok=True)

    offset = 0
    layer_info = []

    print(f"\n{'Layer':<15} {'Type':<6} {'Shape':<22} {'#floats':>8}  {'offset_start':>12}")
    print("-" * 70)

    for name, ltype, shape, has_bias in LAYOUT:

        if ltype in ("conv", "convt"):
            # ── weight ────────────────────────────────────────────────────────
            n_weight = int(np.prod(shape))
            weight   = raw[offset : offset + n_weight].reshape(shape)
            offset  += n_weight

            weight_fp = to_fp(weight)
            fname_w   = f"{out_dir}/{name}_weight.hex"
            write_hex(fname_w, weight_fp)

            print(f"{name:<15} {'conv':<6} {str(shape):<22} {n_weight:>8,}  loaded")

            # ── bias (conv2d di kode lo default punya bias kecuali disable) ───
            # PyTorch Conv2d default bias=True, tapi di kode GAN tidak set bias=False
            # → ada bias. ConvTranspose2d juga ada bias.
            # has_bias=True berarti ada 1D bias tensor dengan ukuran C_out
            if has_bias:
                c_out  = shape[0]
                bias   = raw[offset : offset + c_out]
                offset += c_out
                bias_fp = to_fp(bias)
                write_hex(f"{out_dir}/{name}_bias.hex", bias_fp)

            layer_info.append((name, ltype, shape))

        elif ltype == "bn":
            # ── BatchNorm2d params (urutan PyTorch): ──────────────────────────
            #   weight (gamma), bias (beta), running_mean, running_var, num_batches_tracked
            c = shape[0]

            gamma   = raw[offset:offset+c]; offset += c
            beta    = raw[offset:offset+c]; offset += c
            r_mean  = raw[offset:offset+c]; offset += c
            r_var   = raw[offset:offset+c]; offset += c
            nb      = raw[offset:offset+1]; offset += 1   # num_batches_tracked (int64 stored as float? check)

            # ── Precompute fused params ────────────────────────────────────────
            scale_f  = gamma / np.sqrt(r_var + EPS)
            offset_f = beta  - r_mean * scale_f

            scale_fp  = to_fp(scale_f)
            offset_fp = to_fp(offset_f)

            write_hex(f"{out_dir}/{name}_scale.hex",  scale_fp)
            write_hex(f"{out_dir}/{name}_offset.hex", offset_fp)

            print(f"{name:<15} {'bn':<6} {str(shape):<22} {5*c+1:>8,}  "
                  f"scale=[{scale_f.min():.3f},{scale_f.max():.3f}]  "
                  f"offset=[{offset_f.min():.3f},{offset_f.max():.3f}]")

            layer_info.append((name, ltype, shape))

    print("-" * 70)
    print(f"Total floats parsed : {offset:,} / {total_floats:,}")

    if offset != total_floats:
        diff = total_floats - offset
        print(f"\n⚠  WARNING: {diff} floats remaining — cek layout atau ada layer ekstra")
        print("   Kemungkinan: num_batches_tracked disimpan sebagai int64 (2×float32)")
        # Try dengan nb sebagai int64 (2 float32 words)
        print("   Coba jalankan dengan flag --nb64 jika ada mismatch")
    else:
        print(f"\n✓  Semua float terbaca persis.")

    # ── Summary ────────────────────────────────────────────────────────────────
    hex_files = sorted(os.listdir(out_dir))
    print(f"\nOutput dir: {out_dir}/")
    print(f"Files generated ({len(hex_files)}):")
    for f in hex_files:
        fpath = f"{out_dir}/{f}"
        lines = sum(1 for _ in open(fpath))
        print(f"  {f:<40}  {lines:>6} entries")

    print("\nCara load ke RTL testbench:")
    print('  $readmemh("weights_hex/E1_conv_weight.hex", weight_mem);')
    print('  $readmemh("weights_hex/E2_bn_scale.hex",   scale_mem);')
    print('  $readmemh("weights_hex/E2_bn_offset.hex",  offset_mem);')
    print("=" * 60)


if __name__ == "__main__":
    main()
