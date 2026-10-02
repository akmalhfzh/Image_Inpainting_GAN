#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

from models import Generator, load_weights_bin

DW     = 16
SCALE  = 256.0
MAX_VAL =  32767
MIN_VAL = -32768

def to_fp(x):
    x_scaled  = np.round(x * SCALE).astype(np.int32)
    x_clipped = np.clip(x_scaled, MIN_VAL, MAX_VAL)
    return [f"{int(v) & 0xFFFF:04x}" for v in x_clipped.flatten()]

def write_hex(fname, hex_list):
    with open(fname, "w") as f:
        f.write("\n".join(hex_list) + "\n")
    print(f"  -> Saved: {fname} ({len(hex_list)} entries)")

def extract_model_from_bin():
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "weights/gw_latest.bin"
    out_dir  = "weights_hex"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f" Extracting from: {bin_path}")
    print("=" * 60)

    device = torch.device("cpu")
    model  = Generator(channels=3)

    if os.path.exists(bin_path):
        print(f"Loading weights from {bin_path}...")
        load_weights_bin(model, bin_path, device)
    else:
        print(f"[WARNING] {bin_path} not found — using random weights (RTL smoke test only)")

    state_dict = model.state_dict()

    # ── Layer map: (rtl_name, conv_idx_str, is_convtranspose, has_bn, bn_idx_str)
    # conv_idx = index di model.Sequential untuk conv layer
    # is_convtranspose: True untuk decoder D1-D5 (mode=1 di RTL)
    # has_bn: apakah layer ini punya BatchNorm di PyTorch
    # bn_idx: index BatchNorm setelah conv di Sequential
    layer_map = [
        # rtl_name  conv_idx  is_transpose  has_bn  bn_idx
        ("E1",  "0",  False, False, None),
        ("E2",  "2",  False, True,  "3"),
        ("E3",  "5",  False, True,  "6"),
        ("E4",  "8",  False, True,  "9"),
        ("E5",  "11", False, True,  "12"),
        ("BOT", "14", False, False, None),
        ("D1",  "15", True,  True,  "16"),
        ("D2",  "18", True,  True,  "19"),
        ("D3",  "21", True,  True,  "22"),
        ("D4",  "24", True,  True,  "25"),
        ("D5",  "27", True,  True,  "28"),
        ("OUT", "30", False, False, None),
    ]

    EPS = 1e-5

    for rtl_name, conv_idx, is_transpose, has_bn, bn_idx in layer_map:
        print(f"\nProcessing Layer: {rtl_name}")

        conv_w_key = f"model.{conv_idx}.weight"
        conv_b_key = f"model.{conv_idx}.bias"

        conv_w = state_dict[conv_w_key].numpy()
        conv_b = (state_dict[conv_b_key].numpy()
                  if conv_b_key in state_dict
                  else np.zeros(conv_w.shape[0]))

        if has_bn and bn_idx is not None:
            gamma   = state_dict[f"model.{bn_idx}.weight"].numpy()
            beta    = state_dict[f"model.{bn_idx}.bias"].numpy()
            r_mean  = state_dict[f"model.{bn_idx}.running_mean"].numpy()
            r_var   = state_dict[f"model.{bn_idx}.running_var"].numpy()

            scale_f  = gamma / np.sqrt(r_var + EPS)
            offset_f = beta + scale_f * (conv_b - r_mean)

            # Fold scale ke weight
            # Conv2d       weight shape: [C_out, C_in, kH, kW]
            # ConvTranspose weight shape: [C_in, C_out, kH, kW]
            if not is_transpose:
                # output channel = dim 0
                conv_w_folded = conv_w * scale_f.reshape(-1, 1, 1, 1)
            else:
                # output channel = dim 1 untuk ConvTranspose2d
                conv_w_folded = conv_w * scale_f.reshape(1, -1, 1, 1)

            write_hex(f"{out_dir}/{rtl_name}_weight.hex", to_fp(conv_w_folded))
            write_hex(f"{out_dir}/{rtl_name}_bias.hex",   to_fp(offset_f))
            print(f"  -> Full BN folding applied (scale→weight, offset→bias)")

        else:
            # Layer tanpa BN: simpan weight dan bias as-is
            write_hex(f"{out_dir}/{rtl_name}_weight.hex", to_fp(conv_w))
            write_hex(f"{out_dir}/{rtl_name}_bias.hex",   to_fp(conv_b))
            print(f"  -> No BN — weight and bias extracted directly")

    print("\nExtraction complete!")
    print("NOTE: Tidak ada lagi _scale.hex / _offset.hex — sim_main.cpp")
    print("      hanya perlu load _weight.hex (wr_type=0) dan _bias.hex (wr_type=3)")

if __name__ == "__main__":
    extract_model_from_bin()
