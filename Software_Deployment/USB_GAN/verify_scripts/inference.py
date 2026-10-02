#!/usr/bin/env python3

import os
import glob
import random
import argparse
from PIL import Image
import numpy as np
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

try:
    from models import Generator, load_weights_bin
except ModuleNotFoundError:
    print("\n[ERROR] models.py tidak ditemukan!\n")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inference Context Encoder GAN")
parser.add_argument("--weights",   type=str, default="./weights/gw_latest.bin")
parser.add_argument("--data_dir",  type=str, default=None)
parser.add_argument("--img_path",  type=str, default=None)
parser.add_argument("--img_size",  type=int, default=64)
parser.add_argument("--mask_size", type=int, default=16)
parser.add_argument("--mask_x",    type=int, default=None)
parser.add_argument("--mask_y",    type=int, default=None)
parser.add_argument("--channels",  type=int, default=3)
parser.add_argument("--output",    type=str, default="inference_result.png")
parser.add_argument("--no_show",   action="store_true")
parser.add_argument("--dump_hex",  action="store_true",
                    help="Dump per-layer activations for RTL verification")
opt = parser.parse_args()

if not os.path.exists(opt.weights):
    raise FileNotFoundError(f"File bobot tidak ditemukan: {opt.weights}")

cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

generator = Generator(channels=opt.channels).to(device)
load_weights_bin(generator, opt.weights, device)
generator.eval()

# ── Quantization helpers ──────────────────────────────────────────────────────
DW     = 16
FRAC_W = 8
SCALE  = 1 << FRAC_W  # 256

def quantize_tensor(x):
    """Quantize float tensor to Q7.8 and back to float (simulates RTL truncation)"""
    x_q = torch.round(x * SCALE).clamp(-32768, 32767) / SCALE
    return x_q

def tensor_to_hex(x):
    """Convert float tensor to Q7.8 hex strings"""
    x_np = x.cpu().numpy().flatten()
    x_int = np.round(x_np * SCALE).astype(np.int32)
    x_int = np.clip(x_int, -32768, 32767).astype(np.int16)
    return [f"{int(v) & 0xFFFF:04x}" for v in x_int]

# ── Image selection ───────────────────────────────────────────────────────────
if opt.img_path:
    img_path = opt.img_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
elif opt.data_dir:
    all_files = sorted(
        glob.glob(os.path.join(opt.data_dir, "**", "*.png"),  recursive=True) +
        glob.glob(os.path.join(opt.data_dir, "**", "*.jpg"),  recursive=True) +
        glob.glob(os.path.join(opt.data_dir, "**", "*.jpeg"), recursive=True) +
        glob.glob(os.path.join(opt.data_dir, "*.png")) +
        glob.glob(os.path.join(opt.data_dir, "*.jpg")) +
        glob.glob(os.path.join(opt.data_dir, "*.jpeg"))
    )
    all_files = list(dict.fromkeys(all_files))
    if not all_files:
        raise FileNotFoundError(f"Tidak ada file gambar di: {opt.data_dir}")
    img_path = random.choice(all_files)
else:
    raise ValueError("Berikan --img_path atau --data_dir")

# ── Mask position ─────────────────────────────────────────────────────────────
mask_x = opt.mask_x if opt.mask_x is not None else (opt.img_size - opt.mask_size) // 2
mask_y = opt.mask_y if opt.mask_y is not None else (opt.img_size - opt.mask_size) // 2
mask_x = max(0, min(mask_x, opt.img_size - opt.mask_size))
mask_y = max(0, min(mask_y, opt.img_size - opt.mask_size))

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

real_img   = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
masked_img = real_img.clone()
masked_img[:, :, mask_y:mask_y+opt.mask_size, mask_x:mask_x+opt.mask_size] = 1.0

# ── [FIX-1] Quantize input to Q7.8 BEFORE inference ──────────────────────────
masked_img_q = quantize_tensor(masked_img)

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    if opt.dump_hex:
        os.makedirs("outputs/py_ref", exist_ok=True)

        # Save quantized input image (same as what RTL uses)
        with open("outputs/py_ref/input_image.hex", "w") as f:
            for h in tensor_to_hex(masked_img_q):
                f.write(h + "\n")

        # [FIX-2] Layer-by-layer quantized inference
        # Run each layer, quantize output, then feed to next layer.
        # This matches RTL behavior where each layer output is truncated to Q7.8.
        layer_names = [
            "E1", "E2", "E3", "E4", "E5", "BOT",
            "D1", "D2", "D3", "D4", "D5", "OUT"
        ]
        # Map: layer_name → last index in model.Sequential for that layer
        # (index of activation function or last op)
        layer_end_indices = [1, 4, 7, 10, 13, 14, 17, 20, 23, 26, 29, 31]

        x = masked_img_q.clone()
        layer_name_idx = 0
        prev_end = 0

        print("[PyTorch] Running quantized layer-by-layer inference...")

        for name, end_idx in zip(layer_names, layer_end_indices):
            # Run layers from prev_end to end_idx (inclusive)
            for i in range(prev_end, end_idx + 1):
                x = generator.model[i](x)

            # Dump float output THEN quantize for next layer
            # The hex dump represents quantized values (same as RTL output)
            x_q = quantize_tensor(x)

            hex_file = f"outputs/py_ref/ref_{name}.hex"
            with open(hex_file, "w") as f:
                for h in tensor_to_hex(x_q):
                    f.write(h + "\n")
            print(f"  → Dumped {name:<4} shape {tuple(x.shape)} to {hex_file}")

            # Feed QUANTIZED output to next layer (matches RTL truncation)
            x = x_q
            prev_end = end_idx + 1

        gen_full = x  # final output (after Tanh, quantized)
    else:
        gen_full = generator(masked_img_q)

# Tempel hasil
inpainted_img = masked_img.clone()
inpainted_img[:, :,
              mask_y:mask_y+opt.mask_size,
              mask_x:mask_x+opt.mask_size] = \
    gen_full[:, :,
             mask_y:mask_y+opt.mask_size,
             mask_x:mask_x+opt.mask_size]

# ── Visualisasi ───────────────────────────────────────────────────────────────
def denorm(x):
    return ((x.cpu().squeeze().permute(1, 2, 0) + 1) / 2.0).clamp(0, 1).numpy()

if opt.no_show:
    matplotlib.use("Agg")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Ground Truth", "Input (Masked)", "AI Inpainted"]
images = [real_img, masked_img, inpainted_img]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(denorm(img))
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"mask={opt.mask_size}x{opt.mask_size}  pos=({mask_x},{mask_y})", fontsize=9, y=0.02)
plt.tight_layout()
plt.savefig(opt.output, dpi=150, bbox_inches="tight")
if not opt.no_show:
    plt.show()
