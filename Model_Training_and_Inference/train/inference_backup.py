import os
import glob
import random
import argparse
from PIL import Image

import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

from models import Generator, load_weights_bin


# ──────────────────────────────────────────────
# Argumen CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inference Context Encoder GAN")
parser.add_argument("--weights",   type=str, default="./weights/gw_latest.bin",
                    help="Path ke file bobot generator (.bin)")
parser.add_argument("--data_dir",  type=str, default=None,
                    help="Folder gambar untuk dipilih secara acak")
parser.add_argument("--img_path",  type=str, default=None,
                    help="Path gambar spesifik (override --data_dir)")
parser.add_argument("--img_size",  type=int, default=64)
parser.add_argument("--mask_size", type=int, default=32)
parser.add_argument("--channels",  type=int, default=3)
parser.add_argument("--output",    type=str, default="inference_result.png",
                    help="Path file output gambar hasil")
parser.add_argument("--no_show",   action="store_true",
                    help="Jangan tampilkan window matplotlib (berguna di server headless)")
opt = parser.parse_args()

# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────
if not os.path.exists(opt.weights):
    raise FileNotFoundError(
        f"File bobot tidak ditemukan: {opt.weights}\n"
        "Jalankan train.py terlebih dahulu minimal 1 epoch."
    )

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Device: {device}")

# ──────────────────────────────────────────────
# Muat Generator
# ──────────────────────────────────────────────
print(f"Memuat bobot dari: {opt.weights}")
generator = Generator(channels=opt.channels).to(device)
load_weights_bin(generator, opt.weights, device)
generator.eval()

# ──────────────────────────────────────────────
# Pilih gambar input
# ──────────────────────────────────────────────
if opt.img_path:
    img_path = opt.img_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
elif opt.data_dir:
    all_files = sorted(glob.glob(os.path.join(opt.data_dir, "*.jpg")))
    if not all_files:
        raise FileNotFoundError(f"Tidak ada .jpg di: {opt.data_dir}")
    img_path = random.choice(all_files)
else:
    raise ValueError("Berikan --img_path atau --data_dir")

print(f"Gambar input: {img_path}")

# ──────────────────────────────────────────────
# Transformasi
# ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

real_img   = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
masked_img = real_img.clone()

# Buat lubang kotak putih di tengah
i = (opt.img_size - opt.mask_size) // 2
masked_img[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = 1.0

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
with torch.no_grad():
    gen_full = generator(masked_img)

# Tempel tambalan ke gambar bolong
inpainted_img = masked_img.clone()
inpainted_img[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = \
		gen_full[:, :, i : i + opt.mask_size, i : i + opt.mask_size]

# ──────────────────────────────────────────────
# Visualisasi & simpan
# ──────────────────────────────────────────────
def denorm(x):
    """[-1,1] → [0,1] numpy array siap di-imshow."""
    return ((x.cpu().squeeze().permute(1, 2, 0) + 1) / 2.0).clamp(0, 1).numpy()

if opt.no_show:
    matplotlib.use("Agg")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles     = ["Gambar Asli (Ground Truth)", "Input (Masked)", "Hasil Tambalan AI"]
images     = [real_img, masked_img, inpainted_img]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(denorm(img))
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(opt.output, dpi=150, bbox_inches="tight")
print(f"Hasil disimpan: {opt.output}")

if not opt.no_show:
    plt.show()
