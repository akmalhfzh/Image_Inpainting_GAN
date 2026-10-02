import os
import glob
import random
import argparse
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

# Asumsi models.py sejajar dengan script pemanggil, atau tambahkan 'sys.path.append(..)' jika perlu
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
parser.add_argument("--mask_size", type=int, default=32,
                    help="Ukuran mask (default: 32)")
parser.add_argument("--mask_x",    type=int, default=None,
                    help="Posisi x kiri atas mask (default: tengah)")
parser.add_argument("--mask_y",    type=int, default=None,
                    help="Posisi y kiri atas mask (default: tengah)")
parser.add_argument("--channels",  type=int, default=3)
parser.add_argument("--output",    type=str, default="inference_result.png",
                    help="Path file output gambar hasil")
parser.add_argument("--no_show",   action="store_true",
                    help="Jangan tampilkan window matplotlib")
parser.add_argument("--dump_hex",  action="store_true", 
                    help="Dump aktivasi per layer untuk RTL verification")
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
# Sistem PyTorch Hooks untuk Dump Aktivasi
# ──────────────────────────────────────────────
layer_activations = {}
def get_activation(name):
    def hook(model, input, output):
        layer_activations[name] = output.detach().cpu().numpy()
    return hook

if opt.dump_hex:
    os.makedirs("outputs/py_ref", exist_ok=True)
    layer_names = [
        "E1", "E2", "E3", "E4", "E5", "BOT", "D1", "D2", "D3", "D4", "D5", "OUT"
    ]
    # Indeks disesuaikan untuk struktur nn.Sequential arsitektur LITE
    # (Setelah modifikasi agar simetris)
    hook_indices = [1, 4, 7, 10, 13, 14, 17, 20, 23, 26, 29, 31] 
    
    for i, idx in enumerate(hook_indices):
        if idx < len(generator.model):
            generator.model[idx].register_forward_hook(get_activation(layer_names[i]))
        else:
            print(f"[WARN] Indeks hook {idx} melebihi panjang model!")

# ──────────────────────────────────────────────
# Pilih gambar input
# ──────────────────────────────────────────────
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
    print(f"Total file ditemukan: {len(all_files)}")

else:
    raise ValueError("Berikan --img_path atau --data_dir")

print(f"Gambar input: {img_path}")

# ──────────────────────────────────────────────
# Hitung posisi mask
# ──────────────────────────────────────────────
if opt.mask_x is None:
    mask_x = (opt.img_size - opt.mask_size) // 2
else:
    mask_x = opt.mask_x

if opt.mask_y is None:
    mask_y = (opt.img_size - opt.mask_size) // 2
else:
    mask_y = opt.mask_y

mask_x = max(0, min(mask_x, opt.img_size - opt.mask_size))
mask_y = max(0, min(mask_y, opt.img_size - opt.mask_size))

print(f"Mask: ukuran={opt.mask_size}x{opt.mask_size}  posisi=({mask_x}, {mask_y})")

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

# Buat lubang di posisi yang ditentukan
masked_img[:, :,
           mask_y : mask_y + opt.mask_size,
           mask_x : mask_x + opt.mask_size] = 1.0

# ──────────────────────────────────────────────
# INFERENCE & DUMPING
# ──────────────────────────────────────────────
with torch.no_grad():
    gen_full = generator(masked_img)

if opt.dump_hex:
    print("\n[PyTorch] Merekam aktivasi per layer (Q7.8 hex format)...")
    DW, FRAC_W, SCALE = 16, 8, 256
    
    for name, act in layer_activations.items():
        # Kuantisasi ke Fixed-Point
        act_fp = np.round(act * SCALE).astype(np.int32)
        act_fp = np.clip(act_fp, -(1<<(DW-1)), (1<<(DW-1))-1).astype(np.int16)
        
        hex_file = f"outputs/py_ref/ref_{name}.hex"
        with open(hex_file, "w") as f:
            for v in act_fp.flatten():
                f.write(f"{int(v) & 0xFFFF:04x}\n")
        print(f"  → Dumped {name:<4} shape {act.shape} to {hex_file}")

# Tempel hasil generate ke area mask
inpainted_img = masked_img.clone()
inpainted_img[:, :,
              mask_y : mask_y + opt.mask_size,
              mask_x : mask_x + opt.mask_size] = \
    gen_full[:, :,
             mask_y : mask_y + opt.mask_size,
             mask_x : mask_x + opt.mask_size]

# ──────────────────────────────────────────────
# Visualisasi & simpan
# ──────────────────────────────────────────────
def denorm(x):
    """[-1,1] -> [0,1] numpy array siap di-imshow."""
    return ((x.cpu().squeeze().permute(1, 2, 0) + 1) / 2.0).clamp(0, 1).numpy()

if opt.no_show:
    matplotlib.use("Agg")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Gambar Asli (Ground Truth)", "Input (Masked)", "Hasil Tambalan AI"]
images = [real_img, masked_img, inpainted_img]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(denorm(img))
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(
    f"mask={opt.mask_size}x{opt.mask_size}  pos=({mask_x},{mask_y})",
    fontsize=9, y=0.02
)
plt.tight_layout()
plt.savefig(opt.output, dpi=150, bbox_inches="tight")
print(f"Hasil disimpan: {opt.output}")

if not opt.no_show:
    plt.show()
