import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=64, mask_size=32,
                 min_mask_size=10, mode="train"):
        self.transform      = transforms.Compose(transforms_)
        self.img_size       = img_size
        self.mask_size      = mask_size
        self.min_mask_size  = min_mask_size
        self.mode           = mode

        # Mendukung struktur folder FFHQ (subfolders) maupun flat folder
        self.files = sorted(
            glob.glob("%s/**/*.png" % root, recursive=True) +
            glob.glob("%s/**/*.jpg" % root, recursive=True) +
            glob.glob("%s/**/*.jpeg" % root, recursive=True) +
            glob.glob("%s/*.png" % root) +
            glob.glob("%s/*.jpg" % root) +
            glob.glob("%s/*.jpeg" % root)
        )
        # Menghapus duplikat
        self.files = sorted(list(dict.fromkeys(self.files)))

        if not self.files:
            raise FileNotFoundError(f"Tidak ada file gambar ditemukan di: {root}")

        print(f"[Dataset] Mode: {mode} | Ditemukan {len(self.files)} gambar.")

        split_idx  = int(len(self.files) * 0.9)
        self.files = self.files[:split_idx] if mode == "train" else self.files[split_idx:]

    def apply_random_mask(self, img):
        mask_size = np.random.randint(self.min_mask_size, self.mask_size + 1)
        max_pos = self.img_size - mask_size
        y1 = np.random.randint(0, max_pos)
        x1 = np.random.randint(0, max_pos)

        masked_img = img.clone()
        masked_img[:, y1:y1 + mask_size, x1:x1 + mask_size] = 1
        return masked_img, (y1, x1, mask_size)

    def apply_center_mask(self, img):
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i:i + self.mask_size, i:i + self.mask_size] = 1
        return masked_img, (i, i, self.mask_size)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        img = self.transform(img)
        if self.mode == "train":
            masked_img, (y1, x1, ms) = self.apply_random_mask(img)
        else:
            masked_img, (y1, x1, ms) = self.apply_center_mask(img)
        return img, masked_img, torch.tensor([y1, x1, ms])

    def __len__(self):
        return len(self.files)


# ──────────────────────────────────────────────
# Generator Lite (Optimasi PYNQ-Z1)
# ──────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        # Struktur Arsitektur Lite:
        # Input: 64x64
        # Encoder: 64 -> 32 -> 16 -> 8 -> 4 -> 2
        # Decoder: 2 -> 4 -> 8 -> 16 -> 32 -> 64
        self.model = nn.Sequential(
            *downsample(channels, 32, normalize=False), # E1: 64 -> 32
            *downsample(32, 64),                        # E2: 32 -> 16
            *downsample(64, 128),                       # E3: 16 -> 8
            *downsample(128, 256),                      # E4: 8 -> 4
            *downsample(256, 512),                      # E5: 4 -> 2
            
            # Bottleneck: Dikecilkan dari 4000 ke 512
            nn.Conv2d(512, 512, 1),                     # B1: 2x2
            
            *upsample(512, 256),                        # D1: 2 -> 4
            *upsample(256, 128),                        # D2: 4 -> 8
            *upsample(128, 64),                         # D3: 8 -> 16
            *upsample(64, 32),                          # D4: 16 -> 32
            *upsample(32, 32),                          # D5: 32 -> 64
            
            nn.Conv2d(32, channels, 3, 1, 1),           # Output: 64x64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────
# Discriminator Lite
# ──────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()

        def block(in_f, out_f, stride, normalize):
            layers = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Channel dikurangi untuk efisiensi VRAM dan BRAM
        layers, in_f = [], channels
        for out_f, stride, norm in [(32, 2, False), (64, 2, True), (128, 2, True), (256, 1, True)]:
            layers.extend(block(in_f, out_f, stride, norm))
            in_f = out_f
            
        layers.append(nn.Conv2d(in_f, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_weights_bin(model, filepath):
    """Menyimpan bobot model ke format raw binary (.bin) untuk ekstraksi RTL"""
    with open(filepath, 'wb') as f:
        for name, param in model.state_dict().items():
            raw_bytes = param.cpu().numpy().astype(np.float32).tobytes()
            f.write(raw_bytes)

def load_weights_bin(model, filepath, device):
    """Memuat bobot dari file raw binary (.bin)"""
    with open(filepath, 'rb') as f:
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            num_elements = param.numel()
            raw_bytes    = f.read(num_elements * 4)

            if not raw_bytes:
                raise EOFError(f"File {filepath} tidak cocok dengan struktur model.")

            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            param.copy_(torch.from_numpy(arr).view_as(param).to(device))
