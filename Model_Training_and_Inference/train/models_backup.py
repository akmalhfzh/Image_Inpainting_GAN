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
    def __init__(self, root, transforms_=None, img_size=64, mask_size=32, mode="train"):
        self.transform  = transforms.Compose(transforms_)
        self.img_size   = img_size
        self.mask_size  = mask_size
        self.mode       = mode
        self.files      = sorted(glob.glob("%s/*.jpg" % root))
        if not self.files:
            raise FileNotFoundError(f"Tidak ada file .jpg di: {root}")

        split_idx  = int(len(self.files) * 0.9)
        self.files = self.files[:split_idx] if mode == "train" else self.files[split_idx:]

    def apply_random_mask(self, img):
        y1, x1      = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2      = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img  = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):
        i          = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        img = self.transform(img)
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


# ──────────────────────────────────────────────
# Generator  (encoder-decoder / context encoder)
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

        # 64→32→16→8→4→2  (bottleneck)  →4→8→16→32  (output mask 32×32)
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),   # 64→32
            *downsample(64,  64),                          # 32→16
            *downsample(64,  128),                         # 16→8
            *downsample(128, 256),                         # 8→4
            *downsample(256, 512),                         # 4→2
            nn.Conv2d(512, 4000, 1),                       # bottleneck
            *upsample(4000, 512),                          # 2→4
            *upsample(512,  256),                          # 4→8
            *upsample(256,  128),                          # 8→16
            *upsample(128,  64),                           # 16→32
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────
# Discriminator  (PatchGAN on the masked patch)
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

        layers, in_f = [], channels
        for out_f, stride, norm in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(block(in_f, out_f, stride, norm))
            in_f = out_f
        layers.append(nn.Conv2d(in_f, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


# ──────────────────────────────────────────────
# Weight initialisation
# ──────────────────────────────────────────────
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_weights_bin(model, filepath):
    """Menyimpan bobot model secara sekuensial ke format raw binary (.bin)"""
    with open(filepath, 'wb') as f:
        for name, param in model.state_dict().items():
            # Pindahkan ke CPU, ubah ke NumPy float32, lalu ambil raw bytes-nya
            raw_bytes = param.cpu().numpy().astype(np.float32).tobytes()
            f.write(raw_bytes)

def load_weights_bin(model, filepath, device):
    """Memuat bobot dari file raw binary (.bin) ke dalam model PyTorch"""
    with open(filepath, 'rb') as f:
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            num_elements = param.numel()
            # 1 float32 = 4 bytes
            raw_bytes = f.read(num_elements * 4) 
            
            if not raw_bytes:
                raise EOFError(f"File {filepath} terlalu pendek atau struktur model tidak cocok.")
            
            # Ubah bytes kembali menjadi array numpy float32
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            
            # Copy nilai ke dalam parameter model dan pindahkan ke device (CPU/GPU)
            param.copy_(torch.from_numpy(arr).view_as(param).to(device))
