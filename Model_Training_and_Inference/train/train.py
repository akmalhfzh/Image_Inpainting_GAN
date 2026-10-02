import os
import argparse
from pandas import DataFrame
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, ImageDataset, weights_init_normal, save_weights_bin, load_weights_bin


# ──────────────────────────────────────────────
# Argumen CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Context Encoder GAN (FFHQ / CelebA)")
parser.add_argument("--data_dir",        type=str,   default="./data/ffhq/thumbnails128x128",
                    help="Path ke folder gambar FFHQ (.png) atau CelebA (.jpg)")
parser.add_argument("--n_epochs",        type=int,   default=400)
parser.add_argument("--batch_size",      type=int,   default=16)
parser.add_argument("--lr",              type=float, default=0.0002)
parser.add_argument("--b1",              type=float, default=0.5)
parser.add_argument("--b2",              type=float, default=0.999)
parser.add_argument("--n_cpu",           type=int,   default=4)
parser.add_argument("--img_size",        type=int,   default=64,  help="Ukuran gambar (64x64)")
parser.add_argument("--mask_size",       type=int,   default=32,  help="Ukuran mask maksimum")
parser.add_argument("--min_mask_size",   type=int,   default=10,  help="Ukuran mask minimum saat training")
parser.add_argument("--channels",        type=int,   default=3)
parser.add_argument("--sample_interval", type=int,   default=100, help="Simpan sample setiap N batch")
parser.add_argument("--save_interval",   type=int,   default=1,   help="Simpan model setiap N epoch")
parser.add_argument("--resume",          type=str,   default=None,
                    help="Path ke checkpoint generator (.bin) untuk melanjutkan training (opsional)")
opt = parser.parse_args()
print(opt)

os.makedirs("images",  exist_ok=True)
os.makedirs("weights", exist_ok=True)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Device: {device}")

# PatchGAN patch shape untuk mask 32x32
patch_h = patch_w = opt.mask_size // (2 ** 3)   # = 4
patch   = (1, patch_h, patch_w)

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
generator     = Generator(channels=opt.channels).to(device)
discriminator = Discriminator(channels=opt.channels).to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if opt.resume:
    print(f"Melanjutkan training dari: {opt.resume}")
    load_weights_bin(generator, opt.resume, device)

# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
adversarial_loss = torch.nn.MSELoss().to(device)
pixelwise_loss   = torch.nn.L1Loss().to(device)

# ──────────────────────────────────────────────
# Optimizers
# ──────────────────────────────────────────────
optimizer_G = torch.optim.Adam(generator.parameters(),     lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ──────────────────────────────────────────────
# Dataset & Dataloader
# ──────────────────────────────────────────────
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataset = ImageDataset(opt.data_dir, transforms_=transforms_,
                             img_size=opt.img_size, mask_size=opt.mask_size,
                             min_mask_size=opt.min_mask_size, mode="train")
val_dataset   = ImageDataset(opt.data_dir, transforms_=transforms_,
                             img_size=opt.img_size, mask_size=opt.mask_size,
                             min_mask_size=opt.min_mask_size, mode="val")

dataloader      = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.n_cpu)
test_dataloader = DataLoader(val_dataset,   batch_size=12, shuffle=True, num_workers=1)

print(f"Train: {len(train_dataset)} gambar | Val: {len(val_dataset)} gambar")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ──────────────────────────────────────────────
# Fungsi simpan sample visual
# ──────────────────────────────────────────────
def save_sample(batches_done):
    samples, masked_samples, coords = next(iter(test_dataloader))
    samples        = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    y  = coords[0, 0].item()
    x  = coords[0, 1].item()
    ms = coords[0, 2].item()

    gen_full       = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, y:y+ms, x:x+ms] = gen_full[:, :, y:y+ms, x:x+ms]

    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)
    print(f"  -> Sample disimpan: images/{batches_done}.png")


# ──────────────────────────────────────────────
# History
# ──────────────────────────────────────────────
loss_values_adv   = []
loss_values_pixel = []
loss_values       = []
real_loss_values  = []
fake_loss_values  = []
d_loss_values     = []

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
print(f"\n=== Mulai Training {opt.n_epochs} epoch ===\n")

for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, coords) in enumerate(dataloader):

        y1s        = coords[:, 0]
        x1s        = coords[:, 1]
        mask_sizes = coords[:, 2]

        imgs        = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))

        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake  = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # ── Generator ──
        optimizer_G.zero_grad()
        gen_full = generator(masked_imgs)  # output 64x64 penuh

        # Ekstrak patch tiap sampel, resize ke opt.mask_size untuk Discriminator
        real_patches = []
        gen_patches  = []
        for b in range(imgs.shape[0]):
            y  = int(y1s[b].item())
            x  = int(x1s[b].item())
            ms = int(mask_sizes[b].item())

            real_patch = imgs[b:b+1,     :, y:y+ms, x:x+ms]
            gen_patch  = gen_full[b:b+1, :, y:y+ms, x:x+ms]

            real_patches.append(torch.nn.functional.interpolate(
                real_patch, size=(opt.mask_size, opt.mask_size),
                mode='bilinear', align_corners=False))
            gen_patches.append(torch.nn.functional.interpolate(
                gen_patch, size=(opt.mask_size, opt.mask_size),
                mode='bilinear', align_corners=False))

        real_patches = torch.cat(real_patches, dim=0)
        gen_patches  = torch.cat(gen_patches,  dim=0)

        # Pixel loss hanya pada area mask
        mask = torch.zeros_like(imgs)
        for b in range(imgs.shape[0]):
            y  = int(y1s[b].item())
            x  = int(x1s[b].item())
            ms = int(mask_sizes[b].item())
            mask[b, :, y:y+ms, x:x+ms] = 1.0

        g_adv   = adversarial_loss(discriminator(gen_patches), valid)
        g_pixel = pixelwise_loss(gen_full * mask, imgs * mask)
        g_loss  = 0.001 * g_adv + 0.999 * g_pixel
        g_loss.backward()
        optimizer_G.step()

        # ── Discriminator ──
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_patches),         valid)
        fake_loss = adversarial_loss(discriminator(gen_patches.detach()), fake)
        d_loss    = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # ── Log setiap 50 batch ──
        if i % 50 == 0:
            loss_values_adv.append(g_adv.item())
            loss_values_pixel.append(g_pixel.item())
            loss_values.append(g_loss.item())
            real_loss_values.append(real_loss.item())
            fake_loss_values.append(fake_loss.item())
            d_loss_values.append(d_loss.item())
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G adv: %.4f, pixel: %.4f]" %
                  (epoch + 1, opt.n_epochs, i, len(dataloader),
                   d_loss.item(), g_adv.item(), g_pixel.item()))

        # ── Simpan sample ──
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

    # ── Akhir epoch: simpan loss CSV & model ──
    DataFrame(loss_values_adv,   columns=["g_adv"]).to_csv("g_adv_loss_values.csv",     index=False)
    DataFrame(loss_values_pixel, columns=["g_pixel"]).to_csv("g_pixel_loss_values.csv", index=False)
    DataFrame(loss_values,       columns=["g_loss"]).to_csv("g_loss_values.csv",         index=False)
    DataFrame(real_loss_values,  columns=["real_loss"]).to_csv("real_loss_values.csv",   index=False)
    DataFrame(fake_loss_values,  columns=["fake_loss"]).to_csv("fake_loss_values.csv",   index=False)
    DataFrame(d_loss_values,     columns=["d_loss"]).to_csv("d_loss_values.csv",         index=False)

    if (epoch + 1) % opt.save_interval == 0:
        save_weights_bin(generator,     f"./weights/generator_epoch{epoch+1}.bin")
        save_weights_bin(discriminator, f"./weights/discriminator_epoch{epoch+1}.bin")

    save_weights_bin(generator,     "./weights/gw_latest.bin")
    save_weights_bin(discriminator, "./weights/dw_latest.bin")
    print(f"--> Epoch {epoch+1}/{opt.n_epochs} selesai! Model disimpan di ./weights/")

print("\n=== Training selesai! ===")
