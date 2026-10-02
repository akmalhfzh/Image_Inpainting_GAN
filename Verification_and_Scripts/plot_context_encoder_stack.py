#!/usr/bin/env python3
"""
plot_context_encoder_stack.py — Visualisasi Context Encoder (bentuk V)
=========================================================================
Beda dengan visualize_rtl_hex.py (yang cuma bikin 1 grid channel per layer),
script ini bikin "tumpukan" (deck) beberapa channel representatif per layer,
disusun membentuk V: Encoder (E1..E5) turun di kiri, Bottleneck (BOT) di
ujung bawah, Decoder (D1..D5) naik di kanan, Input di kiri atas, Output di
kanan atas.

TIDAK ada garis skip-connection (beda dengan U-Net) karena model kamu
Context Encoder murni bottleneck.

Usage:
    python3 plot_context_encoder_stack.py --outputs-dir outputs
    python3 plot_context_encoder_stack.py --outputs-dir outputs --n-show 8
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch, Circle

FRAC_W = 8  # Q8.8 fixed-point

LAYER_SHAPES = {
    "E1": (32, 32, 32), "E2": (64, 16, 16), "E3": (128, 8, 8),
    "E4": (256, 4, 4), "E5": (512, 2, 2),
    "BOT": (512, 2, 2),
    "D1": (256, 4, 4), "D2": (128, 8, 8), "D3": (64, 16, 16),
    "D4": (32, 32, 32), "D5": (32, 64, 64),
}
# urutan file rtl_layer_N.hex sesuai visualize_rtl_hex.py
LAYER_FILE_IDX = {
    "E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4,
    "BOT": 5,
    "D1": 6, "D2": 7, "D3": 8, "D4": 9, "D5": 10,
}


# ── Load hex ------------------------------------------------------------------
def load_hex_layer(hex_path: str, shape):
    if not os.path.exists(hex_path):
        print(f"  [WARN] {hex_path} tidak ditemukan")
        return None
    raw = []
    with open(hex_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            val_uint = int(line, 16) & 0xFFFF
            val_int = val_uint if val_uint < 0x8000 else val_uint - 0x10000
            raw.append(val_int)

    c, h, w = shape
    expected = c * h * w
    if len(raw) != expected:
        print(f"  [WARN] {hex_path}: expect {expected}, got {len(raw)}")
        if len(raw) < expected:
            raw.extend([0] * (expected - len(raw)))
        else:
            raw = raw[:expected]

    arr = np.array(raw, dtype=np.float32) / (2 ** FRAC_W)
    return arr.reshape(c, h, w)


# ── Channel -> RGB via colormap -----------------------------------------------
def channel_to_rgb(channel_2d, cmap_name="magma"):
    vmin, vmax = channel_2d.min(), channel_2d.max()
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(channel_2d)
    else:
        norm = (channel_2d - vmin) / (vmax - vmin)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H,W,4) float 0..1
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


# ── Create a stack/deck of multiple channels from a single layer ----------------------
def build_channel_stack(arr, n_show=6, thumb_size=52, dx=7, dy=7,
                         cmap_name="magma", border_color=(255, 255, 255)):
    """
    arr: (C,H,W) feature map satu layer.
    Return: RGB uint8 image berisi tumpukan n_show channel (deck effect),
    channel pertama (index 0) di depan/bawah, channel terakhir di belakang/atas.
    """
    c = arr.shape[0]
    n_show = min(n_show, c)
    idxs = np.linspace(0, c - 1, n_show).astype(int)

    canvas_w = thumb_size + dx * (n_show - 1)
    canvas_h = thumb_size + dy * (n_show - 1)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # gambar dari belakang (index terakhir) ke depan (index pertama)
    for order, i in enumerate(reversed(range(n_show))):
        ch_idx = idxs[i]
        rgb = channel_to_rgb(arr[ch_idx], cmap_name)
        thumb = np.array(
            Image.fromarray(rgb).resize((thumb_size, thumb_size), Image.NEAREST)
        )
        # Thin border to visually separate each feature map
        thumb_bordered = thumb.copy()
        thumb_bordered[0, :, :] = border_color
        thumb_bordered[-1, :, :] = border_color
        thumb_bordered[:, 0, :] = border_color
        thumb_bordered[:, -1, :] = border_color

        x0 = i * dx
        y0 = (n_show - 1 - i) * dy  # channel depan (i kecil) -> y0 besar (bawah)
        canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb_bordered

    return canvas


# ── Load gambar biasa (input/output) ------------------------------------------
def load_plain_image(path):
    if path and os.path.exists(path):
        return np.array(Image.open(path).convert("RGB"))
    return None


# ── Plot utama: bentuk V (Context Encoder, tanpa skip-connection) ------------
def plot_context_encoder(
    input_img,
    output_img,
    encoder_arrs,   # dict label -> (C,H,W) array, urut E1..E5
    bottleneck_arr, # (C,H,W) array untuk BOT
    decoder_arrs,   # dict label -> (C,H,W) array, urut D1..D5
    n_show=6,
    thumb_size=90,
    dx=11, dy=11,
    cmap_name="magma",
    display_size=1.35,   # ukuran fisik tiap node di kanvas (independen dari thumb_size)
    row_gap=1.6,          # jarak vertikal antar layer, must be > display_size to avoid overlapping
    col_gap=4.4,          # jarak horizontal encoder <-> decoder
    save_path=None,
    title="Context Encoder — Feature Map per Layer",
):
    enc_labels = list(encoder_arrs.keys())
    dec_labels = list(decoder_arrs.keys())
    n_side = len(enc_labels)  # asumsi encoder & decoder jumlahnya sama (5)

    fig, ax = plt.subplots(figsize=(6 + n_side * 2.6, 6 + n_side * 2.0))
    ax.axis("off")

    x_enc, x_dec = 0.0, col_gap
    x_in, x_out = x_enc - 2.2, x_dec + 2.2

    node_positions = {}  # label -> (x,y)

    # ---- Encoder: E1 (atas, y=0) turun ke E5 (bawah) ----
    for row, label in enumerate(enc_labels):
        node_positions[label] = (x_enc, row * row_gap)

    # ---- Decoder: D1 (bawah) naik ke D_last (atas) ----
    for row, label in enumerate(reversed(dec_labels)):
        node_positions[label] = (x_dec, row * row_gap)

    # ---- Bottleneck: ujung paling bawah, di tengah ----
    y_bot = n_side * row_gap
    x_bot = (x_enc + x_dec) / 2
    node_positions["BOT"] = (x_bot, y_bot)

    # ---- Input & Output: sejajar baris paling atas ----
    node_positions["INPUT"] = (x_in, 0)
    node_positions["OUTPUT"] = (x_out, 0)

    def draw_arrow(p1, p2):
        ax.annotate(
            "", xy=p2, xytext=p1,
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.4,
                             shrinkA=40, shrinkB=40),
            zorder=1,
        )

    def draw_badge(label, pos, color="#333333"):
        x, y = pos
        bx, by = x, y - display_size / 2 - 0.18
        circ = Circle((bx, by), 0.16, facecolor=color, edgecolor="white",
                      linewidth=1.4, zorder=5)
        ax.add_patch(circ)
        ax.text(bx, by, label, ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold", zorder=6)

    def draw_stack_node(label, arr, pos, badge_color="#333333"):
        stack_img = build_channel_stack(arr, n_show, thumb_size, dx, dy, cmap_name)
        x, y = pos
        aspect = stack_img.shape[1] / stack_img.shape[0]  # w/h
        half_h = display_size / 2
        half_w = half_h * aspect
        ax.imshow(stack_img, extent=[x - half_w, x + half_w,
                                      y + half_h, y - half_h], zorder=2)
        draw_badge(label, pos, badge_color)

    # ---- gambar node stack encoder & decoder & bottleneck ----
    for label in enc_labels:
        draw_stack_node(label, encoder_arrs[label], node_positions[label])
    for label in dec_labels:
        draw_stack_node(label, decoder_arrs[label], node_positions[label])
    draw_stack_node("BOT", bottleneck_arr, node_positions["BOT"], badge_color="#7a3b96")

    # ---- input & output (gambar biasa, bukan stack) ----
    io_half = display_size / 2
    for key, img, color in [("INPUT", input_img, "#c98a1f"), ("OUTPUT", output_img, "#2e5f8a")]:
        x, y = node_positions[key]
        if img is not None:
            box = FancyBboxPatch(
                (x - io_half - 0.05, y - io_half - 0.05), (io_half + 0.05) * 2, (io_half + 0.05) * 2,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=2.2, edgecolor=color, facecolor="white", zorder=2,
            )
            ax.add_patch(box)
            ax.imshow(img, extent=[x - io_half, x + io_half, y + io_half, y - io_half], zorder=3)
        ax.text(x, y - io_half - 0.22, "Input" if key == "INPUT" else "Output",
                 ha="center", va="bottom", fontsize=10, zorder=4)

    # ---- panah pipeline: Input -> E1 -> E2 -> ... -> E5 -> BOT -> D1 -> ... -> D5 -> Output
    chain = ["INPUT"] + enc_labels + ["BOT"] + dec_labels + ["OUTPUT"]
    for a, b in zip(chain[:-1], chain[1:]):
        draw_arrow(node_positions[a], node_positions[b])

    ax.set_xlim(x_in - 1.2, x_out + 1.2)
    ax.set_ylim(y_bot + 1.2, -1.2)  # bottleneck di bawah -> invert
    ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nDiagram disimpan ke: {save_path}")
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs",
                         help="folder berisi rtl_layer_*.hex, py_ref/, output_rtl.png")
    parser.add_argument("--n-show", type=int, default=6, help="jumlah channel ditumpuk per layer")
    parser.add_argument(
        "--input-img",
        default="USB_GAN/testcases/test_kakek/masked_input.png",
        help="path gambar input (default: masked_input.png di testcase test_kakek)",
    )
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    out_dir = args.outputs_dir
    ref_dir = os.path.join(out_dir, "py_ref")

    encoder_arrs, decoder_arrs = {}, {}
    for lbl in ["E1", "E2", "E3", "E4", "E5"]:
        hex_path = os.path.join(out_dir, f"rtl_layer_{LAYER_FILE_IDX[lbl]}.hex")
        encoder_arrs[lbl] = load_hex_layer(hex_path, LAYER_SHAPES[lbl])
    for lbl in ["D1", "D2", "D3", "D4", "D5"]:
        hex_path = os.path.join(out_dir, f"rtl_layer_{LAYER_FILE_IDX[lbl]}.hex")
        decoder_arrs[lbl] = load_hex_layer(hex_path, LAYER_SHAPES[lbl])
    bottleneck_arr = load_hex_layer(
        os.path.join(out_dir, f"rtl_layer_{LAYER_FILE_IDX['BOT']}.hex"), LAYER_SHAPES["BOT"]
    )

    input_img = load_plain_image(args.input_img) if args.input_img else None
    if input_img is None:
        # fallback: pakai file masked_input kalau ada png-nya di folder testcase
        # (isi manual path lewat --input-img kalau tidak ketemu otomatis)
        pass

    output_img = load_plain_image(os.path.join(out_dir, "output_rtl.png"))

    save_path = args.save or os.path.join(out_dir, "viz", "context_encoder_stack.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_context_encoder(
        input_img, output_img,
        encoder_arrs, bottleneck_arr, decoder_arrs,
        n_show=args.n_show,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
