#!/usr/bin/env python3
"""
plot_conv_pipeline.py — Diagram alur Input -> Conv Layer 1..N -> Output
==========================================================================
Merangkai PNG-PNG yang sudah dihasilkan oleh visualize_rtl_hex.py
(di outputs/viz/) menjadi satu diagram pipeline horizontal, mirip
contoh: kotak Input (kuning) -> thumbnail tiap layer -> kotak Output (biru),
disambung panah.

Cocok dipakai langsung dari folder:
    /mnt/ssd_eda/projects/PME/GAN/Context_Encoder_v2/verilog/Integrate5.4/outputs

Usage:
    python3 plot_conv_pipeline.py
    python3 plot_conv_pipeline.py --per-row 7 --save pipeline.png
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FRAC_W = 8  # Q8.8 fixed-point, sama seperti visualize_rtl_hex.py


# ── Helper: load file .hex generik jadi gambar PNG (untuk input) ────────────
def load_hex_as_array(hex_path: str, shape) -> "np.ndarray | None":
    """Baca file hex (signed Q8.8) jadi numpy array shape (C,H,W)."""
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
        print(f"  [WARN] {hex_path}: expect {expected} values, got {len(raw)}")
        if len(raw) < expected:
            raw.extend([0] * (expected - len(raw)))
        else:
            raw = raw[:expected]

    arr = np.array(raw, dtype=np.float32) / (2 ** FRAC_W)
    return arr.reshape(c, h, w)


def hex_to_png(hex_path: str, shape, save_path: str, mode: str = "minmax"):
    """Convert file hex input jadi PNG RGB (kalau belum ada PNG-nya)."""
    arr = load_hex_as_array(hex_path, shape)
    if arr is None:
        return None
    hwc = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    if mode == "tanh":
        clip = np.clip(hwc, -1.0, 1.0)
        img_uint8 = ((clip + 1.0) * 127.5).astype(np.uint8)
    else:  # minmax
        vmin, vmax = hwc.min(), hwc.max()
        if vmax - vmin < 1e-6:
            img_uint8 = np.zeros_like(hwc, dtype=np.uint8)
        else:
            img_uint8 = ((hwc - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    img = Image.fromarray(img_uint8, mode="RGB")
    img.save(save_path)
    print(f"  Converted {hex_path} -> {save_path}")
    return save_path


# ── Fungsi utama: gambar diagram pipeline ────────────────────────────────────
def plot_pipeline(
    input_path,
    layer_items,          # list of (label, image_path)
    output_path,
    save_path=None,
    per_row=7,
    cell_size=1.9,
    input_color="#f6c453",
    output_color="#a9cce3",
    title=None,
):
    """
    Buat diagram: [Input] -> Conv Layer 1 -> Conv Layer 2 -> ... -> [Output]

    Parameters
    ----------
    input_path : str
        Path gambar input (misal gambar masked).
    layer_items : list of (label, path)
        Tiap item = 1 conv layer, label ditampilkan di bawah thumbnail.
    output_path : str
        Path gambar hasil akhir/reconstruction.
    save_path : str, optional
        Simpan hasil ke file ini.
    per_row : int
        Berapa kotak per baris sebelum wrap ke baris berikutnya.
    """
    steps = [("Input\n(Gambar Masked)", input_path, "input")]
    steps += [(label, path, "layer") for label, path in layer_items]
    steps += [("Output\n(Reconstruction)", output_path, "output")]

    n = len(steps)
    rows = (n + per_row - 1) // per_row
    row_gap = 1.45  # jarak vertikal antar baris, adequate space for labels

    fig_w = per_row * cell_size
    fig_h = rows * cell_size * (row_gap / 1.0) * 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, per_row - 0.5)
    ax.set_ylim(-0.5, (rows - 1) * row_gap + 0.75)
    ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=20)

    inset = 0.34  # setengah ukuran gambar di dalam kotak

    for idx, (label, path, kind) in enumerate(steps):
        row = idx // per_row
        col = idx % per_row
        cx, cy = float(col), float(row) * row_gap

        # kotak latar
        if kind == "input":
            face, edge, lw = input_color, "#8a6d1f", 1.8
        elif kind == "output":
            face, edge, lw = output_color, "#2e5f8a", 1.8
        else:
            face, edge, lw = "white", "#999999", 1.0

        box = FancyBboxPatch(
            (cx - 0.45, cy - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw, edgecolor=edge, facecolor=face, zorder=1,
        )
        ax.add_patch(box)

        # gambar thumbnail
        if path and os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            ax.imshow(
                img,
                extent=[cx - inset, cx + inset, cy - inset, cy + inset],
                zorder=2, aspect="auto",
            )
        else:
            ax.text(cx, cy, "?", ha="center", va="center", fontsize=16, zorder=2)

        # label di bawah kotak
        ax.text(cx, cy + 0.58, label, ha="center", va="top", fontsize=8.5, zorder=3)

        # panah ke elemen berikutnya
        if idx < n - 1:
            nrow = (idx + 1) // per_row
            ncol = (idx + 1) % per_row
            ny = float(nrow) * row_gap
            if nrow == row:
                ax.annotate(
                    "", xy=(ncol - 0.45, cy), xytext=(cx + 0.45, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6),
                    zorder=1,
                )
            else:
                ax.annotate(
                    "", xy=(ncol - 0.45, ny), xytext=(cx, cy + 0.45),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#555555", lw=1.6,
                        connectionstyle="angle3,angleA=90,angleB=180",
                    ),
                    zorder=1,
                )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nDiagram disimpan ke: {save_path}")
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs",
                         help="folder yang berisi rtl_layer_*.hex, output_rtl.png, py_ref/, viz/")
    parser.add_argument("--per-row", type=int, default=7)
    parser.add_argument("--save", default=None, help="path file output, default: <outputs-dir>/viz/pipeline.png")
    args = parser.parse_args()

    out_dir = args.outputs_dir
    viz_dir = os.path.join(out_dir, "viz")
    ref_dir = os.path.join(out_dir, "py_ref")

    LAYER_NAMES = {
        0: "E1", 1: "E2", 2: "E3", 3: "E4", 4: "E5",
        5: "BOT",
        6: "D1", 7: "D2", 8: "D3", 9: "D4", 10: "D5",
    }
    LAYER_SHAPES = {
        0: (32, 32, 32), 1: (64, 16, 16), 2: (128, 8, 8), 3: (256, 4, 4),
        4: (512, 2, 2), 5: (512, 2, 2), 6: (256, 4, 4), 7: (128, 8, 8),
        8: (64, 16, 16), 9: (32, 32, 32), 10: (32, 64, 64),
    }

    # ---- 1. Prepare input image (convert from hex if PNG is unavailable) ----
    input_hex = os.path.join(ref_dir, "input_image.hex")
    input_png = os.path.join(viz_dir, "input_image.png")
    if not os.path.exists(input_png):
        # asumsi input shape sama seperti output: (3, 64, 64). Sesuaikan kalau beda.
        hex_to_png(input_hex, (3, 64, 64), input_png, mode="minmax")

    # ---- 2. daftar layer conv (pakai channel-grid yang sudah ada di viz/) ----
    layer_items = []
    for idx in range(0, 11):  # layer 0..10 = E1..D5 (layer 11 = output, dipisah)
        name = LAYER_NAMES[idx]
        png_path = os.path.join(viz_dir, f"layer_{idx}_{name}_channels.png")
        layer_items.append((f"Conv Layer {idx+1}\n({name})", png_path))

    # ---- 3. gambar output akhir ----
    output_png = os.path.join(out_dir, "output_rtl.png")
    if not os.path.exists(output_png):
        output_png = os.path.join(viz_dir, "layer_11_OUT_output.png")

    save_path = args.save or os.path.join(viz_dir, "pipeline.png")

    plot_pipeline(
        input_path=input_png,
        layer_items=layer_items,
        output_path=output_png,
        save_path=save_path,
        per_row=args.per_row,
        title="RTL Forward Pass: Input -> Encoder-Decoder -> Output",
    )


if __name__ == "__main__":
    main()
