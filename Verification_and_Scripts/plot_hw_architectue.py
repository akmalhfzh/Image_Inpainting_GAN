#!/usr/bin/env python3
"""
plot_proposed_hw.py

Generate publication-style diagrams for a proposed FPGA accelerator
for Context Encoder GAN image inpainting.

Figures:
  1. proposed_hw_architecture.png
     System-level FPGA hardware architecture.
  2. cnn_hardware_mapping.png
     Context Encoder Generator mapped to the hardware execution engine.
  3. memory_dataflow.png
     DDR <-> BRAM/tile buffer <-> MAC dataflow for one convolution operation.

The architecture is derived from the supplied RTL and Python model:
- 64x64 RGB input
- 5 encoder stages
- 1x1 bottleneck
- 5 decoder stages
- 3-channel 64x64 output
- 16-bit fixed-point data, FRAC_W=8
- 48-bit accumulation
- 4x4 kernel -> 16 parallel MACs
- two-stage MAC pipeline
- AXI4 master access to DDR
- ping-pong feature-map buffers
- BRAM weight cache and local ifmap tile
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path


OUT = Path("figures")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Common drawing helpers
# ---------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.5,
        fontsize=9, radius=0.03, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, weight=weight,
        wrap=True
    )
    return p


def arrow(ax, x1, y1, x2, y2, text=None, fontsize=8,
          connectionstyle="arc3", lw=1.4, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        connectionstyle=connectionstyle
    )
    ax.add_patch(a)
    if text:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2,
            text, fontsize=fontsize,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
        )
    return a


def setup_ax(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=15)


# ---------------------------------------------------------------------
# 1. Proposed system-level hardware architecture
# ---------------------------------------------------------------------
def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(15, 8))
    setup_ax(ax, "Proposed FPGA Hardware Architecture for Context Encoder GAN Inference")

    # Host/control side
    box(ax, 0.02, 0.72, 0.14, 0.14,
        "Host CPU / PS\n\nControl + image transfer",
        fc="white", fontsize=9, weight="bold")

    box(ax, 0.21, 0.75, 0.12, 0.09,
        "AXI-Lite\nControl Registers",
        fc="white", fontsize=8)

    box(ax, 0.21, 0.59, 0.12, 0.09,
        "AXI4 Master\nData Interface",
        fc="white", fontsize=8)

    # DDR
    box(ax, 0.02, 0.25, 0.14, 0.30,
        "DDR Memory\n\nInput Image\nFeature Maps A/B\nWeights\nOutput Image",
        fc="white", fontsize=9, weight="bold")

    # Main accelerator boundary
    ax.add_patch(Rectangle(
        (0.38, 0.10), 0.59, 0.78,
        fill=False, linewidth=2.0
    ))
    ax.text(0.675, 0.855, "FPGA CNN Accelerator", ha="center",
            va="center", fontsize=12, weight="bold")

    # Controller
    box(ax, 0.42, 0.68, 0.18, 0.12,
        "generator_top\nLayer Controller / FSM",
        fc="white", fontsize=8, weight="bold")

    # Address generator
    box(ax, 0.42, 0.49, 0.18, 0.11,
        "Address Generator\nifmap / weight addresses",
        fc="white", fontsize=8)

    # Conv core
    box(ax, 0.67, 0.68, 0.22, 0.12,
        "conv_unified\nConvolution / Transpose-Conv\nExecution Engine",
        fc="white", fontsize=8, weight="bold")

    # Local memories
    box(ax, 0.67, 0.49, 0.10, 0.11,
        "BRAM\nWeight Cache",
        fc="white", fontsize=8)

    box(ax, 0.79, 0.49, 0.10, 0.11,
        "Local Tile\n16 × IFMAP",
        fc="white", fontsize=8)

    # MAC path
    box(ax, 0.42, 0.27, 0.14, 0.11,
        "16 Parallel\n16×16 MACs",
        fc="white", fontsize=8, weight="bold")

    box(ax, 0.60, 0.27, 0.14, 0.11,
        "2-Stage\nPipeline",
        fc="white", fontsize=8)

    box(ax, 0.78, 0.27, 0.11, 0.11,
        "48-bit\nAccumulator",
        fc="white", fontsize=8)

    # Activation/writeback
    box(ax, 0.42, 0.12, 0.16, 0.09,
        "Activation\nLReLU / ReLU / Tanh",
        fc="white", fontsize=7.5)

    box(ax, 0.63, 0.12, 0.14, 0.09,
        "Quantize /\nSaturate",
        fc="white", fontsize=8)

    box(ax, 0.82, 0.12, 0.09, 0.09,
        "AXI4\nWriteback",
        fc="white", fontsize=8)

    # External connections
    arrow(ax, 0.16, 0.79, 0.21, 0.795, "control", fontsize=7)
    arrow(ax, 0.16, 0.75, 0.21, 0.635, "data", fontsize=7)
    arrow(ax, 0.33, 0.795, 0.42, 0.74, "start/config", fontsize=7)
    arrow(ax, 0.33, 0.635, 0.67, 0.74, "AXI4", fontsize=7)
    arrow(ax, 0.16, 0.40, 0.42, 0.54, "DDR read", fontsize=7)
    arrow(ax, 0.16, 0.32, 0.82, 0.165, "DDR write", fontsize=7)

    # Internal dataflow
    arrow(ax, 0.51, 0.68, 0.51, 0.60, "address/config", fontsize=7)
    arrow(ax, 0.60, 0.545, 0.67, 0.545, "addresses", fontsize=7)
    arrow(ax, 0.77, 0.60, 0.77, 0.49, "weights", fontsize=7)
    arrow(ax, 0.84, 0.60, 0.84, 0.49, "ifmap", fontsize=7)
    arrow(ax, 0.72, 0.49, 0.49, 0.38, "weight tile", fontsize=7)
    arrow(ax, 0.84, 0.49, 0.49, 0.38, "ifmap tile", fontsize=7)
    arrow(ax, 0.56, 0.325, 0.60, 0.325)
    arrow(ax, 0.74, 0.325, 0.78, 0.325)
    arrow(ax, 0.855, 0.27, 0.50, 0.21)
    arrow(ax, 0.58, 0.165, 0.63, 0.165)
    arrow(ax, 0.77, 0.165, 0.82, 0.165)

    # Ping-pong note
    ax.text(
        0.51, 0.92,
        "DDR feature-map ping-pong: FMap-A ↔ FMap-B",
        ha="center", va="center", fontsize=9, style="italic"
    )

    # Legend
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label="Hardware block"),
        Line2D([0], [0], color="black", lw=1.5, label="Data/control path")
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "proposed_hw_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 2. CNN model -> hardware mapping
# ---------------------------------------------------------------------
def draw_cnn_mapping():
    fig, ax = plt.subplots(figsize=(16, 7))
    setup_ax(ax, "Context Encoder Generator: CNN Layer-to-Hardware Mapping")

    layers = [
        ("Input", "64×64×3", "DDR\ninput image"),
        ("E1", "32×32×32", "Conv2D\n4×4 / s2 / p1\nLReLU"),
        ("E2", "16×16×64", "Conv2D + BN\n4×4 / s2 / p1\nLReLU"),
        ("E3", "8×8×128", "Conv2D + BN\n4×4 / s2 / p1\nLReLU"),
        ("E4", "4×4×256", "Conv2D + BN\n4×4 / s2 / p1\nLReLU"),
        ("E5", "2×2×512", "Conv2D + BN\n4×4 / s2 / p1\nLReLU"),
        ("B1", "2×2×512", "1×1 Conv\nBottleneck"),
        ("D1", "4×4×256", "ConvTranspose2D\n4×4 / s2 / p1\nReLU"),
        ("D2", "8×8×128", "ConvTranspose2D\n4×4 / s2 / p1\nReLU"),
        ("D3", "16×16×64", "ConvTranspose2D\n4×4 / s2 / p1\nReLU"),
        ("D4", "32×32×32", "ConvTranspose2D\n4×4 / s2 / p1\nReLU"),
        ("D5", "64×64×32", "ConvTranspose2D\n4×4 / s2 / p1\nReLU"),
        ("Out", "64×64×3", "3×3 Conv\nTanh\nDDR output")
    ]

    x0 = 0.025
    y = 0.58
    w = 0.067
    h = 0.22
    gap = 0.009

    for i, (name, shape, op) in enumerate(layers):
        x = x0 + i * (w + gap)
        fc = "white"

        box(ax, x, y, w, h,
            f"{name}\n{shape}\n\n{op}",
            fc=fc, fontsize=6.5,
            weight="bold" if name in ("Input", "B1", "Out") else "normal",
            radius=0.015)

        if i < len(layers) - 1:
            arrow(ax, x + w, y + h/2, x + w + gap, y + h/2,
                  lw=1.1)

    # Hardware engine below
    box(ax, 0.16, 0.16, 0.18, 0.17,
        "Shared Hardware Engine\n\nconv_unified\nAddress Generator",
        fontsize=8, weight="bold")

    box(ax, 0.40, 0.16, 0.18, 0.17,
        "Local Memory\n\nWeight Cache\nIFMAP Tile\nBRAM",
        fontsize=8)

    box(ax, 0.64, 0.16, 0.18, 0.17,
        "Compute\n\n16 parallel MACs\n2-stage pipeline\n48-bit accumulator",
        fontsize=8, weight="bold")

    box(ax, 0.86, 0.16, 0.10, 0.17,
        "DDR\n\nWeights\nFMaps\nI/O",
        fontsize=7.5)

    # Dashed conceptual mapping lines
    for i in range(len(layers)):
        x = x0 + i * (w + gap) + w/2
        target_x = 0.16 + (i % 3) * 0.24 + 0.09
        arrow(ax, x, y, target_x, 0.33,
              lw=0.8, style="-|>", connectionstyle="arc3,rad=0.12")

    ax.text(
        0.50, 0.055,
        "The same hardware datapath is reused sequentially across all 12 generator layers.",
        ha="center", va="center", fontsize=9, style="italic"
    )

    fig.tight_layout()
    fig.savefig(OUT / "cnn_hardware_mapping.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 3. Detailed memory / MAC dataflow
# ---------------------------------------------------------------------
def draw_memory_dataflow():
    fig, ax = plt.subplots(figsize=(14, 7))
    setup_ax(ax, "Per-Tile Memory and Compute Dataflow")

    # DDR side
    box(ax, 0.04, 0.67, 0.18, 0.19,
        "DDR Memory\n\nWeights\nInput / Feature Maps\nOutput Feature Maps",
        fontsize=9, weight="bold")

    box(ax, 0.29, 0.70, 0.17, 0.13,
        "AXI4 Read / Write\n4-byte aligned bursts",
        fontsize=8)

    # Address path
    box(ax, 0.29, 0.45, 0.17, 0.13,
        "Address Generator\n\nPer-pixel + per-row\nburst address generation",
        fontsize=8)

    # BRAM/local
    box(ax, 0.54, 0.70, 0.16, 0.13,
        "BRAM Weight Cache\n\n16 weight entries\nper input channel",
        fontsize=8)

    box(ax, 0.54, 0.45, 0.16, 0.13,
        "IFMAP Tile Buffer\n\n16 values\nfor 4×4 kernel",
        fontsize=8)

    # MAC
    box(ax, 0.77, 0.58, 0.18, 0.20,
        "MAC Array\n\n16 × (16×16)\nparallel multipliers\n\nBinary adder tree\n\n2-stage pipeline",
        fontsize=8.5, weight="bold")

    # Acc/activation
    box(ax, 0.77, 0.28, 0.18, 0.15,
        "48-bit Accumulator\n\nBias + row_sum\nfixed-point scaling",
        fontsize=8)

    box(ax, 0.51, 0.22, 0.16, 0.15,
        "Activation\n\nLReLU / ReLU / Tanh",
        fontsize=8)

    box(ax, 0.27, 0.22, 0.16, 0.15,
        "Saturation\n16-bit output",
        fontsize=8)

    box(ax, 0.04, 0.22, 0.14, 0.15,
        "Writeback\n\nOutput pixel",
        fontsize=8)

    # Data paths
    arrow(ax, 0.22, 0.77, 0.29, 0.765, "AXI4", fontsize=7)
    arrow(ax, 0.46, 0.765, 0.54, 0.765, "weights", fontsize=7)
    arrow(ax, 0.46, 0.515, 0.54, 0.515, "ifmap", fontsize=7)
    arrow(ax, 0.70, 0.765, 0.77, 0.70, "W[0:15]", fontsize=7)
    arrow(ax, 0.70, 0.515, 0.77, 0.66, "X[0:15]", fontsize=7)
    arrow(ax, 0.86, 0.58, 0.86, 0.43, "row_sum", fontsize=7)
    arrow(ax, 0.77, 0.355, 0.67, 0.295, "acc", fontsize=7)
    arrow(ax, 0.51, 0.295, 0.43, 0.295, "16-bit", fontsize=7)
    arrow(ax, 0.27, 0.295, 0.18, 0.295, "pixel", fontsize=7)

    # Control/address relation
    arrow(ax, 0.375, 0.58, 0.54, 0.58,
          "burst addresses", fontsize=7, style="-|>")
    ax.text(
        0.375, 0.40,
        "Registered address path\nreduces critical path from\nconfiguration → address → AXI burst",
        ha="center", va="center", fontsize=8
    )

    # Ping-pong memory concept
    ax.text(
        0.13, 0.12,
        "Feature-map storage uses ping-pong DDR regions:\n"
        "current layer reads one region while the next layer writes the other.",
        ha="center", va="center", fontsize=8.5, style="italic"
    )

    fig.tight_layout()
    fig.savefig(OUT / "memory_dataflow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    draw_system_architecture()
    draw_cnn_mapping()
    draw_memory_dataflow()

    print("Generated:")
    for f in [
        "proposed_hw_architecture.png",
        "cnn_hardware_mapping.png",
        "memory_dataflow.png",
    ]:
        print("  ", OUT / f)


if __name__ == "__main__":
    main()

