# pyrefly: ignore [missing-import]
import pynq
from pynq import Overlay, allocate
import numpy as np
import cv2
import time
import os

# =====================================================================
# 1. INIT & LOAD OVERLAY
# =====================================================================
print("[PYNQ] Loading Bitstream...")
from pynq import Bitstream, MMIO
bit = Bitstream("design_1_wrapper.bit")
bit.download()
gan_ip = MMIO(0x40000000, 0x1000)  # Bypass HWH parser bug, map manually

# =====================================================================
# 2. ALLOCATE MEMORY BUFFERS
# =====================================================================
print("[PYNQ] Allocating CMA Memory...")
# Allocate buffers for images and feature maps
# We need enough space. FMAP max size is 64x64x64 x 2 bytes ~ 512KB. Let's allocate 1MB each.
img_buf = allocate(shape=(3, 64, 64), dtype=np.int16)
fmap_a  = allocate(shape=(1024 * 1024,), dtype=np.int16)
fmap_a[:] = -1
fmap_a.flush()
fmap_b  = allocate(shape=(1024 * 1024,), dtype=np.int16)
fmap_b[:] = -1
fmap_b.flush()
out_buf = allocate(shape=(3, 64, 64), dtype=np.int16)
out_buf[:] = -1
out_buf.flush()
weight_buf = allocate(shape=(6 * 1024 * 1024,), dtype=np.int16) # ~12MB for 6M words of weights

# =====================================================================
# 3. WRITE MEMORY MAP REGISTERS
# =====================================================================
# Based on gan_axi_wrapper.v
print("[PYNQ] Configuring AXI-Lite Registers...")
gan_ip.write(0x08, img_buf.device_address)
gan_ip.write(0x0C, fmap_a.device_address)
gan_ip.write(0x10, fmap_b.device_address)
gan_ip.write(0x14, out_buf.device_address)
gan_ip.write(0x20, weight_buf.device_address)

# ── Debug: Print all buffer physical addresses ──
print(f"[DEBUG] img_buf    phys=0x{img_buf.device_address:08X}  size={img_buf.nbytes} bytes")
print(f"[DEBUG] fmap_a     phys=0x{fmap_a.device_address:08X}  size={fmap_a.nbytes} bytes")
print(f"[DEBUG] fmap_b     phys=0x{fmap_b.device_address:08X}  size={fmap_b.nbytes} bytes")
print(f"[DEBUG] out_buf    phys=0x{out_buf.device_address:08X}  size={out_buf.nbytes} bytes")
print(f"[DEBUG] weight_buf phys=0x{weight_buf.device_address:08X}  size={weight_buf.nbytes} bytes")

# ── Debug: Read back registers to verify ──
print(f"[DEBUG] Reg 0x08 (img)    = 0x{gan_ip.read(0x08):08X}")
print(f"[DEBUG] Reg 0x0C (fmapA)  = 0x{gan_ip.read(0x0C):08X}")
print(f"[DEBUG] Reg 0x10 (fmapB)  = 0x{gan_ip.read(0x10):08X}")
print(f"[DEBUG] Reg 0x14 (out)    = 0x{gan_ip.read(0x14):08X}")
print(f"[DEBUG] Reg 0x20 (weight) = 0x{gan_ip.read(0x20):08X}")

# =====================================================================
# 4A. LOAD WEIGHTS TO DDR (CMA)
# =====================================================================
print("[PYNQ] Loading Weights to DDR...")
weight_files = {
    0: 'E1_weight.hex',  1: 'E2_weight.hex',  2: 'E3_weight.hex',
    3: 'E4_weight.hex',  4: 'E5_weight.hex',  5: 'BOT_weight.hex',
    6: 'D1_weight.hex',  7: 'D2_weight.hex',  8: 'D3_weight.hex',
    9: 'D4_weight.hex', 10: 'D5_weight.hex', 11: 'OUT_weight.hex'
}
weight_offsets = {
    0: 0, 1: 3072, 2: 68608, 3: 199680, 4: 723968, 5: 2821120,
    6: 3083264, 7: 5180672, 8: 5705088, 9: 5836288, 10: 5869088, 11: 5901888
}

for layer, fname in weight_files.items():
    path = os.path.join("weights_hex", fname)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            offset = weight_offsets[layer]
            # Bulk parse hex strings to numpy array and cast to signed 16-bit int
            vals = [int(line.strip(), 16) for line in lines if line.strip() and not line.strip().startswith('/')]
            if vals:
                vals_np = np.array(vals, dtype=np.uint16).astype(np.int16)
                weight_buf[offset : offset + len(vals_np)] = vals_np
weight_buf.flush()

# =====================================================================
# 4B. LOAD BIASES VIA MMIO
# =====================================================================
print("[PYNQ] Loading Biases to FPGA Registers...")
bias_files = {
    0: 'E1_bias.hex', 1: 'E2_bias.hex', 2: 'E3_bias.hex',
    3: 'E4_bias.hex', 4: 'E5_bias.hex', 5: 'BOT_bias.hex',
    6: 'D1_bias.hex', 7: 'D2_bias.hex', 8: 'D3_bias.hex',
    9: 'D4_bias.hex', 10: 'D5_bias.hex', 11: 'OUT_bias.hex'
}

for layer, fname in bias_files.items():
    path = os.path.join("weights_hex", fname)
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            for addr, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('/'): continue
                val = int(line, 16)
                
                # Config register 0x18: {layer[27:24], type[19:18], addr[17:0]}
                # type=3 for bias
                config_val = (layer << 24) | (3 << 18) | (addr & 0x3FFFF)
                gan_ip.write(0x18, config_val)
                gan_ip.write(0x1C, val) # Write data

# =====================================================================
# 5. PREPARE INPUT IMAGE
# =====================================================================
print("[PYNQ] Preparing Input Image...")
# CLI: pynq_runner.py [img_path] [mask_size]
# Preprocessing MUST match verify/inference.py (the golden reference the RTL
# was verified against), otherwise the FPGA output cannot be compared to it:
#   - resize 64x64 with PIL BICUBIC (torchvision Resize(..., Image.BICUBIC))
#   - normalize (x/127.5 - 1)  == ToTensor + Normalize(0.5, 0.5)
#   - quantize Q7.8 (round * 256, int16)
#   - center square mask set to 1.0  -> 256 in Q7.8
#   - default mask_size = 16 (inference.py default), override via argv[2]
import sys
from PIL import Image
img_path  = sys.argv[1] if len(sys.argv) > 1 else "test_images/kakek.png"
mask_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
if os.path.exists(img_path):
    img = Image.open(img_path).convert("RGB").resize((64, 64), Image.BICUBIC)
    img = np.asarray(img, dtype=np.float32)  # HWC, RGB, [0,255]

    img_normalized = (img / 127.5) - 1.0
    img_q = np.clip(np.round(img_normalized * 256), -32768, 32767).astype(np.int16)

    # HWC -> CHW for DDR layout
    img_chw = np.transpose(img_q, (2, 0, 1))

    # Apply centered square mask only if mask_size > 0
    if mask_size > 0:
        i = (64 - mask_size) // 2
        img_chw[:, i:i+mask_size, i:i+mask_size] = 256
        print(f"[PYNQ] Input '{img_path}'  mask={mask_size}x{mask_size} @ ({i},{i})")
    else:
        print(f"[PYNQ] Input '{img_path}'  mask disabled (mask_size=0)")

    # Copy the (possibly masked) image data to DDR
    np.copyto(img_buf, img_chw)
    img_buf.flush()  # Ensure it's written to DDR
else:
    print(f"Warning: {img_path} not found. Using zeroed input.")

# =====================================================================
# 6. START HARDWARE INFERENCE
# =====================================================================
print("[PYNQ] Starting Hardware Inference...")
start_time = time.time()

# Write 1 to start bit (0x00)
gan_ip.write(0x00, 1)

# Wait for Done bit (bit 0 of register 0x04)
done = 0
last_layer = -1
last_state = -1
while done == 0:
    status = gan_ip.read(0x04)
    done = status & 0x1
    layer = (status >> 4) & 0xF
    state = (status >> 8) & 0xF
    
    if layer != last_layer or state != last_state:
        state_names = {0: "IDLE", 1: "CONV_START", 2: "CONV_WAIT", 3: "NEXT_LAYER", 4: "DONE"}
        s_name = state_names.get(state, f"UNKNOWN({state})")
        print(f"[DEBUG] Layer Index: {layer} | FSM State: {s_name} | done={done}")
        last_layer = layer
        last_state = state
        
    time.sleep(0.05)

end_time = time.time()
print(f"[PYNQ] Inference Complete! Time taken: {(end_time - start_time)*1000:.2f} ms")

# =====================================================================
# 7. READ OUTPUT AND SAVE
# =====================================================================
out_buf.invalidate() # Refresh from DDR
fmap_a.invalidate()
fmap_b.invalidate()
out_chw = np.copy(out_buf)

# ── Debug: Dump raw output buffer statistics ──
print("\n" + "="*60)
print("[DEBUG] === RAW OUTPUT BUFFER (Q8.8 int16) ===")
print(f"  Shape: {out_chw.shape}, dtype: {out_chw.dtype}")
print(f"  min={out_chw.min()}, max={out_chw.max()}, mean={out_chw.mean():.2f}, std={out_chw.std():.2f}")
print(f"  Non-zero count: {np.count_nonzero(out_chw)} / {out_chw.size}")
print(f"  Sample [0,0,:8]  = {out_chw[0, 0, :8]}")
print(f"  Sample [1,0,:8]  = {out_chw[1, 0, :8]}")
print(f"  Sample [2,0,:8]  = {out_chw[2, 0, :8]}")
print(f"  Sample [0,32,:8] = {out_chw[0, 32, :8]}")

# ── Debug: Check intermediate fmap buffers ──
# Layer 0 output goes to fmap_b (ping_pong=0 → ofmap=fmap_b for even layers)
# Shape: 32 channels × 32×32 = 32768 words
fmap_a_arr = np.array(fmap_a[:32768], dtype=np.int16)
fmap_b_arr = np.array(fmap_b[:32768], dtype=np.int16)
print("\n[DEBUG] === INTERMEDIATE FMAP_A (first 32768 words) ===")
print(f"  min={fmap_a_arr.min()}, max={fmap_a_arr.max()}, mean={fmap_a_arr.mean():.2f}, std={fmap_a_arr.std():.2f}")
print(f"  Non-zero: {np.count_nonzero(fmap_a_arr)} / {fmap_a_arr.size}")
print(f"  Sample [:16] = {fmap_a_arr[:16]}")

print("\n[DEBUG] === INTERMEDIATE FMAP_B (first 32768 words) ===")
print(f"  min={fmap_b_arr.min()}, max={fmap_b_arr.max()}, mean={fmap_b_arr.mean():.2f}, std={fmap_b_arr.std():.2f}")
print(f"  Non-zero: {np.count_nonzero(fmap_b_arr)} / {fmap_b_arr.size}")
print(f"  Sample [:16] = {fmap_b_arr[:16]}")

# ── Debug: Check weight buffer ──
wt_sample = np.array(weight_buf[:64], dtype=np.int16)
print("\n[DEBUG] === WEIGHT BUFFER (first 64 words) ===")
print(f"  Non-zero: {np.count_nonzero(wt_sample)} / {wt_sample.size}")
print(f"  Sample [:16] = {wt_sample[:16]}")
print("="*60 + "\n")

# Convert Q8.8 to [-1, 1] then [0, 255]
out_float = out_chw.astype(np.float32) / 256.0
out_img_normalized = (out_float + 1.0) * 127.5
out_img_clamped = np.clip(out_img_normalized, 0, 255).astype(np.uint8)

# ── Composite: paste original pixels back to non-masked region ──────────────
# Matches inference.py behavior:
#   inpainted = original.copy()
#   inpainted[mask_region] = generator_output[mask_region]
# Without this, the output looks wrong even outside the mask because
# the generator is not trained to reproduce pixel-perfect non-mask content.
if os.path.exists(img_path):
    # Re-read the input image (pre-masked or clean) resized to 64x64
    base_img = Image.open(img_path).convert("RGB").resize((64, 64), Image.BICUBIC)
    base_arr = np.asarray(base_img, dtype=np.uint8)   # HWC [0,255]
    base_chw = np.transpose(base_arr, (2, 0, 1))      # CHW [0,255]

    # Start from the base image (contains original pixels outside mask)
    composite_chw = base_chw.copy()

    if mask_size > 0:
        # Mask position is known explicitly
        i = (64 - mask_size) // 2
        composite_chw[:, i:i+mask_size, i:i+mask_size] = \
            out_img_clamped[:, i:i+mask_size, i:i+mask_size]
        print(f"[PYNQ] Composite: generator output pasted into mask region ({i},{i}) size {mask_size}x{mask_size}")
    else:
        # mask_size=0: input is a pre-masked image (e.g., created by create_mask.py)
        # Detect the masked region by finding pixels that are (near-)white = (255,255,255)
        # create_mask.py fills the mask with pure white (255,255,255)
        mask_region_hw = np.all(base_arr >= 250, axis=2)   # HW bool: True where masked
        n_masked = mask_region_hw.sum()
        if n_masked > 0:
            # Paste generator output ONLY into detected mask region
            # non-masked pixels stay as original (from base image)
            composite_chw[:, mask_region_hw] = out_img_clamped[:, mask_region_hw]
            print(f"[PYNQ] Composite: detected {n_masked} masked pixels, filled with generator output")
        else:
            # No white mask detected — fall back to raw generator output
            composite_chw = out_img_clamped
            print("[PYNQ] Composite: no mask region detected, using raw generator output")
else:
    composite_chw = out_img_clamped

# ── Mean color correction: shift inpainted region to match surrounding area ──
# When the generator output is saturated (e.g., too white), shift the inpainted
# region's mean color to match the surrounding border pixels (ring of 4px around
# the mask). This is a simple but effective post-processing step.
try:
    if mask_size > 0:
        mi = (64 - mask_size) // 2
        mask_region_slice = (slice(None), slice(mi, mi + mask_size), slice(mi, mi + mask_size))
    elif os.path.exists(img_path):
        mask_region_hw = np.all(base_arr >= 250, axis=2)
        if not mask_region_hw.any():
            raise ValueError("no mask detected")
        ys, xs = np.where(mask_region_hw)
        mi_y, mi_x = ys.min(), xs.min()
        ms_y = ys.max() - mi_y + 1
        ms_x = xs.max() - mi_x + 1
        mask_region_slice = (slice(None), slice(mi_y, mi_y + ms_y), slice(mi_x, mi_x + ms_x))
        mi = mi_y  # for border sampling below
        mask_size_eff = ms_y
    else:
        raise ValueError("no img_path")

    if mask_size > 0:
        mask_size_eff = mask_size

    # Sample border ring: 4 pixels outside the mask boundary (original skin pixels)
    border_top    = max(mi - 4, 0)
    border_bottom = min(mi + mask_size_eff + 4, 64)
    border_left   = max(mi - 4, 0)
    border_right  = min(mi + mask_size_eff + 4, 64)

    # Get the original (pre-mask) image for border sampling
    if os.path.exists(img_path):
        # Use original unmasked image if reference (argv[3]) is available
        if len(sys.argv) > 3:
            from pathlib import Path as _Path
            _ref = _Path(sys.argv[3])
            if _ref.is_file():
                from PIL import Image as _Image
                _orig_unmasked = np.array(_Image.open(_ref).convert("RGB").resize((64, 64), Image.BICUBIC), dtype=np.uint8)
                orig_for_border = np.transpose(_orig_unmasked, (2, 0, 1))
            else:
                orig_for_border = base_chw
        else:
            orig_for_border = base_chw
    else:
        orig_for_border = composite_chw

    border_region = orig_for_border[
        :,
        border_top:border_bottom,
        border_left:border_right
    ]
    # Mask out the inpainted region itself from border stats
    bh, bw = border_region.shape[1], border_region.shape[2]
    inner_mask = np.zeros((bh, bw), dtype=bool)
    iy0 = mi - border_top
    ix0 = mi - border_left
    inner_mask[iy0:iy0 + mask_size_eff, ix0:ix0 + mask_size_eff] = True

    border_mean = np.array([
        border_region[c][~inner_mask].mean() for c in range(3)
    ], dtype=np.float32)

    # Compute mean of inpainted region in composite
    inpaint_region = composite_chw[mask_region_slice].astype(np.float32)
    inpaint_mean = inpaint_region.mean(axis=(1, 2))

    # Apply per-channel shift so inpainted region matches border mean
    shift = border_mean - inpaint_mean
    corrected = composite_chw.copy().astype(np.float32)
    corrected[mask_region_slice] += shift[:, None, None]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Convert CHW to HWC for OpenCV
    out_img_hwc = np.transpose(corrected, (1, 2, 0))
    out_img_bgr = cv2.cvtColor(out_img_hwc, cv2.COLOR_RGB2BGR)
    print(f"[PYNQ] Color correction: border_mean=({border_mean[0]:.1f},{border_mean[1]:.1f},{border_mean[2]:.1f})"
          f"  inpaint_mean=({inpaint_mean[0]:.1f},{inpaint_mean[1]:.1f},{inpaint_mean[2]:.1f})"
          f"  shift=({shift[0]:.1f},{shift[1]:.1f},{shift[2]:.1f})")
except Exception as e:
    print(f"[PYNQ] Color correction skipped: {e}")
    # Fallback: use composite as-is
    out_img_hwc = np.transpose(composite_chw, (1, 2, 0))
    out_img_bgr = cv2.cvtColor(out_img_hwc, cv2.COLOR_RGB2BGR)

cv2.imwrite("output_pynq.jpg", out_img_bgr)
print("[PYNQ] Output saved to output_pynq.jpg")

print("[PYNQ] Done!")

# ------------------------------------------------------------
# 8. OPTIONAL QUALITY EVALUATION (compare with reference)
# ------------------------------------------------------------
# Usage: python3 pynq_runner.py <img_path> <mask_size> <ref_image_path>
# If a reference image is supplied, compute pixel‑wise quality metrics.
import sys
from pathlib import Path

if len(sys.argv) > 3:
    ref_path = Path(sys.argv[3])
    if ref_path.is_file():
        # Load generated output image
        from PIL import Image
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        gen_img = Image.open("output_pynq.jpg").convert('L')
        ref_img = Image.open(ref_path).convert('L')
        gen_arr = np.array(gen_img, dtype=np.float32)
        ref_arr = np.array(ref_img, dtype=np.float32)
        # Ensure same shape (resize if needed)
        if gen_arr.shape != ref_arr.shape:
            print(f"[WARN] Shape mismatch: generated {gen_arr.shape}, reference {ref_arr.shape}. Resizing reference.")
            ref_img = ref_img.resize(gen_img.size, Image.BICUBIC)
            ref_arr = np.array(ref_img, dtype=np.float32)
        mse_val = np.mean((ref_arr - gen_arr) ** 2)
        psnr_val = psnr(ref_arr, gen_arr, data_range=255)
        ssim_val = ssim(ref_arr, gen_arr, data_range=255)
        print(f"[QUALITY] MSE: {mse_val:.4f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    else:
        print(f"[WARN] Reference image '{ref_path}' not found, skipping quality evaluation.")

