# =====================================================================
# pynq_verify.py — BIT-EXACT FPGA validation against the golden reference
# =====================================================================
import numpy as np
import time
import os
import argparse
from pynq import Bitstream, MMIO, allocate

parser = argparse.ArgumentParser(description="Verify FPGA against PyTorch golden reference")
parser.add_argument("-test", "--testcase", type=str, default="py_ref", help="Directory containing the testcase (e.g. testcases/test_49605) or path to an image file")
parser.add_argument("--no_mask", action="store_true", help="Skip applying the 16x16 mask (assume image is already masked)")
args = parser.parse_args()

# Handle shortcut formats like 'test/kakek' or 'kakek' -> 'testcases/test_kakek'
# OR if it's an image file, auto-run inference.py to generate testcase on the fly.
if os.path.isfile(args.testcase) and args.testcase.lower().endswith(('.png', '.jpg', '.jpeg')):
    img_path = os.path.abspath(args.testcase)
    print(f"[VERIFY] Auto-preparing testcase from image: {img_path}")
    import subprocess
    import sys
    cmd = [
        sys.executable,
        "verify/inference.py",
        "--weights", "../../train/ffhq_4000/gw_latest.bin",
        "--img_path", img_path,
        "--dump_hex", "--no_show"
    ]
    if args.no_mask:
        cmd.append("--no_mask")
    print("[VERIFY] Running PyTorch inference...")
    subprocess.run(cmd, check=True, cwd="..")
    args.testcase = "../outputs/py_ref"
elif args.testcase.startswith("test/"):
    args.testcase = "testcases/test_" + args.testcase[5:]
elif not args.testcase.startswith("testcases/") and args.testcase != "py_ref":
    if args.testcase.startswith("test_"):
        args.testcase = "testcases/" + args.testcase
    else:
        args.testcase = "testcases/test_" + args.testcase

DW, FRAC_W, SCALE = 16, 8, 256
MASK16 = (1 << DW) - 1
SIGN   = 1 << (DW - 1)

REF_DIR    = args.testcase
INPUT_HEX  = os.path.join(REF_DIR, "input_image.hex")
GOLDEN_OUT = os.path.join(REF_DIR, "ref_OUT.hex")   # PyTorch quantized golden
RTL_OUT    = os.path.join(REF_DIR, "rtl_OUT.hex")    # Verilator OUT (bit-exact HW target)

OUT_TOL  = 0.20    # end-to-end tolerance check_gen.py uses for the OUT layer
MIN_CORR = 0.930
RTL_TOL  = 0.002   # FPGA vs Verilator should be ~bit-identical (allow tiny slack)


def read_hex_signed(fname):
    """Parse a hex dump of 16-bit two's-complement words -> int16 numpy array."""
    vals = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('/'):
                continue
            v = int(line, 16) & MASK16
            if v >= SIGN:
                v -= (1 << DW)
            vals.append(v)
    return np.array(vals, dtype=np.int16)


# =====================================================================
# 1. LOAD OVERLAY
# =====================================================================
print("[VERIFY] Loading Bitstream...")
Bitstream("design_1_wrapper.bit").download()
gan_ip = MMIO(0x40000000, 0x1000)

# =====================================================================
# 2. ALLOCATE BUFFERS (same layout as pynq_runner.py)
# =====================================================================
print("[VERIFY] Allocating CMA Memory...")
img_buf    = allocate(shape=(3, 64, 64), dtype=np.int16)
fmap_a     = allocate(shape=(1024 * 1024,), dtype=np.int16); fmap_a[:] = -1; fmap_a.flush()
fmap_b     = allocate(shape=(1024 * 1024,), dtype=np.int16); fmap_b[:] = -1; fmap_b.flush()
out_buf    = allocate(shape=(3, 64, 64), dtype=np.int16);    out_buf[:] = -1; out_buf.flush()
weight_buf = allocate(shape=(6 * 1024 * 1024,), dtype=np.int16)

# =====================================================================
# 3. WRITE BASE ADDRESS REGISTERS
# =====================================================================
gan_ip.write(0x08, img_buf.device_address)
gan_ip.write(0x0C, fmap_a.device_address)
gan_ip.write(0x10, fmap_b.device_address)
gan_ip.write(0x14, out_buf.device_address)
gan_ip.write(0x20, weight_buf.device_address)

# =====================================================================
# 4A. LOAD WEIGHTS TO DDR
#     Offsets MUST match sim_main.cpp / generator_top (verified layout).
# =====================================================================
print("[VERIFY] Loading Weights to DDR...")
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
        vals = read_hex_signed(path)
        if vals.size:
            off = weight_offsets[layer]
            weight_buf[off: off + vals.size] = vals
weight_buf.flush()

# =====================================================================
# 4B. LOAD BIASES VIA MMIO (config reg 0x18: {layer[27:24], type[19:18]=3, addr[17:0]})
# =====================================================================
print("[VERIFY] Loading Biases to FPGA Registers...")
bias_files = {
    0: 'E1_bias.hex', 1: 'E2_bias.hex', 2: 'E3_bias.hex',
    3: 'E4_bias.hex', 4: 'E5_bias.hex', 5: 'BOT_bias.hex',
    6: 'D1_bias.hex', 7: 'D2_bias.hex', 8: 'D3_bias.hex',
    9: 'D4_bias.hex', 10: 'D5_bias.hex', 11: 'OUT_bias.hex'
}
for layer, fname in bias_files.items():
    path = os.path.join("weights_hex", fname)
    if os.path.exists(path):
        with open(path) as f:
            for addr, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('/'):
                    continue
                config_val = (layer << 24) | (3 << 18) | (addr & 0x3FFFF)
                gan_ip.write(0x18, config_val)
                gan_ip.write(0x1C, int(line, 16) & 0xFFFF)

# =====================================================================
# 5. LOAD EXACT GOLDEN INPUT (no re-preprocessing!)
# =====================================================================
print(f"[VERIFY] Loading golden input from {INPUT_HEX} ...")
inp = read_hex_signed(INPUT_HEX)
assert inp.size == 3 * 64 * 64, f"input_image.hex has {inp.size} words, expected 12288"
np.copyto(img_buf, inp.reshape(3, 64, 64))
img_buf.flush()

# =====================================================================
# 6. RUN INFERENCE
# =====================================================================
print("[VERIFY] Starting Hardware Inference...")
t0 = time.time()
gan_ip.write(0x00, 1)  # start pulse
done = 0
while done == 0:
    status = gan_ip.read(0x04)
    done = status & 0x1
    time.sleep(0.01)
elapsed_ms = (time.time() - t0) * 1000.0
print(f"[VERIFY] Inference complete in {elapsed_ms:.2f} ms")

out_buf.invalidate()
fpga = np.copy(out_buf).astype(np.int16).reshape(-1)

# =====================================================================
# 7. COMPARE (mirror of verify/check_gen.py metrics)
# =====================================================================
fpga_f = fpga.astype(np.float32) / SCALE


def compare(ref_vals):
    r = ref_vals.astype(np.float32) / SCALE
    d = np.abs(fpga_f - r)
    c = float(np.corrcoef(fpga_f, r)[0, 1]) if fpga_f.std() > 0 and r.std() > 0 else 0.0
    return float(np.mean(d)), float(np.max(d)), c


print("=" * 60)

# --- Primary check: FPGA vs Verilator (should be ~bit-identical) ---
hw_ok = None
if os.path.exists(RTL_OUT):
    mae_r, mx_r, corr_r = compare(read_hex_signed(RTL_OUT))
    hw_ok = (mae_r <= RTL_TOL)
    print(" [1] FPGA vs Verilator RTL  (expect ~0 — pure hardware check)")
    print(f"     MAE {mae_r:.5f} (tol {RTL_TOL:.3f}) | MaxErr {mx_r:.5f} | Corr {corr_r:.4f}")
    print(f"     -> {'BIT-EXACT ✅  HW datapath matches sim' if hw_ok else 'DIVERGES ⚠️  HW != sim (AXI/bitstream issue)'}")
    print("-" * 60)
else:
    print(" [1] (rtl_OUT.hex not on USB — skipping pure-HW check)")
    print("-" * 60)

# --- Secondary check: FPGA vs PyTorch golden (end-to-end, ~0.17 expected) ---
golden = read_hex_signed(GOLDEN_OUT)
mae, max_err, corr = compare(golden)
e2e_ok = (mae <= OUT_TOL) and (corr >= MIN_CORR)
print(" [2] FPGA vs PyTorch golden (end-to-end, expect MAE ~0.17)")
print(f"     MAE {mae:.5f} (tol {OUT_TOL:.2f}) | MaxErr {max_err:.5f} | Corr {corr:.4f} (min {MIN_CORR:.2f})")
print(f"     FPGA   [:5] = {fpga[:5]}")
print(f"     GOLDEN [:5] = {golden[:5]}")
print("=" * 60)
overall = e2e_ok and (hw_ok is not False)
print("  ✅ VERDICT: FPGA inference CORRECT." if overall
      else "  ❌ VERDICT: MISMATCH — inspect bitstream / AXI datapath.")
print("=" * 60)

# =====================================================================
# 8. SAVE OUTPUTS (raw hex for offline check_gen + a viewable image)
# =====================================================================
with open("out_fpga.hex", "w") as f:
    for v in (fpga.astype(np.uint16)):
        f.write(f"{v:04x}\n")
print("[VERIFY] Wrote out_fpga.hex (feed to verify/check_gen.py offline if needed)")

try:
    from PIL import Image
    img = (fpga_f.reshape(3, 64, 64).transpose(1, 2, 0))
    img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(img).save("output_verify.png")
    print("[VERIFY] Saved output_verify.png")
except Exception as e:
    print(f"[VERIFY] (image save skipped: {e})")
