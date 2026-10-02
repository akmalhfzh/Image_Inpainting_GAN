#!/usr/bin/env python3
import numpy as np
import os, sys
from PIL import Image

# Konfigurasi Hardware (Q7.8)
DW     = 16          
FRAC_W = 8           
SCALE  = 1 << FRAC_W # 256
H = W  = 64          
C      = 3           

def prepare_input(image_path):
    os.makedirs("inputs", exist_ok=True)
    output_hex = "inputs/img.hex"
    recon_png  = "inputs/reconstructed_input.png"

    # --- 1. ENCODING: Image -> Hex ---
    try:
        img = Image.open(image_path).convert("RGB").resize((W, H), Image.LANCZOS)
        img_np = np.array(img, dtype=np.float32)
        
        # Normalisasi: [0, 255] -> [-1.0, 1.0]
        img_float = (img_np / 127.5) - 1.0 
        
        # Transpose: [H, W, C] -> [C, H, W]
        img_float = img_float.transpose(2, 0, 1)
        
        # Fixed-Point Quantization
        img_fp = np.round(img_float * SCALE).astype(np.int32)
        limit = 1 << (DW - 1)
        img_fp = np.clip(img_fp, -limit, limit - 1).astype(np.int16)

        # Write to Hex
        MASK = 0xFFFF
        with open(output_hex, "w") as f:
            for v in img_fp.flatten():
                f.write(f"{int(v) & MASK:04x}\n")
        
        print(f"[SUCCESS] Hex file written to {output_hex}")

    except Exception as e:
        print(f"[ERROR] Encoding failed: {e}")
        return

    # --- 2. DECODING (Verification): Hex -> Image ---
    print("[VERIFY] Reconstructing image from hex to verify...")
    
    try:
        # Baca kembali file hex yang baru saja dibuat
        with open(output_hex, "r") as f:
            hex_lines = f.readlines()

        # Konversi hex string kembali ke signed integer 16-bit
        raw_vals = []
        for h in hex_lines:
            val = int(h.strip(), 16)
            if val >= 0x8000: # Koreksi untuk signed negative
                val -= 0x10000
            raw_vals.append(val)

        # Reshape kembali ke format [C, H, W]
        recon_fp = np.array(raw_vals).reshape(C, H, W)

        # De-quantization: Fixed-point -> Float
        # Formula: $Value_{float} = \frac{Value_{fixed}}{256}$
        recon_float = recon_fp.astype(np.float32) / SCALE

        # Denormalization: [-1.0, 1.0] -> [0, 255]
        recon_final = (recon_float + 1.0) * 127.5
        recon_final = np.clip(recon_final, 0, 255).astype(np.uint8)

        # Transpose balik ke format Image: [C, H, W] -> [H, W, C]
        recon_final = recon_final.transpose(1, 2, 0)

        # Simpan sebagai PNG untuk pengecekan visual
        Image.fromarray(recon_final).save(recon_png)
        print(f"[SUCCESS] Verification image saved to {recon_png}")
        print("=> Buka file tersebut untuk memastikan gambar tidak hancur/berubah warna.")

    except Exception as e:
        print(f"[ERROR] Reconstruction failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_input.py <image.png>")
    else:
        prepare_input(sys.argv[1])
