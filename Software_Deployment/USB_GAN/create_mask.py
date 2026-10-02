#!/usr/bin/env python3
import sys
from PIL import Image, ImageDraw

if len(sys.argv) < 3:
    print("Usage: python3 create_mask.py <input_image> <output_image>")
    sys.exit(1)

# Resize to 64x64 and apply 16x16 white mask at the center
img = Image.open(sys.argv[1]).convert("RGB").resize((64, 64), Image.BICUBIC)
draw = ImageDraw.Draw(img)
draw.rectangle([24, 24, 39, 39], fill=(255, 255, 255))
img.save(sys.argv[2])

print(f"✅ Masked image saved to {sys.argv[2]}")
