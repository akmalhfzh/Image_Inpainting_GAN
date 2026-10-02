import os
import glob
from PIL import Image

testcases = glob.glob('testcases/*')
for tc in testcases:
    img_path = os.path.join(tc, 'original_input.png')
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        # Ensure it's 64x64 if not already, or we can just apply a proportional mask
        # inference.py resizes to 64x64 then applies a 16x16 mask
        img = img.resize((64, 64), Image.BICUBIC)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # mask_size 16, centered -> x: 24 to 40, y: 24 to 40
        draw.rectangle([24, 24, 39, 39], fill=(255, 255, 255))
        out_path = os.path.join(tc, 'masked_input.png')
        img.save(out_path)
        print(f"Created {out_path}")
