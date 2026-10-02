import argparse
import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_image(path: Path) -> np.ndarray:
    """Load image as grayscale floating point array in [0, 255]."""
    img = Image.open(path).convert('L')  # convert to grayscale
    return np.array(img, dtype=np.float32)

def compute_metrics(ref: np.ndarray, gen: np.ndarray):
    """Return MSE, PSNR, SSIM between two images of same shape."""
    if ref.shape != gen.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {gen.shape}")
    mse_val = np.mean((ref - gen) ** 2)
    psnr_val = psnr(ref, gen, data_range=255)
    ssim_val = ssim(ref, gen, data_range=255)
    return mse_val, psnr_val, ssim_val

def evaluate_folder(ref_dir: Path, gen_dir: Path, out_csv: Path):
    ref_images = sorted(ref_dir.glob('*.png')) + sorted(ref_dir.glob('*.jpg')) + sorted(ref_dir.glob('*.jpeg'))
    gen_images = sorted(gen_dir.glob('*.png')) + sorted(gen_dir.glob('*.jpg')) + sorted(gen_dir.glob('*.jpeg'))

    # Build a mapping by filename (without extension) for matching pairs
    ref_map = {p.stem: p for p in ref_images}
    gen_map = {p.stem: p for p in gen_images}

    common_keys = sorted(set(ref_map.keys()) & set(gen_map.keys()))
    if not common_keys:
        raise RuntimeError('No matching image pairs found between reference and generated folders.')

    with out_csv.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'MSE', 'PSNR (dB)', 'SSIM'])
        for key in common_keys:
            ref_img = load_image(ref_map[key])
            gen_img = load_image(gen_map[key])
            mse_val, psnr_val, ssim_val = compute_metrics(ref_img, gen_img)
            writer.writerow([key, f"{mse_val:.4f}", f"{psnr_val:.2f}", f"{ssim_val:.4f}"])
    print(f'Quality report written to {out_csv}')

def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics between reference and generated images.')
    parser.add_argument('--ref', required=True, type=Path, help='Directory containing reference images')
    parser.add_argument('--gen', required=True, type=Path, help='Directory containing generated images')
    parser.add_argument('--out', default='quality_report.csv', type=Path, help='Output CSV file path')
    args = parser.parse_args()
    evaluate_folder(args.ref, args.gen, args.out)

if __name__ == '__main__':
    main()
