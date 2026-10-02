#!/usr/bin/env python3
"""
check_img.py
────────────
Diagnostic tool untuk mengecek validitas inputs/img.hex
"""
import os
import sys

FILE_PATH = "inputs/img.hex"
EXPECTED_LINES = 3 * 64 * 64  # 12288

def main():
    print("=" * 50)
    print(f" DIAGNOSTIK: {FILE_PATH}")
    print("=" * 50)

    if not os.path.exists(FILE_PATH):
        print(f"[FAIL ✗] File tidak ditemukan di path: {FILE_PATH}")
        print("         Pastikan skrip dijalankan dari root proyek (Integrate3).")
        sys.exit(1)

    with open(FILE_PATH, "r") as f:
        # Baca semua baris, abaikan baris kosong (jika ada trailing newline)
        lines = [line.strip() for line in f.readlines() if line.strip()]

    num_lines = len(lines)
    print(f"Jumlah Baris : {num_lines} (Target: {EXPECTED_LINES})")
    
    if num_lines != EXPECTED_LINES:
        print("[FAIL ✗] Ukuran file tidak sesuai dengan 3x64x64!")
    else:
        print("[PASS ✓] Ukuran file presisi.")

    non_zero_count = 0
    try:
        for line in lines:
            val = int(line, 16)
            if val != 0:
                non_zero_count += 1
    except ValueError as e:
        print(f"[FAIL ✗] Ditemukan karakter bukan hexadecimal: {e}")
        sys.exit(1)

    print(f"Data Non-Zero: {non_zero_count} baris dari {num_lines}")

    if non_zero_count == 0:
        print("[FAIL ✗] File HANYA BERISI NOL. Gambar gagal dikonversi atau hitam pekat.")
    else:
        print("[PASS ✓] Ditemukan data valid (bukan nol).")

    print("-" * 50)
    print("Cuplikan 10 Pixel Pertama (Hex):")
    print(" ".join(lines[:10]))
    print("=" * 50)

if __name__ == "__main__":
    main()
