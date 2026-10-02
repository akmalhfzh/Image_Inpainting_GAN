#!/bin/bash
DATA_DIR="./data/celeba/img_align_celeba"
WEIGHTS="./weights/gw_latest.bin"
IMG_SIZE=64
MASK_SIZE=32   # ← Bebas diubah: 10, 16, 20, 32, dst
OUTPUT="inference_result.png"
NO_SHOW=false

# Override dari argumen CLI
# Usage: bash run_inference.sh [img_path] [output] [mask_size]
if [ -n "$1" ]; then IMG_PATH="$1"; fi
if [ -n "$2" ]; then OUTPUT="$2"; fi
if [ -n "$3" ]; then MASK_SIZE="$3"; fi  # ← BARU: mask size dari CLI

echo "============================================"
echo " Context Encoder GAN — Inference (v3)"
echo "============================================"
echo " Weights    : $WEIGHTS"
echo " Image size : ${IMG_SIZE}x${IMG_SIZE}"
echo " Mask size  : ${MASK_SIZE}x${MASK_SIZE}"   # ← sekarang fleksibel
echo " Output     : $OUTPUT"
echo "============================================"

# Cek file bobot
if [ ! -f "$WEIGHTS" ]; then
    echo "[ERROR] File bobot tidak ditemukan: $WEIGHTS"
    echo "        Jalankan run_train.sh terlebih dahulu!"
    exit 1
fi

# Bangun argumen sumber gambar
if [ -n "$IMG_PATH" ]; then
    if [ ! -f "$IMG_PATH" ]; then
        echo "[ERROR] File gambar tidak ditemukan: $IMG_PATH"
        exit 1
    fi
    echo " Input      : $IMG_PATH"
    SRC_ARG="--img_path $IMG_PATH"
else
    if [ ! -d "$DATA_DIR" ]; then
        echo "[ERROR] Folder dataset tidak ditemukan: $DATA_DIR"
        echo "        Berikan path gambar: bash run_inference.sh ./foto.jpg"
        exit 1
    fi
    echo " Input      : acak dari $DATA_DIR"
    SRC_ARG="--data_dir $DATA_DIR"
fi

# Argumen no_show
NO_SHOW_ARG=""
if [ "$NO_SHOW" = true ]; then
    NO_SHOW_ARG="--no_show"
fi

echo ""

# Jalankan inference
python inference.py \
    $SRC_ARG              \
    --weights   "$WEIGHTS"  \
    --img_size  "$IMG_SIZE"  \
    --mask_size "$MASK_SIZE" \
    --output    "$OUTPUT"    \
    $NO_SHOW_ARG

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo " Inference selesai!"
    echo " Hasil tersimpan di: $OUTPUT"
    echo "============================================"
else
    echo ""
    echo "[ERROR] Inference gagal dengan exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
