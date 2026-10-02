#!/bin/bash

# ──────────────────────────────────────────────────────────────────
#  Context Encoder GAN — Training Script (LITE Version for PYNQ-Z1)
# ──────────────────────────────────────────────────────────────────

# ── 1. Konfigurasi Eksperimen ──────────────────
DATA_DIR="./data/ffhq" 
EXP_NAME="ffhq_lite_64x64"

# Hyperparameters
N_EPOCHS=1500
BATCH_SIZE=64      
LR=0.0002          
N_CPU=8            

# Dimensi Gambar & Mask
IMG_SIZE=64         
MASK_SIZE=32       
MIN_MASK_SIZE=16 

# Interval Logging & Saving
SAMPLE_INTERVAL=200
SAVE_INTERVAL=20
RESUME="" 
# ───────────────────────────────────────────────

echo "========================================================="
echo "   STARTING CONTEXT ENCODER TRAINING (LITE ARCH)"
echo "========================================================="
echo " Experiment : $EXP_NAME"
echo " Data Path  : $DATA_DIR"
echo " GPU Device : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Batch Size : $BATCH_SIZE"
echo "========================================================="

# ── Validasi Dataset ───────────────────────
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Dataset tidak ditemukan di $DATA_DIR"
    exit 1
fi

IMG_COUNT=$(find "$DATA_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)
echo "[INFO] Ditemukan $IMG_COUNT gambar."

# ── Folder Output ──────────────────────────
mkdir -p "./outputs/$EXP_NAME/weights"
mkdir -p "./outputs/$EXP_NAME/samples"

# ── Jalankan Python Training ───────────────
stdbuf -oL python3 train.py \
    --data_dir "$DATA_DIR" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --n_cpu "$N_CPU" \
    --img_size "$IMG_SIZE" \
    --mask_size "$MASK_SIZE" \
    --min_mask_size "$MIN_MASK_SIZE" \
    --sample_interval "$SAMPLE_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    ${RESUME:+--resume "$RESUME"}

if [ $? -eq 0 ]; then
    echo "========================================================="
    echo " Training Berhasil!"
    echo " Model Akhir: ./weights/gw_latest.bin"
    echo "========================================================="
else
    echo "========================================================="
    echo " [ERROR] Training berhenti secara tidak normal."
    echo "========================================================="
    exit 1
fi
