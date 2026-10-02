#!/bin/bash
DATA_DIR="./data/celeba/img_align_celeba"
N_EPOCHS=400
BATCH_SIZE=16
LR=0.0002
N_CPU=4
IMG_SIZE=64
MASK_SIZE=32
SAMPLE_INTERVAL=100
SAVE_INTERVAL=1
RESUME=""           
echo "============================================"
echo " Context Encoder GAN — Training"
echo "============================================"
echo " Data dir      : $DATA_DIR"
echo " Epochs        : $N_EPOCHS"
echo " Batch size    : $BATCH_SIZE"
echo " Image size    : ${IMG_SIZE}x${IMG_SIZE}"
echo " Mask size     : ${MASK_SIZE}x${MASK_SIZE}"
echo " Learning rate : $LR"
echo "============================================"

if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Folder dataset tidak ditemukan: $DATA_DIR"
    echo "        Pastikan sudah download dan ekstrak dataset."
    exit 1
fi

IMG_COUNT=$(find "$DATA_DIR" -maxdepth 1 -iname "*.jpg" | wc -l)
if [ "$IMG_COUNT" -eq 0 ]; then
    echo "[ERROR] Tidak ada file .jpg di: $DATA_DIR"
    exit 1
fi
echo " Jumlah gambar : $IMG_COUNT"
echo ""

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo " Resume dari   : $RESUME"
fi

MIN_MASK_SIZE=10

# Jalankan training
python3 train.py \
    --data_dir        "$DATA_DIR"        \
    --n_epochs        "$N_EPOCHS"        \
    --batch_size      "$BATCH_SIZE"      \
    --lr              "$LR"              \
    --n_cpu           "$N_CPU"           \
    --img_size        "$IMG_SIZE"        \
    --mask_size       "$MASK_SIZE"       \
    --min_mask_size   "$MIN_MASK_SIZE"
    --sample_interval "$SAMPLE_INTERVAL" \
    --save_interval   "$SAVE_INTERVAL"   \
    $RESUME_ARG

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo " Training selesai!"
    echo " Model tersimpan di: ./weights/gw_latest.pth"
    echo "============================================"
else
    echo ""
    echo "[ERROR] Training gagal dengan exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
