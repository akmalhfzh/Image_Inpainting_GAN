#!/bin/bash

# ============================================
# setup.sh — Install semua dependencies
# Penggunaan: bash setup.sh
# ============================================

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python tidak ditemukan."
    exit 1
fi

# Gunakan python -m pip (lebih reliable daripada pip/pip3 langsung)
$PYTHON -m pip --version &> /dev/null
if [ $? -ne 0 ]; then
    echo "pip module tidak ada, mencoba install via ensurepip..."
    $PYTHON -m ensurepip --upgrade 2>/dev/null
    $PYTHON -m pip --version &> /dev/null
    if [ $? -ne 0 ]; then
        echo "Mencoba install pip via apt..."
        sudo apt-get install -y python3-pip
        $PYTHON -m pip --version &> /dev/null
        if [ $? -ne 0 ]; then
            echo "[ERROR] pip tidak bisa diaktifkan."
            exit 1
        fi
    fi
fi

PIP="$PYTHON -m pip"

echo "============================================"
echo " Setup Dependencies — Context Encoder GAN"
echo "============================================"
echo " Python : $PYTHON"
echo " Pip    : $PIP"
echo ""

echo "[1/4] Upgrade pip..."
$PIP install --upgrade pip

echo ""
echo "[2/4] Install PyTorch..."
# Deteksi CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+")
    echo "     CUDA terdeteksi: $CUDA_VER"
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "     CUDA tidak terdeteksi, install CPU version..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[3/4] Install dependencies lainnya..."
$PIP install \
    pandas \
    numpy \
    Pillow \
    matplotlib

echo ""
echo "[4/4] Verifikasi instalasi..."
$PYTHON -c "
import torch, torchvision, pandas, numpy, PIL, matplotlib
print(f'  torch       : {torch.__version__}')
print(f'  torchvision : {torchvision.__version__}')
print(f'  pandas      : {pandas.__version__}')
print(f'  numpy       : {numpy.__version__}')
print(f'  Pillow      : {PIL.__version__}')
print(f'  matplotlib  : {matplotlib.__version__}')
print(f'  CUDA ready  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================"
echo " Setup selesai! Sekarang jalankan:"
echo "   ./run_train.sh"
echo "============================================"

