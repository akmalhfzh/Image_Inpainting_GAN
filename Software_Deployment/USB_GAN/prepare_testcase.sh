#!/bin/bash
if [ -z "$2" ]; then
    echo "Usage: ./prepare_testcase.sh <path_to_image> <testcase_name>"
    echo "Example: ./prepare_testcase.sh ../../49605.png test_49605"
    exit 1
fi

IMG_PATH=$(realpath "$1")
TEST_NAME=$2
TEST_DIR="testcases/$TEST_NAME"

echo "=> Creating testcase: $TEST_NAME"
mkdir -p "$TEST_DIR"

echo "=> Running inference.py to extract reference hex files..."
cd ..
# Run inference
/mnt/ssd_eda/python_envs/eda_venv/bin/python3 verify/inference.py --weights ../../train/ffhq_4000/gw_latest.bin --img_path "$IMG_PATH" --dump_hex --no_show

echo "=> Copying files to USB_GAN/$TEST_DIR"
cp outputs/py_ref/input_image.hex USB_GAN/"$TEST_DIR"/
cp outputs/py_ref/ref_OUT.hex USB_GAN/"$TEST_DIR"/
cp outputs/py_ref/ref_*.hex USB_GAN/"$TEST_DIR"/
cp "$IMG_PATH" USB_GAN/"$TEST_DIR"/original_input.png
if [ -f "inference_result.png" ]; then
    cp inference_result.png USB_GAN/"$TEST_DIR"/pytorch_result.png
fi

echo "=> Done! You can now run verification on PYNQ with:"
echo "   sudo -E python3 pynq_verify.py --testcase $TEST_DIR"
