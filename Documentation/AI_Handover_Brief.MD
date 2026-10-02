Project Handover Brief: Context Encoder GAN Accelerator (Integrate 5.4)
1. Project Context & Objectives
Goal: Design, simulate, and implement a hardware accelerator (Verilog RTL) for the Generator part of a Context Encoder GAN used for image inpainting/generation.
The system uses Hardware-Software Co-Design, targeting a Xilinx Zynq/PYNQ FPGA board.
The current version is Integrate 5.4, which already successfully generates bitstreams (`design_1_wrapper.bit`) and has verified RTL simulation using Verilator.
Crucial Note on RTL Implementation:
The Verilog RTL for this accelerator was built completely from scratch by hand. It is not generated using High-Level Synthesis (HLS) or automatic tools. The RTL logic is explicitly designed to follow the mathematical equations and functional behavior of the PyTorch model precisely, ensuring exact matching between hardware and software.
2. Model Architecture & Training
The GAN model was trained from scratch using PyTorch with the FFHQ dataset (64x64 resolution, randomly masked). The training scripts are located at `/mnt/ssd_eda/projects/PME/GAN/Context_Encoder_v2/train`.
The architecture consists of a Lite Generator and Discriminator, specifically optimized for PYNQ-Z1 resource constraints:
Generator (Lite) Architecture
Input: 64x64x3 image. Total 12 layers (5 Encoder + 1 Bottleneck + 5 Decoder + 1 Output).
Encoder (5 Layers): Progressively downsamples using `Conv2d (k=4, s=2, p=1)`, `BatchNorm2d` (except E1), and `LeakyReLU(0.2)`.
E1: 3->32 (64x64 -> 32x32), E2: 32->64 (32x32 -> 16x16), E3: 64->128 (16x16 -> 8x8), E4: 128->256 (8x8 -> 4x4), E5: 256->512 (4x4 -> 2x2).
Bottleneck (1 Layer): `Conv2d (k=1, s=1, p=0)` to maintain 2x2 spatial dimension but map features (512 -> 512).
Decoder (5 Layers): Progressively upsamples using `ConvTranspose2d (k=4, s=2, p=1)`, `BatchNorm2d`, and `ReLU`.
D1: 512->256 (2x2 -> 4x4), D2: 256->128 (4x4 -> 8x8), D3: 128->64 (8x8 -> 16x16), D4: 64->32 (16x16 -> 32x32), D5: 32->32 (32x32 -> 64x64).
Output Layer: `Conv2d (k=3, s=1, p=1)` and `Tanh` activation, outputting the final 64x64x3 generated image.
Discriminator (Lite) Architecture
4 Convolutional layers progressively downsampling using `Conv2d (k=3, s=2, p=1)`, `InstanceNorm2d`, and `LeakyReLU(0.2)`. (Channels: 3->32, 32->64, 64->128, 128->256).
Final output layer (`Conv2d k=3, s=1, p=1`) maps to a single patch prediction scalar value.
3. Directory Structure & Key Components
Path: `/mnt/ssd_eda/projects/PME/GAN/Context_Encoder_v2/verilog/Integrate5.4`
A. RTL Design (`rtl/`)
`generator_top.v`: The top-level module of the Generator. It integrates the memory interfaces and computational blocks.
`gan_axi_wrapper.v`: Wraps the top module with AXI4/AXI-Lite interfaces for seamless integration with the Zynq Processing System (PS).
`conv_unified.v` & `mac_array.v`: The core computational engine. It uses a unified Multiply-Accumulate (MAC) array to process convolutions efficiently.
`addr_gen.v`: Address generator for fetching feature maps and weights from memory.
`batchnorm2d.v`: Hardware implementation of Batch Normalization.
`leaky_relu.v`, `relu.v`, `tanh_act.v`: Activation function modules.
B. Verification & Software Reference (`verify/` & `outputs/`)
Python Reference:
`verify/extract_weights.py`: Extracts weights from a trained PyTorch model (`gw_latest.bin`) into `.hex` files.
`verify/inference.py`: Runs the PyTorch reference model to generate ground truth outputs and input hex files.
`verify/check_gen.py`: Compares the Verilator RTL simulation output (`outputs/rtl_layer_*.hex`) layer-by-layer against the PyTorch reference to ensure bit-accuracy/functional correctness.
C. Simulation (`tb/`)
`sim_main.cpp`: C++ testbench for Verilator. It models a 128MB AXI DDR memory. It loads weights, biases, and the input image, drives the AXI handshakes, and dumps intermediate feature maps to `outputs/rtl_layer_X.hex`.
`tb_generator_top.v`: Standard Verilog testbench (often used with Icarus Verilog for VCD waveform generation).
D. FPGA Implementation (`build_bd_bitstream.tcl`, `project_pynq_bd/`)
Vivado 2023.2 is used for synthesis and implementation.
The TCL script `build_bd_bitstream.tcl` automates the creation of a Block Design (`design_1.bd`), connects the `gan_axi_wrapper` to the Zynq PS, and generates the bitstream (`design_1_wrapper.bit`) and hardware handoff file (`design_1_wrapper.hwh`).
3. Development Workflow (`Makefile`)
The project utilizes a `Makefile` to automate the entire flow:
`make weights`: Extracts PyTorch weights to `weights_hex/`.
`make ref`: Runs PyTorch inference and saves reference Hex files (`outputs/py_ref/`).
`make verilate` & `make build`: Compiles the Verilog RTL and `sim_main.cpp` into a cycle-accurate Verilator executable.
`make sim`: Runs the Verilator simulation.
`make check`: Verifies the RTL output against the PyTorch reference.
`make vivado_bitstream`: Triggers Vivado in batch mode to build the FPGA block design and synthesize the `.bit` file.
4. Memory Map & Data Flow
The Verilator memory model and Zynq PS share the following memory map for the accelerator:
`0x0000000` (BASE_WEIGHT): Weights storage.
`0x4000000` (BASE_IMG): Input Image storage.
`0x5000000` (BASE_FMAP_A) & `0x6000000` (BASE_FMAP_B): Ping-pong buffers for intermediate feature maps between layers.
`0x7000000` (BASE_OUT): Final output image storage.
5. Current State & Next Steps
The hardware logic is completely functional and verified via Verilator.
The Vivado block design has been integrated, and the `.bit` and `.hwh` files have been successfully generated for the PYNQ board.
Output images (e.g., `output_pynq_mbappe.jpg`, `output_verify_kakek.png`) indicate that hardware-in-the-loop (HIL) testing or on-board inference has already been executed successfully.
Focus for the AI: Depending on the user's prompt, the next steps could involve optimizing the MAC array, reducing resource utilization, adding new features to the AXI wrapper, debugging a specific layer's precision issue, or adapting the code for a new FPGA board. Use this brief to maintain full context of the project architecture.
