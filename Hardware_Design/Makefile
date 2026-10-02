# =============================================================================
# Makefile — Hardware-Software Co-Design (Verilator & Vivado)
# =============================================================================

# -----------------------------------------------------------------------------
# Toolchain & Configuration
# -----------------------------------------------------------------------------
VERILATOR = verilator
THREADS   ?= 15
BUILD_DIR = build_verilator
TOP_MODULE = generator_top

# --- Icarus Verilog (VCD Toolchain) ---
IVERILOG = iverilog
VVP      = vvp
V_TB     = tb/tb_generator_top.v
V_OUT    = build/tb_gen.vvp

# --- Vivado ---
VIVADO   = /mnt/ssd_eda/eda_tools/Xilinx/Vivado/2023.2/bin/vivado
VIVADO_MODE = -mode batch

# --- PYTHON PATH ---
PYTHON = /mnt/ssd_eda/python_envs/eda_venv/bin/python3

# Flags Verilator: 
V_FLAGS = -Wall -Wno-fatal --cc --exe --threads $(THREADS) --trace -O3 \
          --Mdir $(BUILD_DIR) -Irtl/

# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
RTL_SHARED = rtl/mac_array.v \
             rtl/addr_gen.v \
             rtl/conv_unified.v \
             rtl/leaky_relu.v \
             rtl/relu.v \
             rtl/tanh_act.v

RTL_TOP    = rtl/generator_top.v
CPP_TB     = tb/sim_main.cpp

# -----------------------------------------------------------------------------
# Python Verification Paths
# -----------------------------------------------------------------------------
WEIGHT_BIN ?= ../../train/ffhq_4000/gw_latest.bin
INPUT_IMG  ?= ../49605.png

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
.PHONY: all all_verilator all_vivado weights ref verilate build sim check sim_icarus wave vivado_synth vivado_bitstream open_vivado clean

# Default jalankan simulasi Verilator
all: all_verilator

# (VERILATOR)
all_verilator: weights ref build sim check

# (VIVADO)
all_vivado: weights vivado_bitstream

weights:
	@echo "=> [STEP 1] Mempersiapkan Bumbu (Ekstraksi Bobot)..."
	@mkdir -p weights_hex
	$(PYTHON) verify/extract_weights.py $(WEIGHT_BIN)

ref:
	@echo "=> [STEP 2] Mempersiapkan Kunci Jawaban & Input (PyTorch Inference)..."
	@mkdir -p outputs/py_ref
	$(PYTHON) verify/inference.py --weights $(WEIGHT_BIN) --img_path $(INPUT_IMG) --dump_hex --no_show

verilate: $(RTL_TOP) $(RTL_SHARED) $(CPP_TB)
	@mkdir -p $(BUILD_DIR)
	$(VERILATOR) $(V_FLAGS) $(RTL_TOP) $(RTL_SHARED) $(CPP_TB)

build: verilate
	@echo "=> [STEP 3] Kompilasi Hardware (Verilator Build)..."
	$(MAKE) -j$(THREADS) -C $(BUILD_DIR) -f V$(TOP_MODULE).mk V$(TOP_MODULE)

sim:
	@echo "=> [STEP 4] Menjalankan Simulasi RTL..."
	@mkdir -p outputs
	./$(BUILD_DIR)/V$(TOP_MODULE)

check:
	@echo "=> [STEP 5] Membandingkan Hasil RTL vs PyTorch..."
	$(PYTHON) verify/check_gen.py

sim_icarus: weights ref
	@mkdir -p build
	@mkdir -p outputs
	@echo "=> Compiling with Icarus Verilog..."
	$(IVERILOG) -Wall -I rtl/ -o $(V_OUT) $(V_TB) $(RTL_TOP) $(RTL_SHARED)
	@echo "=> Running VVP to generate VCD..."
	$(VVP) $(V_OUT)

vivado_synth:
	@echo "=> [VIVADO] Menjalankan Sintesis FPGA..."
	@mkdir -p build_fpga
	$(VIVADO) $(VIVADO_MODE) -source build_bd_bitstream.tcl

vivado_bitstream:
	@echo "=> [VIVADO] Menjalankan Sintesis, Implementasi & Generate Bitstream..."
	@mkdir -p build_fpga
	$(VIVADO) $(VIVADO_MODE) -source build_bd_bitstream.tcl

open_vivado:
	$(VIVADO) project_pynq_bd/gan_accelerator_bd.xpr &

wave:
	gtkwave generator_top.vcd &

clean:
	@echo "=> Membersihkan file temporary..."
	rm -rf $(BUILD_DIR) build outputs weights_hex inputs *.vcd build_fpga .Xil project_pynq
	@echo "Cleaned build artifacts."