# build_bd_bitstream.tcl
# Script untuk membuat Block Design (Zynq PS + GAN IP) dan men-generate Bitstream

set project_name "gan_accelerator_bd"
set project_dir "./project_pynq_bd"
set part_name "xc7z020clg400-1"

# Create Project
create_project -force $project_name $project_dir -part $part_name

# Add RTL Files
puts "=> ADDING RTL FILES..."
add_files ./rtl/mac_array.v
add_files ./rtl/addr_gen.v
add_files ./rtl/conv_unified.v
add_files ./rtl/leaky_relu.v
add_files ./rtl/relu.v
add_files ./rtl/tanh_act.v
add_files ./rtl/generator_top.v
add_files ./rtl/gan_axi_wrapper.v
update_compile_order -fileset sources_1

# Create Block Design
puts "=> CREATING BLOCK DESIGN..."
create_bd_design "design_1"

# Add Zynq Processing System & Configure
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

# Enable S_AXI_HP0 (High Performance) for DDR access
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

# Set PL fabric clock FCLK_CLK0 to 100 MHz.
# With 2-stage pipeline in mac_array:
#   Stage 1 critical path (16-bit x 16-bit DSP multiply): ~8-10 ns  -> 100-125 MHz
#   Stage 2 critical path (4-level 48-bit adder tree):    ~6-8  ns  -> 125-167 MHz
# Bottleneck is Stage 1; 100 MHz is the conservative target.
# If WNS > 2 ns in the timing report, increase to 120 MHz and re-synthesize.
# Set PL fabric clock FCLK_CLK0 to 60 MHz.
# After two pipeline stages:
#   1. MAC array: prod_reg (Stage 1) + sum_out_reg (Stage 2) — ~10 ns/stage
#   2. Addr-gen: cfg_* -> addr_gen -> weight/ifmap_addrs_r (SETUP) — ~13-16 ns
#              weight_addrs_r -> burst_calc -> m_axi_arlen (AR)   — ~13-15 ns
# Bottleneck = addr-gen SETUP or AR path (~15 ns max).
# 60 MHz (period=16.7 ns) gives ~1.7 ns margin; increase to 75 MHz if WNS > 3 ns.
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {60}] [get_bd_cells processing_system7_0]

# Add Custom IP (gan_axi_wrapper)
create_bd_cell -type module -reference gan_axi_wrapper gan_axi_wrapper_0

# Connection Automation
puts "=> RUNNING CONNECTION AUTOMATION..."
# Connect AXI-Lite (Zynq GP0 -> gan_axi_wrapper s_axi)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/gan_axi_wrapper_0/s_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins gan_axi_wrapper_0/s_axi]

# Connect AXI4 Master (gan_axi_wrapper m_axi -> Zynq HP0)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/gan_axi_wrapper_0/m_axi} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# (Optional safety) Force connect clocks if not handled by automation
# connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins gan_axi_wrapper_0/aclk]
# connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins gan_axi_wrapper_0/aresetn]

save_bd_design
validate_bd_design

# Create Top-Level Wrapper
puts "=> CREATING HDL WRAPPER..."
set wrapper_path [make_wrapper -files [get_files [current_bd_design].bd] -top]
add_files -norecurse $wrapper_path
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Run Synthesis & Implementation
puts "=> STARTING SYNTHESIS..."
launch_runs synth_1 -jobs 10
wait_on_run synth_1

puts "=> STARTING IMPLEMENTATION..."
launch_runs impl_1 -jobs 10
wait_on_run impl_1

puts "=> GENERATING BITSTREAM..."
launch_runs impl_1 -to_step write_bitstream -jobs 10
wait_on_run impl_1

puts "=> COPYING BITSTREAM AND HWH FILES TO ROOT AND USB_GAN DIRECTORIES..."
set bit_path "./project_pynq_bd/gan_accelerator_bd.runs/impl_1/design_1_wrapper.bit"
set hwh_path "./project_pynq_bd/gan_accelerator_bd.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh"

if {[file exists $bit_path]} {
    file copy -force $bit_path "./design_1_wrapper.bit"
    file copy -force $bit_path "./USB_GAN/design_1_wrapper.bit"
    file copy -force $bit_path "./USB_GAN/gan_accelerator_bd_wrapper.bit"
    puts " -> Copied design_1_wrapper.bit successfully."
} else {
    puts " -> ERROR: Bitstream not found at $bit_path"
}

if {[file exists $hwh_path]} {
    file copy -force $hwh_path "./design_1_wrapper.hwh"
    file copy -force $hwh_path "./design_1.hwh"
    file copy -force $hwh_path "./USB_GAN/design_1_wrapper.hwh"
    file copy -force $hwh_path "./USB_GAN/design_1.hwh"
    file copy -force $hwh_path "./USB_GAN/gan_accelerator_bd_wrapper.hwh"
    puts " -> Copied design_1.hwh to all target filenames successfully."
} else {
    puts " -> ERROR: HWH file not found at $hwh_path"
}

puts "================================================="
puts "=> BISTREAM BERHASIL DIBUAT DENGAN BLOCK DESIGN! <="
puts "================================================="
