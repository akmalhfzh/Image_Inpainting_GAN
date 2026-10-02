# scripts/build_bitstream.tcl

# PYNQ-Z1 (Part Number xc7z020clg400-1)
create_project -force gan_accelerator ./project_pynq -part xc7z020clg400-1

add_files ./rtl/mac_array.v
add_files ./rtl/addr_gen.v
add_files ./rtl/conv_unified.v
add_files ./rtl/leaky_relu.v
add_files ./rtl/relu.v
add_files ./rtl/tanh_act.v
add_files ./rtl/generator_top.v
add_files ./rtl/gan_axi_wrapper.v
set_property top gan_axi_wrapper [current_fileset]

# Sintesis
launch_runs synth_1 -jobs 10
wait_on_run synth_1

puts "=> SINTESIS SELESAI! Buka GUI untuk merakit Block Design."
