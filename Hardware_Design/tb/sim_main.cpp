// =============================================================================
// sim_main.cpp — AXI DDR Memory Model Verilator Simulation
// =============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include "Vgenerator_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

// -----------------------------------------------------------------------------
// AXI DDR Memory Simulation
// -----------------------------------------------------------------------------
// We simulate 128 MB of DDR memory. Addresses are byte addresses.
// Because m_axi_rdata is 16-bit, we store it as a vector of uint16_t (2 bytes).
// Max byte address = 128MB. So size = 64M elements.
std::vector<uint16_t> ddr_mem((128 * 1024 * 1024) / 2, 0);

struct WeightFile {
    const char* fname;
    int         layer;
    int         type;
};

// Memory Map Constants
const uint32_t BASE_WEIGHT = 0x0000000;   // Weights start at 0
const uint32_t BASE_IMG    = 0x4000000;   // ~64MB
const uint32_t BASE_FMAP_A = 0x5000000;   // ~80MB
const uint32_t BASE_FMAP_B = 0x6000000;   // ~96MB
const uint32_t BASE_OUT    = 0x7000000;   // ~112MB

// -----------------------------------------------------------------------------
// Synchronous AXI Slave Memory Logic
// -----------------------------------------------------------------------------
static int ar_burst_count = 0;
static uint32_t ar_addr = 0;
static bool ar_active = false;

static int aw_burst_count = 0;
static uint32_t aw_addr = 0;
static bool aw_active = false;
static bool b_active = false;

static void eval_axi_comb(Vgenerator_top* top) {
    top->m_axi_arready = !ar_active;
    
    if (ar_active) {
        top->m_axi_rvalid = 1;
        uint32_t aligned = ar_addr & ~0x3u;
        uint16_t lower = ddr_mem[(aligned / 2) % ddr_mem.size()];
        uint16_t upper = ddr_mem[(aligned / 2 + 1) % ddr_mem.size()];
        top->m_axi_rdata = ((uint32_t)upper << 16) | (uint32_t)lower;
        top->m_axi_rresp = 0;
        top->m_axi_rlast = (ar_burst_count == 1);
    } else {
        top->m_axi_rvalid = 0;
        top->m_axi_rlast  = 0;
        top->m_axi_rdata  = 0;
    }

    top->m_axi_awready = !aw_active && !b_active;
    top->m_axi_wready  = aw_active;
    top->m_axi_bvalid  = b_active;
    top->m_axi_bresp   = 0;
}

static void step_axi_posedge(bool arvalid, bool arready, uint32_t araddr, uint8_t arlen, uint8_t arsize,
                             bool rvalid, bool rready,
                             bool awvalid, bool awready, uint32_t awaddr, uint8_t awlen, uint8_t awsize,
                             bool wvalid, bool wready, uint32_t wdata, uint8_t wstrb, bool wlast,
                             bool bvalid, bool bready) {
    // Read Channel Handshake
    if (arvalid && arready && !ar_active) {
        ar_addr = araddr;
        ar_burst_count = arlen + 1;
        ar_active = true;
    } else if (ar_active && rvalid && rready) {
        uint32_t step_bytes = (arsize == 1) ? 2 : 4;
        ar_addr += step_bytes;
        ar_burst_count--;
        if (ar_burst_count == 0) {
            ar_active = false;
        }
    }

    // Write Address Channel
    if (awvalid && awready && !aw_active) {
        aw_addr = awaddr;
        aw_burst_count = awlen + 1;
        aw_active = true;
    } else if (aw_active && wvalid && wready) {
        uint32_t step_bytes = (awsize == 1) ? 2 : 4;
        uint32_t aligned = aw_addr & ~0x3u;
        if (wstrb & 0x3) {
            ddr_mem[(aligned / 2) % ddr_mem.size()] = wdata & 0xFFFF;
        }
        if (wstrb & 0xC) {
            ddr_mem[(aligned / 2 + 1) % ddr_mem.size()] = (wdata >> 16) & 0xFFFF;
        }
        aw_addr += step_bytes;
        aw_burst_count--;
        if (wlast || aw_burst_count == 0) {
            aw_active = false;
            b_active  = true;
        }
    }

    // Write Response Channel
    if (b_active && bvalid && bready) {
        b_active = false;
    }
}

static void tick(Vgenerator_top* top, VerilatedVcdC* tfp = nullptr, uint64_t* vcd_time = nullptr) {
    eval_axi_comb(top);
    top->clk = 0; top->eval();
    if (tfp && vcd_time) { tfp->dump(*vcd_time); (*vcd_time)++; }

    eval_axi_comb(top);

    // Sample handshakes BEFORE clock posedge updates flip-flops!
    bool s_arvalid = top->m_axi_arvalid;
    bool s_arready = top->m_axi_arready;
    uint32_t s_araddr = top->m_axi_araddr;
    uint8_t  s_arlen  = top->m_axi_arlen;
    uint8_t  s_arsize = top->m_axi_arsize;

    bool s_rvalid  = top->m_axi_rvalid;
    bool s_rready  = top->m_axi_rready;

    bool s_awvalid = top->m_axi_awvalid;
    bool s_awready = top->m_axi_awready;
    uint32_t s_awaddr = top->m_axi_awaddr;
    uint8_t  s_awlen  = top->m_axi_awlen;
    uint8_t  s_awsize = top->m_axi_awsize;

    bool s_wvalid  = top->m_axi_wvalid;
    bool s_wready  = top->m_axi_wready;
    uint32_t s_wdata  = top->m_axi_wdata;
    uint8_t  s_wstrb  = top->m_axi_wstrb;
    bool     s_wlast  = top->m_axi_wlast;

    bool s_bvalid  = top->m_axi_bvalid;
    bool s_bready  = top->m_axi_bready;

    top->clk = 1; top->eval();

    step_axi_posedge(s_arvalid, s_arready, s_araddr, s_arlen, s_arsize,
                     s_rvalid, s_rready,
                     s_awvalid, s_awready, s_awaddr, s_awlen, s_awsize,
                     s_wvalid, s_wready, s_wdata, s_wstrb, s_wlast,
                     s_bvalid, s_bready);

    eval_axi_comb(top);
    top->eval();
    if (tfp && vcd_time) { tfp->dump(*vcd_time); (*vcd_time)++; }
}

// -----------------------------------------------------------------------------
// Loaders
// -----------------------------------------------------------------------------
void load_hex_to_ddr(const char* filename, uint32_t base_byte_addr) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "  [WARN] Missing: " << filename << std::endl;
        return;
    }
    std::string line;
    uint32_t word_offset = 0;
    size_t count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '/') continue;
        int16_t val = static_cast<int16_t>(std::stoul(line, nullptr, 16) & 0xFFFF);
        ddr_mem[(base_byte_addr / 2) + word_offset] = val;
        word_offset++;
        count++;
    }
    std::cout << "  -> Loaded " << count << " entries from " << filename << std::endl;
}

void load_bias_to_reg(Vgenerator_top* top, const char* filename, int layer) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    uint32_t addr = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '/') continue;
        top->wr_layer = layer;
        top->wr_type  = 3; // bias type
        top->wr_addr  = addr++;
        int16_t val   = static_cast<int16_t>(std::stoul(line, nullptr, 16) & 0xFFFF);
        top->wr_data  = val;
        top->wr_en    = 1;
        tick(top);
    }
    top->wr_en = 0;
    tick(top);
}

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vgenerator_top* top = new Vgenerator_top;

    // Reset
    top->rst_n = 0; top->clk = 0; top->start = 0; top->wr_en = 0;
    for (int i = 0; i < 20; i++) tick(top);
    top->rst_n = 1;
    tick(top);

    // Provide memory bases
    top->reg_img_base    = BASE_IMG;
    top->reg_fmap_a_base = BASE_FMAP_A;
    top->reg_fmap_b_base = BASE_FMAP_B;
    top->reg_out_base    = BASE_OUT;
    top->reg_weight_base = BASE_WEIGHT;

    // =========================================================================
    // [1] Load Input Image to DDR
    // =========================================================================
    std::cout << "[Verilator] Loading Input Image to DDR..." << std::endl;
    load_hex_to_ddr("outputs/py_ref/input_image.hex", BASE_IMG);

    // =========================================================================
    // [2] Load Weights to DDR & Biases to Registers
    // =========================================================================
    std::cout << "[Verilator] Loading Weights to DDR & Biases to Registers..." << std::endl;

    std::vector<WeightFile> w_list = {
        {"weights_hex/E1_weight.hex",  0,  0}, {"weights_hex/E1_bias.hex",    0,  3},
        {"weights_hex/E2_weight.hex",  1,  0}, {"weights_hex/E2_bias.hex",    1,  3},
        {"weights_hex/E3_weight.hex",  2,  0}, {"weights_hex/E3_bias.hex",    2,  3},
        {"weights_hex/E4_weight.hex",  3,  0}, {"weights_hex/E4_bias.hex",    3,  3},
        {"weights_hex/E5_weight.hex",  4,  0}, {"weights_hex/E5_bias.hex",    4,  3},
        {"weights_hex/BOT_weight.hex", 5,  0}, {"weights_hex/BOT_bias.hex",   5,  3},
        {"weights_hex/D1_weight.hex",  6,  0}, {"weights_hex/D1_bias.hex",    6,  3},
        {"weights_hex/D2_weight.hex",  7,  0}, {"weights_hex/D2_bias.hex",    7,  3},
        {"weights_hex/D3_weight.hex",  8,  0}, {"weights_hex/D3_bias.hex",    8,  3},
        {"weights_hex/D4_weight.hex",  9,  0}, {"weights_hex/D4_bias.hex",    9,  3},
        {"weights_hex/D5_weight.hex",  10, 0}, {"weights_hex/D5_bias.hex",    10, 3},
        {"weights_hex/OUT_weight.hex", 11, 0}, {"weights_hex/OUT_bias.hex",   11, 3},
    };

    uint32_t current_weight_offset_bytes = BASE_WEIGHT;
    for (const auto& w : w_list) {
        if (w.type == 0) {
            // Write to DDR
            // But wait, the generator_top uses hardcoded cfg_weight_base
            // We must place the weights exactly where generator_top expects them!
            uint32_t expected_word_offset = 0;
            switch(w.layer) {
                case 0: expected_word_offset=0; break;
                case 1: expected_word_offset=3072; break;
                case 2: expected_word_offset=68608; break;
                case 3: expected_word_offset=199680; break;
                case 4: expected_word_offset=723968; break;
                case 5: expected_word_offset=2821120; break;
                case 6: expected_word_offset=3083264; break;
                case 7: expected_word_offset=5180672; break;
                case 8: expected_word_offset=5705088; break;
                case 9: expected_word_offset=5836288; break;
                case 10: expected_word_offset=5869088; break;
                case 11: expected_word_offset=5901888; break;
            }
            load_hex_to_ddr(w.fname, BASE_WEIGHT + expected_word_offset * 2);
        } else {
            // Write bias to registers
            load_bias_to_reg(top, w.fname, w.layer);
        }
    }

    // =========================================================================
    // [3] Run Inference & Dump VCD
    // =========================================================================
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("generator_top.vcd");
    uint64_t vcd_time = 0;

    std::cout << "[Verilator] Starting Inference (VCD enabled for first 50k cycles)..." << std::endl;
    top->start = 1;
    tick(top, tfp, &vcd_time);
    top->start = 0;

    uint64_t main_time   = 0;
    const uint64_t LIMIT = 1500000000ULL; // 1.5 Billion cycles

    bool prev_layer_done = false;
    const char* layer_names[] = {"E1", "E2", "E3", "E4", "E5", "BOT", "D1", "D2", "D3", "D4", "D5", "OUT"};
    
    while (!top->done && main_time < LIMIT) {
        if (main_time < 100000ULL) {
            tick(top, tfp, &vcd_time);
        } else {
            tick(top, nullptr, nullptr);
        }
        
        // Dump feature map when layer is done
        if (top->dbg_layer_done && !prev_layer_done) {
            int layer = top->dbg_layer_idx;
            uint32_t base = (layer == 11) ? BASE_OUT : ((layer % 2 == 0) ? BASE_FMAP_B : BASE_FMAP_A);
            
            // To be accurate with ping-pong, if layer 0 finishes, it wrote to FMAP_A (if ping_pong=0)
            // Wait, in generator_top.v: cur_ofmap_base = (layer==11)? OUT : (ping_pong? FMAP_A : FMAP_B)
            // layer 0: ping_pong=0 -> writes to FMAP_B
            // layer 1: ping_pong=1 -> writes to FMAP_A
            // Let's just dump both FMAP_A and FMAP_B entirely, but check_gen.py needs exactly 1 file per layer
            // For now, check_gen.py expects `outputs/rtl_layer_X.hex`
            // Let's dump the region that was just written.
            uint32_t dump_base = BASE_OUT;
            if (layer < 11) {
                dump_base = (layer % 2 == 0) ? BASE_FMAP_B : BASE_FMAP_A;
            }
            
            char fname[256];
            snprintf(fname, sizeof(fname), "outputs/rtl_layer_%d.hex", layer);
            std::ofstream out_file(fname);
            uint32_t expected_words = 0;
            switch(layer) {
                case 0:  expected_words = 32768; break;
                case 1:  expected_words = 16384; break;
                case 2:  expected_words = 8192; break;
                case 3:  expected_words = 4096; break;
                case 4:  expected_words = 2048; break;
                case 5:  expected_words = 2048; break;
                case 6:  expected_words = 4096; break;
                case 7:  expected_words = 8192; break;
                case 8:  expected_words = 16384; break;
                case 9:  expected_words = 32768; break;
                case 10: expected_words = 131072; break;
                case 11: expected_words = 12288; break;
            }
            for (uint32_t i = 0; i < expected_words; i++) {
                uint16_t val = ddr_mem[(dump_base / 2) + i];
                out_file << std::hex << std::setw(4) << std::setfill('0') << val << "\n";
            }
            std::cout << "  [SIM] Dumped " << fname << " (" << expected_words << " words) at cycle " << main_time << std::endl;
        }
        prev_layer_done = top->dbg_layer_done;

        main_time++;
        if (main_time % 1000000 == 0) {
            std::cout << "  [SIM] " << main_time / 1000000 << "M cycles elapsed..." << std::endl;
        }
    }

    if (main_time >= LIMIT) {
        std::cerr << "[ERROR] Simulation TIMEOUT after " << main_time << " cycles!" << std::endl;
    } else {
        std::cout << "[Verilator] Done after " << main_time << " cycles." << std::endl;
        
        // Dump the output from DDR
        std::ofstream out_file("outputs/rtl_gen_img.hex");
        std::ofstream out_file2("outputs/output_rtl.hex");
        uint32_t out_words = 3 * 64 * 64;
        for (uint32_t i = 0; i < out_words; i++) {
            uint16_t val = ddr_mem[(BASE_OUT / 2) + i];
            out_file << std::hex << std::setw(4) << std::setfill('0') << val << "\n";
            out_file2 << std::hex << std::setw(4) << std::setfill('0') << val << "\n";
        }
        std::cout << "  -> Output saved to outputs/rtl_gen_img.hex and outputs/output_rtl.hex" << std::endl;
    }

    if (tfp) { tfp->close(); delete tfp; }
    top->final();
    delete top;
    return 0;
}
