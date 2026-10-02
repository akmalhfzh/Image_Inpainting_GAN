`timescale 1ns/1ps
`default_nettype none

module addr_gen #(
    parameter DW    = 16,
    parameter N_MAC = 16
)(
    input  wire [11:0] c_out, c_in,
    input  wire [9:0]  h_out, w_out,

    input  wire [9:0]  cfg_h_in, cfg_w_in,
    input  wire [11:0] cfg_c_in, cfg_c_out,
    input  wire [3:0]  cfg_k, cfg_stride, cfg_pad,
    input  wire        cfg_mode,
    input  wire [25:0] cfg_weight_base,

    output reg [N_MAC*18-1:0] ifmap_addrs,
    output reg [N_MAC*26-1:0] weight_addrs,
    output reg [N_MAC-1:0]    valid_vec
);
    // ── Pre-compute intermediate widths ──────────────────────────────────────
    // hw_in = cfg_h_in * cfg_w_in, max = 64*64 = 4096 → 13-bit, pakai 18-bit
    wire [17:0] hw_in  = {8'b0, cfg_h_in} * {8'b0, cfg_w_in};

    // c_in * hw_in, max = 511*4096 = 2,093,056 → 21-bit → pakai [25:0]
    // tapi ifmap_addrs hanya 18-bit per tap, jadi kita batasi dan deteksi overflow
    // Untuk layer valid: max ifmap_sz = 512*2*2 = 2048 → addr max = 2047 = 11-bit ✓
    // Paling besar: E1 input 3*64*64=12288, addr max=12287=14-bit ✓
    // Gunakan 18-bit intermediate untuk ifmap address (MAX_IFMAP_SZ = 2^18)
    wire [17:0] cin_x_hw = c_in[11:0] * hw_in[13:0];   // max 511*4096=2M, perlu 21-bit
    // Untuk safety pakai 25-bit intermediate lalu truncate ke 18-bit
    wire [24:0] cin_x_hw_wide = {13'b0, c_in} * {7'b0, hw_in};

    // ── Weight address intermediates (26-bit) ─────────────────────────────────
    wire [7:0]  kk         = {4'b0, cfg_k} * {4'b0, cfg_k};   // max 4*4=16
    wire [25:0] cin_x_kk   = {14'b0, cfg_c_in}  * {18'b0, kk};
    wire [25:0] cout_x_kk  = {14'b0, cfg_c_out} * {18'b0, kk};
    wire [25:0] stride_cout = {14'b0, c_out} * cin_x_kk;
    wire [25:0] step_cin    = {14'b0, c_in}  * {18'b0, kk};
    wire [25:0] tr_stride_cin = {14'b0, c_in}  * cout_x_kk;
    wire [25:0] tr_step_cout  = {14'b0, c_out} * {18'b0, kk};

    integer t, dy, dx;
    integer in_y, in_x, ifmap_y, ifmap_x;
    integer hw_in_int;   // integer copy untuk arithmetic di always block

    always @(*) begin
        hw_in_int = cfg_h_in * cfg_w_in;   // integer: no overflow di 32-bit

        for (t = 0; t < N_MAC; t = t + 1) begin
            if (t < cfg_k * cfg_k) begin

                // ── dy, dx dari flat index t ─────────────────────────────────
                if (cfg_k == 4)      begin dy = t / 4; dx = t % 4; end
                else if (cfg_k == 3) begin dy = t / 3; dx = t % 3; end
                else                 begin dy = 0;     dx = 0;     end

                if (cfg_mode == 0) begin
                    // ════════════════════════════════════════════════════════
                    // Conv2d: in_y = h_out*stride - pad + dy
                    // ════════════════════════════════════════════════════════
                    in_y = h_out * cfg_stride - cfg_pad + dy;
                    in_x = w_out * cfg_stride - cfg_pad + dx;

                    if (in_y >= 0 && in_y < cfg_h_in &&
                        in_x >= 0 && in_x < cfg_w_in) begin
                        valid_vec[t] = 1'b1;
                        // ifmap addr = c_in*H_in*W_in + in_y*W_in + in_x
                        // Semua integer 32-bit, tidak overflow
                        ifmap_addrs[t*18 +: 18] = c_in * hw_in_int
                                                  + in_y * cfg_w_in + in_x;
                    end else begin
                        valid_vec[t]            = 1'b0;
                        ifmap_addrs[t*18 +: 18] = 18'd0;
                    end

                    // weight addr = base + c_out*(C_in*k*k) + c_in*(k*k) + t
                    weight_addrs[t*26 +: 26] = cfg_weight_base
                                              + stride_cout
                                              + step_cin
                                              + t[25:0];

                end else begin
                    // ════════════════════════════════════════════════════════
                    // ConvTranspose2d: in_y = h_out + pad - dy
                    // ════════════════════════════════════════════════════════
                    in_y = h_out + cfg_pad - dy;
                    in_x = w_out + cfg_pad - dx;

                    if (in_y >= 0 &&
                        (in_y % cfg_stride) == 0 &&
                        in_x >= 0 &&
                        (in_x % cfg_stride) == 0) begin

                        ifmap_y = in_y / cfg_stride;
                        ifmap_x = in_x / cfg_stride;

                        if (ifmap_y < cfg_h_in && ifmap_x < cfg_w_in) begin
                            valid_vec[t] = 1'b1;
                            ifmap_addrs[t*18 +: 18] = c_in * hw_in_int
                                                     + ifmap_y * cfg_w_in
                                                     + ifmap_x;
                        end else begin
                            valid_vec[t]            = 1'b0;
                            ifmap_addrs[t*18 +: 18] = 18'd0;
                        end
                    end else begin
                        valid_vec[t]            = 1'b0;
                        ifmap_addrs[t*18 +: 18] = 18'd0;
                    end

                    // weight addr (ConvTranspose2d)
                    // PyTorch storage: [C_in, C_out, kH, kW]
                    // → base + c_in*(C_out*k*k) + c_out*(k*k) + t
                    weight_addrs[t*26 +: 26] = cfg_weight_base
                                              + tr_stride_cin
                                              + tr_step_cout
                                              + t[25:0];
                end

            end else begin
                valid_vec[t]             = 1'b0;
                ifmap_addrs[t*18 +: 18]  = 18'd0;
                weight_addrs[t*26 +: 26] = 26'd0;
            end
        end
    end

endmodule
`default_nettype wire
