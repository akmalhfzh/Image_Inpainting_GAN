// =============================================================================
// batchnorm2d.v — Synced Channel Version
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module batchnorm2d #(
    parameter DW = 16,
    parameter FRAC_W = 8,
    parameter C = 512
)(
    input wire clk,
    input wire rst_n,
    
    // Write Interface (Weights)
    input wire wr_en,
    input wire wr_sel,      // 0: scale, 1: offset
    input wire [8:0] wr_addr,
    input wire signed [DW-1:0] wr_data,

    // Data Interface
    input wire [11:0] c_idx_in,  // [FIX] Kabel channel index langsung dari FSM
    input wire x_valid,
    input wire signed [DW-1:0] x_in,

    output reg y_valid,
    output reg signed [DW-1:0] y_out
);

    reg signed [DW-1:0] scale_mem [0:C-1];
    reg signed [DW-1:0] offset_mem [0:C-1];

    always @(posedge clk) begin
        if (wr_en) begin
            if (wr_sel == 0) scale_mem[wr_addr] <= wr_data;
            else             offset_mem[wr_addr] <= wr_data;
        end
    end

    // Pipeline 1 Cycle
    reg signed [DW-1:0] x_in_r;
    reg x_valid_r;
    reg signed [DW-1:0] scale_r, offset_r;

    wire signed [31:0] mult_res = $signed(x_in_r) * $signed(scale_r);
    wire signed [31:0] shifted_res = (mult_res >>> FRAC_W) + $signed({{16{offset_r[DW-1]}}, offset_r});

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_valid_r <= 0; y_valid <= 0;
            y_out <= 0;
        end else begin
            x_valid_r <= x_valid;
            x_in_r    <= x_in;
            
            // [FIX] Anti Desync: Selalu baca bumbu sesuai channel index dari luar
            scale_r   <= scale_mem[c_idx_in];
            offset_r  <= offset_mem[c_idx_in];

            y_valid <= x_valid_r;
            if (x_valid_r) begin
                if (shifted_res > 32767)       y_out <= 16'sd32767;
                else if (shifted_res < -32768) y_out <= -16'sd32768;
                else                           y_out <= shifted_res[15:0];
            end
        end
    end
endmodule
`default_nettype wire
