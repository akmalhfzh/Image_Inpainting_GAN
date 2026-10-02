`default_nettype none

module leaky_relu #(
    parameter DW     = 16,  // total data width (signed)
    parameter FRAC_W = 8    // fractional bits
)(
    input  wire                   valid_in,
    input  wire signed [DW-1:0]   x_in,
    output wire                   valid_out,
    output wire signed [DW-1:0]   y_out
);
    // 0.2 × 2^FRAC_W = 0.2 × 256 = 51.2 → 51
    localparam signed [DW-1:0] SLOPE = 51;

    // negative branch: (x * 51) >> 8
    // use 2×DW to avoid overflow during multiply
    wire signed [2*DW-1:0] neg_prod = x_in * SLOPE;
    wire signed [DW-1:0]   neg_val  = neg_prod >>> FRAC_W;  // arithmetic right shift

    assign valid_out = valid_in;
    assign y_out     = (x_in >= 0) ? x_in : neg_val;

endmodule
`default_nettype wire
