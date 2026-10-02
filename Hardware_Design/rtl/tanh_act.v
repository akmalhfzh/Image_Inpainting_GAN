`default_nettype none

module tanh_act #(
    parameter DW     = 16,
    parameter FRAC_W = 8
)(
    input  wire                   clk,
    input  wire                   valid_in,
    input  wire signed [DW-1:0]   x_in,
    output wire                   valid_out,
    output wire signed [DW-1:0]   y_out
);
    // ONE = 1.0 in Q7.8
    localparam signed [DW-1:0] ONE = (1 << FRAC_W);  // 256

    // ── LUT: tanh(k × 0.125) × 256 for k = 0..32 ─────────────────────────
    // Values computed as: round(tanh(k/8) * 256)
    // k=0:  tanh(0.000) = 0.000  → 0
    // k=1:  tanh(0.125) = 0.125  → 32
    // k=2:  tanh(0.250) = 0.245  → 63
    // k=3:  tanh(0.375) = 0.360  → 92
    // k=4:  tanh(0.500) = 0.462  → 118
    // k=5:  tanh(0.625) = 0.555  → 142
    // k=6:  tanh(0.750) = 0.635  → 163
    // k=7:  tanh(0.875) = 0.704  → 180
    // k=8:  tanh(1.000) = 0.762  → 195
    // k=9:  tanh(1.125) = 0.811  → 208
    // k=10: tanh(1.250) = 0.848  → 217
    // k=11: tanh(1.375) = 0.878  → 225
    // k=12: tanh(1.500) = 0.905  → 232
    // k=13: tanh(1.625) = 0.925  → 237
    // k=14: tanh(1.750) = 0.942  → 241
    // k=15: tanh(1.875) = 0.956  → 245
    // k=16: tanh(2.000) = 0.964  → 247
    // k=17: tanh(2.125) = 0.973  → 249
    // k=18: tanh(2.250) = 0.978  → 250
    // k=19: tanh(2.375) = 0.983  → 252
    // k=20: tanh(2.500) = 0.987  → 253
    // k=21: tanh(2.625) = 0.989  → 253
    // k=22: tanh(2.750) = 0.992  → 254
    // k=23: tanh(2.875) = 0.993  → 254
    // k=24: tanh(3.000) = 0.995  → 255
    // k=25: tanh(3.125) = 0.996  → 255
    // k=26: tanh(3.250) = 0.997  → 255
    // k=27: tanh(3.375) = 0.997  → 255
    // k=28: tanh(3.500) = 0.998  → 256
    // k=29: tanh(3.625) = 0.998  → 256
    // k=30: tanh(3.750) = 0.999  → 256
    // k=31: tanh(3.875) = 0.999  → 256
    // k=32: tanh(4.000) = 1.000  → 256  (saturate)
    reg [DW-1:0] lut [0:32];
    initial begin
        lut[ 0]=16'd0;  lut[ 1]=16'd32;  lut[ 2]=16'd63;  lut[ 3]=16'd92;
        lut[ 4]=16'd118;lut[ 5]=16'd142; lut[ 6]=16'd163; lut[ 7]=16'd180;
        lut[ 8]=16'd195;lut[ 9]=16'd208; lut[10]=16'd217; lut[11]=16'd225;
        lut[12]=16'd232;lut[13]=16'd237; lut[14]=16'd241; lut[15]=16'd245;
        lut[16]=16'd247;lut[17]=16'd249; lut[18]=16'd250; lut[19]=16'd252;
        lut[20]=16'd253;lut[21]=16'd253; lut[22]=16'd254; lut[23]=16'd254;
        lut[24]=16'd255;lut[25]=16'd255; lut[26]=16'd255; lut[27]=16'd255;
        lut[28]=16'd256;lut[29]=16'd256; lut[30]=16'd256; lut[31]=16'd256;
        lut[32]=16'd256;
    end

    // ── Combinational stage ────────────────────────────────────────────────
    // Absolute value
    wire signed [DW-1:0] x_abs   = x_in[DW-1] ? -x_in : x_in;

    // Segment index: seg = |x| / 0.125  = |x| >> (FRAC_W - 3)
    // FRAC_W=8, seg_width=0.125=2^-3 → seg = x_abs >> (8-3) = x_abs >> 5
    wire [DW-1:0] seg_full = x_abs >> (FRAC_W - 3);          // = |x| / 0.125

    // Clamp to 31 (max segment index, saturate above 4.0)
    wire [4:0]  seg_idx  = (seg_full >= 32) ? 5'd31 : seg_full[4:0];
    wire        sat      = (seg_full >= 32);

    // Fractional part within segment: lower (FRAC_W-3) = 5 bits of x_abs
    wire [FRAC_W-4:0] frac = x_abs[FRAC_W-4:0];  // 4 bits for FRAC_W=8

    // LUT lookup
    wire [DW-1:0] lo   = lut[seg_idx];
    wire [DW-1:0] hi   = lut[seg_idx + 1];
    wire [DW-1:0] diff = (hi >= lo) ? hi - lo : 16'd0;

    // Linear interpolation: result = lo + diff * frac / seg_width
    // seg_width in frac bits = 2^(FRAC_W-3) = 32  (for FRAC_W=8)
    // so result = lo + (diff * frac) >> (FRAC_W-3)
    wire [2*DW-1:0] interp_full = diff * {{(DW-FRAC_W+3){1'b0}}, frac};
    wire [DW-1:0]   interp_val  = lo + (interp_full >> (FRAC_W - 3));

    // Saturate or interpolate
    wire [DW-1:0]  mag   = sat ? ONE : interp_val;

    // Apply sign
    wire signed [DW-1:0] tanh_comb = x_in[DW-1] ? -$signed(mag) : $signed(mag);

    // ── Combinational output (0 cycle latency, matches relu/leaky_relu) ─────
    wire _unused_clk = clk;
    assign valid_out = valid_in;
    assign y_out     = tanh_comb;

endmodule
`default_nettype wire
