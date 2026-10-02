// =============================================================================
// conv_unified (AXI4 Master, Per-Pixel Tile Strategy)
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module conv_unified #(
    parameter DW    = 16,
    parameter ACC_W = 48,
    parameter FRAC_W = 8
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // ---------------------------------------------------
    // AXI4 Master Interface
    // ---------------------------------------------------
    output reg  [31:0] m_axi_araddr,
    output reg  [7:0]  m_axi_arlen,
    output reg  [2:0]  m_axi_arsize,
    output reg  [1:0]  m_axi_arburst,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rlast,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,

    output reg  [31:0] m_axi_awaddr,
    output reg  [7:0]  m_axi_awlen,
    output reg  [2:0]  m_axi_awsize,
    output reg  [1:0]  m_axi_awburst,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    output reg  [31:0] m_axi_wdata,
    output reg  [3:0]  m_axi_wstrb,
    output reg         m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,

    // ---------------------------------------------------
    // Configuration
    // ---------------------------------------------------
    input  wire [9:0]  cfg_h_in,  cfg_w_in,
    input  wire [9:0]  cfg_h_out, cfg_w_out,
    input  wire [11:0] cfg_c_in,  cfg_c_out,
    input  wire [3:0]  cfg_k, cfg_stride, cfg_pad,
    input  wire        cfg_mode,
    input  wire [25:0] cfg_weight_base,
    input  wire [31:0] cfg_ifmap_base,
    input  wire [31:0] cfg_ofmap_base,
    input  wire [31:0] reg_weight_base,

    input  wire signed [DW-1:0] bias_val,
    output wire [11:0]          current_c_out,

    // Activation selection
    input  wire [1:0]           cur_act // 0: LReLU, 1: ReLU, 2: Tanh, 3: None
);

    localparam N_MAC = 16;
    localparam FRAC_W_LOCAL = FRAC_W;

    // ---------------------------------------------------
    // Tile Buffers (BRAM) - Stores exactly 1 patch
    // ---------------------------------------------------
    (* ram_style = "block" *) reg signed [15:0] weight_cache_0 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_1 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_2 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_3 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_4 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_5 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_6 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_7 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_8 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_9 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_10 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_11 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_12 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_13 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_14 [0:511];
    (* ram_style = "block" *) reg signed [15:0] weight_cache_15 [0:511];

    reg signed [DW-1:0] weight_tile [0:15];
    reg signed [DW-1:0] ifmap_tile  [0:15];
    reg signed [ACC_W-1:0] acc_reg;

    reg [11:0] c_out_cnt, c_in_cnt, fill_c_in_cnt;
    reg [9:0]  h_cnt, w_cnt;
    reg [4:0]  wc;       // weight-word counter (0..k*k-1) during weight fill
    reg [4:0]  tap_cnt;  // ifmap tap counter (0..15) during ifmap gather (transpose path)
    reg [2:0]  fr;       // [STEP2] fetch-row counter (0..k-1) for conv2d row-burst
    reg [4:0]  beat_cnt; // [STEP2] beat index within a row-burst
    reg        ph;       // [HW-SAFE] half phase within a 4-byte beat (0=low,1=high)
    assign current_c_out = c_out_cnt;

    // ---------------------------------------------------
    // FSM State Definitions
    // ---------------------------------------------------
    localparam ST_IDLE          = 4'd0;
    localparam ST_FILL_W_AR     = 4'd1;
    localparam ST_FILL_W_R      = 4'd2;
    localparam ST_FETCH_I_AR    = 4'd3;
    localparam ST_FETCH_I_R     = 4'd4;
    localparam ST_COMPUTE_ISSUE = 4'd5;
    localparam ST_WB_AW         = 4'd6;
    localparam ST_WB_W          = 4'd7;
    localparam ST_WB_B          = 4'd8;
    localparam ST_COMPUTE_WAIT  = 4'd9;
    localparam ST_FILL_W_SETUP  = 4'd10;
    localparam ST_FETCH_I_SETUP = 4'd11;

    reg [3:0] state;

    // ---------------------------------------------------
    // Address Generator (Combinational)
    // ---------------------------------------------------
    wire [N_MAC*18-1:0] ifmap_addrs_w;
    wire [N_MAC*26-1:0] weight_addrs_w;
    wire [N_MAC-1:0]    valid_vec_w;

    // ST_FILL_W_SETUP also uses fill_c_in_cnt so addr_gen sees correct c_in
    // while weight_addrs_r is being registered (1 cycle before ST_FILL_W_AR).
    wire [11:0] effective_c_in = (state == ST_FILL_W_SETUP || state == ST_FILL_W_AR || state == ST_FILL_W_R) ? fill_c_in_cnt : c_in_cnt;

    addr_gen u_addr_gen (
        .c_out(c_out_cnt), .c_in(effective_c_in), .h_out(h_cnt), .w_out(w_cnt),
        .cfg_h_in(cfg_h_in), .cfg_w_in(cfg_w_in),
        .cfg_c_in(cfg_c_in), .cfg_c_out(cfg_c_out),
        .cfg_k(cfg_k), .cfg_stride(cfg_stride), .cfg_pad(cfg_pad),
        .cfg_mode(cfg_mode), .cfg_weight_base(cfg_weight_base),
        .ifmap_addrs(ifmap_addrs_w), .weight_addrs(weight_addrs_w),
        .valid_vec(valid_vec_w)
    );

    // -----------------------------------------------------------------------
    // [ADDR-GEN PIPELINE] Registered addr_gen outputs
    // Captured in SETUP states (1 cycle before AR states) to break the
    // critical path: cfg_* -> addr_gen (comb.) -> burst_calc -> m_axi_arlen.
    // After this register, each sub-path is approximately half the original.
    // -----------------------------------------------------------------------
    reg [N_MAC*26-1:0] weight_addrs_r;
    reg [N_MAC*18-1:0] ifmap_addrs_r;
    reg [N_MAC-1:0]    valid_vec_r;

    always @(posedge clk) begin
        weight_addrs_r <= weight_addrs_w;
        ifmap_addrs_r  <= ifmap_addrs_w;
        valid_vec_r    <= valid_vec_w;
    end

    wire [4:0]  kk_local   = cfg_k * cfg_k;                       // taps per kernel (<=16)

    // =======================================================================
    // [HW-SAFE BURST] All ifmap/weight reads use FULL-WIDTH 4-byte beats
    // (arsize=010, INCR, 4-byte-aligned start) — safe for the Zynq AXI
    // interconnect (no narrow bursts). Each beat carries TWO 16-bit elements:
    //   rdata[15:0]  = even element,  rdata[31:16] = odd element.
    // We consume ONE element per cycle across two phases (ph=0 low half,
    // ph=1 high half) so a single demux suffices, and only advance the AXI
    // beat (rready) on the high phase. A run of N contiguous elements starting
    // at element e0 spans nbeats = ceil((e0[0]+N)/2) beats; element e0+pos lands
    // in beat (e0[0]+pos)/2, half (e0[0]+pos)&1.
    // =======================================================================

    // ---- WEIGHT run: whole kernel (kk contiguous elems) from w_addr0 ----
    wire [25:0] w_addr0  = weight_addrs_w[0 +: 26];      // tap-0 weight element
    wire        w_off    = w_addr0[0];                   // start offset in first word
    wire [25:0] w_ealign = {w_addr0[25:1], 1'b0};        // 4-byte-aligned start elem
    wire [4:0]  w_nbeats = ({4'b0, w_off} + kk_local + 5'd1) >> 1; // ceil((off+kk)/2)
    wire [5:0]  w_pos    = ph ? ({wc, 1'b0} - {5'b0, w_off} + 6'd1)  // hi: 2*wc-off+1
                              : ({wc, 1'b0} - {5'b0, w_off});        // lo: 2*wc-off
    wire        w_lo_under = (w_off & (wc == 5'd0));     // 2*wc<off only at wc=0,off=1
    wire        w_pos_vld  = (ph ? 1'b1 : ~w_lo_under) && (w_pos < {1'b0, kk_local});
    wire signed [15:0] w_half = ph ? m_axi_rdata[31:16] : m_axi_rdata[15:0];

    // -----------------------------------------------------------------------
    // [STEP2/3] Per-kernel-row ifmap burst (both conv2d & transpose).
    // Within kernel row `fr` the VALID taps map to a CONTIGUOUS run of ifmap
    // elements (addresses +1 per element). Invalid tile slots are masked by
    // valid_vec in the MAC -> no zeroing needed.
    //   conv2d (mode0): valid dx contiguous & in order. base = first valid dx;
    //                   run-pos p -> slot = base_tap + p.
    //   transpose (mode1): valid dx strided & REVERSED but ifmap_x consecutive.
    //                   base = LAST valid dx (lowest addr); run-pos p ->
    //                   slot = base_tap - p*stride.
    // -----------------------------------------------------------------------
    wire [4:0] row_t0 = fr * cfg_k;                 // first tap index of row fr
    wire rvb0 =                  valid_vec_w[row_t0 + 5'd0];
    wire rvb1 = (cfg_k > 4'd1) ? valid_vec_w[row_t0 + 5'd1] : 1'b0;
    wire rvb2 = (cfg_k > 4'd2) ? valid_vec_w[row_t0 + 5'd2] : 1'b0;
    wire rvb3 = (cfg_k > 4'd3) ? valid_vec_w[row_t0 + 5'd3] : 1'b0;
    wire       row_any     = rvb0 | rvb1 | rvb2 | rvb3;
    wire [2:0] row_dx_first = rvb0 ? 3'd0 : rvb1 ? 3'd1 : rvb2 ? 3'd2 : 3'd3; // min valid dx
    wire [2:0] row_dx_last  = rvb3 ? 3'd3 : rvb2 ? 3'd2 : rvb1 ? 3'd1 : 3'd0; // max valid dx
    wire [2:0] row_base_dx  = (cfg_mode == 1'b0) ? row_dx_first : row_dx_last;
    wire [2:0] row_cnt      = {2'b0,rvb0} + {2'b0,rvb1} + {2'b0,rvb2} + {2'b0,rvb3}; // #valid elems
    wire [4:0] row_base_tap = row_t0 + {2'b0,row_base_dx};      // tile slot of burst base tap
    wire [17:0] row_base_addr = ifmap_addrs_w[row_base_tap*18 +: 18]; // lowest address of the run

    wire        i_off    = row_base_addr[0];
    wire [17:0] i_ealign = {row_base_addr[17:1], 1'b0};
    wire [4:0]  i_nbeats = ({4'b0, i_off} + {2'b0, row_cnt} + 5'd1) >> 1; // ceil((off+cnt)/2)
    wire [5:0]  i_pos    = ph ? ({beat_cnt, 1'b0} - {5'b0, i_off} + 6'd1)
                              : ({beat_cnt, 1'b0} - {5'b0, i_off});
    wire        i_lo_under = (i_off & (beat_cnt == 5'd0));
    wire        i_pos_vld  = (ph ? 1'b1 : ~i_lo_under) && (i_pos < {3'b0, row_cnt});
    wire signed [15:0] i_half = ph ? m_axi_rdata[31:16] : m_axi_rdata[15:0];
    wire [5:0]  i_pos_x_str = i_pos[4:0] * cfg_stride;
    wire [4:0]  i_burst_slot = (cfg_mode == 1'b0)
                             ? (row_base_tap + i_pos[4:0])
                             : (row_base_tap - i_pos_x_str[4:0]);

    // -----------------------------------------------------------------------
    // [AXI-SAFE] 4KB Boundary Crossing Prevention Logic
    // AMBA AXI4 protocol forbids INCR bursts from crossing a 4KB (0x1000) page
    // boundary. We calculate the remaining beats in the current 4KB page and
    // clamp the burst length (arlen) accordingly.
    // -----------------------------------------------------------------------
    wire [31:0] w_base_byte   = reg_weight_base + ({6'b0, w_ealign} << 1);
    wire [31:0] w_cur_addr    = w_base_byte + ({26'b0, wc} << 2);
    wire [4:0]  w_rem_beats   = w_nbeats - wc;
    wire [10:0] w_page_beats  = 11'd1024 - {1'b0, w_cur_addr[11:2]};
    wire [4:0]  w_burst_beats = ({6'b0, w_rem_beats} <= w_page_beats) ? w_rem_beats : w_page_beats[4:0];

    wire [31:0] i_base_byte   = cfg_ifmap_base + ({14'b0, i_ealign} << 1);
    wire [31:0] i_cur_addr    = i_base_byte + ({26'b0, beat_cnt} << 2);
    wire [4:0]  i_rem_beats   = i_nbeats - beat_cnt;
    wire [10:0] i_page_beats  = 11'd1024 - {1'b0, i_cur_addr[11:2]};
    wire [4:0]  i_burst_beats = ({6'b0, i_rem_beats} <= i_page_beats) ? i_rem_beats : i_page_beats[4:0];

    // -----------------------------------------------------------------------
    // [ADDR-GEN PIPELINE] Registered-base burst signals
    // Derived from weight_addrs_r / ifmap_addrs_r / valid_vec_r.
    // Used exclusively in _AR states so the path from the register to
    // m_axi_arlen_reg is short. The _R states continue to use the
    // combinational versions (non-critical for timing).
    // -----------------------------------------------------------------------
    // --- Weight side ---
    wire [25:0] w_addr0_r      = weight_addrs_r[0 +: 26];
    wire        w_off_r        = w_addr0_r[0];
    wire [25:0] w_ealign_r     = {w_addr0_r[25:1], 1'b0};
    wire [4:0]  w_nbeats_r     = ({4'b0, w_off_r} + kk_local + 5'd1) >> 1;
    wire [31:0] w_base_byte_r  = reg_weight_base + ({6'b0, w_ealign_r} << 1);
    wire [31:0] w_cur_addr_r   = w_base_byte_r + ({26'b0, wc} << 2);
    wire [4:0]  w_rem_beats_r  = w_nbeats_r - wc;
    wire [10:0] w_page_beats_r = 11'd1024 - {1'b0, w_cur_addr_r[11:2]};
    wire [4:0]  w_burst_beats_r = ({6'b0, w_rem_beats_r} <= w_page_beats_r) ? w_rem_beats_r : w_page_beats_r[4:0];

    // --- Ifmap side ---
    wire        rvb0_r = valid_vec_r[row_t0 + 5'd0];
    wire        rvb1_r = (cfg_k > 4'd1) ? valid_vec_r[row_t0 + 5'd1] : 1'b0;
    wire        rvb2_r = (cfg_k > 4'd2) ? valid_vec_r[row_t0 + 5'd2] : 1'b0;
    wire        rvb3_r = (cfg_k > 4'd3) ? valid_vec_r[row_t0 + 5'd3] : 1'b0;
    wire        row_any_r      = rvb0_r | rvb1_r | rvb2_r | rvb3_r;
    wire [2:0]  row_dx_first_r = rvb0_r ? 3'd0 : rvb1_r ? 3'd1 : rvb2_r ? 3'd2 : 3'd3;
    wire [2:0]  row_dx_last_r  = rvb3_r ? 3'd3 : rvb2_r ? 3'd2 : rvb1_r ? 3'd1 : 3'd0;
    wire [2:0]  row_base_dx_r  = (cfg_mode == 1'b0) ? row_dx_first_r : row_dx_last_r;
    wire [2:0]  row_cnt_r      = {2'b0,rvb0_r} + {2'b0,rvb1_r} + {2'b0,rvb2_r} + {2'b0,rvb3_r};
    wire [4:0]  row_base_tap_r = row_t0 + {2'b0, row_base_dx_r};
    wire [17:0] row_base_addr_r = ifmap_addrs_r[row_base_tap_r * 18 +: 18];
    wire        i_off_r        = row_base_addr_r[0];
    wire [17:0] i_ealign_r     = {row_base_addr_r[17:1], 1'b0};
    wire [4:0]  i_nbeats_r     = ({4'b0, i_off_r} + {2'b0, row_cnt_r} + 5'd1) >> 1;
    wire [31:0] i_base_byte_r  = cfg_ifmap_base + ({14'b0, i_ealign_r} << 1);
    wire [31:0] i_cur_addr_r   = i_base_byte_r + ({26'b0, beat_cnt} << 2);
    wire [4:0]  i_rem_beats_r  = i_nbeats_r - beat_cnt;
    wire [10:0] i_page_beats_r = 11'd1024 - {1'b0, i_cur_addr_r[11:2]};
    wire [4:0]  i_burst_beats_r = ({6'b0, i_rem_beats_r} <= i_page_beats_r) ? i_rem_beats_r : i_page_beats_r[4:0];

    // ---------------------------------------------------
    // Weight cache write (one element/cycle, demuxed by run-position) + read
    // ---------------------------------------------------
    wire we_weight = (state == ST_FILL_W_R && m_axi_rvalid && w_pos_vld);

    always @(posedge clk) begin
        if (we_weight) begin
            case (w_pos[3:0])
                4'd0:  weight_cache_0 [fill_c_in_cnt] <= w_half;
                4'd1:  weight_cache_1 [fill_c_in_cnt] <= w_half;
                4'd2:  weight_cache_2 [fill_c_in_cnt] <= w_half;
                4'd3:  weight_cache_3 [fill_c_in_cnt] <= w_half;
                4'd4:  weight_cache_4 [fill_c_in_cnt] <= w_half;
                4'd5:  weight_cache_5 [fill_c_in_cnt] <= w_half;
                4'd6:  weight_cache_6 [fill_c_in_cnt] <= w_half;
                4'd7:  weight_cache_7 [fill_c_in_cnt] <= w_half;
                4'd8:  weight_cache_8 [fill_c_in_cnt] <= w_half;
                4'd9:  weight_cache_9 [fill_c_in_cnt] <= w_half;
                4'd10: weight_cache_10[fill_c_in_cnt] <= w_half;
                4'd11: weight_cache_11[fill_c_in_cnt] <= w_half;
                4'd12: weight_cache_12[fill_c_in_cnt] <= w_half;
                4'd13: weight_cache_13[fill_c_in_cnt] <= w_half;
                4'd14: weight_cache_14[fill_c_in_cnt] <= w_half;
                4'd15: weight_cache_15[fill_c_in_cnt] <= w_half;
                default: ;
            endcase
        end
        weight_tile[0]  <= weight_cache_0 [c_in_cnt];
        weight_tile[1]  <= weight_cache_1 [c_in_cnt];
        weight_tile[2]  <= weight_cache_2 [c_in_cnt];
        weight_tile[3]  <= weight_cache_3 [c_in_cnt];
        weight_tile[4]  <= weight_cache_4 [c_in_cnt];
        weight_tile[5]  <= weight_cache_5 [c_in_cnt];
        weight_tile[6]  <= weight_cache_6 [c_in_cnt];
        weight_tile[7]  <= weight_cache_7 [c_in_cnt];
        weight_tile[8]  <= weight_cache_8 [c_in_cnt];
        weight_tile[9]  <= weight_cache_9 [c_in_cnt];
        weight_tile[10] <= weight_cache_10[c_in_cnt];
        weight_tile[11] <= weight_cache_11[c_in_cnt];
        weight_tile[12] <= weight_cache_12[c_in_cnt];
        weight_tile[13] <= weight_cache_13[c_in_cnt];
        weight_tile[14] <= weight_cache_14[c_in_cnt];
        weight_tile[15] <= weight_cache_15[c_in_cnt];
    end

    // MAC Array
    wire [N_MAC*DW-1:0] weight_vec, ifmap_vec;
    genvar t;
    generate
        for (t = 0; t < N_MAC; t = t + 1) begin : MAC_MAP
            assign weight_vec[t*DW +: DW] = weight_tile[t];
            assign ifmap_vec [t*DW +: DW] = ifmap_tile[t];
        end
    endgenerate

    localparam MAC_PIPE = 2; // must match PIPE_STAGES in mac_array.v

    wire signed [ACC_W-1:0] row_sum;
    wire mac_out_valid;
    // in_valid: pulse high for exactly 1 cycle when weight_tile/ifmap_tile are stable
    wire mac_in_valid = (state == ST_COMPUTE_ISSUE);

    mac_array #(.DW(DW), .ACC_W(ACC_W), .K(4), .PIPE_STAGES(MAC_PIPE)) u_mac_array (
        .clk        (clk),
        .weight_vec (weight_vec),
        .ifmap_vec  (ifmap_vec),
        .valid_vec  (valid_vec_w),
        .in_valid   (mac_in_valid),
        .sum_out    (row_sum),
        .out_valid  (mac_out_valid)
    );

    // Bias logic
    wire is_first = (c_in_cnt == 12'd0);
    wire is_last  = (cfg_c_in > 0) && (c_in_cnt == cfg_c_in - 1);
    wire all_done = is_last &&
                   (w_cnt     == cfg_w_out - 1) &&
                   (h_cnt     == cfg_h_out - 1) &&
                   (c_out_cnt == cfg_c_out - 1);

    wire signed [ACC_W-1:0] bias_extended =
        {{(ACC_W-DW){bias_val[DW-1]}}, bias_val} <<< FRAC_W_LOCAL;
    wire signed [ACC_W-1:0] acc_next =
        is_first ? (bias_extended + row_sum) : (acc_reg + row_sum);

    // Activations
    // [FIX-A] Writeback must use acc_reg (the fully-accumulated sum latched at the
    // last ST_COMPUTE), NOT acc_next. In the writeback states acc_next = acc_reg +
    // row_sum, which would re-add the last input channel's contribution (double-count).
    wire signed [ACC_W-1:0] acc_rounded  = acc_reg + (48'sd1 << (FRAC_W_LOCAL-1));
    wire signed [ACC_W-FRAC_W_LOCAL-1:0] acc_shifted = acc_rounded[ACC_W-1:FRAC_W_LOCAL];

    localparam signed [DW-1:0] SAT_MAX = 16'sh7FFF;
    localparam signed [DW-1:0] SAT_MIN = 16'sh8000;
    wire sat_pos = ($signed(acc_shifted) >  $signed(40'sd32767));
    wire sat_neg = ($signed(acc_shifted) < -$signed(40'sd32768));
    wire signed [DW-1:0] acc_trunc = sat_pos ? SAT_MAX : (sat_neg ? SAT_MIN : acc_shifted[DW-1:0]);

    wire signed [DW-1:0] lrelu_out, relu_out, tanh_out_w;
    wire lrelu_valid, relu_valid, tanh_valid;

    leaky_relu #(.DW(DW),.FRAC_W(FRAC_W_LOCAL)) u_lrelu (
        .valid_in(1'b1), .x_in(acc_trunc),
        .valid_out(lrelu_valid), .y_out(lrelu_out));
    relu #(.DW(DW)) u_relu (
        .valid_in(1'b1), .x_in(acc_trunc),
        .valid_out(relu_valid),  .y_out(relu_out));
    tanh_act #(.DW(DW),.FRAC_W(FRAC_W_LOCAL)) u_tanh (
        .clk(clk), .valid_in(1'b1), .x_in(acc_trunc),
        .valid_out(tanh_valid), .y_out(tanh_out_w));

    wire signed [DW-1:0] final_pixel =
        (cur_act == 2'd0) ? lrelu_out :
        (cur_act == 2'd1) ? relu_out :
        (cur_act == 2'd2) ? tanh_out_w : acc_trunc;

    wire [19:0] hw_out            = {10'b0, cfg_h_out} * {10'b0, cfg_w_out};
    wire [31:0] ofmap_word_offset = {20'b0, c_out_cnt} * {12'b0, hw_out} + {22'b0, h_cnt} * {22'b0, cfg_w_out} + {22'b0, w_cnt};
    wire [31:0] ofmap_byte_addr   = cfg_ofmap_base + (ofmap_word_offset << 1);

    // ---------------------------------------------------
    // FSM
    // ---------------------------------------------------

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done  <= 0;
            c_out_cnt <= 0; h_cnt <= 0; w_cnt <= 0; c_in_cnt <= 0;
            fill_c_in_cnt <= 0; wc <= 0; tap_cnt <= 0;
            fr <= 0; beat_cnt <= 0; ph <= 0;
            m_axi_arvalid <= 0; m_axi_rready <= 0;
            m_axi_awvalid <= 0; m_axi_wvalid <= 0; m_axi_bready <= 0;
            acc_reg <= 0;
        end else begin
            done <= 0;
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        c_out_cnt <= 0; h_cnt <= 0; w_cnt <= 0; c_in_cnt <= 0;
                        fill_c_in_cnt <= 0; wc <= 0; tap_cnt <= 0; fr <= 0;
                        state <= ST_FILL_W_SETUP; // go through setup to register addr_gen outputs
                    end
                end

                // --- FILL WEIGHT CACHE (one full-width burst per c_in) ---
                // [HW-SAFE] Burst the whole kernel (kk contiguous elems) for this
                // c_in. 4-byte-aligned start, arsize=010; two phases per beat.
                // [PIPELINE] Use _r signals: path is weight_addrs_r (FF) -> burst_calc -> m_axi_arlen
                // weight_addrs_r was captured in ST_FILL_W_SETUP (previous cycle).
                ST_FILL_W_AR: begin
                    m_axi_araddr  <= w_cur_addr_r;
                    m_axi_arlen   <= {3'd0, (w_burst_beats_r - 5'd1)};
                    m_axi_arsize  <= 3'b010;  // 4 bytes/beat (full width)
                    m_axi_arburst <= 2'b01;   // INCR
                    m_axi_arvalid <= 1'b1;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;  // phase 0: read low half, do not advance
                        ph            <= 1'b0;
                        state         <= ST_FILL_W_R;
                    end
                end

                ST_FILL_W_R: begin
                    if (m_axi_rvalid) begin
                        // we_weight writes the in-run element (w_pos) this cycle
                        if (ph == 1'b0) begin
                            ph           <= 1'b1;
                            m_axi_rready <= 1'b1;  // advance the beat on the high phase
                        end else begin
                            ph           <= 1'b0;
                            m_axi_rready <= 1'b0;
                            if (m_axi_rlast) begin
                                if (wc == w_nbeats - 5'd1) begin
                                    wc <= 5'd0;
                                    if (fill_c_in_cnt == cfg_c_in - 1) begin
                                        fill_c_in_cnt <= 0;
                                        tap_cnt       <= 0;
                                        fr            <= 0;
                                        beat_cnt      <= 0;
                                        state         <= ST_FETCH_I_SETUP; // addr_gen -> ifmap_addrs_r
                                    end else begin
                                        fill_c_in_cnt <= fill_c_in_cnt + 1;
                                        state         <= ST_FILL_W_SETUP;  // addr_gen -> weight_addrs_r
                                    end
                                end else begin
                                    wc    <= wc + 5'd1;
                                    state <= ST_FILL_W_SETUP;  // addr_gen -> weight_addrs_r
                                end
                            end else begin
                                wc <= wc + 5'd1;   // next beat
                            end
                        end
                    end
                end

                // --- FETCH IFMAP (per-kernel-row burst; conv2d & transpose) ---
                // [STEP2/3] One burst per kernel row over the contiguous valid-tap
                // run; the row_* helpers pick base/slot stepping per cfg_mode.
                // [PIPELINE] row_any_r uses valid_vec_r (registered in SETUP), i_cur_addr_r
                // and i_burst_beats_r break the cfg_* -> ifmap_addrs -> m_axi_arlen path.
                // valid_vec_r contains the FULL K*K valid vector; indexing with current fr is safe.
                ST_FETCH_I_AR: begin
                    if (fr == cfg_k[2:0]) begin
                        state <= ST_COMPUTE_ISSUE;
                    end else if (!row_any_r) begin
                        // whole row has no valid tap (padding / out-of-range): skip
                        fr <= fr + 3'd1;
                    end else begin
                        m_axi_araddr  <= i_cur_addr_r;
                        m_axi_arlen   <= {3'd0, (i_burst_beats_r - 5'd1)};
                        m_axi_arsize  <= 3'b010;  // 4 bytes/beat (full width)
                        m_axi_arburst <= 2'b01;
                        m_axi_arvalid <= 1'b1;
                        if (m_axi_arvalid && m_axi_arready) begin
                            m_axi_arvalid <= 1'b0;
                            m_axi_rready  <= 1'b0;  // phase 0: read low half, do not advance
                            ph            <= 1'b0;
                            state         <= ST_FETCH_I_R;
                        end
                    end
                end

                ST_FETCH_I_R: begin
                    if (m_axi_rvalid) begin
                        if (i_pos_vld) ifmap_tile[i_burst_slot[3:0]] <= i_half;
                        if (ph == 1'b0) begin
                            ph           <= 1'b1;
                            m_axi_rready <= 1'b1;  // advance the beat on the high phase
                        end else begin
                            ph           <= 1'b0;
                            m_axi_rready <= 1'b0;
                            if (m_axi_rlast) begin
                                if (beat_cnt == i_nbeats - 5'd1) begin
                                    fr       <= fr + 3'd1;
                                    beat_cnt <= 0;
                                    state    <= ST_FETCH_I_SETUP; // re-register addr_gen for new fr
                                end else begin
                                    beat_cnt <= beat_cnt + 5'd1;
                                    state    <= ST_FETCH_I_SETUP; // re-register for new beat_cnt
                                end
                            end else begin
                                beat_cnt <= beat_cnt + 5'd1;
                            end
                        end
                    end
                end

                // --- COMPUTE ISSUE: present data to MAC pipeline for 1 cycle ---
                // mac_in_valid is driven combinationally (= state==ST_COMPUTE_ISSUE).
                // At this clock edge: prod_reg in mac_array captures the products.
                ST_COMPUTE_ISSUE: begin
                    state <= ST_COMPUTE_WAIT;
                end

                // --- COMPUTE WAIT: drain pipeline, latch result when valid ---
                // mac_out_valid asserts MAC_PIPE (=2) cycles after ISSUE.
                // sum_out (= sum_out_reg in mac_array) is stable when mac_out_valid=1.
                // Overhead vs original: +2 cycles per c_in — negligible vs DDR latency.
                ST_COMPUTE_WAIT: begin
                    if (mac_out_valid) begin
                        acc_reg <= acc_next; // acc_next uses the now-valid row_sum
                        if (is_last) begin
                            state <= ST_WB_AW;
                        end else begin
                            c_in_cnt <= c_in_cnt + 1;
                            tap_cnt  <= 0;
                            fr       <= 0;
                            beat_cnt <= 0;
                            state    <= ST_FETCH_I_SETUP; // register addr_gen for next c_in
                        end
                    end
                end

                // --- ADDR-GEN PIPELINE SETUP STATES ---
                // One cycle where addr_gen outputs (weight_addrs_w / ifmap_addrs_w /
                // valid_vec_w) propagate into their registered counterparts (_r).
                // The subsequent AR state uses the _r signals, making its path:
                //   weight_addrs_r (FF) -> burst_calc -> m_axi_arlen_reg
                // instead of:
                //   cfg_* -> addr_gen -> burst_calc -> m_axi_arlen_reg  (was 27 ns!)
                ST_FILL_W_SETUP: begin
                    state <= ST_FILL_W_AR;
                end

                ST_FETCH_I_SETUP: begin
                    state <= ST_FETCH_I_AR;
                end

                // --- WRITEBACK OFMAP (1 Beat) ---
                ST_WB_AW: begin
                    m_axi_awaddr  <= ofmap_byte_addr & 32'hFFFFFFFC; // Force 4-byte alignment
                    m_axi_awlen   <= 8'd0;    // 1 beat
                    m_axi_awsize  <= 3'b010;  // 4 bytes (FULL WIDTH burst to avoid narrow-burst bugs)
                    m_axi_awburst <= 2'b01;   // INCR
                    m_axi_awvalid <= 1'b1;

                    // Prepare wdata, wstrb, and wlast NOW so they are ready when wvalid goes high
                    if (ofmap_byte_addr & 2) begin
                        m_axi_wdata  <= {final_pixel, 16'b0};
                        m_axi_wstrb  <= 4'b1100;
                    end else begin
                        m_axi_wdata  <= {16'b0, final_pixel};
                        m_axi_wstrb  <= 4'b0011;
                    end
                    m_axi_wlast  <= 1'b1;

                    if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b1;
                        state         <= ST_WB_W;
                    end
                end

                ST_WB_W: begin
                    // Data is already driven. Just wait for wready.
                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        m_axi_wlast  <= 1'b0;
                        m_axi_wstrb  <= 4'b0000;
                        m_axi_bready <= 1'b1;
                        state        <= ST_WB_B;
                    end
                end

                ST_WB_B: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bready <= 1'b0;

                        c_in_cnt <= 0;
                        if (all_done) begin
                            done  <= 1;
                            state <= ST_IDLE;
                        end else begin
                            if (w_cnt == cfg_w_out - 1) begin
                                w_cnt <= 0;
                                if (h_cnt == cfg_h_out - 1) begin
                                    h_cnt     <= 0;
                                    c_out_cnt <= c_out_cnt + 1;
                                    fill_c_in_cnt <= 0; // Trigger fill cache for next c_out
                                end else begin
                                    h_cnt <= h_cnt + 1;
                                end
                            end else begin
                                w_cnt <= w_cnt + 1;
                            end

                            // If we just finished a c_out layer, refill the weight cache.
                            if (w_cnt == cfg_w_out - 1 && h_cnt == cfg_h_out - 1) begin
                                wc    <= 0;
                                state <= ST_FILL_W_SETUP;  // addr_gen -> weight_addrs_r
                            end else begin
                                tap_cnt  <= 0;
                                fr       <= 0;
                                beat_cnt <= 0;
                                state    <= ST_FETCH_I_SETUP; // addr_gen -> ifmap_addrs_r
                            end
                        end
                    end
                end
            endcase
        end
    end
endmodule
`default_nettype wire
