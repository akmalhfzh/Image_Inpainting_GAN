// =============================================================================
// generator_top (AXI DDR-Backed)
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module generator_top #(
    parameter DW     = 16,
    parameter FRAC_W = 8,
    parameter ACC_W  = 48
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // AXI4 Master Interface
    output wire [31:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    output wire        m_axi_arvalid,
    input  wire        m_axi_arready,
    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rlast,
    input  wire        m_axi_rvalid,
    output wire        m_axi_rready,

    output wire [31:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output wire        m_axi_awvalid,
    input  wire        m_axi_awready,
    output wire [31:0] m_axi_wdata,
    output wire [3:0]  m_axi_wstrb,
    output wire        m_axi_wlast,
    output wire        m_axi_wvalid,
    input  wire        m_axi_wready,
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output wire        m_axi_bready,

    // DDR Base addresses from AXI-Lite
    input  wire [31:0] reg_img_base,
    input  wire [31:0] reg_fmap_a_base,
    input  wire [31:0] reg_fmap_b_base,
    input  wire [31:0] reg_out_base,
    input  wire [31:0] reg_weight_base,

    // Bias write interface (from AXI-Lite wrapper)
    input  wire                 wr_en,
    input  wire [3:0]           wr_layer,
    input  wire [1:0]           wr_type,
    input  wire [17:0]          wr_addr,
    input  wire signed [DW-1:0] wr_data,
    
    // Debug
    output wire                 dbg_layer_done,
    output wire [3:0]           dbg_layer_idx,
    output wire [3:0]           dbg_state
);
    localparam ST_IDLE       = 4'd0;
    localparam ST_CONV_START = 4'd1;
    localparam ST_CONV_WAIT  = 4'd2;
    localparam ST_NEXT_LAYER = 4'd3;
    localparam ST_DONE       = 4'd4;

    reg [3:0] state, layer_idx;
    reg ping_pong;

    assign dbg_layer_idx  = layer_idx;
    assign dbg_layer_done = (state == ST_NEXT_LAYER) || (state == ST_DONE);
    assign dbg_state      = state;

    reg [11:0] cur_c_out;
    reg [9:0]  cur_h_out, cur_w_out;
    reg [1:0]  cur_act;

    reg [9:0]  cfg_h_in, cfg_w_in, cfg_h_out, cfg_w_out;
    reg [11:0] cfg_c_in, cfg_c_out;
    reg [3:0]  cfg_k, cfg_stride, cfg_pad;
    reg        cfg_mode;
    reg [25:0] cfg_weight_base;
    reg [11:0] cfg_bias_base;

    reg signed [DW-1:0] bias_mem [0:4095];

    // DDR addresses for current layer
    wire [31:0] cur_ifmap_base = 
        (layer_idx == 0) ? reg_img_base :
        (ping_pong)      ? reg_fmap_b_base :
                           reg_fmap_a_base;

    wire [31:0] cur_ofmap_base = 
        (layer_idx == 11) ? reg_out_base :
        (ping_pong)       ? reg_fmap_a_base :
                            reg_fmap_b_base;

    reg  conv_start_r;
    wire conv_done_w;

    wire [11:0] current_c_out;
    wire signed [DW-1:0] current_bias = bias_mem[cfg_bias_base + current_c_out];

    conv_unified #(.DW(DW),.ACC_W(ACC_W),.FRAC_W(FRAC_W)) u_CONV_CORE (
        .clk(clk), .rst_n(rst_n), .start(conv_start_r), .done(conv_done_w),
        
        // AXI4 Master
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),

        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),

        // Config
        .cfg_h_in(cfg_h_in), .cfg_w_in(cfg_w_in),
        .cfg_h_out(cfg_h_out), .cfg_w_out(cfg_w_out),
        .cfg_c_in(cfg_c_in), .cfg_c_out(cfg_c_out),
        .cfg_k(cfg_k), .cfg_stride(cfg_stride), .cfg_pad(cfg_pad),
        .cfg_mode(cfg_mode), .cfg_weight_base(cfg_weight_base),
        .cfg_ifmap_base(cur_ifmap_base), .cfg_ofmap_base(cur_ofmap_base),
        .reg_weight_base(reg_weight_base),
        .cur_act(cur_act),
        
        .bias_val(current_bias),
        .current_c_out(current_c_out)
    );

    reg [11:0] target_bias_base;

    always @(*) begin
        case (wr_layer)
            4'd0:  target_bias_base=12'd0;
            4'd1:  target_bias_base=12'd32;
            4'd2:  target_bias_base=12'd96;
            4'd3:  target_bias_base=12'd224;
            4'd4:  target_bias_base=12'd480;
            4'd5:  target_bias_base=12'd992;
            4'd6:  target_bias_base=12'd1504;
            4'd7:  target_bias_base=12'd1760;
            4'd8:  target_bias_base=12'd1888;
            4'd9:  target_bias_base=12'd1952;
            4'd10: target_bias_base=12'd1984;
            4'd11: target_bias_base=12'd2016;
            default: target_bias_base=12'd0;
        endcase
    end

    always @(posedge clk) begin
        if (wr_en && wr_type == 2'd3) begin
            bias_mem[target_bias_base + wr_addr[11:0]] <= wr_data;
        end
    end

    task set_layer_info;
        input [3:0] idx;
        begin
            case (idx)
                4'd0:  begin cfg_c_in=12'd3;   cfg_c_out=12'd32;  cfg_h_in=10'd64; cfg_w_in=10'd64; cfg_h_out=10'd32; cfg_w_out=10'd32; cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd0;       cur_act=2'd0; cfg_bias_base=12'd0;    end
                4'd1:  begin cfg_c_in=12'd32;  cfg_c_out=12'd64;  cfg_h_in=10'd32; cfg_w_in=10'd32; cfg_h_out=10'd16; cfg_w_out=10'd16; cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd3072;    cur_act=2'd0; cfg_bias_base=12'd32;   end
                4'd2:  begin cfg_c_in=12'd64;  cfg_c_out=12'd128; cfg_h_in=10'd16; cfg_w_in=10'd16; cfg_h_out=10'd8;  cfg_w_out=10'd8;  cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd68608;   cur_act=2'd0; cfg_bias_base=12'd96;   end
                4'd3:  begin cfg_c_in=12'd128; cfg_c_out=12'd256; cfg_h_in=10'd8;  cfg_w_in=10'd8;  cfg_h_out=10'd4;  cfg_w_out=10'd4;  cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd199680;  cur_act=2'd0; cfg_bias_base=12'd224;  end
                4'd4:  begin cfg_c_in=12'd256; cfg_c_out=12'd512; cfg_h_in=10'd4;  cfg_w_in=10'd4;  cfg_h_out=10'd2;  cfg_w_out=10'd2;  cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd723968;  cur_act=2'd0; cfg_bias_base=12'd480;  end
                4'd5:  begin cfg_c_in=12'd512; cfg_c_out=12'd512; cfg_h_in=10'd2;  cfg_w_in=10'd2;  cfg_h_out=10'd2;  cfg_w_out=10'd2;  cfg_k=4'd1; cfg_stride=4'd1; cfg_pad=4'd0; cfg_mode=1'b0; cfg_weight_base=26'd2821120; cur_act=2'd3; cfg_bias_base=12'd992;  end
                4'd6:  begin cfg_c_in=12'd512; cfg_c_out=12'd256; cfg_h_in=10'd2;  cfg_w_in=10'd2;  cfg_h_out=10'd4;  cfg_w_out=10'd4;  cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b1; cfg_weight_base=26'd3083264; cur_act=2'd1; cfg_bias_base=12'd1504; end
                4'd7:  begin cfg_c_in=12'd256; cfg_c_out=12'd128; cfg_h_in=10'd4;  cfg_w_in=10'd4;  cfg_h_out=10'd8;  cfg_w_out=10'd8;  cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b1; cfg_weight_base=26'd5180672; cur_act=2'd1; cfg_bias_base=12'd1760; end
                4'd8:  begin cfg_c_in=12'd128; cfg_c_out=12'd64;  cfg_h_in=10'd8;  cfg_w_in=10'd8;  cfg_h_out=10'd16; cfg_w_out=10'd16; cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b1; cfg_weight_base=26'd5705088; cur_act=2'd1; cfg_bias_base=12'd1888; end
                4'd9:  begin cfg_c_in=12'd64;  cfg_c_out=12'd32;  cfg_h_in=10'd16; cfg_w_in=10'd16; cfg_h_out=10'd32; cfg_w_out=10'd32; cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b1; cfg_weight_base=26'd5836288; cur_act=2'd1; cfg_bias_base=12'd1952; end
                4'd10: begin cfg_c_in=12'd32;  cfg_c_out=12'd32;  cfg_h_in=10'd32; cfg_w_in=10'd32; cfg_h_out=10'd64; cfg_w_out=10'd64; cfg_k=4'd4; cfg_stride=4'd2; cfg_pad=4'd1; cfg_mode=1'b1; cfg_weight_base=26'd5869088; cur_act=2'd1; cfg_bias_base=12'd1984; end
                4'd11: begin cfg_c_in=12'd32;  cfg_c_out=12'd3;   cfg_h_in=10'd64; cfg_w_in=10'd64; cfg_h_out=10'd64; cfg_w_out=10'd64; cfg_k=4'd3; cfg_stride=4'd1; cfg_pad=4'd1; cfg_mode=1'b0; cfg_weight_base=26'd5901888; cur_act=2'd2; cfg_bias_base=12'd2016; end
                default: begin cfg_c_in=12'd0; cfg_c_out=12'd0; cfg_h_in=10'd0; cfg_w_in=10'd0; cfg_h_out=10'd0; cfg_w_out=10'd0; cfg_k=4'd0; cfg_stride=4'd0; cfg_pad=4'd0; cfg_mode=1'b0; cfg_weight_base=26'd0; cur_act=2'd3; cfg_bias_base=12'd0; end
            endcase
            cur_c_out = cfg_c_out;
            cur_h_out = cfg_h_out;
            cur_w_out = cfg_w_out;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 0; layer_idx <= 0;
            ping_pong <= 0; conv_start_r <= 0;
        end else begin
            done         <= 0;
            conv_start_r <= 0;

            case (state)
            ST_IDLE: begin
                if (start) begin
                    layer_idx <= 4'd0; ping_pong <= 0;
                    set_layer_info(4'd0);
                    state  <= ST_CONV_START;
                    $display("[TOP] Inference start (AXI mode)");
                end
            end

            ST_CONV_START: begin
                conv_start_r <= 1'b1;
                state        <= ST_CONV_WAIT;
            end

            ST_CONV_WAIT: begin
                if (conv_done_w) begin
                    state <= ST_NEXT_LAYER;
                end
            end

            ST_NEXT_LAYER: begin
                if (layer_idx == 4'd11) begin
                    state  <= ST_DONE;
                end else begin
                    ping_pong <= ~ping_pong;
                    layer_idx <= layer_idx + 1;
                    set_layer_info(layer_idx + 1);
                    state  <= ST_CONV_START;
                    $display("[TOP] Next layer %0d", layer_idx + 1);
                end
            end

            ST_DONE: begin
                done  <= 1'b1;
                state <= ST_IDLE;
                $display("[TOP] Inference complete.");
            end
            endcase
        end
    end

endmodule
`default_nettype wire
