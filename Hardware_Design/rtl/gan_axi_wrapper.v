`timescale 1ns / 1ps
`default_nettype none

module gan_axi_wrapper #(
    parameter DW = 16,
    parameter FRAC_W = 8,
    parameter ACC_W = 48
)(
    // CLOCK & RESET
    input  wire        aclk,
    input  wire        aresetn,

    // AXI4-LITE SLAVE (Control & Addresses)
    input  wire [5:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [5:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,

    // AXI4 MASTER (DDR Access)
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
    
    // AXI Standard Sideband Signals
    output wire [3:0]  m_axi_awcache,
    output wire [2:0]  m_axi_awprot,
    output wire [3:0]  m_axi_awqos,
    output wire [3:0]  m_axi_arcache,
    output wire [2:0]  m_axi_arprot,
    output wire [3:0]  m_axi_arqos
);

    // =========================================================================
    // AXI4-LITE LOGIC
    // =========================================================================
    reg awready, wready, bvalid, arready, rvalid;
    reg [31:0] rdata;
    
    assign s_axi_awready = awready;
    assign s_axi_wready  = wready;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = bvalid;
    assign s_axi_arready = arready;
    assign s_axi_rdata   = rdata;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = rvalid;

    // Registers
    reg [31:0] slv_reg0; // 0x00: Control (Bit 0: Start)
    wire [31:0] slv_reg1; // 0x04: Status  (Bit 0: Done)
    reg [31:0] slv_reg2; // 0x08: Img Base Address
    reg [31:0] slv_reg3; // 0x0C: FMAP A Base
    reg [31:0] slv_reg4; // 0x10: FMAP B Base
    reg [31:0] slv_reg5; // 0x14: Out Base Address
    reg [31:0] slv_reg6; // 0x18: Bias Write Config
    reg [31:0] slv_reg7; // 0x1C: Bias Data
    reg [31:0] slv_reg8; // 0x20: Weight Base Address

    wire slv_reg_wren = wready && s_axi_wvalid && awready && s_axi_awvalid;
    reg  start_pulse;
    wire done_sig;
    reg done_latch;
    always @(posedge aclk) begin
        if (!aresetn) done_latch <= 0;
        else if (start_pulse) done_latch <= 0;
        else if (done_sig) done_latch <= 1;
    end
    
    wire [3:0] dbg_layer_idx;
    wire [3:0] dbg_state;
    assign slv_reg1 = {20'b0, dbg_state, dbg_layer_idx, 3'b0, done_latch};

    reg bias_wr_en;

    always @(posedge aclk) begin
        if (!aresetn) begin
            awready <= 0; wready <= 0; bvalid <= 0;
            slv_reg0 <= 0; slv_reg2 <= 0; slv_reg3 <= 0;
            slv_reg4 <= 0; slv_reg5 <= 0; slv_reg6 <= 0; slv_reg7 <= 0; slv_reg8 <= 0;
            start_pulse <= 0; bias_wr_en <= 0;
        end else begin
            start_pulse <= 0;
            bias_wr_en <= 0;
            
            if (~awready && s_axi_awvalid && s_axi_wvalid) awready <= 1;
            else awready <= 0;
            
            if (~wready && s_axi_wvalid && s_axi_awvalid) wready <= 1;
            else wready <= 0;
            
            if (slv_reg_wren) begin
                case (s_axi_awaddr[5:2])
                    4'h0: begin slv_reg0 <= s_axi_wdata; if(s_axi_wdata[0]) start_pulse <= 1; end
                    4'h2: slv_reg2 <= s_axi_wdata;
                    4'h3: slv_reg3 <= s_axi_wdata;
                    4'h4: slv_reg4 <= s_axi_wdata;
                    4'h5: slv_reg5 <= s_axi_wdata;
                    4'h6: slv_reg6 <= s_axi_wdata;
                    4'h7: begin slv_reg7 <= s_axi_wdata; bias_wr_en <= 1; end
                    4'h8: slv_reg8 <= s_axi_wdata;
                endcase
            end
            
            if (s_axi_awready && s_axi_awvalid && s_axi_wready && s_axi_wvalid && ~bvalid) begin
                bvalid <= 1;
            end else if (s_axi_bready && bvalid) begin
                bvalid <= 0;
            end
        end
    end

    wire slv_reg_rden = arready && s_axi_arvalid && ~rvalid;
    always @(posedge aclk) begin
        if (!aresetn) begin
            arready <= 0; rvalid <= 0; rdata <= 0;
        end else begin
            if (~arready && s_axi_arvalid) arready <= 1;
            else arready <= 0;
            
            if (slv_reg_rden) begin
                rvalid <= 1;
                case (s_axi_araddr[5:2])
                    4'h0: rdata <= slv_reg0;
                    4'h1: rdata <= slv_reg1;
                    4'h2: rdata <= slv_reg2;
                    4'h3: rdata <= slv_reg3;
                    4'h4: rdata <= slv_reg4;
                    4'h5: rdata <= slv_reg5;
                    4'h6: rdata <= slv_reg6;
                    4'h7: rdata <= slv_reg7;
                    4'h8: rdata <= slv_reg8;
                    default: rdata <= 0;
                endcase
            end else if (s_axi_rready && rvalid) begin
                rvalid <= 0;
            end
        end
    end

    // =========================================================================
    // Generator Top Instantiation
    // =========================================================================
    generator_top #(.DW(DW), .FRAC_W(FRAC_W), .ACC_W(ACC_W)) u_gen (
        .clk(aclk),
        .rst_n(aresetn),
        .start(start_pulse),
        .done(done_sig),

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

        .reg_img_base(slv_reg2),
        .reg_fmap_a_base(slv_reg3),
        .reg_fmap_b_base(slv_reg4),
        .reg_out_base(slv_reg5),
        .reg_weight_base(slv_reg8),

        .dbg_layer_idx(dbg_layer_idx),
        .dbg_state(dbg_state),

        .wr_en(bias_wr_en),
        .wr_layer(slv_reg6[27:24]), // Same mapping as old
        .wr_type(slv_reg6[19:18]),
        .wr_addr(slv_reg6[17:0]),
        .wr_data(slv_reg7[DW-1:0])
    );

    // AXI sideband signal assignments for Zynq HP0 / AXI SmartConnect compatibility
    assign m_axi_arcache = 4'b0011; // Normal Non-cacheable Bufferable
    assign m_axi_awcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_awqos   = 4'b0000;

endmodule
`default_nettype wire
