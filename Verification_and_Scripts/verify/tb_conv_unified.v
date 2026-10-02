// =============================================================================
// tb_conv_unified.v  —  Testbench untuk conv_unified.v
//
// Test dua mode:
//   TEST 0: MODE=0 (Conv2d)       — sama seperti tb_conv2dcore_par
//   TEST 1: MODE=1 (ConvTranspose) — sama seperti tb_convtranspose2dcore_par
//
// Karena MODE adalah parameter (bukan runtime), kita instantiate DUT dua kali
// dengan nama berbeda dan tes secara berurutan dalam satu file.
//
// Untuk test Conv2d    : localparam TEST = 0
// Untuk test ConvTranspose: localparam TEST = 1
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_conv_unified;

    localparam DW      = 16;
    localparam ACC_W   = 40;
    localparam K       = 4;
    localparam STRIDE  = 2;
    localparam PAD     = 1;
    localparam DILA    = 1;
    localparam USE_BIAS= 0;

    // ── Select test mode - adjust 0/1 and configure C_IN/C_OUT/H/W ──────
    localparam TEST_MODE = 0;   // 0=Conv2d, 1=ConvTranspose2d

    // Conv2d test params (E1: 3→64, 64×64→32×32)
    localparam C_IN_CONV  = 3;
    localparam C_OUT_CONV = 64;
    localparam H_IN_CONV  = 64;
    localparam W_IN_CONV  = 64;

    // ConvTranspose test params (D4: 128→64, 16×16→32×32)
    localparam C_IN_CONVT  = 128;
    localparam C_OUT_CONVT = 64;
    localparam H_IN_CONVT  = 16;
    localparam W_IN_CONVT  = 16;

    // Select aktif
    localparam C_IN  = (TEST_MODE==0) ? C_IN_CONV  : C_IN_CONVT;
    localparam C_OUT = (TEST_MODE==0) ? C_OUT_CONV : C_OUT_CONVT;
    localparam H_IN  = (TEST_MODE==0) ? H_IN_CONV  : H_IN_CONVT;
    localparam W_IN  = (TEST_MODE==0) ? W_IN_CONV  : W_IN_CONVT;

    localparam H_OUT = (TEST_MODE==0)
        ? (H_IN + 2*PAD - DILA*(K-1) - 1) / STRIDE + 1
        : (H_IN - 1)*STRIDE - 2*PAD + K;
    localparam W_OUT = (TEST_MODE==0)
        ? (W_IN + 2*PAD - DILA*(K-1) - 1) / STRIDE + 1
        : (W_IN - 1)*STRIDE - 2*PAD + K;

    localparam IFMAP_SZ  = C_IN  * H_IN  * W_IN;
    localparam WEIGHT_SZ = C_OUT * C_IN  * K * K;
    localparam OFMAP_SZ  = C_OUT * H_OUT * W_OUT;
    localparam THEO_PAR  = C_OUT * H_OUT * W_OUT * C_IN;
    localparam THEO_SEQ  = THEO_PAR * K * K;

    // ── Clock ─────────────────────────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    // ── DUT ───────────────────────────────────────────────────────────────────
    reg                      start   = 0;
    wire                     done;
    reg                      wr_en   = 0;
    reg  [1:0]               wr_sel  = 0;
    reg  [17:0]              wr_addr = 0;
    reg  signed [DW-1:0]    wr_data = 0;
    reg  [17:0]              rd_addr = 0;
    wire signed [ACC_W-1:0]  rd_data;

    conv_unified #(
        .DW(DW), .ACC_W(ACC_W),
        .C_IN(C_IN), .C_OUT(C_OUT),
        .H_IN(H_IN), .W_IN(W_IN),
        .K(K), .STRIDE(STRIDE), .PAD(PAD),
        .DILA(DILA), .USE_BIAS(USE_BIAS),
        .MODE(TEST_MODE)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .wr_en(wr_en), .wr_sel(wr_sel),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_addr(rd_addr), .rd_data(rd_data)
    );

    // ── VCD ───────────────────────────────────────────────────────────────────
    initial begin
        $dumpfile("conv_unified.vcd");
        $dumpvars(0, tb_conv_unified);
    end

    // ── Staging arrays ────────────────────────────────────────────────────────
    reg [DW-1:0] ifmap_stage  [0:IFMAP_SZ-1];
    reg [DW-1:0] weight_stage [0:WEIGHT_SZ-1];

    // ── Write task ────────────────────────────────────────────────────────────
    task load_mem;
        input [1:0]           sel;
        input [17:0]          addr;
        input signed [DW-1:0] data;
        begin
            @(posedge clk); #1;
            wr_en=1; wr_sel=sel; wr_addr=addr; wr_data=data;
            @(posedge clk); #1;
            wr_en=0;
        end
    endtask

    integer i, fd, timeout_cnt;
    string vec_dir;

    initial begin
        vec_dir = (TEST_MODE==0) ? "vectors" : "vectors_t";

        $display("=================================================================");
        $display("  tb_conv_unified  MODE=%0d (%0s)",
                 TEST_MODE, (TEST_MODE==0)?"Conv2d":"ConvTranspose2d");
        $display("  C_IN=%0d C_OUT=%0d  H_IN=%0d W_IN=%0d",C_IN,C_OUT,H_IN,W_IN);
        $display("  H_OUT=%0d  W_OUT=%0d  K=%0d S=%0d P=%0d",H_OUT,W_OUT,K,STRIDE,PAD);
        $display("  IFMAP=%0d  WEIGHT=%0d  OFMAP=%0d",IFMAP_SZ,WEIGHT_SZ,OFMAP_SZ);
        $display("  Theoretical parallel: %0d cycles  (%0dx vs seq)",THEO_PAR,K*K);
        $display("=================================================================");

        // Tentukan folder vektor berdasarkan mode
        if (TEST_MODE == 0) begin
            $readmemh("vectors/ifmap.hex",  ifmap_stage);
            $readmemh("vectors/weight.hex", weight_stage);
        end else begin
            $readmemh("vectors_t/ifmap.hex",  ifmap_stage);
            $readmemh("vectors_t/weight.hex", weight_stage);
        end
        $display("[TB] Hex files loaded.");

        rst_n=0; repeat(4) @(posedge clk);
        rst_n=1; repeat(2) @(posedge clk);

        $display("[TB] Loading ifmap  (%0d words)...", IFMAP_SZ);
        for (i=0; i<IFMAP_SZ; i=i+1)
            load_mem(2'd0, i[17:0], $signed(ifmap_stage[i]));

        $display("[TB] Loading weights (%0d words)...", WEIGHT_SZ);
        for (i=0; i<WEIGHT_SZ; i=i+1)
            load_mem(2'd1, i[17:0], $signed(weight_stage[i]));

        $display("[TB] Starting compute...");

        fork
            begin start=1; @(posedge clk); #1; start=0; end
            begin
                timeout_cnt=0;
                while (!done && timeout_cnt < THEO_PAR+1000) begin
                    @(posedge clk); timeout_cnt=timeout_cnt+1;
                end
            end
        join

        if (!done) begin
            $display("[TB] ERROR: TIMEOUT!"); $finish;
        end

        $display("[TB] Done! cycles=%0d  speedup=%0dx",
                 timeout_cnt, THEO_SEQ/timeout_cnt);

        @(posedge clk); #1;

        // Dump output ke folder yang sesuai
        if (TEST_MODE == 0)
            fd = $fopen("vectors/ofmap_rtl.hex","w");
        else
            fd = $fopen("vectors_t/ofmap_rtl.hex","w");

        if (fd==0) begin $display("[TB] ERROR: open file"); $finish; end

        for (i=0; i<OFMAP_SZ; i=i+1) begin
            rd_addr=i[17:0]; #1;
            $fwrite(fd, "%010x\n", rd_data[ACC_W-1:0]);
        end
        $fclose(fd);

        $display("[TB] Output written.");
        $display("[TB] Run: python3 check_output.py  (atau check_output_transpose.py)");
        $display("=================================================================");
        $finish;
    end

endmodule
`default_nettype wire
