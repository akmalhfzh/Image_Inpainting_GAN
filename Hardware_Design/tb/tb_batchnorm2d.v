// =============================================================================
// tb_batchnorm2d.v  —  Testbench untuk batchnorm2d.v
//
// Flow:
//   1. gen_vectors_bn.py  → vectors_bn/scale.hex, offset.hex, input.hex, expected.hex
//   2. TB ini load hex, stream ke DUT, dump output
//   3. check_bn.py bandingkan hasil
//
// Layer yang ditest: BN setelah E1 conv2d
//   C=64, H=32, W=32  (output E1: [64, 32, 32])
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_batchnorm2d;

    localparam DW     = 16;
    localparam FRAC_W = 8;
    localparam C      = 4;    // Set small value for rapid verification
    localparam H      = 4;
    localparam W      = 4;

    localparam TOTAL  = C * H * W;   // total pixel yang distream

    // ── Clock ────────────────────────────────────────────────────────────────
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n = 0;

    // ── DUT wires ─────────────────────────────────────────────────────────────
    reg                   wr_en   = 0;
    reg                   wr_sel  = 0;
    reg [$clog2(C)-1:0]  wr_addr = 0;
    reg signed [DW-1:0]  wr_data = 0;

    reg                   x_valid = 0;
    reg  signed [DW-1:0] x_in    = 0;
    wire                  y_valid;
    wire signed [DW-1:0] y_out;

    batchnorm2d #(
        .DW(DW), .FRAC_W(FRAC_W),
        .C(C), .H(H), .W(W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_sel(wr_sel), .wr_addr(wr_addr), .wr_data(wr_data),
        .x_valid(x_valid), .x_in(x_in),
        .y_valid(y_valid), .y_out(y_out)
    );

    // ── Staging ───────────────────────────────────────────────────────────────
    reg signed [DW-1:0] scale_stage  [0:C-1];
    reg signed [DW-1:0] offset_stage [0:C-1];
    reg signed [DW-1:0] input_stage  [0:TOTAL-1];

    // ── Load helper ───────────────────────────────────────────────────────────
    task load_param;
        input            sel;
        input [31:0]     ch;
        input signed [DW-1:0] data;
        begin
            @(posedge clk); #1;
            wr_en   = 1; wr_sel = sel;
            wr_addr = ch[$clog2(C)-1:0];
            wr_data = data;
            @(posedge clk); #1;
            wr_en   = 0;
        end
    endtask

    localparam PIPE_LAT = 2;   // pipeline latency batchnorm2d = 2 clock

    // ── VCD dump — harus di initial tersendiri, jalan paling awal ────────────
    initial begin
        $dumpfile("tb_batchnorm2d.vcd");
        $dumpvars(0, tb_batchnorm2d);   // dump semua sinyal di module ini
    end

    // ── Debug signals — visible di waveform sebagai named group ──────────────
    // Menunjukkan state TB: phase, pixel index, channel yang sedang diproses
    reg [1:0]  dbg_phase;        // 0=reset 1=load_param 2=streaming 3=drain
    reg [15:0] dbg_pixel_in;     // index pixel input yang sedang dikirim
    reg [15:0] dbg_pixel_out;    // index pixel output yang sudah dikumpulkan
    reg [7:0]  dbg_channel;      // channel yang sedang aktif (dari ch_cnt DUT)

    // Tap internal DUT signals untuk waveform
    // (channel counter dan pixel counter dari batchnorm2d.v)
    wire [$clog2(C)-1:0]  dbg_dut_ch_cnt  = dut.ch_cnt;
    wire                   dbg_dut_s1valid = dut.stage1_valid;
    wire signed [DW-1:0]   dbg_dut_s1mul   = dut.stage1_mul;
    wire signed [DW-1:0]   dbg_dut_s1off   = dut.stage1_offset;
    // scale dan offset yang sedang aktif untuk channel saat ini
    wire signed [DW-1:0]   dbg_scale_active = dut.scale_mem[dut.ch_cnt];
    wire signed [DW-1:0]   dbg_offset_active= dut.offset_mem[dut.ch_cnt];

    integer i, j, fd, out_cnt, timeout_cnt;
    integer out_buf [0:TOTAL-1];

    // ── Inisialisasi out_buf ke 0 (hindari nilai 'x') ─────────────────────────
    integer init_idx;
    initial begin
        for (init_idx = 0; init_idx < TOTAL; init_idx = init_idx + 1)
            out_buf[init_idx] = 0;
    end

    initial begin
        // Init debug signals
        dbg_phase     = 0;
        dbg_pixel_in  = 0;
        dbg_pixel_out = 0;
        dbg_channel   = 0;
    end

    initial begin
        $display("=============================================================");
        $display("  tb_batchnorm2d  C=%0d H=%0d W=%0d  TOTAL=%0d pixels", C, H, W, TOTAL);
        $display("  Pipeline latency = %0d clocks", PIPE_LAT);
        $display("=============================================================");

        // Load hex vectors
        $readmemh("vectors_bn/scale.hex",  scale_stage);
        $readmemh("vectors_bn/offset.hex", offset_stage);
        $readmemh("vectors_bn/input.hex",  input_stage);
        $display("[TB] Vectors loaded.");

        // Reset
        dbg_phase = 0;   // phase: RESET
        rst_n = 0; repeat(4) @(posedge clk);
        rst_n = 1; repeat(2) @(posedge clk);
        $display("[TB] Reset done.");

        // Load scale & offset per channel
        dbg_phase = 1;   // phase: LOAD_PARAM
        for (i = 0; i < C; i = i + 1) begin
            dbg_channel = i;
            load_param(1'b0, i, scale_stage[i]);
            load_param(1'b1, i, offset_stage[i]);
        end
        $display("[TB] Parameters loaded (%0d channels).", C);

        // ── Stream input: drive TOTAL pixels satu per clock ──────────────────
        $display("[TB] Streaming %0d pixels...", TOTAL);
        dbg_phase = 2;   // phase: STREAMING
        out_cnt = 0;

        for (i = 0; i < TOTAL; i = i + 1) begin
            @(posedge clk); #1;
            x_valid      = 1;
            x_in         = input_stage[i];
            dbg_pixel_in = i;
            // Kumpulkan output kalau sudah valid (mulai dari pixel ke-3 dst)
            if (y_valid) begin
                out_buf[out_cnt] = $signed(y_out);
                dbg_pixel_out    = out_cnt;
                out_cnt = out_cnt + 1;
            end
        end

        // ── Flush pipeline: matikan input, tunggu sisa PIPE_LAT output ──────
        dbg_phase = 3;   // phase: DRAIN
        @(posedge clk); #1;
        x_valid = 0;
        x_in    = 0;

        // Drain: tunggu sampai semua TOTAL output keluar atau timeout
        timeout_cnt = 0;
        while (out_cnt < TOTAL && timeout_cnt < TOTAL + 10) begin
            @(posedge clk); #1;
            if (y_valid) begin
                out_buf[out_cnt] = $signed(y_out);
                dbg_pixel_out    = out_cnt;
                out_cnt = out_cnt + 1;
            end
            timeout_cnt = timeout_cnt + 1;
        end

        if (out_cnt != TOTAL)
            $display("[TB] WARNING: hanya %0d dari %0d output terkumpul!", out_cnt, TOTAL);
        else
            $display("[TB] All %0d outputs collected.", out_cnt);

        // ── Tulis hasil ───────────────────────────────────────────────────────
        fd = $fopen("vectors_bn/output_rtl.hex", "w");
        if (fd == 0) begin
            $display("[TB] ERROR: cannot open vectors_bn/output_rtl.hex"); $finish;
        end
        for (j = 0; j < TOTAL; j = j + 1)
            $fwrite(fd, "%04x\n", out_buf[j] & 32'h0000FFFF);
        $fclose(fd);

        $display("[TB] Output written → vectors_bn/output_rtl.hex");
        $display("[TB] Run: python3 check_bn.py");
        $display("=============================================================");
        $finish;
    end

endmodule
`default_nettype wire
