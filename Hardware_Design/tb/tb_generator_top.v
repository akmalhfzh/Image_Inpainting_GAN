// =============================================================================
// tb_generator_top.v  —  Testbench untuk generator_top.v (LITE Model Fixed + Bias)
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_generator_top;
    localparam DW    = 16;
    localparam FRAC_W= 8;
    localparam ACC_W = 48; // [FIX] Mengikuti ACC_W terbaru di conv_unified
    localparam IMG_SZ = 3 * 64 * 64; // 12,288
    localparam OUT_SZ = 3 * 64 * 64;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;

    // ── DUT ───────────────────────────────────────────────────────────────────
    reg                   start       = 0;
    wire                  done;
    reg                   img_wr_en   = 0;
    reg  [13:0]           img_wr_addr = 0;
    reg  signed [DW-1:0]  img_wr_data = 0;
    
    reg                   wr_en       = 0;
    reg  [3:0]            wr_layer    = 0;
    reg  [1:0]            wr_type     = 0;
    reg  [17:0]           wr_addr     = 0;
    reg  signed [DW-1:0]  wr_data     = 0;
    
    reg  [13:0]           out_rd_addr = 0;
    wire signed [DW-1:0]  out_rd_data;

    generator_top #(.DW(DW),.FRAC_W(FRAC_W),.ACC_W(ACC_W)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),.done(done),
        .img_wr_en(img_wr_en),.img_wr_addr(img_wr_addr),.img_wr_data(img_wr_data),
        .wr_en(wr_en),.wr_layer(wr_layer),.wr_type(wr_type),
        .wr_addr(wr_addr),.wr_data(wr_data),
        .out_rd_addr(out_rd_addr),.out_rd_data(out_rd_data)
    );

    // ── VCD ───────────────────────────────────────────────────────────────────
    initial begin
        $dumpfile("generator_top.vcd");
        $dumpvars(0, tb_generator_top);
    end

    // ── Staging (Ukuran Memori Disinkronkan dengan Model LITE) ────────────────
    reg signed [DW-1:0] img_stage    [0:12287];
    
    reg signed [DW-1:0] wt_E1_s      [0:1535];
    reg signed [DW-1:0] wt_E2_s      [0:32767];
    reg signed [DW-1:0] bn_sc_E2_s   [0:63];  reg signed [DW-1:0] bn_of_E2_s [0:63];
    reg signed [DW-1:0] wt_E3_s      [0:131071];
    reg signed [DW-1:0] bn_sc_E3_s   [0:127]; reg signed [DW-1:0] bn_of_E3_s [0:127];
    reg signed [DW-1:0] wt_E4_s      [0:524287];
    reg signed [DW-1:0] bn_sc_E4_s   [0:255]; reg signed [DW-1:0] bn_of_E4_s [0:255];
    reg signed [DW-1:0] wt_E5_s      [0:2097151];
    reg signed [DW-1:0] bn_sc_E5_s   [0:511]; reg signed [DW-1:0] bn_of_E5_s [0:511];
    
    reg signed [DW-1:0] wt_BOT_s     [0:262143];
    
    reg signed [DW-1:0] wt_D1_s      [0:2097151];
    reg signed [DW-1:0] bn_sc_D1_s   [0:255]; reg signed [DW-1:0] bn_of_D1_s [0:255];
    reg signed [DW-1:0] wt_D2_s      [0:524287];
    reg signed [DW-1:0] bn_sc_D2_s   [0:127]; reg signed [DW-1:0] bn_of_D2_s [0:127];
    reg signed [DW-1:0] wt_D3_s      [0:131071];
    reg signed [DW-1:0] bn_sc_D3_s   [0:63];  reg signed [DW-1:0] bn_of_D3_s [0:63];
    reg signed [DW-1:0] wt_D4_s      [0:32767];
    reg signed [DW-1:0] bn_sc_D4_s   [0:31];  reg signed [DW-1:0] bn_of_D4_s [0:31];
    reg signed [DW-1:0] wt_D5_s      [0:16383];
    reg signed [DW-1:0] bn_sc_D5_s   [0:31];  reg signed [DW-1:0] bn_of_D5_s [0:31];
    
    reg signed [DW-1:0] wt_OUT_s     [0:863];

    // [FIX] ── Staging untuk Bias Konvolusi ─────────────────────────────────────
    reg signed [DW-1:0] bs_E1_s      [0:31];
    reg signed [DW-1:0] bs_BOT_s     [0:511];
    reg signed [DW-1:0] bs_OUT_s     [0:2];

    // ── Write tasks ───────────────────────────────────────────────────────────
    task write_weight;
        input [3:0]  layer; input [1:0] wtype;
        input [17:0] addr;  input signed [DW-1:0] data;
        begin
            @(posedge clk); #1;
            wr_en=1; wr_layer=layer; wr_type=wtype; wr_addr=addr; wr_data=data;
            @(posedge clk); #1; wr_en=0;
        end
    endtask

    task write_img;
        input [13:0] addr; input signed [DW-1:0] data;
        begin
            @(posedge clk); #1;
            img_wr_en=1; img_wr_addr=addr; img_wr_data=data;
            @(posedge clk); #1; img_wr_en=0;
        end
    endtask

    integer i, fd, timeout_cnt;

    initial begin
        $display("=================================================================");
        $display("  tb_generator_top  —  LITE Model (Fixed memory bounds & Bias)");
        $display("=================================================================");
        
        $display("[TB] 1/3 Starting $readmemh (Loading hex files to TB memory)...");
        
        // Membaca Hex File Weights & BN
        $readmemh("weights_hex/E1_conv_weight.hex",  wt_E1_s);
        $readmemh("weights_hex/E2_conv_weight.hex",  wt_E2_s);
        $readmemh("weights_hex/E2_bn_scale.hex",     bn_sc_E2_s);
        $readmemh("weights_hex/E2_bn_offset.hex",    bn_of_E2_s);
        $readmemh("weights_hex/E3_conv_weight.hex",  wt_E3_s);
        $readmemh("weights_hex/E3_bn_scale.hex",     bn_sc_E3_s);
        $readmemh("weights_hex/E3_bn_offset.hex",    bn_of_E3_s);
        $readmemh("weights_hex/E4_conv_weight.hex",  wt_E4_s);
        $readmemh("weights_hex/E4_bn_scale.hex",     bn_sc_E4_s);
        $readmemh("weights_hex/E4_bn_offset.hex",    bn_of_E4_s);
        $readmemh("weights_hex/E5_conv_weight.hex",  wt_E5_s);
        $readmemh("weights_hex/E5_bn_scale.hex",     bn_sc_E5_s);
        $readmemh("weights_hex/E5_bn_offset.hex",    bn_of_E5_s);
        
        $readmemh("weights_hex/BOT_conv_weight.hex", wt_BOT_s);
        
        $readmemh("weights_hex/D1_convt_weight.hex", wt_D1_s);
        $readmemh("weights_hex/D1_bn_scale.hex",     bn_sc_D1_s);
        $readmemh("weights_hex/D1_bn_offset.hex",    bn_of_D1_s);
        $readmemh("weights_hex/D2_convt_weight.hex", wt_D2_s);
        $readmemh("weights_hex/D2_bn_scale.hex",     bn_sc_D2_s);
        $readmemh("weights_hex/D2_bn_offset.hex",    bn_of_D2_s);
        $readmemh("weights_hex/D3_convt_weight.hex", wt_D3_s);
        $readmemh("weights_hex/D3_bn_scale.hex",     bn_sc_D3_s);
        $readmemh("weights_hex/D3_bn_offset.hex",    bn_of_D3_s);
        $readmemh("weights_hex/D4_convt_weight.hex", wt_D4_s);
        $readmemh("weights_hex/D4_bn_scale.hex",     bn_sc_D4_s);
        $readmemh("weights_hex/D4_bn_offset.hex",    bn_of_D4_s);
        $readmemh("weights_hex/D5_convt_weight.hex", wt_D5_s);
        $readmemh("weights_hex/D5_bn_scale.hex",     bn_sc_D5_s);
        $readmemh("weights_hex/D5_bn_offset.hex",    bn_of_D5_s);
        
        $readmemh("weights_hex/OUT_conv_weight.hex", wt_OUT_s);

        // [FIX] Membaca Hex File Bias
        $readmemh("weights_hex/E1_conv_bias.hex",    bs_E1_s);
        $readmemh("weights_hex/BOT_conv_bias.hex",   bs_BOT_s);
        $readmemh("weights_hex/OUT_conv_bias.hex",   bs_OUT_s);

        // Membaca Image Input
        $readmemh("inputs/img.hex",                  img_stage);
        
        $display("[TB] --> Hex files loaded successfully.");

        rst_n=0; repeat(4) @(posedge clk);
        rst_n=1; repeat(2) @(posedge clk);

        // Load weights sequentially
        $display("[TB] 2/3 Beginning weight transfer to DUT RAM...");
        
        $display("     - Writing E1, E2, E3, E4...");
        for(i=0;i<1536;    i=i+1) write_weight(0,0,i,wt_E1_s[i]);
        
        for(i=0;i<32768;   i=i+1) write_weight(1,0,i,wt_E2_s[i]);
        for(i=0;i<64;      i=i+1) write_weight(1,1,i,bn_sc_E2_s[i]);
        for(i=0;i<64;      i=i+1) write_weight(1,2,i,bn_of_E2_s[i]);
        
        for(i=0;i<131072;  i=i+1) write_weight(2,0,i,wt_E3_s[i]);
        for(i=0;i<128;     i=i+1) write_weight(2,1,i,bn_sc_E3_s[i]);
        for(i=0;i<128;     i=i+1) write_weight(2,2,i,bn_of_E3_s[i]);
        
        for(i=0;i<524288;  i=i+1) write_weight(3,0,i,wt_E4_s[i]);
        for(i=0;i<256;     i=i+1) write_weight(3,1,i,bn_sc_E4_s[i]);
        for(i=0;i<256;     i=i+1) write_weight(3,2,i,bn_of_E4_s[i]);
        
        $display("     - Writing E5 (2M params)...");
        for(i=0;i<2097152; i=i+1) write_weight(4,0,i,wt_E5_s[i]);
        for(i=0;i<512;     i=i+1) write_weight(4,1,i,bn_sc_E5_s[i]);
        for(i=0;i<512;     i=i+1) write_weight(4,2,i,bn_of_E5_s[i]);
        
        $display("     - Writing BOT...");
        for(i=0;i<262144;  i=i+1) write_weight(5,0,i,wt_BOT_s[i]);
        
        $display("     - Writing D1...");
        for(i=0;i<2097152; i=i+1) write_weight(6,0,i,wt_D1_s[i]);
        for(i=0;i<256;     i=i+1) write_weight(6,1,i,bn_sc_D1_s[i]);
        for(i=0;i<256;     i=i+1) write_weight(6,2,i,bn_of_D1_s[i]);
        
        $display("     - Writing D2, D3, D4, D5, OUT...");
        for(i=0;i<524288;  i=i+1) write_weight(7,0,i,wt_D2_s[i]);
        for(i=0;i<128;     i=i+1) write_weight(7,1,i,bn_sc_D2_s[i]);
        for(i=0;i<128;     i=i+1) write_weight(7,2,i,bn_of_D2_s[i]);
        
        for(i=0;i<131072;  i=i+1) write_weight(8,0,i,wt_D3_s[i]);
        for(i=0;i<64;      i=i+1) write_weight(8,1,i,bn_sc_D3_s[i]);
        for(i=0;i<64;      i=i+1) write_weight(8,2,i,bn_of_D3_s[i]);
        
        for(i=0;i<32768;   i=i+1) write_weight(9,0,i,wt_D4_s[i]);
        for(i=0;i<32;      i=i+1) write_weight(9,1,i,bn_sc_D4_s[i]);
        for(i=0;i<32;      i=i+1) write_weight(9,2,i,bn_of_D4_s[i]);
        
        for(i=0;i<16384;   i=i+1) write_weight(10,0,i,wt_D5_s[i]);
        for(i=0;i<32;      i=i+1) write_weight(10,1,i,bn_sc_D5_s[i]);
        for(i=0;i<32;      i=i+1) write_weight(10,2,i,bn_of_D5_s[i]);
        
        for(i=0;i<864;     i=i+1) write_weight(11,0,i,wt_OUT_s[i]);

        // [FIX] ── Transfer Bias ke DUT ─────────────────────────────────────────
        $display("     - Writing Bias E1, BOT, OUT...");
        
        // Layer 0 (E1): Tipe Tulis=3 (Bias), Alamat Base=0
        for(i=0; i<32; i=i+1)  write_weight(0, 3, i, bs_E1_s[i]);
        
        // Layer 5 (BOT): Tipe Tulis=3 (Bias), Alamat Base=32
        for(i=0; i<512; i=i+1) write_weight(5, 3, 32+i, bs_BOT_s[i]);
        
        // Layer 11 (OUT): Tipe Tulis=3 (Bias), Alamat Base=544
        for(i=0; i<3; i=i+1)   write_weight(11, 3, 544+i, bs_OUT_s[i]);
        
        $display("[TB] --> All parameters successfully loaded to DUT.");

        $display("[TB] 3/3 Writing image data to DUT...");
        for(i=0;i<IMG_SZ;i=i+1) write_img(i[13:0], img_stage[i]);
        $display("[TB] --> Input image loaded.");

        @(posedge clk); #1; start=1;
        @(posedge clk); #1; start=0;
        
        $display("=================================================================");
        $display("[TB] Inference start pulse triggered. Waiting for calculation...");
        $display("=================================================================");

        timeout_cnt=0;
        while (!done && timeout_cnt < 100_000_000) begin
            @(posedge clk);
            timeout_cnt=timeout_cnt+1;
            
            // Print progress every 100,000 cycles
            if (timeout_cnt % 100_000 == 0)
                $display("[TB] Progress: %0d Kcycles | layer_idx: %0d", 
                         timeout_cnt/1000, dut.layer_idx);
        end

        if (!done) begin 
            $display("[TB] TIMEOUT REACHED! (100,000,000 cycles).");
            $finish; 
        end
        
        $display("=================================================================");
        $display("[TB] DONE! Inference completed in %0d cycles.", timeout_cnt);

        @(posedge clk); #1;
        fd = $fopen("outputs/output_rtl.hex","w");
        if (fd==0) begin 
            $display("[TB] ERROR: Cannot open outputs/output_rtl.hex for writing"); 
            $finish;
        end
        
        for (i=0; i<OUT_SZ; i=i+1) begin
            out_rd_addr=i[13:0];
            #1;
            $fwrite(fd, "%04x\n", out_rd_data & 16'hFFFF);
        end
        $fclose(fd);
        
        $display("[TB] Output successfully written to -> outputs/output_rtl.hex");
        $display("[TB] Run: python3 verify/check_gen.py to verify results.");
        $display("=================================================================");
        $finish;
    end

endmodule
`default_nettype wire
