// =============================================================================
// mac_array.v  —  K*K Parallel MAC Array (with optional 2-stage pipeline)
//
// PIPE_STAGES parameter:
//   0 = fully combinational (original behavior, backward compat)
//   2 = 2-stage pipeline:
//         Stage 1: register products after multiply   (cuts DSP critical path)
//         Stage 2: register adder tree output         (cuts adder critical path)
//       sum_out valid PIPE_STAGES clock cycles after in_valid is asserted.
// =============================================================================
`default_nettype none

module mac_array #(
    parameter DW          = 16,   // data width per element
    parameter ACC_W       = 48,   // accumulator width
    parameter K           = 4,    // kernel size (K*K MAC units)
    parameter PIPE_STAGES = 2     // 0 = combinational, 2 = 2-stage pipeline
)(
    input  wire                     clk,         // required for pipeline registers
    input  wire [K*K*DW-1:0]       weight_vec,
    input  wire [K*K*DW-1:0]       ifmap_vec,
    input  wire [K*K-1:0]          valid_vec,
    input  wire                     in_valid,    // pulse 1 when inputs are stable
    output wire signed [ACC_W-1:0]  sum_out,
    output wire                     out_valid    // sum_out valid PIPE_STAGES cycles later
);
    localparam N_MAC = K * K;

    // =========================================================================
    // Stage 0: Combinational multiply (K*K parallel 16×16 signed multipliers)
    // =========================================================================
    wire signed [2*DW-1:0]  prod    [0:N_MAC-1];
    wire signed [ACC_W-1:0] prod_sx [0:N_MAC-1]; // sign-extended to ACC_W

    genvar m;
    generate
        for (m = 0; m < N_MAC; m = m + 1) begin : MUL
            wire signed [DW-1:0] wval = weight_vec[m*DW +: DW];
            // Zero out invalid taps (padding / out-of-range positions)
            wire signed [DW-1:0] ival = valid_vec[m] ? ifmap_vec[m*DW +: DW]
                                                     : {DW{1'b0}};

            // [FIX] Cast to 2*DW BEFORE multiply to avoid truncation
            wire signed [2*DW-1:0] wval_32 = $signed(wval);
            wire signed [2*DW-1:0] ival_32 = $signed(ival);

            assign prod[m]    = wval_32 * ival_32;
            // Sign-extend 32-bit product to accumulator width
            assign prod_sx[m] = {{(ACC_W - 2*DW){prod[m][2*DW-1]}}, prod[m]};
        end
    endgenerate

    // =========================================================================
    // Pipeline Stage 1: Register products after multiply
    // Breaks the DSP multiply → adder-tree critical path into two shorter paths.
    // When PIPE_STAGES==0, prod_reg is unused; synthesis will optimize it away.
    // =========================================================================
    reg signed [ACC_W-1:0] prod_reg [0:N_MAC-1];

    genvar g;
    generate
        for (g = 0; g < N_MAC; g = g + 1) begin : STAGE1_REG
            always @(posedge clk) begin
                prod_reg[g] <= prod_sx[g];
            end
        end
    endgenerate

    // =========================================================================
    // Flatten for adder tree:
    //   PIPE_STAGES >= 1 → use registered products (prod_reg)
    //   PIPE_STAGES == 0 → use combinational products (prod_sx), original path
    // PIPE_STAGES is a compile-time constant; synthesis optimises away the mux.
    // =========================================================================
    wire [N_MAC*ACC_W-1:0] prod_flat;
    genvar i;
    generate
        for (i = 0; i < N_MAC; i = i + 1) begin : FLAT
            assign prod_flat[i*ACC_W +: ACC_W] =
                (PIPE_STAGES >= 1) ? prod_reg[i] : prod_sx[i];
        end
    endgenerate

    // =========================================================================
    // Binary Adder Tree  (same recursive structure as before)
    // =========================================================================
    wire signed [ACC_W-1:0] tree_out;
    adder_tree #(.W(ACC_W), .N(N_MAC)) u_adder_tree (
        .in_vec (prod_flat),
        .out_sum(tree_out)
    );

    // =========================================================================
    // Pipeline Stage 2: Register adder-tree output
    // When PIPE_STAGES < 2, sum_out_reg is unused; synthesis optimises it away.
    // =========================================================================
    reg signed [ACC_W-1:0] sum_out_reg;
    always @(posedge clk) begin
        sum_out_reg <= tree_out;
    end

    assign sum_out = (PIPE_STAGES >= 2) ? sum_out_reg : tree_out;

    // =========================================================================
    // Valid shift register — delays in_valid by PIPE_STAGES cycles
    // =========================================================================
    generate
        if (PIPE_STAGES == 0) begin : COMB_VALID
            // Combinational: no latency, out_valid mirrors in_valid
            assign out_valid = in_valid;
        end else if (PIPE_STAGES == 1) begin : PIPE1_VALID
            reg valid_sr1;
            always @(posedge clk) valid_sr1 <= in_valid;
            assign out_valid = valid_sr1;
        end else begin : PIPE2_VALID
            // PIPE_STAGES >= 2: shift register of depth PIPE_STAGES
            // in_valid enters at LSB, exits at MSB after PIPE_STAGES cycles
            reg [PIPE_STAGES-1:0] valid_sr;
            always @(posedge clk) begin
                valid_sr <= {valid_sr[PIPE_STAGES-2:0], in_valid};
            end
            assign out_valid = valid_sr[PIPE_STAGES-1];
        end
    endgenerate

endmodule

// =============================================================================
// adder_tree  —  Recursive Binary Adder Tree (unchanged)
// =============================================================================
module adder_tree #(
    parameter W = 48,
    parameter N = 16
)(
    input  wire [N*W-1:0]     in_vec,
    output wire signed [W-1:0] out_sum
);
    generate
        if (N == 0) begin : empty
            assign out_sum = {W{1'b0}};
        end else if (N == 1) begin : leaf
            assign out_sum = in_vec[0 +: W];
        end else if (N == 2) begin : node
            assign out_sum = $signed(in_vec[0 +: W]) + $signed(in_vec[W +: W]);
        end else begin : branch
            wire signed [W-1:0] sum_left;
            wire signed [W-1:0] sum_right;
            localparam LEFT_N  = N / 2;
            localparam RIGHT_N = N - LEFT_N;

            adder_tree #(.W(W), .N(LEFT_N)) left_tree (
                .in_vec (in_vec[0 +: LEFT_N*W]),
                .out_sum(sum_left)
            );
            adder_tree #(.W(W), .N(RIGHT_N)) right_tree (
                .in_vec (in_vec[LEFT_N*W +: RIGHT_N*W]),
                .out_sum(sum_right)
            );

            assign out_sum = sum_left + sum_right;
        end
    endgenerate
endmodule
`default_nettype wire
