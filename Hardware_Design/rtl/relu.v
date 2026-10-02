`default_nettype none

module relu #(
    parameter DW = 16
)(
    input  wire                   valid_in,
    input  wire signed [DW-1:0]   x_in,
    output wire                   valid_out,
    output wire signed [DW-1:0]   y_out
);
    assign valid_out = valid_in;
    assign y_out     = (x_in >= 0) ? x_in : {DW{1'b0}};

endmodule
`default_nettype wire
