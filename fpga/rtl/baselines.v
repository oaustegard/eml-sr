// Native Q16.16 baselines, synthesized with the same flow as eml_cell.v.

// Saturating adder: out = x + y.  Latency 2 (input and output registers).
module q16_add (
    input  wire               clk,
    input  wire signed [31:0] x,
    input  wire signed [31:0] y,
    output reg  signed [31:0] out
);
    reg signed [31:0] x0, y0;
    wire signed [32:0] s = x0 + y0;
    always @(posedge clk) begin
        x0 <= x;
        y0 <= y;
        if (s > 33'sh0_7FFF_FFFF)       out <= 32'sh7FFF_FFFF;
        else if (s < -33'sh0_8000_0000) out <= 32'sh8000_0000;
        else                            out <= s[31:0];
    end
endmodule

// Rounding, saturating multiplier: out = x * y.  Latency 3.
module q16_mul (
    input  wire               clk,
    input  wire signed [31:0] x,
    input  wire signed [31:0] y,
    output reg  signed [31:0] out
);
    reg signed [31:0] x0, y0;
    reg signed [63:0] p;
    wire signed [63:0] pr = (p + 64'sd32768) >>> 16;
    always @(posedge clk) begin
        x0 <= x;
        y0 <= y;
        p  <= x0 * y0;
        if (pr > 64'sh0_7FFF_FFFF)       out <= 32'sh7FFF_FFFF;
        else if (pr < -64'sh8000_0000)   out <= 32'sh8000_0000;
        else                             out <= pr[31:0];
    end
endmodule

// Multiply-accumulate: acc <= sat(acc + x * y).  The conventional unit an
// EML datapath would replace in a dot product.
module q16_mac (
    input  wire               clk,
    input  wire               clr,
    input  wire signed [31:0] x,
    input  wire signed [31:0] y,
    output reg  signed [31:0] acc
);
    reg signed [31:0] x0, y0;
    reg signed [63:0] p;
    reg               clr0, clr1;
    wire signed [32:0] pr = (p + 64'sd32768) >>> 16;
    wire signed [33:0] s  = (clr1 ? 34'sd0 : acc) + pr;
    always @(posedge clk) begin
        x0 <= x; y0 <= y; clr0 <= clr; clr1 <= clr0;
        p  <= x0 * y0;
        if (s > 34'sh0_7FFF_FFFF)       acc <= 32'sh7FFF_FFFF;
        else if (s < -34'sh0_8000_0000) acc <= 32'sh8000_0000;
        else                            acc <= s[31:0];
    end
endmodule
