// Fixed-point EML cell: out = exp(x) - ln(y), signed Q16.16 in and out.
// Bit-exact with fpga/eml_model.py (checked by fpga/tb/).
//
// exp: x*log2(e) -> integer part n (shift) + 24-bit fraction; 2^frac from a
//      256-entry table with linear interpolation (one 17x16 multiply).
// ln:  leading-one position p + mantissa m in [1,2); ln(m) from a 256-entry
//      table with linear interpolation (one 16x16 multiply); plus (p-16)*ln2
//      from a 31-entry constant table.
// Fully pipelined, one result per clock.  Latency LATENCY = 10: x/y sampled
// on clock edge 0, out and out_valid updated on edge 9.
//
// Tables are read with $readmemh from the working directory (run the tools
// from fpga/rtl/, or copy the .hex files next to the simulation).

module eml_cell (
    input  wire               clk,
    input  wire               in_valid,
    input  wire signed [31:0] x,
    input  wire signed [31:0] y,
    output reg                out_valid,
    output reg  signed [31:0] out,
    output reg                invalid      // y <= 0 (ln undefined on the real branch)
);
    localparam LATENCY = 10;

    // ---------------------------------------------------------------- ROMs
    reg [30:0] exp_base  [0:255];
    reg [16:0] exp_slope [0:255];   // slopes in units of 2^-24
    reg [29:0] ln_base   [0:255];
    reg [15:0] ln_slope  [0:255];
    initial begin
        $readmemh("exp_base.hex",  exp_base);
        $readmemh("exp_slope.hex", exp_slope);
        $readmemh("ln_base.hex",   ln_base);
        $readmemh("ln_slope.hex",  ln_slope);
    end

    localparam signed [26:0] K_LOG2E = 27'sd48408813;   // round(log2(e) * 2^25)
    localparam signed [31:0] X_HI    = 32'sh000F_FFFF;  // clamp x to [-16, 16)
    localparam signed [31:0] X_LO    = -32'sh0010_0000;

    reg [LATENCY-1:0] v;

    // ---------------------------------------------------------- stage 0
    reg signed [31:0] x0, y0;
    always @(posedge clk) begin
        x0 <= x;
        y0 <= y;
    end

    // ---------------------------------------------------------- stage 1
    // exp: clamp x.  ln: invalid flag + leading-one detect.
    reg  signed [20:0] xc1;
    reg  [4:0]         p1;
    reg  [30:0]        y1;
    reg                inv1;

    integer k;
    reg [4:0] lod;
    always @(*) begin
        lod = 5'd0;
        for (k = 1; k < 31; k = k + 1)
            if (y0[k]) lod = k[4:0];
    end

    always @(posedge clk) begin
        xc1  <= (x0 > X_HI) ? X_HI[20:0] : (x0 < X_LO) ? X_LO[20:0] : x0[20:0];
        inv1 <= (y0[31] == 1'b1) || (y0 == 32'sd0);
        p1   <= lod;
        y1   <= (y0[31] == 1'b1 || y0 == 32'sd0) ? 31'd1 : y0[30:0];
    end

    // ---------------------------------------------------------- stage 1b
    // exp: constant multiply.  ln: normalise the mantissa.
    reg signed [47:0] t1;
    reg [30:0]        norm1;
    reg [4:0]         p1b;
    reg               inv1b;
    always @(posedge clk) begin
        t1    <= xc1 * K_LOG2E;
        norm1 <= y1 << (5'd30 - p1);      // Q1.30, [2^30, 2^31)
        p1b   <= p1;
        inv1b <= inv1;
    end

    // ---------------------------------------------------------- stage 2
    // split t into n.f, normalise y, read the tables (synchronous ROM).
    wire signed [30:0] t24 = t1[47:17];          // Q.24, floor
    wire signed [5:0]  n_w = t24[29:24];
    wire [7:0]         eidx = t24[23:16];
    wire [30:0]        norm = norm1;
    wire [7:0]         lidx = norm[29:22];

    reg [30:0] e0_2;
    reg [16:0] es_2;
    reg [29:0] l0_2;
    reg [15:0] ls_2;
    reg [15:0] er_2, lr_2;
    reg signed [5:0] n_2;
    reg [4:0]  p_2;
    reg        inv_2;
    always @(posedge clk) begin
        e0_2  <= exp_base[eidx];
        es_2  <= exp_slope[eidx];
        l0_2  <= ln_base[lidx];
        ls_2  <= ln_slope[lidx];
        er_2  <= t24[15:0];
        lr_2  <= norm[21:6];
        n_2   <= n_w;
        p_2   <= p1b;
        inv_2 <= inv1b;
    end

    // ---------------------------------------------------------- stage 2b
    // ROM output register (maps onto the block RAM's own output register).
    reg [30:0] e0_2b;
    reg [16:0] es_2b;
    reg [29:0] l0_2b;
    reg [15:0] ls_2b;
    reg [15:0] er_2b, lr_2b;
    reg signed [5:0] n_2b;
    reg [4:0]  p_2b;
    reg        inv_2b;
    always @(posedge clk) begin
        e0_2b <= e0_2;  es_2b <= es_2;  l0_2b <= l0_2;  ls_2b <= ls_2;
        er_2b <= er_2;  lr_2b <= lr_2;  n_2b  <= n_2;   p_2b  <= p_2;
        inv_2b <= inv_2;
    end

    // ---------------------------------------------------------- stage 3
    reg [32:0] ep_3;
    reg [31:0] lp_3;
    reg [30:0] e0_3;
    reg [29:0] l0_3;
    reg signed [5:0] n_3;
    reg [4:0]  p_3;
    reg        inv_3;
    always @(posedge clk) begin
        ep_3  <= es_2b * er_2b;
        lp_3  <= ls_2b * lr_2b;
        e0_3  <= e0_2b;
        l0_3  <= l0_2b;
        n_3   <= n_2b;
        p_3   <= p_2b;
        inv_3 <= inv_2b;
    end

    // ---------------------------------------------------------- stage 4
    reg [31:0]        m_4;      // Q.30 mantissa of 2^frac
    reg signed [36:0] ln30_4;   // Q.30 ln(y)
    reg signed [5:0]  n_4;
    reg               inv_4;
    reg  signed [37:0] pln2;                 // (p - 16) * ln2, Q.30
    integer j;
    always @(*) begin
        pln2 = 38'sd0;
        for (j = 0; j < 31; j = j + 1)
            if (p_3 == j[4:0]) pln2 = (j - 16) * 38'sd744261118;
    end
    always @(posedge clk) begin
        m_4    <= {1'b0, e0_3} + {9'd0, ep_3[32:10]};
        ln30_4 <= pln2 + $signed({7'd0, l0_3}) + $signed({15'd0, lp_3[31:10]});
        n_4    <= n_3;
        inv_4  <= inv_3;
    end

    // ---------------------------------------------------------- stage 5a
    // exp: shift amount, rounding constant, saturate/flush flags.
    wire signed [6:0] sh  = 7'sd14 - n_4;          // right shift of m
    wire [32:0]       rnd = (sh > 0) ? (33'd1 << (sh - 1)) : 33'd0;
    reg  [32:0]       mr_5;
    reg  [5:0]        sh_5;
    reg               sat_5, zero_5;
    reg signed [32:0] ln_5a;
    reg               inv_5a;
    always @(posedge clk) begin
        mr_5   <= {1'b0, m_4} + rnd;
        sh_5   <= sh[5:0];
        sat_5  <= (n_4 >= 6'sd15);
        zero_5 <= (sh >= 7'sd32);
        ln_5a  <= (ln30_4 + 37'sd8192) >>> 14;
        inv_5a <= inv_4;
    end

    // ---------------------------------------------------------- stage 5b
    wire [32:0]        eq = mr_5 >> sh_5;
    reg  signed [32:0] exp_5;
    reg  signed [32:0] ln_5;
    reg                inv_5;
    always @(posedge clk) begin
        if (sat_5)                     exp_5 <= 33'sh0_7FFF_FFFF;
        else if (zero_5)               exp_5 <= 33'sd0;
        else if (eq > 33'h0_7FFF_FFFF) exp_5 <= 33'sh0_7FFF_FFFF;
        else                           exp_5 <= $signed(eq);
        ln_5  <= ln_5a;
        inv_5 <= inv_5a;
    end

    // ---------------------------------------------------------- stage 6
    wire signed [33:0] d = exp_5 - ln_5;
    always @(posedge clk) begin
        if (inv_5 || d > 34'sh0_7FFF_FFFF) out <= 32'sh7FFF_FFFF;
        else if (d < -34'sh0_8000_0000)    out <= 32'sh8000_0000;
        else                               out <= d[31:0];
        invalid <= inv_5;
    end

    always @(posedge clk) begin
        v         <= {v[LATENCY-2:0], in_valid};
        out_valid <= v[LATENCY-2];
    end
    initial v = 0;
endmodule
