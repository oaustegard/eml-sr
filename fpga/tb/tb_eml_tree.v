// Streams test vectors through a generated EML tree (`TREE), one per clock, and checks every
// output bit-for-bit against the Python model (fpga/eml_model.py).
// Vector file lines: x y expected_out expected_invalid  (hex, 32/32/32/1 bits)
`timescale 1ns/1ps
module tb_eml_tree;
    reg clk = 0;
    always #5 clk = ~clk;

    reg               in_valid = 0;
    reg  signed [31:0] x = 0, y = 0;
    wire              out_valid, invalid;
    wire signed [31:0] out;

    `TREE dut(.clk(clk), .in_valid(in_valid), .x0(x), .x1(y),
                 .out_valid(out_valid), .out(out), .invalid(invalid));

    integer fd, n_in = 0, n_out = 0, n_bad = 0, rc;
    reg [31:0] vx, vy, vo;
    reg [0:0]  vi;
    reg [31:0] exp_out [0:1048575];
    reg        exp_inv [0:1048575];
    reg done_reading = 0;

    initial begin
        fd = $fopen("vectors.txt", "r");
        if (fd == 0) begin $display("no vectors.txt"); $finish; end
    end

    always @(posedge clk) begin
        if (!done_reading) begin
            rc = $fscanf(fd, "%h %h %h %h\n", vx, vy, vo, vi);
            if (rc == 4) begin
                x <= vx; y <= vy; in_valid <= 1;
                exp_out[n_in] = vo; exp_inv[n_in] = vi;
                n_in = n_in + 1;
            end else begin
                in_valid <= 0; done_reading = 1;
            end
        end else in_valid <= 0;

        if (out_valid) begin
            if (out !== exp_out[n_out] || invalid !== exp_inv[n_out]) begin
                n_bad = n_bad + 1;
                if (n_bad <= 10)
                    $display("MISMATCH #%0d: got %h/%b want %h/%b", n_out,
                             out, invalid, exp_out[n_out], exp_inv[n_out]);
            end
            n_out = n_out + 1;
            if (done_reading && n_out == n_in) begin
                $display("RESULT vectors=%0d mismatches=%0d", n_out, n_bad);
                $finish;
            end
        end
    end
endmodule
