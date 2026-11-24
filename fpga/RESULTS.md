# EML cell on an FPGA: synthesis and simulation results

One fixed-point EML cell, `eml(x, y) = exp(x) − ln(y)`, fits on a Lattice ECP5 in 1,124 logic cells, 6 DSP multipliers and 4 block RAMs, runs at 88 MHz with one result per clock, and is accurate to about 1 LSB of its Q16.16 format (RMSE 5.4 × 10⁻⁶ on the grid the analog study used, against 10.3 mV for the analog op-amp cell). Built as a pure EML tree, a multiply costs 21× the logic and 7× the DSPs of a native multiplier, and an add costs 78× the logic of a native adder.

No board was used. "Measured" here means Yosys synthesis, nextpnr place-and-route with static timing, and Icarus Verilog simulation. Power was not measured.

## Design

`rtl/eml_cell.v`, signed Q16.16 in and out (32 bits, 16 fraction bits, LSB = 1.53 × 10⁻⁵).

- **exp(x)** = 2^(x·log₂e). One constant multiply gives t = x·log₂e. The integer part of t is a barrel shift, and 2^frac comes from a 256-entry table with linear interpolation (one 17×16 multiply).
- **ln(y)** = (p − 16)·ln 2 + ln(m). A leading-one detector gives p, the mantissa m ∈ [1, 2) indexes a 256-entry table with linear interpolation (one 16×16 multiply), and (p − 16)·ln 2 comes from a 31-entry constant table.
- Each table entry is offset by half its segment's chord error, so the interpolation error is centred instead of one-sided.
- 10 pipeline stages, one result per clock. Inputs outside the range saturate: exp saturates above x ≈ 10.4, y ≤ 0 raises `invalid` and saturates the output (real branch only; there is no complex mode).

I chose tables with interpolation over CORDIC because the ECP5 has DSP blocks and block RAM to spare, and this gives one result per clock at roughly 1 LSB error. A multiplier-free CORDIC would need about 16–20 iterations per function, so it would be smaller on DSP-less parts and much slower or much deeper. I did not build it.

`eml_model.py` is a bit-exact Python model. The RTL matches it on 262,409 vectors (random in-domain, random 32-bit patterns, edges, every table segment and every power of two) with 0 mismatches. The three tree builds also match the model, 60,000 vectors each. A mutation check confirmed the testbench catches wrong outputs.

Trees come from `gen_tree.py`, which turns an `e(L,R)` witness string into pipelined Verilog: one `eml_cell` per node, constant `1` leaves, and delay registers so every node's two inputs belong to the same sample. The witnesses are the repo's own minimal gadgets from `benchmarks/perm_det_gadgets.py`: x·y in 8 nodes, x + y in 9 and x − y in 5.

## Area and speed

ECP5 LFE5U-85F (CABGA381), Yosys `synth_ecp5`, nextpnr-ecp5 seed 1. "Logic" is nextpnr's TRELLIS_COMB count (LUT4s plus carry and mux halves).

| Design | Logic | FFs | DSP 18×18 | BRAM | Fmax (MHz) | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Q16.16 adder (native) | 73 | 96 | 0 | 0 | 180 | 2 |
| Q16.16 multiplier (native) | 238 | 145 | 4 | 0 | 84 | 3 |
| Q16.16 MAC (native) | 277 | 132 | 4 | 0 | 84 | 3 |
| **EML cell** | **1,124** | **708** | **6** | **4** | **88** | **10** |
| x − y as EML tree (5 nodes) | 3,522 | 3,328 | 20 | 18 | 78 | 40 |
| x · y as EML tree (8 nodes) | 5,035 | 5,493 | 29 | 29 | 76 | 80 |
| x + y as EML tree (9 nodes) | 5,703 | 5,958 | 34 | 32 | 72 | 80 |

Synthesis folds the constant `1` leaves. ln(1) = 0 removes most of a node's ln path, so the trees come out at about 56% of (nodes × one cell) in logic and 60% in DSPs.

iCE40-HX8K (CT256), which has no DSP blocks, so multipliers become LUTs:

| Design | Logic cells | BRAM (4 kbit) | Fmax (MHz) |
|---|---:|---:|---:|
| Q16.16 adder | 132 | 0 | 110 |
| Q16.16 multiplier | 3,183 | 0 | 48 |
| Q16.16 MAC | 2,659 | 0 | 53 |
| EML cell | 3,309 | 7 | 65 |
| x − y as EML tree | does not fit (9,036 LUT4 > 7,680) | | |

On DSP-less fabric one EML cell costs about as much as one 32-bit multiplier, because the cell's multiplies are narrow (17×16, 16×16 and 21×27 with a constant) while the native one is 32×32.

## Accuracy

Errors are against float64 evaluated on the quantized inputs, so they measure the cell, not the input rounding (`accuracy.py`, `results/accuracy.json`).

| Region | RMSE | Max error |
|---|---:|---:|
| Analog study's grid, 20 × 41 over x ∈ [−1, 1], y ∈ [0.1, 2] | 5.4 × 10⁻⁶ (0.36 LSB) | 1.3 × 10⁻⁵ (0.85 LSB) |
| Same box, 10⁶ random points | 6.3 × 10⁻⁶ (0.41 LSB) | 1.7 × 10⁻⁵ (1.09 LSB) |
| ln term alone, y ∈ [2⁻¹⁶, 32767] | 0.29 LSB | 0.56 LSB |
| exp term alone, relative, where exp(x) > 64 | | 6.1 × 10⁻⁷ |

Absolute error grows with the output because exp is accurate to about 6 × 10⁻⁷ relative and a 32-bit word cannot hold more. Near the top of the range (outputs ≈ 30,000) the absolute error reaches 0.017.

The trees lose accuracy, mostly because their intermediates swing exponentially through a fixed-point format. Inputs log-uniform in [0.01, 8]:

| Tree | Valid on real branch | Saturated | Relative error p50 / p99 / max | Native op |
|---|---:|---:|---|---|
| x · y (8 nodes) | 100% | 2.6% | 2.3 × 10⁻⁵ / 5.1 × 10⁻⁴ / 8.1 × 10⁻⁴ | 0.5 LSB |
| x + y (9 nodes) | 84% | < 0.01% | 6.9 × 10⁻⁶ / 4.5 × 10⁻⁴ / 2.2 × 10⁻³ | exact |
| x − y (5 nodes) | 100% | 0% | 0 / 4.8 × 10⁻⁵ / 4.5 × 10⁻³ (max 5 LSB) | exact |

- The multiply tree computes exp(e − ln x − ln y) = e^e/(xy) along the way. That overflows Q16.16 when xy < 4.6 × 10⁻⁴ (the 2.6% saturated) and loses relative precision when xy is large, because the intermediate is then a small number held to a fixed absolute resolution.
- The add tree is only valid on the real branch when x + y < e, because its accumulator takes ln(e − x − y). The 16% invalid share is that region. On the complex branch the gadgets are exact everywhere, but this hardware has no complex mode.
- A fixed-point EML datapath therefore wants a floating-point or logarithmic number format for the intermediates. That is the main design lesson from building the trees.

## Comparison with the analog cells

Analog numbers are cited from `analog/spice/README.md` (ngspice, 2N3904-class BJTs, behavioural op-amps), not re-measured.

| | FPGA cell (this work) | Analog op-amp cell | Analog translinear cell |
|---|---|---|---|
| DC RMSE on the 20 × 41 grid | 5.4 × 10⁻⁶ | 10.3 mV (1.03 × 10⁻²) | 16.2 mV |
| Max error on the grid | 1.3 × 10⁻⁵ | 26.0 mV | 53.5 mV |
| Speed | 88 M results/s, 114 ns latency | 501 kHz −3 dB bandwidth, 3.4 µs settle | 178 kHz, 4.1 µs |
| Temperature | Bit-exact at any temperature that meets timing | RMSE 200 mV at 25 °C, 8.5 V at 55 °C (uncompensated V_T divider) | same issue |
| Depth-2 cascade | Bit-exact; error propagates as computed in `accuracy.py` | 20.3 mV RMSE | |
| Hardware per cell | 1,124 logic cells, 6 DSPs, 4 BRAMs | 3 BJTs, 6 op-amps, about 19 resistors | similar |

The digital cell is about 1,900× more accurate on the same grid and has no drift. The analog cell's case rests on power and area per operation in custom silicon, which neither study measured.

## MAC crossover

The open question from the analog exploration was when an EML cell beats a conventional multiply-accumulate. A dot-product term a·b + acc built from the repo's gadgets, `SUM(PROD(a_i, b_i), ...)`, costs about 13 EML nodes per term (12 for k = 1, 12.9 for k = 8 and k = 16, from `perm_det_gadgets`).

On the ECP5, a realized EML node averages 632 logic cells, 3.7 DSPs and 3.6 BRAMs (the mul and add trees, constant folding included). That puts one EML MAC term at about 8,200 logic cells, 48 DSPs and 47 BRAMs, against 277 logic cells and 4 DSPs for the native MAC: about 30× the logic, 12× the DSPs, block RAM the MAC doesn't need, roughly 120 cycles of latency instead of 3, and a real-branch domain limit (partial sums must stay below e). On an FPGA, EML loses for multiply-accumulate at every size.

For analog, the same count gives a lower bound in devices. At 13 cells × (3 BJTs + 6 op-amps) per term, with a small op-amp at about 8–10 transistors, an EML MAC term needs roughly 700–1,000 transistors plus about 250 resistors. A crossbar MAC cell is one transistor and one resistive element per weight, with ADCs shared across a column. That is about three orders of magnitude in device count before noise and temperature, so EML is not a MAC substrate in analog either. This is a device-count estimate from the existing netlists, not a 22 nm layout or power study.

EML pays off only where a node replaces a transcendental the conventional design would need anyway. exp(x) is 1 node and ln(x) is 3, against an exp or ln unit that is a large fraction of the cell. The case for an EML fabric is uniformity: one cell type, one tile, and a tree from `eml_sr` maps straight onto hardware with no operator library. That is worth something for small, fixed, transcendental-heavy expressions (log compression, ratios, softplus-like nonlinearities). It is not worth it for arithmetic, where native adders and multipliers are 20–80× cheaper.

## Reproducing

```bash
pip install yowasp-yosys yowasp-nextpnr-ecp5 yowasp-nextpnr-ice40   # synthesis
sudo apt-get install iverilog                                       # simulation

python fpga/eml_model.py          # regenerate rtl/*.hex tables
python fpga/gen_tree.py           # regenerate rtl/eml_tree_*.v
cd fpga/tb && python make_vectors.py 200000 && cp ../rtl/*.hex . \
  && iverilog -g2012 -o sim ../rtl/eml_cell.v tb_eml_cell.v && vvp -n sim
python make_tree_vectors.py mul 40000 && iverilog -g2012 -DTREE=eml_tree_mul \
  -o simt ../rtl/eml_cell.v ../rtl/eml_tree_mul.v tb_eml_tree.v && vvp -n simt
cd ../.. && python fpga/synth.py --target ecp5      # about 5 minutes
python fpga/synth.py q16_add q16_mul q16_mac eml_cell --target ice40
python fpga/accuracy.py
python -m pytest tests/test_fpga_model.py
```

Toolchain versions used: yowasp-yosys and yowasp-nextpnr from PyPI on 2026-11-24, Icarus Verilog 12.0. Fmax is from one nextpnr seed and moves a few MHz between seeds.

## Files

| Path | Role |
|---|---|
| `eml_model.py` | Bit-exact model, table generation |
| `rtl/eml_cell.v` | The cell |
| `rtl/baselines.v` | Native adder, multiplier, MAC |
| `rtl/eml_tree_{mul,add,sub}.v` | Generated trees |
| `rtl/*.hex` | ROM images |
| `gen_tree.py` | Witness string to pipelined Verilog; tree model |
| `tb/` | Testbenches and vector generators |
| `synth.py` | Synthesis and place-and-route driver |
| `accuracy.py` | Error analysis |
| `results/synth.json`, `results/accuracy.json` | Measured numbers |
