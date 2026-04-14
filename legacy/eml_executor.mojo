"""EML Executor: A compiled transformer whose only compute instruction is eml(x,y) = exp(x) - ln(y).

Architecture mirrors llm-as-computer but with a 3-opcode ISA:
  PUSH val  — push a complex constant onto the stack
  EML       — pop y (top), pop x (second), push exp(x) - ln(y)
  HALT      — stop execution

Every stack read is a parabolic attention dot product + argmax.
Every stack write appends a parabolic key-value pair to the context.
The feed-forward layer is trivial: one EML computation.

Programs are RPN sequences. From the paper (Odrzywolek 2026):
  e      = 1 1 EML
  exp(x) = x 1 EML
  ln(x)  = 1 1 x EML 1 EML EML
"""

from std.math import exp, log, cos, sin, sqrt, abs


def _atan2(y: Float64, x: Float64) -> Float64:
    """Manual atan2 using atan(y/x) with quadrant correction."""
    # Handle special cases
    var pi: Float64 = 3.141592653589793
    if x > 0:
        return _atan(y / x)
    elif x < 0:
        if y >= 0:
            return _atan(y / x) + pi
        else:
            return _atan(y / x) - pi
    else:  # x == 0
        if y > 0:
            return pi / 2.0
        elif y < 0:
            return -(pi / 2.0)
        else:
            return 0.0


def _atan(x: Float64) -> Float64:
    """Arctangent via Chebyshev-like rational approximation."""
    # Range reduction: atan(x) for |x| > 1 uses atan(x) = pi/2 - atan(1/x)
    var pi_2: Float64 = 1.5707963267948966
    if x > 1.0:
        return pi_2 - _atan_core(1.0 / x)
    elif x < -1.0:
        return -pi_2 - _atan_core(1.0 / x)
    else:
        return _atan_core(x)


def _atan_core(x: Float64) -> Float64:
    """Polynomial approximation for atan(x), |x| <= 1. Abramowitz & Stegun."""
    var x2 = x * x
    # 7th order minimax: max error ~1e-7
    return x * (0.9998660 + x2 * (-0.3302995 + x2 * (0.1801410 + x2 * (-0.0851330 + x2 * 0.0208351))))
from std.time import perf_counter_ns

# ─── Complex64 ────────────────────────────────────────────────────

struct Complex64(Writable, Copyable, Movable, ImplicitlyCopyable):
    """Complex number with Float64 components."""
    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re
        self.im = im

    def __add__(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    def __sub__(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    def __mul__(self, other: Self) -> Self:
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def mag_sq(self) -> Float64:
        return self.re * self.re + self.im * self.im

    def mag(self) -> Float64:
        return sqrt(self.mag_sq())

    def write_to(self, mut writer: Some[Writer]):
        if abs(self.im) < 1e-12:
            writer.write(self.re)
        elif abs(self.re) < 1e-12:
            writer.write(self.im, "i")
        else:
            if self.im >= 0:
                writer.write(self.re, "+", self.im, "i")
            else:
                writer.write(self.re, self.im, "i")


def complex_exp(z: Complex64) -> Complex64:
    """exp(a + bi) = e^a (cos b + i sin b)."""
    var ea = exp(z.re)
    return Complex64(ea * cos(z.im), ea * sin(z.im))


def complex_ln(z: Complex64) -> Complex64:
    """Principal branch: ln(z) = ln|z| + i·arg(z).
    Special case: ln(0) → (-inf, 0) to match IEEE 754 / the paper's convention.
    """
    var m = z.mag()
    if m < 1e-300:
        return Complex64(-1e308, 0.0)  # -inf stand-in
    return Complex64(log(m), _atan2(z.im, z.re))


def eml_op(x: Complex64, y: Complex64) -> Complex64:
    """The EML Sheffer operator: eml(x, y) = exp(x) - ln(y)."""
    return complex_exp(x) - complex_ln(y)


# ─── Parabolic Attention Stack ────────────────────────────────────

struct ParabolicStack:
    """Stack using parabolic key encoding for attention-based reads.

    Write at address a: key = (2a, -a²)
    Read for address a: query = (a, 1), score = dot(key, query)
    Score = 2a·j + (-j²)·1 = -(j-a)² + a²  →  peaks at j = a.
    """
    var keys_0: List[Float64]    # first component of 2D key
    var keys_1: List[Float64]    # second component
    var values_re: List[Float64]
    var values_im: List[Float64]
    var sp: Int                  # stack pointer (next write address)
    var count: Int               # total entries (including overwrites)

    def __init__(out self):
        self.keys_0 = List[Float64]()
        self.keys_1 = List[Float64]()
        self.values_re = List[Float64]()
        self.values_im = List[Float64]()
        self.sp = 0
        self.count = 0

    def push(mut self, val: Complex64):
        """Append a parabolic key-value pair for address sp."""
        self.sp += 1
        var addr = Float64(self.sp)
        self.keys_0.append(2.0 * addr)
        self.keys_1.append(-(addr * addr))
        self.values_re.append(val.re)
        self.values_im.append(val.im)
        self.count += 1

    def attention_read(self, addr: Int) -> Complex64:
        """Read stack[addr] via parabolic dot product + argmax.
        On tied scores, picks the most recent entry (highest index).
        """
        var q0 = Float64(addr)
        var q1: Float64 = 1.0
        var best_score = -1e308
        var best_idx = 0
        for i in range(self.count):
            var score = self.keys_0[i] * q0 + self.keys_1[i] * q1
            if score >= best_score:  # >= ensures last-writer-wins
                best_score = score
                best_idx = i
        return Complex64(self.values_re[best_idx], self.values_im[best_idx])

    def read_top(self) -> Complex64:
        """Read value at current stack pointer via attention."""
        return self.attention_read(self.sp)

    def read_second(self) -> Complex64:
        """Read value at sp-1 via attention."""
        return self.attention_read(self.sp - 1)

    def pop_two(mut self) -> Tuple[Complex64, Complex64]:
        """Pop top two: returns (second, top) = (x, y) for eml(x, y).
        Uses attention for reads, then decrements sp.
        """
        var top = self.read_top()        # y
        var second = self.read_second()  # x
        self.sp -= 2
        return (second^, top^)


# ─── Opcodes ──────────────────────────────────────────────────────

comptime OP_HALT = 0
comptime OP_PUSH = 1
comptime OP_EML = 2


struct Instruction(Copyable, Movable):
    var opcode: Int
    var val_re: Float64
    var val_im: Float64

    def __init__(out self, opcode: Int, val_re: Float64 = 0.0, val_im: Float64 = 0.0):
        self.opcode = opcode
        self.val_re = val_re
        self.val_im = val_im


# ─── Parabolic Program Memory ────────────────────────────────────

struct ProgramMemory:
    """Program stored as parabolic key-value pairs — instruction fetch is attention."""
    var keys_0: List[Float64]
    var keys_1: List[Float64]
    var opcodes: List[Int]
    var args_re: List[Float64]
    var args_im: List[Float64]

    def __init__(out self, prog: List[Instruction]):
        self.keys_0 = List[Float64]()
        self.keys_1 = List[Float64]()
        self.opcodes = List[Int]()
        self.args_re = List[Float64]()
        self.args_im = List[Float64]()
        for i in range(len(prog)):
            var addr = Float64(i)
            self.keys_0.append(2.0 * addr)
            self.keys_1.append(-(addr * addr))
            self.opcodes.append(prog[i].opcode)
            self.args_re.append(prog[i].val_re)
            self.args_im.append(prog[i].val_im)

    def fetch(self, ip: Int) -> Tuple[Int, Float64, Float64]:
        """Fetch instruction at ip via parabolic attention."""
        var q0 = Float64(ip)
        var q1: Float64 = 1.0
        var best_score = -1e308
        var best_idx = 0
        var n = len(self.opcodes)
        for i in range(n):
            var score = self.keys_0[i] * q0 + self.keys_1[i] * q1
            if score >= best_score:
                best_score = score
                best_idx = i
        return (self.opcodes[best_idx], self.args_re[best_idx], self.args_im[best_idx])


# ─── EML Executor ─────────────────────────────────────────────────

struct EMLExecutor:
    """Compiled transformer executor with 3-opcode ISA: PUSH, EML, HALT.

    Every memory access (instruction fetch + stack read/write) uses
    parabolic attention. The only compute operation is eml(x,y).
    """
    var trace_enabled: Bool

    def __init__(out self, trace: Bool = False):
        self.trace_enabled = trace

    def execute(self, prog: List[Instruction], max_steps: Int = 10000) -> Complex64:
        """Execute an EML program. Returns top of stack at HALT."""
        var pmem = ProgramMemory(prog)
        var stack = ParabolicStack()
        var ip = 0
        var steps = 0

        while steps < max_steps:
            # Instruction fetch via attention
            var fetched = pmem.fetch(ip)
            var opcode = fetched[0]
            var arg_re = fetched[1]
            var arg_im = fetched[2]

            if opcode == OP_HALT:
                if self.trace_enabled:
                    print("  HALT at step", steps)
                break

            if opcode == OP_PUSH:
                var val = Complex64(arg_re, arg_im)
                stack.push(val)
                if self.trace_enabled:
                    print("  PUSH", val, " sp=", stack.sp)

            elif opcode == OP_EML:
                # Pop two via attention, compute eml, push result
                var pair = stack.pop_two()
                var x = pair[0]
                var y = pair[1]
                var result = eml_op(x, y)
                stack.push(result)
                if self.trace_enabled:
                    print("  EML(", x, ",", y, ") =", result, " sp=", stack.sp)

            ip += 1
            steps += 1

        return stack.read_top()


# ─── Program Builders ─────────────────────────────────────────────

def push(val: Float64) -> Instruction:
    return Instruction(OP_PUSH, val)

def push_complex(re: Float64, im: Float64) -> Instruction:
    return Instruction(OP_PUSH, re, im)

def eml() -> Instruction:
    return Instruction(OP_EML)

def halt() -> Instruction:
    return Instruction(OP_HALT)


# ─── Programs from the paper ─────────────────────────────────────

def prog_e() -> List[Instruction]:
    """e = eml(1, 1) = exp(1) - ln(1) = e - 0 = e."""
    return [push(1), push(1), eml(), halt()]

def prog_exp(x: Float64) -> List[Instruction]:
    """exp(x) = eml(x, 1) = exp(x) - ln(1) = exp(x)."""
    return [push(x), push(1), eml(), halt()]

def prog_ln(x: Float64) -> List[Instruction]:
    """ln(x) = eml(1, eml(eml(1, x), 1)).
    RPN: 1 1 x EML 1 EML EML
    """
    return [push(1), push(1), push(x), eml(), push(1), eml(), eml(), halt()]

def prog_zero() -> List[Instruction]:
    """0 = eml(1, eml(1, 1)).
    RPN: 1 1 1 EML EML
    Since eml(1,1) = e, then eml(1, e) = exp(1) - ln(e) = e - 1... 
    
    Actually from the paper, 0 has K=7. Let me use:
    0 = ln(1) = eml(1, eml(eml(1, 1), 1))
    Wait, ln(1) = 0. So: 0 = ln(1).
    RPN for ln(1): 1 1 1 EML 1 EML EML
    """
    return [push(1), push(1), push(1), eml(), push(1), eml(), eml(), halt()]

def prog_subtract(x: Float64, y: Float64) -> List[Instruction]:
    """x - y via EML.
    
    From the paper's bootstrapping chain:
    First: exp(x) = eml(x, 1)
    Then: ln(x) = eml(1, eml(eml(1, x), 1))
    Then: x - y = eml(x, 1) - eml(y, 1) ... but we need to build subtraction
    from just EML.
    
    Actually, from the paper: x - y has K=11 in direct search.
    x - y = ln(exp(x) / exp(y)) = ln(exp(x-y))
    But we don't have division yet...
    
    The simplest route: x - y = ln(exp(x)/exp(y)).
    But we need exp and ln and division, which themselves need subtraction.
    
    The compiler route uses the bootstrapped chain. For this demo,
    we just verify the core operations.
    """
    # For the demo, compute exp(x) - exp(y) indirectly isn't trivial.
    # Instead, show that eml(x, 1) = exp(x) and build from there.
    # Direct: eml(x, eml(y, 1)) = exp(x) - ln(exp(y)) = exp(x) - y
    # That gives us exp(x) - y, not x - y.
    # 
    # For x - y we need: eml(ln(x), eml(ln(y), 1)) ... but that requires
    # the full compiler chain. Let's demonstrate what we can.
    #
    # Actually: x - y = eml(x, eml(y, 1)) when x replaces exp(x)... no.
    # eml(x, eml(y, 1)) = exp(x) - ln(exp(y)) = exp(x) - y
    #
    # From the paper's Table 4: x - y has K=11 (direct search).
    # The RPN code would be 11 tokens. Without the full compiler
    # chain we can't reconstruct this, but we CAN verify the building
    # blocks (exp, ln, e, π).
    
    # So let's just provide a stub that computes exp(x) - y as a demo
    # eml(x, eml(y, 1)) = exp(x) - ln(exp(y)) = exp(x) - y
    return [push(x), push(y), push(1), eml(), eml(), halt()]

def prog_neg_one() -> List[Instruction]:
    """-1 via EML. K=15 from direct search.
    From bootstrapping: -1 = 0 - 1 but we need subtraction...
    Actually from the paper: -x = complex route via ln(-1) = iπ etc.
    
    For demo: -1 = eml(0, eml(1,1)) = exp(0) - ln(e) = 1 - 1 = 0... no.
    
    Let's use: eml(ln(1), eml(1, 1)) = exp(0) - ln(e) = 1 - 1 = 0.
    
    Actually for -1 we need the full chain. Let's skip to what we can
    verify cleanly.
    """
    # Placeholder — needs full compiler chain
    return [push(1), halt()]

def prog_pi() -> List[Instruction]:
    """π = -i · ln(-1). Requires complex intermediates.
    
    From the paper's chain: once we have i and ln, we can get π.
    This requires the full bootstrapping. For demo, we show that
    eml operates correctly over complex numbers by computing
    eml(iπ, 1) = exp(iπ) = -1 (Euler's formula).
    """
    # Demonstrate Euler: push iπ, push 1, EML → should give -1
    var pi: Float64 = 3.141592653589793
    return [push_complex(0.0, pi), push(1), eml(), halt()]


# ─── Tests and Benchmarks ────────────────────────────────────────

def test_basic() raises:
    """Verify core EML programs against known values."""
    var exec = EMLExecutor(trace=True)

    print("=== Test: e = eml(1,1) ===")
    var result_e = exec.execute(prog_e())
    var expected_e: Float64 = 2.718281828459045
    var err_e = abs(result_e.re - expected_e)
    print("  Result:", result_e, " Expected:", expected_e, " Error:", err_e)
    if err_e > 1e-10:
        raise Error("e computation failed")
    print("  PASS")
    print()

    print("=== Test: exp(2) ===")
    var result_exp = exec.execute(prog_exp(2.0))
    var expected_exp = exp(Float64(2.0))
    var err_exp = abs(result_exp.re - expected_exp)
    print("  Result:", result_exp, " Expected:", expected_exp, " Error:", err_exp)
    if err_exp > 1e-10:
        raise Error("exp(2) computation failed")
    print("  PASS")
    print()

    print("=== Test: ln(5) ===")
    var result_ln = exec.execute(prog_ln(5.0))
    var expected_ln = log(Float64(5.0))
    var err_ln = abs(result_ln.re - expected_ln)
    print("  Result:", result_ln, " Expected:", expected_ln, " Error:", err_ln)
    if err_ln > 1e-10:
        raise Error("ln(5) computation failed")
    print("  PASS")
    print()

    print("=== Test: ln(0.5) ===")
    var result_ln2 = exec.execute(prog_ln(0.5))
    var expected_ln2 = log(Float64(0.5))
    var err_ln2 = abs(result_ln2.re - expected_ln2)
    print("  Result:", result_ln2, " Expected:", expected_ln2, " Error:", err_ln2)
    if err_ln2 > 1e-10:
        raise Error("ln(0.5) computation failed")
    print("  PASS")
    print()

    print("=== Test: 0 = ln(1) ===")
    var result_zero = exec.execute(prog_zero())
    var err_zero = abs(result_zero.re)
    print("  Result:", result_zero, " Expected: 0  Error:", err_zero)
    if err_zero > 1e-10:
        raise Error("zero computation failed")
    print("  PASS")
    print()

    print("=== Test: Euler's formula eml(iπ, 1) = exp(iπ) = -1 ===")
    var result_euler = exec.execute(prog_pi())
    var err_euler_re = abs(result_euler.re - (-1.0))
    var err_euler_im = abs(result_euler.im)
    print("  Result:", result_euler, " Expected: -1  Error re:", err_euler_re, " im:", err_euler_im)
    if err_euler_re > 1e-10 or err_euler_im > 1e-10:
        raise Error("Euler formula failed")
    print("  PASS")
    print()


def bench_eml(n_iters: Int):
    """Benchmark: execute ln(x) program n_iters times."""
    var exec = EMLExecutor(trace=False)
    var prog = prog_ln(7.389056)  # ln(e²) ≈ 2

    # Warmup
    for _ in range(100):
        _ = exec.execute(prog)

    var start = perf_counter_ns()
    for _ in range(n_iters):
        _ = exec.execute(prog)
    var elapsed = perf_counter_ns() - start

    var ns_per = elapsed // UInt(n_iters)
    var steps_per_exec = 8  # 7 instructions + halt
    var total_steps = UInt(n_iters) * UInt(steps_per_exec)
    var steps_per_sec = total_steps * 1_000_000_000 // elapsed

    print("=== Benchmark: ln(x) via EML ===")
    print("  Iterations:", n_iters)
    print("  Total ns:", elapsed)
    print("  ns/execution:", ns_per)
    print("  Steps/sec:", steps_per_sec)
    print("  (each execution = 7 instructions + HALT = 8 steps)")
    print("  (each step = 1 attention fetch + 0-1 attention reads + 1 EML op)")


def bench_exp_chain(n_iters: Int):
    """Benchmark: execute exp(x) n_iters times (simplest non-trivial program)."""
    var exec = EMLExecutor(trace=False)
    var prog = prog_exp(1.5)

    # Warmup
    for _ in range(100):
        _ = exec.execute(prog)

    var start = perf_counter_ns()
    for _ in range(n_iters):
        _ = exec.execute(prog)
    var elapsed = perf_counter_ns() - start

    var ns_per = elapsed // UInt(n_iters)
    print("=== Benchmark: exp(x) via EML ===")
    print("  Iterations:", n_iters)
    print("  ns/execution:", ns_per)


def main() raises:
    print("╔══════════════════════════════════════════════════╗")
    print("║  EML Executor — Compiled Transformer            ║")
    print("║  ISA: {PUSH, EML, HALT}                         ║")
    print("║  Memory: Parabolic attention (2D keys)          ║")
    print("║  Compute: eml(x,y) = exp(x) - ln(y) over ℂ    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    test_basic()

    print()
    bench_exp_chain(100_000)
    print()
    bench_eml(100_000)
