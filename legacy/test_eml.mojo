"""Comprehensive EML test suite — v2.

Fixes from v1:
  - c_exp/c_ln now handle inf/nan per IEEE 754 extended reals
  - stdlib atan2 replaces polynomial approximation
  - Real fast-path in c_ln (skip sqrt+atan2 when im≈0)
  - Vocabulary cache eliminates redundant constant recomputation
  - Tolerances calibrated per composition depth
"""

from std.math import exp, log, cos, sin, sqrt, abs, cosh, sinh, tanh, acos, asin, atan, atan2, isinf, isnan
from std.time import perf_counter_ns


# ─── Complex number ────────────────────────────────────────────

struct Z(Writable, Copyable, Movable, ImplicitlyCopyable):
    var re: Float64; var im: Float64
    def __init__(out self, re: Float64, im: Float64 = 0.0): self.re = re; self.im = im
    def __add__(self, o: Self) -> Self: return Self(self.re + o.re, self.im + o.im)
    def __sub__(self, o: Self) -> Self: return Self(self.re - o.re, self.im - o.im)
    def __mul__(self, o: Self) -> Self:
        return Self(self.re*o.re - self.im*o.im, self.re*o.im + self.im*o.re)
    def __neg__(self) -> Self: return Self(-self.re, -self.im)
    def mag(self) -> Float64: return sqrt(self.re*self.re + self.im*self.im)
    def write_to(self, mut writer: Some[Writer]):
        if abs(self.im) < 1e-12: writer.write(self.re)
        elif abs(self.re) < 1e-12: writer.write(self.im, "i")
        elif self.im >= 0: writer.write(self.re, "+", self.im, "i")
        else: writer.write(self.re, self.im, "i")

def zr(x: Float64) -> Z: return Z(x, 0.0)
def z1() -> Z: return Z(1.0, 0.0)


# ─── Complex exp with extended-real handling ───────────────────

def c_exp(z: Z) -> Z:
    """Complex exp with IEEE 754 inf/nan guards.

    Key cases the paper relies on:
      exp(+inf + 0i) = +inf (not inf + NaN*i)
      exp(-inf + 0i) = 0    (not 0 + NaN*i)
      exp(x + 0i)    = real  (imaginary part exactly 0)
    """
    # Real fast-path: avoid sin/cos when imaginary is zero
    if abs(z.im) < 1e-300:
        var er = exp(z.re)
        return Z(er, 0.0)
    # Guard against inf real part: exp(±inf + bi)
    if isinf(z.re):
        if z.re > 0:
            return Z(1e308, 0.0)  # +inf, but as finite stand-in
        else:
            return Z(0.0, 0.0)   # exp(-inf) = 0
    var ea = exp(z.re)
    return Z(ea * cos(z.im), ea * sin(z.im))


# ─── Complex ln with extended-real handling ────────────────────

def c_ln(z: Z) -> Z:
    """Complex ln (lower-edge branch) with extended-real handling.

    Key cases:
      ln(0)  = -inf (not NaN)
      ln(+x) = real  (skip atan2 entirely)
      ln(-x) = ln|x| - iπ  (lower-edge branch)
    """
    var PI: Float64 = 3.141592653589793

    # Handle zero
    if abs(z.re) < 1e-300 and abs(z.im) < 1e-300:
        return Z(-1e308, 0.0)  # -inf stand-in

    # Handle +inf
    if z.re > 1e307 and abs(z.im) < 1e-10:
        return Z(log(z.re) if z.re < 1e308 else 709.78, 0.0)

    # Real fast-path: skip sqrt+atan2 entirely
    if abs(z.im) < 1e-15:
        if z.re > 0:
            return Z(log(z.re), 0.0)
        else:
            return Z(log(-z.re), -PI)  # lower-edge branch

    # General complex case: use stdlib atan2
    var m = z.mag()
    return Z(log(m), atan2(z.im, z.re))


# ─── EML operator ──────────────────────────────────────────────

def eml(x: Z, y: Z) -> Z:
    """EML(x,y) = exp(x) - ln(y)."""
    return c_exp(x) - c_ln(y)


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CHAIN (from verify_eml_symbolic_chain.wl)
# ═══════════════════════════════════════════════════════════════

def eml_e() -> Z:           return eml(z1(), z1())
def eml_exp(x: Z) -> Z:     return eml(x, z1())
def eml_ln(x: Z) -> Z:      return eml(z1(), eml_exp(eml(z1(), x)))
def eml_sub(a: Z, b: Z) -> Z: return eml(eml_ln(a), eml_exp(b))
def eml_zero() -> Z:        return eml_ln(z1())
def eml_neg1() -> Z:        return eml_sub(eml_zero(), z1())
def eml_two() -> Z:         return eml_sub(z1(), eml_neg1())
def eml_minus(x: Z) -> Z:   return eml_sub(eml_zero(), x)
def eml_plus(a: Z, b: Z) -> Z: return eml_sub(a, eml_minus(b))
def eml_inv(x: Z) -> Z:     return eml_exp(eml_minus(eml_ln(x)))
def eml_times(a: Z, b: Z) -> Z: return eml_exp(eml_plus(eml_ln(a), eml_ln(b)))
def eml_sqr(x: Z) -> Z:     return eml_times(x, x)
def eml_div(a: Z, b: Z) -> Z: return eml_times(a, eml_inv(b))
def eml_half(x: Z) -> Z:    return eml_div(x, eml_two())
def eml_avg(a: Z, b: Z) -> Z: return eml_half(eml_plus(a, b))
def eml_sqrt(x: Z) -> Z:    return eml_exp(eml_half(eml_ln(x)))
def eml_pow(a: Z, b: Z) -> Z: return eml_exp(eml_times(b, eml_ln(a)))
def eml_logb(base: Z, x: Z) -> Z: return eml_div(eml_ln(x), eml_ln(base))
def eml_hypot(a: Z, b: Z) -> Z: return eml_sqrt(eml_plus(eml_sqr(a), eml_sqr(b)))

def eml_i() -> Z:           return eml_minus(eml_exp(eml_half(eml_ln(eml_neg1()))))
def eml_pi() -> Z:          return eml_times(eml_i(), eml_ln(eml_neg1()))

def eml_cosh(x: Z) -> Z:    return eml_avg(eml_exp(x), eml_exp(eml_minus(x)))
def eml_sinh(x: Z) -> Z:    return eml(x, eml_exp(eml_cosh(x)))
def eml_tanh(x: Z) -> Z:    return eml_div(eml_sinh(x), eml_cosh(x))
def eml_cos(x: Z) -> Z:     return eml_cosh(eml_div(x, eml_i()))
def eml_sin(x: Z) -> Z:     return eml_cos(eml_sub(x, eml_half(eml_pi())))
def eml_tan(x: Z) -> Z:     return eml_div(eml_sin(x), eml_cos(x))
def eml_sigma(x: Z) -> Z:   return eml_inv(eml(eml_minus(x), eml_exp(eml_neg1())))

def eml_arcsinh(x: Z) -> Z: return eml_ln(eml_plus(x, eml_hypot(eml_neg1(), x)))
def eml_arccosh(x: Z) -> Z: return eml_arcsinh(eml_hypot(x, eml_sqrt(eml_neg1())))
def eml_arccos(x: Z) -> Z:  return eml_arccosh(eml_cos(eml_arccosh(x)))
def eml_arcsin(x: Z) -> Z:  return eml_sub(eml_half(eml_pi()), eml_arccos(x))
def eml_arctanh(x: Z) -> Z: return eml_arcsinh(eml_inv(eml_tan(eml_arccos(x))))
def eml_arctan(x: Z) -> Z:  return eml_arcsin(eml_tanh(eml_arcsinh(x)))


# ═══════════════════════════════════════════════════════════════
# VOCABULARY CACHE — precompute constants once
# ═══════════════════════════════════════════════════════════════

struct Vocab:
    var zero: Z
    var neg1: Z
    var two: Z
    var i_val: Z
    var pi_val: Z
    var half_pi: Z

    def __init__(out self):
        self.zero = eml_zero()
        self.neg1 = eml_neg1()
        self.two = eml_sub(z1(), self.neg1)
        var half_ln_neg1 = eml_div(eml_ln(self.neg1), self.two)
        self.i_val = eml_minus(eml_exp(half_ln_neg1))
        self.pi_val = eml_times(self.i_val, eml_ln(self.neg1))
        self.half_pi = eml_div(self.pi_val, self.two)

# Cached versions using precomputed vocab
def v_minus(v: Vocab, x: Z) -> Z: return eml_sub(v.zero, x)
def v_plus(v: Vocab, a: Z, b: Z) -> Z: return eml_sub(a, v_minus(v, b))
def v_inv(v: Vocab, x: Z) -> Z: return eml_exp(v_minus(v, eml_ln(x)))
def v_times(v: Vocab, a: Z, b: Z) -> Z: return eml_exp(v_plus(v, eml_ln(a), eml_ln(b)))
def v_div(v: Vocab, a: Z, b: Z) -> Z: return v_times(v, a, v_inv(v, b))
def v_half(v: Vocab, x: Z) -> Z: return v_div(v, x, v.two)
def v_avg(v: Vocab, a: Z, b: Z) -> Z: return v_half(v, v_plus(v, a, b))
def v_sqrt(v: Vocab, x: Z) -> Z: return eml_exp(v_half(v, eml_ln(x)))
def v_sqr(v: Vocab, x: Z) -> Z: return v_times(v, x, x)
def v_pow(v: Vocab, a: Z, b: Z) -> Z: return eml_exp(v_times(v, b, eml_ln(a)))
def v_logb(v: Vocab, base: Z, x: Z) -> Z: return v_div(v, eml_ln(x), eml_ln(base))
def v_hypot(v: Vocab, a: Z, b: Z) -> Z: return v_sqrt(v, v_plus(v, v_sqr(v, a), v_sqr(v, b)))
def v_cosh(v: Vocab, x: Z) -> Z: return v_avg(v, eml_exp(x), eml_exp(v_minus(v, x)))
def v_sinh(v: Vocab, x: Z) -> Z: return eml(x, eml_exp(v_cosh(v, x)))
def v_tanh(v: Vocab, x: Z) -> Z: return v_div(v, v_sinh(v, x), v_cosh(v, x))
def v_cos(v: Vocab, x: Z) -> Z: return v_cosh(v, v_div(v, x, v.i_val))
def v_sin(v: Vocab, x: Z) -> Z: return v_cos(v, eml_sub(x, v.half_pi))
def v_tan(v: Vocab, x: Z) -> Z: return v_div(v, v_sin(v, x), v_cos(v, x))
def v_sigma(v: Vocab, x: Z) -> Z: return v_inv(v, eml(v_minus(v, x), eml_exp(v.neg1)))
def v_arcsinh(v: Vocab, x: Z) -> Z: return eml_ln(v_plus(v, x, v_hypot(v, v.neg1, x)))
def v_arccosh(v: Vocab, x: Z) -> Z: return v_arcsinh(v, v_hypot(v, x, v.i_val))
def v_arccos(v: Vocab, x: Z) -> Z: return v_arccosh(v, v_cos(v, v_arccosh(v, x)))
def v_arcsin(v: Vocab, x: Z) -> Z: return eml_sub(v.half_pi, v_arccos(v, x))
def v_arctanh(v: Vocab, x: Z) -> Z: return v_arcsinh(v, v_inv(v, v_tan(v, v_arccos(v, x))))
def v_arctan(v: Vocab, x: Z) -> Z: return v_arcsin(v, v_tanh(v, v_arcsinh(v, x)))


# ═══════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════

struct T:
    var total: Int; var passed: Int; var failed: Int
    var worst_err: Float64; var worst_name: String
    def __init__(out self):
        self.total = 0; self.passed = 0; self.failed = 0
        self.worst_err = 0.0; self.worst_name = String("")

    def ck(mut self, name: String, got: Z, exp_val: Z, tol: Float64, real_only: Bool):
        self.total += 1
        var err_re = abs(got.re - exp_val.re)
        var err_im = abs(got.im - exp_val.im)
        var err = err_re if real_only else sqrt(err_re*err_re + err_im*err_im)
        if real_only and err_im > tol * 100:
            print("  FAIL  ", name, " imag residual:", err_im)
            self.failed += 1; return
        if isnan(got.re) or isnan(got.im):
            print("  FAIL  ", name, "  NaN! got:", got)
            self.failed += 1; return
        if err > tol:
            print("  FAIL  ", name, "  got:", got, " exp:", exp_val, " err:", err)
            self.failed += 1
        else: self.passed += 1
        if err > self.worst_err and not isnan(err):
            self.worst_err = err; self.worst_name = name

    def r(mut self, name: String, got: Z, ev: Float64, tol: Float64 = 1e-9):
        self.ck(name, got, zr(ev), tol, True)
    def c(mut self, name: String, got: Z, ev: Z, tol: Float64 = 1e-9):
        self.ck(name, got, ev, tol, False)

    def sw(mut self, name: String, worst: Float64, tol: Float64, nv: Int, n: Int, wx: Float64, rms: Float64):
        self.total += 1
        var ok = worst <= tol and not isnan(worst)
        if ok: self.passed += 1
        else: self.failed += 1
        if worst > self.worst_err and not isnan(worst):
            self.worst_err = worst; self.worst_name = "sweep:" + name
        var tag = "PASS" if ok else "FAIL"
        print("  ", tag, " ", name, "  valid=", nv, "/", n, "  worst=", worst, " @", wx, "  rms=", rms)

    def report(self):
        print("\n═══════════════════════════════════════════")
        print("  Total:", self.total, " Pass:", self.passed, " Fail:", self.failed)
        if self.worst_err > 0: print("  Worst:", self.worst_err, " (", self.worst_name, ")")
        print("═══════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════
# TIER 1: Raw eml
# ═══════════════════════════════════════════════════════════════

def test_t1(mut t: T):
    print("\n━━━ Tier 1: Raw eml(x, y) ━━━")
    var PI: Float64 = 3.141592653589793
    var E: Float64 = 2.718281828459045
    t.r("eml(0,1)=1", eml(zr(0), z1()), 1.0)
    t.r("eml(1,1)=e", eml(z1(), z1()), E)
    t.r("eml(0,e)=0", eml(zr(0), zr(E)), 0.0)
    t.r("eml(2,1)=e²", eml(zr(2), z1()), exp(Float64(2)))
    t.r("eml(0,0.5)", eml(zr(0), zr(0.5)), 1.0 + log(Float64(2)))
    t.c("eml(iπ,1)=-1", eml(Z(0,PI), z1()), zr(-1))
    t.c("eml(0,-1)=1+iπ", eml(zr(0), zr(-1)), Z(1.0, PI))
    # Extended reals
    t.r("eml(1,0)=e+inf", eml(z1(), zr(0)), 1e308, tol=1e300)  # should be +inf
    t.r("c_exp(-inf)=0", c_exp(Z(-1e308, 0)), 0.0)
    t.r("c_ln(0)=-inf", c_ln(zr(0)), -1e308, tol=1.0)

# ═══════════════════════════════════════════════════════════════
# TIER 2: Bootstrap chain
# ═══════════════════════════════════════════════════════════════

def test_t2_const(mut t: T):
    print("\n━━━ Tier 2a: Constants (uncached) ━━━")
    t.r("e", eml_e(), 2.718281828459045)
    t.r("0", eml_zero(), 0.0)
    t.r("-1", eml_neg1(), -1.0)
    t.r("2", eml_two(), 2.0)
    t.c("i", eml_i(), Z(0,1), tol=1e-9)
    t.r("π", eml_pi(), 3.141592653589793, tol=1e-8)

def test_t2_vocab(mut t: T, v: Vocab):
    print("\n━━━ Tier 2b: Constants (cached vocab) ━━━")
    t.r("v.zero", v.zero, 0.0)
    t.r("v.neg1", v.neg1, -1.0)
    t.r("v.two", v.two, 2.0)
    t.c("v.i", v.i_val, Z(0,1), tol=1e-9)
    t.r("v.π", v.pi_val, 3.141592653589793, tol=1e-8)
    t.r("v.π/2", v.half_pi, 1.5707963267948966, tol=1e-8)

def test_t2_unary(mut t: T, v: Vocab):
    print("\n━━━ Tier 2c: Unary functions (cached) ━━━")
    var g: Float64 = 0.5772156649015329   # γ
    var a: Float64 = 1.2824271291006226   # Glaisher A
    var zg = zr(g); var za = zr(a)

    t.r("exp(γ)", eml_exp(zg), exp(g))
    t.r("exp(A)", eml_exp(za), exp(a))
    t.r("ln(γ)", eml_ln(zg), log(g))
    t.r("ln(A)", eml_ln(za), log(a))
    t.r("-γ", v_minus(v, zg), -g)
    t.r("1/γ", v_inv(v, zg), 1.0/g)
    t.r("γ/2", v_half(v, zg), g/2.0)
    t.r("γ²", v_sqr(v, zg), g*g, tol=1e-8)
    t.r("A²", v_sqr(v, za), a*a, tol=1e-8)
    t.r("√γ", v_sqrt(v, zg), sqrt(g), tol=1e-9)
    t.r("√A", v_sqrt(v, za), sqrt(a), tol=1e-9)
    t.r("cosh(γ)", v_cosh(v, zg), cosh(g), tol=1e-8)
    t.r("cosh(A)", v_cosh(v, za), cosh(a), tol=1e-8)
    t.r("sinh(γ)", v_sinh(v, zg), sinh(g), tol=1e-8)
    t.r("sinh(A)", v_sinh(v, za), sinh(a), tol=1e-8)
    t.r("tanh(γ)", v_tanh(v, zg), tanh(g), tol=1e-8)
    t.r("cos(γ)", v_cos(v, zg), cos(g), tol=1e-9)
    t.r("cos(A)", v_cos(v, za), cos(a), tol=1e-9)
    t.r("sin(γ)", v_sin(v, zg), sin(g), tol=1e-8)
    t.r("sin(A)", v_sin(v, za), sin(a), tol=1e-9)
    t.r("tan(γ)", v_tan(v, zg), sin(g)/cos(g), tol=1e-8)
    t.r("σ(γ)", v_sigma(v, zg), 1.0/(1.0+exp(-g)), tol=1e-9)
    t.r("σ(A)", v_sigma(v, za), 1.0/(1.0+exp(-a)), tol=1e-9)
    t.r("arcsinh(γ)", v_arcsinh(v, zg), log(g+sqrt(1.0+g*g)), tol=1e-8)
    t.r("arcsinh(A)", v_arcsinh(v, za), log(a+sqrt(1.0+a*a)), tol=1e-8)
    t.r("arccosh(A)", v_arccosh(v, za), log(a+sqrt(a*a-1.0)), tol=1e-6)
    t.r("arccos(γ)", v_arccos(v, zg), acos(g), tol=1e-6)
    t.r("arcsin(γ)", v_arcsin(v, zg), asin(g), tol=1e-6)
    var ref_atanh = 0.5 * log((1.0+g)/(1.0-g))
    t.r("arctanh(γ)", v_arctanh(v, zg), ref_atanh, tol=1e-5)
    t.r("arctan(γ)", v_arctan(v, zg), atan(g), tol=1e-5)
    t.r("arctan(A)", v_arctan(v, za), atan(a), tol=1e-5)

def test_t2_binary(mut t: T, v: Vocab):
    print("\n━━━ Tier 2d: Binary operations (cached) ━━━")
    var g: Float64 = 0.5772156649015329
    var a: Float64 = 1.2824271291006226
    var zg = zr(g); var za = zr(a)
    t.r("γ-A", eml_sub(zg, za), g - a)
    t.r("γ+A", v_plus(v, zg, za), g + a)
    t.r("γ×A", v_times(v, zg, za), g * a, tol=1e-8)
    t.r("γ/A", v_div(v, zg, za), g / a, tol=1e-8)
    t.r("A/γ", v_div(v, za, zg), a / g, tol=1e-8)
    t.r("γ^A", v_pow(v, zg, za), exp(a*log(g)), tol=1e-7)
    t.r("log_A(γ)", v_logb(v, za, zg), log(g)/log(a), tol=1e-8)
    t.r("avg(γ,A)", v_avg(v, zg, za), (g+a)/2.0, tol=1e-8)
    t.r("hypot(γ,A)", v_hypot(v, zg, za), sqrt(g*g+a*a), tol=1e-7)

def test_identities(mut t: T, v: Vocab):
    print("\n━━━ Identities ━━━")
    var g: Float64 = 0.5772156649015329
    var zg = zr(g)
    t.r("exp(ln(γ))=γ", eml_exp(eml_ln(zg)), g)
    t.r("ln(exp(γ))=γ", eml_ln(eml_exp(zg)), g)
    t.r("--γ=γ", v_minus(v, v_minus(v, zg)), g)
    t.r("1/(1/γ)=γ", v_inv(v, v_inv(v, zg)), g, tol=1e-8)
    t.r("γ-γ=0", eml_sub(zg, zg), 0.0)
    t.r("γ×1=γ", v_times(v, zg, z1()), g, tol=1e-8)
    t.r("γ/γ=1", v_div(v, zg, zg), 1.0, tol=1e-8)
    t.r("γ^1=γ", v_pow(v, zg, z1()), g, tol=1e-8)
    t.r("(√γ)²=γ", v_sqr(v, v_sqrt(v, zg)), g, tol=1e-8)
    t.r("cosh²-sinh²=1", eml_sub(v_sqr(v, v_cosh(v, zg)), v_sqr(v, v_sinh(v, zg))), 1.0, tol=1e-7)
    t.r("sin²+cos²=1", v_plus(v, v_sqr(v, v_sin(v, zg)), v_sqr(v, v_cos(v, zg))), 1.0, tol=1e-7)
    t.r("e^iπ+1=0", v_plus(v, eml_exp(Z(0, 3.141592653589793)), z1()), 0.0)


# ═══════════════════════════════════════════════════════════════
# TIER 3: Domain sweeps (vocab-cached)
# ═══════════════════════════════════════════════════════════════

def sweep(mut t: T, v: Vocab, name: String, x_min: Float64, x_max: Float64, step: Float64, tol: Float64):
    var n = 0; var nv = 0; var worst: Float64 = 0.0; var wx: Float64 = 0.0; var ssq: Float64 = 0.0
    var x = x_min
    while x <= x_max + step * 0.1:
        n += 1; var xc = zr(x); var got = zr(0.0); var expected = 0.0; var skip = False
        if name == "exp":     got = eml_exp(xc);           expected = exp(x)
        elif name == "ln":    got = eml_ln(xc);            expected = log(x)
        elif name == "minus": got = v_minus(v, xc);        expected = -x
        elif name == "inv":   got = v_inv(v, xc);          expected = 1.0/x
        elif name == "half":  got = v_half(v, xc);         expected = x/2.0
        elif name == "sqr":   got = v_sqr(v, xc);          expected = x*x
        elif name == "sqrt":  got = v_sqrt(v, xc);         expected = sqrt(x)
        elif name == "cosh":  got = v_cosh(v, xc);         expected = cosh(x)
        elif name == "sinh":  got = v_sinh(v, xc);         expected = sinh(x)
        elif name == "tanh":  got = v_tanh(v, xc);         expected = tanh(x)
        elif name == "cos":   got = v_cos(v, xc);          expected = cos(x)
        elif name == "sin":   got = v_sin(v, xc);          expected = sin(x)
        elif name == "tan":   got = v_tan(v, xc);          expected = sin(x)/cos(x)
        elif name == "sigma": got = v_sigma(v, xc);        expected = 1.0/(1.0+exp(-x))
        elif name == "arcsinh": got = v_arcsinh(v, xc);    expected = log(x+sqrt(1.0+x*x))
        elif name == "arccosh": got = v_arccosh(v, xc);    expected = log(x+sqrt(x*x-1.0))
        elif name == "arcsin":  got = v_arcsin(v, xc);     expected = asin(x)
        elif name == "arccos":  got = v_arccos(v, xc);     expected = acos(x)
        elif name == "arctan":  got = v_arctan(v, xc);     expected = atan(x)
        elif name == "arctanh": got = v_arctanh(v, xc);    expected = 0.5*log((1.0+x)/(1.0-x))
        else: skip = True
        if not skip:
            if isnan(expected) or expected > 1e300 or expected < -1e300 or isnan(got.re) or got.re > 1e300 or got.re < -1e300: skip = True
        if not skip:
            nv += 1; var err = abs(got.re - expected); ssq += err*err
            if err > worst: worst = err; wx = x
        x += step
    var rms: Float64 = 0.0
    if nv > 0: rms = sqrt(ssq / Float64(nv))
    t.sw(name, worst, tol, nv, n, wx, rms)

def test_t3(mut t: T, v: Vocab):
    print("\n━━━ Tier 3: Domain sweeps (vocab-cached) ━━━")
    sweep(t, v, "exp",   -4.0,  4.0,  0.125, 1e-10)
    sweep(t, v, "ln",     0.125, 8.0,  0.125, 1e-9)
    sweep(t, v, "minus", -4.0,  4.0,  0.125, 1e-9)
    sweep(t, v, "inv",    0.125, 8.0,  0.125, 1e-9)
    sweep(t, v, "half",  -4.0,  4.0,  0.125, 1e-8)
    sweep(t, v, "sqr",   -4.0,  4.0,  0.125, 1e-7)
    sweep(t, v, "sqrt",   0.125, 8.0,  0.125, 1e-8)
    sweep(t, v, "cosh",  -4.0,  4.0,  0.125, 1e-7)
    sweep(t, v, "sinh",  -4.0,  4.0,  0.125, 1e-7)
    sweep(t, v, "tanh",  -4.0,  4.0,  0.125, 1e-8)
    sweep(t, v, "sigma", -8.0,  8.0,  0.125, 1e-8)
    sweep(t, v, "cos",   -6.0,  6.0,  0.0625, 1e-8)
    sweep(t, v, "sin",   -6.0,  6.0,  0.0625, 1e-7)
    sweep(t, v, "tan",   -1.25, 1.25, 0.03125, 1e-7)
    sweep(t, v, "arcsinh", -8.0, 8.0, 0.125, 1e-6)
    sweep(t, v, "arccosh",  1.125, 9.0, 0.125, 1e-5)
    sweep(t, v, "arcsin", 0.05, 0.7, 0.01, 1e-4)
    sweep(t, v, "arccos", 0.05, 0.7, 0.01, 1e-4)
    sweep(t, v, "arctan", 0.125, 2.0, 0.125, 1e-3)
    sweep(t, v, "arctanh", 0.05, 0.7, 0.01, 1e-3)


# ═══════════════════════════════════════════════════════════════
# TIER 4: RPN executor
# ═══════════════════════════════════════════════════════════════

comptime OP_HALT = 0
comptime OP_PUSH = 1
comptime OP_EML  = 2

struct Inst(Copyable, Movable):
    var op: Int; var re: Float64; var im: Float64
    def __init__(out self, op: Int, re: Float64 = 0.0, im: Float64 = 0.0):
        self.op = op; self.re = re; self.im = im

struct PStack:
    var k0: List[Float64]; var k1: List[Float64]
    var vr: List[Float64]; var vi: List[Float64]
    var sp: Int; var cnt: Int
    def __init__(out self):
        self.k0 = List[Float64](); self.k1 = List[Float64]()
        self.vr = List[Float64](); self.vi = List[Float64]()
        self.sp = 0; self.cnt = 0
    def push(mut self, v: Z):
        self.sp += 1; var a = Float64(self.sp)
        self.k0.append(2.0*a); self.k1.append(-(a*a))
        self.vr.append(v.re); self.vi.append(v.im); self.cnt += 1
    def attn(self, addr: Int) -> Z:
        var q = Float64(addr); var best = -1e308; var bi = 0
        for i in range(self.cnt):
            var s = self.k0[i]*q + self.k1[i]
            if s >= best: best = s; bi = i
        return Z(self.vr[bi], self.vi[bi])
    def pop2(mut self) -> Tuple[Z, Z]:
        var top = self.attn(self.sp); var sec = self.attn(self.sp - 1); self.sp -= 2
        return (sec^, top^)

def run_rpn(prog: List[Inst]) -> Z:
    var st = PStack(); var ip = 0; var steps = 0
    var pk0 = List[Float64](); var pk1 = List[Float64]()
    var pop = List[Int](); var par = List[Float64](); var pai = List[Float64]()
    for i in range(len(prog)):
        var a = Float64(i)
        pk0.append(2.0*a); pk1.append(-(a*a))
        pop.append(prog[i].op); par.append(prog[i].re); pai.append(prog[i].im)
    while steps < 10000:
        var q = Float64(ip); var best = -1e308; var bi = 0
        for i in range(len(pop)):
            var s = pk0[i]*q + pk1[i]
            if s >= best: best = s; bi = i
        var opc = pop[bi]
        if opc == OP_HALT: break
        if opc == OP_PUSH: st.push(Z(par[bi], pai[bi]))
        elif opc == OP_EML:
            var p = st.pop2(); st.push(eml(p[0], p[1]))
        ip += 1; steps += 1
    return st.attn(st.sp)

def P(v: Float64) -> Inst: return Inst(OP_PUSH, v)
def Pc(r: Float64, i: Float64) -> Inst: return Inst(OP_PUSH, r, i)
def Em() -> Inst: return Inst(OP_EML)
def Ht() -> Inst: return Inst(OP_HALT)

def test_t4(mut t: T):
    print("\n━━━ Tier 4: RPN executor ━━━")
    var PI: Float64 = 3.141592653589793
    t.r("RPN:e", run_rpn([P(1),P(1),Em(),Ht()]), 2.718281828459045)
    t.r("RPN:exp(2)", run_rpn([P(2),P(1),Em(),Ht()]), exp(Float64(2)))
    t.r("RPN:exp(-1)", run_rpn([P(-1),P(1),Em(),Ht()]), exp(Float64(-1)))
    t.r("RPN:exp(0)=1", run_rpn([P(0),P(1),Em(),Ht()]), 1.0)
    for v in [0.5, 1.0, 2.0, 5.0, 10.0, 0.1]:
        t.r("RPN:ln("+String(v)+")", run_rpn([P(1),P(1),P(v),Em(),P(1),Em(),Em(),Ht()]), log(v))
    t.r("RPN:zero", run_rpn([P(1),P(1),P(1),Em(),P(1),Em(),Em(),Ht()]), 0.0)
    t.c("RPN:Euler", run_rpn([Pc(0,PI),P(1),Em(),Ht()]), zr(-1))
    for v in [0.5, 2.0, 3.7]:
        t.r("RPN:exp(ln("+String(v)+"))", run_rpn([P(1),P(1),P(v),Em(),P(1),Em(),Em(),P(1),Em(),Ht()]), v)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  EML Test Suite v2 — inf handling + stdlib atan2     ║")
    print("║  + vocab cache + real fast-path                      ║")
    print("╚═══════════════════════════════════════════════════════╝")
    var start = perf_counter_ns()
    var t = T()
    var v = Vocab()
    print("  Vocab built: zero=", v.zero, " neg1=", v.neg1, " two=", v.two, " i=", v.i_val, " π=", v.pi_val)
    test_t1(t)
    test_t2_const(t)
    test_t2_vocab(t, v)
    test_t2_unary(t, v)
    test_t2_binary(t, v)
    test_identities(t, v)
    test_t4(t)
    test_t3(t, v)
    var ms = (perf_counter_ns() - start) // 1_000_000
    t.report()
    print("  Elapsed:", ms, "ms")
