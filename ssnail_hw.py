"""SSNAIL hardware numerics: exactly what the datapath computes.

The emulator uses these in hardware-numerics mode (Machine.hw = True); the
VHDL is checked against them.  See ssnail-numerics.md for the decisions.
Every function spells out its arithmetic order, because F32 rounding makes
the order part of the result.

Conventions: all intermediate values are IEEE single (np.float32) with
round-to-nearest-even; "sequential sum" means acc = acc + x[i] for i in
order, starting from +0, rounding after every addition.
"""

import numpy as np

F32 = np.float32


def seq_sum(v):
    """Sequential F32 sum (numpy's accumulate is sequential)."""
    v = np.asarray(v, dtype=F32)
    return F32(0) if v.size == 0 else np.add.accumulate(v, dtype=F32)[-1]


def recip(x):
    """1/x, correctly rounded (the hardware's iterative unit: within 1 ulp)."""
    return F32(1.0 / float(x))


def rsqrt(x):
    """1/sqrt(x), correctly rounded (hardware: within 1 ulp)."""
    return F32(1.0 / np.sqrt(float(x)))


# --- Q8_0 quantisation of the GEMV input, as llama.cpp quantize_row_q8_0 ------
def quantize_q8_0(x):
    """x (F32, length a multiple of 32) -> (d as F16 per block, int8 q).
    llama.cpp: d = amax/127; id = d ? 1/d : 0; q = roundf(x*id) (half away
    from zero); the stored scale is d rounded to F16."""
    x = np.asarray(x, dtype=F32).reshape(-1, 32)
    amax = np.max(np.abs(x), axis=1).astype(F32)
    d = (amax / F32(127)).astype(F32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, F32(1) / np.where(d != 0, d, F32(1)), F32(0)).astype(F32)
    xi = (x * idv[:, None]).astype(F32)
    q = np.sign(xi) * np.floor(np.abs(xi) + F32(0.5))          # roundf
    return d.astype(np.float16), q.astype(np.int8)


# --- GEMV ----------------------------------------------------------------------
def gemv_quant(raw_blocks, fmt, rows, cols, x):
    """Quantised weights (Q8_0 = 8, Q4_0 = 2) against a Q8_0-quantised x.
    Per row: acc = 0; for each block b in order:
        isum = sum over the block of q_w * q_x           (exact integer)
        acc  = acc + F32(isum) * (F32(d_w) * F32(d_x))   (each op rounded)
    as llama.cpp's ggml_vec_dot_q8_0_q8_0 / q4_0_q8_0."""
    dx, qx = quantize_q8_0(x)
    nb = cols // 32
    if fmt == 8:
        blk = np.frombuffer(raw_blocks, dtype=np.uint8).reshape(rows, nb, 34)
        dw = blk[:, :, 0:2].copy().view(np.float16)[..., 0]
        qw = blk[:, :, 2:].view(np.int8).astype(np.int32)
    elif fmt == 2:
        blk = np.frombuffer(raw_blocks, dtype=np.uint8).reshape(rows, nb, 18)
        dw = blk[:, :, 0:2].copy().view(np.float16)[..., 0]
        nib = blk[:, :, 2:]
        lo = (nib & 0x0F).astype(np.int32) - 8                 # elements 0..15
        hi = (nib >> 4).astype(np.int32) - 8                   # elements 16..31
        qw = np.concatenate([lo, hi], axis=2)
    else:
        raise ValueError(fmt)
    isum = np.einsum("rbk,bk->rb", qw, qx.astype(np.int32))    # exact
    scale = (dw.astype(F32) * dx.astype(F32)[None, :]).astype(F32)
    terms = (isum.astype(F32) * scale).astype(F32)
    return np.add.accumulate(terms, axis=1, dtype=F32)[:, -1]


def gemv_float(w, x):
    """Float weights: products into a wide exact accumulator, rounded once to
    F32 at the end of each row.  (Emulated as an exactly rounded F64 sum,
    then F32: identical except in vanishingly rare double-rounding ties.)"""
    import math
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=F32).astype(np.float64)
    prods = w * x[None, :]                                     # exact in F64
    return np.array([F32(math.fsum(r)) for r in prods], dtype=F32)


# --- Function tables -------------------------------------------------------------
SEGMENTS = 256


class Table:
    """Piecewise-linear f(x) over [lo, hi): y = c0[i] + c1[i] * x (each op
    rounded), segment i = floor((x - lo) * (SEGMENTS / (hi - lo))).
    Below lo: below(x); at or above hi: above(x).  Coefficients are the
    chords through each segment's end points, rounded to F32; the hardware
    holds this exact table."""

    def __init__(self, f, lo, hi, below, above):
        self.lo, self.hi, self.below, self.above = F32(lo), F32(hi), below, above
        xs = np.linspace(lo, hi, SEGMENTS + 1, dtype=np.float64)
        ys = f(xs)
        c1 = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        c0 = ys[:-1] - c1 * xs[:-1]
        self.c0, self.c1 = c0.astype(F32), c1.astype(F32)
        self.k = F32(SEGMENTS / (hi - lo))

    def __call__(self, x):
        x = np.asarray(x, dtype=F32)
        i = np.floor(((x - self.lo).astype(F32) * self.k).astype(F32)).astype(np.int64)
        inside = (x >= self.lo) & (x < self.hi)
        ic = np.clip(i, 0, SEGMENTS - 1)
        y = (self.c0[ic] + (self.c1[ic] * x).astype(F32)).astype(F32)
        y = np.where(inside, y, np.where(x < self.lo, self.below(x), self.above(x)))
        return y.astype(F32)


_silu = lambda x: x / (1 + np.exp(-x))
_gelu = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
SILU = Table(_silu, -12.0, 12.0, lambda x: np.zeros_like(x), lambda x: x)
GELU = Table(_gelu, -6.0, 6.0, lambda x: np.zeros_like(x), lambda x: x)
EXP = Table(np.exp, -16.0, 0.0, lambda x: np.zeros_like(x), lambda x: np.ones_like(x))


# --- Vector operations -------------------------------------------------------------
def rmsnorm(x, g, eps):
    """ss = seq_sum(x*x); m = ss * recip(N); r = rsqrt(m + eps); y = (x*r)*g."""
    x = np.asarray(x, dtype=F32)
    ss = seq_sum((x * x).astype(F32))
    m = F32(ss * recip(F32(x.size)))
    r = rsqrt(F32(m + F32(eps)))
    return ((x * r).astype(F32) * np.asarray(g, dtype=F32)).astype(F32)


def layernorm(x, g, b, eps):
    """mu = seq_sum(x) * recip(N); d = x - mu; var = seq_sum(d*d) * recip(N);
    r = rsqrt(var + eps); y = ((d*r)*g) + b."""
    x = np.asarray(x, dtype=F32)
    rn = recip(F32(x.size))
    mu = F32(seq_sum(x) * rn)
    d = (x - mu).astype(F32)
    var = F32(seq_sum((d * d).astype(F32)) * rn)
    r = rsqrt(F32(var + F32(eps)))
    return ((((d * r).astype(F32)) * np.asarray(g, dtype=F32)).astype(F32)
            + np.asarray(b, dtype=F32)).astype(F32)


def silumul(x, y):
    return (SILU(x) * np.asarray(y, dtype=F32)).astype(F32)


def gelu(x):
    return GELU(x)


def meanrows(M):
    """Column-wise sequential sums over the rows, times recip(rows)."""
    M = np.asarray(M, dtype=F32)
    s = np.add.accumulate(M, axis=0, dtype=F32)[-1]
    return (s * recip(F32(M.shape[0]))).astype(F32)


def attention_head(q, K, V):
    """One head.  q: (hd,) F32; K, V: (T, hd) as stored (F16 -> F32 exact).
    s_t = seq_sum(q * k_t) * recip(sqrt(hd))   [scale = rsqrt(hd)]
    m = max(s); e_t = EXP(s_t - m); z = seq_sum(e); w_t = e_t * recip(z)
    out = sequential over t of out + w_t * v_t."""
    q = np.asarray(q, dtype=F32)
    K = np.asarray(K, dtype=F32)
    V = np.asarray(V, dtype=F32)
    prods = (K * q[None, :]).astype(F32)
    s = np.add.accumulate(prods, axis=1, dtype=F32)[:, -1]
    s = (s * rsqrt(F32(q.size))).astype(F32)
    e = EXP((s - s.max()).astype(F32))
    w = (e * recip(seq_sum(e))).astype(F32)
    out = np.zeros(q.size, dtype=F32)
    for t in range(K.shape[0]):
        out = (out + (w[t] * V[t]).astype(F32)).astype(F32)
    return out
