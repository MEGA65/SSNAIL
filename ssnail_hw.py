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
    """1 / sqrt(x) as two correctly rounded steps: F32(1 / F32(sqrt(x))).
    (Within 1 ulp of an exactly rounded rsqrt; matches the hardware, which
    has a square root and a divider, bit for bit.)"""
    return recip(np.sqrt(F32(x), dtype=F32))


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
    return d.astype(np.float16), roundf(xi).astype(np.int8)


def roundf(v):
    """C roundf (half away from zero), exactly.  Not floor(|v| + 0.5) in
    F32: that addition can round (0.49999997 + 0.5 -> 1.0).  Every step here
    is exact in F32, as in gguf-py."""
    v = np.asarray(v, dtype=F32)
    a = np.abs(v)
    fl = np.floor(a)
    return (np.sign(v) * (fl + np.floor(F32(2) * (a - fl)))).astype(F32)


# --- GEMV ----------------------------------------------------------------------
def gemv_quant(raw_blocks, fmt, rows, cols, x):
    """Quantised weights (Q8_0 = 8, Q4_0 = 2, Q5_0 = 6, IQ4_NL = 20, MXFP4 = 39) against
    a Q8_0-quantised x (the K-quants, IQ4_XS and TQ2_0: gemv_kquant).
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
    elif fmt == 6:
        # Q5_0: F16 d, 32 high bits (bit j of the little-endian word for
        # element j), then Q4_0's nibbles; q = (nibble | high << 4) - 16
        blk = np.frombuffer(raw_blocks, dtype=np.uint8).reshape(rows, nb, 22)
        dw = blk[:, :, 0:2].copy().view(np.float16)[..., 0]
        qh = blk[:, :, 2:6].copy().view(np.uint32)[..., 0].astype(np.int32)
        hb = (qh[..., None] >> np.arange(32, dtype=np.int32)) & 1
        nib = blk[:, :, 6:]
        qw = np.concatenate([nib & 0x0F, nib >> 4], axis=2).astype(np.int32) + (hb << 4) - 16
    elif fmt == 20:
        # IQ4_NL: as Q4_0, the nibbles index kvalues_iq4nl
        blk = np.frombuffer(raw_blocks, dtype=np.uint8).reshape(rows, nb, 18)
        dw = blk[:, :, 0:2].copy().view(np.float16)[..., 0]
        nib = blk[:, :, 2:]
        qw = np.concatenate([IQ4_VALUES[nib & 0x0F], IQ4_VALUES[nib >> 4]], axis=2)
    elif fmt == 39:
        # MXFP4: E8M0 exponent e (scale ggml_e8m0_to_fp32_half(e), a power of
        # two), then 16 bytes of E2M1 nibbles (doubled: kvalues_mxfp4), low
        # nibbles first
        blk = np.frombuffer(raw_blocks, dtype=np.uint8).reshape(rows, nb, 17)
        e = blk[:, :, 0].astype(np.uint32)
        bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - 1) << np.uint32(23))
        dw = bits.astype(np.uint32).view(F32)
        kv = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int32)
        nib = blk[:, :, 1:]
        qw = np.concatenate([kv[nib & 0x0F], kv[nib >> 4]], axis=2)
    elif fmt in K_FORMATS:
        return gemv_kquant(raw_blocks, fmt, rows, cols, x)
    else:
        raise ValueError(fmt)
    isum = np.einsum("rbk,bk->rb", qw, qx.astype(np.int32))    # exact
    scale = (dw.astype(F32) * dx.astype(F32)[None, :]).astype(F32)
    terms = (isum.astype(F32) * scale).astype(F32)
    return np.add.accumulate(terms, axis=1, dtype=F32)[:, -1]


# --- K-quants (256-weight super-blocks) -----------------------------------------
K_FORMATS = {10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
             23: "IQ4_XS", 35: "TQ2_0",
             16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S", 18: "IQ3_XXS", 21: "IQ3_S"}
K_BYTES = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 23: 136, 35: 66,
           16: 66, 17: 74, 22: 82, 18: 98, 21: 110}
# IQ4_NL / IQ4_XS: the 16 values a 4-bit index stands for (ggml's kvalues_iq4nl)
IQ4_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                      dtype=np.int64)


def quantize_q8_k(x):
    """The GEMV input for K-quant weights: one F32 scale per 256 elements,
    and the sums of each 16 quantised elements (as llama.cpp's Q8_K, with
    quantize_q8_0's arithmetic: d = amax/127; id = d ? 1/d : 0;
    q = roundf(x*id); d is not rounded to F16)."""
    x = np.asarray(x, dtype=F32).reshape(-1, 256)
    amax = np.max(np.abs(x), axis=1).astype(F32)
    d = (amax / F32(127)).astype(F32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, F32(1) / np.where(d != 0, d, F32(1)), F32(0)).astype(F32)
    q = roundf((x * idv[:, None]).astype(F32)).astype(np.int64)
    return d, q, q.reshape(-1, 16, 16).sum(axis=2)


# --- IQ2 / IQ3 codebooks (ggml's iq2xxs_grid .. iq3s_grid) ----------------------
def _iq_grid(name):
    """the codebook as small integers, (entries, values per entry), from
    gguf-py's packed copy (which ggml's tables are checked against there)"""
    from gguf import quants as GQ
    c = getattr(GQ, name)
    c.init_grid()
    return np.rint(c.grid.reshape(c.grid_shape)).astype(np.int64)


_IQ_GRIDS = {}


def iq_grid(fmt):
    """fmt's codebook: IQ2_XXS 256 x 8, IQ2_XS 512 x 8, IQ2_S 1024 x 8 (values
    8, 25, 43), IQ3_XXS 256 x 4 (4 .. 62), IQ3_S 512 x 4 (1 .. 15)"""
    if fmt not in _IQ_GRIDS:
        _IQ_GRIDS[fmt] = _iq_grid({16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S",
                                   18: "IQ3_XXS", 21: "IQ3_S"}[fmt])
    return _IQ_GRIDS[fmt]


def iq_ksigns():
    """ggml's ksigns_iq2xs: 7 stored sign bits, the eighth their parity"""
    k = np.arange(128)
    par = np.array([bin(v).count("1") & 1 for v in k])
    return k | (par << 7)


def _sign_bits(sb):
    """bytes of 8 sign bits -> +1 / -1 per bit, bit 0 first"""
    bits = (sb[..., None].astype(np.int64) >> np.arange(8)) & 1
    return (1 - 2 * bits).reshape(*sb.shape[:-1], -1)


def kquant_parts(raw, fmt, n):
    """n super-blocks -> (q (n, 256) ints, sc (n, 16) per 16 weights, m (n, 16)
    or None, d (n) F32, dmin (n) F32 or None): weight k of 16-group g is
    d * sc[g] * q - dmin * m[g], as gguf-py's dequantisers."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, K_BYTES[fmt])
    f16 = lambda a: a.copy().view(np.float16)[:, 0].astype(F32)
    if fmt == 10:                                              # Q2_K
        scales, qs, d, dmin = b[:, :16], b[:, 16:80], f16(b[:, 80:82]), f16(b[:, 82:84])
        q = (qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3
        return q.reshape(n, 256).astype(np.int64), (scales & 15).astype(np.int64), \
            (scales >> 4).astype(np.int64), d, dmin
    if fmt == 11:                                              # Q3_K
        hmask, qs, sb, d = b[:, :32], b[:, 32:96], b[:, 96:108], f16(b[:, 108:110])
        ls = (sb[:, :8].reshape(n, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)).reshape(n, 16)
        hs = (sb[:, 8:].reshape(n, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)).reshape(n, 16)
        sc = ((ls & 15) | ((hs & 3) << 4)).astype(np.int64) - 32
        ql = (qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)).reshape(n, 256) & 3
        qh = (hmask.reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)).reshape(n, 256) & 1
        return ql.astype(np.int64) - 4 * (1 - qh.astype(np.int64)), sc, None, d, None
    if fmt in (12, 13):                                        # Q4_K, Q5_K
        d, dmin, sb = f16(b[:, 0:2]), f16(b[:, 2:4]), b[:, 4:16].reshape(n, 3, 4)
        dd, mm, md = sb[:, 0], sb[:, 1], sb[:, 2]
        sc = np.concatenate([dd & 0x3F, (md & 0x0F) | ((dd >> 2) & 0x30)], axis=1).astype(np.int64)
        mn = np.concatenate([mm & 0x3F, (md >> 4) | ((mm >> 2) & 0x30)], axis=1).astype(np.int64)
        qs = b[:, 16:144] if fmt == 12 else b[:, 48:176]
        q = ((qs.reshape(n, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(n, 256)
        q = q.astype(np.int64)
        if fmt == 13:
            qh = (b[:, 16:48].reshape(n, -1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1
            q = q | (qh.reshape(n, 256).astype(np.int64) << 4)
        return q, np.repeat(sc, 2, axis=1), np.repeat(mn, 2, axis=1), d, dmin
    if fmt == 23:                                              # IQ4_XS
        d = f16(b[:, 0:2])
        sh = b[:, 2].astype(np.int64) | (b[:, 3].astype(np.int64) << 8)
        sl = b[:, 4:8].astype(np.int64)
        ib = np.arange(8)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 15) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        nib = b[:, 8:136].reshape(n, 8, 16)
        q = IQ4_VALUES[np.concatenate([nib & 15, nib >> 4], axis=2)].reshape(n, 256)
        return q, np.repeat(ls - 32, 2, axis=1), None, d, None
    if fmt == 35:                                              # TQ2_0
        qs, d = b[:, :64], f16(b[:, 64:66])
        q = (qs.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3
        return q.reshape(n, 256).astype(np.int64) - 1, np.ones((n, 16), dtype=np.int64), None, d, None
    # IQ2 / IQ3: q = +-grid value; per 16 or 32 weights an odd integer
    # scale 2s + 1; d times an exact power of two (llama.cpp's
    # (0.5 + s) * 0.25 = (2s + 1) / 8, (0.5 + s) * 0.5 = (2s + 1) / 4)
    if fmt == 16:                                              # IQ2_XXS
        d = f16(b[:, 0:2]) * F32(0.125)
        aux = b[:, 2:66].copy().view(np.uint32).reshape(n, 8, 2).astype(np.int64)
        idx = b[:, 2:66].reshape(n, 8, 8)[:, :, 0:4]           # 4 grid indices / 32
        sg = iq_ksigns()[(aux[:, :, 1:2] >> (7 * np.arange(4))) & 127]       # (n, 8, 4)
        q = iq_grid(16)[idx] * _sign_bits(sg.reshape(n, 8, 4)).reshape(n, 8, 4, 8)
        sc = np.repeat(2 * (aux[:, :, 1] >> 28) + 1, 2, axis=1)
        return q.reshape(n, 256), sc, None, d, None
    if fmt == 17:                                              # IQ2_XS
        d = f16(b[:, 0:2]) * F32(0.125)
        qs = b[:, 2:66].copy().view(np.uint16).astype(np.int64)                # (n, 32)
        sb = b[:, 66:74].astype(np.int64)
        s16 = np.stack([sb & 15, sb >> 4], axis=2).reshape(n, 16)
        q = iq_grid(17)[qs & 511] * _sign_bits(iq_ksigns()[qs >> 9][..., None]).reshape(n, 32, 8)
        return q.reshape(n, 256), 2 * s16 + 1, None, d, None
    if fmt == 22:                                              # IQ2_S
        d = f16(b[:, 0:2]) * F32(0.125)
        qs, signs, qh, sb = b[:, 2:34], b[:, 34:66], b[:, 66:74], b[:, 74:82].astype(np.int64)
        hi = (qh[:, :, None].astype(np.int64) >> (2 * np.arange(4))) & 3      # (n, 8, 4)
        idx = qs.astype(np.int64) | (hi.reshape(n, 32) << 8)
        q = iq_grid(22)[idx] * _sign_bits(signs[..., None]).reshape(n, 32, 8)
        s16 = np.stack([sb & 15, sb >> 4], axis=2).reshape(n, 16)
        return q.reshape(n, 256), 2 * s16 + 1, None, d, None
    if fmt == 18:                                              # IQ3_XXS
        d = f16(b[:, 0:2]) * F32(0.25)
        idx = b[:, 2:66].astype(np.int64)                      # 64 x 4 values
        aux = b[:, 66:98].copy().view(np.uint32).astype(np.int64)             # (n, 8)
        sg = iq_ksigns()[(aux[:, :, None] >> (7 * np.arange(4))) & 127]      # (n, 8, 4)
        q = iq_grid(18)[idx].reshape(n, 8, 4, 8) * _sign_bits(sg.reshape(n, 8, 4)).reshape(n, 8, 4, 8)
        sc = np.repeat(2 * (aux >> 28) + 1, 2, axis=1)
        return q.reshape(n, 256), sc, None, d, None
    if fmt == 21:                                              # IQ3_S
        d = f16(b[:, 0:2])
        qs, qh, signs, sb = b[:, 2:66], b[:, 66:74], b[:, 74:106], b[:, 106:110].astype(np.int64)
        hi = (qh[:, :, None].astype(np.int64) >> np.arange(8)) & 1           # (n, 8, 8)
        idx = qs.astype(np.int64) | (hi.reshape(n, 64) << 8)
        q = iq_grid(21)[idx].reshape(n, 256) * _sign_bits(signs)
        s32 = np.stack([sb & 15, sb >> 4], axis=2).reshape(n, 8)
        return q, np.repeat(2 * s32 + 1, 2, axis=1), None, d, None
    if fmt == 14:                                              # Q6_K
        ql, qh, sc, d = b[:, :128], b[:, 128:192], b[:, 192:208].view(np.int8), f16(b[:, 208:210])
        lo = ((ql.reshape(n, -1, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(n, -1, 32)
        hi = ((qh.reshape(n, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, -1, 32)
        q = (lo.astype(np.int64) | (hi.astype(np.int64) << 4)) - 32
        return q.reshape(n, 256), sc.astype(np.int64), None, d, None
    raise ValueError(fmt)


def gemv_kquant(raw, fmt, rows, cols, x):
    """K-quant weights against a Q8_K-style input (quantize_q8_k).  Per row:
    acc = 0; for each super-block s in order, with exact integers
        S1 = sum over 16-groups g of sc[g] * (sum of q_w * q_x over g)
        S2 = sum over 16-groups g of m[g] * (sum of q_x over g)
    and then (each op rounded)
        t = F32(S1) * (d * dx)                    (formats without mins)
        t = F32(S1) * (d * dx) - F32(S2) * (dmin * dx)
        acc = acc + t"""
    dx, qx, bsum = quantize_q8_k(x)
    nb = cols // 256
    q, sc, m, d, dmin = kquant_parts(raw, fmt, rows * nb)
    isum = np.einsum("rbgk,bgk->rbg", q.reshape(rows, nb, 16, 16), qx.reshape(nb, 16, 16))
    S1 = np.sum(sc.reshape(rows, nb, 16) * isum, axis=2)
    t = (S1.astype(F32) * (d.reshape(rows, nb) * dx[None, :]).astype(F32)).astype(F32)
    if m is not None:
        S2 = np.sum(m.reshape(rows, nb, 16) * bsum[None, :, :], axis=2)
        t2 = (S2.astype(F32) * (dmin.reshape(rows, nb) * dx[None, :]).astype(F32)).astype(F32)
        t = (t - t2).astype(F32)
    return np.add.accumulate(t, axis=1, dtype=F32)[:, -1]


FLOAT_LANES = 8


def dot8(w, x):
    """The float lane's dot product, row by row of w (rows, n) with x (n):
    each product rounded to F32; product i added into partial sum i mod 8
    (each starting from -0, the identity of addition); then
    ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)), every operation
    rounded.  (The pipelined adder's latency is 8: each partial's last sum
    is back just as its next product arrives.)"""
    w = np.atleast_2d(np.asarray(w, dtype=F32))
    x = np.asarray(x, dtype=F32)
    with np.errstate(all="ignore"):
        prods = (w * x[None, :]).astype(F32)
        rows, n = prods.shape
        pad = (-n) % FLOAT_LANES
        if pad:
            prods = np.concatenate([prods, np.full((rows, pad), F32(-0.0), dtype=F32)], axis=1)
        p = np.full((rows, FLOAT_LANES), F32(-0.0), dtype=F32)
        for g in prods.reshape(rows, -1, FLOAT_LANES).transpose(1, 0, 2):
            p = (p + g).astype(F32)
        q = [(p[:, 2 * i] + p[:, 2 * i + 1]).astype(F32) for i in range(4)]
        return (((q[0] + q[1]).astype(F32)) + ((q[2] + q[3]).astype(F32))).astype(F32)


def gemv_float(w, x):
    """Float weights (F32, F16 and BF16, exactly as F32): dot8 per row."""
    return dot8(w, x)


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
_softplus = lambda x: np.log1p(np.exp(x))
SOFTPLUS = Table(_softplus, -12.0, 12.0, lambda x: np.zeros_like(x), lambda x: x)


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


def vexp(x):
    """The softmax's exp table (inputs <= 0; >= 0 gives 1)."""
    return EXP(x)


def softplus(x):
    return SOFTPLUS(x)


def ssm_update(h, a, u, B):
    """h[r][c] = (a[r] * h[r][c]) + (u[r] * B[c]), each product and the sum
    rounded to F32."""
    h = np.asarray(h, dtype=F32)
    t = (np.asarray(a, dtype=F32)[:, None] * h).astype(F32)
    s = (np.asarray(u, dtype=F32)[:, None] * np.asarray(B, dtype=F32)[None, :]).astype(F32)
    return (t + s).astype(F32)


def meanrows(M):
    """Column-wise sequential sums over the rows, times recip(rows)."""
    M = np.asarray(M, dtype=F32)
    s = np.add.accumulate(M, axis=0, dtype=F32)[-1]
    return (s * recip(F32(M.shape[0]))).astype(F32)


def attention_head(q, K, V):
    """One head.  q: (hd,) F32; K, V: (T, hd) as stored (F16 -> F32 exact).
    s_t = dot8(k_t, q) * recip(sqrt(hd))       [scale = rsqrt(hd)]
    m = max(s); e_t = EXP(s_t - m); z = seq_sum(e); w_t = e_t * recip(z)
    out = sequential over t of out + w_t * v_t."""
    q = np.asarray(q, dtype=F32)
    K = np.asarray(K, dtype=F32)
    V = np.asarray(V, dtype=F32)
    s = (dot8(K, q) * rsqrt(F32(q.size))).astype(F32)
    e = EXP((s - s.max()).astype(F32))
    w = (e * recip(seq_sum(e))).astype(F32)
    out = np.zeros(q.size, dtype=F32)
    for t in range(K.shape[0]):
        out = (out + (w[t] * V[t]).astype(F32)).astype(F32)
    return out
