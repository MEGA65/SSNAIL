"""Simple quantisers (Q4_K, Q6_K, IQ4_NL, IQ2 / IQ3) for the tools.

gguf-py dequantises these formats but cannot make them.  These pack exactly
the layouts gguf-py's dequantisers read (checked by round trip in
tests/test_hw.py), with plain per-group scale choices rather than
llama.cpp's error-minimising search: good enough for test models and for
repacking float weights, not a replacement for llama.cpp's quantiser.
"""
import numpy as np

F16 = np.float16


def _f16(v):
    return np.asarray(v, dtype=np.float32).astype(F16)


def quantize_q6_k(x):
    """x (F32, length a multiple of 256) -> Q6_K blocks (210 bytes each):
    per 16 weights a signed 8-bit scale, q in -32..31, one F16 d."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 256)
    n = x.shape[0]
    g = x.reshape(n, 16, 16)
    amax = np.abs(g).max(axis=2)                                   # (n, 16)
    s = amax / 32                                                  # group step
    d = _f16(s.max(axis=1) / 127).astype(np.float32)               # (n,)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d[:, None] > 0, np.round(s / d[:, None]), 0)
    sc = np.clip(sc, -128, 127).astype(np.int64)
    step = d[:, None] * sc                                         # (n, 16)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(step[:, :, None] > 0, np.round(g / step[:, :, None]), 0)
    L = (np.clip(q, -32, 31).astype(np.int64) + 32).reshape(n, 256)
    e = np.arange(256)
    h, ep = e // 128, e % 128
    ql = np.zeros((n, 128), dtype=np.int64)
    qh = np.zeros((n, 64), dtype=np.int64)
    np.add.at(ql, (slice(None), 64 * h + ep % 64), (L & 15) << (4 * (ep // 64)))
    np.add.at(qh, (slice(None), 32 * h + ep % 32), (L >> 4) << (2 * (ep // 32)))
    out = np.concatenate([ql.astype(np.uint8), qh.astype(np.uint8),
                          sc.astype(np.int8).view(np.uint8),
                          _f16(d).reshape(n, 1).view(np.uint8)], axis=1)
    return out.reshape(-1)


def quantize_q4_k(x):
    """x (F32, length a multiple of 256) -> Q4_K blocks (144 bytes each):
    per 32 weights a 6-bit scale and min, q in 0..15, F16 d and dmin
    (weight = d * sc * q - dmin * m)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 256)
    n = x.shape[0]
    g = x.reshape(n, 8, 32)
    mn = np.minimum(g.min(axis=2), 0)                              # (n, 8)
    mx = g.max(axis=2)
    s = (mx - mn) / 15
    d = _f16(s.max(axis=1) / 63).astype(np.float32)
    dmin = _f16((-mn).max(axis=1) / 63).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d[:, None] > 0, np.round(s / d[:, None]), 0)
        m = np.where(dmin[:, None] > 0, np.round(-mn / dmin[:, None]), 0)
    sc = np.clip(sc, 0, 63).astype(np.int64)
    m = np.clip(m, 0, 63).astype(np.int64)
    step = d[:, None] * sc
    off = dmin[:, None] * m
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(step[:, :, None] > 0, np.round((g + off[:, :, None]) / step[:, :, None]), 0)
    q = np.clip(q, 0, 15).astype(np.int64)                         # (n, 8, 32)
    scales = np.zeros((n, 12), dtype=np.int64)
    for j in range(8):
        if j < 4:
            scales[:, j] |= sc[:, j]
            scales[:, 4 + j] |= m[:, j]
        else:
            scales[:, 8 + j - 4] |= (sc[:, j] & 15) | ((m[:, j] & 15) << 4)
            scales[:, j - 4] |= (sc[:, j] >> 4) << 6
            scales[:, 4 + j - 4] |= (m[:, j] >> 4) << 6
    qs = np.zeros((n, 128), dtype=np.int64)
    for j in range(8):
        qs[:, 32 * (j // 2):32 * (j // 2) + 32] |= q[:, j, :] << (4 * (j % 2))
    out = np.concatenate([_f16(d).reshape(n, 1).view(np.uint8), _f16(dmin).reshape(n, 1).view(np.uint8),
                          scales.astype(np.uint8), qs.astype(np.uint8)], axis=1)
    return out.reshape(-1)


IQ4_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                      dtype=np.float32)


def quantize_iq4_nl(x):
    """x (F32, length a multiple of 32) -> IQ4_NL blocks (18 bytes each): F16
    d = amax / 127, each weight the nearest of d * kvalues_iq4nl (low nibbles
    are elements 0-15, high 16-31)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    n = x.shape[0]
    amax = np.abs(x).max(axis=1)
    d = _f16(amax / 127).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(d[:, None] > 0, x / d[:, None], 0)
    idx = np.argmin(np.abs(v[:, :, None] - IQ4_VALUES[None, None, :]), axis=2)
    qs = (idx[:, :16] | (idx[:, 16:] << 4)).astype(np.uint8)
    return np.concatenate([_f16(d).reshape(n, 1).view(np.uint8), qs], axis=1).reshape(-1)


# --- IQ2 / IQ3 (codebook) quantisers ---------------------------------------------
# Plain nearest-codebook-entry choices: per 16 or 32 weights an odd scale
# 2s + 1 from the sub-block's largest magnitude, then for each 8 (IQ2) or 4
# (IQ3) weights the entry nearest their magnitudes, the signs from x.  The
# ksigns formats store 7 signs and imply the eighth (even parity): an odd
# group flips its smallest weight.  No importance matrix, no search over
# scales: test models and repacking, not llama.cpp's quality.

def _grid(fmt):
    import ssnail_hw as H
    return H.iq_grid(fmt).astype(np.float32)


# (values per index, weights per scale, d factor, largest grid value)
_IQ = {16: (8, 32, 0.125), 17: (8, 16, 0.125), 22: (8, 16, 0.125),
       18: (4, 32, 0.25), 21: (4, 32, 1.0)}


def _iq_choose(x, fmt):
    """x (n, 256) -> (d F16 (n,), s (n, 256 / per) 0..15, idx (n, 256 / k),
    neg (n, 256) bools)"""
    k, per, f = _IQ[fmt]
    g = _grid(fmt)
    n = x.shape[0]
    a = np.abs(x)
    sub = a.reshape(n, -1, per).max(axis=2)                       # (n, subs)
    step = sub / g.max()                                          # ideal grid step
    d = _f16(step.max(axis=1) / (f * 31)).astype(np.float32)      # sc <= 31
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d[:, None] > 0, step / (d[:, None] * f), 0)
    s = np.clip(np.round((sc - 1) / 2), 0, 15).astype(np.int64)
    st = (d[:, None] * f * (2 * s + 1)).astype(np.float32)        # actual step
    stw = np.repeat(st, per, axis=1)                              # (n, 256)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(stw > 0, a / stw, 0).reshape(-1, k)          # groups of k
    g2 = (g * g).sum(axis=1)
    idx = np.concatenate([(g2[None, :] - 2 * (v[i:i + 8192] @ g.T)).argmin(axis=1)
                          for i in range(0, len(v), 8192)]).reshape(n, -1)
    neg = x < 0
    return _f16(d), s, idx, neg


def _even_signs(neg, x):
    """ksigns: the 8 signs of each 8 must have even parity; an odd 8 flips
    its smallest weight's sign"""
    n8 = neg.reshape(-1, 8).copy()
    odd = n8.sum(axis=1) % 2 == 1
    sm = np.abs(x).reshape(-1, 8).argmin(axis=1)
    n8[odd, sm[odd]] = ~n8[odd, sm[odd]]
    return n8


def _bits(b8):
    """(m, 8) bools -> (m,) bytes, element 0 in bit 0"""
    return (b8.astype(np.int64) << np.arange(8)).sum(axis=1)


def quantize_iq(x, fmt):
    """x (F32, length a multiple of 256) -> IQ2_XXS / IQ2_XS / IQ2_S /
    IQ3_XXS / IQ3_S blocks, in gguf-py's layouts"""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 256)
    n = x.shape[0]
    d, s, idx, neg = _iq_choose(x, fmt)
    dh = d.reshape(n, 1).view(np.uint8)
    if fmt in (16, 17, 18):
        s8 = _bits(_even_signs(neg, x)).reshape(n, 32) & 127      # 7 stored
    if fmt == 16:                                                 # IQ2_XXS
        aux1 = (s8.reshape(n, 8, 4) << (7 * np.arange(4))).sum(axis=2) | (s[:, :] << 28)
        blk = np.concatenate([idx.reshape(n, 8, 4).astype(np.uint8),
                              aux1.astype(np.uint32).reshape(n, 8, 1).view(np.uint8)], axis=2)
        return np.concatenate([dh, blk.reshape(n, 64)], axis=1).reshape(-1)
    if fmt == 17:                                                 # IQ2_XS
        qs = (idx | (s8 << 9)).astype(np.uint16)                  # (n, 32)
        sc = (s[:, 0::2] | (s[:, 1::2] << 4)).astype(np.uint8)    # (n, 8)
        return np.concatenate([dh, qs.view(np.uint8), sc], axis=1).reshape(-1)
    if fmt == 22:                                                 # IQ2_S
        qs = (idx & 255).astype(np.uint8)                         # (n, 32)
        qh = ((idx >> 8).reshape(n, 8, 4) << (2 * np.arange(4))).sum(axis=2).astype(np.uint8)
        signs = _bits(neg.reshape(-1, 8)).reshape(n, 32).astype(np.uint8)
        sc = (s[:, 0::2] | (s[:, 1::2] << 4)).astype(np.uint8)
        return np.concatenate([dh, qs, signs, qh, sc], axis=1).reshape(-1)
    if fmt == 18:                                                 # IQ3_XXS
        aux = (s8.reshape(n, 8, 4) << (7 * np.arange(4))).sum(axis=2) | (s << 28)
        return np.concatenate([dh, idx.astype(np.uint8),
                               aux.astype(np.uint32).view(np.uint8)], axis=1).reshape(-1)
    # IQ3_S
    qs = (idx & 255).astype(np.uint8)                             # (n, 64)
    qh = ((idx >> 8).reshape(n, 8, 8) << np.arange(8)).sum(axis=2).astype(np.uint8)
    signs = _bits(neg.reshape(-1, 8)).reshape(n, 32).astype(np.uint8)
    sc = (s[:, 0::2] | (s[:, 1::2] << 4)).astype(np.uint8)        # (n, 4)
    return np.concatenate([dh, qs, qh, signs, sc], axis=1).reshape(-1)


QUANTISERS = {12: quantize_q4_k, 14: quantize_q6_k, 20: quantize_iq4_nl,
              16: lambda x: quantize_iq(x, 16), 17: lambda x: quantize_iq(x, 17),
              22: lambda x: quantize_iq(x, 22), 18: lambda x: quantize_iq(x, 18),
              21: lambda x: quantize_iq(x, 21)}


def quantize_k(w, fmt):
    """w (rows, cols), cols a multiple of the block size -> bytes, rows of
    fmt blocks"""
    return QUANTISERS[fmt](np.asarray(w, dtype=np.float32).reshape(-1)).tobytes()
