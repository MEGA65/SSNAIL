"""Simple K-quant quantisers (Q4_K, Q6_K) for the tools.

gguf-py dequantises the K-quants but cannot make them.  These pack exactly
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


QUANTISERS = {12: quantize_q4_k, 14: quantize_q6_k}


def quantize_k(w, fmt):
    """w (rows, cols), cols a multiple of 256 -> bytes, rows of fmt blocks"""
    return QUANTISERS[fmt](np.asarray(w, dtype=np.float32).reshape(-1)).tobytes()
