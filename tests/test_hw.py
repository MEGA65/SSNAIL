#!/usr/bin/env python3
"""Hardware-numerics mode: the definitions in ssnail_hw.py, and what they cost
against the float reference on whole models."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gguf import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402
import ssnail_convert as C  # noqa: E402
import ssnail_hw as H  # noqa: E402
import ssnail_sim as S  # noqa: E402
import test_gpt2_bert as TG  # noqa: E402
import test_pipeline as TP  # noqa: E402

F32 = np.float32
ok = True


def report(cond, text):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {text}")


rng = np.random.default_rng(7)

# 1. Q8_0 quantisation is bit-identical to llama.cpp's (via gguf-py)
x = (rng.standard_normal(32 * 64) * rng.choice([0.01, 1, 50], 32 * 64)).astype(F32)
x[:32] = 0                                    # an all-zero block
x[32:64] = np.arange(32, dtype=F32) * 0.5     # exact .5 cases
# a block where x * id lands just below .5: 0.49999997 + 0.5 rounds up in F32
x[64:96] = F32(0)
x[64] = F32(127.0)
x[65] = np.nextafter(F32(0.5), F32(0))
d, q = H.quantize_q8_0(x)
mine = np.concatenate([np.concatenate([d[i:i + 1].view(np.uint8), q[i].view(np.uint8)])
                       for i in range(len(d))])
ref = np.asarray(quantize(x.reshape(1, -1), Q.Q8_0)).reshape(-1)
report(np.array_equal(mine, ref), "Q8_0 input quantisation identical to llama.cpp's")

# 2. The vectorised GEMV equals a plain scalar reading of the definition
rows, cols = 7, 96
for fmt, name in ((8, "Q8_0"), (2, "Q4_0"), (6, "Q5_0")):
    W = rng.standard_normal((rows, cols)).astype(F32)
    raw = np.asarray(quantize(W, Q(fmt))).tobytes()
    xv = rng.standard_normal(cols).astype(F32)
    fast = H.gemv_quant(raw, fmt, rows, cols, xv)
    dx, qx = H.quantize_q8_0(xv)
    bs = {8: 34, 2: 18, 6: 22}[fmt]
    slow = []
    for r in range(rows):
        acc = F32(0)
        for blk in range(cols // 32):
            o = (r * (cols // 32) + blk) * bs
            dw = F32(np.frombuffer(raw[o:o + 2], dtype=np.float16)[0])
            if fmt == 8:
                qw = np.frombuffer(raw[o + 2:o + 34], dtype=np.int8).astype(int)
            elif fmt == 6:
                qh = int.from_bytes(raw[o + 2:o + 6], "little")
                nib = np.frombuffer(raw[o + 6:o + 22], dtype=np.uint8).astype(int)
                qw = np.array([((nib[j % 16] >> (4 * (j // 16))) & 15) + (((qh >> j) & 1) << 4) - 16
                               for j in range(32)])
            else:
                nib = np.frombuffer(raw[o + 2:o + 18], dtype=np.uint8).astype(int)
                qw = np.concatenate([(nib & 15) - 8, (nib >> 4) - 8])
            isum = int(np.dot(qw, qx[blk].astype(int)))
            acc = F32(acc + F32(F32(isum) * F32(dw * F32(dx[blk]))))
        slow.append(acc)
    report(np.array_equal(fast, np.array(slow, dtype=F32)),
           f"GEMV {name}: vectorised emulator == scalar definition (bit-exact)")

# 2. (cont.) Q5_0's layout as gguf-py reads it: weight = F16 d * q
W = rng.standard_normal((3, 64)).astype(F32)
raw = np.asarray(quantize(W, Q.Q5_0)).tobytes()
blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 22)
qh = blk[:, 2:6].copy().view(np.uint32)[:, 0].astype(np.int64)
q = np.concatenate([blk[:, 6:] & 15, blk[:, 6:] >> 4], axis=1).astype(np.int64) \
    + (((qh[:, None] >> np.arange(32)) & 1) << 4) - 16
mine = (blk[:, 0:2].copy().view(np.float16).astype(F32) * q).reshape(-1)
report(np.array_equal(mine, dequantize(blk.reshape(3, -1), Q.Q5_0).reshape(-1)),
       "Q5_0 layout: d * q equals gguf-py's dequantisation")

# 2a. IQ4_NL (no gguf-py quantiser: random blocks with a sane d)
rows, cols = 5, 96
b = rng.integers(0, 256, (rows * cols // 32, 18), dtype=np.uint8)
b[:, 0:2] = (rng.standard_normal((len(b), 1)) * 0.01).astype(np.float16).view(np.uint8)
raw = b.tobytes()
xv = rng.standard_normal(cols).astype(F32)
fast = H.gemv_quant(raw, 20, rows, cols, xv)
dx, qx = H.quantize_q8_0(xv)
slow = []
for r_ in range(rows):
    acc = F32(0)
    for blk in range(cols // 32):
        o = (r_ * (cols // 32) + blk) * 18
        dw = F32(np.frombuffer(raw[o:o + 2], dtype=np.float16)[0])
        nib = np.frombuffer(raw[o + 2:o + 18], dtype=np.uint8).astype(int)
        qw = np.concatenate([H.IQ4_VALUES[nib & 15], H.IQ4_VALUES[nib >> 4]])
        acc = F32(acc + F32(F32(int(np.dot(qw, qx[blk].astype(int)))) * F32(dw * F32(dx[blk]))))
    slow.append(acc)
report(np.array_equal(fast, np.array(slow, dtype=F32)),
       "GEMV IQ4_NL: vectorised emulator == scalar definition (bit-exact)")
ref = dequantize(b.reshape(rows, -1), Q.IQ4_NL).reshape(rows, cols).astype(np.float64) @ xv
report(np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) < 0.03, "GEMV IQ4_NL: hw vs float")

# 2b. K-quants: the parts are gguf-py's dequantisation, and the vectorised
# GEMV equals a scalar reading of its definition (ssnail_hw.gemv_kquant)
D_OFF = {10: [80, 82], 11: [108], 12: [0, 2], 13: [0, 2], 14: [208], 23: [0], 35: [64],
         16: [0], 17: [0], 22: [0], 18: [0], 21: [0]}


def k_random(fmt, n):
    """n random super-blocks (gguf-py has no K-quant quantisers) with sane
    F16 d / dmin"""
    b = rng.integers(0, 256, (n, H.K_BYTES[fmt]), dtype=np.uint8)
    for o in D_OFF[fmt]:
        b[:, o:o + 2] = (rng.standard_normal((n, 1)) * 0.01).astype(np.float16).view(np.uint8)
    return b.tobytes()


for fmt, name in H.K_FORMATS.items():
    rows, cols = 8, 512
    nb = cols // 256
    raw = k_random(fmt, rows * nb)
    q, sc, mn, d, dmin = H.kquant_parts(raw, fmt, rows * nb)
    w = d[:, None].astype(np.float64) * np.repeat(sc, 16, axis=1) * q
    if mn is not None:
        w = w - dmin[:, None].astype(np.float64) * np.repeat(mn, 16, axis=1)
    ref = dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(rows, -1), Q(fmt)).reshape(-1, 256)
    report(np.allclose(w, ref, rtol=1e-6, atol=1e-9), f"{name}: unpacked parts == gguf-py dequantize")
    xv = rng.standard_normal(cols).astype(F32)
    fast = H.gemv_quant(raw, fmt, rows, cols, xv)
    dx, qx, bsum = H.quantize_q8_k(xv)
    slow = []
    for r in range(rows):
        acc = F32(0)
        for s_ in range(nb):
            i = r * nb + s_
            S1 = sum(int(sc[i, g]) * int(np.dot(q[i, 16 * g:16 * g + 16], qx[s_, 16 * g:16 * g + 16]))
                     for g in range(16))
            t = F32(F32(S1) * F32(d[i] * dx[s_]))
            if mn is not None:
                S2 = sum(int(mn[i, g]) * int(bsum[s_, g]) for g in range(16))
                t = F32(t - F32(F32(S2) * F32(dmin[i] * dx[s_])))
            acc = F32(acc + t)
        slow.append(acc)
    report(np.array_equal(fast, np.array(slow, dtype=F32)),
           f"GEMV {name}: vectorised emulator == scalar definition (bit-exact)")
    # the datapath against gguf-py's weights and the quantised x (only F32
    # rounding apart), then against float x (x's 8-bit quantisation too)
    W = ref.reshape(rows, cols).astype(np.float64)
    yq = W @ (qx.reshape(-1) * np.repeat(dx, 256)).astype(np.float64)
    report(np.max(np.abs(fast - yq)) <= 1e-5 * np.max(np.abs(yq)), f"GEMV {name}: hw == dequantised w . x")
    yf = W @ xv
    rel = np.sqrt(np.mean((fast - yf) ** 2) / np.mean(yf ** 2))
    report(rel < 0.03, f"GEMV {name}: hw vs float rms rel.err {rel:.1e}")

# 2c. The tools' K-quant quantisers pack what gguf-py dequantises
import ssnail_kquant as KQ  # noqa: E402
for fmt, name in ((12, "Q4_K"), (14, "Q6_K")):
    W = (rng.standard_normal((4, 512)) * 0.05).astype(F32)
    W[0, :256] = 0
    raw = KQ.quantize_k(W, fmt)
    back = dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(4, -1), Q(fmt)).reshape(4, 512)
    err = np.sqrt(np.mean((back - W) ** 2)) / np.sqrt(np.mean(W ** 2))
    report(len(raw) == 4 * 2 * H.K_BYTES[fmt] and err < (0.1 if fmt == 12 else 0.03)
           and np.all(back[0, :256] == 0), f"{name} quantiser: round trip rel.rms {err:.3f}")

# 3. Function tables
for tab, f, lo, hi, name in ((H.SILU, lambda v: v / (1 + np.exp(-v)), -10, 10, "SiLU"),
                             (H.GELU, lambda v: 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3))), -8, 8, "GELU"),
                             (H.EXP, np.exp, -20, 0, "exp")):
    v = np.linspace(lo, hi, 20001).astype(F32)
    err = np.max(np.abs(tab(v).astype(np.float64) - f(v.astype(np.float64))))
    report(err < 2e-3, f"{name} table: max abs error {err:.1e} over [{lo}, {hi}]")

# 4. Whole models: hardware numerics vs the float reference, same inputs
with tempfile.TemporaryDirectory() as d:
    cases = [("llama Q8_0", lambda p: TP.make_model(p, Q.Q8_0)),
             ("llama Q4_0", lambda p: TP.make_model(p, Q.Q4_0)),
             ("llama F16", lambda p: TP.make_model(p, Q.F16)),
             ("llama Q4_K", lambda p: TP.make_model(p, Q.Q4_K)),
             ("llama Q6_K", lambda p: TP.make_model(p, Q.Q6_K)),
             ("llama IQ4_NL", lambda p: TP.make_model(p, Q.IQ4_NL)),
             ("llama Q5_0", lambda p: TP.make_model(p, Q.Q5_0)),
             ("gpt2 Q8_0", lambda p: TG.make_gpt2(p, Q.Q8_0)),
             ("bert Q8_0", lambda p: TG.make_bert(p, Q.Q8_0))]
    for name, maker in cases:
        g = os.path.join(d, "m.gguf")
        maker(g)
        model = C.Model(g)
        image = C.convert(model, 32, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False)
        if "_K" in name or "IQ4" in name or "Q5_0" in name:
            b = C.build(model, 32, "keep", C.FMT_BY_NAME["q8_0"], 8, "f16", "General")
            kept = all(e[2] == e[1] and not e[4] for e in b.log if e[1] == name.split()[1])
            report(kept, f"{name:11s} matrices kept native (not repacked)")
        text = "once upon a time the happy dog sat on the mat" if "bert" not in name \
            else "throw the lamp at the troll"
        ids = C.encode(model, text)
        outs = []
        for hw in (False, True):
            m = S.Machine(image)
            m.hw = hw
            if "bert" in name:
                v, _ = S.encode_sequence(m, ids)
                outs.append(v[None, :])
            else:
                seen = []
                m.on_argmax = seen.append
                S.generate(m, ids, 1)
                outs.append(np.array(seen))
        ref, hw = outs
        rel = np.max(np.abs(hw - ref)) / np.max(np.abs(ref))
        if "bert" in name:
            cos = float(hw[0] @ ref[0] / np.linalg.norm(hw[0]) / np.linalg.norm(ref[0]))
            report(cos > 0.999, f"{name:11s} hw vs float: rel.err {rel:.1e}, cosine {cos:.5f}")
        else:
            agree = np.mean(np.argmax(hw, 1) == np.argmax(ref, 1))
            report(rel < 0.05 and agree >= 0.9,
                   f"{name:11s} hw vs float: rel.err {rel:.1e}, argmax agrees at "
                   f"{100 * agree:.0f}% of {len(ref)} positions")

print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
