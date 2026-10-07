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
from gguf.quants import quantize  # noqa: E402
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
for fmt, name in ((8, "Q8_0"), (2, "Q4_0")):
    W = rng.standard_normal((rows, cols)).astype(F32)
    raw = np.asarray(quantize(W, Q(fmt))).tobytes()
    xv = rng.standard_normal(cols).astype(F32)
    fast = H.gemv_quant(raw, fmt, rows, cols, xv)
    dx, qx = H.quantize_q8_0(xv)
    bs = 34 if fmt == 8 else 18
    slow = []
    for r in range(rows):
        acc = F32(0)
        for blk in range(cols // 32):
            o = (r * (cols // 32) + blk) * bs
            dw = F32(np.frombuffer(raw[o:o + 2], dtype=np.float16)[0])
            if fmt == 8:
                qw = np.frombuffer(raw[o + 2:o + 34], dtype=np.int8).astype(int)
            else:
                nib = np.frombuffer(raw[o + 2:o + 18], dtype=np.uint8).astype(int)
                qw = np.concatenate([(nib & 15) - 8, (nib >> 4) - 8])
            isum = int(np.dot(qw, qx[blk].astype(int)))
            acc = F32(acc + F32(F32(isum) * F32(dw * F32(dx[blk]))))
        slow.append(acc)
    report(np.array_equal(fast, np.array(slow, dtype=F32)),
           f"GEMV {name}: vectorised emulator == scalar definition (bit-exact)")

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
             ("gpt2 Q8_0", lambda p: TG.make_gpt2(p, Q.Q8_0)),
             ("bert Q8_0", lambda p: TG.make_bert(p, Q.Q8_0))]
    for name, maker in cases:
        g = os.path.join(d, "m.gguf")
        maker(g)
        model = C.Model(g)
        image = C.convert(model, 32, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False)
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
