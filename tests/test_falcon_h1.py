#!/usr/bin/env python3
"""Falcon-H1 checks (slow; needs models/Falcon-H1-Tiny-90M-Instruct-BF16.gguf, and
transformers for the first and last):
  1. the numpy reference (falcon_h1_ref.py) against transformers' FalconH1
  2. the emulator (float numerics) on a Q8_0 image against the reference with
     the same Q8_0 rounding
  3. the image's chat wrapping against transformers' chat template"""
import os, struct, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
GGUF = os.path.join(os.path.dirname(HERE), "models", "Falcon-H1-Tiny-90M-Instruct-BF16.gguf")
HF = "tiiuae/Falcon-H1-Tiny-90M-Instruct"
if not os.path.exists(GGUF):
    print(f"SKIP  no {GGUF}"); sys.exit(0)
import ssnail_convert as C, ssnail_sim as S, ssnail_isa as I
from falcon_h1_ref import FalconH1Ref
from gguf import GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize
SCALE = C.FALCON_H1_LOGIT_SCALE[(512, 24)]
bad = 0
def check(ok, what):
    global bad
    print(("PASS  " if ok else "FAIL  ") + what); bad += not ok
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(HF)
    hf = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).eval()
except Exception as e:
    tk = hf = None
    print(f"SKIP  transformers checks ({e})")
m = C.Model(GGUF)
ids = C.encode(m, "Once upon a time, there was a little robot who", bos=False)
if hf is not None:
    with torch.no_grad():
        hl = hf(torch.tensor([ids])).logits[0].numpy()
    ref = FalconH1Ref(GGUF)
    rl = np.array([ref.step(t) for t in ids]) * SCALE
    e = (np.abs(rl - hl).max(axis=1) / np.abs(hl).max(axis=1)).max()
    check(e < 1e-2 and (rl.argmax(1) == hl.argmax(1)).all(), f"reference vs transformers: rel err {e:.1e}")
    pre, post, stop = C.chat_template_parts(m)
    want = tk.apply_chat_template([{"role": "user", "content": "Hi there"}], add_generation_prompt=True)
    want = list(want["input_ids"] if hasattr(want, "keys") else want)
    check(pre + C.encode(m, "Hi there", bos=False) + post == want and m.field("tokenizer.ggml.tokens")[stop] == "<|im_end|>",
          "chat wrapping matches the chat template, stop <|im_end|>")
with tempfile.TemporaryDirectory() as tmp:
    img = C.convert(m, 64, "q8_0", I.FMT_Q8_0, 120, verbose=False, kv="f32")
    mach = S.Machine(img)
    toks = S.header(mach, I.H_TOKENS) - I.HYPERRAM_BASE
    for i, t in enumerate(ids):
        mach.mem[toks + 4 * i: toks + 4 * i + 4] = struct.pack("<I", t)
    S.set_rt(mach, I.R_POS, 0); S.set_rt(mach, I.R_PROMPT_LEN, len(ids))
    S.set_rt(mach, I.R_N_GENERATE, 1); S.set_rt(mach, I.R_STOP_REQUEST, 0)
    got = []
    mach.on_argmax = lambda v: got.append(v.copy()); mach.temperature = 0.0
    mach.run(S.header(mach, I.H_CODE))
ref = FalconH1Ref(GGUF)
for n, w in list(ref.W.items()):
    if w.ndim == 2 and min(w.shape) > 1 and "conv1d" not in n:
        ref.W[n] = dequantize(quantize(w.astype(np.float32), Q.Q8_0), Q.Q8_0).astype(np.float64).reshape(w.shape)
ref.cls = ref.W["token_embd.weight"]
e = max(np.abs(got[i] - ref.step(t) * SCALE).max() / np.abs(got[i]).max() for i, t in enumerate(ids))
check(e < 1e-4, f"emulator (Q8_0 image) vs reference: rel err {e:.1e}")
print("ALL PASSED" if not bad else "FAILURES"); sys.exit(1 if bad else 0)
