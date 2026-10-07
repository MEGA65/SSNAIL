#!/usr/bin/env python3
"""End-to-end test of the converter + emulator.

Builds small random Llama-architecture GGUF files (several weight formats,
with grouped-query attention and untied/tied classifiers), converts them, runs
the SSNAIL emulator, and compares every position's logits against an
independent numpy implementation of the Llama forward pass that reads the
GGUF directly.
"""

import os
import struct
import sys
import tempfile

import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ssnail_convert as C   # noqa: E402
import ssnail_sim as S       # noqa: E402


def make_model(path, wtype, dim=256, layers=2, heads=8, kv_heads=4, hidden=512,
               tied=False, seed=1):
    rng = np.random.default_rng(seed)
    pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    words = ["\u2581the", "\u2581a", "\u2581cat", "\u2581dog", "\u2581sat", "\u2581on",
             "\u2581mat", "\u2581once", "\u2581upon", "\u2581time", "\u2581was", "\u2581happy"]
    # Realistic SentencePiece shape: single characters plus every prefix of
    # each word, so BPE merges have a path to the whole word.
    chars = ["\u2581"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    prefixes = []
    for wd in words:
        for k in range(2, len(wd) + 1):
            if wd[:k] not in prefixes:
                prefixes.append(wd[:k])
    pieces += chars + prefixes
    pieces += [f"\u2581w{i}" for i in range(400 - len(pieces))]
    vocab = len(pieces)
    w = gguf.GGUFWriter(path, "llama")
    w.add_context_length(128)
    w.add_embedding_length(dim)
    w.add_block_count(layers)
    w.add_feed_forward_length(hidden)
    w.add_head_count(heads)
    w.add_head_count_kv(kv_heads)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_tokenizer_model("llama")
    w.add_token_list(pieces)
    w.add_token_scores([0.0] * 259 + [float(len(pc)) for pc in pieces[259:]])
    w.add_token_types([1] * vocab)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    def mat(name, rows, cols, scale=None):
        m = (rng.standard_normal((rows, cols)) * (scale or 1 / np.sqrt(cols))).astype(np.float32)
        if wtype == Q.F32:
            w.add_tensor(name, m)
        elif wtype == Q.F16:
            w.add_tensor(name, m.astype(np.float16))
        else:
            qb = quantize(m, wtype)
            w.add_tensor(name, qb, raw_shape=qb.shape, raw_dtype=wtype)

    def vecf(name, n):
        w.add_tensor(name, (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32))

    kvd = kv_heads * (dim // heads)
    mat("token_embd.weight", vocab, dim, scale=1.0)
    for l in range(layers):
        vecf(f"blk.{l}.attn_norm.weight", dim)
        vecf(f"blk.{l}.ffn_norm.weight", dim)
        mat(f"blk.{l}.attn_q.weight", dim, dim)
        mat(f"blk.{l}.attn_k.weight", kvd, dim)
        mat(f"blk.{l}.attn_v.weight", kvd, dim)
        mat(f"blk.{l}.attn_output.weight", dim, dim)
        mat(f"blk.{l}.ffn_gate.weight", hidden, dim)
        mat(f"blk.{l}.ffn_up.weight", hidden, dim)
        mat(f"blk.{l}.ffn_down.weight", dim, hidden)
    vecf("output_norm.weight", dim)
    if not tied:
        mat("output.weight", vocab, dim)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# --- Independent reference ----------------------------------------------------
class Reference:
    def __init__(self, path):
        r = GGUFReader(path)
        f = lambda k: r.fields[k].parts[r.fields[k].data[0]].tolist()[0]
        self.dim = f("llama.embedding_length")
        self.L = f("llama.block_count")
        self.nh = f("llama.attention.head_count")
        self.nkv = f("llama.attention.head_count_kv")
        self.eps = f("llama.attention.layer_norm_rms_epsilon")
        self.theta = f("llama.rope.freq_base")
        self.hd = self.dim // self.nh
        self.W = {}
        for t in r.tensors:
            raw = np.asarray(t.data)
            if t.tensor_type in (Q.F32, Q.F16):
                a = raw.astype(np.float64)
            else:
                a = dequantize(raw.view(np.uint8), t.tensor_type).astype(np.float64)
            ne = [int(x) for x in t.shape]
            self.W[t.name] = a.reshape(list(reversed(ne)) if len(ne) > 1 else ne)
        self.cls = self.W.get("output.weight", self.W["token_embd.weight"])

    def logits(self, tokens):
        dim, hd, nh, nkv = self.dim, self.hd, self.nh, self.nkv
        K = [[] for _ in range(self.L)]
        V = [[] for _ in range(self.L)]
        out = []
        rms = lambda v, g: v / np.sqrt(np.mean(v * v) + self.eps) * g

        def rope(v, pos):
            v = v.reshape(-1, hd).copy()
            for i in range(0, hd, 2):
                a = pos / self.theta ** (i / hd)
                c, s = np.cos(a), np.sin(a)
                v0, v1 = v[:, i].copy(), v[:, i + 1].copy()
                v[:, i], v[:, i + 1] = v0 * c - v1 * s, v0 * s + v1 * c
            return v.reshape(-1)

        for pos, tok in enumerate(tokens):
            x = self.W["token_embd.weight"][tok].copy()
            for l in range(self.L):
                g = lambda n: self.W[f"blk.{l}.{n}.weight"]
                xb = rms(x, g("attn_norm"))
                q = rope(g("attn_q") @ xb, pos)
                K[l].append(rope(g("attn_k") @ xb, pos))
                V[l].append(g("attn_v") @ xb)
                Ks, Vs = np.array(K[l]), np.array(V[l])
                att = np.zeros(dim)
                for h in range(nh):
                    kh = h // (nh // nkv)
                    s = Ks[:, kh * hd:(kh + 1) * hd] @ q[h * hd:(h + 1) * hd] / np.sqrt(hd)
                    a = np.exp(s - s.max()); a /= a.sum()
                    att[h * hd:(h + 1) * hd] = a @ Vs[:, kh * hd:(kh + 1) * hd]
                x = x + g("attn_output") @ att
                xb = rms(x, g("ffn_norm"))
                gt = g("ffn_gate") @ xb
                x = x + g("ffn_down") @ (gt / (1 + np.exp(-gt)) * (g("ffn_up") @ xb))
            out.append(self.cls @ rms(x, self.W["output_norm.weight"]))
        return np.array(out)


def run_case(name, wtype, conv_wtype="keep", tied=False, tol=2e-3, need_exact_tokens=True):
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "m.gguf")
        make_model(g, wtype, tied=tied)
        model = C.Model(g)
        image = C.convert(model, 64, conv_wtype, C.FMT_BY_NAME["q8_0"], 8, verbose=False)
        m = S.Machine(image)
        seen = []
        m.on_argmax = seen.append
        prompt = C.encode(model, "once upon a time the cat")
        gen, reason = S.generate(m, prompt, 12)
        seq = prompt + gen
        ref = Reference(g).logits(seq)
        sim = np.array(seen)
        n = len(sim)
        err = np.max(np.abs(sim - ref[:n])) / np.max(np.abs(ref[:n]))
        ref_next = list(np.argmax(ref[len(prompt) - 1:n], axis=1))
        tok_ok = ref_next == gen[:len(ref_next)]
        ok = err < tol and (tok_ok or not need_exact_tokens)
        print(f"{'PASS' if ok else 'FAIL'}  {name:34s} rel.err {err:.2e}  "
              f"tokens {'match' if tok_ok else 'DIFFER'}  ({len(gen)} generated, "
              f"stop {reason}, {len(image)} bytes)")
        return ok


def test_tokenizer_and_stops():
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "m.gguf")
        make_model(g, Q.F32)
        model = C.Model(g)
        ids = C.encode(model, "the cat sat")
        words = [model.field("tokenizer.ggml.tokens")[i] for i in ids]
        ok1 = words == ["<s>", "\u2581the", "\u2581cat", "\u2581sat"]
        image = C.convert(model, 16, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False)
        m = S.Machine(image)
        gen, reason = S.generate(m, ids, 100)
        ok2 = reason == 3 and len(ids) + len(gen) == 17     # ran into the context limit
        m = S.Machine(image)
        gen0, reason0 = S.generate(m, ids, 0)
        ok3 = gen0 == [] and reason0 == 1
        ok = ok1 and ok2 and ok3
        print(f"{'PASS' if ok else 'FAIL'}  tokenizer / context-full / n=0       "
              f"({words}, ctx stop {reason} after {len(gen)}, n=0 -> {reason0})")
        return ok


if __name__ == "__main__":
    results = [
        run_case("F32 weights", Q.F32),
        run_case("F16 weights", Q.F16),
        run_case("Q8_0 weights (native)", Q.Q8_0),
        run_case("Q4_0 weights (native)", Q.Q4_0),
        run_case("Q8_0, tied classifier", Q.Q8_0, tied=True),
        run_case("F32 repacked to Q4_0", Q.F32, conv_wtype="q4_0", tol=0.3,
                 need_exact_tokens=False),
        test_tokenizer_and_stops(),
    ]
    try:
        results.append(run_case("Q4_K repacked to Q8_0 (fallback)", Q.Q4_K, tol=0.05,
                                need_exact_tokens=False))
    except NotImplementedError as e:
        print(f"SKIP  Q4_K input: gguf-py cannot quantize it ({e})")
    print("ALL PASSED" if all(results) else "FAILURES")
    sys.exit(0 if all(results) else 1)
