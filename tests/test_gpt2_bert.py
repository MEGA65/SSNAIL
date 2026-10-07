#!/usr/bin/env python3
"""End-to-end tests for the gpt2 and bert back ends: random small models
written as GGUF, converted, run on the emulator, and compared against
independent numpy implementations that read the GGUF directly."""

import os
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

GELU = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def ln(v, w, b, eps):
    v = v - v.mean()
    return v / np.sqrt(np.mean(v * v) + eps) * w + b


class Writer:
    def __init__(self, path, arch, wtype, seed):
        self.w = gguf.GGUFWriter(path, arch)
        self.wtype = wtype
        self.rng = np.random.default_rng(seed)

    def mat(self, name, rows, cols, scale=None):
        m = (self.rng.standard_normal((rows, cols)) * (scale or 1 / np.sqrt(cols))).astype(np.float32)
        if self.wtype == Q.F32:
            self.w.add_tensor(name, m)
        elif self.wtype == Q.F16:
            self.w.add_tensor(name, m.astype(np.float16))
        else:
            qb = quantize(m, self.wtype)
            self.w.add_tensor(name, qb, raw_shape=qb.shape, raw_dtype=self.wtype)

    def vec(self, name, n, centre=0.0, scale=0.1):
        self.w.add_tensor(name, (centre + scale * self.rng.standard_normal(n)).astype(np.float32))

    def norm(self, name, n):
        self.vec(f"{name}.weight", n, 1.0)
        self.vec(f"{name}.bias", n, 0.0)

    def close(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


def load(path):
    r = GGUFReader(path)
    W = {}
    for t in r.tensors:
        raw = np.asarray(t.data)
        a = raw.astype(np.float64) if t.tensor_type in (Q.F32, Q.F16) else \
            dequantize(raw.view(np.uint8), t.tensor_type).astype(np.float64)
        ne = [int(x) for x in t.shape]
        W[t.name] = a.reshape(list(reversed(ne)) if len(ne) > 1 else ne)
    return W


def attention(q, K, V, nh):
    hd = q.size // nh
    out = np.zeros_like(q)
    for h in range(nh):
        sl = slice(h * hd, (h + 1) * hd)
        s = K[:, sl] @ q[sl] / np.sqrt(hd)
        a = np.exp(s - s.max()); a /= a.sum()
        out[sl] = a @ V[:, sl]
    return out


# --- GPT-2 --------------------------------------------------------------------
GPT2_WORDS = ["the", "cat", "sat", "on", "mat", "once", "upon", "time", "happy", "dog", "a"]


def make_gpt2(path, wtype, dim=128, layers=2, heads=4, hidden=512, ctx=64, tied=True, seed=3,
              add_bos=None):
    B2U = C.B2U
    tokens = [B2U[b] for b in range(256)]
    merges = []
    # Space-prefixed variants first, so their merges rank higher, as in a real
    # GPT-2 vocabulary.
    for prefix in ("\u0120", ""):                  # 'Ġ' = space
        for wd in GPT2_WORDS:
            parts = ([prefix] if prefix else []) + list(wd)
            cur = parts[0]
            for ch in parts[1:]:
                mg = f"{cur} {ch}"
                if mg not in merges:
                    merges.append(mg)
                cur = cur + ch
                if cur not in tokens:
                    tokens.append(cur)
    tokens.append("<|endoftext|>")
    vocab = len(tokens)
    vocab_pad = (vocab + 31) // 32 * 32
    tokens += [f"<|pad{i}|>" for i in range(vocab_pad - vocab)]
    W = Writer(path, "gpt2", wtype, seed)
    w = W.w
    w.add_context_length(ctx)
    w.add_embedding_length(dim)
    w.add_block_count(layers)
    w.add_feed_forward_length(hidden)
    w.add_head_count(heads)
    w.add_layer_norm_eps(1e-5)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_token_types([1] * (vocab - 1) + [3] + [3] * (vocab_pad - vocab))
    w.add_eos_token_id(vocab - 1)
    if add_bos is not None:
        w.add_bos_token_id(vocab - 1)
        w.add_bool("tokenizer.ggml.add_bos_token", add_bos)
    W.mat("token_embd.weight", vocab_pad, dim, scale=1.0)
    W.mat("position_embd.weight", ctx, dim, scale=0.5)
    for l in range(layers):
        b = f"blk.{l}."
        W.norm(b + "attn_norm", dim)
        W.mat(b + "attn_qkv.weight", 3 * dim, dim)
        W.vec(b + "attn_qkv.bias", 3 * dim)
        W.mat(b + "attn_output.weight", dim, dim)
        W.vec(b + "attn_output.bias", dim)
        W.norm(b + "ffn_norm", dim)
        W.mat(b + "ffn_up.weight", hidden, dim)
        W.vec(b + "ffn_up.bias", hidden)
        W.mat(b + "ffn_down.weight", dim, hidden)
        W.vec(b + "ffn_down.bias", dim)
    W.norm("output_norm", dim)
    if not tied:
        W.mat("output.weight", vocab_pad, dim)
    W.close()


def ref_gpt2(path, tokens, heads=4, eps=1e-5):
    W = load(path)
    dim = W["token_embd.weight"].shape[1]
    L = sum(1 for k in W if k.endswith("attn_qkv.weight"))
    cls = W.get("output.weight", W["token_embd.weight"])
    K = [[] for _ in range(L)]
    V = [[] for _ in range(L)]
    out = []
    for pos, t in enumerate(tokens):
        x = W["token_embd.weight"][t] + W["position_embd.weight"][pos]
        for l in range(L):
            g = lambda n: W[f"blk.{l}.{n}"]
            h = ln(x, g("attn_norm.weight"), g("attn_norm.bias"), eps)
            qkv = g("attn_qkv.weight") @ h + g("attn_qkv.bias")
            q, k, v = qkv[:dim], qkv[dim:2 * dim], qkv[2 * dim:]
            K[l].append(k); V[l].append(v)
            x = x + g("attn_output.weight") @ attention(q, np.array(K[l]), np.array(V[l]), heads) \
                + g("attn_output.bias")
            h = ln(x, g("ffn_norm.weight"), g("ffn_norm.bias"), eps)
            x = x + g("ffn_down.weight") @ GELU(g("ffn_up.weight") @ h + g("ffn_up.bias")) \
                + g("ffn_down.bias")
        out.append(cls @ ln(x, W["output_norm.weight"], W["output_norm.bias"], eps))
    return np.array(out)


def gpt2_case(name, wtype, tied=True):
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "m.gguf")
        make_gpt2(g, wtype, tied=tied)
        model = C.Model(g)
        image = C.convert(model, 48, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False)
        m = S.Machine(image)
        seen = []
        m.on_argmax = seen.append
        prompt = C.encode(model, "once upon a time the cat")
        gen, reason = S.generate(m, prompt, 10)
        ref = ref_gpt2(g, prompt + gen)
        sim = np.array(seen)
        err = np.max(np.abs(sim - ref[:len(sim)])) / np.max(np.abs(ref))
        tok_ok = list(np.argmax(ref[len(prompt) - 1:len(sim)], axis=1)) == gen
        words = [model.field("tokenizer.ggml.tokens")[i] for i in prompt]
        tk_ok = words == ["once", "\u0120upon", "\u0120a", "\u0120time", "\u0120the", "\u0120cat"]
        ok = err < 2e-3 and tok_ok and tk_ok
        print(f"{'PASS' if ok else 'FAIL'}  gpt2 {name:28s} rel.err {err:.2e}  tokens "
              f"{'match' if tok_ok else 'DIFFER'}  tokenizer {'ok' if tk_ok else words}")
        return ok


# --- BERT ---------------------------------------------------------------------
def make_bert(path, wtype, dim=128, layers=2, heads=4, hidden=256, ctx=32, pooling=1, seed=5):
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for wd in ["the", "cat", "play", "dog", "run", "jump", "lamp", "troll", "throw", "at", "a"]:
        tokens.append("\u2581" + wd)
    tokens += ["ing", "s", "ed", "er", ",", ".", "\u2581,", "\u2581."]
    tokens += [f"\u2581w{i}" for i in range((len(tokens) + 31) // 32 * 32 - len(tokens))]
    vocab = len(tokens)
    W = Writer(path, "bert", wtype, seed)
    w = W.w
    w.add_context_length(ctx)
    w.add_embedding_length(dim)
    w.add_block_count(layers)
    w.add_feed_forward_length(hidden)
    w.add_head_count(heads)
    w.add_layer_norm_eps(1e-12)
    w.add_bool("bert.attention.causal", False)
    w.add_uint32("bert.pooling_type", pooling)
    w.add_tokenizer_model("bert")
    w.add_token_list(tokens)
    w.add_token_types([3] * 5 + [1] * (vocab - 5))
    w.add_uint32("tokenizer.ggml.cls_token_id", 2)
    w.add_uint32("tokenizer.ggml.seperator_token_id", 3)
    W.mat("token_embd.weight", vocab, dim, scale=1.0)
    W.mat("token_types.weight", 2, dim, scale=0.3)
    W.mat("position_embd.weight", ctx, dim, scale=0.5)
    W.norm("token_embd_norm", dim)
    for l in range(layers):
        b = f"blk.{l}."
        for n in ("attn_q", "attn_k", "attn_v", "attn_output"):
            W.mat(b + n + ".weight", dim, dim)
            W.vec(b + n + ".bias", dim)
        W.norm(b + "attn_output_norm", dim)
        W.mat(b + "ffn_up.weight", hidden, dim)
        W.vec(b + "ffn_up.bias", hidden)
        W.mat(b + "ffn_down.weight", dim, hidden)
        W.vec(b + "ffn_down.bias", dim)
        W.norm(b + "layer_output_norm", dim)
    W.close()


def ref_bert(path, tokens, heads=4, eps=1e-12, pooling=1):
    W = load(path)
    L = sum(1 for k in W if k.endswith("attn_q.weight"))
    X = np.array([ln(W["token_embd.weight"][t] + W["position_embd.weight"][i]
                     + W["token_types.weight"][0],
                     W["token_embd_norm.weight"], W["token_embd_norm.bias"], eps)
                  for i, t in enumerate(tokens)])
    for l in range(L):
        g = lambda n: W[f"blk.{l}.{n}"]
        Qm = X @ g("attn_q.weight").T + g("attn_q.bias")
        Km = X @ g("attn_k.weight").T + g("attn_k.bias")
        Vm = X @ g("attn_v.weight").T + g("attn_v.bias")
        new = []
        for i in range(len(tokens)):
            x = X[i] + g("attn_output.weight") @ attention(Qm[i], Km, Vm, heads) + g("attn_output.bias")
            x = ln(x, g("attn_output_norm.weight"), g("attn_output_norm.bias"), eps)
            x = x + g("ffn_down.weight") @ GELU(g("ffn_up.weight") @ x + g("ffn_up.bias")) \
                + g("ffn_down.bias")
            new.append(ln(x, g("layer_output_norm.weight"), g("layer_output_norm.bias"), eps))
        X = np.array(new)
    return X[0] if pooling == 2 else X.mean(axis=0)


def bert_case(name, wtype, pooling=1):
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, "m.gguf")
        make_bert(g, wtype, pooling=pooling)
        model = C.Model(g)
        image = C.convert(model, None, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False)
        ids = C.encode(model, "Throw the lamp at the troll, playing cats.")
        words = [model.field("tokenizer.ggml.tokens")[i] for i in ids]
        expect = ["[CLS]", "\u2581throw", "\u2581the", "\u2581lamp", "\u2581at", "\u2581the",
                  "\u2581troll", "\u2581,", "\u2581play", "ing", "\u2581cat", "s", "\u2581.", "[SEP]"]
        vec, reason = S.encode_sequence(S.Machine(image), ids)
        ref = ref_bert(g, ids, pooling=pooling)
        err = np.max(np.abs(vec - ref)) / np.max(np.abs(ref))
        ok = err < 2e-3 and words == expect and reason == 1
        print(f"{'PASS' if ok else 'FAIL'}  bert {name:28s} rel.err {err:.2e}  "
              f"tokenizer {'ok' if words == expect else words}  stop {reason}")
        return ok


def main():
    results = [
        gpt2_case("F32, tied classifier", Q.F32),
        gpt2_case("Q8_0, untied classifier", Q.Q8_0, tied=False),
        gpt2_case("Q4_0", Q.Q4_0),
        bert_case("F32, mean pooling", Q.F32),
        bert_case("Q8_0, CLS pooling", Q.Q8_0, pooling=2),
        bert_case("F16, mean pooling", Q.F16),
    ]
    results.append(layouts_case())
    print("ALL PASSED" if all(results) else "FAILURES")
    sys.exit(0 if all(results) else 1)


def layouts_case():
    """The same model in the 8, 64 and 72 MB layouts must behave identically;
    big layouts are loaded through the staging buffer with read-back."""
    import struct
    import ssnail_isa as I
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for arch, maker in (("gpt2", make_gpt2), ("bert", make_bert)):
            g = os.path.join(d, arch + ".gguf")
            maker(g, Q.Q8_0)
            model = C.Model(g)
            ids = C.encode(model, "throw the lamp at the troll" if arch == "bert" else "once upon a time")
            results = {}
            for mb in (8, 64, 72):
                image = C.convert(model, 32, "keep", C.FMT_BY_NAME["q8_0"], mb, verbose=False)
                m = S.Machine(image)
                base = struct.unpack("<I", image[I.H_LOAD_BASE:I.H_LOAD_BASE + 4])[0]
                tok = S.header(m, I.H_TOKENS)
                if arch == "bert":
                    vec, _ = S.encode_sequence(m, ids)
                    results[mb] = vec.tobytes()
                else:
                    results[mb] = tuple(S.generate(m, ids, 8)[0])
                staged = getattr(m, "load_sectors", 0)
                where = "window" if I.HYPERRAM_BASE <= tok < I.WINDOW_END else "after scratch"
                print(f"      {arch} {mb:2d} MB: payload at ${base:07X}, tokens in {where}, "
                      f"loaded as {staged} sectors through the load port")
                if staged == 0 or (mb > 8 and where != "window"):
                    ok = False
            same = results[8] == results[64] == results[72]
            ok = ok and same
            print(f"{'PASS' if same else 'FAIL'}  {arch} identical across 8/64/72 MB layouts")
    return ok


if __name__ == "__main__":
    main()
