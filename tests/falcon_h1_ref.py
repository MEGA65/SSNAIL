"""Falcon-H1 reference forward pass from a GGUF (numpy, float64), one token at
a time with the Mamba-2 recurrent state and conv history, as the hardware will
run it.  Semantics: transformers' FalconH1 (single-token decode path)."""
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import dequantize


class FalconH1Ref:
    def __init__(self, path):
        r = GGUFReader(path)
        def f(k):
            v = r.fields["falcon-h1." + k]
            return v.parts[v.data[0]].tolist()[0]
        self.dim, self.L = f("embedding_length"), f("block_count")
        self.nh, self.nkv = f("attention.head_count"), f("attention.head_count_kv")
        self.hd = f("attention.key_length")
        self.eps, self.theta = f("attention.layer_norm_rms_epsilon"), f("rope.freq_base")
        self.d_ssm, self.d_state = f("ssm.inner_size"), f("ssm.state_size")
        self.n_sh, self.d_conv = f("ssm.time_step_rank"), f("ssm.conv_kernel")
        self.sh = self.d_ssm // self.n_sh            # channels per SSM head
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
        self.dexp = np.exp            # (the decay exp: a hook for numerics experiments)
        self.reset()

    def reset(self):
        self.K = [[] for _ in range(self.L)]
        self.V = [[] for _ in range(self.L)]
        cd = self.d_ssm + 2 * self.d_state
        self.conv = [np.zeros((self.d_conv - 1, cd)) for _ in range(self.L)]
        self.h = [np.zeros((self.n_sh, self.sh, self.d_state)) for _ in range(self.L)]
        self.pos = 0

    def rms(self, v, g):
        return v / np.sqrt(np.mean(v * v) + self.eps) * g

    def rope(self, v, pos):                      # rotate-half (as transformers)
        hd = self.hd
        v = v.reshape(-1, hd).copy()
        half = hd // 2
        inv = 1.0 / self.theta ** (np.arange(half) * 2 / hd)
        a = pos * inv
        c, s = np.cos(a), np.sin(a)
        x0, x1 = v[:, :half].copy(), v[:, half:].copy()
        v[:, :half] = x0 * c - x1 * s
        v[:, half:] = x1 * c + x0 * s
        return v.reshape(-1)

    def step(self, tok):
        W, pos = self.W, self.pos
        x = W["token_embd.weight"][tok].copy()
        for l in range(self.L):
            g = lambda n: W[f"blk.{l}.{n}"]
            xb = self.rms(x, g("attn_norm.weight"))
            # Mamba-2
            p = g("ssm_in.weight") @ xb
            z, xbc, dt = p[:self.d_ssm], p[self.d_ssm:2 * self.d_ssm + 2 * self.d_state], p[-self.n_sh:]
            hist = np.vstack([self.conv[l], xbc[None, :]])             # oldest first, 4 rows
            self.conv[l] = hist[1:]
            cw = g("ssm_conv1d.weight")                                # (channels, 4)
            u = (hist.T * cw).sum(axis=1) + g("ssm_conv1d.bias")
            u = u / (1 + np.exp(-u))                                   # SiLU
            xs, B, C = u[:self.d_ssm], u[self.d_ssm:self.d_ssm + self.d_state], u[-self.d_state:]
            dt = np.log1p(np.exp(dt + g("ssm_dt.bias")))               # softplus
            A = g("ssm_a").reshape(-1)
            dA = self.dexp(dt * A)
            xh = xs.reshape(self.n_sh, self.sh)
            self.h[l] = self.h[l] * dA[:, None, None] + (dt[:, None] * xh)[:, :, None] * B[None, None, :]
            y = (self.h[l] @ C) + g("ssm_d").reshape(-1)[:, None] * xh
            y = y.reshape(-1) * (z / (1 + np.exp(-z)))
            m_out = g("ssm_out.weight") @ y
            # attention (in parallel, on the same xb)
            q = self.rope(g("attn_q.weight") @ xb, pos)
            self.K[l].append(self.rope(g("attn_k.weight") @ xb, pos))
            self.V[l].append(g("attn_v.weight") @ xb)
            Ks, Vs = np.array(self.K[l]), np.array(self.V[l])
            hd, att = self.hd, np.zeros(self.nh * self.hd)
            for hh in range(self.nh):
                kh = hh // (self.nh // self.nkv)
                s = Ks[:, kh * hd:(kh + 1) * hd] @ q[hh * hd:(hh + 1) * hd] / np.sqrt(hd)
                a = np.exp(s - s.max()); a /= a.sum()
                att[hh * hd:(hh + 1) * hd] = a @ Vs[:, kh * hd:(kh + 1) * hd]
            x = x + m_out + g("attn_output.weight") @ att
            xb = self.rms(x, W[f"blk.{l}.ffn_norm"] if f"blk.{l}.ffn_norm" in W else g("ffn_norm.weight"))
            gt = g("ffn_gate.weight") @ xb
            x = x + g("ffn_down.weight") @ (gt / (1 + np.exp(-gt)) * (g("ffn_up.weight") @ xb))
        self.pos += 1
        return self.cls @ self.rms(x, W["output_norm.weight"])
