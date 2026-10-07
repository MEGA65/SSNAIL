#!/usr/bin/env python3
"""Reference emulator for SSNAIL memory images.

Executes an image produced by ssnail_convert.py exactly as the SSNAIL
hardware should, using float arithmetic.  It is the golden model: the
hardware is checked against it, and it is checked against independent
implementations (see tests/).

    ssnail_sim.py model.ssnail --tokens 1,450,4799 -n 40

Token ids are model-specific: get them with ssnail_convert.py tokenize, or use
ssnail_chat.py, which tokenizes for you.
"""

import argparse
import struct
import sys
import time

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

import ssnail_isa as I
import ssnail_hw as H


def row_bytes(fmt, n):
    block, tsize = GGML_QUANT_SIZES[GGMLQuantizationType(fmt)]
    assert n % block == 0, f"{n} elements is not a whole number of blocks"
    return n // block * tsize


class SsnailError(Exception):
    pass


def gelu(x):
    """GELU, tanh approximation (as ggml uses for GPT-2 and BERT)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class Machine:
    def __init__(self, image, mem_bytes=None):
        if image[:4] != I.MAGIC:
            raise SsnailError("not an SSNAIL image")
        u32 = lambda f: struct.unpack("<I", image[f:f + 4])[0]
        if mem_bytes is None:
            need = u32(I.H_MEM_NEEDED)
            mem_bytes = max(8 << 20, (need + (1 << 20) - 1) & ~((1 << 20) - 1))
        self.mem = bytearray(mem_bytes)
        self.R = [0] * 16
        self.A = [0] * 16
        self.N = [0] * 3
        self.pc = 0
        self.stats = dict(instructions=0, bytes_read=0, bytes_written=0)
        self.on_argmax = self.on_sync = None
        self.trace = False
        self.temperature, self.top_k = 0.0, 0
        self.rep_penalty, self.rep_window = 1.0, 64
        self.rng = np.random.default_rng()
        # Hardware-numerics mode: compute exactly as the SSNAIL datapath
        # (ssnail_hw.py).  Off: float reference.
        self.hw = False
        load_image(self, image)
        self.stats = dict(instructions=0, bytes_read=0, bytes_written=0)

    # --- memory ---------------------------------------------------------------
    def _off(self, addr, n):
        off = addr - I.HYPERRAM_BASE
        if off < 0 or off + n > len(self.mem):
            raise SsnailError(f"address fault: ${addr:07X} (+{n}) at PC ${self.pc:07X}")
        return off

    def read(self, addr, n):
        o = self._off(addr, n)
        self.stats["bytes_read"] += n
        return bytes(self.mem[o:o + n])

    def write(self, addr, data):
        o = self._off(addr, len(data))
        self.stats["bytes_written"] += len(data)
        self.mem[o:o + len(data)] = data

    def u32(self, addr):
        return struct.unpack("<I", self.read(addr, 4))[0]

    def put_u32(self, addr, v):
        self.write(addr, struct.pack("<I", v & 0xFFFFFFFF))

    def f32v(self, addr, n):
        return np.frombuffer(self.read(addr, 4 * n), dtype="<f4").astype(np.float32)

    def put_f32v(self, addr, v):
        self.write(addr, np.asarray(v, dtype="<f4").tobytes())

    def ea(self, operand):
        """Resolve an address operand."""
        if operand & 0x80000000:
            return self.A[(operand >> 24) & 0xF] + (operand & 0xFFFFFF)
        return operand & 0xFFFFFFF

    def weights(self, fmt, addr, rows, cols):
        raw = np.frombuffer(self.read(addr, rows * row_bytes(fmt, cols)), dtype=np.uint8)
        if fmt == I.FMT_F32:
            w = raw.view("<f4")
        elif fmt == I.FMT_F16:
            w = raw.view("<f2")
        else:
            w = dequantize(raw, GGMLQuantizationType(fmt))
        return np.asarray(w, dtype=np.float32).reshape(rows, cols)

    # --- execution --------------------------------------------------------------
    def reg(self, i):
        return 0 if i == I.RZERO else self.R[i]

    def run(self, start, max_instructions=50_000_000):
        self.pc = start
        while True:
            if self.stats["instructions"] >= max_instructions:
                raise SsnailError("instruction limit reached")
            if self.pc & 15:
                raise SsnailError(f"alignment fault: PC ${self.pc:07X}")
            op, a, b, x, y, z = I.decode(self.read(self.pc, 16))
            self.stats["instructions"] += 1
            if self.trace:
                print(f"${self.pc:07X} {I.OPNAMES.get(op, hex(op)):8s} a={a} b={b} "
                      f"X=${x:08X} Y=${y:08X} Z=${z:08X}", file=sys.stderr)
            npc = self.pc + 16
            N0, N1, N2 = self.N
            if op == I.NOP:
                pass
            elif op == I.SYNC:
                if self.on_sync:
                    self.on_sync()
            elif op == I.HALT:
                return
            elif op == I.COPY:
                self.write(self.ea(y), self.read(self.ea(x), z & 0xFFFF))
            elif op == I.SETN:
                self.N = [x, y, z]
            elif op == I.LI:
                self.R[a] = z
            elif op == I.LDR:
                self.R[a] = self.u32(self.ea(x))
            elif op == I.STR:
                self.put_u32(self.ea(x), self.reg(a))
            elif op == I.ADDI:
                zs = z - (1 << 32) if z & 0x80000000 else z
                self.R[a] = (self.reg(b) + zs) & 0xFFFFFFFF
            elif op == I.LEA:
                self.A[a] = (self.ea(x) + self.reg(b) * z) & 0xFFFFFFF
            elif op in (I.BEQ, I.BNE, I.BLT):
                ra, rb = self.reg(a), self.reg(b)
                if (op == I.BEQ and ra == rb) or (op == I.BNE and ra != rb) \
                        or (op == I.BLT and ra < rb):
                    npc = self.ea(x)
            elif op == I.GEMV:
                xv = self.f32v(self.ea(y), N0)
                if self.hw and a in (I.FMT_Q8_0, I.FMT_Q4_0):
                    raw = self.read(self.ea(x), N1 * row_bytes(a, N0))
                    v = H.gemv_quant(raw, a, N1, N0, xv)
                elif self.hw:
                    v = H.gemv_float(self.weights(a, self.ea(x), N1, N0), xv)
                else:
                    v = (self.weights(a, self.ea(x), N1, N0) @ xv).astype(np.float32)
                if b & I.GEMV_ACC:
                    v = (v + self.f32v(self.ea(z), N1)).astype(np.float32)
                if b & I.GEMV_F16OUT:
                    self.write(self.ea(z), np.asarray(v, dtype="<f2").tobytes())
                else:
                    self.put_f32v(self.ea(z), v)
            elif op == I.CVT16:
                v = self.f32v(self.ea(x), N0)
                self.write(self.ea(z), v.astype("<f2").tobytes())
            elif op == I.DEQROW:
                self.put_f32v(self.ea(z), self.weights(a, self.ea(x), 1, N0)[0])
            elif op == I.ARGMAX:
                v = self.f32v(self.ea(x), N0)
                if self.on_argmax:
                    self.on_argmax(v.copy())
                self.R[a] = self.choose(v)
            elif op in (I.VADD, I.VMUL, I.SILUMUL):
                p, q = self.f32v(self.ea(x), N0), self.f32v(self.ea(y), N0)
                if op == I.VADD:
                    r = p + q
                elif op == I.VMUL:
                    r = p * q
                elif self.hw:
                    r = H.silumul(p, q)
                else:
                    r = p / (1.0 + np.exp(-p)) * q
                self.put_f32v(self.ea(z), r)
            elif op == I.RMSNORM:
                eps = struct.unpack("<f", struct.pack("<I", N2))[0]
                p, g = self.f32v(self.ea(x), N0), self.f32v(self.ea(y), N0)
                if self.hw:
                    r = H.rmsnorm(p, g, eps)
                else:
                    r = p / np.sqrt(np.mean(p.astype(np.float64) ** 2) + eps) * g
                self.put_f32v(self.ea(z), r)
            elif op == I.ROPE:
                vaddr = self.ea(x)
                v = self.f32v(vaddr, N0)
                cs = self.f32v(self.ea(y), N1)          # N1/2 (cos, sin) pairs
                c = np.tile(cs[0::2], N0 // N1)
                s = np.tile(cs[1::2], N0 // N1)
                v0, v1 = v[0::2].copy(), v[1::2].copy()
                v[0::2] = v0 * c - v1 * s
                v[1::2] = v0 * s + v1 * c
                self.put_f32v(vaddr, v)
            elif op == I.ATTN:
                self.attn(self.ea(x), self.ea(y), self.ea(z), self.reg(a))
            elif op == I.LAYERNORM:
                eps = struct.unpack("<f", struct.pack("<I", N2))[0]
                gb = self.f32v(self.ea(y), 2 * N0)
                if self.hw:
                    r = H.layernorm(self.f32v(self.ea(x), N0), gb[:N0], gb[N0:], eps)
                else:
                    p = self.f32v(self.ea(x), N0).astype(np.float64)
                    p = p - p.mean()
                    r = p / np.sqrt(np.mean(p * p) + eps) * gb[:N0] + gb[N0:]
                self.put_f32v(self.ea(z), r)
            elif op == I.GELU:
                p = self.f32v(self.ea(x), N0)
                self.put_f32v(self.ea(z), H.gelu(p) if self.hw else gelu(p))
            elif op == I.MEANROWS:
                rows = self.reg(a)
                M = self.f32v(self.ea(x), rows * N0).reshape(rows, N0)
                self.put_f32v(self.ea(z), H.meanrows(M) if self.hw else M.mean(axis=0))
            else:
                raise SsnailError(f"illegal opcode ${op:02X} at PC ${self.pc:07X}")
            self.pc = npc

    def recent_tokens(self, n):
        """The last n tokens of context (for the repetition penalty)."""
        u32 = lambda f: struct.unpack("<I", self.mem[f:f + 4])[0]
        pos = u32(I.R_POS)
        t0 = u32(I.H_TOKENS) - I.HYPERRAM_BASE
        lo = max(0, pos + 1 - n)
        return list(struct.unpack(f"<{pos + 1 - lo}I", self.mem[t0 + 4 * lo:t0 + 4 * (pos + 1)]))

    def choose(self, logits):
        if self.temperature <= 0:
            return int(np.argmax(logits))
        z = logits.astype(np.float64)
        if self.rep_penalty != 1.0:
            # As llama.cpp: shrink the logits of recently used tokens
            for t in set(self.recent_tokens(self.rep_window)):
                if t < len(z):
                    z[t] = z[t] / self.rep_penalty if z[t] > 0 else z[t] * self.rep_penalty
        z = z / self.temperature
        if self.top_k > 0 and self.top_k < len(z):
            cut = np.partition(z, -self.top_k)[-self.top_k]
            z = np.where(z >= cut, z, -np.inf)
        p = np.exp(z - z.max())
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def kv(self, addr, n, fmt):
        if fmt == I.KV_F16:
            return np.frombuffer(self.read(addr, 2 * n), dtype="<f2").astype(np.float32)
        return self.f32v(addr, n)

    def attn(self, block, q_a, out_a, pos):
        k_a, v_a, nh, nkv, hd, kvf, _r1, _r2 = struct.unpack(
            "<8I", self.read(block, 32))
        kvd = nkv * hd
        q = self.f32v(q_a, nh * hd).reshape(nh, hd)
        K = self.kv(k_a, (pos + 1) * kvd, kvf).reshape(pos + 1, nkv, hd)
        V = self.kv(v_a, (pos + 1) * kvd, kvf).reshape(pos + 1, nkv, hd)
        out = np.empty((nh, hd), dtype=np.float32)
        group = nh // nkv
        if self.hw:
            for h in range(nh):
                out[h] = H.attention_head(q[h], K[:, h // group, :], V[:, h // group, :])
            self.put_f32v(out_a, out.reshape(-1))
            return
        for h in range(nh):
            s = K[:, h // group, :] @ q[h] / np.sqrt(hd)
            s = np.exp(s - s.max())
            s /= s.sum()
            out[h] = s @ V[:, h // group, :]
        self.put_f32v(out_a, out.reshape(-1))


# --- Load port (SSNAIL registers $14-$18) --------------------------------------
class LoadPort:
    """Model of SSNAIL's load port: a pointer, a data register, and a 1 KB
    ring indexed by destination address.  Complete 256-byte blocks go to RAM
    as the pointer leaves them; a commit flushes the rest, then moves the
    pointer.  The model writes memory exactly when the hardware would."""

    def __init__(self, m):
        self.m = m
        self.ptr = self.drain = 0
        self.error = False
        self.ring = {}
        self.sectors = 0

    def ready(self):
        return True       # the model drains instantly

    def commit(self, new_ptr, clear_error=False):
        self._flush(self.ptr)
        self.ptr = self.drain = new_ptr
        if clear_error:
            self.error = False

    def write(self, data):
        for b in data:
            self.ring[self.ptr] = b
            self.ptr += 1
            if self.ptr & 0xFF == 0:
                self._flush(self.ptr)

    def _flush(self, end):
        if end <= self.drain:
            return
        o = self.drain - I.HYPERRAM_BASE
        if o < 0 or end - I.HYPERRAM_BASE > len(self.m.mem):
            self.error = True
        else:
            self.m.mem[o:end - I.HYPERRAM_BASE] = bytes(self.ring.pop(a, 0)
                                                       for a in range(self.drain, end))
        self.drain = end


def load_image(m, image):
    """The MEGA65 loader: stream the file through the load port, a 512-byte
    sector at a time, moving the pointer at each segment boundary (which
    need not fall on a sector boundary)."""
    u32 = lambda f: struct.unpack("<I", image[f:f + 4])[0]
    host_len = u32(I.H_HOST_LEN)
    segments = [(I.HYPERRAM_BASE, I.HEADER_SIZE)]
    if host_len:
        segments.append((u32(I.H_HOST_BASE), host_len))
    segments.append((u32(I.H_LOAD_BASE) or I.HYPERRAM_BASE + I.HEADER_SIZE,
                     len(image) - I.HEADER_SIZE - host_len))
    seg_starts, off = {}, 0
    for addr, n in segments:
        seg_starts[off] = addr
        off += n
    lp = LoadPort(m)
    lp.commit(I.HYPERRAM_BASE, clear_error=True)
    for sec in range(0, len(image), 512):
        while not lp.ready():
            pass
        sector = image[sec:sec + 512]
        cut = sec
        # a segment may start anywhere in the sector, including its first byte
        for s0 in sorted(k for k in seg_starts if k > 0 and sec <= k < sec + len(sector)):
            lp.write(sector[cut - sec:s0 - sec])
            lp.commit(seg_starts[s0])
            cut = s0
        lp.write(sector[cut - sec:])
        lp.sectors += 1
    lp.commit(I.HYPERRAM_BASE)          # final flush
    if lp.error:
        raise SsnailError("load port reported ERROR (data outside RAM)")
    m.load_sectors = lp.sectors


# --- Host-side helpers (what MEGA65 software would do) -------------------------
def description(m):
    """The image's description (header H_DESC): "name (arch) [type]"."""
    raw = bytes(m.mem[I.H_DESC:I.H_DESC + I.H_DESC_SIZE])
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def header(m, field):
    return struct.unpack("<I", m.mem[field:field + 4])[0]


def set_rt(m, field, value):
    m.mem[field:field + 4] = struct.pack("<I", value & 0xFFFFFFFF)


def vocab_piece(m, tok):
    vocab = header(m, I.H_VOCAB) - I.HYPERRAM_BASE
    o0, o1 = struct.unpack("<II", m.mem[vocab + 4 * tok: vocab + 4 * tok + 8])
    return bytes(m.mem[vocab + o0: vocab + o1])


def generate(m, prompt, n_generate):
    """Write the prompt, run the script, and return the generated tokens."""
    tokens = header(m, I.H_TOKENS) - I.HYPERRAM_BASE
    for i, t in enumerate(prompt):
        m.mem[tokens + 4 * i: tokens + 4 * i + 4] = struct.pack("<I", t)
    set_rt(m, I.R_POS, 0)
    set_rt(m, I.R_PROMPT_LEN, len(prompt))
    set_rt(m, I.R_N_GENERATE, n_generate)
    set_rt(m, I.R_GENERATED, 0)
    set_rt(m, I.R_STOP_REQUEST, 0)
    m.run(header(m, I.H_CODE))
    n = header(m, I.R_GENERATED)
    out = [struct.unpack("<I", m.mem[tokens + 4 * i: tokens + 4 * i + 4])[0]
           for i in range(len(prompt), len(prompt) + n)]
    return out, header(m, I.R_STOP_REASON)


def encode_sequence(m, tokens):
    """Run an encoder (BERT) image over a token sequence; returns the
    pooled output vector."""
    taddr = header(m, I.H_TOKENS) - I.HYPERRAM_BASE
    for i, t in enumerate(tokens):
        m.mem[taddr + 4 * i: taddr + 4 * i + 4] = struct.pack("<I", t)
    set_rt(m, I.R_PROMPT_LEN, len(tokens))
    m.run(header(m, I.H_CODE))
    out = header(m, I.H_OUTPUT) - I.HYPERRAM_BASE
    n = header(m, I.H_OUTDIM)
    return np.frombuffer(bytes(m.mem[out:out + 4 * n]), dtype="<f4").copy(), \
        header(m, I.R_STOP_REASON)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--tokens", required=True,
                    help="comma-separated prompt token ids (see ssnail_convert.py tokenize)")
    ap.add_argument("-n", type=int, default=32, help="tokens to generate")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--bandwidth", type=float, default=130.0,
                    help="assumed sustained MB/s, for the speed estimate")
    args = ap.parse_args()

    m = Machine(open(args.image, "rb").read())
    m.trace = args.trace
    prompt = [int(t) for t in args.tokens.split(",")]
    t0 = time.time()
    if struct.unpack("<H", m.mem[6:8])[0] & I.H_FLAG_ENCODER:
        vec, reason = encode_sequence(m, prompt)
        print(f"embedding ({len(vec)} values, |v| = {np.linalg.norm(vec):.4f}):")
        print(np.array2string(vec[:16], precision=4) + (" ..." if len(vec) > 16 else ""))
        mb = (m.stats["bytes_read"] + m.stats["bytes_written"]) / 1e6
        print(f"[stop reason {reason}, {m.stats['instructions']} instructions; RAM traffic "
              f"{mb:.2f} MB -> ~{mb / args.bandwidth * 1000:.0f} ms at {args.bandwidth:.0f} MB/s]",
              file=sys.stderr)
        return
    out, reason = generate(m, prompt, args.n)
    dt = time.time() - t0
    text = b"".join(vocab_piece(m, t) for t in prompt + out)
    print(text.decode("utf-8", errors="replace"))
    print(f"\n[{len(out)} tokens, stop reason {reason}, {m.stats['instructions']} "
          f"instructions, {dt:.2f}s emulated]", file=sys.stderr)
    steps = len(prompt) + len(out)
    mb = (m.stats["bytes_read"] + m.stats["bytes_written"]) / 1e6
    print(f"[RAM traffic {mb / steps:.2f} MB/token -> ~{args.bandwidth / (mb / steps):.1f} "
          f"tokens/s at {args.bandwidth:.0f} MB/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
