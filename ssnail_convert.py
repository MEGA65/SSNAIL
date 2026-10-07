#!/usr/bin/env python3
"""Convert a GGUF model into an SSNAIL memory image for the MEGA65.

    ssnail_convert.py convert model.gguf -o model.ssnail [--ctx 256] [--wtype keep]
    ssnail_convert.py tokenize model.gguf "Once upon a time"
    ssnail_convert.py info model.gguf

Supported architectures (GGUF general.architecture):
    llama   Llama family, TinyStories llama2.c exports, SmolLM, ...  (decoder)
    gpt2    GPT-2 family, e.g. the TinyStories GPT-2 models             (decoder)
    bert    BERT family, e.g. all-MiniLM sentence embedders             (encoder)

The image is loaded at $8000000 (attic RAM).  Layout, low to high:

    $8000000  header + runtime block (256 bytes)
    $8000100  SSNAIL script
              weights, norms, tables, attention parameter blocks, vocab
    --------  end of file
              scratch activations and KV cache (not in the file)
              token buffer: u32 tokens, last, open-ended

Weight matrices already in a natively supported format (F32, F16, BF16,
Q8_0, Q4_0) are copied byte-for-byte; others (e.g. the K-quants in a Q4_K_M
file) are repacked to --fallback (Q8_0 by default).  --wtype forces one
format for every matrix; 'mixed' uses Q8_0 for the layers and Q4_0 for the
big embedding/classifier; 'auto' picks the best that fits (keep, q8_0,
mixed, q4_0), asking first if that means fewer bits than an already-quantised
source.
"""

import argparse
import re
import struct

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize, quantize

import ssnail_isa as I

FMT_BY_NAME = {"f32": I.FMT_F32, "f16": I.FMT_F16, "bf16": I.FMT_BF16,
               "q8_0": I.FMT_Q8_0, "q4_0": I.FMT_Q4_0}
ARCHES = ("llama", "gpt2", "bert")


class ImageTooBig(Exception):
    pass


FLOAT_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16,
               GGMLQuantizationType.BF16, GGMLQuantizationType.F64}


def bpw(fmt):
    """Bits per weight of a ggml type."""
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType(fmt)]
    return size * 8 / block


def source_bpw(m):
    """Element-weighted bits per weight of the source's weight matrices."""
    tot = bits = 0
    for t in m.r.tensors:
        if len(t.shape) > 1 and t.name.endswith(".weight"):
            n = int(np.prod([int(x) for x in t.shape]))
            tot += n
            bits += n * bpw(int(t.tensor_type))
    return bits / max(tot, 1)


def matrix_params(m):
    return sum(int(np.prod([int(x) for x in t.shape])) for t in m.r.tensors
               if len(t.shape) > 1)


AUTO_ORDER = (("keep", "the file's own formats"), ("q8_0", "Q8_0"),
              ("mixed", "Q8_0 layers, Q4_0 embedding/classifier"), ("q4_0", "Q4_0"))


def too_big_message(m, gguf_path, mem_mb, ctx_given):
    """Explain how to get a bigger memory size, suggesting only sizes that the
    model should fit at Q4_0 (estimate)."""
    import os
    stem = os.path.basename(gguf_path)
    stem = stem[:-5] if stem.endswith(".gguf") else stem
    need = matrix_params(m) * 4.5 / 8 * 1.05 + 2 * 1048576
    lines = [f"{stem} does not fit in {mem_mb:g} MB"
             + (" of attic RAM." if mem_mb == 8 else ".")]
    opts = [(72, "attic RAM + SDRAM"), (64, "SDRAM, leaving attic RAM free")]
    fits = [(mb, what) for mb, what in opts if mb > mem_mb and need <= mb * 1048576]
    if fits:
        lines.append("On an R4-R6 (with SDRAM), use a larger memory size:")
        for mb, what in fits:
            lines.append(f"  make {stem}.{mb}mb.chat      ({what})")
        lines.append(f"  or ATTICRAM={fits[0][0]} make {stem}.chat   "
                     f"(python: --mem-mb {fits[0][0]})")
    elif mem_mb < 72:
        lines.append("It is probably too big even for 72 MB (attic RAM + SDRAM).")
    lines.append("Or try a smaller context: CONTEXT_WINDOW=128 (python: --ctx 128).")
    return "\n".join(lines)


# --- GGUF access ------------------------------------------------------------------
class Model:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.tensors = {t.name: t for t in self.r.tensors}
        self.arch = self.field("general.architecture")

    def field(self, key, default=None):
        f = self.r.fields.get(key)
        if f is None:
            return default
        if f.types and f.types[0].name == "ARRAY":
            vals = [f.parts[i] for i in f.data]
            if f.types[-1].name == "STRING":
                return [bytes(v).decode("utf-8", errors="replace") for v in vals]
            return [v.tolist()[0] for v in vals]
        v = f.parts[f.data[0]]
        if f.types[0].name == "STRING":
            return bytes(v).decode("utf-8")
        return v.tolist()[0]

    def af(self, key, default=None):
        """Architecture-specific field, e.g. af('embedding_length')."""
        return self.field(f"{self.arch}.{key}", default)

    def tensor_f32(self, name):
        t = self.tensors[name]
        raw = np.asarray(t.data)
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F64,
                             GGMLQuantizationType.F16):
            w = raw.astype(np.float32)
        else:
            w = dequantize(raw.view(np.uint8), t.tensor_type).astype(np.float32)
        return w.reshape(-1)

    def shape(self, name):
        """(rows, cols) for a matrix, (n,) for a vector."""
        ne = [int(x) for x in self.tensors[name].shape]
        return tuple(reversed(ne)) if len(ne) > 1 else (ne[0],)

    def has(self, name):
        return name in self.tensors


def params(m):
    if m.arch not in ARCHES:
        raise SystemExit(f"Unsupported architecture '{m.arch}' (supported: {', '.join(ARCHES)})")
    p = dict(
        dim=m.af("embedding_length"),
        n_layers=m.af("block_count"),
        hidden=m.af("feed_forward_length"),
        n_heads=m.af("attention.head_count"),
        ctx_train=m.af("context_length", 2048),
    )
    p["n_kv_heads"] = m.af("attention.head_count_kv", p["n_heads"])
    if m.arch == "llama":
        p["eps"] = m.af("attention.layer_norm_rms_epsilon", 1e-5)
        p["rope_base"] = m.af("rope.freq_base", 10000.0)
    else:
        p["eps"] = m.af("attention.layer_norm_epsilon", 1e-5)
    if m.arch == "bert":
        p["pooling"] = m.af("pooling_type", 1)        # 1 = mean, 2 = CLS
    p["head_dim"] = p["dim"] // p["n_heads"]
    p["kv_dim"] = p["n_kv_heads"] * p["head_dim"]
    p["vocab"] = m.shape("token_embd.weight")[0]
    return p


# --- Tokenizers -------------------------------------------------------------------
def bytes_to_unicode():
    """GPT-2's byte <-> printable-character mapping."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def tokenizer_kind(m):
    return m.field("tokenizer.ggml.model", "llama")


def piece_bytes(kind, piece, special):
    """What a vocab entry prints as."""
    if special:
        return b""
    if kind == "gpt2":
        try:
            return bytes(U2B[c] for c in piece)
        except KeyError:
            return piece.encode("utf-8")
    if kind == "llama" and len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            pass
    return piece.replace("\u2581", " ").encode("utf-8")


def fold_ascii(raw):
    """Bytes as plain ASCII (see ssnail_tok.AsciiFolder)."""
    from ssnail_tok import AsciiFolder
    return AsciiFolder().feed(raw).encode("ascii")


def is_plain_ascii(raw):
    return all(b in (9, 10) or 32 <= b < 127 for b in raw)


def special_flags(m):
    types = m.field("tokenizer.ggml.token_type") or []
    tokens = m.field("tokenizer.ggml.tokens")
    out = []
    for i, t in enumerate(tokens):
        tt = types[i] if i < len(types) else 1
        out.append(tt in (2, 3, 5) or t in ("<s>", "</s>", "<unk>", "[CLS]", "[SEP]",
                                            "[PAD]", "[UNK]", "[MASK]"))
    return out


def wants_bos(m):
    """Whether prompts start with the BOS token: the GGUF's add_bos_token if
    present, else the family default (SentencePiece yes, GPT-2 BPE no)."""
    v = m.field("tokenizer.ggml.add_bos_token")
    if v is not None:
        return bool(v)
    return tokenizer_kind(m) == "llama"


def encode(m, text, bos=True):
    kind = tokenizer_kind(m)
    tokens = m.field("tokenizer.ggml.tokens")
    lookup = {t: i for i, t in enumerate(tokens)}
    if kind == "llama":
        ids = _encode_spm(m, tokens, lookup, text)
        if bos and wants_bos(m):
            ids.insert(0, m.field("tokenizer.ggml.bos_token_id", 1))
    elif kind == "gpt2":
        ids = _encode_gpt2(m, lookup, text)
        bid = m.field("tokenizer.ggml.bos_token_id")
        if bos and wants_bos(m) and bid is not None:
            ids.insert(0, bid)
    elif kind == "bert":
        ids = _encode_wordpiece(lookup, text, uncased=True)
        cls = m.field("tokenizer.ggml.cls_token_id", m.field("tokenizer.ggml.bos_token_id", 101))
        sep = m.field("tokenizer.ggml.seperator_token_id",
                      m.field("tokenizer.ggml.eos_token_id", 102))
        ids = [cls] + ids + [sep]
    else:
        raise SystemExit(f"tokenizer '{kind}' not supported; pass token ids instead")
    return ids


def _encode_spm(m, tokens, lookup, text):
    scores = m.field("tokenizer.ggml.scores") or [0.0] * len(tokens)
    ids = []
    for ch in ("\u2581" + text.replace(" ", "\u2581")):
        if ch in lookup:
            ids.append(lookup[ch])
        else:
            for b in ch.encode("utf-8"):
                ids.append(lookup[f"<0x{b:02X}>"])
    while True:
        best, best_i = None, -1
        for i in range(len(ids) - 1):
            j = lookup.get(tokens[ids[i]] + tokens[ids[i + 1]])
            if j is not None and (best is None or scores[j] > scores[best]):
                best, best_i = j, i
        if best is None:
            return ids
        ids[best_i:best_i + 2] = [best]


# GPT-2 pre-tokenizer.  The real pattern uses \p{L}/\p{N}; this ASCII-centric
# version is exact for English text.
_GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\u00c0-\uffff]+| ?[0-9]+| ?[^\sA-Za-z0-9\u00c0-\uffff]+|\s+(?!\S)|\s+""")


def _encode_gpt2(m, lookup, text):
    merges = m.field("tokenizer.ggml.merges") or []
    rank = {tuple(mg.split(" ", 1)): i for i, mg in enumerate(merges)}
    ids = []
    for word in _GPT2_PAT.findall(text):
        parts = [B2U[b] for b in word.encode("utf-8")]
        while len(parts) > 1:
            pairs = [(rank.get((parts[i], parts[i + 1]), 1 << 30), i) for i in range(len(parts) - 1)]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        ids += [lookup[p] for p in parts]
    return ids


def _encode_wordpiece(lookup, text, uncased=True):
    """WordPiece as stored by llama.cpp's converter: word-initial pieces carry
    a '\u2581' prefix, continuation pieces are bare."""
    if uncased:
        text = text.lower()
    words = re.findall(r"\w+|[^\w\s]", text)
    unk = lookup.get("[UNK]", 100)
    ids = []
    for w in words:
        start, pieces = 0, []
        while start < len(w):
            end = len(w)
            while end > start:
                sub = ("\u2581" if start == 0 else "") + w[start:end]
                if sub in lookup:
                    pieces.append(lookup[sub])
                    break
                end -= 1
            if end == start:
                pieces = [unk]
                break
            start = end
        ids += pieces
    return ids


# --- Image builder ---------------------------------------------------------------
class Asm:
    """Tiny assembler with forward label references."""

    def __init__(self):
        self.code = []
        self.labels = {}

    def label(self, name):
        self.labels[name] = len(self.code)

    def __call__(self, op, a=0, b=0, x=0, y=0, z=0):
        self.code.append([op, a, b, x, y, z])

    def jump(self, target):
        self(I.BEQ, I.RZERO, I.RZERO, target)

    def assemble(self, base):
        out = bytearray()
        for op, a, b, x, y, z in self.code:
            if isinstance(x, str):
                x = base + 16 * self.labels[x]
            out += I.encode(op, a, b, x, y, z)
        return bytes(out)


def rowbytes(fmt, n):
    block, tsize = GGML_QUANT_SIZES[GGMLQuantizationType(fmt)]
    return n // block * tsize


class Builder:
    def __init__(self, m, p, ctx, policy, fallback, mem_mb, kv="f16"):
        self.m, self.p, self.ctx = m, p, ctx
        self.kv16 = kv == "f16"
        self.kv_esz = 2 if self.kv16 else 4
        self.kv_fmt = I.KV_F16 if self.kv16 else I.KV_F32
        self.policy, self.fallback = policy, fallback
        self.mem_mb = mem_mb
        self.big = mem_mb > 8
        self.header = bytearray(I.HEADER_SIZE)
        if not self.big:
            self.base = I.HYPERRAM_BASE + I.HEADER_SIZE
            self.top = I.HYPERRAM_BASE + int(mem_mb * 0x100000)
        elif mem_mb == 64:
            self.base, self.top = I.SDRAM_BASE, I.MEM_TOP
        else:
            self.base = I.WINDOW_END
            self.top = min(I.MEM_TOP, I.HYPERRAM_BASE + int(mem_mb * 0x100000))
        if self.big and ctx + 1 > I.TOKENS_AREA_SIZE // 4:
            raise SystemExit(f"--ctx {ctx} too large: the host window holds "
                             f"{I.TOKENS_AREA_SIZE // 4 - 1} tokens")
        self.data = bytearray()          # payload, loaded at self.base
        self.lossy = []                  # weights stored with fewer bits than the source
        # 64 MB images: the CPU-side tables go in a separate host segment in
        # attic RAM, after the host window; the model goes to SDRAM.
        self.host = bytearray() if mem_mb == 64 else None
        self.host_base = I.WINDOW_END
        self.log = []
        self.code_slots = 64 + 64 * p["n_layers"]
        self.code_addr = self.add(bytes(16 * self.code_slots), align=16)
        self.vocab()     # CPU-side tables early, so they stay in attic RAM
        self.scratch = None
        self.A = Asm()

    @property
    def here(self):
        return self.base + len(self.data)

    def patch(self, addr, blob):
        o = addr - self.base
        self.data[o:o + len(blob)] = blob

    def tokens(self):
        """Token buffer: in the host window for big images, else last."""
        self.tokens_addr = I.TOKENS_AREA if self.big else self.scratch
        return self.tokens_addr

    def output_buffer(self, nbytes):
        if self.big:
            if nbytes > I.OUTPUT_AREA_SIZE:
                raise SystemExit("output vector too large for the host window")
            return I.OUTPUT_AREA
        return self.salloc(nbytes)

    def add(self, blob, align=32):
        while len(self.data) % align:
            self.data.append(0)
        addr = self.here
        self.data += blob
        return addr

    def wt(self, name):
        """Add a weight matrix; returns (fmt, address, bytes per row)."""
        m = self.m
        t = m.tensors[name]
        src = int(t.tensor_type)
        rows, cols = m.shape(name)
        big_table = name in ("token_embd.weight", "output.weight")
        if self.policy == "keep" and src in I.NATIVE_FORMATS:
            fmt, blob, rp = src, np.asarray(t.data).tobytes(), False
        else:
            if self.policy == "keep":
                fmt = self.fallback
            elif self.policy == "mixed":
                fmt = I.FMT_Q4_0 if big_table else I.FMT_Q8_0
            else:
                fmt = FMT_BY_NAME[self.policy]
            w = m.tensor_f32(name).reshape(rows, cols)
            if cols % GGML_QUANT_SIZES[GGMLQuantizationType(fmt)][0]:
                fmt = I.FMT_F16
            if fmt == I.FMT_F32:
                blob = w.astype("<f4").tobytes()
            elif fmt == I.FMT_F16:
                blob = w.astype("<f2").tobytes()
            else:
                blob = quantize(w, GGMLQuantizationType(fmt)).tobytes()
            rp = True
        addr = self.add(blob)
        self.log.append((name, GGMLQuantizationType(src).name,
                         GGMLQuantizationType(fmt).name, len(blob), rp))
        if src not in FLOAT_TYPES and bpw(fmt) < bpw(src):
            self.lossy.append((name, rows * cols, src, fmt))
        return fmt, addr, rowbytes(fmt, cols)

    def vec(self, *names):
        """Add one or more vectors back to back as f32 (e.g. gain then bias)."""
        return self.add(b"".join(self.m.tensor_f32(n).astype("<f4").tobytes() for n in names))

    def add_host(self, blob, align=32):
        """Add a table the CPU reads: it must land in attic RAM."""
        if self.host is None:
            addr = self.add(blob, align)
            if addr + len(blob) > I.SDRAM_BASE:
                raise SystemExit("internal error: host table beyond attic RAM")
            return addr
        while len(self.host) % align:
            self.host.append(0)
        addr = self.host_base + len(self.host)
        self.host += blob
        return addr

    def vocab(self):
        """Vocab table (raw UTF-8 bytes) and the tokenizer block."""
        m = self.m
        kind = tokenizer_kind(m)
        tokens = m.field("tokenizer.ggml.tokens")
        specials = special_flags(m)
        raw = [piece_bytes(kind, t, sp) for t, sp in zip(tokens, specials)]
        # Raw bytes (UTF-8).  Byte-level vocabularies split characters across
        # tokens, so folding to ASCII has to happen on the output stream, in
        # the front end, not per token.
        offs, blob = [], bytearray()
        base = 4 * (len(raw) + 1)
        for pc in raw:
            offs.append(base + len(blob))
            blob += pc
        offs.append(base + len(blob))
        self.vocab_addr = self.add_host(struct.pack(f"<{len(offs)}I", *offs) + bytes(blob))
        self.n_vocab = n = len(tokens)

        # Which tokens can appear when encoding typed (ASCII) text
        is_byte = [kind == "llama" and len(t) == 6 and t.startswith("<0x") and t.endswith(">")
                   for t in tokens]
        usable = [r != b"" and not sp and not ib and is_plain_ascii(r)
                  for r, sp, ib in zip(raw, specials, is_byte)]
        index = sorted((i for i in range(n) if usable[i]), key=lambda i: raw[i])
        # Duplicate texts (rare): keep the first, which sorts by id
        dedup, last = [], None
        for i in index:
            if raw[i] != last:
                dedup.append(i)
            last = raw[i]
        index = dedup

        lookup = {raw[i]: i for i in index}
        prio = [None] * n
        if kind == "gpt2":
            for r, mg in enumerate(m.field("tokenizer.ggml.merges") or []):
                a, b_ = mg.split(" ", 1)
                try:
                    t = lookup.get(bytes(U2B[c] for c in a + b_))
                except KeyError:
                    continue
                if t is not None and (prio[t] is None or r < prio[t]):
                    prio[t] = r
            tk, flags = I.TOK_BPE, 0
        elif kind == "llama":
            scores = m.field("tokenizer.ggml.scores") or [0.0] * n
            order = sorted((i for i in index if len(raw[i]) > 1), key=lambda i: (-scores[i], i))
            for r, i in enumerate(order):
                prio[i] = r
            tk, flags = I.TOK_SPM, 0
        elif kind == "bert":
            tk, flags = I.TOK_WORDPIECE, I.TF_LOWERCASE
        else:
            raise SystemExit(f"tokenizer '{kind}' not supported")
        big = n > 0xFFFE
        if big:
            flags |= I.TF_PRIO32
        none = 0xFFFFFFFF if big else 0xFFFF
        prio = [none if v is None else v for v in prio]

        byte_init = [I.NO_TOKEN] * 256
        for b_ in range(256):
            t = lookup.get(bytes([b_]))
            if t is None and kind == "llama":
                t = tokens.index(f"<0x{b_:02X}>") if f"<0x{b_:02X}>" in tokens else None
            if t is not None:
                byte_init[b_] = t
        if kind == "bert":
            begin = m.field("tokenizer.ggml.cls_token_id", m.field("tokenizer.ggml.bos_token_id", I.NO_TOKEN))
            end = m.field("tokenizer.ggml.seperator_token_id",
                          m.field("tokenizer.ggml.eos_token_id", I.NO_TOKEN))
        else:
            bid = m.field("tokenizer.ggml.bos_token_id", 1 if kind == "llama" else None)
            begin = bid if (wants_bos(m) and bid is not None) else I.NO_TOKEN
            end = I.NO_TOKEN
        unk = m.field("tokenizer.ggml.unknown_token_id",
                      tokens.index("[UNK]") if "[UNK]" in tokens else 0)

        a_index = self.add_host(struct.pack(f"<{len(index)}I", *index))
        a_prio = self.add_host(struct.pack(f"<{n}{'I' if big else 'H'}", *prio))
        a_bytes = self.add_host(struct.pack("<256I", *byte_init))
        self.tok_addr = self.add_host(struct.pack("<BBHIIIIIIII", tk, flags, 0, n, a_index,
                                                  len(index), a_prio, a_bytes,
                                                  begin & 0xFFFFFFFF, end & 0xFFFFFFFF, unk))
        self.tok_bytes = 4 * len(index) + (4 if big else 2) * n + 1024 + I.TOKBLOCK_SIZE

    def attn_block(self, k_cache, v_cache):
        p = self.p
        return self.add(struct.pack("<8I", k_cache, v_cache, p["n_heads"],
                                    p["n_kv_heads"], p["head_dim"], 0, 0, 0))

    def end_of_file(self):
        while len(self.data) % 256:
            self.data.append(0)
        self.file_end = self.here
        self.scratch = self.file_end

    def salloc(self, nbytes):
        a = self.scratch
        self.scratch += (nbytes + 255) // 256 * 256
        return a

    def report(self, mem_mb):
        m, p, top = self.m, self.p, self._top
        print(f"{m.arch}: dim {p['dim']}, {p['n_layers']} layers, {p['n_heads']} heads "
              f"({p['n_kv_heads']} kv), hidden {p['hidden']}, vocab {p['vocab']}, ctx {self.ctx}")
        rep = [e for e in self.log if e[4]]
        print(f"weights: {sum(e[3] for e in self.log) / 1e6:.2f} MB in {len(self.log)} "
              f"tensors, {len(rep)} repacked")
        shown = self.log[:2] + [e for e in rep[:6] if e not in self.log[:2]][:4]
        for name, src, dst, n, rp in shown:
            print(f"  {name:28s} {src:>6s} -> {dst:6s} {n:9d} bytes" + ("  (repacked)" if rp else ""))
        print(f"script: {self._code_len // 16} instructions at ${self.code_addr:07X}")
        print(f"payload: {len(self.data)} bytes at ${self.base:07X}-${self.file_end - 1:07X}; "
              f"scratch to ${self.scratch:07X}; tokens at ${self.tokens_addr:07X}")
        cfg = I.MEM_CONFIGS.get(mem_mb, "custom")
        print(f"memory used to ${top - 1:07X} with {self.ctx}-token context ({mem_mb:g} MB: {cfg})")
        print(f"tokenizer: {self.tok_bytes / 1024:.0f} KB of encode tables in attic RAM "
              f"(plus the vocab table)")

    def finish(self, output, outdim, flags, hyperram_mb, verbose):
        code = self.A.assemble(self.code_addr)
        if len(code) > 16 * self.code_slots:
            raise SystemExit("internal error: script larger than reserved space")
        self.patch(self.code_addr, code)
        m = self.m
        H = self.header
        H[0:4] = I.MAGIC
        H[4:8] = struct.pack("<HH", I.IMAGE_VERSION, flags)
        bos = m.field("tokenizer.ggml.bos_token_id", 1)
        eos = m.field("tokenizer.ggml.eos_token_id", 2)
        for f, v in ((I.H_CODE, self.code_addr), (I.H_DATA, self.code_addr + 16 * self.code_slots),
                     (I.H_SCRATCH, self.file_end), (I.H_TOKENS, self.tokens_addr),
                     (I.H_VOCAB, self.vocab_addr), (I.H_NVOCAB, self.n_vocab),
                     (I.H_MAXCTX, self.ctx), (I.H_BOS, bos), (I.H_EOS, eos),
                     (I.H_PAYLOAD_LEN, len(self.data)),
                     (I.H_OUTPUT, output), (I.H_OUTDIM, outdim)):
            H[f:f + 4] = struct.pack("<I", v & 0xFFFFFFFF)
        if self.big:
            top = self.scratch
            memcfg = bytes([8, 0x88, 64, 0])
            staging = (I.STAGING_AREA, I.STAGING_SIZE)
        else:
            top = self.tokens_addr + 4 * (self.ctx + 1)
            memcfg = bytes([8, 0, 0, 0])
            staging = (0, 0)
        total = top - I.HYPERRAM_BASE
        host_len = len(self.host) if self.host is not None else 0
        host_len += -host_len % 256
        if self.host is not None:
            self.host += bytes(host_len - len(self.host))
        for f, v in ((I.H_HOST_BASE, self.host_base if host_len else 0), (I.H_HOST_LEN, host_len),
                     (I.H_TOKENIZER, self.tok_addr),
                     (I.H_MEM_NEEDED, total), (I.H_ARCH, I.ARCH_CODES[m.arch]),
                     (I.H_LOAD_BASE, self.base), (I.H_STAGING, staging[0]),
                     (I.H_STAGING_SIZE, staging[1])):
            H[f:f + 4] = struct.pack("<I", v & 0xFFFFFFFF)
        H[I.H_MEMCFG:I.H_MEMCFG + 4] = memcfg
        self._code_len, self._top = len(code), top
        if top > self.top:
            raise ImageTooBig(f"Image needs {total / 1048576:.2f} MB, more than the "
                              f"{hyperram_mb:g} MB available")
        return bytes(self.header) + bytes(self.host or b"") + bytes(self.data)


# Register conventions used by the generated scripts
R_POS, R_TOK, R_NEXT, R_PLEN, R_NGEN, R_EOS, R_POS1, R_REASON, R_GEN, R_CTX, R_TMP, R_T, R_N = range(13)
RT = lambda f: I.HYPERRAM_BASE + f
a_ = [I.reg_addr(i) for i in range(16)]


def decoder_prologue(A, ctx, eos):
    A(I.LDR, R_POS, x=RT(I.R_POS))
    A(I.LDR, R_PLEN, x=RT(I.R_PROMPT_LEN))
    A(I.LDR, R_NGEN, x=RT(I.R_N_GENERATE))
    A(I.LI, R_GEN, z=0)
    A(I.STR, R_GEN, x=RT(I.R_GENERATED))
    A(I.LI, R_EOS, z=eos)
    A(I.LI, R_CTX, z=ctx)
    A(I.LI, R_REASON, z=I.STOP_COUNT)
    A(I.BEQ, R_NGEN, I.RZERO, "done")
    A.label("loop")
    A(I.BLT, R_POS, R_CTX, "room")
    A(I.LI, R_REASON, z=I.STOP_CTX_FULL)
    A.jump("done")
    A.label("room")


def decoder_epilogue(A, tokens_addr):
    """After ARGMAX into R_NEXT: write the token, count, stop checks, loop."""
    A(I.ADDI, R_POS1, R_POS, z=1)
    A(I.BLT, R_POS1, R_PLEN, "advance")     # still inside the prompt
    A(I.LEA, 0, R_POS1, tokens_addr, z=4)
    A(I.STR, R_NEXT, x=a_[0])
    A(I.ADDI, R_GEN, R_GEN, z=1)
    A(I.STR, R_GEN, x=RT(I.R_GENERATED))
    A(I.SYNC)                                # token + count visible to the CPU
    A(I.LI, R_REASON, z=I.STOP_EOS)
    A(I.BEQ, R_NEXT, R_EOS, "stop")
    A(I.LI, R_REASON, z=I.STOP_COUNT)
    A(I.BEQ, R_GEN, R_NGEN, "stop")
    A(I.LDR, R_TMP, x=RT(I.R_STOP_REQUEST))
    A(I.LI, R_REASON, z=I.STOP_REQUESTED)
    A(I.BNE, R_TMP, I.RZERO, "stop")
    A.label("advance")
    A(I.ADDI, R_POS, R_POS, z=1)
    A(I.STR, R_POS, x=RT(I.R_POS))
    A.jump("loop")
    A.label("stop")
    A(I.ADDI, R_POS, R_POS, z=1)             # the new token is next to process
    A.label("done")
    A(I.STR, R_POS, x=RT(I.R_POS))
    A(I.STR, R_REASON, x=RT(I.R_STOP_REASON))
    A(I.SYNC)
    A(I.HALT)


# --- llama ------------------------------------------------------------------------
def build_llama(b):
    m, p, A, ctx = b.m, b.p, b.A, b.ctx
    dim, hd, kvd, hidden, vocab = p["dim"], p["head_dim"], p["kv_dim"], p["hidden"], p["vocab"]
    embd = b.wt("token_embd.weight")
    L = []
    for l in range(p["n_layers"]):
        g = lambda n: f"blk.{l}.{n}.weight"
        L.append(dict(attn_norm=b.vec(g("attn_norm")), ffn_norm=b.vec(g("ffn_norm")),
                      wq=b.wt(g("attn_q")), wk=b.wt(g("attn_k")), wv=b.wt(g("attn_v")),
                      wo=b.wt(g("attn_output")), w1=b.wt(g("ffn_gate")),
                      w3=b.wt(g("ffn_up")), w2=b.wt(g("ffn_down"))))
    out_norm = b.vec("output_norm.weight")
    cls = b.wt("output.weight") if m.has("output.weight") else embd
    j = np.arange(hd // 2, dtype=np.float64)
    ang = np.arange(ctx, dtype=np.float64)[:, None] * (1.0 / p["rope_base"] ** (2 * j / hd))[None, :]
    rope = b.add(np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype("<f4").tobytes())
    blocks = [b.add(bytes(32)) for _ in L]
    b.end_of_file()
    x, xb, q, att = (b.salloc(4 * dim) for _ in range(4))
    kscr = b.salloc(4 * kvd)
    hb, hb2 = b.salloc(4 * hidden), b.salloc(4 * hidden)
    logits = b.salloc(4 * vocab)
    esz = b.kv_esz
    kc = [b.salloc(esz * ctx * kvd) for _ in L]
    vc = [b.salloc(esz * ctx * kvd) for _ in L]
    for l, blk in enumerate(blocks):
        b.patch(blk, struct.pack("<8I", kc[l], vc[l], p["n_heads"], p["n_kv_heads"], hd,
                                 b.kv_fmt, 0, 0))
    tokens_addr = b.tokens()
    eps = I.f32_bits(p["eps"])

    decoder_prologue(A, ctx, m.field("tokenizer.ggml.eos_token_id", 2))
    A(I.LEA, 0, R_POS, tokens_addr, z=4)
    A(I.LDR, R_TOK, x=a_[0])
    A(I.SETN, x=dim)
    A(I.LEA, 1, R_TOK, embd[1], z=embd[2])
    A(I.DEQROW, embd[0], x=a_[1], z=x)
    A(I.LEA, 2, R_POS, rope, z=4 * hd)
    for l, W in enumerate(L):
        A(I.SETN, x=dim, z=eps)
        A(I.RMSNORM, x=x, y=W["attn_norm"], z=xb)
        A(I.SETN, x=dim, y=dim)
        A(I.GEMV, W["wq"][0], x=W["wq"][1], y=xb, z=q)
        A(I.SETN, x=dim, y=kvd)
        A(I.LEA, 3, R_POS, kc[l], z=esz * kvd)
        A(I.LEA, 4, R_POS, vc[l], z=esz * kvd)
        if b.kv16:
            # K is rotated in F32, then stored as F16; V goes straight in
            A(I.GEMV, W["wk"][0], x=W["wk"][1], y=xb, z=kscr)
            A(I.GEMV, W["wv"][0], I.GEMV_F16OUT, x=W["wv"][1], y=xb, z=a_[4])
        else:
            A(I.GEMV, W["wk"][0], x=W["wk"][1], y=xb, z=a_[3])
            A(I.GEMV, W["wv"][0], x=W["wv"][1], y=xb, z=a_[4])
        A(I.SETN, x=dim, y=hd)
        A(I.ROPE, x=q, y=a_[2])
        A(I.SETN, x=kvd, y=hd)
        if b.kv16:
            A(I.ROPE, x=kscr, y=a_[2])
            A(I.CVT16, x=kscr, z=a_[3])
        else:
            A(I.ROPE, x=a_[3], y=a_[2])
        A(I.ATTN, R_POS, x=blocks[l], y=q, z=att)
        A(I.SETN, x=dim, y=dim)
        A(I.GEMV, W["wo"][0], I.GEMV_ACC, x=W["wo"][1], y=att, z=x)
        A(I.SETN, x=dim, z=eps)
        A(I.RMSNORM, x=x, y=W["ffn_norm"], z=xb)
        A(I.SETN, x=dim, y=hidden)
        A(I.GEMV, W["w1"][0], x=W["w1"][1], y=xb, z=hb)
        A(I.GEMV, W["w3"][0], x=W["w3"][1], y=xb, z=hb2)
        A(I.SETN, x=hidden)
        A(I.SILUMUL, x=hb, y=hb2, z=hb)
        A(I.SETN, x=hidden, y=dim)
        A(I.GEMV, W["w2"][0], I.GEMV_ACC, x=W["w2"][1], y=hb, z=x)
    A(I.SETN, x=dim, z=eps)
    A(I.RMSNORM, x=x, y=out_norm, z=xb)
    A(I.SETN, x=dim, y=vocab)
    A(I.GEMV, cls[0], x=cls[1], y=xb, z=logits)
    A(I.SETN, x=vocab)
    A(I.ARGMAX, R_NEXT, x=logits)
    decoder_epilogue(A, tokens_addr)
    return logits, vocab, 0


# --- gpt2 -------------------------------------------------------------------------
def build_gpt2(b):
    m, p, A, ctx = b.m, b.p, b.A, b.ctx
    dim, hidden, vocab = p["dim"], p["hidden"], p["vocab"]
    wte = b.wt("token_embd.weight")
    wpe = b.wt("position_embd.weight")
    L = []
    for l in range(p["n_layers"]):
        g = lambda n, s="weight": f"blk.{l}.{n}.{s}"
        L.append(dict(
            ln1=b.vec(g("attn_norm"), g("attn_norm", "bias")),
            ln2=b.vec(g("ffn_norm"), g("ffn_norm", "bias")),
            wqkv=b.wt(g("attn_qkv")), bqkv=b.vec(g("attn_qkv", "bias")),
            wo=b.wt(g("attn_output")), bo=b.vec(g("attn_output", "bias")),
            wup=b.wt(g("ffn_up")), bup=b.vec(g("ffn_up", "bias")),
            wdn=b.wt(g("ffn_down")), bdn=b.vec(g("ffn_down", "bias"))))
    lnf = b.vec("output_norm.weight", "output_norm.bias")
    cls = b.wt("output.weight") if m.has("output.weight") else wte
    blocks = [b.add(bytes(32)) for _ in L]
    b.end_of_file()
    x, xb, q, att, tmp, kscr, vscr = (b.salloc(4 * dim) for _ in range(7))
    hb = b.salloc(4 * hidden)
    logits = b.salloc(4 * vocab)
    esz = b.kv_esz
    kc = [b.salloc(esz * ctx * dim) for _ in L]
    vc = [b.salloc(esz * ctx * dim) for _ in L]
    for l, blk in enumerate(blocks):
        b.patch(blk, struct.pack("<8I", kc[l], vc[l], p["n_heads"], p["n_heads"],
                                 p["head_dim"], b.kv_fmt, 0, 0))
    tokens_addr = b.tokens()
    eps = I.f32_bits(p["eps"])
    eos = m.field("tokenizer.ggml.eos_token_id", 50256)

    decoder_prologue(A, ctx, eos)
    A(I.LEA, 0, R_POS, tokens_addr, z=4)
    A(I.LDR, R_TOK, x=a_[0])
    A(I.SETN, x=dim)
    A(I.LEA, 1, R_TOK, wte[1], z=wte[2])
    A(I.DEQROW, wte[0], x=a_[1], z=x)
    A(I.LEA, 1, R_POS, wpe[1], z=wpe[2])
    A(I.DEQROW, wpe[0], x=a_[1], z=tmp)
    A(I.VADD, x=x, y=tmp, z=x)
    for l, W in enumerate(L):
        wf, wa, wrb = W["wqkv"]
        A(I.SETN, x=dim, z=eps)
        A(I.LAYERNORM, x=x, y=W["ln1"], z=xb)
        A(I.SETN, x=dim, y=dim)
        A(I.GEMV, wf, x=wa, y=xb, z=q)                          # rows 0..dim-1
        A(I.LEA, 3, R_POS, kc[l], z=esz * dim)
        A(I.LEA, 4, R_POS, vc[l], z=esz * dim)
        kd, vd = (kscr, vscr) if b.kv16 else (a_[3], a_[4])
        A(I.GEMV, wf, x=wa + dim * wrb, y=xb, z=kd)             # rows dim..2dim-1
        A(I.GEMV, wf, x=wa + 2 * dim * wrb, y=xb, z=vd)         # rows 2dim..3dim-1
        A(I.SETN, x=dim)
        A(I.VADD, x=q, y=W["bqkv"], z=q)
        A(I.VADD, x=kd, y=W["bqkv"] + 4 * dim, z=kd)
        A(I.VADD, x=vd, y=W["bqkv"] + 8 * dim, z=vd)
        if b.kv16:
            A(I.CVT16, x=kscr, z=a_[3])
            A(I.CVT16, x=vscr, z=a_[4])
        A(I.ATTN, R_POS, x=blocks[l], y=q, z=att)
        A(I.SETN, x=dim, y=dim)
        A(I.GEMV, W["wo"][0], I.GEMV_ACC, x=W["wo"][1], y=att, z=x)
        A(I.SETN, x=dim, z=eps)
        A(I.VADD, x=x, y=W["bo"], z=x)
        A(I.LAYERNORM, x=x, y=W["ln2"], z=xb)
        A(I.SETN, x=dim, y=hidden)
        A(I.GEMV, W["wup"][0], x=W["wup"][1], y=xb, z=hb)
        A(I.SETN, x=hidden)
        A(I.VADD, x=hb, y=W["bup"], z=hb)
        A(I.GELU, x=hb, z=hb)
        A(I.SETN, x=hidden, y=dim)
        A(I.GEMV, W["wdn"][0], I.GEMV_ACC, x=W["wdn"][1], y=hb, z=x)
        A(I.SETN, x=dim)
        A(I.VADD, x=x, y=W["bdn"], z=x)
    A(I.SETN, x=dim, z=eps)
    A(I.LAYERNORM, x=x, y=lnf, z=xb)
    A(I.SETN, x=dim, y=vocab)
    A(I.GEMV, cls[0], x=cls[1], y=xb, z=logits)
    A(I.SETN, x=vocab)
    A(I.ARGMAX, R_NEXT, x=logits)
    decoder_epilogue(A, tokens_addr)
    return logits, vocab, 0


# --- bert -------------------------------------------------------------------------
def build_bert(b):
    """Encoder: processes tokens[0..prompt_len) as one sequence, layer by layer,
    and leaves the pooled embedding at H_OUTPUT."""
    m, p, A, ctx = b.m, b.p, b.A, b.ctx
    dim, hidden = p["dim"], p["hidden"]
    tok = b.wt("token_embd.weight")
    pos = b.wt("position_embd.weight")
    typ = b.wt("token_types.weight") if m.has("token_types.weight") else None
    embn = b.vec("token_embd_norm.weight", "token_embd_norm.bias")
    L = []
    for l in range(p["n_layers"]):
        g = lambda n, s="weight": f"blk.{l}.{n}.{s}"
        L.append(dict(
            wq=b.wt(g("attn_q")), bq=b.vec(g("attn_q", "bias")),
            wk=b.wt(g("attn_k")), bk=b.vec(g("attn_k", "bias")),
            wv=b.wt(g("attn_v")), bv=b.vec(g("attn_v", "bias")),
            wo=b.wt(g("attn_output")), bo=b.vec(g("attn_output", "bias")),
            ln1=b.vec(g("attn_output_norm"), g("attn_output_norm", "bias")),
            wup=b.wt(g("ffn_up")), bup=b.vec(g("ffn_up", "bias")),
            wdn=b.wt(g("ffn_down")), bdn=b.vec(g("ffn_down", "bias")),
            ln2=b.vec(g("layer_output_norm"), g("layer_output_norm", "bias"))))
    blocks = [b.add(bytes(32)) for _ in L]
    b.end_of_file()
    X = b.salloc(4 * ctx * dim)          # activations, one row per token
    Q = b.salloc(4 * ctx * dim)
    esz = b.kv_esz
    K = b.salloc(esz * ctx * dim)
    V = b.salloc(esz * ctx * dim)
    att, tmp = b.salloc(4 * dim), b.salloc(4 * dim)
    kscr, vscr = b.salloc(4 * dim), b.salloc(4 * dim)
    out = b.output_buffer(4 * dim)
    hb = b.salloc(4 * hidden)
    for blk in blocks:
        b.patch(blk, struct.pack("<8I", K, V, p["n_heads"], p["n_heads"], p["head_dim"],
                                 b.kv_fmt, 0, 0))
    tokens_addr = b.tokens()
    eps = I.f32_bits(p["eps"])
    row = 4 * dim
    R_NM1 = R_POS1

    A(I.LI, R_GEN, z=0)
    A(I.STR, R_GEN, x=RT(I.R_GENERATED))
    A(I.LDR, R_N, x=RT(I.R_PROMPT_LEN))
    A(I.LI, R_CTX, z=ctx + 1)
    A(I.LI, R_REASON, z=I.STOP_CTX_FULL)
    A(I.BLT, R_N, R_CTX, "fits")
    A.jump("done")
    A.label("fits")
    A(I.LI, R_REASON, z=I.STOP_COUNT)
    A(I.BEQ, R_N, I.RZERO, "done")
    A(I.ADDI, R_NM1, R_N, z=0xFFFFFFFF)          # n - 1: last position attended
    # Embeddings: x[t] = LN(tok[t] + pos[t] + type[0])
    A(I.LI, R_T, z=0)
    A.label("emb")
    A(I.LEA, 0, R_T, tokens_addr, z=4)
    A(I.LDR, R_TOK, x=a_[0])
    A(I.LEA, 6, R_T, X, z=row)
    A(I.SETN, x=dim, z=eps)
    A(I.LEA, 1, R_TOK, tok[1], z=tok[2])
    A(I.DEQROW, tok[0], x=a_[1], z=a_[6])
    A(I.LEA, 1, R_T, pos[1], z=pos[2])
    A(I.DEQROW, pos[0], x=a_[1], z=tmp)
    A(I.VADD, x=a_[6], y=tmp, z=a_[6])
    if typ:
        A(I.DEQROW, typ[0], x=typ[1], z=tmp)
        A(I.VADD, x=a_[6], y=tmp, z=a_[6])
    A(I.LAYERNORM, x=a_[6], y=embn, z=a_[6])
    A(I.ADDI, R_T, R_T, z=1)
    A(I.BLT, R_T, R_N, "emb")
    for l, W in enumerate(L):
        # Pass 1: q, k, v for every token
        A(I.LI, R_T, z=0)
        A.label(f"qkv{l}")
        A(I.LEA, 6, R_T, X, z=row)
        A(I.LEA, 7, R_T, Q, z=row)
        A(I.LEA, 3, R_T, K, z=esz * dim)
        A(I.LEA, 4, R_T, V, z=esz * dim)
        kd, vd = (kscr, vscr) if b.kv16 else (a_[3], a_[4])
        A(I.SETN, x=dim, y=dim)
        A(I.GEMV, W["wq"][0], x=W["wq"][1], y=a_[6], z=a_[7])
        A(I.GEMV, W["wk"][0], x=W["wk"][1], y=a_[6], z=kd)
        A(I.GEMV, W["wv"][0], x=W["wv"][1], y=a_[6], z=vd)
        A(I.VADD, x=a_[7], y=W["bq"], z=a_[7])
        A(I.VADD, x=kd, y=W["bk"], z=kd)
        A(I.VADD, x=vd, y=W["bv"], z=vd)
        if b.kv16:
            A(I.CVT16, x=kscr, z=a_[3])
            A(I.CVT16, x=vscr, z=a_[4])
        A(I.ADDI, R_T, R_T, z=1)
        A(I.BLT, R_T, R_N, f"qkv{l}")
        # Pass 2: bidirectional attention, then the rest of the layer
        A(I.LI, R_T, z=0)
        A.label(f"att{l}")
        A(I.LEA, 6, R_T, X, z=row)
        A(I.LEA, 7, R_T, Q, z=row)
        A(I.ATTN, R_NM1, x=blocks[l], y=a_[7], z=att)
        A(I.SETN, x=dim, y=dim, z=eps)
        A(I.GEMV, W["wo"][0], I.GEMV_ACC, x=W["wo"][1], y=att, z=a_[6])
        A(I.VADD, x=a_[6], y=W["bo"], z=a_[6])
        A(I.LAYERNORM, x=a_[6], y=W["ln1"], z=a_[6])
        A(I.SETN, x=dim, y=hidden)
        A(I.GEMV, W["wup"][0], x=W["wup"][1], y=a_[6], z=hb)
        A(I.SETN, x=hidden)
        A(I.VADD, x=hb, y=W["bup"], z=hb)
        A(I.GELU, x=hb, z=hb)
        A(I.SETN, x=hidden, y=dim)
        A(I.GEMV, W["wdn"][0], I.GEMV_ACC, x=W["wdn"][1], y=hb, z=a_[6])
        A(I.SETN, x=dim, z=eps)
        A(I.VADD, x=a_[6], y=W["bdn"], z=a_[6])
        A(I.LAYERNORM, x=a_[6], y=W["ln2"], z=a_[6])
        A(I.ADDI, R_T, R_T, z=1)
        A(I.BLT, R_T, R_N, f"att{l}")
    # Pooling
    A(I.SETN, x=dim)
    if p.get("pooling", 1) == 2:                 # CLS
        A(I.COPY, x=X, y=out, z=4 * dim)
    else:                                        # mean
        A(I.MEANROWS, R_N, x=X, z=out)
    A(I.LI, R_REASON, z=I.STOP_COUNT)
    A.label("done")
    A(I.STR, R_REASON, x=RT(I.R_STOP_REASON))
    A(I.SYNC)
    A(I.HALT)
    return out, dim, I.H_FLAG_ENCODER


BUILDERS = dict(llama=build_llama, gpt2=build_gpt2, bert=build_bert)


def convert(m, ctx, policy, fallback, hyperram_mb, verbose=True, confirm=None, kv="f16"):
    """confirm(message) -> bool is asked before 'auto' stores weights with
    fewer bits than an already-quantised source; None = never ask."""
    if policy == "auto":
        for pol, what in AUTO_ORDER:
            try:
                b = build(m, ctx, pol, fallback, hyperram_mb, kv)
            except ImageTooBig:
                continue
            if b.lossy and confirm is not None:
                n = sum(e[1] for e in b.lossy)
                tot = matrix_params(m)
                srcs = sorted({GGMLQuantizationType(e[2]).name for e in b.lossy})
                dsts = sorted({GGMLQuantizationType(e[3]).name for e in b.lossy})
                msg = (f"To fit in {hyperram_mb:g} MB, 'auto' chose {what}, which stores "
                       f"{100 * n / tot:.0f}% of the weights with fewer bits than the file "
                       f"({'/'.join(srcs)} -> {'/'.join(dsts)}).\n"
                       f"Quality will drop.  Better options: a larger memory size, or a GGUF "
                       f"of this model already in {'/'.join(dsts)} (or F16/F32).\n"
                       f"To accept this without being asked, give the format explicitly: "
                       f"WTYPE={pol} (python: --wtype {pol}).")
                if not confirm(msg):
                    raise SystemExit("Not converted.")
            if verbose:
                print(f"auto: using --wtype {pol} ({what})")
                b.report(hyperram_mb)
            return b.image
        raise ImageTooBig(f"Does not fit in {hyperram_mb:g} MB even with Q4_0 weights")
    b = build(m, ctx, policy, fallback, hyperram_mb, kv)
    if verbose:
        b.report(hyperram_mb)
        if b.lossy:
            print(f"note: {len(b.lossy)} weight tensors stored with fewer bits than the source")
    return b.image


def build(m, ctx, policy, fallback, hyperram_mb, kv="f16"):
    p = params(m)
    if ctx is None:
        ctx = min(p["ctx_train"], 256 if m.arch != "bert" else 128)
    b = Builder(m, p, ctx, policy, fallback, hyperram_mb, kv)
    output, outdim, flags = BUILDERS[m.arch](b)
    b.image = b.finish(output, outdim, flags, hyperram_mb, verbose=False)
    return b


def ask(msg):
    import sys
    print("\nWARNING: " + msg, file=sys.stderr)
    if not sys.stdin.isatty():
        print("(not interactive, so not converting)", file=sys.stderr)
        return False
    try:
        return input("Convert anyway? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="build an SSNAIL image")
    c.add_argument("gguf")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--ctx", type=int,
                   help="context / maximum sequence length (default min(model, 256); 128 for bert)")
    c.add_argument("--wtype", default="keep", choices=["keep", "auto", "mixed"] + list(FMT_BY_NAME),
                   help="weight format: keep native formats; force one; 'mixed' (Q8_0 layers, "
                        "Q4_0 embedding/classifier); or 'auto' (best that fits: keep, q8_0, "
                        "mixed, q4_0; asks before going below the file's own precision)")
    c.add_argument("--kv", default="f16", choices=["f16", "f32"],
                   help="KV cache format (default f16; f32 for exact reference runs)")
    c.add_argument("--fallback", default="q8_0", choices=list(FMT_BY_NAME),
                   help="format for weights not natively supported (with --wtype keep)")
    c.add_argument("--mem-mb", "--hyperram-mb", dest="hyperram_mb", type=float, default=8,
                   help="linear memory from $8000000 available to the image: 8 (attic RAM), "
                        "64 (SDRAM) or 72 (both, R6); default 8")
    t = sub.add_parser("tokenize", help="print token ids for a prompt")
    t.add_argument("gguf")
    t.add_argument("text")
    t.add_argument("--no-bos", action="store_true")
    i = sub.add_parser("info", help="show model metadata and tensor formats")
    i.add_argument("gguf")
    args = ap.parse_args()

    m = Model(args.gguf)
    if args.cmd == "convert":
        try:
            image = convert(m, args.ctx, args.wtype, FMT_BY_NAME[args.fallback], args.hyperram_mb,
                            confirm=ask, kv=args.kv)
        except ImageTooBig:
            raise SystemExit(too_big_message(m, args.gguf, args.hyperram_mb, args.ctx is not None))
        open(args.output, "wb").write(image)
        print(f"wrote {args.output}: load at $8000000")
    elif args.cmd == "tokenize":
        print(",".join(str(t) for t in encode(m, args.text, bos=not args.no_bos)))
    else:
        print(f"architecture {m.arch}, tokenizer {tokenizer_kind(m)}")
        print(params(m))
        for t in m.r.tensors:
            print(f"  {t.name:32s} {t.tensor_type.name:6s} {list(t.shape)}")


if __name__ == "__main__":
    main()
