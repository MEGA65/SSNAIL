#!/usr/bin/env python3
"""Interactive front end for the SSNAIL emulator.

    ssnail_chat.py model.ssnail

Self-contained: the image carries its own tokenizer, so nothing else is
needed.  Defaults:
  * sampling at temperature 0.8, top-k 40, with a mild repetition penalty,
    which reads far better than greedy decoding on small models.
    --greedy gives exactly what the v1 hardware does (it is greedy).

Decoder models (llama, gpt2): type text and the model continues it, with the
context kept between turns.  An empty line asks for more of the same.
Encoder models (bert): each line is embedded, and the most similar earlier
lines are shown.

It drives the image the way MEGA65 software would: it writes tokens and the
runtime block, starts the script, and prints tokens as the script publishes
them (at each SYNC).

Commands:  /reset  /n N  /temp T  /topk K  /penalty P  /greedy  /hw  /eos  /seed S  /stats  /quit
"""

import argparse
import struct
import sys

import numpy as np

import ssnail_isa as I
import ssnail_sim as S

ARCH_NAMES = {v: k for k, v in I.ARCH_CODES.items()}


class Tokenizer:
    """Tokenizes with the tables inside the image (ssnail_tok.py), exactly as
    the MEGA65 will.  Decodes with the image's ASCII-folded vocab table."""

    def __init__(self, m):
        from ssnail_tok import ImageTokenizer, AsciiFolder
        self.m = m
        self.arch = ARCH_NAMES.get(S.header(m, I.H_ARCH), "llama")
        self.it = ImageTokenizer(m.mem)
        # Token text is raw UTF-8; fold the output stream to ASCII for
        # display, across token boundaries (as the MEGA65 front end will)
        self.folder = AsciiFolder()

    def encode(self, text, first):
        return self.it.encode(text, first=first)

    def decode(self, t):
        return S.vocab_piece(self.m, t)

    def show(self, t):
        return self.folder.feed(self.decode(t))


def write_tokens(m, start, ids):
    base = S.header(m, I.H_TOKENS) - I.HYPERRAM_BASE
    for i, t in enumerate(ids):
        o = base + 4 * (start + i)
        m.mem[o:o + 4] = struct.pack("<I", t)


def read_token(m, i):
    o = S.header(m, I.H_TOKENS) - I.HYPERRAM_BASE + 4 * i
    return struct.unpack("<I", m.mem[o:o + 4])[0]


def eos_probability(m, logits):
    """Probability the sampler gives the end-of-text token (after temperature
    and top-k, as Machine.choose does; repetition penalty ignored)."""
    eos = S.header(m, I.H_EOS)
    if eos >= len(logits):
        return 0.0
    z = np.asarray(logits, dtype=np.float64)
    if m.temperature <= 0:
        return 1.0 if int(np.argmax(z)) == eos else 0.0
    z = z / m.temperature
    if 0 < m.top_k < len(z):
        cut = np.partition(z, -m.top_k)[-m.top_k]
        if z[eos] < cut:
            return 0.0
        z = np.where(z >= cut, z, -np.inf)
    p = np.exp(z - z.max())
    return float(p[eos] / p.sum())


class Session:
    def __init__(self, m, tok, n):
        self.m, self.tok, self.n = m, tok, n
        self.show_eos = False
        self.started = False
        self.shown = 0

    def reset(self):
        self.started = False

    def turn(self, text):
        m = self.m
        if not self.started:
            ids = self.tok.encode(text, first=True)
            if not ids:
                return
            write_tokens(m, 0, ids)
            S.set_rt(m, I.R_POS, 0)
            S.set_rt(m, I.R_PROMPT_LEN, len(ids))
            self.started = True
        else:
            pos = S.header(m, I.R_POS)
            ids = self.tok.encode(text, first=False) if text else []
            write_tokens(m, pos + 1, ids)
            S.set_rt(m, I.R_PROMPT_LEN, pos + 1 + len(ids))
        plen = S.header(m, I.R_PROMPT_LEN)
        S.set_rt(m, I.R_N_GENERATE, self.n)
        S.set_rt(m, I.R_STOP_REQUEST, 0)
        self.shown = 0

        def on_sync():
            # What the MEGA65 CPU does: poll the count, print new tokens
            g = S.header(m, I.R_GENERATED)
            while self.shown < g:
                t = read_token(m, plen + self.shown)
                sys.stdout.write(self.tok.show(t))
                sys.stdout.flush()
                self.shown += 1

        def on_argmax(logits):
            # Only while generating, not during the prompt
            if self.show_eos and S.header(m, I.R_POS) + 1 >= plen:
                p = eos_probability(m, logits)
                if p >= 0.05:
                    sys.stdout.write(f"<EOS {100 * p:.0f}%>")
                    sys.stdout.flush()

        m.on_sync = on_sync
        m.on_argmax = on_argmax
        m.run(S.header(m, I.H_CODE))
        reason = S.header(m, I.R_STOP_REASON)
        print()
        if reason == I.STOP_EOS:
            print("[end of text]")
        elif reason == I.STOP_CTX_FULL:
            print(f"[context full at {S.header(m, I.H_MAXCTX)} tokens: /reset to start again]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-n", type=int, default=120, help="tokens per turn (default 120)")
    ap.add_argument("--hw", action="store_true",
                    help="hardware numerics (Q8_0 GEMV inputs, function tables, ...) "
                         "instead of the float reference")
    ap.add_argument("--greedy", action="store_true",
                    help="greedy decoding, exactly as the v1 hardware")
    ap.add_argument("--penalty", type=float, default=1.15,
                    help="repetition penalty over the last 64 tokens (default 1.15; 1 = off)")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="sampling temperature (0 = greedy, as the v1 hardware)")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--eos", action="store_true",
                    help="show the end-of-text probability (<EOS n%%>) when it is 5%% or more")
    ap.add_argument("--prompt", help="run one prompt non-interactively and exit")
    args = ap.parse_args()

    m = S.Machine(open(args.image, "rb").read())
    m.temperature = 0.0 if args.greedy else args.temp
    m.top_k = args.topk
    m.hw = args.hw
    m.rep_penalty = 1.0 if args.greedy else args.penalty
    m.rng = np.random.default_rng(args.seed)
    tok = Tokenizer(m)
    encoder = bool(struct.unpack("<H", m.mem[6:8])[0] & I.H_FLAG_ENCODER)
    if S.description(m):
        print(f"[{S.description(m)}]")
    print(f"[{tok.arch} {'encoder' if encoder else 'decoder'}, context "
          f"{S.header(m, I.H_MAXCTX)}, vocab {S.header(m, I.H_NVOCAB)}]")
    if not encoder:
        print("[" + ("hardware numerics" if m.hw else "float reference numerics") + "; "
              + ("greedy, as the v1 hardware" if m.temperature <= 0 else
                     f"sampling: temperature {m.temperature}, top-k {m.top_k}, "
                     f"repetition penalty {m.rep_penalty}") + "]")
    sess = Session(m, tok, args.n)
    sess.show_eos = args.eos
    seen = []

    def handle(line):
        if encoder:
            if not line:
                return
            ids = tok.encode(line, first=True)
            vec, reason = S.encode_sequence(m, ids)
            if reason != I.STOP_COUNT:
                print(f"[too long: {len(ids)} tokens, maximum {S.header(m, I.H_MAXCTX)}]")
                return
            v = vec / (np.linalg.norm(vec) or 1)
            ranked = sorted(((float(v @ w), t) for t, w in seen), reverse=True)[:3]
            for score, text in ranked:
                print(f"  {score:+.3f}  {text}")
            if not ranked:
                print(f"  [{len(vec)}-dimensional embedding stored]")
            seen.append((line, v))
        else:
            sess.turn(line)

    if args.prompt is not None:
        handle(args.prompt)
        return
    print("Type text; /quit to exit, /reset for a fresh context.  "
          "An empty line continues the story.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        cmd = line.split()
        if cmd and cmd[0].startswith("/"):
            c, arg = cmd[0], (cmd[1] if len(cmd) > 1 else None)
            if c == "/quit":
                return
            elif c == "/reset":
                sess.reset(); seen.clear(); print("[context cleared]")
            elif c == "/n" and arg:
                sess.n = int(arg)
            elif c == "/temp" and arg:
                m.temperature = float(arg)
            elif c == "/topk" and arg:
                m.top_k = int(arg)
            elif c == "/eos":
                sess.show_eos = not sess.show_eos
                print(f"[end-of-text probability display {'on' if sess.show_eos else 'off'}: "
                      f"shown as <EOS n%> when the sampler gives it 5% or more]")
            elif c == "/hw":
                m.hw = not m.hw
                print(f"[{'hardware' if m.hw else 'float reference'} numerics]")
            elif c == "/penalty" and arg:
                m.rep_penalty = float(arg)
            elif c == "/greedy":
                m.temperature, m.rep_penalty = 0.0, 1.0
                print("[greedy, as the v1 hardware; /temp T to sample again]")
            elif c == "/seed" and arg:
                m.rng = np.random.default_rng(int(arg))
            elif c == "/stats":
                st = m.stats
                print(f"[{st['instructions']} instructions, "
                      f"{(st['bytes_read'] + st['bytes_written']) / 1e6:.1f} MB RAM traffic, "
                      f"pos {S.header(m, I.R_POS)}]")
            else:
                print(__doc__.split("Commands:")[1].strip())
            continue
        handle(line)


if __name__ == "__main__":
    main()
