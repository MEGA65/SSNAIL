"""Tokenizer that uses only the tables inside an SSNAIL image.

This is the reference for the MEGA65-native tokenizer: it reads memory the
way the 45GS02 will (u32 fields, a binary search over a sorted index, short
string compares against the vocab table), and uses no Python dictionaries
or GGUF data.  Input is plain ASCII, as typed on a MEGA65.

Algorithms (the tokenizer block format is documented in ssnail_isa.py):

  SPM (Llama) and BPE (GPT-2):
    1. Pre-split the text into chunks (BPE only: GPT-2's rules, ASCII form).
       SPM uses the whole text, with a leading space added.
    2. Start each chunk as one token per byte (the byte table).
    3. Repeatedly: for each adjacent pair, look up the concatenated text in
       the sorted index; among pairs that form a token, merge the one with
       the lowest priority (leftmost on ties).  Stop when none do.
  WordPiece (BERT):
    Lowercase; split into words (letters/digits) and single punctuation
    marks; for each word, greedily take the longest piece that is in the
    index (" " + text for the first piece of a word, bare text after it);
    a word with no match becomes [UNK].  Add [CLS] ... [SEP].
"""

import struct

import ssnail_isa as I


class ImageTokenizer:
    def __init__(self, mem, base=I.HYPERRAM_BASE):
        self.mem, self.base = mem, base
        tb = self.u32(base + I.H_TOKENIZER)
        hdr = self.read(tb, I.TOKBLOCK_SIZE)
        (self.kind, self.flags, _, self.n_vocab, self.a_index, self.n_index, self.a_prio,
         self.a_bytes, self.begin, self.end, self.unk) = struct.unpack("<BBHIIIIIIII", hdr)
        self.vocab = self.u32(base + I.H_VOCAB)
        self.lookups = 0          # count of index probes, for cost estimates

    # --- memory access, as the 45GS02 would do it ---------------------------
    def read(self, addr, n):
        o = addr - self.base
        return bytes(self.mem[o:o + n])

    def u32(self, addr):
        return struct.unpack("<I", self.read(addr, 4))[0]

    def piece(self, t):
        o0 = self.u32(self.vocab + 4 * t)
        o1 = self.u32(self.vocab + 4 * t + 4)
        return self.read(self.vocab + o0, o1 - o0)

    def prio(self, t):
        if self.flags & I.TF_PRIO32:
            return self.u32(self.a_prio + 4 * t)
        v = struct.unpack("<H", self.read(self.a_prio + 2 * t, 2))[0]
        return 0xFFFFFFFF if v == 0xFFFF else v

    def find(self, text):
        """Binary search of the sorted index; token id or None."""
        lo, hi = 0, self.n_index - 1
        while lo <= hi:
            self.lookups += 1
            mid = (lo + hi) // 2
            t = self.u32(self.a_index + 4 * mid)
            p = self.piece(t)
            if p == text:
                return t
            if p < text:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    # --- encoding -------------------------------------------------------------
    def encode(self, text, first=True):
        data = text.encode("ascii", errors="replace")
        if self.kind == I.TOK_WORDPIECE:
            return self._wordpiece(data)
        if self.kind == I.TOK_SPM:
            ids = self._merge(b" " + data)
            if first and self.begin != I.NO_TOKEN:
                ids.insert(0, self.begin)
            return ids
        ids = [self.begin] if first and self.begin != I.NO_TOKEN else []
        for chunk in gpt2_chunks(data):
            ids += self._merge(chunk)
        return ids

    def _merge(self, chunk):
        toks = []
        for b in chunk:
            t = self.u32(self.a_bytes + 4 * b)
            if t != I.NO_TOKEN:
                toks.append(t)
        texts = [self.piece(t) for t in toks]
        while len(toks) > 1:
            best, best_i = None, -1
            for i in range(len(toks) - 1):
                t = self.find(texts[i] + texts[i + 1])
                if t is not None:
                    pr = self.prio(t)
                    if pr != 0xFFFFFFFF and (best is None or pr < best[0]):
                        best, best_i = (pr, t), i
            if best is None:
                break
            toks[best_i:best_i + 2] = [best[1]]
            texts[best_i:best_i + 2] = [texts[best_i] + texts[best_i + 1]]
        return toks

    def _wordpiece(self, data):
        if self.flags & I.TF_LOWERCASE:
            data = data.lower()
        words, i = [], 0
        while i < len(data):
            c = data[i:i + 1]
            if c.isalnum() or c == b"_":
                j = i
                while j < len(data) and (data[j:j + 1].isalnum() or data[j:j + 1] == b"_"):
                    j += 1
                words.append(data[i:j])
                i = j
            elif c.isspace():
                i += 1
            else:
                words.append(c)
                i += 1
        ids = [self.begin] if self.begin != I.NO_TOKEN else []
        for w in words:
            start, pieces = 0, []
            while start < len(w):
                end = len(w)
                while end > start:
                    t = self.find((b" " if start == 0 else b"") + w[start:end])
                    if t is not None:
                        pieces.append(t)
                        break
                    end -= 1
                if end == start:
                    pieces = [self.unk]
                    break
                start = end
            ids += pieces
        if self.end != I.NO_TOKEN:
            ids.append(self.end)
        return ids


def gpt2_chunks(data):
    """GPT-2's pre-tokenizer for ASCII: contractions, then runs of letters,
    digits, or other symbols, each optionally preceded by one space; runs of
    whitespace stay together except a single space before the next word."""
    out, i, n = [], 0, len(data)
    alpha = lambda c: 65 <= c <= 90 or 97 <= c <= 122
    digit = lambda c: 48 <= c <= 57
    space = lambda c: c in (9, 10, 11, 12, 13, 32)
    other = lambda c: not (alpha(c) or digit(c) or space(c))
    while i < n:
        if data[i] == 39:                                  # '  contractions
            m = next((s for s in (b"s", b"t", b"re", b"ve", b"m", b"ll", b"d")
                      if data[i + 1:i + 1 + len(s)] == s), None)
            if m is not None:
                out.append(data[i:i + 1 + len(m)])
                i += 1 + len(m)
                continue
        j = i + 1 if data[i] == 32 and i + 1 < n and not space(data[i + 1]) else i
        for cls in (alpha, digit, other):
            if j < n and cls(data[j]):
                k = j
                while k < n and cls(data[k]):
                    k += 1
                out.append(data[i:k])
                i = k
                break
        else:
            # whitespace run; leave one space for the following word
            k = i
            while k < n and space(data[k]):
                k += 1
            if k < n and k - i > 1 and data[k - 1] == 32:
                k -= 1
            out.append(data[i:k])
            i = k
    return out


SAMPLE = """Once upon a time, there was a little girl named Lily. She loved to play outside.
One day, she saw a big, red ball in the park! "Can I play?" she asked.
It's 3:45pm and we've got 1,024 apples... don't you think that's   odd?
The elf and the dwarf walked through the forest; they'd never seen a troll.
Throw the lamp at the troll.  Mr. Smith's dog (aged 12) ran away -- fast!
x=y+2*z; email@example.com #hashtag $100 50% a_b C:\\path\\file.txt
"""


def check(image_path, gguf_path, text=None):
    """Compare the image tokenizer with the reference GGUF tokenizer."""
    import ssnail_convert as C
    import ssnail_sim as S
    m = S.Machine(open(image_path, "rb").read())
    it = ImageTokenizer(m.mem)
    model = C.Model(gguf_path)
    lines = [l for l in (text or SAMPLE).splitlines() if l.strip()]
    bad = 0
    for line in lines:
        a = it.encode(line, first=True)
        b = C.encode(model, line, bos=True)
        if a != b:
            bad += 1
            print(f"DIFFER: {line!r}\n  image: {a}\n  gguf:  {b}")
    total = sum(len(it.encode(l)) for l in lines)
    print(f"{len(lines) - bad}/{len(lines)} lines identical ({total} tokens); "
          f"{it.lookups / max(total, 1):.0f} index probes per token")
    return bad == 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "check":
        txt = open(sys.argv[4]).read() if len(sys.argv) > 4 else None
        sys.exit(0 if check(sys.argv[2], sys.argv[3], txt) else 1)
    print("usage: ssnail_tok.py check image.ssnail model.gguf [textfile]")


# --- Display: raw UTF-8 token bytes -> plain ASCII ----------------------------
# Token text in the image is raw UTF-8, and byte-level vocabularies (GPT-2)
# can split one character across several tokens.  So folding to ASCII is done
# on the output byte stream by the front end, with this small state machine.
# The MEGA65 version needs: the pending bytes (up to 3), and the table below.

FOLD = {0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'",
        0x201C: '"', 0x201D: '"', 0x201E: '"', 0x2033: '"', 0x00AB: '"', 0x00BB: '"',
        0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
        0x2212: "-", 0x2026: "...", 0x00A0: " ", 0x00D7: "x", 0x00DF: "ss",
        0x00E6: "ae", 0x00C6: "AE", 0x0153: "oe", 0x0152: "OE", 0x00F8: "o", 0x00D8: "O",
        0x2022: "*", 0x00B7: ".", 0x2122: "(TM)", 0x00A9: "(c)", 0x00AE: "(R)"}
# Latin-1 letters with accents: the base letter (index = code point - 0xC0)
LATIN1_BASE = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYTsaaaaaaaceeeeiiiidnooooo/ouuuuyty"


def fold_char(cp):
    if cp in (9, 10) or 32 <= cp < 127:
        return chr(cp)
    if cp in FOLD:
        return FOLD[cp]
    if 0xC0 <= cp <= 0xFF:
        return LATIN1_BASE[cp - 0xC0]
    return "?"


class AsciiFolder:
    """Feed raw bytes as tokens arrive; get back printable ASCII."""

    def __init__(self):
        self.pending, self.need, self.cp = 0, 0, 0

    def feed(self, data):
        out = []
        for b in data:
            if self.need:
                if b & 0xC0 == 0x80:                  # continuation byte
                    self.cp = (self.cp << 6) | (b & 0x3F)
                    self.need -= 1
                    if self.need == 0:
                        out.append(fold_char(self.cp))
                    continue
                out.append("?")                       # broken sequence
                self.need = 0
            if b < 0x80:
                out.append(fold_char(b))
            elif b & 0xE0 == 0xC0:
                self.cp, self.need = b & 0x1F, 1
            elif b & 0xF0 == 0xE0:
                self.cp, self.need = b & 0x0F, 2
            elif b & 0xF8 == 0xF0:
                self.cp, self.need = b & 0x07, 3
            else:
                out.append("?")
        return "".join(out)
