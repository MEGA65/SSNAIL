"""SHITE (Spreadsheet Hardware Integrated Technology Engine): the reference
model.  It defines the hardware's results exactly; the layout and semantics
are in mega65-core's docs/shite/README.md.

The instructions work on any object with read(addr, n) -> bytes and
write(addr, data), as ssnail_sim.Machine (SSNAIL runs them as opcodes
$40-$43).  Sheet builds sheets in such a memory for tests.
"""
import struct

GRANULE = 16
EMPTY_ROW = 0xFFFFFFFF
MAGIC = b"SHT1"

# cell kinds (header byte 2, bits 3-0) and flags (bits 7-4)
K_EMPTY, K_TEXT, K_FORMULA, K_F64, K_I64, K_I32, K_ERROR, K_UNCOMPILED = range(8)
K_FREE = 15
F_CHANGED, F_ERROR, F_HAS_FORMAT = 0x10, 0x20, 0x40

MAX_ROWS, MAX_COLS = 1 << 20, 1 << 24
MAX_BLOCK = 4096                   # granules in a row block (12-bit links)
# descriptor fields
D_ROWTAB, D_NROWS, D_NCOLS, D_CL, D_CLCAP, D_PREVCL = 4, 8, 12, 16, 20, 24
D_ROUND, D_CHANGED, D_EVALS, D_ERRORS, D_FLAGS, D_BADROW = 28, 32, 36, 40, 44, 48
D_PREVCHANGED, D_NAMES, D_DECSEP, D_ARGSEP = 52, 56, 60, 61
BAD_DESCRIPTOR = 0xFFFFFFFF
# row block header fields
R_ROW, R_LEN, R_NCELLS, R_ROOT, R_FREE, R_CHANGED, R_FLAGS = 0, 4, 6, 8, 10, 12, 14
ROW_STALE = 1


class ShiteFormatError(Exception):
    """A malformed descriptor or row block: SSNAIL ends the job with
    ERR_FORMAT, the row in the descriptor (BAD_DESCRIPTOR: the descriptor
    itself)."""
    def __init__(self, row, why):
        super().__init__(f"row {row}: {why}")
        self.row = row


def u16(m, a):
    return struct.unpack("<H", m.read(a, 2))[0]


def u32(m, a):
    return struct.unpack("<I", m.read(a, 4))[0]


def put16(m, a, v):
    m.write(a, struct.pack("<H", v & 0xFFFF))


def put32(m, a, v):
    m.write(a, struct.pack("<I", v & 0xFFFFFFFF))


# The cell header: column (24 bits), kind and flags, length (granules),
# tree links (24 bits: left in bits 11-0, right in 23-12)
def hdr_col(h):
    return h[0] | h[1] << 8 | h[2] << 16


def hdr_links(h):
    v = h[5] | h[6] << 8 | h[7] << 16
    return v & 0xFFF, v >> 12


def pack_hdr(col, kf, n, left=0, right=0):
    v = left | right << 12
    return bytes([col & 0xFF, col >> 8 & 0xFF, col >> 16 & 0xFF, kf, n,
                  v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF])


class Descriptor:
    def __init__(self, m, a):
        if m.read(a, 4) != MAGIC:
            raise ShiteFormatError(BAD_DESCRIPTOR, "no sheet descriptor")
        self.m, self.a = m, a
        self.rowtab = u32(m, a + D_ROWTAB)
        self.nrows = u32(m, a + D_NROWS)
        self.ncols = u32(m, a + D_NCOLS)

    def row_block(self, row):
        """The row's block address, or None (an empty row, or out of range)."""
        if row >= self.nrows:
            return None
        b = u32(self.m, self.rowtab + 4 * row)
        return None if b == EMPTY_ROW else b


def cells_in_block(m, b, row):
    """(granule offset, header bytes) of each cell but FREE ones, in memory
    order, checking the block's structure."""
    if u32(m, b + R_ROW) != row:
        raise ShiteFormatError(row, "row number does not match the row table")
    blen, free = u16(m, b + R_LEN), u16(m, b + R_FREE)
    if free == 0 or free > blen or blen > MAX_BLOCK:
        raise ShiteFormatError(row, "first free granule outside the block")
    out, g = [], 1
    while g < free:
        h = m.read(b + GRANULE * g, 8)
        n = h[4]
        if n == 0 or g + n > free:
            raise ShiteFormatError(row, f"cell at granule {g}: bad length {n}")
        if h[3] & 0x0F != K_FREE:
            out.append((g, h))
        g += n
    return out


def find_cell(m, desc_addr, row, col):
    """FIND_CELL: the cell's address, or 0 (a tree walk from the root)."""
    d = Descriptor(m, desc_addr)
    b = d.row_block(row)
    if b is None:
        return 0
    g = u16(m, b + R_ROOT)
    while g:
        c = b + GRANULE * g
        h = m.read(c, 8)
        cc = hdr_col(h)
        if cc == col:
            return c
        left, right = hdr_links(h)
        g = left if col < cc else right
    return 0


def balanced_links(n):
    """For cells sorted 0..n-1: (root index, left[], right[]) of the tree
    REBALANCE_ROW builds (root of lo..hi at (lo + hi) // 2); -1 = none."""
    left, right = [-1] * n, [-1] * n

    def build(lo, hi):
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        left[mid] = build(lo, mid - 1)
        right[mid] = build(mid + 1, hi)
        return mid
    return build(0, n - 1), left, right


def rebalance_row(m, desc_addr, first, last):
    """REBALANCE_ROW for rows first..last (rows without a block skipped)."""
    d = Descriptor(m, desc_addr)
    for row in range(first, min(last, d.nrows - 1) + 1):
        b = d.row_block(row)
        if b is None:
            continue
        cells = sorted(cells_in_block(m, b, row), key=lambda gh: hdr_col(gh[1]))
        cols = [hdr_col(h) for _, h in cells]
        if any(cols[i] == cols[i + 1] for i in range(len(cols) - 1)):
            raise ShiteFormatError(row, "two cells in one column")
        root, left, right = balanced_links(len(cells))
        gr = lambda i: 0 if i < 0 else cells[i][0]
        for i, (g, h) in enumerate(cells):
            m.write(b + GRANULE * g, pack_hdr(hdr_col(h), h[3], h[4], gr(left[i]), gr(right[i])))
        put16(m, b + R_NCELLS, len(cells))
        put16(m, b + R_ROOT, gr(root))
        put16(m, b + R_FLAGS, u16(m, b + R_FLAGS) & ~ROW_STALE)


# --- building sheets (tests, tools) -------------------------------------------
def granules(nbytes):
    return (nbytes + GRANULE - 1) // GRANULE


class Sheet:
    """A sheet to lay out in memory: cells added in any order, each a kind
    and its body bytes (after the 8-byte header)."""
    def __init__(self, nrows, ncols):
        self.nrows, self.ncols = nrows, ncols
        self.rows = {}                      # row -> [(col, kind | flags, body)]

    def add(self, row, col, kind, body=b"", flags=0):
        assert 0 <= row < min(self.nrows, MAX_ROWS) and 0 <= col < min(self.ncols, MAX_COLS)
        self.rows.setdefault(row, []).append((col, kind | flags, bytes(body)))

    def number(self, row, col, v, kind=None, fmt=None, flags=0):
        if kind is None:
            kind = K_F64 if isinstance(v, float) else K_I64
        val = (struct.pack("<d", v) if kind == K_F64 else
               struct.pack("<q", v) if kind == K_I64 else struct.pack("<iI", v, 0xFFFFFFFF if v < 0 else 0))
        if fmt is not None:
            flags |= F_HAS_FORMAT
        self.add(row, col, kind, val + (bytes(fmt) if fmt is not None else b""), flags)

    def text(self, row, col, s, fmt=bytes(8)):
        t = s.encode("ascii")
        self.add(row, col, K_TEXT, bytes(fmt) + struct.pack("<H", len(t)) + t)

    def free(self, row, ngranules):
        """A FREE hole of ngranules (the header's granule included)."""
        self.add(row, 0, K_FREE, bytes(GRANULE * ngranules - 8))

    def layout(self, m, base, slack=2, balanced=True, change_cap=64):
        """Write the sheet at base (16-aligned): descriptor, change lists, row
        table, row blocks (each with `slack` spare granules).  Returns the
        descriptor's address.  balanced: build each row's tree (as
        REBALANCE_ROW would); else leave the links 0 and the rows stale."""
        a = base
        desc = a; a += 64
        cl = a; a += 16 * granules(8 * change_cap)
        prev = a; a += 16 * granules(8 * change_cap)
        rowtab = a; a += 16 * granules(4 * self.nrows)
        m.write(desc, MAGIC + struct.pack("<IIIIII", rowtab, self.nrows, self.ncols,
                                          cl, change_cap, prev) + bytes(36))
        for r in range(self.nrows):
            put32(m, rowtab + 4 * r, EMPTY_ROW)
        for r in sorted(self.rows):
            cells = []
            for col, kf, body in self.rows[r]:
                n = granules(8 + len(body))
                cells.append([col, kf, n, body])
            free = 1 + sum(c[2] for c in cells)
            blen = free + slack
            b = a; a += GRANULE * blen
            put32(m, rowtab + 4 * r, b)
            m.write(b, bytes(GRANULE * blen))
            g = 1
            for c in cells:
                c.append(g)
                m.write(b + GRANULE * g, pack_hdr(c[0], c[1], c[2]) + c[3])
                g += c[2]
            assert blen <= MAX_BLOCK, "row too long for one row block"
            m.write(b, struct.pack("<IHHHHHH", r, blen, 0, 0, free, 0, ROW_STALE))
        if balanced:
            rebalance_row(m, desc, 0, self.nrows - 1)
        return desc


# --- values -------------------------------------------------------------------------
# A value on the evaluation stack: kind 'i' (I64, with fixed point k: the
# format byte, bit 7 binary), 'f' (F64), 't' (text bytes), 'e' (error code),
# 'z' (an empty cell: I64 0, skipped by aggregates), 'r' (a range: r1, c1,
# r2, c2).  ref: it came from a reference (aggregates skip its text).
E_DIV0, E_VALUE, E_REF, E_NUM, E_CIRC, E_NA, E_STACK, E_NAME, E_COMPILE = range(1, 10)
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
DEC_KMAX, BIN_KMAX = 18, 62
FLT_MIN = 2.2250738585072014e-308


class V:
    __slots__ = ("kind", "v", "k", "ref")

    def __init__(self, kind, v=0, k=0, ref=False):
        self.kind, self.v, self.k, self.ref = kind, v, k, ref

    def __repr__(self):
        return f"V({self.kind},{self.v!r},k={self.k},ref={self.ref})"


def err(code):
    return V("e", code)


def ftz(x):
    """Subnormals flushed to zero, keeping the sign."""
    import math
    return math.copysign(0.0, x) if x != 0 and abs(x) < FLT_MIN else x


def to_f64(a):
    """An integer value (with its fixed point) as F64."""
    import math
    v = 0 if a.kind == "z" else a.v
    if a.kind == "f":
        return a.v
    k = a.k & 0x7F
    if k == 0:
        return ftz(float(v))
    if a.k & 0x80:
        return ftz(math.ldexp(float(v), -k))
    return ftz(float(v) / float(10 ** k))


def i64(v, k=0):
    return V("i", v, k) if I64_MIN <= v <= I64_MAX else err(E_NUM)


def scale_up(v, k_from, k_to, binary):
    """v at fixed point k_from rescaled up to k_to (exactly), or None on
    overflow."""
    r = v << (k_to - k_from) if binary else v * 10 ** (k_to - k_from)
    return r if I64_MIN <= r <= I64_MAX else None


def div_away(num, den):
    """num / den rounded half away from zero (exact integers)."""
    q = (2 * abs(num) + abs(den)) // (2 * abs(den))
    return q if (num < 0) == (den < 0) else -q


def same_base(a, b):
    """Both integers whose fixed points can be aligned (the same base, or
    one of them an ordinary integer): (base binary?, ka, kb), or None."""
    ka, kb = a.k & 0x7F, b.k & 0x7F
    if ka and kb and (a.k & 0x80) != (b.k & 0x80):
        return None
    binary = bool((a.k if ka else b.k) & 0x80)
    return binary, ka, kb


def as_num(a):
    """A scalar operand: 'z' is I64 0; ranges and text are #VALUE!."""
    if a.kind == "z":
        return V("i", 0, 0)
    if a.kind in ("r", "t"):
        return err(E_VALUE)
    return a


def binop(op, a, b):
    """ADD SUB MUL DIV and the comparisons (section 4)."""
    a, b = as_num(a), as_num(b)
    if a.kind == "e":
        return a
    if b.kind == "e":
        return b
    if a.kind == "i" and b.kind == "i":
        sb = same_base(a, b)
        if sb is not None:
            binary, ka, kb = sb
            if op in ("ADD", "SUB") or op in CMP_OPS:
                k = max(ka, kb)
                va, vb = scale_up(a.v, ka, k, binary), scale_up(b.v, kb, k, binary)
                if va is None or vb is None:
                    return err(E_NUM)
                kb_ = k | (0x80 if binary and k else 0)
                if op == "ADD":
                    return i64(va + vb, kb_)
                if op == "SUB":
                    return i64(va - vb, kb_)
                return V("i", int(CMP_OPS[op](va, vb)))
            if op == "MUL":
                k = ka + kb
                if k <= (BIN_KMAX if binary else DEC_KMAX):
                    return i64(a.v * b.v, k | (0x80 if binary and k else 0))
            if op == "DIV":
                if b.v == 0:
                    return err(E_DIV0)
                if ka and ka == kb and not binary:
                    return i64(div_away(a.v * 10 ** ka, b.v), ka)
    # F64
    x, y = to_f64(a), to_f64(b)
    if op in CMP_OPS:
        return V("i", int(CMP_OPS[op](x, y)))
    if op == "DIV" and y == 0:
        return err(E_DIV0)
    try:
        r = {"ADD": lambda: x + y, "SUB": lambda: x - y, "MUL": lambda: x * y,
             "DIV": lambda: x / y}[op]()
    except OverflowError:
        r = float("inf")
    return V("f", ftz(r))


import operator as _op
CMP_OPS = {"EQ": _op.eq, "NE": _op.ne, "LT": _op.lt, "LE": _op.le, "GT": _op.gt, "GE": _op.ge}


def neg(a):
    a = as_num(a)
    if a.kind == "e":
        return a
    if a.kind == "i":
        return i64(-a.v, a.k)
    return V("f", -a.v)


# --- bytecode -----------------------------------------------------------------------
OP_END, OP_I8, OP_I32, OP_I64, OP_F64, OP_STR, OP_FIX, OP_REF, OP_RECT, OP_HOSTCALL, \
    OP_JZ, OP_JMP, OP_CALL = range(13)
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG, OP_PCT, OP_CONCAT = range(0x10, 0x18)
OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE = range(0x18, 0x1E)
FN_SUM, FN_AVERAGE, FN_MIN, FN_MAX, FN_COUNT = range(5)
STACK_DEPTH = 32
_BINOPS = {OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL", OP_DIV: "DIV", OP_EQ: "EQ",
           OP_NE: "NE", OP_LT: "LT", OP_LE: "LE", OP_GT: "GT", OP_GE: "GE"}


class Abort(Exception):
    """Ends a formula's evaluation with an error result."""
    def __init__(self, code):
        self.code = code


def varlen(b, i):
    """A one- or two-byte count / identifier at b[i]: (value, next i)."""
    if i >= len(b):
        raise Abort(E_STACK)
    if b[i] < 0x80:
        return b[i], i + 1
    if i + 1 >= len(b):
        raise Abort(E_STACK)
    return (b[i] & 0x7F) | b[i + 1] << 7, i + 2


def sext(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def ref_at(b, i, frow, fcol):
    """A reference (short form, or REF and the long form) at b[i]:
    (row, col, next i), absolute and wrapped."""
    if i < len(b) and b[i] >= 0x80:
        if i + 4 > len(b):
            raise Abort(E_STACK)
        w = int.from_bytes(b[i:i + 4], "big")
        ra, rv = w >> 30 & 1, w >> 15 & 0x7FFF
        ca, cv = w >> 14 & 1, w & 0x3FFF
        row = rv if ra else (frow + sext(rv, 15)) % MAX_ROWS
        col = cv if ca else (fcol + sext(cv, 14)) % MAX_COLS
        return row, col, i + 4
    if i < len(b) and b[i] == OP_REF:
        if i + 7 > len(b):
            raise Abort(E_STACK)
        w = int.from_bytes(b[i + 1:i + 7], "big")
        ra, ca = w >> 47 & 1, w >> 46 & 1
        rv, cv = w >> 24 & 0xFFFFF, w & 0xFFFFFF
        row = rv if ra else (frow + rv) % MAX_ROWS
        col = cv if ca else (fcol + cv) % MAX_COLS
        return row, col, i + 7
    raise Abort(E_STACK)


class Sheetview:
    """Reading cells of a sheet in memory for evaluation."""
    def __init__(self, m, d):
        self.m, self.d = m, d

    def find(self, row, col):
        if row >= self.d.nrows or col >= self.d.ncols:
            return None
        b = self.d.row_block(row)
        if b is None:
            return None
        g = u16(self.m, b + R_ROOT)
        while g:
            c = b + GRANULE * g
            h = self.m.read(c, 8)
            cc = hdr_col(h)
            if cc == col:
                return c
            left, right = hdr_links(h)
            g = left if col < cc else right
        return None

    def value(self, c):
        """A cell's value (c its address, or None: no cell)."""
        m = self.m
        if c is None:
            return V("z", ref=True)
        h = m.read(c, 8)
        kind, flags = h[3] & 0x0F, h[3] & 0xF0
        if kind == K_EMPTY:
            return V("z", ref=True)
        if kind == K_TEXT:
            n = u16(m, c + 16)
            return V("t", m.read(c + 18, n), ref=True)
        if kind == K_F64:
            return V("f", ftz(struct.unpack("<d", m.read(c + 8, 8))[0]), ref=True)
        if kind in (K_I64, K_I32):
            v = (struct.unpack("<q", m.read(c + 8, 8))[0] if kind == K_I64 else
                 struct.unpack("<i", m.read(c + 8, 4))[0])
            k = m.read(c + 18, 1)[0] if flags & F_HAS_FORMAT else 0
            return V("i", v, k, ref=True)
        if kind == K_ERROR:
            return V("e", m.read(c + 8, 1)[0], ref=True)
        if kind == K_FORMULA:
            rk = m.read(c + 24, 1)[0]
            if rk == K_F64:
                return V("f", ftz(struct.unpack("<d", m.read(c + 16, 8))[0]), ref=True)
            if rk in (K_I64, K_I32):
                return V("i", struct.unpack("<q", m.read(c + 16, 8))[0], m.read(c + 10, 1)[0], ref=True)
            if rk == K_ERROR:
                return V("e", m.read(c + 16, 1)[0], ref=True)
            if rk == K_TEXT:
                n = u16(m, c + 16)
                return V("t", m.read(c + 32 + u16(m, c + 26), n), ref=True)
            return V("z", ref=True)                 # never computed
        if kind == K_UNCOMPILED:
            return V("e", E_COMPILE, ref=True)
        return V("z", ref=True)

    def row_cells(self, row, c1, c2):
        """The addresses of a row's cells with c1 <= column <= c2, in column
        order (an in-order walk, at most 32 deep)."""
        b = self.d.row_block(row)
        if b is None:
            return []
        out = []

        def walk(g, depth):
            if not g:
                return
            if depth > 32:
                raise ShiteFormatError(row, "tree deeper than 32")
            c = b + GRANULE * g
            h = self.m.read(c, 8)
            cc = hdr_col(h)
            left, right = hdr_links(h)
            if c1 < cc:
                walk(left, depth + 1)
            if c1 <= cc <= c2:
                out.append(c)
            if cc < c2:
                walk(right, depth + 1)
        walk(u16(self.m, b + R_ROOT), 1)
        return out


def aggregate(sv, fid, args):
    """SUM AVERAGE MIN MAX COUNT over the arguments, in order (section 4)."""
    nums = []

    def take(v, from_cells):
        if v.kind == "e":
            raise Abort(v.v)
        if v.kind in ("i", "f"):
            nums.append(v)
        elif v.kind in ("t", "z"):
            if not from_cells:
                raise Abort(E_VALUE)
    for a in args:
        if a.kind == "r":
            r1, c1, r2, c2 = a.v
            for r in range(r1, min(r2, sv.d.nrows - 1) + 1):
                for c in sv.row_cells(r, c1, c2):
                    take(sv.value(c), True)
        else:
            take(a, a.ref)
    if fid == FN_COUNT:
        return V("i", len(nums))
    if fid in (FN_MIN, FN_MAX):
        if not nums:
            return V("i", 0)
        best = nums[0]
        for v in nums[1:]:
            better = binop("LT", v, best) if fid == FN_MIN else binop("GT", v, best)
            if better.kind == "e":
                return better
            if better.v:
                best = v
        return V(best.kind, best.v, best.k)
    s = V("i", 0)
    for v in nums:
        s = binop("ADD", s, v)
        if s.kind == "e":
            return s
    if fid == FN_SUM:
        return s
    if not nums:
        return err(E_DIV0)
    return binop("DIV", s, V("i", len(nums)))


def evaluate(sv, code, frow, fcol):
    """A formula's value: its bytecode run (section 5)."""
    st = []

    def push(v):
        if len(st) >= STACK_DEPTH:
            raise Abort(E_STACK)
        st.append(v)

    def pop():
        if not st:
            raise Abort(E_STACK)
        return st.pop()
    i = 0
    try:
        while True:
            if i >= len(code):
                raise Abort(E_STACK)
            op = code[i]
            if op >= 0x80:
                r, c, i = ref_at(code, i, frow, fcol)
                push(sv.value(sv.find(r, c)) if r < sv.d.nrows and c < sv.d.ncols else err(E_REF))
                continue
            i += 1
            if op == OP_END:
                if len(st) != 1:
                    raise Abort(E_STACK)
                return st[0]
            if op == OP_I8:
                push(V("i", sext(code[i], 8))); i += 1
            elif op == OP_I32:
                push(V("i", struct.unpack("<i", code[i:i + 4])[0])); i += 4
            elif op == OP_I64:
                push(V("i", struct.unpack("<q", code[i:i + 8])[0])); i += 8
            elif op == OP_F64:
                push(V("f", ftz(struct.unpack("<d", code[i:i + 8])[0]))); i += 8
            elif op == OP_FIX:
                a = pop()
                k = code[i]; i += 1
                push(V("i", a.v, k) if a.kind == "i" else (a if a.kind == "e" else err(E_VALUE)))
            elif op == OP_REF:
                r, c, i = ref_at(code, i - 1, frow, fcol)
                push(sv.value(sv.find(r, c)) if r < sv.d.nrows and c < sv.d.ncols else err(E_REF))
            elif op == OP_RECT:
                r1, c1, i = ref_at(code, i, frow, fcol)
                r2, c2, i = ref_at(code, i, frow, fcol)
                if max(r1, r2) >= sv.d.nrows or max(c1, c2) >= sv.d.ncols:
                    push(err(E_REF))
                else:
                    push(V("r", (min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2))))
            elif op == OP_HOSTCALL:
                n, i = varlen(code, i)
                i += n
                r, c, i = ref_at(code, i, frow, fcol)
                push(sv.value(sv.find(r, c)) if r < sv.d.nrows and c < sv.d.ncols else err(E_REF))
            elif op == OP_CALL:
                fid, i = varlen(code, i)
                argc, i = varlen(code, i)
                if argc > len(st):
                    raise Abort(E_STACK)
                args = st[len(st) - argc:]
                del st[len(st) - argc:]
                push(aggregate(sv, fid, args) if fid <= FN_COUNT else err(E_NAME))
            elif op in _BINOPS:
                b = pop(); a = pop()
                push(binop(_BINOPS[op], a, b))
            elif op == OP_NEG:
                push(neg(pop()))
            elif op in (OP_POW, OP_PCT, OP_CONCAT):         # not in this phase
                if op != OP_PCT:
                    pop()
                pop()
                push(err(E_NAME))
            elif op == OP_STR:
                n, i = varlen(code, i)
                i += n
                push(err(E_NAME))
            elif op in (OP_JZ, OP_JMP):
                raise Abort(E_NAME)
            else:
                raise Abort(E_STACK)
    except Abort as e:
        return err(e.code)


def store_form(v, fk):
    """A formula's value as stored: (result kind, 8 bytes), rescaled to the
    cell's fixed point fk (section 4)."""
    if v.kind == "z":
        v = V("i", 0)
    if v.kind == "i":
        k, kf = v.k & 0x7F, fk & 0x7F
        if k and kf and (v.k & 0x80) == (fk & 0x80):
            binary = bool(fk & 0x80)
            if kf >= k:
                r = scale_up(v.v, k, kf, binary)
                if r is None:
                    return K_ERROR, struct.pack("<Q", E_NUM)
            else:
                r = (div_away(v.v, 1 << (k - kf)) if binary else div_away(v.v, 10 ** (k - kf)))
            return K_I64, struct.pack("<q", r)
        if k == 0 and kf:
            r = scale_up(v.v, 0, kf, bool(fk & 0x80))
            if r is None:
                return K_ERROR, struct.pack("<Q", E_NUM)
            return K_I64, struct.pack("<q", r)
        if k == 0:
            return K_I64, struct.pack("<q", v.v)
        v = V("f", to_f64(v))
    if v.kind == "f":
        import math
        if math.isnan(v.v) or math.isinf(v.v):
            return K_ERROR, struct.pack("<Q", E_NUM)
        return K_F64, struct.pack("<d", ftz(v.v))
    if v.kind == "e":
        return K_ERROR, struct.pack("<Q", v.v)
    return K_ERROR, struct.pack("<Q", E_VALUE)           # text, ranges: later phases


def recalc(m, desc_addr, r1, r2, c1, c2, begin=True, end=True):
    """RECALC: returns the cells this call changed (section 6)."""
    d = Descriptor(m, desc_addr)
    sv = Sheetview(m, d)
    a = desc_addr
    if begin:
        for f in (D_CHANGED, D_EVALS, D_ERRORS):
            put32(m, a + f, 0)
        m.write(a + D_FLAGS, bytes([m.read(a + D_FLAGS, 1)[0] & ~3 & 0xFF]))
    changed_call = 0
    for row in range(r1, min(r2, d.nrows - 1) + 1):
        b = d.row_block(row)
        if b is None:
            continue
        row_changed = 0
        for g, h in cells_in_block(m, b, row):
            col = hdr_col(h)
            if h[3] & 0x0F != K_FORMULA or not c1 <= col <= c2:
                continue
            c = b + GRANULE * g
            n = u16(m, c + 26)
            code = m.read(c + 32, n)
            v = evaluate(sv, code, row, col)
            rk, rv = store_form(v, m.read(c + 10, 1)[0])
            put32(m, a + D_EVALS, u32(m, a + D_EVALS) + 1)
            if rk == K_ERROR:
                put32(m, a + D_ERRORS, u32(m, a + D_ERRORS) + 1)
            if m.read(c + 24, 1)[0] != rk or m.read(c + 16, 8) != rv:
                m.write(c + 16, rv)
                m.write(c + 24, bytes([rk]))
                fl = (h[3] & 0x0F) | (h[3] & 0xC0) | F_CHANGED | (F_ERROR if rk == K_ERROR else 0)
                m.write(c + 3, bytes([fl]))
                k = u32(m, a + D_CHANGED)
                cap = u32(m, a + D_CLCAP)
                if k < cap:
                    m.write(u32(m, a + D_CL) + 8 * k, struct.pack("<II", row, col))
                else:
                    m.write(a + D_FLAGS, bytes([m.read(a + D_FLAGS, 1)[0] | 1]))
                put32(m, a + D_CHANGED, k + 1)
                row_changed += 1
                changed_call += 1
        put16(m, b + R_CHANGED, row_changed)
    if end:
        k, pk = u32(m, a + D_CHANGED), u32(m, a + D_PREVCHANGED)
        cl, pcl, cap = u32(m, a + D_CL), u32(m, a + D_PREVCL), u32(m, a + D_CLCAP)
        if pcl:
            same = k == pk and k <= cap and m.read(cl, 8 * k) == m.read(pcl, 8 * k)
            fl = m.read(a + D_FLAGS, 1)[0]
            m.write(a + D_FLAGS, bytes([fl | 2 if same else fl & ~2 & 0xFF]))
            put32(m, a + D_CL, pcl)
            put32(m, a + D_PREVCL, cl)
        put32(m, a + D_ROUND, u32(m, a + D_ROUND) + 1)
        put32(m, a + D_PREVCHANGED, k)
    return changed_call


# --- assembling formulas (tests, tools) ----------------------------------------------
class F:
    """A formula's bytecode, built in postfix order."""
    def __init__(self):
        self.b = bytearray()

    def i(self, v):
        if -128 <= v < 128:
            self.b += bytes([OP_I8, v & 0xFF])
        elif -(1 << 31) <= v < 1 << 31:
            self.b += bytes([OP_I32]) + struct.pack("<i", v)
        else:
            self.b += bytes([OP_I64]) + struct.pack("<q", v)
        return self

    def f(self, x):
        self.b += bytes([OP_F64]) + struct.pack("<d", x)
        return self

    def fix(self, k):
        self.b += bytes([OP_FIX, k])
        return self

    def _ref(self, row, col, rabs, cabs, long=False):
        """row / col: numbers (absolute) or offsets (relative)."""
        rs = row if rabs else row
        short_ok = (0 <= row < 32768 if rabs else -16384 <= row < 16384) and \
                   (0 <= col < 16384 if cabs else -8192 <= col < 8192)
        if short_ok and not long:
            w = 1 << 31 | rabs << 30 | (row & 0x7FFF) << 15 | cabs << 14 | (col & 0x3FFF)
            return w.to_bytes(4, "big")
        w = rabs << 47 | cabs << 46 | (rs % MAX_ROWS) << 24 | (col % MAX_COLS)
        return bytes([OP_REF]) + w.to_bytes(6, "big")

    def ref(self, row, col, rabs=False, cabs=False, long=False):
        self.b += self._ref(row, col, rabs, cabs, long)
        return self

    def rect(self, a, b):
        """a, b: (row, col, rabs, cabs) corners."""
        self.b += bytes([OP_RECT]) + self._ref(*a) + self._ref(*b)
        return self

    def host(self, text, row, col, rabs=True, cabs=True):
        t = text.encode("ascii")
        self.b += bytes([OP_HOSTCALL]) + varlen_bytes(len(t)) + t + self._ref(row, col, rabs, cabs)
        return self

    def call(self, fid, argc):
        self.b += bytes([OP_CALL]) + varlen_bytes(fid) + varlen_bytes(argc)
        return self

    def op(self, o):
        self.b += bytes([o])
        return self

    def end(self):
        self.b += bytes([OP_END])
        return bytes(self.b)


def varlen_bytes(v):
    return bytes([v]) if v < 0x80 else bytes([0x80 | v & 0x7F, v >> 7])


def formula_body(code, fmt=bytes(8), text_cap=0):
    """A FORMULA cell's body (after the header): format, result (never
    computed), its info, the bytecode, the result text area."""
    return bytes(fmt) + bytes(8) + struct.pack("<BBHI", 0, text_cap, len(code), 0) + code + bytes(text_cap)


def _sheet_formula(self, row, col, code, fmt=bytes(8), text_cap=0):
    self.add(row, col, K_FORMULA, formula_body(code, fmt, text_cap))


Sheet.formula = _sheet_formula
