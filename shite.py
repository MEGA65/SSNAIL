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
    return list(iter_cells(m, b, row))


def iter_cells(m, b, row):
    """As cells_in_block, one cell at a time: a malformed cell raises only
    when the walk reaches it (RECALC has evaluated the cells before it)."""
    if u32(m, b + R_ROW) != row:
        raise ShiteFormatError(row, "row number does not match the row table")
    blen, free = u16(m, b + R_LEN), u16(m, b + R_FREE)
    if free == 0 or free > blen or blen > MAX_BLOCK:
        raise ShiteFormatError(row, "first free granule outside the block")
    g = 1
    while g < free:
        h = m.read(b + GRANULE * g, 8)
        n = h[4]
        if n == 0 or g + n > free:
            raise ShiteFormatError(row, f"cell at granule {g}: bad length {n}")
        if h[3] & 0x0F != K_FREE:
            yield g, h
        g += n


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
# 'z' (blank: an empty cell or none; I64 0 to arithmetic, skipped by
# aggregates), 'r' (a range: r1, c1, r2, c2), 'l' (a cell list).  ref: it came from a reference (aggregates skip its text).
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


def bad_k(k):
    """A fixed point past its limits (decimal 18, binary 62): #NUM!."""
    return (k & 0x7F) > (BIN_KMAX if k & 0x80 else DEC_KMAX)


def as_num(a):
    """A scalar operand: 'z' is I64 0; ranges and text are #VALUE!; an
    integer whose fixed point is past its limits is #NUM!."""
    if a.kind == "z":
        return V("i", 0, 0)
    if a.kind == "i" and bad_k(a.k):
        return err(E_NUM)
    if a.kind in ("r", "t", "l"):
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
    OP_JZ, OP_JMP, OP_CALL, OP_CELLS = range(14)
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG, OP_PCT, OP_CONCAT = range(0x10, 0x18)
OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE = range(0x18, 0x1E)
FN_SUM, FN_AVERAGE, FN_MIN, FN_MAX, FN_COUNT, FN_COUNTA, FN_PRODUCT, FN_ABS, FN_INT, \
    FN_TRUNC, FN_ROUND, FN_SIGN, FN_SQRT, FN_MOD, FN_AND, FN_OR, FN_NOT = range(17)
AGGREGATES = (FN_SUM, FN_AVERAGE, FN_MIN, FN_MAX, FN_COUNT, FN_COUNTA, FN_PRODUCT, FN_AND, FN_OR)
FN_POWER = 29
ARITY = {FN_ABS: 1, FN_INT: 1, FN_TRUNC: 1, FN_SIGN: 1, FN_MOD: 2, FN_NOT: 1, FN_ROUND: 2,
         FN_SQRT: 1, FN_POWER: 2}
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


def long_ref(w, frow, fcol):
    """The long form's 48-bit payload: (row, col), absolute and wrapped."""
    ra, ca = w >> 47 & 1, w >> 46 & 1
    rv, cv = w >> 24 & 0xFFFFF, w & 0xFFFFFF
    return (rv if ra else (frow + rv) % MAX_ROWS), (cv if ca else (fcol + cv) % MAX_COLS)


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
        row, col = long_ref(int.from_bytes(b[i + 1:i + 7], "big"), frow, fcol)
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
    """The aggregates over the arguments, in order (section 5.7)."""
    def items():
        """(value, from cells) of every argument, ranges and lists expanded."""
        for a in args:
            if a.kind == "l":
                for r, c in a.v:
                    if r >= sv.d.nrows or c >= sv.d.ncols:
                        raise Abort(E_REF)
                    yield sv.value(sv.find(r, c)), True
            elif a.kind == "r":
                r1, c1, r2, c2 = a.v
                for r in range(r1, min(r2, sv.d.nrows - 1) + 1):
                    for c in sv.row_cells(r, c1, c2):
                        yield sv.value(c), True
            else:
                yield a, a.ref
    if fid == FN_COUNTA:                 # every value but blanks, errors too
        return V("i", sum(1 for v, _ in items() if v.kind != "z"))
    nums = []
    for v, from_cells in items():
        if v.kind == "e":
            raise Abort(v.v)
        if v.kind in ("i", "f"):
            nums.append(v)
        elif v.kind in ("t", "z"):
            if not from_cells:
                raise Abort(E_VALUE)
    if fid == FN_COUNT:
        return V("i", len(nums))
    if fid in (FN_AND, FN_OR):
        if not nums:
            return err(E_VALUE)
        for v in nums:
            if v.kind == "i" and bad_k(v.k):
                return err(E_NUM)
        t = [v.v != 0 for v in nums]
        return V("i", int(all(t) if fid == FN_AND else any(t)))
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
    if fid == FN_PRODUCT:
        if not nums:
            return V("i", 0)
        p = V("i", 1)
        for v in nums:
            p = binop("MUL", p, v)
            if p.kind == "e":
                return p
        return p
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


def truth(v):
    """IF's and NOT's test: zero (blank, I64 0, F64 +-0) is false; errors
    and text end the formula (#VALUE! for text)."""
    if v.kind == "e":
        raise Abort(v.v)
    if v.kind in ("t", "r", "l"):
        raise Abort(E_VALUE)
    if v.kind == "z":
        return False
    if v.kind == "i" and bad_k(v.k):
        raise Abort(E_NUM)
    return v.v != 0


def zsign(r, x):
    """r (a float), with x's sign when it is zero (IEEE floor and trunc)."""
    import math
    return math.copysign(0.0, x) if r == 0 else float(r)


def scalar_fn(fid, args):
    """ABS INT TRUNC SIGN MOD NOT (section 5.7)."""
    import math
    if len(args) != ARITY[fid]:
        return err(E_VALUE)
    if fid == FN_POWER:
        return power(*args)
    if fid == FN_ROUND:
        return round_fn(*args)
    if fid == FN_SQRT:
        return sqrt_fn(args[0])
    if fid == FN_NOT:
        a = as_num(args[0])
        if a.kind == "e":
            return a
        return V("i", int(a.v == 0))
    a = as_num(args[0])
    if a.kind == "e":
        return a
    k, binary = a.k & 0x7F, bool(a.k & 0x80)
    if fid == FN_ABS:
        return i64(abs(a.v), a.k) if a.kind == "i" else V("f", abs(a.v))
    if fid == FN_SIGN:
        if a.kind == "f" and math.isnan(a.v):
            return err(E_NUM)
        return V("i", (a.v > 0) - (a.v < 0))
    if fid in (FN_INT, FN_TRUNC):
        if a.kind == "f":
            if math.isinf(a.v) or math.isnan(a.v):
                return a
            return V("f", zsign(math.floor(a.v) if fid == FN_INT else math.trunc(a.v), a.v))
        if k == 0:
            return V("i", a.v)
        if fid == FN_INT:
            return V("i", a.v >> k if binary else a.v // 10 ** k)
        m = abs(a.v) >> k if binary else abs(a.v) // 10 ** k
        return V("i", -m if a.v < 0 else m)
    # MOD: a - b * floor(a / b), the divisor's sign
    b = as_num(args[1])
    if b.kind == "e":
        return b
    if b.v == 0:
        return err(E_DIV0)
    if a.kind == "i" and b.kind == "i":
        sb = same_base(a, b)
        if sb is not None:
            binary, ka, kb = sb
            kk = max(ka, kb)
            va, vb = scale_up(a.v, ka, kk, binary), scale_up(b.v, kb, kk, binary)
            if va is None or vb is None:
                return err(E_NUM)
            return V("i", va % vb, kk | (0x80 if binary and kk else 0))
    x, y = to_f64(a), to_f64(b)
    q = ftz(x / y) if y != 0 else float("nan")
    if math.isinf(q) or math.isnan(q):
        return V("f", float("nan"))
    t = ftz(y * zsign(math.floor(q), q))
    return V("f", ftz(x - t))


def f_round_away(y):
    """An F64 rounded half away from zero to an integer, exactly (a zero
    keeps y's sign)."""
    import math
    if math.isinf(y) or math.isnan(y) or abs(y) >= 2.0 ** 52:
        return y
    t = math.trunc(y)
    f = y - t                                # (exact)
    r = t + (1 if f >= 0.5 else -1 if f <= -0.5 else 0)
    return zsign(r, y)


def f_div(x, y):
    """F64 x / y with IEEE's infinities for a zero y (callers exclude 0/0)."""
    import math
    if y == 0:
        return math.copysign(float("inf"), x) * math.copysign(1.0, y)
    return ftz(x / y)


def f_powi(x, n):
    """x^n for an integer n, F64, by square and multiply from the low bit
    (a square only while bits remain), then 1 / r for n < 0."""
    r, b, m = 1.0, x, abs(n)
    while m:
        if m & 1:
            r = ftz(r * b)
        m >>= 1
        if m:
            b = ftz(b * b)
    return f_div(1.0, r) if n < 0 else r


def power(a, b):
    """a ^ b (section 5.7)."""
    a, b = as_num(a), as_num(b)
    if a.kind == "e":
        return a
    if b.kind == "e":
        return b
    if not (b.kind == "i" and b.k & 0x7F == 0):
        return err(E_NAME)                   # (a fractional exponent: the transcendentals)
    n = b.v
    zero = a.v == 0
    if zero and n == 0:
        return err(E_NUM)
    if zero and n < 0:
        return err(E_DIV0)
    if a.kind == "i" and n >= 0:
        k, binary = a.k & 0x7F, bool(a.k & 0x80)
        if k * n <= (BIN_KMAX if binary else DEC_KMAX):
            p = a.v ** n
            if not I64_MIN <= p <= I64_MAX:
                return err(E_NUM)
            return V("i", p, (k * n) | (0x80 if binary and k * n else 0))
    return V("f", f_powi(to_f64(a), n))


def f_parts(x):
    """A finite nonzero F64 as (m, E): |x| = m x 2^E, m < 2^53."""
    import struct as _s
    b = _s.unpack("<Q", _s.pack("<d", x))[0]
    e, f = b >> 52 & 0x7FF, b & ((1 << 52) - 1)
    return (1 << 52 | f), e - 1075


def round_fn(a, b):
    """ROUND(x, n): half away from zero, n an integer -18..18, exactly
    (section 5.7)."""
    import math
    a, b = as_num(a), as_num(b)
    if a.kind == "e":
        return a
    if b.kind == "e":
        return b
    if not (b.kind == "i" and b.k & 0x7F == 0) or not -18 <= b.v <= 18:
        return err(E_NUM)
    n = b.v
    k, binary = a.k & 0x7F, bool(a.k & 0x80)
    if a.kind == "i" and not (k and binary):
        if n >= k:
            return V("i", a.v, a.k)
        q = div_away(a.v, 10 ** (k - n))
        if n >= 0:
            return V("i", q, n)
        return i64(q * 10 ** -n)
    x = to_f64(a)
    if x == 0 or math.isinf(x) or math.isnan(x):
        return V("f", x)
    neg = x < 0
    m, e = f_parts(x)
    if n >= 0:
        p = m * 10 ** n                       # |x| x 10^n = p x 2^e, exactly
        if e >= 0 or (p >> -e) >= 1 << 53:
            return V("f", x)                  # (no digits to round, or within an ulp)
        sh = -e
        nn = (p + (1 << (sh - 1))) >> sh       # half away from zero (on the magnitude)
        r = f_div(float(nn), float(10 ** n))
    else:
        if abs(x) >= 2.0 ** 63:
            return V("f", x)
        whole = m << e if e >= 0 else m >> -e  # the integer part: the fraction cannot matter
        q, rem = divmod(whole, 10 ** -n)
        nn = q + (rem >= 5 * 10 ** (-n - 1))
        r = float(nn * 10 ** -n)
    return V("f", ftz(-r if neg else r))


def sqrt_fn(a):
    import math
    a = as_num(a)
    if a.kind == "e":
        return a
    x = to_f64(a)
    if x < 0:
        return err(E_NUM)
    return V("f", ftz(math.sqrt(x)))


def pct(a):
    """%: / 100, exactly for integers (two more decimals of fixed point)."""
    a = as_num(a)
    if a.kind == "e":
        return a
    k = a.k & 0x7F
    if a.kind == "i" and not (k and a.k & 0x80) and k + 2 <= DEC_KMAX:
        return V("i", a.v, k + 2)
    return V("f", ftz(to_f64(a) / 100.0))


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
                if i + 6 > len(code):
                    raise Abort(E_STACK)
                r, c = long_ref(int.from_bytes(code[i:i + 6], "big"), frow, fcol)
                i += 6
                push(sv.value(sv.find(r, c)) if r < sv.d.nrows and c < sv.d.ncols else err(E_REF))
            elif op == OP_CELLS:
                n, i = varlen(code, i)
                cells = []
                for _ in range(n):
                    r, c, i = ref_at(code, i, frow, fcol)
                    cells.append((r, c))
                push(V("l", cells))
            elif op == OP_CALL:
                fid, i = varlen(code, i)
                argc, i = varlen(code, i)
                if argc > len(st):
                    raise Abort(E_STACK)
                args = st[len(st) - argc:]
                del st[len(st) - argc:]
                push(aggregate(sv, fid, args) if fid in AGGREGATES else
                     scalar_fn(fid, args) if fid in ARITY else err(E_NAME))
            elif op in _BINOPS:
                b = pop(); a = pop()
                push(binop(_BINOPS[op], a, b))
            elif op == OP_NEG:
                push(neg(pop()))
            elif op == OP_PCT:
                push(pct(pop()))
            elif op == OP_POW:
                b = pop(); a = pop()
                push(power(a, b))
            elif op == OP_CONCAT:                           # not in this phase
                pop()
                pop()
                push(err(E_NAME))
            elif op == OP_STR:
                n, i = varlen(code, i)
                i += n
                push(err(E_NAME))
            elif op in (OP_JZ, OP_JMP):
                if i + 2 > len(code):
                    raise Abort(E_STACK)
                off = code[i] | code[i + 1] << 8
                i += 2
                if op == OP_JMP or not truth(pop()):
                    i += off
            else:
                raise Abort(E_STACK)
    except Abort as e:
        return err(e.code)


def store_form(v, fk):
    """A formula's value as stored: (result kind, 8 bytes), rescaled to the
    cell's fixed point fk (section 4)."""
    if v.kind == "z":
        v = V("i", 0)
    if v.kind == "i" and (bad_k(v.k) or bad_k(fk)):
        return K_ERROR, struct.pack("<Q", E_NUM)
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
        for g, h in iter_cells(m, b, row):
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
        self.b += bytes([OP_HOSTCALL]) + varlen_bytes(len(t)) + t + self._ref(row, col, rabs, cabs, long=True)[1:]
        return self

    def cells(self, refs):
        """refs: (row, col, rabs, cabs) each."""
        self.b += bytes([OP_CELLS]) + varlen_bytes(len(refs)) + b"".join(self._ref(*r) for r in refs)
        return self

    def call(self, fid, argc):
        self.b += bytes([OP_CALL]) + varlen_bytes(fid) + varlen_bytes(argc)
        return self

    def op(self, o):
        self.b += bytes([o])
        return self

    def jz(self):
        """JZ with its offset to fill in (patch(at))."""
        self.b += bytes([OP_JZ, 0, 0])
        return len(self.b)

    def jmp(self):
        self.b += bytes([OP_JMP, 0, 0])
        return len(self.b)

    def patch(self, at):
        """The jump ending at `at` lands here."""
        off = len(self.b) - at
        self.b[at - 2:at] = bytes([off & 0xFF, off >> 8])
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
