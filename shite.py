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

# descriptor fields
D_ROWTAB, D_NROWS, D_NCOLS, D_CL, D_CLCAP, D_PREVCL = 4, 8, 10, 12, 16, 20
D_ROUND, D_CHANGED, D_EVALS, D_ERRORS, D_FLAGS, D_BADROW = 24, 28, 32, 36, 40, 42
# row block header fields
R_ROW, R_LEN, R_NCELLS, R_ROOT, R_FREE, R_CHANGED, R_FLAGS = 0, 2, 4, 6, 8, 10, 12
ROW_STALE = 1


class ShiteFormatError(Exception):
    """A malformed descriptor or row block: SSNAIL ends the job with
    ERR_FORMAT, the row in the descriptor (0xFFFF: the descriptor itself)."""
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


class Descriptor:
    def __init__(self, m, a):
        if m.read(a, 4) != MAGIC:
            raise ShiteFormatError(0xFFFF, "no sheet descriptor")
        self.m, self.a = m, a
        self.rowtab = u32(m, a + D_ROWTAB)
        self.nrows = u16(m, a + D_NROWS)
        self.ncols = u16(m, a + D_NCOLS)

    def row_block(self, row):
        """The row's block address, or None (an empty row, or out of range)."""
        if row >= self.nrows:
            return None
        b = u32(self.m, self.rowtab + 4 * row)
        return None if b == EMPTY_ROW else b


def cells_in_block(m, b, row):
    """(granule offset, header bytes) of each cell but FREE ones, in memory
    order, checking the block's structure."""
    if u16(m, b + R_ROW) != row:
        raise ShiteFormatError(row, "row number does not match the row table")
    blen, free = u16(m, b + R_LEN), u16(m, b + R_FREE)
    if free == 0 or free > blen:
        raise ShiteFormatError(row, "first free granule outside the block")
    out, g = [], 1
    while g < free:
        h = m.read(b + GRANULE * g, 8)
        n = h[3]
        if n == 0 or g + n > free:
            raise ShiteFormatError(row, f"cell at granule {g}: bad length {n}")
        if h[2] & 0x0F != K_FREE:
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
        cc = u16(m, c)
        if cc == col:
            return c
        g = u16(m, c + (4 if col < cc else 6))
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
        cells = sorted(cells_in_block(m, b, row), key=lambda gh: struct.unpack("<H", gh[1][:2])[0])
        cols = [struct.unpack("<H", h[:2])[0] for _, h in cells]
        if any(cols[i] == cols[i + 1] for i in range(len(cols) - 1)):
            raise ShiteFormatError(row, "two cells in one column")
        root, left, right = balanced_links(len(cells))
        gr = lambda i: 0 if i < 0 else cells[i][0]
        for i, (g, _) in enumerate(cells):
            put16(m, b + GRANULE * g + 4, gr(left[i]))
            put16(m, b + GRANULE * g + 6, gr(right[i]))
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
        cl = a; a += 16 * granules(4 * change_cap)
        prev = a; a += 16 * granules(4 * change_cap)
        rowtab = a; a += 16 * granules(4 * self.nrows)
        m.write(desc, MAGIC + struct.pack("<IHHIHHI", rowtab, self.nrows, self.ncols,
                                          cl, change_cap, 0, prev) + bytes(40))
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
                m.write(b + GRANULE * g, struct.pack("<HBBHH", c[0], c[1], c[2], 0, 0) + c[3])
                g += c[2]
            m.write(b, struct.pack("<HHHHHHHH", r, blen, 0, 0, free, 0, ROW_STALE, 0))
        if balanced:
            rebalance_row(m, desc, 0, self.nrows - 1)
        return desc
