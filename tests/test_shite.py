#!/usr/bin/env python3
"""SHITE reference model: sheet layout, REBALANCE_ROW, FIND_CELL (also as
SSNAIL instructions through the emulator)."""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import shite as SH
import ssnail_isa as I

ok = True


def check(c, what):
    global ok
    ok &= bool(c)
    print(f"{'PASS' if c else 'FAIL'}  {what}")


class Mem:
    def __init__(self, n=1 << 20, base=I.HYPERRAM_BASE):
        self.b, self.base = bytearray(n), base

    def read(self, a, n):
        return bytes(self.b[a - self.base:a - self.base + n])

    def write(self, a, d):
        self.b[a - self.base:a - self.base + len(d)] = d


def tree(m, b):
    """(in-order columns, height) of a row block's tree."""
    def walk(g):
        if not g:
            return [], 0
        c = b + 16 * g
        l, hl = walk(SH.u16(m, c + 4))
        r, hr = walk(SH.u16(m, c + 6))
        return l + [SH.u16(m, c)] + r, 1 + max(hl, hr)
    return walk(SH.u16(m, b + SH.R_ROOT))


rng = np.random.default_rng(5)
m = Mem()
sh = SH.Sheet(40, 300)
want = {}
for r in range(40):
    if r % 7 == 3:
        continue                                   # some empty rows
    n = int(rng.integers(0, 60))
    cols = rng.choice(300, size=n, replace=False)  # any order
    for i, c in enumerate(cols):
        c = int(c)
        if i % 9 == 4:
            sh.free(r, int(rng.integers(1, 4)))    # holes
        if i % 3 == 0:
            sh.number(r, c, float(rng.standard_normal()))
        elif i % 3 == 1:
            sh.number(r, c, int(rng.integers(-1000, 1000)), fmt=bytes([1, 2, 0, 1, 0, 0, 0, 0]))
        else:
            sh.text(r, c, "x" * int(rng.integers(0, 40)))
        want[(r, c)] = True
desc = sh.layout(m, I.HYPERRAM_BASE + 0x1000, balanced=False)
d = SH.Descriptor(m, desc)
check(all(SH.find_cell(m, desc, r, c) == 0 for (r, c) in list(want)[:20]),
      "before REBALANCE (stale rows, no links) nothing is found")
SH.rebalance_row(m, desc, 0, 39)
good = True
for r in range(40):
    b = d.row_block(r)
    if b is None:
        continue
    cols, h = tree(m, b)
    n = SH.u16(m, b + SH.R_NCELLS)
    good &= cols == sorted(cols) and n == len(cols) and h == (n.bit_length() if n else 0)
    good &= SH.u16(m, b + SH.R_FLAGS) & SH.ROW_STALE == 0
check(good, "REBALANCE_ROW: every row's tree in column order, all cells, minimal height, not stale")
found = all(SH.u16(m, SH.find_cell(m, desc, r, c)) == c for (r, c) in want)
miss = all(SH.find_cell(m, desc, r, c) == 0 for r in range(40) for c in (300, 301, 65535)
           if True) and all(SH.find_cell(m, desc, r, c) == 0
                            for r in range(40) for c in range(0, 300, 7) if (r, c) not in want)
check(found and miss, f"FIND_CELL finds all {len(want)} cells and nothing else")
check(SH.find_cell(m, desc, 40, 0) == 0 and SH.find_cell(m, desc, 9999, 1) == 0, "FIND_CELL beyond the rows: 0")

# the same through SSNAIL instructions
import ssnail_sim as S
from ssnail_convert import Asm
mach = S.Machine.__new__(S.Machine)
mach.mem = bytearray(1 << 20)
mach.R, mach.A, mach.N, mach.pc = [0] * 16, [0] * 16, [0] * 3, 0
mach.stats = dict(instructions=0, bytes_read=0, bytes_written=0)
mach.on_argmax = mach.on_sync = None
mach.trace = False
mach.hw = True
m2 = Mem()
sh.layout(m2, I.HYPERRAM_BASE + 0x1000, balanced=False)
mach.mem[:len(m2.b)] = m2.b
(r0, c0) = sorted(want)[5]
A = Asm()
A(I.LI, 1, z=0); A(I.LI, 2, z=39); A(I.REBALANCE, 1, 2, x=desc)
A(I.LI, 3, z=r0); A(I.LI, 4, z=c0); A(I.FINDCELL, 3, 4, x=desc)
A(I.HALT)
code = A.assemble(I.HYPERRAM_BASE)
mach.mem[:len(code)] = code
mach.run(I.HYPERRAM_BASE)
check(bytes(mach.mem[0x1000:0x80000]) == bytes(m.b[0x1000:0x80000]) and mach.R[3] == SH.find_cell(m, desc, r0, c0) != 0,
      "as SSNAIL instructions: the same memory, FIND_CELL's address in R[a]")

# malformed rows
m3 = Mem()
s3 = SH.Sheet(4, 10)
s3.number(2, 5, 1.0); s3.number(2, 5, 2.0)
d3 = s3.layout(m3, I.HYPERRAM_BASE + 0x1000, balanced=False)
try:
    SH.rebalance_row(m3, d3, 0, 3); bad = None
except SH.ShiteFormatError as e:
    bad = e.row
check(bad == 2, "two cells in one column: a format error, row 2")
b3 = SH.Descriptor(m3, d3).row_block(2)
m3.write(b3 + 16 + 3, bytes([0]))
try:
    SH.rebalance_row(m3, d3, 0, 3); bad = None
except SH.ShiteFormatError as e:
    bad = e.row
check(bad == 2, "a zero cell length: a format error")
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
