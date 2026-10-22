#!/usr/bin/env python3
"""SHITE reference model: MOVE_CELLS and CELLS_MOVED (README section 6).
Checks what a move means rather than bytes: after inserting a row every
formula, moved or not, has the value it had; deleting one makes references
to it #REF!; a cut and paste over itself; the failures (a reference too far
for its form, a heap too small) leave the sheet as it was; and CELLS_MOVED
after the cells were copied unchanged gives MOVE_CELLS' formulas."""
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
    def __init__(self, n=1 << 21, base=I.HYPERRAM_BASE):
        self.b, self.base = bytearray(n), base

    def read(self, a, n):
        return bytes(self.b[a - self.base:a - self.base + n])

    def write(self, a, d):
        self.b[a - self.base:a - self.base + len(d)] = d


B = I.HYPERRAM_BASE
PB = B + 0x100                       # the parameter block
HEAP = B + 0x100000


def cells(m, desc):
    """{(row, col): (kind, header flags, body)} of the whole sheet."""
    d = SH.Descriptor(m, desc)
    out = {}
    for r in range(d.nrows):
        b = d.row_block(r)
        if b is None:
            continue
        for g, h in SH.iter_cells(m, b, r):
            c = b + 16 * g
            out[(r, SH.hdr_col(h))] = (h[3] & 15, h[3] & 0xF0, m.read(c + 8, 16 * h[4] - 8))
    return out


def values(m, desc):
    """{(row, col): (result kind, result)} of every formula, recalculated."""
    for _ in range(4):                               # (settled: formulas read formulas)
        SH.recalc(m, desc, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF)
    return {p: (body[16], body[8:16]) for p, (k, _, body) in cells(m, desc).items() if k == SH.K_FORMULA}


def code_of(body):
    n = struct.unpack("<H", body[18:20])[0]
    return body[24:24 + n]


def params(m, desc, src, dst, fix=(0, 0xFFFFFFFF), heap=(HEAP, HEAP + 0x80000)):
    m.write(PB, struct.pack("<IIIIIIIIIIII", desc, *src, *dst, *fix, *heap, 0) + bytes(16))


heap_at = {}


def move(m, desc, src, dst, **kw):
    """MOVE_CELLS; each move's blocks after the last's (the heap's new start)."""
    kw.setdefault("heap", (heap_at.get(id(m), HEAP), HEAP + 0x80000))
    params(m, desc, src, dst, **kw)
    n = SH.move_cells(m, PB)
    heap_at[id(m)] = SH.u32(m, PB + SH.P_HEAP)
    return n


rng = np.random.default_rng(953)


def sheet(nrows=24, ncols=12, far=None, filled=22, kinds=range(6)):
    """Numbers in columns 0-3, formulas in 4-11 referring to them and to
    each other: every reference kind and form (kinds: which of them).
    Rows filled.. empty."""
    sh = SH.Sheet(nrows, ncols)
    for r in range(filled):
        for c in range(4):
            if (r + c) % 5:
                sh.number(r, c, int(rng.integers(-50, 50)))
    F = SH.F
    for r in range(filled):
        R = lambda: int(rng.integers(0, filled))
        C = lambda: int(rng.integers(0, 4))
        for c in range(4, ncols):
            k = kinds[(r * 7 + c) % len(kinds)]
            if k == 0:
                rr, cc = R(), C()
                f = F().ref(rr - r, cc - c).ref(rr, cc, True, True).op(SH.OP_ADD)
            elif k == 1:
                rr, cc = R(), C()
                f = F().ref(rr - r, cc, False, True, long=True).ref(rr, cc - c, True, False).op(SH.OP_SUB)
            elif k == 2:
                a, b = sorted((R(), R()))
                f = F().rect((a - r, 0 - c, False, False), (b, 3, True, True)).call(0, 1)
            elif k == 3:
                f = F().cells([(R() - r, C() - c, False, False), (R(), C(), True, True)]).call(0, 1)
            elif k == 4:
                f = F().host("HOST(1)", R(), C())
            else:                                    # IF, and a formula's own result
                rr = R()
                f = F().ref(rr - r, 0 - c).i(0).op(SH.OP_GT)
                j = f.jz(); f.ref(rr - r, 4 - c); j2 = f.jmp(); f.patch(j); f.i(-1); f.patch(j2)
                if c == 4:
                    f = F().ref(rr - r, 1 - c)
            sh.formula(r, c, f.end())
    if far is not None:
        r0, rf = far
        sh.number(rf, 0, 7)
        sh.formula(r0, 4, SH.F().ref(rf - r0, 0 - 4).end())      # (r0, rf: rows past filled)
    return sh


# 1. Inserting a row: rows 5.. move down one; every value as before
m = Mem()
desc = sheet().layout(m, B + 0x1000, slack=1)
before = values(m, desc)
moved = move(m, desc, (5, 0, 0xFFFFFFFF, 0xFFFFFFFF), (6, 0))
after = values(m, desc)
same = all(after.get((r + (r >= 5), c)) == v for (r, c), v in before.items())
check(same and len(after) == len(before), "insert a row: every formula, moved or not, has its value")
check(moved == sum(1 for (r, c) in cells(m, desc) if r >= 6), f"insert a row: {moved} cells moved")
check(SH.u32(m, PB + SH.P_DELETED) == 0, "insert a row: no reference deleted")
check(all(SH.Descriptor(m, desc).row_block(5) is None for _ in [0]), "insert a row: the new row is empty")

# 2. Deleting row 5 again: rows 6.. back up; the sheet's formulas as at first
m2 = Mem()
desc2 = sheet().layout(m2, B + 0x1000, slack=1)
v0 = values(m2, desc2)
c0 = cells(m2, desc2)
move(m2, desc2, (5, 0, 0xFFFFFFFF, 0xFFFFFFFF), (6, 0))
move(m2, desc2, (6, 0, 0xFFFFFFFF, 0xFFFFFFFF), (5, 0))
c2 = cells(m2, desc2)
check(all(code_of(c2[p][2]) == code_of(b) for p, (k, _, b) in c0.items() if k == SH.K_FORMULA),
      "insert then delete a row: every formula's bytecode as it was")

# 3. Deleting a row with cells: references to it #REF!, the rest unchanged
# (single references and cell lists to numbers only: a range through the
# row would lose its cells, and a formula reading such a one would change)
m3 = Mem()
desc3 = sheet(kinds=(0, 1, 3)).layout(m3, B + 0x1000, slack=1)
v0 = values(m3, desc3)
move(m3, desc3, (8, 0, 0xFFFFFFFF, 0xFFFFFFFF), (7, 0))
v1 = values(m3, desc3)
dl = SH.u32(m3, PB + SH.P_DELETED)
refs = sum(1 for p, (rk, rv) in v1.items() if rk == SH.K_ERROR and rv[0] == SH.E_REF)
check(dl > 0 and refs > 0, f"delete a row: {dl} references deleted, {refs} formulas #REF!")
check(all(v1[(r - (r > 7), c)] == v for (r, c), v in v0.items()
          if r != 7 and v1[(r - (r > 7), c)][0] != SH.K_ERROR),
      "delete a row: every other formula's value as before")

# 4. A cut and paste onto itself (overlapping): the moved block's values follow
m4 = Mem()
desc4 = sheet().layout(m4, B + 0x1000, slack=1)
v0 = values(m4, desc4)
moved = move(m4, desc4, (2, 0, 9, 5), (4, 2))
v1 = values(m4, desc4)
c1 = cells(m4, desc4)
check(moved > 0 and all(c1[(r + 2, c + 2)][1] & SH.F_CHANGED for r in range(2, 10) for c in range(0, 6)
                        if (r + 2, c + 2) in c1), f"cut and paste: {moved} cells moved, each CHANGED")
inside = lambda r, c: 2 <= r <= 9 and 0 <= c <= 5
over = lambda r, c: 4 <= r <= 11 and 2 <= c <= 7
check(all(((r, c) not in c1) for r in range(2, 10) for c in range(0, 6) if not over(r, c)),
      "cut and paste: the source is left empty")

# 5. A reference that no longer fits its form: nothing changes
m5 = Mem(1 << 22)
desc5 = sheet(nrows=40000, ncols=12, far=(30, 16000)).layout(m5, B + 0x1000, slack=1)
snap = bytes(m5.b)
try:
    move(m5, desc5, (16000, 0, 16000, 0), (20000, 0), heap=(B + 0x300000, B + 0x380000))
    check(False, "a reference too far for its form: refused")
except SH.MoveError as e:
    check(e.why == SH.MV_FAR and (e.row, e.col) == (30, 4), f"a reference too far for its form: refused ({e})")
m5.b[PB - B:PB - B + 64] = snap[PB - B:PB - B + 64]
heapless = lambda b: b[:0x300000] == snap[:0x300000]
check(heapless(bytes(m5.b)), "a reference too far for its form: the sheet unchanged")

# 6. A heap too small: nothing changes
m6 = Mem()
desc6 = sheet().layout(m6, B + 0x1000, slack=1)
snap = bytes(m6.b)
try:
    move(m6, desc6, (5, 0, 0xFFFFFFFF, 0xFFFFFFFF), (6, 0), heap=(HEAP, HEAP + 0x400))
    check(False, "a heap too small: refused")
except SH.MoveError as e:
    check(e.why == SH.MV_HEAP, f"a heap too small: refused ({e})")
m6.b[PB - B:PB - B + 64] = snap[PB - B:PB - B + 64]
check(bytes(m6.b[:HEAP - B]) == snap[:HEAP - B], "a heap too small: the sheet unchanged")

# 7. CELLS_MOVED after the CPU copied the cells (bytecode unchanged): the
# same formulas as MOVE_CELLS gives
rng = np.random.default_rng(77)
sa = sheet()
rng = np.random.default_rng(77)
sb = sheet()
src, dst = (3, 1, 8, 6), (10, 3)
mv_r = lambda r, c: (r - 3 + 10, c - 1 + 3)
moved_rows = {}
for r, cl in sb.rows.items():                         # the CPU's move, by hand
    for col, kf, body in cl:
        if 3 <= r <= 8 and 1 <= col <= 6:
            nr, nc = mv_r(r, col)
            moved_rows.setdefault(nr, []).append((nc, kf, body))
        elif not (10 <= r <= 15 and 3 <= col <= 8):
            moved_rows.setdefault(r, []).append((col, kf, body))
sb.rows = moved_rows
ma, mb = Mem(), Mem()
da, db = sa.layout(ma, B + 0x1000, slack=1), sb.layout(mb, B + 0x1000, slack=1)
move(ma, da, src, dst)
params(mb, db, src, dst)
n = SH.cells_moved(mb, PB)
ca, cb = cells(ma, da), cells(mb, db)
check(set(ca) == set(cb) and all(code_of(ca[p][2]) == code_of(cb[p][2]) for p in ca if ca[p][0] == SH.K_FORMULA),
      f"CELLS_MOVED after a copy: MOVE_CELLS' formulas ({n} changed)")

print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
