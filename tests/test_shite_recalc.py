#!/usr/bin/env python3
"""SHITE reference model: RECALC (values, fixed point, F64, aggregates,
errors, host calls, rounds, change lists)."""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import shite as SH
from shite import F
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


def result(m, desc, r, c):
    """(result kind, value) of a formula cell."""
    a = SH.find_cell(m, desc, r, c)
    rk = m.read(a + 24, 1)[0]
    raw = m.read(a + 16, 8)
    if rk == SH.K_F64:
        return "f", struct.unpack("<d", raw)[0]
    if rk == SH.K_I64:
        return "i", struct.unpack("<q", raw)[0]
    return {SH.K_ERROR: "e"}.get(rk, rk), raw[0]


DEC2 = bytes([1, 2, 2, 0, 0, 0, 0, 0])        # fixed, 2 decimals, k = 2
m = Mem()
sh = SH.Sheet(20, 40)
# row 0: numbers; row 1: text, an error, an uncompiled formula
sh.number(0, 0, 10); sh.number(0, 1, 32); sh.number(0, 2, 0.5)
sh.number(0, 3, 1999, fmt=DEC2)             # 19.99
sh.number(0, 4, 3)
sh.text(1, 0, "hello")
sh.add(1, 1, SH.K_ERROR, struct.pack("<Q", SH.E_NA))
sh.add(1, 2, SH.K_UNCOMPILED, bytes(8) + struct.pack("<BBHHH", 0, 0, 0, 3, 0) + b"=A1")
sh.number(1, 3, 1e300)
# row 2: formulas (relative references: row offset, column offset)
sh.formula(2, 0, F().ref(-2, 0).ref(-2, 1).op(SH.OP_ADD).end())                  # =A1+B1 -> 42
sh.formula(2, 1, F().ref(0, -1).i(2).op(SH.OP_MUL).end())                        # =A3*2 -> 84
sh.formula(2, 2, F().ref(-2, -2).ref(-2, -2).op(SH.OP_ADD).ref(-2, 2, cabs=True).op(SH.OP_MUL).end())  # (A1+A1)*$C1 -> 10.0
sh.formula(2, 3, F().ref(-2, 0).ref(-2, 1).op(SH.OP_MUL).end(), fmt=DEC2)         # D1*E1 = 59.97, k 2
sh.formula(2, 4, F().ref(0, 0, True, True).i(7).op(SH.OP_DIV).end())              # =$A$1/7 -> F64
sh.formula(2, 5, F().f(0.1).f(0.2).op(SH.OP_ADD).end())
sh.formula(2, 6, F().i(1).i(0).op(SH.OP_DIV).end())                               # #DIV/0!
sh.formula(2, 7, F().ref(25, 0).end())                                            # off the sheet: #REF!
sh.formula(2, 8, F().ref(-1, -6).end())                                           # C2 uncompiled: #COMPILE!
sh.formula(2, 9, F().i(2).i(3).op(SH.OP_POW).end())                               # not in this phase: #NAME?
sh.formula(2, 10, F().rect((0, 0, True, True), (1, 4, True, True)).call(SH.FN_SUM, 1).end())  # error in range: #N/A
sh.formula(2, 11, F().rect((0, 0, True, True), (0, 4, True, True)).call(SH.FN_SUM, 1).end())  # 10+32+0.5+19.99+3
sh.formula(2, 12, F().rect((0, 0, True, True), (0, 4, True, True)).call(SH.FN_COUNT, 1).end())
sh.formula(2, 13, F().rect((0, 0, True, True), (0, 4, True, True)).call(SH.FN_MIN, 1).end())
sh.formula(2, 14, F().rect((0, 0, True, True), (0, 4, True, True)).call(SH.FN_MAX, 1).end())
sh.formula(2, 15, F().ref(-2, -15).ref(-2, -14).i(5).call(SH.FN_AVERAGE, 3).end())  # (10+32+5)/3
sh.formula(2, 16, F().ref(-1, -16).i(1).call(SH.FN_SUM, 2).end())                 # text from a reference skipped: 1
sh.formula(2, 17, F().ref(-1, -14, long=True).f(1e10).op(SH.OP_MUL).end())        # 1e310: #NUM!
sh.formula(2, 18, F().host("VLOOKUP(A1,B:C,2)", 19, 39).end())                    # a host call's result cell
sh.formula(2, 19, F().i(1).i(2).op(SH.OP_LT).ref(0, -19).ref(0, -18).op(SH.OP_GT).op(SH.OP_ADD).end())  # 1 + 0
sh.number(19, 39, 123)
# row 3: forward references (need a second round), row 4: an oscillation
sh.formula(3, 0, F().ref(2, 0).i(1).op(SH.OP_ADD).end())                          # =A6+1
sh.formula(5, 0, F().ref(-3, 1).i(1).op(SH.OP_ADD).end())                         # =B3+1 = 85
sh.formula(4, 0, F().ref(0, 1).i(1).op(SH.OP_SUB).op(SH.OP_NEG).end())            # A5 = -(B5-1)
sh.formula(4, 1, F().ref(0, -1).end())                                            # B5 = A5
desc = sh.layout(m, I.HYPERRAM_BASE + 0x1000, change_cap=64)

ALL = 0xFFFFFFFF
n1 = SH.recalc(m, desc, 0, ALL, 0, ALL)
R = lambda r, c: result(m, desc, r, c)
check(R(2, 0) == ("i", 42) and R(2, 1) == ("i", 84) and R(2, 2) == ("f", 10.0), "integers, relative and $ references, mixed F64")
check(R(2, 3) == ("i", 5997), "fixed point: 19.99 x 3 = 59.97 (I64 5997 at k = 2)")
check(R(2, 4) == ("f", 10 / 7) and R(2, 5) == ("f", 0.1 + 0.2), "division gives F64; IEEE double")
check(R(2, 6) == ("e", SH.E_DIV0) and R(2, 7) == ("e", SH.E_REF) and R(2, 8) == ("e", SH.E_COMPILE)
      and R(2, 9) == ("e", SH.E_NAME) and R(2, 10) == ("e", SH.E_NA) and R(2, 17) == ("e", SH.E_NUM),
      "errors: #DIV/0!, #REF!, #COMPILE!, #NAME?, #N/A from a range, #NUM! on overflow")
check(R(2, 11) == ("f", ((10 + 32) + 0.5) + 19.99 + 3) and R(2, 12) == ("i", 5)
      and R(2, 13) == ("f", 0.5) and R(2, 14) == ("i", 32), "SUM (a left fold, F64 from the first F64), COUNT, MIN, MAX")
check(R(2, 15) == ("f", 47 / 3) and R(2, 16) == ("i", 1), "AVERAGE of references and a value; text from a reference skipped")
check(R(2, 18) == ("i", 123) and R(2, 19) == ("i", 1), "a host call reads its result cell; comparisons")
check(R(5, 0) == ("i", 85) and R(3, 0) == ("i", 1), "a forward reference sees the old value in round 1")
d = SH.Descriptor(m, desc)
chg = SH.u32(m, desc + SH.D_CHANGED)
check(n1 == chg and SH.u32(m, desc + SH.D_ROUND) == 1 and SH.u32(m, desc + SH.D_EVALS) == 24,
      f"round 1: {chg} changed, counters")
SH.recalc(m, desc, 0, ALL, 0, ALL)
check(R(3, 0) == ("i", 86), "round 2: the forward reference settles")
rounds = []
for k in range(6):
    n = SH.recalc(m, desc, 0, ALL, 0, ALL)
    rounds.append((n, m.read(desc + SH.D_FLAGS, 1)[0] & 2))
check(all(n == 2 for n, _ in rounds) and all(f for _, f in rounds[1:]),
      "the oscillating pair keeps changing, and its change set repeats (flag)")
cl = SH.u32(m, desc + SH.D_PREVCL)
entries = [struct.unpack("<II", m.read(cl + 8 * i, 8)) for i in range(2)]
check(sorted(entries) == [(4, 0), (4, 1)], "the change list holds the cycle's cells")
a = SH.find_cell(m, desc, 4, 0)
check(m.read(a + 3, 1)[0] & SH.F_CHANGED, "CHANGED set on a changed cell")
# a region: only row 2's columns 0-1, both as one round
m2 = Mem(); desc2 = sh.layout(m2, I.HYPERRAM_BASE + 0x1000, change_cap=64)
n = SH.recalc(m2, desc2, 2, 2, 0, 1)
check(n == 2 and result(m2, desc2, 2, 2)[1] == 0, "RECALC of a region evaluates only that region")
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
