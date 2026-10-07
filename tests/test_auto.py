#!/usr/bin/env python3
"""'auto' weight-format choice, the 'mixed' step, the question before storing
weights below the source's precision, and the too-big message."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
from gguf import GGMLQuantizationType as Q  # noqa: E402
import ssnail_convert as C  # noqa: E402
import test_gpt2_bert as TG  # noqa: E402


def run(args):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "ssnail_convert.py")] + args,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


ok = True
with tempfile.TemporaryDirectory() as d:
    f16 = os.path.join(d, "fp.gguf")
    q8 = os.path.join(d, "q8.gguf")
    # Big vocabulary so the embedding dominates, as in real small models
    TG.make_gpt2(f16, Q.F16, dim=256, hidden=1024)
    TG.make_gpt2(q8, Q.Q8_0, dim=256, hidden=1024)
    out = os.path.join(d, "o.ssnail")
    sizes = {}
    for pol in ("keep", "q8_0", "mixed", "q4_0"):
        b = C.build(C.Model(f16), 32, pol, C.FMT_Q8 if hasattr(C, "FMT_Q8") else 8, 8)
        sizes[pol] = len(b.image)
    order_ok = sizes["keep"] > sizes["q8_0"] > sizes["mixed"] > sizes["q4_0"]
    ok &= order_ok
    print(f"{'PASS' if order_ok else 'FAIL'}  sizes keep > q8_0 > mixed > q4_0: {sizes}")

    def mb_for(pol):        # a memory size where 'pol' is the first that fits
        return sizes[pol] / 1048576 + 0.6

    # From an F16 source, quantising is normal: no question asked
    rc, txt = run(["convert", f16, "-o", out, "--wtype", "auto", "--ctx", "32",
                   "--mem-mb", "1.5"])
    t1 = rc == 0 and "auto: using --wtype" in txt and "WARNING" not in txt
    ok &= t1
    print(f"{'PASS' if t1 else 'FAIL'}  F16 source: auto quantises without asking "
          f"({[l for l in txt.splitlines() if l.startswith('auto:')]})")

    # From a Q8_0 source, going lower needs consent: non-interactive refuses
    rc, txt = run(["convert", q8, "-o", out, "--wtype", "auto", "--ctx", "32",
                   "--mem-mb", "1.5"])
    t2 = rc != 0 and "WARNING" in txt and "WTYPE=" in txt and "Not converted" in txt
    ok &= t2
    print(f"{'PASS' if t2 else 'FAIL'}  Q8_0 source: auto asks before going lower, "
          f"and says how to accept it")

    # ... but an explicit format is consent
    rc, txt = run(["convert", q8, "-o", out, "--wtype", "q4_0", "--ctx", "32", "--mem-mb", "1.5"])
    t3 = rc == 0 and "WARNING" not in txt
    ok &= t3
    print(f"{'PASS' if t3 else 'FAIL'}  explicit --wtype q4_0 converts without asking")

    # Too big: the message names the larger sizes and how to ask for them
    rc, txt = run(["convert", f16, "-o", out, "--wtype", "auto", "--ctx", "32",
                   "--mem-mb", "0"])
    t4 = rc != 0 and "fp.72mb.chat" in txt and "ATTICRAM=72" in txt and "CONTEXT_WINDOW" in txt
    ok &= t4
    print(f"{'PASS' if t4 else 'FAIL'}  too-big message suggests larger sizes:\n"
          + "\n".join("      " + l for l in txt.strip().splitlines()))
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
