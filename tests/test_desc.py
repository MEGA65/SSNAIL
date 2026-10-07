"""The header description: "name (arch) [type]", type from --type."""
import os
import subprocess
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gguf import GGMLQuantizationType as Q
import ssnail_convert as C
import ssnail_sim as S
import test_pipeline as TP

ok = True
d = tempfile.mkdtemp()
g = os.path.join(d, "m.gguf")
TP.make_model(g, Q.Q8_0)
model = C.Model(g)
# nameless model: "arch dims [type]"
for kw, want in (({}, "llama 256d 2L 8H [General]"), ({"model_type": "Story"}, "llama 256d 2L 8H [Story]")):
    m = S.Machine(C.convert(model, 32, "keep", 8, 8, verbose=False, **kw))
    desc = S.description(m)
    good = desc == want
    ok = ok and good
    print(f"{'PASS' if good else 'FAIL'}  description {desc!r}")
# named model: "name (arch) [type]"
orig = model.field
model.field = lambda k, dflt=None: "TinyTest" if k == "general.name" else orig(k, dflt)
m = S.Machine(C.convert(model, 32, "keep", 8, 8, verbose=False, model_type="Story"))
good = S.description(m) == "TinyTest (llama) [Story]"
ok = ok and good
print(f"{'PASS' if good else 'FAIL'}  description {S.description(m)!r}")
model.field = orig
# the command line: --type is optional, and recorded when given
out = os.path.join(d, "x.ssnail")
here = os.path.join(os.path.dirname(__file__), "..")
for extra, want in (([], "[General]"), (["--type", "Code"], "[Code]")):
    r = subprocess.run([sys.executable, os.path.join(here, "ssnail_convert.py"), "convert", g,
                        "-o", out] + extra, capture_output=True, text=True)
    m = S.Machine(open(out, "rb").read()) if r.returncode == 0 else None
    good = m is not None and S.description(m).endswith(want)
    ok = ok and good
    print(f"{'PASS' if good else 'FAIL'}  convert {' '.join(extra) or '(no --type)'} -> "
          f"{S.description(m) if m else r.stderr.strip()[-80:]!r}")
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
