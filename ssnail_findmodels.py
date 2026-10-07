#!/usr/bin/env python3
"""Find GGUF models on Hugging Face that SSNAIL can run.

    ssnail_findmodels.py                      # default searches for small models
    ssnail_findmodels.py tinystories minilm   # your own search terms
    ssnail_findmodels.py --mem 8 --limit 40

For each candidate repository it reports the architecture, parameter count,
which memory sizes it should fit (and at which weight format), the best file
to download (unquantised if available, since the converter quantises better
from F16/F32 than from an already-quantised file), and the command to get it.

Fit is an estimate from the parameter count (weights only, plus a margin for
activations, KV cache and the script).  The converter has the final word:
`make foo.all` builds every size that really fits.

Needs network access to huggingface.co; uses only the standard library.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api"
SUPPORTED = ("llama", "gpt2", "bert")
DEFAULT_QUERIES = ["tinystories", "smollm", "minilm", "bert-tiny", "bert-mini",
                   "tinyllama", "stories15m", "stories260k", "gpt2"]

# Bits per weight by quantisation tag in a filename, for estimating parameter
# counts when the API does not report them.
BPW = [("f32", 32), ("fp32", 32), ("bf16", 16), ("f16", 16), ("fp16", 16),
       ("q8_0", 8.5), ("q6_k", 6.56), ("q5_k", 5.5), ("q5_1", 6), ("q5_0", 5.5),
       ("q4_k", 4.5), ("q4_1", 5), ("q4_0", 4.5), ("iq4", 4.5), ("q3_k", 3.44),
       ("iq3", 3.4), ("q2_k", 2.63), ("iq2", 2.3), ("iq1", 1.8), ("tq1_0", 1.69),
       ("tq2_0", 2.06)]
FORMATS = [("q8_0", 8.5), ("q4_0", 4.5)]     # what the converter can produce
MEM_SIZES = (8, 64, 72)
MARGIN = 0.80        # weights may use this fraction of memory


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ssnail-findmodels/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(query, limit):
    q = urllib.parse.urlencode([("search", query), ("filter", "gguf"),
                                ("sort", "downloads"), ("direction", "-1"),
                                ("limit", str(limit)), ("expand[]", "gguf"),
                                ("expand[]", "downloads"), ("expand[]", "likes")])
    return get_json(f"{API}/models?{q}")


def gguf_files(repo):
    """[(path, size)] of .gguf files in a repository."""
    out = []
    for e in get_json(f"{API}/models/{repo}/tree/main?recursive=true"):
        if e.get("type") == "file" and e.get("path", "").lower().endswith(".gguf"):
            out.append((e["path"], e.get("size") or (e.get("lfs") or {}).get("size", 0)))
    return out


def file_bpw(name):
    n = name.lower()
    for tag, bpw in BPW:
        if re.search(r"(^|[.\-_])" + re.escape(tag), n):
            return bpw
    return None


def guess_arch(repo, files):
    """Architecture from the repository/file names when the API omits it."""
    s = (repo + " " + " ".join(f for f, _ in files)).lower()
    if "bert" in s or "minilm" in s or "mpnet" in s or "e5-" in s or "bge" in s:
        return "bert"
    if "gpt2" in s or "gpt-2" in s or "tinystories-gpt" in s:
        return "gpt2"
    if any(k in s for k in ("llama", "smollm", "stories", "tinyllama", "mistral", "qwen")):
        return "llama?"
    return None


def assess(model, files):
    gg = model.get("gguf") or {}
    arch = gg.get("architecture") or guess_arch(model["id"], files)
    params = gg.get("total")
    best = None
    if files:
        # Prefer unquantised, then the highest-precision quantised file
        ranked = sorted(files, key=lambda f: -(file_bpw(f[0]) or 0))
        best = ranked[0]
        if not params:
            for f, size in ranked:
                bpw = file_bpw(f)
                if bpw and size:
                    params = int(size * 8 / bpw)
                    break
    fits = {}
    if params:
        for mb in MEM_SIZES:
            budget = mb * 1048576 * MARGIN - (64 * 1024 if mb > 8 else 0)
            fits[mb] = next((fmt for fmt, bpw in FORMATS if params * bpw / 8 <= budget), None)
    return arch, params, best, fits


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="*", help=f"search terms (default: {' '.join(DEFAULT_QUERIES)})")
    ap.add_argument("--mem", type=int, choices=MEM_SIZES,
                    help="only show models estimated to fit this memory size")
    ap.add_argument("--limit", type=int, default=15, help="repositories per search term")
    ap.add_argument("--all-arch", action="store_true",
                    help="also list architectures SSNAIL can't run yet")
    args = ap.parse_args()

    seen, rows = set(), []
    for q in args.queries or DEFAULT_QUERIES:
        try:
            models = search(q, args.limit)
        except (urllib.error.URLError, TimeoutError) as e:
            sys.exit(f"Can't reach Hugging Face ({e}).")
        for model in models:
            repo = model["id"]
            if repo in seen:
                continue
            seen.add(repo)
            try:
                files = gguf_files(repo)
            except (urllib.error.URLError, TimeoutError, ValueError):
                continue
            arch, params, best, fits = assess(model, files)
            ok_arch = arch and arch.rstrip("?") in SUPPORTED
            if not (ok_arch or args.all_arch) or not best:
                continue
            if args.mem and not fits.get(args.mem):
                continue
            if not any(fits.values()) and not args.all_arch:
                continue
            rows.append((model.get("downloads", 0), repo, arch, params, best, fits))
        print(f"searched '{q}': {len(rows)} candidates so far", file=sys.stderr)

    rows.sort(key=lambda r: (r[3] or 1 << 62, -r[0]))
    if not rows:
        print("No suitable models found.  Try other search terms, or --all-arch.")
        return
    print(f"\n{'repository':48s} {'arch':7s} {'params':>7s} {'8MB':>5s} {'64MB':>5s} "
          f"{'72MB':>5s} {'downloads':>9s}")
    for dl, repo, arch, params, best, fits in rows:
        f = lambda mb: (fits.get(mb) or "-").replace("_0", "")
        print(f"{repo[:48]:48s} {arch or '?':7s} {human(params) if params else '?':>7s} "
              f"{f(8):>5s} {f(64):>5s} {f(72):>5s} {dl:>9d}")
    print("\nFit columns: weight format the converter would likely use (q8 / q4), or - if "
          "it won't fit.\n'llama?' = architecture guessed from the name; check with make foo.info.\n")
    print("To fetch the best file of each (unquantised where available):")
    for dl, repo, arch, params, best, fits in rows:
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(best[0])}"
        local = best[0].split("/")[-1]
        print(f"  make fetch URL={url}" + (f"   # {local}" if local else ""))


if __name__ == "__main__":
    main()
