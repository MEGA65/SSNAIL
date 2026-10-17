#!/usr/bin/env python3
"""Find GGUF models on Hugging Face that SSNAIL can run.

    ssnail_findmodels.py                      # default searches for small models
    ssnail_findmodels.py tinystories minilm   # your own search terms
    ssnail_findmodels.py --mem 8 --limit 40
    ssnail_findmodels.py --urls models/models.txt   # also write the download list
    ssnail_findmodels.py --fetch models/models.txt --dir models   # get what's missing

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
SUPPORTED = ("llama", "gpt2", "falcon-h1")    # decoders: what the MEGA65 client chats with
ENCODERS = ("bert",)                           # listed only with --encoders
DEFAULT_QUERIES = ["tinystories", "smollm", "tinyllama", "stories15m", "stories260k", "gpt2",
                   "falcon-h1-tiny"]
ENCODER_QUERIES = ["minilm", "bert-tiny", "bert-mini"]
# Models found not to work (by base_name: every quantisation of them): broken files
# (TinyStories-LLaMA2-20M-256h-4l-GQA gives nonsense even in an independent
# reference), or architectures the converter can't read (mixture of experts, ...)
BROKEN = {"tinystories-llama2-20m-256h-4l-gqa", "stories15m-moe", "stories260k-infill",
          "stories15m-quant-ds", "minilmv2-bertscore-distilled"}

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
    if "falcon-h1" in s:
        return "falcon-h1"
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


def base_name(name):
    """A file's model, without its quantisation and packaging tags: the same
    model from different repositories (stories15M-q8_0, llama2.c-stories15M.f16,
    stories15M-bf16, ...) gives the same key."""
    n = name.lower()
    n = re.sub(r"\.gguf$", "", n)
    n = re.sub(r"[-_.](i1|imat)(?=[-_.]|$)", "", n)
    n = re.sub(r"[-_.](f32|fp32|bf16|f16|fp16|i?q\d(_[a-z0-9]+)*|tq\d_\d|mxfp4)(?=[-_.]|$)", "", n)
    n = re.sub(r"[-_.]?(ggml[-_.]model|gguf)(?=[-_.]|$)", "", n)
    n = re.sub(r"^(llama2\.c|xenova)[-_.]", "", n)
    return re.sub(r"[-_.]+", "-", n).strip("-")


def fetch_list(listfile, outdir):
    """Download each URL in listfile (one per line, # comments) to outdir,
    skipping files already there; a partial download is kept as .part and
    continued next time.  Returns the number that failed."""
    import os
    os.makedirs(outdir, exist_ok=True)
    failed = 0
    for line in open(listfile):
        url = line.split("#")[0].strip()
        if not url:
            continue
        name = urllib.parse.unquote(url.rstrip("/").split("/")[-1].split("?")[0])
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            continue
        part = path + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        print(f"fetching {name}" + (f" (from {have} bytes)" if have else ""), file=sys.stderr)
        try:
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(part, "ab" if have and r.status == 206 else "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(part, path)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  failed: {e}", file=sys.stderr)
            failed += 1
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="*", help=f"search terms (default: {' '.join(DEFAULT_QUERIES)})")
    ap.add_argument("--mem", type=int, choices=MEM_SIZES,
                    help="only show models estimated to fit this memory size")
    ap.add_argument("--limit", type=int, default=15, help="repositories per search term")
    ap.add_argument("--encoders", action="store_true",
                    help="also list encoders (BERT, MiniLM: embeddings, not text)")
    ap.add_argument("--names", metavar="FILE",
                    help="no search: print the file name of each URL in FILE (for make)")
    ap.add_argument("--all-arch", action="store_true",
                    help="also list architectures SSNAIL can't run yet")
    ap.add_argument("--fetch", metavar="FILE",
                    help="no search: download each URL listed in FILE into --dir, unless "
                         "the file is already there")
    ap.add_argument("--dir", default=".", help="where --fetch puts the files (default .)")
    ap.add_argument("--urls", metavar="FILE",
                    help="also write the best file's URL of each candidate to FILE, one per "
                         "line (for make allmodels): split files and repeated names left out")
    args = ap.parse_args()
    if args.fetch:
        sys.exit(fetch_list(args.fetch, args.dir))
    if args.names:
        for line in open(args.names):
            url = line.split("#")[0].strip()
            if url:
                print(urllib.parse.unquote(url.rstrip("/").split("/")[-1].split("?")[0]))
        return
    supported = SUPPORTED + (ENCODERS if args.encoders else ())
    queries = args.queries or DEFAULT_QUERIES + (ENCODER_QUERIES if args.encoders else [])

    seen, rows = set(), []
    for q in queries:
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
            ok_arch = arch and arch.rstrip("?") in supported
            if best and base_name(best[0].split("/")[-1]) in BROKEN:
                continue
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
    urls, names = [], set()
    for dl, repo, arch, params, best, fits in rows:
        url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(best[0])}"
        local = best[0].split("/")[-1]
        print(f"  make fetch URL={url}" + (f"   # {local}" if local else ""))
        # (the converter reads one file: a split model's first part is not one;
        # two repositories' files of one name would land on the same file)
        if not re.search(r"-\d{5}-of-\d{5}\.gguf$", local) and local not in names:
            names.add(local)
            urls.append((base_name(local), file_bpw(local) or 0, url))
    # one file per model: the most precise of a model's files (the converter
    # quantises better from F16 / F32 than from an already-quantised file);
    # rows come smallest model first, so ties keep the first
    best_of = {}
    for key, bpw, url in urls:
        if key not in best_of or bpw > best_of[key][0]:
            best_of[key] = (bpw, url)
    urls = [u for k, (b, u) in best_of.items()]
    if args.urls:
        with open(args.urls, "w") as f:
            f.write("# SSNAIL candidates from ssnail_findmodels.py: one URL per line (edit freely)\n")
            f.write("".join(u + "\n" for u in urls))
        print(f"wrote {len(urls)} URLs to {args.urls}", file=sys.stderr)


if __name__ == "__main__":
    main()
