#!/usr/bin/env python3
"""Offline test of ssnail_findmodels.py with mocked Hugging Face API replies."""
import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ssnail_findmodels as F  # noqa: E402

MB = 1 << 20
FAKE_SEARCH = [
    {"id": "afrideva/Tinystories-gpt-0.1-3m-GGUF", "downloads": 2039,
     "gguf": {"architecture": "gpt2", "total": 6960000}},
    {"id": "someone/MiniLM-L6-GGUF", "downloads": 500},                  # no gguf info
    {"id": "big/Llama-3-8B-GGUF", "downloads": 99999,
     "gguf": {"architecture": "llama", "total": 8030000000}},
    {"id": "odd/Mamba-130m-GGUF", "downloads": 50, "gguf": {"architecture": "mamba", "total": 130e6}},
]
FAKE_TREES = {
    "afrideva/Tinystories-gpt-0.1-3m-GGUF": [
        {"type": "file", "path": "tinystories-gpt-0.1-3m.Q2_K.gguf", "size": int(7.75 * MB)},
        {"type": "file", "path": "tinystories-gpt-0.1-3m.Q8_0.gguf", "size": int(9.58 * MB)},
        {"type": "file", "path": "tinystories-gpt-0.1-3m.fp16.gguf", "size": 16 * MB},
        {"type": "file", "path": "README.md", "size": 1880}],
    "someone/MiniLM-L6-GGUF": [
        {"type": "file", "path": "all-MiniLM-L6-v2.Q8_0.gguf", "size": int(24 * MB)}],
    "big/Llama-3-8B-GGUF": [{"type": "file", "path": "llama3.Q4_K_M.gguf", "size": 4900 * MB}],
    "odd/Mamba-130m-GGUF": [{"type": "file", "path": "mamba.f16.gguf", "size": 260 * MB}],
}


def fake_get_json(url, timeout=30):
    if "/tree/main" in url:
        repo = url.split("/api/models/")[1].split("/tree/")[0]
        return FAKE_TREES[repo]
    return FAKE_SEARCH


F.get_json = fake_get_json
out = io.StringIO()
sys.argv = ["ssnail_findmodels.py", "anything"]
with redirect_stdout(out), redirect_stderr(io.StringIO()):
    F.main()
text = out.getvalue()
print(text)
checks = [
    ("TinyStories GPT-2 listed, 8 MB needs q4", "afrideva/Tinystories-gpt-0.1-3m-GGUF" in text
     and [l for l in text.splitlines() if l.startswith("afrideva")][0].split()[3] == "q4"),
    ("fp16 file chosen for download", "tinystories-gpt-0.1-3m.fp16.gguf" in text),
    ("MiniLM found by name, params estimated", "someone/MiniLM-L6-GGUF" in text),
    ("8B Llama excluded (fits nowhere)", "Llama-3-8B" not in text),
    ("unsupported arch excluded", "Mamba" not in text),
]
ok = all(c for _, c in checks)
for name, c in checks:
    print(f"{'PASS' if c else 'FAIL'}  {name}")
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
