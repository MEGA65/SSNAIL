#!/usr/bin/env python3
"""Check the C tokenizer (tok_host) against ssnail_tok.py on images: the
reference sample plus edge cases.  tok_check.py image.ssnail ..."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ssnail_sim as S
import ssnail_tok as T

EXTRA = ["", " ", "a", "Hello", "  two  spaces", "Tom's dog isn't here, they'll see",
         "ALL CAPS SHOUTING", "x" * 70, "1234567890 3.14159 1e10",
         "tabs\tand\tsymbols ~!@#$%^&*()_+{}|:<>?", "a" * 150, "end."]

def main():
    host = os.path.join(HERE, "tok_host")
    lines = [l for l in T.SAMPLE.splitlines() if l.strip()] + EXTRA
    bad = 0
    for path in sys.argv[1:]:
        m = S.Machine(open(path, "rb").read())
        it = T.ImageTokenizer(m.mem)
        r = subprocess.run([host, path], input="\n".join(lines) + "\n", capture_output=True, text=True)
        out = r.stdout.split("\n")
        nb = 0
        for k, line in enumerate(lines):
            want = it.encode(line, first=True)
            got = [int(x) for x in out[2 * k].split(",")] if out[2 * k] else []
            fold = T.AsciiFolder()
            want_txt = "".join(fold.feed(it.piece(t)) for t in want)
            if got != want or out[2 * k + 1] != want_txt:
                nb += 1
                if nb <= 2:
                    print(f"  DIFFER {line[:40]!r}\n    C:  {got[:16]}\n    py: {want[:16]}")
        print(f"{'ok ' if nb == 0 else 'BAD'} {os.path.basename(path)}: {len(lines) - nb}/{len(lines)} lines")
        bad += nb
    sys.exit(1 if bad else 0)

main()
