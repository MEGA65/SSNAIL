#!/usr/bin/env python3
"""The image-only tokenizer must match the reference GGUF tokenizer exactly,
for all three tokenizer families, in every memory layout."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gguf import GGMLQuantizationType as Q  # noqa: E402
import ssnail_convert as C  # noqa: E402
import ssnail_sim as S  # noqa: E402
import ssnail_tok as T  # noqa: E402
import test_pipeline as TP  # noqa: E402
import test_gpt2_bert as TG  # noqa: E402

TEXTS = T.SAMPLE.splitlines() + [
    "the cat sat on the mat", "once upon a time the happy dog",
    "throw the lamp at the troll, playing cats.", "", "   ", "a", "the  cat",
    "Throw The LAMP!", "cats dogs runs jumped", "the\tcat\nsat"]

ok = True
with tempfile.TemporaryDirectory() as d:
    for name, maker in (("spm (llama)", lambda p: TP.make_model(p, Q.Q8_0)),
                        ("bpe (gpt2)", lambda p: TG.make_gpt2(p, Q.Q8_0)),
                        ("wordpiece (bert)", lambda p: TG.make_bert(p, Q.Q8_0))):
        g = os.path.join(d, "m.gguf")
        maker(g)
        model = C.Model(g)
        for mb in (8, 64, 72):
            image = C.convert(model, 32, "keep", C.FMT_BY_NAME["q8_0"], mb, verbose=False)
            it = T.ImageTokenizer(S.Machine(image).mem)
            bad = [t for t in TEXTS if it.encode(t) != C.encode(model, t)]
            for t in bad[:3]:
                print(f"   {t!r}: image {it.encode(t)} vs gguf {C.encode(model, t)}")
            ok &= not bad
            print(f"{'PASS' if not bad else 'FAIL'}  {name:18s} {mb:2d} MB layout: "
                  f"{len(TEXTS) - len(bad)}/{len(TEXTS)} texts identical")
    # BOS follows tokenizer.ggml.add_bos_token (GPT-2 tokenizer: off by default)
    for add_bos in (None, False, True):
        g = os.path.join(d, "b.gguf")
        TG.make_gpt2(g, Q.Q8_0, add_bos=add_bos)
        model = C.Model(g)
        it = T.ImageTokenizer(S.Machine(C.convert(model, 32, "keep", C.FMT_BY_NAME["q8_0"], 8,
                                                  verbose=False)).mem)
        a, b = it.encode("once upon a time"), C.encode(model, "once upon a time")
        eos = model.field("tokenizer.ggml.eos_token_id")
        bok = a == b and ((a[0] == eos) == bool(add_bos))
        ok &= bok
        print(f"{'PASS' if bok else 'FAIL'}  BOS with add_bos_token={add_bos}: "
              f"{'starts with BOS' if a[0] == eos else 'no BOS'}")

    # Display folding on the output stream, including characters split
    # across tokens (as GPT-2 byte-level vocabularies do)
    text = "\u201cHi\u201d \u2014 caf\u00e9\u2026 \u2603 soon\u2019!".encode()
    folder = T.AsciiFolder()
    pieces = [text[i:i + 1] for i in range(len(text))]       # worst case: one byte each
    fold = "".join(folder.feed(p) for p in pieces)
    fok = fold == '"Hi" - cafe... ? soon\'!'
    ok &= fok
    print(f"{'PASS' if fok else 'FAIL'}  ASCII folding across token boundaries: {fold!r}")
    # The image's vocab table holds raw UTF-8, not folded text
    with tempfile.TemporaryDirectory() as d2:
        g = os.path.join(d2, "m.gguf")
        TG.make_gpt2(g, Q.Q8_0)
        mm = S.Machine(C.convert(C.Model(g), 32, "keep", C.FMT_BY_NAME["q8_0"], 8, verbose=False))
        raw = S.vocab_piece(mm, 0xE2)        # token n is byte n in this test vocab
        rok = raw == b"\xe2"
        ok &= rok
        print(f"{'PASS' if rok else 'FAIL'}  vocab table keeps raw bytes ({raw!r})")
print("ALL PASSED" if ok else "FAILURES")
sys.exit(0 if ok else 1)
