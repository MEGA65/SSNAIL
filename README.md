# SSNAIL tools

PC-side tools for SSNAIL, the MEGA65 LLM inference accelerator.

| File | Purpose |
|---|---|
| `ssnail_convert.py` | GGUF → SSNAIL memory image; prompt tokenizer; model info |
| `ssnail_sim.py` | Reference emulator: executes an image exactly as the hardware should |
| `ssnail_findmodels.py` | Searches Hugging Face for GGUF models SSNAIL can run, with size estimates |
| `ssnail_tok.py` | Tokenizer that uses only the image's tables: the reference for the MEGA65-native one |
| `ssnail_chat.py` | Interactive front end to the emulator: chat/continue for decoders, similarity search for encoders |
| `ssnail_isa.py` | Instruction set and image layout definitions, shared by both |
| `tests/test_pipeline.py` | Llama end-to-end tests against an independent numpy implementation |
| `tests/test_gpt2_bert.py` | The same for GPT-2 and BERT |

Requirements: Python 3.9+, `pip install gguf numpy`.

## Makefile

```
make foo.ssnail                          # from foo.gguf; architecture detected
CONTEXT_WINDOW=128 make foo.ssnail
make foo.ssnail WTYPE=q4_0 ATTICRAM=8
make foo.run PROMPT="Once upon a time" TOKENS=60   # convert if needed, then emulate
make foo.chat [TEMP=0]                   # interactive session (TEMP=0: greedy)
make foo.info
make foo.tokcheck [TEXT=file.txt]        # image tokenizer vs the GGUF's
make test
```

Memory sizes:

| Target | Memory | Where the model goes |
|---|---|---|
| `foo.8mb.ssnail` (= `foo.ssnail`) | attic RAM, all boards | everything from $8000000 |
| `foo.64mb.ssnail` | R4–R6 | SDRAM ($8800000 on), plus the 64 KB host window; leaves $8010000–$87FFFFF of attic RAM free |
| `foo.72mb.ssnail` | R4–R6 | from $8010000 through attic RAM and on into SDRAM |

See "Large images and the host window" below for how these are loaded.

`WTYPE` defaults to `auto`, which tries in turn: the file's own formats,
Q8_0, `mixed` (Q8_0 layers with the large embedding/classifier at Q4_0), then
Q4_0.  If the choice would store weights with fewer bits than an
already-quantised source (say a Q8_0 file squeezed to Q4_0), it warns and
asks first; giving `WTYPE` explicitly counts as consent.  Converting from
F16/F32 never asks.  If a model doesn't fit at all, the message lists the
larger memory sizes it should fit (`make foo.72mb.chat`, or `ATTICRAM=72`).  Use the Makefile from another directory with
`make -f path/to/ssnail_tools/Makefile foo.ssnail`.  Images rebuild when the
converter changes.

```
python3 ssnail_convert.py info model.gguf
python3 ssnail_convert.py convert model.gguf -o model.ssnail --ctx 256
python3 ssnail_convert.py tokenize model.gguf "Once upon a time"
python3 ssnail_chat.py model.ssnail --gguf model.gguf          # interactive
python3 ssnail_chat.py model.ssnail --gguf model.gguf --prompt "Once upon a time"
python3 ssnail_sim.py model.ssnail --tokens <ids from tokenize> -n 64
python3 tests/test_pipeline.py
python3 tests/test_gpt2_bert.py
```

## Tokenizer in the image

Images are self-contained: everything needed to turn typed text into tokens
and tokens back into text is in the image, in attic RAM where the CPU can
read it.  `ssnail_tok.py` is the reference implementation of exactly what
the MEGA65-native tokenizer will do: binary searches over a sorted index,
short string compares, and a per-token merge priority.

| Family | Method | Encode tables |
|---|---|---|
| SentencePiece (llama) | merge adjacent pieces, highest score first | sorted index + priority + byte table |
| byte-level BPE (gpt2) | GPT-2 pre-split, then merge by rank | same; priority = rank of the merge that made the token |
| WordPiece (bert) | lowercase, greedy longest match per word | sorted index |

For GPT-2 vocabularies (about 50k tokens) the encode tables are about 350 KB,
on top of the vocab table.  The vocab table holds each token's raw bytes
(UTF-8).  Display translation is the front end's job: byte-level
vocabularies split characters across tokens, so it must fold the output
*stream*, not single tokens.  `ssnail_tok.AsciiFolder` is the reference: a
small UTF-8 state machine plus a table mapping curly quotes, dashes,
ellipses and accented Latin-1 letters to ASCII, and anything else to `?`.
Typed text is ASCII, so only tokens whose text is plain ASCII are in the
encode index.  BOS is added when the GGUF's `add_bos_token` says so (default:
yes for SentencePiece, no for GPT-2 BPE).

GPT-2's merge list is replaced by one priority per token.  This is exact on
the test vocabularies; `make foo.tokcheck` compares it with the real
tokenizer on any model, and should be run on each new model family.

## Interactive mode

`ssnail_chat.py` drives an image the way MEGA65 software would: it writes the
tokens and runtime block, starts the script, and prints each token when the
script publishes it (at SYNC), keeping the context between turns.

- Decoders: type text and the model continues it; an empty line continues.
  When the context fills, `/reset`.
- Encoders (BERT): each line is embedded and the three most similar earlier
  lines are shown with their cosine similarity.
- `/n N`, `/temp T`, `/topk K`, `/penalty P`, `/greedy`, `/hw`, `/eos`,
  `/seed S`, `/stats`, `/reset`, `/quit`.
- `/eos` (or `--eos`) shows `<EOS n%>` wherever the sampler gives the
  end-of-text token 5% or more: useful for seeing whether a model ends its
  stories, or runs straight on into the next one.

Tokenization uses only the image (see "Tokenizer in the image"), so no GGUF is
needed.  Token ids are model-specific, so
ids from one model's tokenizer produce gibberish prompts in another.

Defaults are chosen so nothing else is needed: output is sampled (temperature 0.8, top-k 40, repetition penalty 1.15 over the last
64 tokens).  `--greedy` (or `/greedy`, or `make foo.chat TEMP=0`) gives
exactly what the v1 hardware does.

Sampling and the repetition penalty are an emulator-only preview: the v1
hardware ARGMAX is greedy.
Small models decoded greedily tend to loop on names and phrases ("Lily,
Lily, Ben's mommy"); sampling at 0.7–1.0 with top-k 40 reads much better, and
is a strong argument for hardware sampling (an LFSR plus a cumulative-sum
scan over the logits).

## Large images and the host window

Terminology: *attic RAM* is the 8 MB the CPU sees at $8000000 (HyperRAM
on every board).  SDRAM (R4–R6) is reachable only through SSNAIL.  SSNAIL's
own region registers keep the chip names ("HyperRAM size", "SDRAM base").

Images bigger than 8 MB use SSNAIL's R4–R6 map: attic RAM at $8000000, SDRAM
at $8800000 (SSNAIL registers: HyperRAM size 8, SDRAM base $88, size 64; the
header's `$60` bytes say so).  The CPU only ever sees attic RAM, so the first
64 KB of attic RAM is a **host window** that the CPU and SSNAIL share:

| Address | Contents |
|---|---|
| $8000000 | Header and runtime block |
| $8000080 | Loader job: 4 instruction slots |
| $8000100 | Output vector (BERT embedding), up to 3840 bytes |
| $8001000 | Token buffer, 16 KB (up to 4095 tokens of context) |
| $8005000 | Staging buffer, 44 KB |
| $8010000 | (72 MB images: the model starts here) |

The CPU reads tokens and outputs directly from the window, so nothing has to
be read back out of SDRAM at run time.

**Loading.**  Attic RAM and SDRAM both respond at $8000000-$87FFFFF on the
CPU side, so the CPU can't write a big image itself.  Instead it streams the
file through SSNAIL's **load port**, which writes each byte wherever the
SSNAIL address map says (attic RAM or SDRAM):

| Register | Use |
|---|---|
| $FFD7514-$FFD7516 | Load pointer bits 0-23 (staged until $17 is written) |
| $FFD7517 | Write: pointer bits 24-27, and commit (flushing any pending bytes to the old pointer first); bit 6 also clears ERROR.  Read: bit 7 READY, bit 6 ERROR |
| $FFD7518 | Data: one byte per write, at the pointer, which advances |

The loader:

1. Set SSNAIL's region registers from header `$60`.
2. Set the pointer to $8000000 and commit, with bit 6 set (clear ERROR).
3. For each 512-byte sector of the file: wait for READY, have the SD card
   controller read the sector, DMA it to $FFD7518 (destination held), and
   move on.  Where a segment boundary falls inside a sector (header -> host
   segment -> payload; see header `$64`, `$70`, `$74`), DMA up to the
   boundary, write the next segment's address to the pointer and commit,
   then DMA the rest.  No sector alignment is needed.
4. Commit once more at the end to flush, wait for READY, and check ERROR.

SSNAIL buffers bytes in a 1 KB ring indexed by destination address, and
writes each 256-byte block to RAM as soon as the pointer leaves it, followed
by a cache invalidate on that RAM port.  READY means there is room for
another 512 bytes, no commit is outstanding, and no job is running.  The
emulator loads every image this way, so the protocol is exercised on each
run.  (The host window's 44 KB staging buffer is no longer used by loading.)

## Getting models

`make findmodels` searches Hugging Face for GGUF models in architectures SSNAIL
supports, and estimates which memory sizes each fits and at which weight
format:

```
make findmodels                               # default searches for small models
make findmodels QUERY="tinystories minilm" MEM=8
make fetch URL=https://huggingface.co/<repo>/resolve/main/<file>.gguf
```

It prefers unquantised (F16/F32) files to download, prints a `make fetch`
line for each, and uses only the Python standard library.  The fit columns
are estimates from the parameter count; `make foo.all` is the final word.
Architecture is read from Hugging Face's GGUF metadata where available,
otherwise guessed from names (shown as `llama?`).

GGUF files on Hugging Face can be downloaded directly; no account is needed
for public models.  On a model page, open **Files and versions** and use the
download arrow, or fetch the `resolve` URL:

```
curl -L -O https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF/resolve/main/tinystories-gpt-0.1-3m.fp16.gguf
```

or with the Hugging Face CLI (`pip install huggingface_hub`):

```
huggingface-cli download afrideva/Tinystories-gpt-0.1-3m-GGUF tinystories-gpt-0.1-3m.fp16.gguf --local-dir .
```

Prefer the F16/F32 file and let the converter quantise (`--wtype q4_0` or
`q8_0`): requantising an already-quantised K-quant file loses more quality.


The emulator prints the RAM traffic per token and an estimated token rate at
a given sustained bandwidth (default 130 MB/s).

## Status

- Architectures:
  - `llama`: Llama 1/2/3 family, llama2.c TinyStories exports, SmolLM, ...
    Grouped-query attention, tied or untied classifier.
  - `gpt2`: GPT-2 family, including the TinyStories GPT-2 models.
  - `bert`: BERT encoders (e.g. all-MiniLM sentence embedders), mean or CLS
    pooling.  The image leaves the pooled embedding at header `$30`.
- Weight formats executed natively: F32, F16, BF16, Q8_0, Q4_0.  Weights in
  any other ggml format that gguf-py can dequantize (e.g. the Q4_K/Q6_K in a
  Q4_K_M file) are repacked to `--fallback` (default Q8_0).  `--wtype` forces
  one format for every matrix.
- Prompt tokenization: SentencePiece (`llama`), byte-level BPE (`gpt2`) and
  WordPiece (`bert`, uncased).  The GPT-2 pre-tokenizer is exact for English
  text; unusual Unicode may split differently from the reference.
- KV caches are F16 by default (`--kv f32` for exact reference runs); see
  `ssnail-numerics.md`.  The emulator has two numerics modes: the float
  reference, and **hardware numerics** (`Machine.hw = True`, `ssnail_chat.py
  --hw`, `make foo.chat HW=1`), defined bit by bit in `ssnail_hw.py`, which
  the VHDL is checked against.  On the test models, hardware numerics change
  logits by about 1% and never change the greedy choice.
- Activations, norm weights and the RoPE table are float32.
  This defines the reference semantics; the hardware will use narrower
  fixed-point formats and be checked against the emulator within a tolerance.
  The float32 KV cache is the main memory cost: 8 bytes × layers × ctx ×
  kv_dim.

Verified: logits at every position (and BERT's pooled embeddings) match
independent references to about 3e-7 relative error, for F32, F16, Q8_0 and
Q4_0 weights, GQA, tied and untied classifiers, and both pooling types.  All
three tokenizers are tested.  Not verified here: K-quant input files (gguf-py can't
create them for the test).  The repack path is exercised with F32 → Q4_0.

## Memory image layout

The image is loaded at $8000000 (attic RAM):

| Address | Contents |
|---|---|
| $8000000 | Header and runtime block (256 bytes) |
| $8000100 | SSNAIL script: the complete per-token program, looping |
| ... | Weights, norm vectors, RoPE table, vocab table, attention parameter blocks |
| *end of file* | |
| `H_SCRATCH` | Activations and KV cache (not in the file; no need to load or clear) |
| `H_TOKENS` | Token buffer (u32 per token), last so it is open-ended |

(That is the 8 MB layout.  Larger images keep the header, tokens and outputs in
a 64 KB host window at $8000000; see "Large images and the host window".)

### Header (static, written by the converter)

| Offset | Field |
|---|---|
| $00 | Magic `SSNL` |
| $04 | u16 version (1), u16 flags |
| $08 | Address of the script |
| $0C | Start of weights |
| $10 | Start of scratch (= end of the file, in SSNAIL addresses) |
| $14 | Token buffer |
| $18 | Vocab table: u32 offsets[n_vocab + 1] relative to the table, then the bytes |
| $1C | n_vocab |
| $20 | Maximum context (KV cache capacity) |
| $24 | BOS token |
| $28 | EOS token |
| $2C | Payload length in bytes (the file is the 256-byte header, then the host segment if any, then the payload) |
| $30 | Address of the f32 output vector: logits, or the pooled embedding for encoders |
| $34 | Length of the output vector |
| $38 | Bytes of memory needed from $8000000, including scratch, KV cache and token buffer |
| $3C | Architecture: 0 llama, 1 gpt2, 2 bert |
| $60 | SSNAIL region setup: bytes HyperRAM MB, SDRAM base MB, SDRAM MB, 0 |
| $64 | Payload load address ($8000100 for 8 MB images, so the file is a plain memory image) |
| $68 | Staging buffer address (0 = load directly) |
| $6C | Staging buffer size |
| $70 | Host segment address (64 MB images: CPU-side tables in attic RAM at $8010000) |
| $74 | Host segment length (0 = the tables are in the payload, already in attic RAM) |
| $78 | Tokenizer block address (format in `ssnail_isa.py`) |

Header `$06` flags: bit 0 set = encoder image (BERT).

### Runtime block (shared by the CPU and the script)

| Offset | Field | Written by |
|---|---|---|
| $40 | `pos`: next position to process | CPU, then script |
| $44 | `prompt_len`: `tokens[0..prompt_len)` are the prompt | CPU |
| $48 | `n_generate`: maximum tokens to generate this run | CPU |
| $4C | `generated`: tokens generated so far this run | script |
| $50 | `stop_request`: non-zero = stop after the current token | CPU |
| $54 | `stop_reason`: 1 count, 2 EOS, 3 context full, 4 stop request | script |

### Running a generation (what MEGA65 software does)

1. Load the image file at $8000000 (larger images: see the loading steps
   under "Large images and the host window").
2. Write the prompt tokens to `H_TOKENS`, then set `pos = 0`,
   `prompt_len`, `n_generate`, `stop_request = 0`.
3. Write the address at header $08 to the SSNAIL job pointer, then set GO.
4. While SSNAIL is busy, poll `generated`.  Each new token is at
   `tokens[prompt_len + generated - 1]`.  Decode it with the vocab table.
   The script SYNCs after writing each token and its count, so the CPU sees
   them without stale cache contents.
5. To continue later, append tokens after the last one, set
   `prompt_len = pos + 1 + appended`, keep `pos`, and GO again.

For an encoder image: write the tokens (including [CLS] and [SEP]), set
`prompt_len`, GO, and read the embedding from the address at header `$30`
once SSNAIL is idle.  The sequence may be up to the header's maximum context
in length; longer sequences stop with reason 3 and produce no output.

## Instruction set v1 (draft)

16 bytes per instruction: `op, a, b, 0, X:u32, Y:u32, Z:u32`.

Address operands: bit 31 clear means an absolute 28-bit address.  Bit 31 set
means `A[bits 27–24] + bits 23–0`.

Registers: R0–R15 (32-bit; R15 reads as 0), A0–A15 (28-bit addresses), and
N0–N2 (lengths, set by SETN).

| Op | Name | Semantics |
|---|---|---|
| $00 | NOP | |
| $01 | HALT | job done |
| $02 | SYNC | invalidate CPU-side read caches on every port |
| $03 | COPY | copy `Z & $FFFF` bytes from addr(X) to addr(Y) |
| $07 | SETN | N0 = X, N1 = Y, N2 = Z |
| $08 | LI | R[a] = Z |
| $09 | LDR | R[a] = u32 at addr(X) |
| $0A | STR | u32 at addr(X) = R[a] |
| $0B | ADDI | R[a] = R[b] + signed Z |
| $0C | LEA | A[a] = addr(X) + R[b] × Z |
| $0D/$0E/$0F | BEQ/BNE/BLT | if R[a] ==/!=/< R[b] (unsigned): PC = addr(X) |
| $10 | GEMV | y at addr(Z) = W at addr(X) · x at addr(Y).  W is N1 rows × N0 columns in format `a` (ggml type number).  Flag `b` bit 0 accumulates into y |
| $11 | DEQROW | f32 vector at addr(Z) = N0 elements of format `a` from addr(X) |
| $12 | ARGMAX | R[a] = index of the maximum of N0 f32 values at addr(X) |
| $20 | VADD | Z = X + Y (N0 f32) |
| $21 | VMUL | Z = X × Y |
| $22 | RMSNORM | Z = X · rsqrt(mean(X²) + eps) · Y, with eps = N2 as float32 bits |
| $23 | SILUMUL | Z = silu(X) · Y |
| $24 | ROPE | rotate adjacent pairs of the N0-element vector at addr(X) in place, using N1/2 (cos, sin) pairs at addr(Y); N1 = head dimension |
| $25 | ATTN | q at addr(Y) attends to K/V rows 0..R[a] inclusive; result at addr(Z).  addr(X) is an 8 × u32 parameter block: K cache, V cache, n_heads, n_kv_heads, head_dim, 3 × reserved.  Causal decoders pass the current position; BERT passes n−1 for every token |
| $26 | LAYERNORM | Z = (X − mean) · rsqrt(var + eps) · G + B; G at addr(Y), B immediately after it; eps = N2 |
| $27 | GELU | Z = gelu(X), tanh approximation (as ggml) |
| $28 | MEANROWS | Z = mean of R[a] consecutive N0-element f32 rows starting at addr(X) |
| $29 | CVT16 | Z (N0 × F16) = X (N0 × F32), round to nearest even (storing K/V rows in an F16 cache) |

GEMV flag `b` bit 1 writes y as F16 instead of F32.  The ATTN parameter
block's sixth word is the KV cache format: 0 F32, 1 F16.

Notes for the hardware:

- Rows of a matrix are contiguous, so a GEMV is a single linear stream.  Row
  starts (and embedding rows reached via LEA) are not necessarily 8-byte
  aligned.  The unpacker must handle a misaligned start by fetching from the
  aligned address below it and discarding the leading bytes.
- GEMV with the accumulate flag is the residual add.  The fused forms
  (SwiGLU pairing, folded norms, argmax in the classifier output stage) are
  hardware optimisations of the same semantics, so the converter can start
  emitting them once the hardware supports them.
- BERT runs each layer in two passes over the sequence (q/k/v for all
  tokens, then attention and the rest), so every weight is streamed once per
  token.  A batched GEMV (several input vectors per weight pass) would cut
  that substantially, and would also speed up decoder prompt prefill.
- ATTN, RMSNORM, LAYERNORM, SILUMUL and GELU are macro-ops.  The hardware can implement them
  as microcoded sequences of its vector unit.
