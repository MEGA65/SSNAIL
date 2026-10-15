# SSNAIL numerics: decisions

Status: **decided**.  This fixes what the datapath (step 2 onwards) computes,
so the converter, the reference emulator and the hardware all target the
same formats.

## Principle

Memory bandwidth decides speed, and almost all of it is weights.  Activation
vectors are small (one row of `dim` or `hidden` values per op), so their
format barely affects speed and can favour accuracy and simple hardware.  The
one large activation structure is the KV cache, which is re-read every token,
so it gets a compact format.

## Formats

| Data | Format in RAM | Notes |
|---|---|---|
| Weight matrices | as converted: F32, F16, BF16, Q8_0, Q4_0 (ggml layouts) | unchanged from ISA v1 |
| Activation vectors (residual stream, q, attention output, FFN intermediates, logits) | **F32** | unchanged from ISA v1; small, so cost is negligible |
| Norm gains/biases, RoPE table | **F32** | unchanged |
| **KV cache** | **F16** | changed from F32: halves the largest activation structure and its per-token traffic |
| GEMV input vector, inside SSNAIL | **Q8_0**: int8, one F16 scale per 32 elements, exactly llama.cpp's `quantize_row_q8_0` | for Q8_0, Q4_0, IQ4_NL and MXFP4 weights |
| GEMV input vector, K-quant weights | **Q8_K-style**: int8, one F32 scale per 256 elements, with Q8_0's arithmetic (`quantize_q8_k`); plus the sum of each 16 quantised elements | for Q2_K .. Q6_K, IQ4_XS, TQ2_0 and IQ2 / IQ3 weights |

## Arithmetic

- **GEMV, quantised weights (Q8_0, Q4_0):** the input vector is quantised to
  Q8_0 as it is loaded into the on-chip buffer.  Per 32-element block,
  Σ q_w·q_x is computed exactly as an integer; it is multiplied by
  d_w·d_x and accumulated into the row result in F32, block by block in
  order.  This matches llama.cpp's CPU dot products, so results can be
  compared with llama.cpp directly.
- **GEMV, MXFP4 weights:** as Q4_0, with the E2M1 values doubled (ggml's
  kvalues) as the integers and d_w = ggml_e8m0_to_fp32_half(e), a power of
  two.
- **GEMV, IQ4_NL weights:** as Q4_0, with the 4-bit indices mapped through
  ggml's kvalues_iq4nl to the integers.
- **GEMV, K-quant weights (Q2_K .. Q6_K, and IQ4_XS and TQ2_0 the same way:
  IQ4_XS's 32-group scale less 32 with kvalues_iq4nl quants, TQ2_0 a scale of
  1 and quants less 1):** per
  256-weight super-block, two exact integers: S1 = Σ over 16-weight groups g
  of sc[g]·(Σ q_w·q_x over g), and, for the formats with mins (Q2_K, Q4_K,
  Q5_K), S2 = Σ m[g]·(Σ q_x over g).  Then, each operation rounded:
  t = F32(S1)·(d·d_x) − F32(S2)·(dmin·d_x), and acc = acc + t, super-block
  by super-block in order.  q_w is the raw quant less any fixed offset
  (Q3_K −4·(1−h), Q6_K −32), sc the 16-group's integer scale (Q4_K / Q5_K
  repeat each 32-group's), as gguf-py's dequantisers.  Like llama.cpp's
  Q8_K dot products in structure; not bit-identical to them.
- **GEMV, IQ2 / IQ3 weights (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S):** the
  K-quant arithmetic above, with q_w = ± the codebook value (ggml's grids:
  8 / 25 / 43 for IQ2, 4 .. 62 or 1 .. 15 for IQ3), sc = 2s + 1 per 16 or 32
  weights, and d = F32(d) / 8 (IQ2) or / 4 (IQ3_XXS), exact: llama.cpp's
  d·(0.5 + s)·0.25 is d/8·(2s + 1).  GEMV only: DEQROW does not take them,
  so the converter repacks IQ2 / IQ3 embedding tables.
- **GEMV, float weights (F32, F16, BF16):** the input is not quantised.
  The pipelined float lane: each product w·x rounded to F32; product i
  added into partial sum i mod 8 (each starting from −0); the row's result
  ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)), every operation
  rounded (`dot8`).  (An exact wide accumulator was considered: about 2k
  LUTs for accuracy inference does not need.)
- **ATTN:** scores s_t = dot8(k_t, q)·rsqrt(hd) (the float lane, as float
  GEMV); softmax as below; out = out + w_t·v_t sequentially over t for each
  element (also in the float lane).  K and V are streamed: 8-byte aligned,
  a head's row a whole number of 8 bytes.
- **Vector unit (VADD, VMUL, RMSNORM, LAYERNORM, SILUMUL, GELU, ROPE,
  MEANROWS, softmax inside ATTN):** one pipelined F32 add/multiply datapath.
  Vectors are short, so one element per cycle is ample.
- **Reciprocal and reciprocal square root:** iterative, to full F32 precision.
- **exp, SiLU, GELU:** piecewise-linear tables of **256** segments,
  y = c0[i] + c1[i]·x (each operation rounded to F32), segment
  i = floor((x − lo)·(256 / (hi − lo))):

  | Function | Range [lo, hi) | Below lo | At or above hi | Max abs error |
  |---|---|---|---|---|
  | SiLU | [−12, 12) | 0 | x | 5.5e-4 |
  | GELU | [−6, 6) | 0 | x | 2.2e-4 |
  | exp (softmax, inputs ≤ 0) | [−16, 0) | 0 | 1 | 4.7e-4 |

  (128 segments over ±8 left SiLU 2.7e-3 out at the clamp edges.)  Each
  table is 256 × 2 F32 coefficients, about 16 kbit of LUT RAM.
- **KV cache writes:** F32 → F16, round to nearest even, either by GEMV's
  F16-output flag (V, for Llama) or by the new **CVT16** instruction (K,
  which is rotated by ROPE in F32 first; and K/V for GPT-2 and BERT, whose
  biases are added in F32 first).  ATTN reads K and V as F16 (exactly
  representable in F32).

## Agreement between hardware and emulator

The emulator gains a **hardware-numerics mode** that implements exactly the
above.  Hardware is checked against it:

- **bit-exact** for everything integer or by-construction exact: Q8_0
  quantisation of the input, integer block sums, the float lane,
  F32 → F16 conversion, the scalar/control instructions;
- **within 1 ulp per F32 operation** for the vector unit and the iterative
  reciprocal/rsqrt;
- **exactly the tabulated values** for exp, SiLU and GELU (the tables are
  generated by the converter's tools, and the emulator uses the same tables).

The existing float32 emulator mode remains as the accuracy reference; the
difference between the two modes measures what the hardware formats cost in
model quality (expected to be small: this is what llama.cpp does on CPUs).

## Implemented (tools)

- ISA: GEMV flag bit 1 = F16 output; **CVT16** ($29); ATTN parameter block
  word 6 = KV format (0 F32, 1 F16).
- Converter: F16 KV caches by default, `--kv f32` for reference runs.
- Emulator: `Machine.hw` selects hardware numerics; every operation's
  arithmetic order is spelled out in `ssnail_hw.py`.
- Measured on the test models: Q8_0 input quantisation identical to
  llama.cpp's; GEMV bit-exact against a scalar reading of the definition;
  hardware vs float numerics change logits by about 1% (Q4_0/Q8_0) or
  0.03% (F16 weights), the greedy token at every position is unchanged,
  and BERT embeddings keep cosine similarity 0.99998.
