"""SSNAIL instruction set, v1 (draft), and memory-image layout.

Shared by the converter (ssnail_convert.py) and the reference emulator
(ssnail_sim.py).  The emulator defines the semantics; the hardware must match
it (to within the tolerances of its fixed-point arithmetic).

Instruction format: 16 bytes, 16-byte aligned, little-endian.

    byte 0      opcode
    byte 1      a   (register number or weight format)
    byte 2      b   (register number or flags)
    byte 3      c   (reserved, 0)
    bytes 4-7   X   (u32)
    bytes 8-11  Y   (u32)
    bytes 12-15 Z   (u32)

Address operands (X/Y/Z where the op takes an address):
    bit 31 = 0  absolute SSNAIL address (28 bits)
    bit 31 = 1  A[bits 27-24] + bits 23-0  (address register + offset)

Registers:
    R0-R15  32-bit integer registers.  R15 always reads as 0.
    A0-A15  28-bit address registers.
    N0-N2   length registers set by SETN.  GEMV uses N0 = columns,
            N1 = rows.  Vector ops use N0 = element count.  RMSNORM uses
            N2 as the float32 bit pattern of epsilon.  ROPE uses N1 = head
            dimension.

Activations, KV cache, norm weights and RoPE tables are float32 in v1.
Weight formats use ggml type numbers directly (see FMT_*).
"""

import struct

# --- Opcodes -----------------------------------------------------------------
NOP, HALT, SYNC, COPY = 0x00, 0x01, 0x02, 0x03
SETN = 0x07     # N0 = X, N1 = Y, N2 = Z
LI = 0x08       # R[a] = Z
LDR = 0x09      # R[a] = u32 at addr(X)
STR = 0x0A      # u32 at addr(X) = R[a]
ADDI = 0x0B     # R[a] = R[b] + (signed) Z
LEA = 0x0C      # A[a] = addr(X) + R[b] * Z
BEQ = 0x0D      # if R[a] == R[b]: PC = X
BNE = 0x0E      # if R[a] != R[b]: PC = X
BLT = 0x0F      # if R[a] <  R[b] (unsigned): PC = X
GEMV = 0x10     # y[addr Z] (+)= W[addr X] . x[addr Y]; a = fmt, b = flags
DEQROW = 0x11   # f32 vector at addr Z = dequant(N0 elements at addr X), a = fmt
ARGMAX = 0x12   # R[a] = argmax of N0 f32 at addr X
VADD = 0x20     # Z = X + Y            (N0 f32)
VMUL = 0x21     # Z = X * Y
RMSNORM = 0x22  # Z = X * rsqrt(mean(X^2) + eps(N2)) * Y
SILUMUL = 0x23  # Z = silu(X) * Y
ROPE = 0x24     # rotate pairs of X in place using (cos, sin) row at Y; N0, N1
ATTN = 0x25     # attention: X = parameter block, Y = q, Z = out; attends
                #   to positions 0..R[a] inclusive
LAYERNORM = 0x26  # Z = (X - mean) * rsqrt(var + eps(N2)) * G + B;
                  #   G at addr(Y), B at addr(Y) + 4*N0
GELU = 0x27     # Z = gelu(X) (tanh approximation, as ggml)
MEANROWS = 0x28 # Z = mean of R[a] rows of N0 f32 starting at addr(X)
CVT16 = 0x29    # Z (N0 x F16) = X (N0 x F32), round to nearest even

OPNAMES = {v: k for k, v in dict(
    NOP=NOP, HALT=HALT, SYNC=SYNC, COPY=COPY, SETN=SETN, LI=LI, LDR=LDR,
    STR=STR, ADDI=ADDI, LEA=LEA, BEQ=BEQ, BNE=BNE, BLT=BLT, GEMV=GEMV,
    DEQROW=DEQROW, ARGMAX=ARGMAX, VADD=VADD, VMUL=VMUL, RMSNORM=RMSNORM,
    SILUMUL=SILUMUL, ROPE=ROPE, ATTN=ATTN, LAYERNORM=LAYERNORM, GELU=GELU,
    MEANROWS=MEANROWS, CVT16=CVT16).items()}

GEMV_ACC = 0x01   # GEMV flag: accumulate into y (residual add)
GEMV_F16OUT = 0x02  # GEMV flag: write y as F16 (e.g. straight into a KV cache)

RZERO = 15        # R15 reads as zero

# --- Weight formats (ggml type numbers) ---------------------------------------
FMT_F32, FMT_F16, FMT_Q4_0, FMT_Q8_0, FMT_BF16 = 0, 1, 2, 8, 30
# Formats the v1 hardware/emulator execute natively.  Anything else is
# repacked by the converter.
NATIVE_FORMATS = {FMT_F32, FMT_F16, FMT_Q4_0, FMT_Q8_0, FMT_BF16}

# --- Image layout ---------------------------------------------------------------
HYPERRAM_BASE = 0x8000000
# SSNAIL's address map (its region registers default to these):
#   R3 (no SDRAM)   $8000000-$87FFFFF  HyperRAM
#   R4-R6           $8000000-$BFFFFFF  SDRAM (the faster memory, first)
#                   $C000000-$C7FFFFF  HyperRAM
# Every image uses one layout from $8000000, so an image of 8 MB or less is
# identical on every board.  The CPU reads the window and tables by mapping
# whichever RAM is at $8000000 (SDRAM when SSNAIL reports it present), and
# writes through SSNAIL's load port.
#
#   $8000000  header + runtime block (256 bytes)
#   $8000080  model description (header bytes $80-$FF, see H_DESC)
#   $8000100  output vector (BERT embedding), up to 3840 bytes
#   $8001000  token buffer, 16 KB (up to 4095 tokens of context)
#   $8005000  script, vocab and tokenizer tables, weights, then scratch
#
# Sizes: 8 MB (any board), 64 MB (R4-R6, all SDRAM), 72 MB (R4-R6, spilling
# into HyperRAM at $C000000; no weight tensor straddles $C000000, since GEMV
# streams each matrix from one RAM).
MEM_CONFIGS = {8: "any board", 64: "R4-R6, all in SDRAM",
               72: "R4-R6, SDRAM then HyperRAM at $C000000"}
WINDOW_END = HYPERRAM_BASE + 0x5000
OUTPUT_AREA = HYPERRAM_BASE + 0x100    # up to 3840 bytes (960 floats)
OUTPUT_AREA_SIZE = 0xF00
TOKENS_AREA = HYPERRAM_BASE + 0x1000   # 16 KB: up to 4096 tokens
TOKENS_AREA_SIZE = 0x4000
SD_END = HYPERRAM_BASE + 64 * 0x100000 # end of SDRAM on R4-R6 ($C000000)
MEM_TOP = HYPERRAM_BASE + 72 * 0x100000
HEADER_SIZE = 0x100
MAGIC = b"SSNL"
IMAGE_VERSION = 1

# Static header fields (offsets from HYPERRAM_BASE)
H_MAGIC = 0x00       # 'SSNL'
H_VERSION = 0x04     # u16 version, u16 flags
H_CODE = 0x08        # address of first instruction
H_DATA = 0x0C        # start of weights/tables
H_SCRATCH = 0x10     # start of activations + KV cache (not in the file)
H_TOKENS = 0x14      # token buffer (u32 tokens), last, open-ended
H_VOCAB = 0x18       # vocab table: u32 offsets[n_vocab + 1], then bytes
H_NVOCAB = 0x1C
H_MAXCTX = 0x20      # KV cache capacity in positions
H_BOS = 0x24
H_EOS = 0x28
H_PAYLOAD_LEN = 0x2C # bytes of payload in the file (after the 256-byte header)
H_OUTPUT = 0x30      # address of the f32 output vector (logits, or the
                     #   pooled embedding for encoder models)
H_OUTDIM = 0x34      # length of the output vector
H_ARCH = 0x3C       # 0 llama, 1 gpt2, 2 bert
ARCH_CODES = {"llama": 0, "gpt2": 1, "bert": 2}
H_MEM_NEEDED = 0x38  # bytes of linear memory the image needs from $8000000,
                     #   including scratch, KV cache and a full token buffer
H_FLAG_ENCODER = 0x0001   # in the u16 flags: encoder (BERT) image

H_MEMCFG = 0x60      # (unused: 0; SSNAIL's default region registers are right)
H_LOAD_BASE = 0x64   # where the payload loads
H_STAGING = 0x68     # (unused: 0)
H_STAGING_SIZE = 0x6C
H_DESC = 0x80        # description, NUL-terminated UTF-8: "name (arch) [type]",
H_DESC_SIZE = 0x80   #   type from convert --type (default General), up to $FF
H_HOST_BASE = 0x70   # (unused: 0)
H_HOST_LEN = 0x74    # (unused: 0)
H_TOKENIZER = 0x78   # address of the tokenizer block

# Tokenizer block (everything the MEGA65 needs to turn ASCII text into tokens;
# the algorithm is in ssnail_tok.py).  All tables live in attic RAM.
#   +0  u8  kind: TOK_SPM, TOK_BPE or TOK_WORDPIECE
#   +1  u8  flags: TF_*
#   +4  u32 n_vocab
#   +8  u32 address of the sorted index: u32 token ids, ordered by piece bytes
#   +12 u32 entries in the sorted index
#   +16 u32 address of the priority table: u16 (or u32 if TF_PRIO32) per
#           token; lower merges first; $FFFF = never produced by a merge
#   +20 u32 address of the byte table: 256 x u32, starting token per byte
#   +24 u32 begin token (BOS / [CLS]), $FFFFFFFF = none
#   +28 u32 end token ([SEP]), $FFFFFFFF = none
#   +32 u32 unknown token
# Piece text comes from the vocab table at H_VOCAB, which holds each token's
# text folded to plain ASCII for display.  Only tokens whose original text is
# plain ASCII are in the sorted index, since typed text is ASCII.
TOK_SPM, TOK_BPE, TOK_WORDPIECE = 0, 1, 2
TF_PRIO32 = 0x08
TF_LOWERCASE = 0x02
TOKBLOCK_SIZE = 36
NO_TOKEN = 0xFFFFFFFF

# Image file = the 256-byte header, then the payload, which loads at
# H_LOAD_BASE.  For 8 MB images H_LOAD_BASE is $8000100, so the file is
# simply a memory image of $8000000 onwards.

# Runtime block (written by the CPU and by the script)
R_POS = 0x40          # next position to process
R_PROMPT_LEN = 0x44   # tokens[0..prompt_len) are the prompt
R_N_GENERATE = 0x48   # maximum tokens to generate in this run
R_GENERATED = 0x4C    # tokens generated so far in this run
R_STOP_REQUEST = 0x50 # CPU writes non-zero to stop after the current token
R_STOP_REASON = 0x54  # 1 count reached, 2 EOS, 3 context full, 4 stop request

STOP_COUNT, STOP_EOS, STOP_CTX_FULL, STOP_REQUESTED = 1, 2, 3, 4

# ATTN parameter block: 8 x u32 (last two reserved, 0)
ATTN_BLOCK_FIELDS = ("k_cache", "v_cache", "n_heads", "n_kv_heads", "head_dim", "kv_format")
KV_F32, KV_F16 = 0, 1


def reg_addr(areg, offset=0):
    """Address operand meaning A[areg] + offset."""
    assert 0 <= areg < 16 and 0 <= offset < (1 << 24)
    return 0x80000000 | (areg << 24) | offset


def encode(op, a=0, b=0, x=0, y=0, z=0):
    return struct.pack("<BBBBIII", op, a & 0xFF, b & 0xFF, 0,
                       x & 0xFFFFFFFF, y & 0xFFFFFFFF, z & 0xFFFFFFFF)


def decode(raw):
    op, a, b, _c, x, y, z = struct.unpack("<BBBBIII", raw)
    return op, a, b, x, y, z


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]
