/*
  KLOPAPIER: a crude 80-column spreadsheet for the MEGA65 (a sheet for
  SHITE), on SSNAIL's
  spreadsheet engine (SHITE, opcodes $40-$47: mega65-core's docs/shite).
  Everything lives in the first 8 MB of HyperRAM (SSNAIL's $8000000, which
  is where the CPU sees it too with $D7FE bit 4 clear), so it runs on an R3
  as well as on boards with SDRAM.

  Keys:
    cursor keys     move              HOME     to A1
    typing          starts an entry   RETURN   edits the cell (or enters it)
    ESC             cancels an entry  DEL      deletes a character
    F1              25 / 50 rows      F3 / F5  the column narrower / wider
    F7              recalculates      Shift+HOME (CLR) clears the cell
  An entry starting with = is a formula (=A1*2, =SUM(A1:A9), =IF(B2>0,1,2));
  anything else a number (19.99 is kept exact, with two decimals) or text.

  The CPU keeps the sheet in memory as the engine expects (row table, row
  blocks of 16-byte cells) and asks SSNAIL for the rest: COMPILE and PARSE
  for an entry, REBALANCE_ROW after a change, RECALC until nothing changes,
  FORMAT_CELLS for the screen, DECOMPILE and RENDER for the edit line.
*/
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <mega65/memory.h>

/* A small sprintf of our own (as ssnail_chat.c): %s %c %u %X, l, width, 0, - */
static void fmt(char *o, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  while (*f) {
    char t[12], *q;
    uint8_t left = 0, zero = 0, lng = 0, w = 0, n;
    uint32_t v;
    if (*f != '%') { *o++ = *f++; continue; }
    f++;
    if (*f == '-') { left = 1; f++; }
    if (*f == '0') { zero = 1; f++; }
    while (*f >= '0' && *f <= '9') w = w * 10 + (*f++ - '0');
    if (*f == 'l') { lng = 1; f++; }
    if (*f == 's' || *f == 'c') {
      const char *str;
      char c1[2];
      if (*f == 's') str = va_arg(ap, const char *);
      else { c1[0] = (char)va_arg(ap, int); c1[1] = 0; str = c1; }
      n = (uint8_t)strlen(str);
      if (!left) while (w > n) { *o++ = ' '; w--; }
      while (*str) *o++ = *str++;
      while (w > n) { *o++ = ' '; w--; }
      f++;
      continue;
    }
    v = lng ? va_arg(ap, uint32_t) : (uint16_t)va_arg(ap, unsigned int);
    q = t + sizeof(t);
    *--q = 0;
    if (*f == 'X') { do { *--q = "0123456789ABCDEF"[v & 15]; v >>= 4; } while (v); }
    else { do { *--q = '0' + (char)(v % 10); v /= 10; } while (v); }
    n = (uint8_t)(t + sizeof(t) - 1 - q);
    if (!left) while (w > n) { *o++ = zero ? '0' : ' '; w--; }
    while (*q) *o++ = *q++;
    while (w > n) { *o++ = ' '; w--; }
    f++;
  }
  *o = 0;
  va_end(ap);
}

/* --- where things are ------------------------------------------------------- */
#define SN 0xFFD7500UL          /* SSNAIL's registers */
#define D7FE 0xFFD37FEUL
#define HR 0x8000000UL          /* HyperRAM: SSNAIL's and the CPU's $8000000 */
#define PROG (HR + 0x0000)      /* the SSNAIL program of the moment */
#define RESULT (HR + 0x0800)    /* its results (STR) */
#define PBLK (HR + 0x0900)      /* a parameter block */
#define DESC (HR + 0x1000)      /* the sheet descriptor */
#define CL1 (HR + 0x1100)       /* the change lists (64 entries each) */
#define CL2 (HR + 0x1300)
#define NAMES (HR + 0x1800)     /* function names */
#define WIDTHS (HR + 0x1C00)    /* the visible columns' widths, for FORMAT_CELLS */
#define SCRATCH (HR + 0x2000)   /* 1 KB for the renderer */
#define TXT (HR + 0x2800)       /* an entry's text */
#define OUTB (HR + 0x2900)      /* the codec's output */
#define SCRBUF (HR + 0x3000)    /* the screen, rendered (80 x 50) */
#define COLBUF (HR + 0x4000)    /* its colours */
#define ROWTAB (HR + 0x10000UL) /* NROWS x 4 */
#define HEAP (HR + 0x20000UL)   /* row blocks */
#define HEAP_END (HR + 0x800000UL)
#define NROWS 1000
#define NCOLS 64
#define SCR 0x50000UL           /* the visible screen (chip RAM, bank 5) */
#define COL 0xFF80000UL         /* colour RAM */

/* cell kinds and flags (README 3.4) */
#define K_EMPTY 0
#define K_TEXT 1
#define K_FORMULA 2
#define K_F64 3
#define K_I64 4
#define K_ERROR 6
#define K_UNCOMPILED 7
#define K_FREE 15
#define F_CHANGED 0x10
#define F_HAS_FORMAT 0x40

/* SSNAIL's opcodes */
#define OP_HALT 0x01
#define OP_LI 0x08
#define OP_STR 0x0A
#define OP_ADDI 0x0B
#define OP_BEQ 0x0D
#define OP_BLT 0x0F
#define OP_RECALC 0x40
#define OP_REBALANCE 0x41
#define OP_FINDCELL 0x42
#define OP_FORMATCELLS 0x43
#define OP_COMPILE 0x44
#define OP_DECOMPILE 0x45
#define OP_RENDER 0x46
#define OP_PARSE 0x47
#define RZERO 15

static uint8_t rows = 50;               /* 25 or 50 */
static uint8_t widths[NCOLS];
static uint16_t top, left;              /* the first visible row, column */
static uint16_t cur_r, cur_c;           /* the cursor's cell */
static uint32_t heap = HEAP;
static char line[80];                   /* the edit line */
static char msg[81];
static uint8_t blk[64];                 /* a cell or parameter block, built here */

/* --- memory ----------------------------------------------------------------- */
static uint32_t rd32(uint32_t a)
{
  return lpeek(a) | ((uint32_t)lpeek(a + 1) << 8) | ((uint32_t)lpeek(a + 2) << 16)
         | ((uint32_t)lpeek(a + 3) << 24);
}
static uint16_t rd16(uint32_t a) { return lpeek(a) | ((uint16_t)lpeek(a + 1) << 8); }
static void wr32(uint32_t a, uint32_t v)
{
  lpoke(a, v); lpoke(a + 1, v >> 8); lpoke(a + 2, v >> 16); lpoke(a + 3, v >> 24);
}
static void wr16(uint32_t a, uint16_t v) { lpoke(a, v); lpoke(a + 1, v >> 8); }
static void put32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void put16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }

/* A DMA does not carry into the next megabyte: copies are split there */
static uint32_t mb_left(uint32_t a) { return 0x100000UL - (a & 0xFFFFFUL); }
static void xcopy(uint32_t from, uint32_t to, uint32_t n)
{
  while (n) {
    uint32_t k = n;
    if (k > 0x8000) k = 0x8000;
    if (k > mb_left(from)) k = mb_left(from);
    if (k > mb_left(to)) k = mb_left(to);
    lcopy(from, to, (uint16_t)k);
    from += k; to += k; n -= k;
  }
}
static void to_ram(uint32_t a, const void *p, uint16_t n) { xcopy((uint32_t)(uint16_t)p, a, n); }
static void from_ram(void *p, uint32_t a, uint16_t n) { xcopy(a, (uint32_t)(uint16_t)p, n); }

/* --- the screen ----------------------------------------------------------- */
static uint8_t sc(uint8_t c)            /* ASCII -> screen code (lower-case set) */
{
  if (c >= 'a' && c <= 'z') return c - 96;
  if (c >= 'A' && c <= 'Z') return c;
  if (c == '@') return 0;
  if (c >= '[' && c <= '_') return c - 64;
  if (c < 32 || c > 126) return '?';
  return c;
}
static uint8_t asc(uint8_t s)           /* and back (the renderer's numbers) */
{
  s &= 0x7F;
  if (s >= 1 && s <= 26) return s + 96;
  if (s == 0) return '@';
  return s;
}

static void put_at(uint8_t x, uint8_t y, uint8_t colour, const char *s)
{
  uint32_t a = SCR + x + y * 80;
  while (*s && x < 80) {
    lpoke(a, sc((uint8_t)*s++));
    lpoke(COL + (a - SCR), colour);
    a++; x++;
  }
}
static void clear_line(uint8_t y) { lfill(SCR + y * 80, ' ', 80); }
static void status(uint8_t colour, const char *s) { clear_line(rows - 1); put_at(0, rows - 1, colour, s); }

static void screen_setup(void)
{
  POKE(0xD05D, PEEK(0xD05D) | 0x80);
  POKE(0xD018, 0x16);
  POKE(0xD054, 0x00);
  POKE(0xD031, rows == 50 ? 0xE8 : 0xE0);
  POKE(0xD016, 0xC9);
  POKE(0xD05D, PEEK(0xD05D) & 0x7F);
  POKE(0xD058, 80); POKE(0xD059, 0);
  POKE(0xD05E, 80);
  POKE(0xD060, SCR & 0xFF); POKE(0xD061, (SCR >> 8) & 0xFF); POKE(0xD062, (SCR >> 16) & 0xFF);
  POKE(0xD063, (PEEK(0xD063) & 0xF0) | ((SCR >> 24) & 0x0F));
  POKE(0xD07B, rows);
  POKE(0xD020, 0); POKE(0xD021, 0);
  lfill(SCR, ' ', 4000);
  lfill(COL, 1, 4000);
}

/* --- SSNAIL jobs ------------------------------------------------------------ */
static uint8_t prog[256];
static uint8_t np;                      /* bytes of prog */

static void ins(uint8_t op, uint8_t a, uint8_t b, uint8_t c, uint32_t x, uint32_t z)
{
  uint8_t *p = prog + np;
  memset(p, 0, 16);
  p[0] = op; p[1] = a; p[2] = b; p[3] = c;
  put32(p + 4, x);
  put32(p + 12, z);
  np += 16;
}
static uint32_t here(void) { return PROG + np; }

/* the program in prog, run (HALT added): SSNAIL's error code (0: none) */
static uint8_t run(void)
{
  uint16_t spin = 0;
  ins(OP_HALT, 0, 0, 0, 0, 0);
  to_ram(PROG, prog, np);
  np = 0;
  lpoke(SN + 8, PROG & 0xFF); lpoke(SN + 9, (PROG >> 8) & 0xFF);
  lpoke(SN + 10, (PROG >> 16) & 0xFF); lpoke(SN + 11, (PROG >> 24) & 0xFF);
  lpoke(SN + 3, 1);
  while (lpeek(SN + 4) & 1) {           /* (busy: the border flickers) */
    if (!++spin) { POKE(0xD020, 1); POKE(0xD020, 0); }
  }
  return lpeek(SN + 5);
}
static void job_error(uint8_t e)
{
  fmt(msg, "SSNAIL error %u at $%lX", e, rd32(SN + 12) & 0xFFFFFFFUL);
  status(2, msg);
}

/* one instruction with R1 = a, R2 = b, then R1 stored: R1 back */
static uint32_t job_r1(uint8_t op, uint32_t a, uint32_t b, uint32_t x)
{
  uint8_t e;
  ins(OP_LI, 1, 0, 0, 0, a);
  ins(OP_LI, 2, 0, 0, 0, b);
  ins(op, 1, 2, 0, x, 0);
  ins(OP_STR, 1, 0, 0, RESULT, 0);
  e = run();
  if (e) { job_error(e); return 0; }
  return rd32(RESULT);
}

static uint32_t find_cell(uint16_t r, uint16_t c) { return job_r1(OP_FINDCELL, r, c, DESC); }
static void rebalance(uint16_t r) { job_r1(OP_REBALANCE, r, r, DESC); }

/* RECALC rounds until one changes nothing (at most 32): rounds taken, or 0
   if it still changes (a cycle, most likely) */
static uint8_t recalc(void)
{
  uint32_t loop, done;
  uint8_t e;
  ins(OP_LI, 2, 0, 0, 0, 0xFFFFFFFFUL);   /* last row, */
  ins(OP_LI, 3, 0, 0, 0, 0);              /* columns 0 .. */
  ins(OP_LI, 4, 0, 0, 0, 0xFFFFFFFFUL);
  ins(OP_LI, 6, 0, 0, 0, 0);              /* rounds */
  ins(OP_LI, 7, 0, 0, 0, 32);
  loop = here();
  ins(OP_LI, 5, 0, 0, 0, 0);              /* first row; then the count */
  ins(OP_RECALC, 5, 2, 3 | 4 << 4, DESC, 3);   /* (BEGIN, END) */
  ins(OP_ADDI, 6, 6, 0, 0, 1);
  done = here() + 32;
  ins(OP_BEQ, 5, RZERO, 0, done, 0);
  ins(OP_BLT, 6, 7, 0, loop, 0);
  ins(OP_STR, 5, 0, 0, RESULT, 0);        /* (done:) */
  ins(OP_STR, 6, 0, 0, RESULT + 4, 0);
  e = run();
  if (e) { job_error(e); return 0; }
  return rd32(RESULT) ? 0 : (uint8_t)rd32(RESULT + 4);
}

/* --- the sheet in memory ----------------------------------------------------- */
static const char *const fn_names[] = {
  "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "PRODUCT", "ABS", "INT", "TRUNC",
  "ROUND", "SIGN", "SQRT", "MOD", "AND", "OR", "NOT", "IF", "EXP", "LN", "LOG10", "SIN",
  "COS", "TAN", "ASIN", "ACOS", "ATAN", "ATAN2", "PI", "POWER", "LEN", "LEFT", "RIGHT",
  "MID", "UPPER", "LOWER", "TRIM", "VALUE", "TEXT", "CONCATENATE", "SUMIF", "COUNTIF",
  "VLOOKUP", "HLOOKUP", "MATCH", "INDEX"
};

static void sheet_setup(void)
{
  uint32_t a = NAMES;
  uint8_t i;
  /* the function names: [identifier][name, NUL] ..., then an empty name */
  for (i = 0; i < sizeof(fn_names) / sizeof(fn_names[0]); i++) {
    const char *n = fn_names[i];
    lpoke(a++, i);
    while (*n) lpoke(a++, *n++);
    lpoke(a++, 0);
  }
  lpoke(a++, 0); lpoke(a++, 0);
  /* the descriptor (README 3.1) */
  memset(blk, 0, 64);
  memcpy(blk, "SHT1", 4);
  put32(blk + 4, ROWTAB); put32(blk + 8, NROWS); put32(blk + 12, NCOLS);
  put32(blk + 16, CL1); put32(blk + 20, 64); put32(blk + 24, CL2);
  put32(blk + 56, NAMES); blk[60] = '.'; blk[61] = ',';
  to_ram(DESC, blk, 64);
  lfill(ROWTAB, 0xFF, NROWS * 4);         /* every row empty */
  heap = HEAP;
  memset(widths, 9, sizeof(widths));
}

static uint32_t row_block(uint16_t r)
{
  uint32_t b = rd32(ROWTAB + 4UL * r);
  return b == 0xFFFFFFFFUL ? 0 : b;
}

/* a new row block of len granules (header written), or 0 */
static uint32_t new_block(uint16_t r, uint16_t len)
{
  uint32_t b = heap;
  if (b + 16UL * len > HEAP_END) return 0;
  heap += 16UL * len;
  memset(blk, 0, 16);
  put32(blk, r); put16(blk + 4, len); put16(blk + 10, 1); put16(blk + 14, 1);   /* (free: 1; stale) */
  to_ram(b, blk, 16);
  return b;
}

/* room for need granules in row r's block (grown, its live cells copied,
   if it is too small): the block, or 0 */
static uint32_t room(uint16_t r, uint8_t need)
{
  uint32_t b = row_block(r), nb;
  uint16_t len, free, g, nfree, nlen;
  if (!b) {
    b = new_block(r, 64);
    if (b) wr32(ROWTAB + 4UL * r, b);
    return b;
  }
  len = rd16(b + 4); free = rd16(b + 10);
  if (free + need <= len) return b;
  nlen = len * 2;
  if (nlen < free + need + 16) nlen = free + need + 16;
  if (nlen > 4096) nlen = 4096;
  nb = new_block(r, nlen);
  if (!nb) return 0;
  nfree = 1;
  for (g = 1; g < free; ) {             /* the live cells, compacted */
    uint8_t kind = lpeek(b + 16UL * g + 3) & 15, n = lpeek(b + 16UL * g + 4);
    if (!n) break;
    if (kind != K_FREE) { xcopy(b + 16UL * g, nb + 16UL * nfree, 16UL * n); nfree += n; }
    g += n;
  }
  if (nfree + need > nlen) return 0;
  wr16(nb + 10, nfree);
  wr32(ROWTAB + 4UL * r, nb);
  rebalance(r);                         /* (the cells moved: the tree rebuilt) */
  return nb;
}

/* cell (r, c) replaced by a new one: kind and flags, its body (after the
   header, n bytes from p); 0 if the row is full */
static uint8_t put_cell(uint16_t r, uint16_t c, uint8_t kf, const uint8_t *p, uint16_t n)
{
  uint8_t need = (uint8_t)((8 + n + 15) / 16);
  uint32_t old, b, a;
  uint16_t free;
  b = room(r, need);                    /* (first: growing moves the cells) */
  if (!b) { status(2, "the row is full"); return 0; }
  old = find_cell(r, c);
  if (old) lpoke(old + 3, K_FREE);      /* (the old cell: a hole) */
  free = rd16(b + 10);
  a = b + 16UL * free;
  {
    uint8_t h[8];
    h[0] = c; h[1] = c >> 8; h[2] = 0; h[3] = kf | F_CHANGED; h[4] = need; h[5] = h[6] = h[7] = 0;
    to_ram(a, h, 8);
  }
  if (n) to_ram(a + 8, p, n);
  wr16(b + 10, free + need);
  lpoke(b + 14, lpeek(b + 14) | 1);     /* (stale: rebalanced now) */
  rebalance(r);
  return 1;
}

static void clear_cell(uint16_t r, uint16_t c)
{
  uint32_t old = find_cell(r, c);
  if (old) { lpoke(old + 3, K_FREE); rebalance(r); }
}

/* --- entries: COMPILE, PARSE ----------------------------------------------- */
static const char *const codec_errors[] = {
  "", "syntax", "unknown function", "bad number", "bad cell reference", "wrong argument count",
  "nested too deeply", "too long", "bad bytecode", "bad name table", "memory", "aborted"
};

static uint8_t enter(const char *t)
{
  uint16_t n = (uint16_t)strlen(t);
  uint8_t e;
  if (!n) { clear_cell(cur_r, cur_c); return 1; }
  if (t[0] == '=') {                    /* a formula: COMPILE */
    to_ram(TXT, t + 1, n - 1);
    memset(blk, 0, 32);
    put32(blk, DESC); put32(blk + 4, TXT); put16(blk + 8, n - 1); put16(blk + 10, 512);
    put32(blk + 12, cur_r); put32(blk + 16, cur_c); put32(blk + 20, OUTB);
    to_ram(PBLK, blk, 32);
    ins(OP_COMPILE, 1, 0, 0, PBLK, 0);
    e = run();
    if (e) { job_error(e); return 0; }
    from_ram(blk, PBLK + 24, 8);
    if (blk[2]) {                       /* the codec's error: kept as UNCOMPILED */
      uint8_t code = blk[2];
      uint16_t pos = blk[4] | (blk[5] << 8), keep = n > 48 ? 48 : n;
      fmt(msg, "%s at %u", code < 12 ? codec_errors[code] : "error", pos + 1);
      memset(blk, 0, 16);
      blk[8] = code; put16(blk + 10, pos); put16(blk + 12, keep);
      memcpy(blk + 16, t, keep);
      if (!put_cell(cur_r, cur_c, K_UNCOMPILED, blk, 16 + keep)) return 0;
      status(2, msg);
      return 1;
    } else {
      uint16_t len = blk[0] | (blk[1] << 8);
      static uint8_t cell[24 + 512];
      memset(cell, 0, 24);
      put16(cell + 18, len);            /* (format, result, info: rk, cap, length) */
      from_ram(cell + 24, OUTB, len);
      return put_cell(cur_r, cur_c, K_FORMULA, cell, 24 + len);
    }
  }
  /* a value: PARSE (a number), else text */
  to_ram(TXT, t, n);
  memset(blk, 0, 32);
  put32(blk, DESC); put32(blk + 4, TXT); put16(blk + 8, n);
  to_ram(PBLK, blk, 32);
  ins(OP_PARSE, 1, 0, 0, PBLK, 0);
  e = run();
  if (e) { job_error(e); return 0; }
  from_ram(blk, PBLK + 16, 16);
  if (blk[8] == K_I64 && blk[9]) {      /* fixed point: shown with its decimals */
    blk[16] = 1; blk[17] = blk[9]; blk[18] = blk[9];
    memset(blk + 19, 0, 5);
    memcpy(blk + 8, blk + 16, 8);
    return put_cell(cur_r, cur_c, K_I64 | F_HAS_FORMAT, blk, 16);
  }
  if (blk[8] == K_I64 || blk[8] == K_F64) return put_cell(cur_r, cur_c, blk[8], blk, 8);
  {
    static uint8_t cell[10 + 80];
    memset(cell, 0, 10);
    put16(cell + 8, n);
    memcpy(cell + 10, t, n);
    return put_cell(cur_r, cur_c, K_TEXT, cell, 10 + n);
  }
}

/* --- the edit line: a cell's contents as text ------------------------------ */
static void col_name(char *o, uint16_t c)
{
  if (c >= 26) { *o++ = 'A' + c / 26 - 1; }
  *o++ = 'A' + c % 26;
  *o = 0;
}

static void contents(char *o)
{
  uint32_t a = find_cell(cur_r, cur_c);
  uint8_t kind, e;
  *o = 0;
  if (!a) return;
  kind = lpeek(a + 3) & 15;
  if (kind == K_TEXT) {
    uint16_t n = rd16(a + 16);
    if (n > 76) n = 76;
    from_ram(o, a + 18, n); o[n] = 0;
  } else if (kind == K_UNCOMPILED) {
    uint16_t n = rd16(a + 20);
    if (n > 48) n = 48;
    from_ram(o, a + 24, n); o[n] = 0;
  } else if (kind == K_FORMULA) {       /* DECOMPILE */
    memset(blk, 0, 32);
    put32(blk, DESC); put32(blk + 4, a + 32); put16(blk + 8, rd16(a + 26)); put16(blk + 10, 76);
    put32(blk + 12, cur_r); put32(blk + 16, cur_c); put32(blk + 20, OUTB);
    to_ram(PBLK, blk, 32);
    ins(OP_DECOMPILE, 1, 0, 0, PBLK, 0);
    e = run();
    if (e) { job_error(e); return; }
    {
      uint16_t n = rd16(PBLK + 24);
      if (n > 76) n = 76;
      o[0] = '=';
      from_ram(o + 1, OUTB, n); o[n + 1] = 0;
    }
  } else if (kind == K_I64 || kind == K_F64) {   /* RENDER, then back to ASCII */
    uint8_t i, k = 0, w = 30;
    memset(blk, 0, 32);
    put32(blk, DESC);
    blk[4] = kind;
    from_ram(blk + 8, a + 8, 8);
    if (lpeek(a + 3) & F_HAS_FORMAT) k = lpeek(a + 18);
    blk[5] = k; blk[6] = w; blk[7] = 1;
    if (k) { blk[16] = 1; blk[17] = k; }  /* (all its decimals) */
    put32(blk + 24, OUTB); put32(blk + 28, SCRATCH);
    to_ram(PBLK, blk, 32);
    ins(OP_RENDER, 1, 0, 0, PBLK, 0);
    e = run();
    if (e) { job_error(e); return; }
    for (i = 0; i < w && lpeek(OUTB + i) == ' '; i++) ;
    for (; i < w; i++) *o++ = asc(lpeek(OUTB + i));
    *o = 0;
  }
}

/* --- drawing ------------------------------------------------------------------ */
static uint8_t grid(void) { return rows - 3; }   /* rows of cells: lines 2 .. rows - 2 */

/* the visible columns from `left`: how many, their widths in WIDTHS (the last one cut) */
static uint8_t visible(uint8_t *w_out)
{
  uint8_t x = 4, n = 0;
  while (left + n < NCOLS && x < 80) {
    uint8_t w = widths[left + n];
    if (x + w > 80) w = 80 - x;
    w_out[n++] = w;
    x += w;
  }
  return n;
}

static void draw(void)
{
  static uint8_t w[NCOLS];
  uint8_t n = visible(w), i, y, x, e, cx = 0, cw = 0;
  char t[8];
  /* the cells: FORMAT_CELLS into the buffer, then copied to the screen */
  lfill(SCRBUF, ' ', 80 * 50);
  lfill(COLBUF, 1, 80 * 50);
  to_ram(WIDTHS, w, n);
  memset(blk, 0, 48);
  put32(blk, DESC); put32(blk + 4, top); put32(blk + 8, top + grid() - 1);
  put32(blk + 12, left); put32(blk + 16, left + n - 1); put32(blk + 20, WIDTHS);
  put32(blk + 24, SCRBUF + 2 * 80 + 4); put16(blk + 28, 80); blk[30] = 1; blk[31] = 0;
  put32(blk + 32, COLBUF + 2 * 80 + 4); put16(blk + 36, 80); blk[38] = 1;
  put32(blk + 40, SCRATCH);
  to_ram(PBLK, blk, 48);
  ins(OP_FORMATCELLS, 0, 0, 0, PBLK, 0);
  e = run();
  if (e) job_error(e);
  xcopy(SCRBUF + 2 * 80, SCR + 2 * 80, 80UL * grid());
  xcopy(COLBUF + 2 * 80, COL + 2 * 80, 80UL * grid());
  /* the column letters and row numbers */
  lfill(SCR + 80, ' ' | 0x80, 80); lfill(COL + 80, 14, 80);
  for (x = 4, i = 0; i < n; i++) {
    col_name(t, left + i);
    put_at(x + (w[i] - (uint8_t)strlen(t)) / 2, 1, 14, t);
    if (left + i == cur_c) { cx = x; cw = w[i]; }
    x += w[i];
  }
  for (x = 0; x < 80; x++) lpoke(SCR + 80 + x, lpeek(SCR + 80 + x) | 0x80);
  for (y = 0; y < grid(); y++) {
    fmt(t, "%4u", top + y + 1);
    put_at(0, 2 + y, 14, t);
    for (x = 0; x < 4; x++) lpoke(SCR + (2 + y) * 80 + x, lpeek(SCR + (2 + y) * 80 + x) | 0x80);
  }
  /* the cursor: its cell in reverse */
  if (cw && cur_r >= top && cur_r < top + grid()) {
    uint32_t a = SCR + (2 + cur_r - top) * 80 + cx;
    for (x = 0; x < cw; x++) lpoke(a + x, lpeek(a + x) ^ 0x80);
  }
}

static void show_line(const char *s, uint8_t editing)
{
  char t[8];
  clear_line(0);
  col_name(t, cur_c);
  fmt(msg, "%s%u", t, cur_r + 1);
  put_at(0, 0, 7, msg);
  put_at(8, 0, editing ? 1 : 15, s);
  if (editing) lpoke(SCR + 8 + strlen(s), 0xA0);   /* (a block cursor) */
}

/* keep the cursor on the screen: its row, and its column wholly */
static void follow(void)
{
  if (cur_r < top) top = cur_r;
  if (cur_r >= top + grid()) top = cur_r - grid() + 1;
  if (cur_c < left) left = cur_c;
  for (;;) {
    uint16_t x = 4, c;
    for (c = left; c <= cur_c; c++) x += widths[c];
    if (x <= 80 || left == cur_c) break;
    left++;
  }
}

static void refresh(void)
{
  follow();
  draw();
  contents(line);
  show_line(line, 0);
}

/* --- the keys ----------------------------------------------------------------- */
static uint8_t key(void)
{
  uint8_t c;
  while (!(c = PEEK(0xD610))) ;
  POKE(0xD610, 0);
  return c;
}

/* an entry, from its first key (0: start from the cell's contents) */
static void edit(uint8_t first)
{
  uint8_t n, c, r;
  if (first) { line[0] = first; line[1] = 0; } else contents(line);
  n = (uint8_t)strlen(line);
  for (;;) {
    show_line(line, 1);
    c = key();
    if (c == 13) break;
    if (c == 27) { refresh(); return; }
    if ((c == 20 || c == 8) && n) line[--n] = 0;
    else if (c >= 32 && c < 127 && n < 70) { line[n++] = c; line[n] = 0; }
  }
  status(1, "");
  if (enter(line)) {
    r = recalc();
    if (!r) status(2, "still changing after 32 rounds: a circular reference?");
    if (cur_r + 1 < NROWS) cur_r++;
  }
  refresh();
}

int main(void)
{
  uint8_t c;
  POKE(0xD020, 2);
  __asm__ volatile("sei");
  mega65_io_enable();
  screen_setup();
  if (lpeek(SN) != 0x53 || !(lpeek(SN + 2) & 8)) {
    put_at(0, 0, 2, "This bitstream has no SHITE (SSNAIL's spreadsheet engine).");
    for (;;) ;
  }
  /* SSNAIL: HyperRAM at $8000000 (8 MB), no SDRAM region; the CPU likewise */
  lpoke(SN + 0x10, 8); lpoke(SN + 0x11, 0); lpoke(SN + 0x12, 0); lpoke(SN + 0x13, 0x80);
  lpoke(D7FE, (lpeek(D7FE) & 0xE0) | 0x0D);
  sheet_setup();
  refresh();
  status(5, "KLOPAPIER: type into a cell; = for a formula; F1 rows, F3/F5 width, F7 recalc");
  for (;;) {
    c = key();
    switch (c) {
      case 0x11: if (cur_r + 1 < NROWS) cur_r++; refresh(); break;     /* down */
      case 0x91: if (cur_r) cur_r--; refresh(); break;                 /* up */
      case 0x1D: if (cur_c + 1 < NCOLS) cur_c++; refresh(); break;     /* right */
      case 0x9D: if (cur_c) cur_c--; refresh(); break;                 /* left */
      case 0x13: cur_r = cur_c = 0; refresh(); break;                  /* HOME */
      case 0x93: clear_cell(cur_r, cur_c); recalc(); refresh(); break; /* CLR */
      case 13: edit(0); break;
      case 0xF1: rows = rows == 50 ? 25 : 50; screen_setup(); refresh(); break;
      case 0xF3: if (widths[cur_c] > 1) widths[cur_c]--; refresh(); break;
      case 0xF5: if (widths[cur_c] < 40) widths[cur_c]++; refresh(); break;
      case 0xF7:
        fmt(msg, "recalculated in %u rounds", recalc());
        refresh();
        status(5, msg);
        break;
      default:
        if (c >= 32 && c < 127) edit(c);
    }
  }
}
