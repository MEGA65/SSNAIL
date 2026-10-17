/*
  SSNAIL image tokenizer in C (see ssnail_tokc.h; the reference is
  ssnail_tok.py, which documents the algorithms and the table format).

  One difference in method, none in result: _merge in the reference probes
  every adjacent pair on every round; here each pair's best token is kept,
  and after a merge only the two pairs touching it are probed again.  The
  choice each round (lowest priority, leftmost on ties) is the same.
*/
#include "ssnail_tokc.h"

#define H_VOCAB 0x18
#define H_TOKENIZER 0x78
#define TF_PRIO32 0x08
#define TF_LOWERCASE 0x02

#define MAXT 160             /* tokens in one chunk (a typed line, + 1) */
#define MAXW MAXT            /* bytes in one WordPiece word (a whole line) */

static uint8_t kind, flags;
static uint32_t vocab, a_index, n_index, a_prio, a_bytes, t_begin, t_end, t_unk;

static uint32_t rd32(uint32_t a)
{
  uint8_t b[4];
  tm_read(a, b, 4);
  return b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

void tok_init(uint32_t base)
{
  uint8_t h[36];
  uint32_t tb = rd32(base + H_TOKENIZER);
  uint8_t i;
  uint32_t *f[8];
  vocab = rd32(base + H_VOCAB);
  tm_read(tb, h, 36);
  kind = h[0];
  flags = h[1];
  f[0] = &a_index; f[1] = &n_index; f[2] = &a_prio; f[3] = &a_bytes;
  f[4] = &t_begin; f[5] = &t_end; f[6] = &t_unk;
  for (i = 0; i < 7; i++) {
    uint8_t *p = h + 8 + 4 * i;
    *f[i] = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  }
}

uint8_t tok_kind(void) { return kind; }

uint8_t tok_piece(uint32_t t, uint8_t *buf)
{
  uint32_t o0 = rd32(vocab + 4 * t);
  uint32_t o1 = rd32(vocab + 4 * t + 4);
  uint32_t n = o1 - o0;
  if (n > 255) n = 255;
  tm_read(vocab + o0, buf, (uint8_t)n);
  return (uint8_t)n;
}

static uint32_t prio(uint32_t t)
{
  uint8_t b[2];
  if (flags & TF_PRIO32) return rd32(a_prio + 4 * t);
  tm_read(a_prio + 2 * t, b, 2);
  if (b[0] == 0xFF && b[1] == 0xFF) return NO_TOKEN;
  return b[0] | ((uint16_t)b[1] << 8);
}

/* Binary search of the sorted index for the piece key[0..n); token or NO_TOKEN */
static uint32_t find(const uint8_t *key, uint8_t n)
{
  static uint8_t p[255];
  int32_t lo = 0, hi = (int32_t)n_index - 1;
  while (lo <= hi) {
    int32_t mid = (lo + hi) >> 1;
    uint32_t t = rd32(a_index + 4 * (uint32_t)mid);
    uint8_t m = tok_piece(t, p), i = 0;
    int8_t c = 0;
    while (i < m && i < n) {
      if (p[i] != key[i]) { c = p[i] < key[i] ? -1 : 1; break; }
      i++;
    }
    if (c == 0) {
      if (m == n) return t;
      c = m < n ? -1 : 1;
    }
    if (c < 0) lo = mid + 1; else hi = mid - 1;
  }
  return NO_TOKEN;
}

/* --- SPM / BPE: merge a chunk's bytes ----------------------------------- */
static uint32_t toks[MAXT];
static uint32_t pr_t[MAXT], pr_p[MAXT];   /* pair i = (toks[i], toks[i+1]) */
static uint8_t kbuf[255];

static void probe(uint16_t i)
{
  uint8_t a = tok_piece(toks[i], kbuf), b;
  static uint8_t tmp[255];
  pr_t[i] = NO_TOKEN;
  pr_p[i] = NO_TOKEN;
  b = tok_piece(toks[i + 1], tmp);
  if ((uint16_t)a + b > 255) return;
  {
    uint8_t j;
    for (j = 0; j < b; j++) kbuf[a + j] = tmp[j];
  }
  pr_t[i] = find(kbuf, a + b);
  if (pr_t[i] != NO_TOKEN) pr_p[i] = prio(pr_t[i]);
}

static uint16_t merge(const uint8_t *s, uint16_t n, uint32_t *out, uint16_t max)
{
  uint16_t m = 0, i;
  for (i = 0; i < n && m < MAXT; i++) {
    uint32_t t = rd32(a_bytes + 4 * (uint32_t)s[i]);
    if (t != NO_TOKEN) toks[m++] = t;
  }
  for (i = 0; i + 1 < m; i++) probe(i);
  while (m > 1) {
    uint16_t bi = 0xFFFF;
    uint32_t bp = NO_TOKEN;
    for (i = 0; i + 1 < m; i++)
      if (pr_p[i] != NO_TOKEN && (bi == 0xFFFF || pr_p[i] < bp)) { bp = pr_p[i]; bi = i; }
    if (bi == 0xFFFF) break;
    toks[bi] = pr_t[bi];
    for (i = bi + 1; i + 1 < m; i++) {        /* drop toks[bi + 1] and its pair */
      toks[i] = toks[i + 1];
      pr_t[i] = pr_t[i + 1];
      pr_p[i] = pr_p[i + 1];
    }
    m--;
    if (bi > 0) probe(bi - 1);
    if (bi + 1 < m) probe(bi);
  }
  for (i = 0; i < m && i < max; i++) out[i] = toks[i];
  return i;
}

/* --- GPT-2's pre-split, ASCII form (gpt2_chunks) ------------------------ */
static uint8_t is_alpha(uint8_t c) { return (c >= 65 && c <= 90) || (c >= 97 && c <= 122); }
static uint8_t is_digit(uint8_t c) { return c >= 48 && c <= 57; }
static uint8_t is_space(uint8_t c) { return c == 32 || (c >= 9 && c <= 13); }
static uint8_t is_other(uint8_t c) { return !(is_alpha(c) || is_digit(c) || is_space(c)); }
static uint8_t cls_of(uint8_t k, uint8_t c)
{
  return k == 0 ? is_alpha(c) : k == 1 ? is_digit(c) : is_other(c);
}

static const char *contr[] = { "s", "t", "re", "ve", "m", "ll", "d" };

static uint16_t bpe(const uint8_t *d, uint16_t n, uint32_t *out, uint16_t max)
{
  uint16_t i = 0, got = 0;
  while (i < n && got < max) {
    uint16_t j, k = i;
    uint8_t c, done = 0;
    if (d[i] == 39) {                         /* ' contractions */
      for (c = 0; c < 7; c++) {
        const char *s = contr[c];
        uint8_t l = 0;
        while (s[l] && i + 1 + l < n && d[i + 1 + l] == (uint8_t)s[l]) l++;
        if (!s[l]) {
          got += merge(d + i, 1 + l, out + got, max - got);
          i += 1 + l;
          done = 1;
          break;
        }
      }
      if (done) continue;
    }
    j = (d[i] == 32 && i + 1 < n && !is_space(d[i + 1])) ? i + 1 : i;
    for (c = 0; c < 3; c++) {
      if (j < n && cls_of(c, d[j])) {
        k = j;
        while (k < n && cls_of(c, d[k])) k++;
        done = 1;
        break;
      }
    }
    if (!done) {                              /* whitespace; one space left for the next word */
      k = i;
      while (k < n && is_space(d[k])) k++;
      if (k < n && k - i > 1 && d[k - 1] == 32) k--;
    }
    got += merge(d + i, k - i, out + got, max - got);
    i = k;
  }
  return got;
}

/* --- WordPiece ------------------------------------------------------------ */
static uint8_t is_wchar(uint8_t c) { return is_alpha(c) || is_digit(c) || c == '_'; }

static uint16_t wordpiece(const uint8_t *d, uint16_t n, uint32_t *out, uint16_t max)
{
  static uint8_t w[MAXW + 1];
  uint16_t i = 0, got = 0;
  if (t_begin != NO_TOKEN && got < max) out[got++] = t_begin;
  while (i < n) {
    uint16_t wl = 0, start;
    if (is_wchar(d[i])) {
      while (i < n && is_wchar(d[i])) { if (wl < MAXW) w[wl++] = d[i]; i++; }
    } else if (is_space(d[i])) {
      i++;
      continue;
    } else {
      w[wl++] = d[i++];
    }
    start = 0;
    {
      uint16_t g0 = got;
      while (start < wl) {
        uint16_t end = wl;
        uint32_t t = NO_TOKEN;
        while (end > start) {
          uint16_t l = 0, q;
          if (start == 0) kbuf[l++] = ' ';
          for (q = start; q < end; q++) kbuf[l++] = w[q];
          t = find(kbuf, (uint8_t)l);
          if (t != NO_TOKEN) break;
          end--;
        }
        if (end == start) { got = g0; if (got < max) out[got++] = t_unk; break; }
        if (got < max) out[got++] = t;
        start = end;
      }
    }
  }
  if (t_end != NO_TOKEN && got < max) out[got++] = t_end;
  return got;
}

uint16_t tok_encode(const char *text, uint8_t first, uint32_t *out, uint16_t max)
{
  static uint8_t d[MAXT];
  uint16_t n = 0, got = 0;
  const uint8_t *s = (const uint8_t *)text;
  if (kind == TOK_SPM) d[n++] = ' ';
  for (; *s && n < MAXT; s++) {
    uint8_t c = *s;
    if (c > 127) c = '?';
    if (kind == TOK_WORDPIECE && (flags & TF_LOWERCASE) && c >= 'A' && c <= 'Z') c += 32;
    d[n++] = c;
  }
  if (kind == TOK_WORDPIECE) return wordpiece(d, n, out, max);
  if (first && t_begin != NO_TOKEN && got < max) out[got++] = t_begin;
  if (kind == TOK_SPM) return got + merge(d, n, out + got, max - got);
  return got + bpe(d, n, out + got, max - got);
}

/* --- Display folding (AsciiFolder) ----------------------------------------- */
static uint8_t f_need;
static uint32_t f_cp;
static const char latin1[] = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYTsaaaaaaaceeeeiiiidnooooo/ouuuuyty";
static const struct { uint16_t cp; const char *s; } fold_tab[] = {
  {0x2018, "'"}, {0x2019, "'"}, {0x201A, "'"}, {0x201B, "'"}, {0x2032, "'"},
  {0x201C, "\""}, {0x201D, "\""}, {0x201E, "\""}, {0x2033, "\""}, {0x00AB, "\""}, {0x00BB, "\""},
  {0x2010, "-"}, {0x2011, "-"}, {0x2012, "-"}, {0x2013, "-"}, {0x2014, "-"}, {0x2015, "-"},
  {0x2212, "-"}, {0x2026, "..."}, {0x00A0, " "}, {0x00D7, "x"}, {0x00DF, "ss"},
  {0x00E6, "ae"}, {0x00C6, "AE"}, {0x0153, "oe"}, {0x0152, "OE"}, {0x00F8, "o"}, {0x00D8, "O"},
  {0x2022, "*"}, {0x00B7, "."}, {0x2122, "(TM)"}, {0x00A9, "(c)"}, {0x00AE, "(R)"},
};

static char *put_cp(uint32_t cp, char *o)
{
  uint8_t i;
  if (cp == 9 || cp == 10 || (cp >= 32 && cp < 127)) { *o++ = (char)cp; return o; }
  for (i = 0; i < sizeof(fold_tab) / sizeof(fold_tab[0]); i++)
    if (fold_tab[i].cp == cp) {
      const char *s = fold_tab[i].s;
      while (*s) *o++ = *s++;
      return o;
    }
  if (cp >= 0xC0 && cp <= 0xFF) { *o++ = latin1[cp - 0xC0]; return o; }
  *o++ = '?';
  return o;
}

void fold_reset(void) { f_need = 0; f_cp = 0; }

void fold_feed(const uint8_t *b, uint8_t n, char *out)
{
  uint8_t i;
  for (i = 0; i < n; i++) {
    uint8_t c = b[i];
    if (f_need) {
      if ((c & 0xC0) == 0x80) {
        f_cp = (f_cp << 6) | (c & 0x3F);
        if (--f_need == 0) out = put_cp(f_cp, out);
        continue;
      }
      *out++ = '?';
      f_need = 0;
    }
    if (c < 0x80) out = put_cp(c, out);
    else if ((c & 0xE0) == 0xC0) { f_cp = c & 0x1F; f_need = 1; }
    else if ((c & 0xF0) == 0xE0) { f_cp = c & 0x0F; f_need = 2; }
    else if ((c & 0xF8) == 0xF0) { f_cp = c & 0x07; f_need = 3; }
    else *out++ = '?';
  }
  *out = 0;
}
