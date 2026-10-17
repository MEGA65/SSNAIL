/*
  SSNAIL chat for the MEGA65: picks a .ssnail image from the SD card's
  /SSNAIL folder (or the root), streams it into SSNAIL's RAM through the load
  port, then chats: typed text is tokenized on the MEGA65 with the image's
  own tables (ssnail_tokc.c), SSNAIL runs the image's script, and each new
  token is printed as it arrives.  The protocol is the README's "Running a
  generation"; the tokenizer's reference is ssnail_tok.py.

  Nothing happens until asked, one step at a time:
    /regs        SSNAIL's registers, and $D7FE
    /scan        find the models on the SD card (/models lists them again)
    /hdr N       model N's header fields, from the file
    /load N [direct|port|slow]   load it: direct (default) = DMA straight
                 into SSNAIL's RAM at $8000000+ through the CPU's view of SDRAM
                 (and of HyperRAM, for a 72 MB image's last 8 MB);
                 port = DMA into the load port (destination held); slow = the
                 load port, byte by byte
    /check       the loaded header and first payload bytes read back from
                 $8000000 (by lpeek and by DMA), against the file
    /tok TEXT    tokenize TEXT with the loaded image's tables, and back
    /step        one token: the script with n_generate = 1, and what it did
    /new         new story (context cleared)    /len N  reply length
    /temp T      sampling temperature (0 = greedy, the default; 0.7-1.0 for stories)
    /seed [N]    a repeatable seed, or (no N) the moment Return is pressed
    /rows 25|50  the screen's height
    /more        (or an empty line) carry on generating where the reply
                 stopped: after a key stopped it, or at /len tokens
  Then plain text chats; a chat model's text is wrapped in its chat template
  (the image's, header $6C).  While SSNAIL generates (the border flickers),
  any key asks it to stop after the current token; a second key aborts it
  outright.  /more then carries on.  Shift + HOME clears the screen.
*/
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <mega65/memory.h>
#include <mega65/fileio.h>
#include <mega65/dirent.h>
#include <mega65/targets.h>
#include "ssnail_tokc.h"

void do_dma(void);              /* (mega65-libc's, not in its header) */

/* A small sprintf of our own (the library's broke the screen here):
   %s %c %u %X, with l, width, 0 and - flags and a precision for %s */
static void fmt(char *o, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  while (*f) {
    char t[12], *q;
    uint8_t left = 0, zero = 0, lng = 0, w = 0, prec = 255, n;
    uint32_t v;
    if (*f != '%') { *o++ = *f++; continue; }
    f++;
    if (*f == '-') { left = 1; f++; }
    if (*f == '0') { zero = 1; f++; }
    while (*f >= '0' && *f <= '9') w = w * 10 + (*f++ - '0');
    if (*f == '.') { prec = 0; f++; while (*f >= '0' && *f <= '9') prec = prec * 10 + (*f++ - '0'); }
    if (*f == 'l') { lng = 1; f++; }
    if (*f == 's' || *f == 'c') {
      const char *str;
      char c1[2];
      if (*f == 's') str = va_arg(ap, const char *);
      else { c1[0] = (char)va_arg(ap, int); c1[1] = 0; str = c1; }
      n = 0;
      while (str[n] && n < prec) n++;
      if (!left) while (w > n) { *o++ = ' '; w--; }
      { uint8_t i; for (i = 0; i < n; i++) *o++ = str[i]; }
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
#define sprintf fmt

static uint32_t parse_u(const char *s)
{
  uint32_t v = 0;
  while (*s == ' ') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return v;
}

#define SN 0xFFD7500UL          /* SSNAIL's registers */
#define D7FE 0xFFD37FEUL
#define IMG 0x8000000UL         /* every image starts here */

/* header and runtime block (ssnail_isa.py) */
#define H_CODE 0x08
#define H_TOKENS 0x14
#define H_MAXCTX 0x20
#define H_PAYLOAD_LEN 0x2C
#define H_MEM_NEEDED 0x38
#define H_ARCH 0x3C
#define H_LOAD_BASE 0x64
#define H_DESC 0x80
#define R_POS 0x40
#define R_PROMPT_LEN 0x44
#define R_N_GENERATE 0x48
#define R_GENERATED 0x4C
#define R_STOP_REQUEST 0x50
#define R_STOP_REASON 0x54
#define R_SAMPLE 0x58
#define R_NOISE 0x5C
#define H_OUTDIM 0x34
#define H_TVEC 0x68
#define H_CHAT 0x6C
#define NOISE_SPAN 4096

#define MAXMODELS 64
#define NAMELEN 64

struct model {
  char name[NAMELEN];
  uint8_t in_dir;                /* 1: /SSNAIL, 0: the root */
  uint32_t mem;
  uint8_t arch;
  char desc[40];
};
static struct model models[MAXMODELS];
static uint8_t n_models;
static int8_t cur = -1;
static uint32_t avail;
static uint8_t sector[512];
static uint8_t tvec_ok;                 /* the image's T vector holds the temperature */
static char buf[4 * 255 + 1];          /* a token's text, folded (one user at a time) */

/* --- screen: an 80 x 50 text screen, output scrolling above the input line */
/* (in bank 5, written only by DMA: the CPU's $E000-$FFFF is the ROM's) */
#define SCR 0x50000UL
#define COL 0xFF80000UL
#define TOP 3
static uint8_t rows = 50;       /* 25 or 50 (/rows) */
static uint8_t BOT = 46;        /* last output row */
static uint8_t INROW = 48;      /* the input line (a rule above it) */
static uint8_t ox, oy = TOP;

static uint8_t sc(uint8_t c)    /* ASCII -> screen code (lower-case set) */
{
  if (c >= 'a' && c <= 'z') return c - 96;
  if (c >= 'A' && c <= 'Z') return c;
  if (c == '@') return 0;
  if (c >= '[' && c <= '_') return c - 64;
  if (c == '{') return '(';
  if (c == '}') return ')';
  if (c == '~') return '-';
  if (c == '`') return 39;
  if (c == '|') return 93;
  if (c < 32 || c > 126) return '?';
  return c;
}

static void put_at(uint8_t x, uint8_t y, uint8_t colour, const char *s)
{
  uint32_t a = SCR + x + y * 80;
  while (*s && x < 80) {
    lpoke(a, sc((uint8_t)*s++));
    lpoke(COL + (a - SCR), colour);
    a++;
    x++;
  }
}

static void clear_rows(uint8_t y0, uint8_t y1)
{
  lfill(SCR + y0 * 80, ' ', (y1 - y0 + 1) * 80);
}

static uint8_t colour = 1;

static void out_newline(void)
{
  ox = 0;
  if (oy < BOT) { oy++; return; }
  lcopy(SCR + (TOP + 1) * 80, SCR + TOP * 80, (BOT - TOP) * 80);
  lcopy(COL + (TOP + 1) * 80, COL + TOP * 80, (BOT - TOP) * 80);
  clear_rows(BOT, BOT);
}

static void out_char(char c)
{
  if (c == '\n') { out_newline(); return; }
  if (c == '\t') c = ' ';
  if (ox >= 80) out_newline();
  lpoke(SCR + ox + oy * 80, sc((uint8_t)c));
  lpoke(COL + ox + oy * 80, colour);
  ox++;
}

static void out_str(const char *s) { while (*s) out_char(*s++); }

/* word wrap: break before a word that would cross the right edge */
static void out_text(const char *s)
{
  while (*s) {
    if (*s != ' ' && *s != '\n' && ox > 0) {
      const char *e = s;
      uint8_t l = 0;
      while (*e && *e != ' ' && *e != '\n') { e++; l++; }
      if (ox + l > 80 && l < 80) out_newline();
    }
    out_char(*s++);
  }
}

/* messages go inline, in the scrolling output; row 1 shows the loaded
   model (and a load's progress) */
static void status(uint8_t c, const char *s)
{
  uint8_t keep = colour;
  colour = c;
  out_text(s);
  out_newline();
  colour = keep;
}

static void row1(uint8_t c, const char *s)
{
  clear_rows(1, 1);
  put_at(0, 1, c, s);
}

static void screen_setup(void)
{
  /* The VIC-II / III registers with hot registers on, so the VIC-IV applies
     them (V400 is what sets 25 or 50 rows); then hot registers off, so that
     nothing undoes the VIC-IV settings written after them */
  POKE(0xD05D, PEEK(0xD05D) | 0x80);
  POKE(0xD018, 0x16);           /* lower case */
  POKE(0xD054, 0x00);
  POKE(0xD031, rows == 50 ? 0xE8 : 0xE0);  /* H640, fast CPU (and V400: 50 rows) */
  POKE(0xD016, 0xC9);
  POKE(0xD05D, PEEK(0xD05D) & 0x7F);
  POKE(0xD058, 80); POKE(0xD059, 0);
  POKE(0xD05E, 80);
  POKE(0xD060, SCR & 0xFF); POKE(0xD061, (SCR >> 8) & 0xFF); POKE(0xD062, (SCR >> 16) & 0xFF);
  POKE(0xD063, (PEEK(0xD063) & 0xF0) | ((SCR >> 24) & 0x0F));
  POKE(0xD07B, rows);
  BOT = rows - 4;
  INROW = rows - 2;
  POKE(0xD021, 0);
  POKE(0xD020, 8);
  lfill(SCR, ' ', 4000);
  lfill(COL, 1, 4000);
  POKE(0xD020, 4);
  put_at(0, 0, 7, "SSNAIL chat");
  lfill(SCR + (INROW - 1) * 80, 64 + 128, 80);   /* (a rule above the input line) */
  lfill(COL + (INROW - 1) * 80, 11, 80);
  ox = 0;
  oy = TOP;
}

/* --- SSNAIL ------------------------------------------------------------- */
static uint32_t rd32(uint32_t a)
{
  return lpeek(a) | ((uint32_t)lpeek(a + 1) << 8) | ((uint32_t)lpeek(a + 2) << 16)
         | ((uint32_t)lpeek(a + 3) << 24);
}

/* A DMA's address does not carry into its megabyte (it wraps within 1 MB),
   so a transfer that would cross a 1 MB boundary is done in two */
static uint16_t mb_room(uint32_t a, uint16_t n)
{
  uint32_t left = 0x100000UL - (a & 0xFFFFFUL);
  return left < n ? (uint16_t)left : n;
}

void tm_read(uint32_t a, uint8_t *dst, uint8_t n)
{
  while (n) {
    uint8_t k = (uint8_t)mb_room(a, n);
    lcopy(a, (uint32_t)(uint16_t)dst, k);
    a += k; dst += k; n -= k;
  }
}

/* The CPU sees SSNAIL's $8000000 RAM (SDRAM on R4-R6) at $8000000, with its
   attic caches and prefetch off; SSNAIL's region registers at their defaults */
static uint8_t ssnail_setup(void)
{
  uint8_t sd;
  if (lpeek(SN) != 0x53) return 0;
  sd = lpeek(SN + 2) & 2;
  lpoke(SN + 0x10, 8);
  lpoke(SN + 0x11, sd ? 0x80 : 0);
  lpoke(SN + 0x12, sd ? 64 : 0);
  lpoke(SN + 0x13, sd ? 0xC0 : 0x80);
  lpoke(D7FE, (lpeek(D7FE) & 0xE2) | (sd ? 0x10 : 0));
  avail = sd ? 72UL << 20 : 8UL << 20;
  return 1;
}

/* load port: commit a new pointer (bit 6 clears ERROR) */
static void lp_wait(void) { while (!(lpeek(SN + 0x17) & 0x80)) ; }

static void lp_commit(uint32_t a, uint8_t clr)
{
  lp_wait();
  lpoke(SN + 0x14, a & 0xFF);
  lpoke(SN + 0x15, (a >> 8) & 0xFF);
  lpoke(SN + 0x16, (a >> 16) & 0xFF);
  lpoke(SN + 0x17, ((a >> 24) & 0x0F) | (clr ? 0x40 : 0));
}

/* n bytes from local memory into the data register (destination held) */
static void lp_dma(const uint8_t *src, uint16_t n)
{
  if (!n) return;
  lp_wait();
  dmalist.option_0b = 0x0b;
  dmalist.option_80 = 0x80;
  dmalist.source_mb = 0;
  dmalist.option_81 = 0x81;
  dmalist.dest_mb = 0xFF;
  dmalist.option_85 = 0x85;
  dmalist.dest_skip = 0;                /* (hold the destination) */
  dmalist.end_of_options = 0;
  dmalist.sub_cmd = 0;
  dmalist.command = 0;                  /* copy */
  dmalist.count = n;
  dmalist.source_addr = (uint16_t)src;
  dmalist.source_bank = 0;
  dmalist.dest_addr = (SN + 0x18) & 0xFFFF;
  dmalist.dest_bank = ((SN + 0x18) >> 16) & 0x0F;
  do_dma();
}

/* the same, byte by byte through the CPU (no DMA): slow but plain */
static void lp_bytes(const uint8_t *src, uint16_t n)
{
  uint16_t i;
  lp_wait();
  for (i = 0; i < n; i++) lpoke(SN + 0x18, src[i]);
}

static uint8_t lp_error(void) { lp_wait(); return lpeek(SN + 0x17) & 0x40; }

/* bytes into SSNAIL's RAM through the load port (and flushed) */
static void lp_write(uint32_t a, const uint8_t *src, uint16_t n)
{
  lp_commit(a, 0);
  lp_dma(src, n);
  lp_commit(a + n, 0);
}

static void lp_write32(uint32_t a, uint32_t v)
{
  uint8_t b[4];
  b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24;
  lp_write(a, b, 4);
}

static uint8_t busy(void) { return lpeek(SN + 4) & 1; }

/* a flicker of thin white lines in the border while something is happening */
static void flick(void) { POKE(0xD020, 1); POKE(0xD020, 0); }

static uint8_t image_there(void)
{
  if (lpeek(IMG) == 'S' && lpeek(IMG + 1) == 'S' && lpeek(IMG + 2) == 'N' && lpeek(IMG + 3) == 'L')
    return 1;
  status(2, "no image header (SSNL) at $8000000: /load the model again");
  return 0;
}

static void go(uint32_t pc)
{
  lpoke(SN + 8, pc & 0xFF);
  lpoke(SN + 9, (pc >> 8) & 0xFF);
  lpoke(SN + 10, (pc >> 16) & 0xFF);
  lpoke(SN + 11, (pc >> 24) & 0xFF);
  lpoke(SN + 3, 1);
}

/* frames, for timing: $D7FA counts video frames */
static uint32_t frames;
static uint8_t last_frame;
static void tick(void)
{
  uint8_t f = lpeek(0xFFD37FAUL);
  frames += (uint8_t)(f - last_frame);
  last_frame = f;
}
static uint8_t fps(void) { return (lpeek(0xFFD306FUL) & 0x80) ? 60 : 50; }

/* --- models on the SD card ---------------------------------------------- */
static uint32_t le32(const uint8_t *p)
{
  return p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void scan(uint8_t in_dir)
{
  uint8_t d;
  struct m65_dirent *e;
  chdirroot();
  if (in_dir && chdir("SSNAIL") == 0xFF) return;
  d = opendir();
  if (d == 0xFF) return;
  while ((e = readdir(d)) && n_models < MAXMODELS) {
    uint8_t l = strlen(e->d_name);
    struct model *m = &models[n_models];
    if (l < 8 || l >= NAMELEN) continue;
    if (strcmp(e->d_name + l - 7, ".ssnail") && strcmp(e->d_name + l - 7, ".SSNAIL")) continue;
    strcpy(m->name, e->d_name);
    m->in_dir = in_dir;
    n_models++;
  }
  closedir(d);
}

/* header of each: size, architecture, description; drop what isn't one */
static void inspect(void)
{
  uint8_t i, k = 0;
  for (i = 0; i < n_models; i++) {
    struct model *m = &models[i];
    uint8_t fd;
    chdirroot();
    if (m->in_dir) chdir("SSNAIL");
    fd = open(m->name);
    if (fd == 0xFF) continue;
    read512(sector);
    close(fd);
    if (sector[0] != 'S' || sector[1] != 'S' || sector[2] != 'N' || sector[3] != 'L') continue;
    m->mem = le32(sector + H_MEM_NEEDED);
    m->arch = sector[H_ARCH];
    strncpy(m->desc, (char *)sector + H_DESC, sizeof(m->desc) - 1);
    m->desc[sizeof(m->desc) - 1] = 0;
    if (k != i) models[k] = *m;
    k++;
  }
  n_models = k;
}

static void list_models(void)
{
  uint8_t i;
  char line[100];
  colour = 3;
  for (i = 0; i < n_models; i++) {
    struct model *m = &models[i];
    sprintf(line, "%2u %-44.44s %3lu MB %s%s\n", i + 1, m->name, (m->mem + 0xFFFFF) >> 20,
            m->arch == 2 ? "(encoder) " : "", m->mem > avail ? "(too big)" : "");
    out_str(line);
    if ((i + 1) % 40 == 0 && i + 1 < n_models) {      /* a page at a time */
      colour = 7;
      out_str("-- a key for more --");
      while (!PEEK(0xD610)) ;
      POKE(0xD610, 0);
      ox = 0;
      clear_rows(oy, oy);
      colour = 3;
    }
  }
  colour = 1;
}

/* stream the file: the 256-byte header to $8000000, the rest to the load
   base (H_LOAD_BASE), one sector at a time by DMA */
#define LOAD_DIRECT 0
#define LOAD_PORT 1
#define LOAD_SLOW 2
static uint8_t load_mode;
static uint32_t dst;                    /* (direct: where the next bytes go) */
/* direct: SSNAIL's $8000000-$BFFFFFF is SDRAM, which the CPU sees at the same
   addresses with $D7FE bit 4 set; its $C000000-$C7FFFFF is HyperRAM, which
   the CPU sees at $8000000 with bit 4 clear */
static void direct(const uint8_t *src, uint16_t n)
{
  uint8_t d7 = lpeek(D7FE);
  while (n) {
    uint16_t k = mb_room(dst, n);
    if (dst < 0xC000000UL) {
      if (dst + k > 0xC000000UL) k = 0xC000000UL - dst;
      lpoke(D7FE, d7 | 0x10);
      lcopy((uint32_t)(uint16_t)src, dst, k);
    } else {
      lpoke(D7FE, d7 & ~0x10);
      lcopy((uint32_t)(uint16_t)src, IMG + (dst - 0xC000000UL), k);
    }
    src += k; dst += k; n -= k;
  }
  lpoke(D7FE, d7);
}

static void lp_send(const uint8_t *src, uint16_t n)
{
  if (load_mode == LOAD_DIRECT) direct(src, n);
  else if (load_mode == LOAD_SLOW) lp_bytes(src, n);
  else lp_dma(src, n);
}
static void lp_to(uint32_t a, uint8_t clr)
{
  if (load_mode == LOAD_DIRECT) dst = a; else lp_commit(a, clr);
}

static uint8_t load(uint8_t i)
{
  struct model *m = &models[i];
  uint8_t fd;
  uint16_t n, first = 1;
  uint32_t total = 0, len, lb;
  char msg[90];
  chdirroot();
  if (m->in_dir) chdir("SSNAIL");
  fd = open(m->name);
  if (fd == 0xFF) { status(2, "can't open the file"); return 0; }
  n = read512(sector);
  if (n < 256) { close(fd); status(2, "not an image"); return 0; }
  lb = le32(sector + H_LOAD_BASE);
  len = 256 + le32(sector + H_PAYLOAD_LEN);
  if (load_mode == LOAD_DIRECT && (lpeek(SN + 2) & 2) == 0) {
    close(fd); status(2, "direct needs SDRAM at $8000000: /load N port"); return 0;
  }
  if (load_mode == LOAD_DIRECT && lb + len > 0xC800000UL) {
    close(fd); status(2, "past the end of HyperRAM ($C800000)"); return 0;
  }
  lp_to(IMG, 1);
  for (;;) {
    if (first) {
      lp_send(sector, 256);
      lp_to(lb, 0);
      lp_send(sector + 256, n - 256);
      first = 0;
    } else {
      lp_send(sector, n);
    }
    total += n;
    if ((total & 0x3FFFF) < 512 || total >= len) {
      sprintf(msg, "loading %s: %lu of %lu KB", m->name, total >> 10, len >> 10);
      row1(7, msg);
    }
    if (total >= len) break;
    flick();
    n = read512(sector);
    if (!n) break;
  }
  close(fd);
  POKE(0xD020, 0);
  if (load_mode != LOAD_DIRECT) {
    lp_commit(lb, 0);                   /* (flush) */
    if (lp_error()) { status(2, "load port error (data outside SSNAIL's RAM)"); return 0; }
  }
  cur = i;
  tvec_ok = 0;
  tok_init(IMG);
  sprintf(msg, "%s: %s", m->name, m->desc);
  row1(5, msg);
  return 1;
}

/* --- chat --------------------------------------------------------------- */
#define MAXIN 96
#define MAXIDS (MAXIN + 34)             /* typed text, and a chat template's wrapping */
static uint32_t ids[MAXIDS];
static uint8_t fresh = 1;               /* next prompt starts the context */
static uint32_t pos, n_gen = 120;
/* sampling: T in thousandths (0 = greedy); the T vector is filled once per
   load; the seed is /seed's, or the moment Return was pressed */
static uint16_t temp_milli;
static int32_t seed = -1;
static uint16_t key_time;

/* m / 1000 as F32 bits, rounded to nearest (integer long division: no float
   library, which would leave no room for the stack); m > 0 */
static uint32_t milli_to_f32(uint16_t m)
{
  uint32_t num = m, den = 1000, q = 0;
  int8_t e = 0, i;
  while (num >= 2 * den) { den <<= 1; e++; }
  while (num < den) { num <<= 1; e--; }
  for (i = 0; i < 24; i++) {            /* den <= num < 2 den: 24 bits of num / den */
    q <<= 1;
    if (num >= den) { num -= den; q |= 1; }
    num <<= 1;
  }
  if (num >= den) {                     /* round half up */
    q++;
    if (q >> 24) { q >>= 1; e++; }
  }
  return ((uint32_t)(127 + e) << 23) | (q & 0x7FFFFFUL);
}

static uint8_t fill_tvec(void)
{
  static uint8_t b[512];
  uint32_t tv = rd32(IMG + H_TVEC), n = rd32(IMG + H_OUTDIM) * 4;
  uint32_t u;
  uint16_t i;
  if (!tv) { status(2, "this image has no sampling: rebuild it with the current converter"); return 0; }
  u = milli_to_f32(temp_milli);
  for (i = 0; i < 512; i += 4) {
    b[i] = u; b[i + 1] = u >> 8; b[i + 2] = u >> 16; b[i + 3] = u >> 24;
  }
  dst = tv;                             /* (direct DMA, split at 1 MB boundaries) */
  while (n) {
    uint16_t k = n > 512 ? 512 : (uint16_t)n;
    direct(b, k);
    n -= k;
  }
  tvec_ok = 1;
  return 1;
}

static void reply(const char *text)
{
  uint32_t tokens = rd32(IMG + H_TOKENS), ctx = rd32(IMG + H_MAXCTX);
  uint32_t at, plen, got = 0, g, t0;
  uint16_t n, i;
  uint8_t stop_sent = 0;
  char msg[80];
  static uint8_t piece[255];

  if (!image_there()) return;

  if (!*text) {                         /* continue: no new tokens */
    if (fresh) { status(2, "nothing to continue: type something first"); return; }
    n = 0;
  } else if (rd32(IMG + H_CHAT)) {      /* a chat model: the template's wrapping */
    uint32_t c = rd32(IMG + H_CHAT);
    uint16_t npre = lpeek(c) | (lpeek(c + 1) << 8), npost = lpeek(c + 2) | (lpeek(c + 3) << 8);
    if (npre + npost > MAXIDS - 8) { status(2, "the chat template is too long for this client"); return; }
    for (i = 0; i < npre; i++) ids[i] = rd32(c + 4 + 4 * i);
    n = npre + tok_encode(text, 0, ids + npre, MAXIDS - npre - npost);
    for (i = 0; i < npost; i++) ids[n + i] = rd32(c + 4 + 4 * (npre + i));
    n += npost;
  } else {
    n = tok_encode(text, fresh, ids, MAXIDS);
  }
  at = fresh ? 0 : pos + 1;
  if (at + n + 1 >= ctx) { status(2, "context full: /new starts again"); return; }
  for (i = 0; i < n; i++) lp_write32(tokens + 4 * (at + i), ids[i]);
  plen = at + n;
  if (fresh) lp_write32(IMG + R_POS, 0);
  lp_write32(IMG + R_PROMPT_LEN, plen);
  lp_write32(IMG + R_N_GENERATE, n_gen);
  lp_write32(IMG + R_STOP_REQUEST, 0);
  if (temp_milli && (tvec_ok || fill_tvec())) {
    lp_write32(IMG + R_SAMPLE, 1);
    lp_write32(IMG + R_NOISE, (uint32_t)(seed >= 0 ? (uint32_t)seed : key_time) % NOISE_SPAN);
  } else if (rd32(IMG + H_TVEC)) {
    lp_write32(IMG + R_SAMPLE, 0);
  }
  if (fresh) fold_reset();
  fresh = 0;
  colour = 13;
  tick(); t0 = frames;
  go(rd32(IMG + H_CODE));
  for (;;) {
    uint8_t b = busy();
    tick();
    g = rd32(IMG + R_GENERATED);
    while (got < g) {
      uint32_t t = rd32(tokens + 4 * (plen + got));
      uint8_t l = tok_piece(t, piece);
      fold_feed(piece, l, buf);
      out_text(buf);
      got++;
    }
    if (!b) break;
    flick();
    if (PEEK(0xD610)) {                 /* a key: stop after this token; */
      POKE(0xD610, 0);
      if (!stop_sent) {
        lp_write32(IMG + R_STOP_REQUEST, 1);
        stop_sent = 1;
        status(7, "(stopping after this token: a key again aborts)");
      } else {                          /* a second: abort SSNAIL */
        lpoke(SN + 3, 2);
        status(2, "aborted");
      }
    }
  }
  POKE(0xD020, 0);
  colour = 1;
  out_newline();
  pos = rd32(IMG + R_POS);
  if (lpeek(SN + 5)) {
    sprintf(msg, "SSNAIL stopped with error %u", lpeek(SN + 5));
    status(2, msg);
    fresh = 1;
    return;
  }
  {
    uint32_t f = frames - t0, r = rd32(IMG + R_STOP_REASON);
    uint32_t tps10 = f ? (got * 10UL * fps()) / f : 0;
    static const char *why[] = { "", "length", "end of text", "context full", "stopped" };
    sprintf(msg, "%lu tokens, %lu.%lu tok/s (%s); position %lu of %lu", got, tps10 / 10,
            tps10 % 10, r < 5 ? why[r] : "?", pos, ctx);
    status(5, msg);
    if (r == 3) fresh = 1;
  }
}

/* --- input line ------------------------------------------------------- */
static void read_line(char *buf, uint8_t max)
{
  uint8_t n = 0;
  clear_rows(INROW, INROW + 1);
  put_at(0, INROW, 7, "> ");
  for (;;) {
    uint8_t c = PEEK(0xD610);
    if (!c) continue;
    POKE(0xD610, 0);
    if (c == 13) { key_time = ((uint16_t)lpeek(0xFFD37FAUL) << 8) ^ PEEK(0xD012) ^ (n * 977u); break; }
    if (c == 0x93) {                    /* shift + HOME (CLR): clear the screen, as on a C64 */
      clear_rows(TOP, BOT);
      ox = 0; oy = TOP;
      continue;
    }
    if ((c == 20 || c == 8) && n) {
      n--;
      lpoke(SCR + INROW * 80 + 2 + n, ' ');
    } else if (c >= 32 && c < 127 && n < max - 1 && n < 77) {
      buf[n] = c;
      lpoke(SCR + INROW * 80 + 2 + n, sc(c));
      lpoke(COL + INROW * 80 + 2 + n, 1);
      n++;
    }
  }
  buf[n] = 0;
}


/* --- step-by-step checks ------------------------------------------- */
static void say(uint8_t c, const char *s) { colour = c; out_str(s); out_newline(); colour = 1; }

static void cmd_found(void)
{
  char m[90];
  uint8_t t = detect_target(), caps = lpeek(SN + 2);

  sprintf(m, "board: %s (target $%02X)", t == TARGET_MEGA65R6 ? "MEGA65 R6" : t == TARGET_MEGA65R5 ? "MEGA65 R5"
          : t == TARGET_MEGA65R4 ? "MEGA65 R4" : t == TARGET_MEGA65R3 ? "MEGA65 R3" : "other", t);
  say(3, m);

  sprintf(m, "SSNAIL v%u: HyperRAM port %s, SDRAM %s", lpeek(SN + 1),
          caps & 1 ? "yes" : "no", caps & 2 ? "PRESENT" : "not present");
  say(3, m);
  if (caps & 2)
    say(3, "map: SDRAM at $8000000 (64 MB, the CPU sees it there too), HyperRAM at $C000000 (8 MB)");
  else
    say(3, "map: HyperRAM at $8000000 (8 MB)");
  sprintf(m, "models up to %lu MB fit", avail >> 20);
  say(3, m);
}

static void cmd_regs(void)
{
  char m[90];
  uint8_t i;
  sprintf(m, "SSNAIL id $%02X version %u caps $%02X  status $%02X error %u  PC $%07lX",
          lpeek(SN), lpeek(SN + 1), lpeek(SN + 2), lpeek(SN + 4), lpeek(SN + 5),
          rd32(SN + 12) & 0xFFFFFFFUL);
  say(3, m);
  sprintf(m, "regions: HyperRAM %u MB at $%02X0 0000, SDRAM %u MB at $%02X0 0000;"
          "  load port $%02X  $D7FE = $%02X",
          lpeek(SN + 0x10), lpeek(SN + 0x13), lpeek(SN + 0x12), lpeek(SN + 0x11),
          lpeek(SN + 0x17), lpeek(D7FE));
  say(3, m);
  (void)i;
}

static uint8_t pick(const char *s)
{
  int k = 0;
  while (*s == ' ') s++;
  while (*s >= '0' && *s <= '9') k = k * 10 + (*s++ - '0');
  if (k < 1 || k > n_models) { say(2, "no such model: /scan lists them"); return 0xFF; }
  return k - 1;
}

static uint8_t read_header(uint8_t i)
{
  uint8_t fd;
  chdirroot();
  if (models[i].in_dir) chdir("SSNAIL");
  fd = open(models[i].name);
  if (fd == 0xFF) { say(2, "can't open the file"); return 0; }
  read512(sector);
  close(fd);
  return 1;
}

static void cmd_hdr(uint8_t i)
{
  char m[90];
  if (!read_header(i)) return;
  sprintf(m, "%s: magic %c%c%c%c  arch %u  code $%07lX  tokens $%07lX  vocab $%07lX",
          models[i].name, sector[0], sector[1], sector[2], sector[3], sector[H_ARCH],
          le32(sector + H_CODE), le32(sector + H_TOKENS), le32(sector + 0x18));
  say(3, m);
  sprintf(m, "  load base $%07lX  payload %lu bytes  memory to $%07lX  context %lu  tokenizer $%07lX",
          le32(sector + H_LOAD_BASE), le32(sector + H_PAYLOAD_LEN),
          IMG + le32(sector + H_MEM_NEEDED), le32(sector + H_MAXCTX), le32(sector + 0x78));
  say(3, m);
}

/* the image as loaded, read back two ways, against the file's first sector */
static void cmd_check(void)
{
  static uint8_t dm[256];
  char m[90];
  uint16_t i, bad_p = 0, bad_d = 0, bad_pp = 0, bad_pd = 0, first = 0xFFFF;
  uint32_t lb;
  if (cur < 0) { say(2, "nothing loaded"); return; }
  if (!read_header(cur)) return;
  lb = le32(sector + H_LOAD_BASE);
  lcopy(IMG, (uint32_t)(uint16_t)dm, 256);
  for (i = 0; i < 256; i++) {
    if (lpeek(IMG + i) != sector[i]) { bad_p++; if (first == 0xFFFF) first = i; }
    if (dm[i] != sector[i]) bad_d++;
  }
  lcopy(lb, (uint32_t)(uint16_t)dm, 256);
  for (i = 0; i < 256; i++) {
    if (lpeek(lb + i) != sector[256 + i]) bad_pp++;
    if (dm[i] != sector[256 + i]) bad_pd++;
  }
  sprintf(m, "header: %u bytes differ by lpeek, %u by DMA; payload's first 256: %u by lpeek, %u by DMA",
          bad_p, bad_d, bad_pp, bad_pd);
  say(bad_p | bad_d | bad_pp | bad_pd ? 2 : 5, m);
  if (first != 0xFFFF) {
    sprintf(m, "  first: +$%02X is $%02X, the file has $%02X", first, lpeek(IMG + first), sector[first]);
    say(2, m);
    sprintf(m, "  RAM:  ");
    for (i = 0; i < 16; i++) sprintf(m + 8 + 3 * i, "%02X ", lpeek(IMG + i));
    say(2, m);
    sprintf(m, "  file: ");
    for (i = 0; i < 16; i++) sprintf(m + 8 + 3 * i, "%02X ", sector[i]);
    say(2, m);
  }
}

/* the whole image, sector by sector, against the file */
static void cmd_verify(void)
{
  static uint8_t ram[512];
  char m[90];
  uint8_t fd;
  uint16_t n, i, firstsec = 1;
  uint32_t off = 0, a, len, lb, bad = 0, first_bad = 0;
  if (cur < 0) { say(2, "nothing loaded"); return; }
  chdirroot();
  if (models[cur].in_dir) chdir("SSNAIL");
  fd = open(models[cur].name);
  if (fd == 0xFF) { say(2, "can't open the file"); return; }
  say(7, "verifying the whole image against the file (as slow as a load)...");
  while ((n = read512(sector)) != 0) {
    if (firstsec) {
      lb = le32(sector + H_LOAD_BASE);
      len = 256 + le32(sector + H_PAYLOAD_LEN);
      (void)len;
    }
    for (i = 0; i < n; ) {                /* the file's bytes [off, off + n) */
      uint32_t fo = off + i;
      uint16_t k = n - i;
      a = fo < 256 ? IMG + fo : lb + (fo - 256);
      if (fo < 256 && fo + k > 256) k = 256 - (uint16_t)fo;
      k = mb_room(a, k);
      lcopy(a, (uint32_t)(uint16_t)ram, k);
      { uint16_t j; for (j = 0; j < k; j++) if (ram[j] != sector[i + j]) {
          if (!bad) first_bad = a + j;
          bad++;
          break;
        } }
      i += k;
    }
    off += n;
    firstsec = 0;
    flick();
  }
  close(fd);
  POKE(0xD020, 0);
  if (!bad) { sprintf(m, "all %lu bytes match the file", off); say(5, m); }
  else { sprintf(m, "%lu chunks differ; the first at $%07lX", bad, first_bad); say(2, m); }
}

static void cmd_tok(const char *text)
{
  static uint32_t t[64];
  static uint8_t piece[255];
  char m[90];
  uint16_t n, i;
  if (cur < 0) { say(2, "nothing loaded"); return; }
  n = tok_encode(text, 1, t, 64);
  colour = 3;
  for (i = 0; i < n; i++) { sprintf(m, "%lu ", t[i]); out_str(m); }
  out_newline();
  fold_reset();
  for (i = 0; i < n; i++) {
    uint8_t l = tok_piece(t[i], piece);
    fold_feed(piece, l, buf);
    out_char('[');
    out_str(buf);
    out_char(']');
  }
  out_newline();
  colour = 1;
}

/* one token, watched: prompt "Once", n_generate = 1 */
static void cmd_step(void)
{
  uint32_t tokens, t;
  uint16_t n, i;
  uint32_t spins = 0;
  char m[90];
  static uint8_t piece[255];
  if (cur < 0) { say(2, "nothing loaded"); return; }
  if (!image_there()) return;
  tokens = rd32(IMG + H_TOKENS);
  n = tok_encode("Once", 1, ids, 8);
  for (i = 0; i < n; i++) lp_write32(tokens + 4 * i, ids[i]);
  lp_write32(IMG + R_POS, 0);
  lp_write32(IMG + R_PROMPT_LEN, n);
  lp_write32(IMG + R_N_GENERATE, 1);
  lp_write32(IMG + R_STOP_REQUEST, 0);
  sprintf(m, "prompt: %u tokens at $%07lX; runtime block reads back pos %lu, prompt_len %lu, n_generate %lu",
          n, tokens, rd32(IMG + R_POS), rd32(IMG + R_PROMPT_LEN), rd32(IMG + R_N_GENERATE));
  say(3, m);
  go(rd32(IMG + H_CODE));
  while (busy() && ++spins < 300000UL) flick();
  POKE(0xD020, 0);
  if (busy()) { lpoke(SN + 3, 2); say(2, "still busy: aborted"); }
  sprintf(m, "after %lu polls: status $%02X error %u PC $%07lX; generated %lu, stop reason %lu, pos %lu",
          spins, lpeek(SN + 4), lpeek(SN + 5), rd32(SN + 12) & 0xFFFFFFFUL,
          rd32(IMG + R_GENERATED), rd32(IMG + R_STOP_REASON), rd32(IMG + R_POS));
  say(busy() || lpeek(SN + 5) ? 2 : 5, m);
  if (rd32(IMG + R_GENERATED)) {
    uint8_t l;
    t = rd32(tokens + 4 * n);
    l = tok_piece(t, piece);
    fold_reset();
    fold_feed(piece, l, buf);
    sprintf(m, "token %lu: [", t);
    colour = 3; out_str(m); out_str(buf); out_str("]"); out_newline(); colour = 1;
  }
  fresh = 1;
}

int main(void)
{
  char line[MAXIN];
  int8_t d;
  /* No KERNAL from here on (keys come from $D610, files from the
     hypervisor): its interrupt would otherwise run, and touch VIC and
     colour RAM, 60 times a second */
  /* (the border shows how far start-up got: red, yellow, orange, purple,
     then black once waiting for input) */
  POKE(0xD020, 2);
  __asm__ volatile("sei");
  mega65_io_enable();
  POKE(0xD020, 7);
  screen_setup();

  if (!ssnail_setup()) { status(2, "no SSNAIL in this bitstream"); for (;;) ; }

  cmd_found();
  cmd_regs();
   
  out_newline();
  say(7, "/scan lists the models on the SD card (in /SSNAIL, else the root).");
  say(7, "/load N [direct|port|slow] loads one (direct, the default, DMAs into SDRAM).");
  say(7, "Then /check (or /verify, the whole image), /tok TEXT and /step; then type to chat.");
  say(7, "While it writes, a key stops it after the current token (twice: at once);");
  say(7, "/more or an empty line carries on.  /temp T samples; Shift+HOME clears the screen.");
  say(7, "/hdr N shows a file's header, /regs SSNAIL's registers, /new a new story.");
  status(7, "ready");
  
  POKE(0xD020, 0);
  (void)d;
  for (;;) {
    read_line(line, sizeof(line));
    if (line[0] == '/') {
      if (!strncmp(line, "/more", 5)) {
        if (cur >= 0) reply("");
      } else if (!strncmp(line, "/models", 7)) list_models();
      else if (!strncmp(line, "/regs", 5)) cmd_regs();
      else if (!strncmp(line, "/rows", 5)) {
        rows = parse_u(line + 5) == 25 ? 25 : 50;
        screen_setup();
        say(7, "screen cleared: /help lists the commands");
      }
      else if (!strncmp(line, "/scan", 5)) {
        say(7, "scanning the SD card...");
        closeall();                     /* (a handle left open made every other scan fail) */
        n_models = 0;
        scan(1);
        if (!n_models) scan(0);         /* (chdir can't tell us /SSNAIL is missing) */
        inspect();
        if (!n_models) say(2, "no .ssnail files in /SSNAIL or the root");
        else list_models();
      } else if (!strncmp(line, "/hdr ", 5)) {
        uint8_t k = pick(line + 5);
        if (k != 0xFF) cmd_hdr(k);
      } else if (!strncmp(line, "/load ", 6)) {
        uint8_t k = pick(line + 6);
        load_mode = strstr(line, "slow") ? LOAD_SLOW : strstr(line, "port") ? LOAD_PORT : LOAD_DIRECT;
        if (k != 0xFF && models[k].mem > avail) say(2, "too big for this board's RAM");
        else if (k != 0xFF) {
          say(7, load_mode == LOAD_SLOW ? "loading through the load port, byte by byte..."
                 : load_mode == LOAD_PORT ? "loading by DMA into the load port..."
                 : "loading by DMA straight into SSNAIL's RAM at $8000000...");
          if (load(k)) say(5, "loaded: /check next");
          fresh = 1;
        }
      } else if (!strncmp(line, "/check", 6)) cmd_check();
      else if (!strncmp(line, "/verify", 7)) cmd_verify();
      else if (!strncmp(line, "/tok ", 5)) cmd_tok(line + 5);
      else if (!strncmp(line, "/step", 5)) cmd_step();
      else if (!strncmp(line, "/new", 4)) { fresh = 1; status(5, "new story"); }
      else if (!strncmp(line, "/temp", 5)) {
        const char *q = line + 5;
        uint16_t ip = 0, fr = 0, sc = 1000;
        char msg2[60];
        while (*q == ' ') q++;
        while (*q >= '0' && *q <= '9') ip = ip * 10 + (*q++ - '0');
        if (*q == '.') { q++; while (*q >= '0' && *q <= '9' && sc > 1) { sc /= 10; fr += (*q++ - '0') * sc; } }
        temp_milli = ip * 1000 + fr;
        if (temp_milli > 5000) temp_milli = 5000;
        tvec_ok = 0;
        if (temp_milli) sprintf(msg2, "temperature %u.%03u (sampling)", temp_milli / 1000, temp_milli % 1000);
        else sprintf(msg2, "temperature 0: greedy");
        say(5, msg2);
      } else if (!strncmp(line, "/seed", 5)) {
        char msg2[60];
        const char *q = line + 5;
        while (*q == ' ') q++;
        if (*q >= '0' && *q <= '9') { seed = (int32_t)(parse_u(q) % NOISE_SPAN); sprintf(msg2, "seed %lu (repeatable)", (uint32_t)seed); }
        else { seed = -1; sprintf(msg2, "seed from the time Return is pressed"); }
        say(5, msg2);
      } else if (!strncmp(line, "/len ", 5)) {
        uint32_t v = parse_u(line + 5);
        if (v) n_gen = v;
      } else status(2, "/regs /scan /models /hdr N /load N [direct|port|slow] /check /verify /tok TEXT /step /new /more /len N /temp T /seed [N] /rows 25|50");
      continue;
    }
    if (cur < 0) { status(2, "no model loaded: /load N"); continue; }
    if (models[cur].arch == 2) { status(2, "an encoder (BERT): no text to generate"); continue; }
    if (!line[0]) { reply(""); continue; }     /* (an empty line: more of the same) */
    colour = 7;
    out_text(line);
    out_newline();
    reply(line);
  }
}
