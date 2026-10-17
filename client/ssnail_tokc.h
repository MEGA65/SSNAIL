/*
  SSNAIL image tokenizer in C: the same algorithms as ssnail_tok.py (the
  reference), reading the tables inside a loaded image.  Portable: the image
  is reached only through tm_read(), which the MEGA65 client implements with
  DMA from SSNAIL's RAM, and the host test with an image file.
*/
#ifndef SSNAIL_TOKC_H
#define SSNAIL_TOKC_H

#include <stdint.h>

#define TOK_SPM 0
#define TOK_BPE 1
#define TOK_WORDPIECE 2
#define NO_TOKEN 0xFFFFFFFFUL

/* image access: n bytes at SSNAIL address a (n <= 255) */
void tm_read(uint32_t a, uint8_t *dst, uint8_t n);

/* read the header's tables; base = the image's address ($8000000) */
void tok_init(uint32_t base);
uint8_t tok_kind(void);

/* text (ASCII, NUL-terminated) -> token ids; returns how many were stored
   (at most max).  first: add the begin token (BOS / [CLS]) where the model
   uses one. */
uint16_t tok_encode(const char *text, uint8_t first, uint32_t *out, uint16_t max);

/* a token's piece: its bytes into buf (at most 255); returns the length */
uint8_t tok_piece(uint32_t t, uint8_t *buf);

/* output folding (ssnail_tok.AsciiFolder): feed a token's raw bytes, get
   printable ASCII (NUL-terminated) in out, which must hold 4 * n + 1 */
void fold_reset(void);
void fold_feed(const uint8_t *b, uint8_t n, char *out);

#endif
